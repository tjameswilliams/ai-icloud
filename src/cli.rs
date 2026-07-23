//! Command-line interface: parsing, dispatch, and human-facing output.
//! All logs go to stderr; stdout is for results.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use crate::config::{self, LoadedConfig};
use crate::doctor;
use crate::embed;
use crate::index::{IndexDb, SearchHit};
use crate::ingest;
use crate::retrieve::{self, RetrievalParams};
use crate::scan;

#[derive(Parser)]
#[command(
    name = "ai-icloud",
    version,
    about = "Local-first iCloud Drive document RAG index and MCP server"
)]
struct Cli {
    /// Path to a config file (default: ~/Library/Application Support/ai-icloud/config.toml)
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Increase log verbosity (-v info, -vv debug); logs go to stderr
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check the environment: source tree, index, endpoints
    Doctor,
    /// Scan the source tree and index new or changed files
    Scan {
        /// Report what would happen without touching the index
        #[arg(long)]
        dry_run: bool,
        /// Skip the embedding pass (chunks index for keyword search only)
        #[arg(long)]
        no_embed: bool,
    },
    /// Search the index
    Search {
        /// The query terms
        #[arg(required = true)]
        query: Vec<String>,
        #[arg(long)]
        limit: Option<u32>,
        /// Keyword (FTS5) search only
        #[arg(long, conflicts_with = "semantic")]
        keyword: bool,
        /// Vector search only
        #[arg(long)]
        semantic: bool,
    },
    /// Show or locate the configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Print the effective configuration (secrets redacted)
    Show,
    /// Print the config file path
    Path,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);
    let loaded = config::load(cli.config.as_deref())?;

    match cli.command {
        Command::Doctor => run_doctor(&loaded),
        Command::Scan { dry_run, no_embed } => run_scan(&loaded, dry_run, no_embed),
        Command::Search {
            query,
            limit,
            keyword,
            semantic,
        } => run_search(&loaded, &query.join(" "), limit, keyword, semantic),
        Command::Config { action } => run_config(&loaded, action),
    }
}

fn init_tracing(verbose: u8) {
    use tracing_subscriber::EnvFilter;
    let default = match verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("ai_icloud={default}")));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

fn run_doctor(loaded: &LoadedConfig) -> Result<()> {
    println!(
        "config: {}{}",
        loaded.path.display(),
        if loaded.from_file { "" } else { " (defaults; no file)" }
    );
    let checks = doctor::run_checks(&loaded.config)?;
    if doctor::print_report(&checks) {
        Ok(())
    } else {
        bail!("one or more checks failed");
    }
}

fn run_scan(loaded: &LoadedConfig, dry_run: bool, no_embed: bool) -> Result<()> {
    let config = &loaded.config;
    let root = config.source_root()?;
    let scanned = scan::scan_tree(&root, &config.source)?;

    if dry_run {
        // Dry run must not create the index; diff against it only if it
        // already exists.
        let states = match config.index_db_path() {
            Ok(p) if p.exists() => IndexDb::open(&p)?.file_states()?,
            _ => Default::default(),
        };
        let plan = scan::plan(scanned, &states);
        println!(
            "would index {} file(s), remove {}, leave {} unchanged",
            plan.to_index.len(),
            plan.to_remove.len(),
            plan.unchanged
        );
        for f in &plan.to_index {
            let note = if f.evicted { " (evicted stub)" } else { "" };
            println!("  + {} [{}]{note}", f.rel_path, f.kind.as_str());
        }
        for r in &plan.to_remove {
            println!("  - {r}");
        }
        return Ok(());
    }

    let mut db = IndexDb::open(&config.index_db_path()?)?;
    let plan = scan::plan(scanned, &db.file_states()?);
    let mut report = ingest::ingest(&mut db, config, plan, chrono::Utc::now().timestamp_millis());
    report.pruned_embeddings = db.prune_orphan_embeddings()?;

    if !no_embed {
        let model_dir = config.index_dir()?.join("models");
        let mut embedder = embed::make_embedder(config, &model_dir)?;
        report.embedded = ingest::embed_missing(&mut db, embedder.as_mut())?;
    }
    db.set_last_scan_ms(chrono::Utc::now().timestamp_millis())?;
    println!("{}", report.summary());
    Ok(())
}

fn run_search(
    loaded: &LoadedConfig,
    query: &str,
    limit: Option<u32>,
    keyword: bool,
    semantic: bool,
) -> Result<()> {
    let config = &loaded.config;
    let db_path = config.index_db_path()?;
    if !db_path.exists() {
        bail!("no index yet — run `ai-icloud scan` first");
    }
    let db = IndexDb::open(&db_path)?;
    let limit = limit.unwrap_or(config.retrieval.result_limit).clamp(1, 100);

    let hits = if keyword {
        db.search(query, limit)?
    } else {
        let query_vec = if db.embedding_count()? > 0 {
            let model_dir = config.index_dir()?.join("models");
            let mut embedder = embed::make_embedder(config, &model_dir)?;
            Some(embedder.embed_query(query)?)
        } else {
            if semantic {
                bail!("no embeddings stored yet — run `ai-icloud scan` without --no-embed");
            }
            None
        };
        if semantic {
            db.vector_search(query_vec.as_deref().unwrap(), limit)?
        } else {
            retrieve::hybrid_search(
                &db,
                query,
                query_vec.as_deref(),
                &RetrievalParams {
                    fts_candidates: config.retrieval.fts_candidates,
                    vector_candidates: config.retrieval.vector_candidates,
                    limit,
                },
            )?
        }
    };

    if hits.is_empty() {
        println!("no results");
        return Ok(());
    }
    println!("{} result(s)", hits.len());
    for hit in &hits {
        println!("{}", format_hit(hit));
    }
    Ok(())
}

fn format_hit(hit: &SearchHit) -> String {
    let mut line = format!("[chunk {}] {}", hit.chunk_id, hit.rel_path);
    if let Some(doc_type) = &hit.doc_type {
        line.push_str(&format!(" ({doc_type})"));
    }
    if let Some(score) = hit.score {
        line.push_str(&format!(" score={score:.3}"));
    }
    line.push_str("\n  ");
    line.push_str(&hit.snippet.replace('\n', " "));
    line
}

fn run_config(loaded: &LoadedConfig, action: ConfigAction) -> Result<()> {
    match action {
        ConfigAction::Show => {
            let toml = loaded
                .config
                .redacted()
                .to_toml()
                .context("could not render configuration")?;
            print!("{toml}");
            Ok(())
        }
        ConfigAction::Path => {
            println!("{}", loaded.path.display());
            Ok(())
        }
    }
}
