//! Drive scanned files through extract → chunk → store, then embed
//! whatever is missing vectors. Each file is one transaction; a crash
//! loses at most the file in flight.

use std::fs::File;
use std::io;
use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use crate::chunk;
use crate::config::Config;
use crate::embed::Embedder;
use crate::extract;
use crate::index::{FileStatus, IndexDb};
use crate::scan::{ScanPlan, ScannedFile};

/// What one ingest run did.
#[derive(Debug, Default, PartialEq)]
pub struct IngestReport {
    pub indexed: usize,
    pub pending: usize,
    pub evicted: usize,
    pub errors: usize,
    pub removed: usize,
    pub unchanged: usize,
    pub embedded: usize,
    pub pruned_embeddings: u64,
}

impl IngestReport {
    pub fn summary(&self) -> String {
        format!(
            "{} indexed, {} pending, {} evicted, {} errors, {} removed, {} unchanged, \
             {} chunks embedded, {} orphan embeddings pruned",
            self.indexed,
            self.pending,
            self.evicted,
            self.errors,
            self.removed,
            self.unchanged,
            self.embedded,
            self.pruned_embeddings
        )
    }
}

/// Execute a scan plan against the index.
pub fn ingest(db: &mut IndexDb, config: &Config, plan: ScanPlan, now_ms: i64) -> IngestReport {
    let mut report = IngestReport {
        unchanged: plan.unchanged,
        ..IngestReport::default()
    };

    for rel_path in &plan.to_remove {
        match db.remove_file(rel_path) {
            Ok(true) => {
                tracing::info!("removed {rel_path}");
                report.removed += 1;
            }
            Ok(false) => {}
            Err(err) => {
                tracing::warn!("could not remove {rel_path}: {err}");
                report.errors += 1;
            }
        }
    }

    for file in &plan.to_index {
        match ingest_one(db, config, file, now_ms) {
            Ok(status) => match status {
                FileStatus::Indexed => report.indexed += 1,
                FileStatus::Pending => report.pending += 1,
                FileStatus::Evicted => report.evicted += 1,
                FileStatus::Error => report.errors += 1,
            },
            Err(err) => {
                // A DB-level failure; the file keeps its previous state.
                tracing::warn!("ingest failed for {}: {err:#}", file.rel_path);
                report.errors += 1;
            }
        }
    }
    report
}

fn ingest_one(
    db: &mut IndexDb,
    config: &Config,
    file: &ScannedFile,
    now_ms: i64,
) -> Result<FileStatus> {
    let kind = file.kind.as_str();

    if file.evicted {
        tracing::info!("evicted (content not local): {}", file.rel_path);
        db.upsert_unextracted_file(
            &file.rel_path,
            kind,
            file.size,
            file.mtime_ms,
            FileStatus::Evicted,
            None,
        )?;
        return Ok(FileStatus::Evicted);
    }

    if !file.kind.extractable() {
        db.upsert_unextracted_file(
            &file.rel_path,
            kind,
            file.size,
            file.mtime_ms,
            FileStatus::Pending,
            None,
        )?;
        return Ok(FileStatus::Pending);
    }

    let extracted = match extract::extract(&file.abs_path, file.kind) {
        Ok(doc) => doc,
        Err(err) => {
            tracing::warn!("extraction failed for {}: {err:#}", file.rel_path);
            db.upsert_unextracted_file(
                &file.rel_path,
                kind,
                file.size,
                file.mtime_ms,
                FileStatus::Error,
                Some(&format!("{err:#}")),
            )?;
            return Ok(FileStatus::Error);
        }
    };

    let sha = match sha256_file(&file.abs_path) {
        Ok(s) => s,
        Err(err) => {
            db.upsert_unextracted_file(
                &file.rel_path,
                kind,
                file.size,
                file.mtime_ms,
                FileStatus::Error,
                Some(&format!("{err:#}")),
            )?;
            return Ok(FileStatus::Error);
        }
    };

    let pieces = chunk::chunk_text(
        &extracted.text,
        config.index.chunk_target_tokens,
        config.index.chunk_overlap_tokens,
    );
    tracing::info!("indexed {} ({} chunks)", file.rel_path, pieces.len());
    db.upsert_indexed_file(
        &file.rel_path,
        kind,
        file.size,
        file.mtime_ms,
        &sha,
        extracted.title.as_deref(),
        &pieces,
        now_ms,
    )?;
    Ok(FileStatus::Indexed)
}

/// Embed every chunk that lacks a vector, committing per batch so an
/// interrupted run resumes where it stopped. Returns how many chunks got
/// embedded.
pub fn embed_missing(db: &mut IndexDb, embedder: &mut dyn Embedder) -> Result<usize> {
    const STORE_BATCH: usize = 256;

    let wiped = db.ensure_embedding_model(&embedder.id())?;
    if wiped {
        tracing::info!("embedding model changed; all stored vectors were invalidated");
    }
    let missing = db.missing_embeddings()?;
    if missing.is_empty() {
        return Ok(0);
    }
    tracing::info!("embedding {} chunks", missing.len());

    let mut done = 0;
    for batch in missing.chunks(STORE_BATCH) {
        let texts: Vec<String> = batch.iter().map(|(_, t)| t.clone()).collect();
        let vectors = embedder
            .embed_docs(&texts)
            .context("embedding batch failed")?;
        anyhow::ensure!(
            vectors.len() == batch.len(),
            "embedder returned {} vectors for {} chunks",
            vectors.len(),
            batch.len()
        );
        let items: Vec<(String, Vec<f32>)> = batch
            .iter()
            .map(|(hash, _)| hash.clone())
            .zip(vectors)
            .collect();
        db.store_embeddings(&embedder.id(), &items)?;
        done += items.len();
    }
    Ok(done)
}

fn sha256_file(path: &Path) -> Result<String> {
    use io::Read;
    let mut file =
        File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("could not read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan;
    use std::fs;
    use tempfile::TempDir;

    fn setup() -> (TempDir, TempDir, IndexDb, Config) {
        let tree = TempDir::new().unwrap();
        let dbdir = TempDir::new().unwrap();
        let db = IndexDb::open(&dbdir.path().join("index.sqlite")).unwrap();
        let mut config = Config::default();
        config.embeddings.provider = "debug-hash".into();
        (tree, dbdir, db, config)
    }

    fn run_scan(tree: &Path, db: &mut IndexDb, config: &Config) -> IngestReport {
        let scanned = scan::scan_tree(tree, &config.source).unwrap();
        let plan = scan::plan(scanned, &db.file_states().unwrap());
        ingest(db, config, plan, 1000)
    }

    #[test]
    fn full_pipeline_indexes_embeds_and_searches() {
        let (tree, _dbdir, mut db, config) = setup();
        fs::write(
            tree.path().join("closing.txt"),
            "The final sale price of the house was $487,500.",
        )
        .unwrap();
        fs::write(tree.path().join("todo.md"), "buy milk\n\nwalk the dog").unwrap();

        let report = run_scan(tree.path(), &mut db, &config);
        assert_eq!(report.indexed, 2);
        assert_eq!(report.errors, 0);

        let mut embedder =
            crate::embed::make_embedder(&config, &_dbdir.path().join("models")).unwrap();
        let n = embed_missing(&mut db, embedder.as_mut()).unwrap();
        assert!(n >= 2);

        let hits = db.search("sale price house", 10).unwrap();
        assert_eq!(hits[0].rel_path, "closing.txt");

        let qv = embedder.embed_query("sale price house").unwrap();
        let vhits = db.vector_search(&qv, 10).unwrap();
        assert_eq!(vhits[0].rel_path, "closing.txt");
    }

    #[test]
    fn rescan_of_unchanged_tree_does_nothing() {
        let (tree, _dbdir, mut db, config) = setup();
        fs::write(tree.path().join("a.txt"), "hello").unwrap();
        let first = run_scan(tree.path(), &mut db, &config);
        assert_eq!(first.indexed, 1);
        let second = run_scan(tree.path(), &mut db, &config);
        assert_eq!(second.indexed, 0);
        assert_eq!(second.unchanged, 1);
    }

    #[test]
    fn deleting_a_file_removes_it_from_the_index() {
        let (tree, _dbdir, mut db, config) = setup();
        let path = tree.path().join("a.txt");
        fs::write(&path, "hello world").unwrap();
        run_scan(tree.path(), &mut db, &config);
        assert_eq!(db.file_count().unwrap(), 1);

        fs::remove_file(&path).unwrap();
        let report = run_scan(tree.path(), &mut db, &config);
        assert_eq!(report.removed, 1);
        assert_eq!(db.file_count().unwrap(), 0);
        assert!(db.search("hello", 10).unwrap().is_empty());
    }

    #[test]
    fn unsupported_kinds_wait_as_pending() {
        let (tree, _dbdir, mut db, config) = setup();
        fs::write(tree.path().join("scan.pdf"), "%PDF-1.4 fake").unwrap();
        let report = run_scan(tree.path(), &mut db, &config);
        assert_eq!(report.pending, 1);
        let states = db.file_states().unwrap();
        assert_eq!(states["scan.pdf"].status, "pending");
    }

    #[test]
    fn evicted_stubs_are_recorded_not_read() {
        let (tree, _dbdir, mut db, config) = setup();
        fs::write(tree.path().join(".notes.txt.icloud"), "placeholder").unwrap();
        let report = run_scan(tree.path(), &mut db, &config);
        assert_eq!(report.evicted, 1);
        let states = db.file_states().unwrap();
        assert_eq!(states["notes.txt"].status, "evicted");
    }

    #[test]
    fn changed_content_is_reindexed_and_orphans_pruned() {
        let (tree, dbdir, mut db, config) = setup();
        let path = tree.path().join("a.txt");
        fs::write(&path, "original words here").unwrap();
        run_scan(tree.path(), &mut db, &config);
        let mut embedder =
            crate::embed::make_embedder(&config, &dbdir.path().join("models")).unwrap();
        embed_missing(&mut db, embedder.as_mut()).unwrap();
        assert_eq!(db.embedding_count().unwrap(), 1);

        // Ensure a different mtime even on coarse filesystems.
        fs::write(&path, "completely different replacement text").unwrap();
        let new_mtime = std::time::SystemTime::now() + std::time::Duration::from_secs(2);
        File::open(&path).unwrap().set_modified(new_mtime).unwrap();

        let report = run_scan(tree.path(), &mut db, &config);
        assert_eq!(report.indexed, 1);
        assert!(db.search("original", 10).unwrap().is_empty());
        assert_eq!(db.search("replacement", 10).unwrap().len(), 1);

        let pruned = db.prune_orphan_embeddings().unwrap();
        assert_eq!(pruned, 1);
        embed_missing(&mut db, embedder.as_mut()).unwrap();
        assert_eq!(db.embedding_count().unwrap(), 1);
    }

    #[test]
    fn embed_missing_is_idempotent() {
        let (tree, dbdir, mut db, config) = setup();
        fs::write(tree.path().join("a.txt"), "some text").unwrap();
        run_scan(tree.path(), &mut db, &config);
        let mut embedder =
            crate::embed::make_embedder(&config, &dbdir.path().join("models")).unwrap();
        assert_eq!(embed_missing(&mut db, embedder.as_mut()).unwrap(), 1);
        assert_eq!(embed_missing(&mut db, embedder.as_mut()).unwrap(), 0);
    }
}
