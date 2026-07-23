//! Diagnostic checks with actionable resolutions: is the source tree
//! readable (TCC/Full Disk Access), is the index healthy, do the
//! configured endpoints answer.

use std::io::ErrorKind;
use std::time::Duration;

use anyhow::Result;

use crate::config::Config;
use crate::embed;
use crate::index::IndexDb;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug)]
pub struct Check {
    pub level: Level,
    pub name: &'static str,
    pub detail: String,
    pub resolution: Option<String>,
}

impl Check {
    fn ok(name: &'static str, detail: impl Into<String>) -> Check {
        Check {
            level: Level::Ok,
            name,
            detail: detail.into(),
            resolution: None,
        }
    }

    fn warn(name: &'static str, detail: impl Into<String>, fix: impl Into<String>) -> Check {
        Check {
            level: Level::Warn,
            name,
            detail: detail.into(),
            resolution: Some(fix.into()),
        }
    }

    fn fail(name: &'static str, detail: impl Into<String>, fix: impl Into<String>) -> Check {
        Check {
            level: Level::Fail,
            name,
            detail: detail.into(),
            resolution: Some(fix.into()),
        }
    }
}

/// Run every check. Never returns Err — problems are results, not errors.
pub fn run_checks(config: &Config) -> Result<Vec<Check>> {
    Ok(vec![
        check_source_root(config),
        check_index(config),
        check_sidecar(config),
        check_embeddings_config(config),
        check_llm_endpoint(config),
    ])
}

fn check_source_root(config: &Config) -> Check {
    let root = match config.source_root() {
        Ok(r) => r,
        Err(e) => return Check::fail("source root", format!("{e:#}"), "check [source] root"),
    };
    match std::fs::read_dir(&root) {
        Ok(mut it) => {
            let _ = it.next();
            Check::ok("source root", format!("{} is readable", root.display()))
        }
        Err(e) if e.kind() == ErrorKind::NotFound => Check::fail(
            "source root",
            format!("{} does not exist", root.display()),
            "is iCloud Drive enabled? Set [source] root if your documents live elsewhere",
        ),
        Err(e) if e.kind() == ErrorKind::PermissionDenied => Check::fail(
            "source root",
            format!("{} exists but reading it was denied", root.display()),
            "grant Full Disk Access to this binary (System Settings → Privacy & Security → \
             Full Disk Access); a launchd service needs the grant on the binary itself, \
             not the terminal",
        ),
        Err(e) => Check::fail(
            "source root",
            format!("could not read {}: {e}", root.display()),
            "check [source] root",
        ),
    }
}

fn check_index(config: &Config) -> Check {
    let path = match config.index_db_path() {
        Ok(p) => p,
        Err(e) => return Check::fail("index", format!("{e:#}"), "check [index] database_path"),
    };
    match IndexDb::open(&path) {
        Ok(db) => {
            let files = db.file_count().unwrap_or(0);
            let chunks = db.chunk_count().unwrap_or(0);
            let embedded = db.embedding_count().unwrap_or(0);
            let statuses = db
                .status_counts()
                .unwrap_or_default()
                .iter()
                .map(|(s, n)| format!("{n} {s}"))
                .collect::<Vec<_>>()
                .join(", ");
            Check::ok(
                "index",
                format!(
                    "{} — {files} files ({statuses}), {chunks} chunks, {embedded} embedded",
                    path.display()
                ),
            )
        }
        Err(e) => Check::fail(
            "index",
            format!("could not open {}: {e}", path.display()),
            "check permissions on the index directory, or move [index] database_path",
        ),
    }
}

fn check_sidecar(config: &Config) -> Check {
    let dir = match config.index_dir() {
        Ok(d) => d.join("bin"),
        Err(e) => return Check::fail("ocr sidecar", format!("{e:#}"), "check [index] database_path"),
    };
    match crate::sidecar::Sidecar::ensure(&dir) {
        Ok(_) => Check::ok(
            "ocr sidecar",
            "Vision OCR helper materialized and answering",
        ),
        Err(e) => Check::fail(
            "ocr sidecar",
            format!("{e:#}"),
            "PDF/image extraction will fail; check permissions on the index directory",
        ),
    }
}

fn check_embeddings_config(config: &Config) -> Check {
    match config.embeddings.provider.as_str() {
        "embedded" => Check::ok(
            "embeddings",
            format!(
                "embedded/{} (weights auto-download on first scan)",
                config.embeddings.model
            ),
        ),
        "debug-hash" => Check::warn(
            "embeddings",
            "debug-hash provider — deterministic test vectors, no semantic quality",
            "set [embeddings] provider = \"embedded\" for real search",
        ),
        "openai-compatible" => match config.embeddings.base_url.as_deref() {
            None => Check::fail(
                "embeddings",
                "openai-compatible provider with no base_url",
                "set [embeddings] base_url",
            ),
            Some(url) if !embed::is_loopback_url(url) && !config.privacy.allow_remote_endpoints => {
                Check::fail(
                    "embeddings",
                    format!("{url} is not loopback and remote endpoints are not allowed"),
                    "use a local endpoint or set [privacy] allow_remote_endpoints = true",
                )
            }
            Some(url) => Check::ok("embeddings", format!("openai-compatible via {url}")),
        },
        other => Check::fail(
            "embeddings",
            format!("unknown provider {other:?}"),
            "use \"embedded\", \"openai-compatible\", or \"debug-hash\"",
        ),
    }
}

/// Informational until the enrichment phase: is the LLM endpoint up, and
/// does our key open it?
fn check_llm_endpoint(config: &Config) -> Check {
    let base = config.llm.base_url.trim_end_matches('/');
    if !embed::is_loopback_url(base) && !config.privacy.allow_remote_endpoints {
        return Check::fail(
            "llm endpoint",
            format!("{base} is not loopback and remote endpoints are not allowed"),
            "use a local endpoint or set [privacy] allow_remote_endpoints = true",
        );
    }
    let mut req = ureq::get(&format!("{base}/models")).timeout(Duration::from_secs(3));
    if let Some(key) = &config.llm.api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    match req.call() {
        Ok(resp) => {
            let body: serde_json::Value = resp.into_json().unwrap_or_default();
            let n = body
                .get("data")
                .and_then(|d| d.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            Check::ok("llm endpoint", format!("{base} answered ({n} models)"))
        }
        Err(ureq::Error::Status(401 | 403, _)) => Check::warn(
            "llm endpoint",
            format!("{base} rejected our credentials"),
            "set [llm] api_key (LM Studio: Developer → API token)",
        ),
        Err(e) => Check::warn(
            "llm endpoint",
            format!("{base} not reachable: {e}"),
            "start the LLM server (LM Studio / Ollama) — only needed for enrichment, \
             not for plain-text indexing",
        ),
    }
}

/// Human-readable report; returns true when nothing failed.
pub fn print_report(checks: &[Check]) -> bool {
    let mut all_ok = true;
    for c in checks {
        let tag = match c.level {
            Level::Ok => "ok  ",
            Level::Warn => "warn",
            Level::Fail => {
                all_ok = false;
                "FAIL"
            }
        };
        println!("[{tag}] {}: {}", c.name, c.detail);
        if let Some(fix) = &c.resolution {
            println!("       ↳ {fix}");
        }
    }
    all_ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn local_config(dir: &TempDir) -> Config {
        let mut c = Config::default();
        c.source.root = dir.path().join("tree").to_string_lossy().into_owned();
        c.index.database_path = dir
            .path()
            .join("index.sqlite")
            .to_string_lossy()
            .into_owned();
        // An unroutable port so the endpoint check fails fast.
        c.llm.base_url = "http://127.0.0.1:1/v1".into();
        c
    }

    #[test]
    fn missing_source_root_fails_with_hint() {
        let dir = TempDir::new().unwrap();
        let checks = run_checks(&local_config(&dir)).unwrap();
        let src = checks.iter().find(|c| c.name == "source root").unwrap();
        assert_eq!(src.level, Level::Fail);
        assert!(src.resolution.is_some());
    }

    #[test]
    fn healthy_local_setup_passes_source_and_index() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("tree")).unwrap();
        let checks = run_checks(&local_config(&dir)).unwrap();
        assert_eq!(
            checks
                .iter()
                .find(|c| c.name == "source root")
                .unwrap()
                .level,
            Level::Ok
        );
        assert_eq!(
            checks.iter().find(|c| c.name == "index").unwrap().level,
            Level::Ok
        );
        // The LLM endpoint being down is a warning, not a failure.
        assert_eq!(
            checks
                .iter()
                .find(|c| c.name == "llm endpoint")
                .unwrap()
                .level,
            Level::Warn
        );
    }

    #[test]
    fn remote_llm_endpoint_without_privacy_flag_fails() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("tree")).unwrap();
        let mut cfg = local_config(&dir);
        cfg.llm.base_url = "https://api.example.com/v1".into();
        let checks = run_checks(&cfg).unwrap();
        assert_eq!(
            checks
                .iter()
                .find(|c| c.name == "llm endpoint")
                .unwrap()
                .level,
            Level::Fail
        );
    }

    #[test]
    fn debug_hash_provider_warns() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("tree")).unwrap();
        let mut cfg = local_config(&dir);
        cfg.embeddings.provider = "debug-hash".into();
        let checks = run_checks(&cfg).unwrap();
        assert_eq!(
            checks
                .iter()
                .find(|c| c.name == "embeddings")
                .unwrap()
                .level,
            Level::Warn
        );
    }
}
