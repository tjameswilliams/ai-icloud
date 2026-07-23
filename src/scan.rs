//! Walk the source tree, apply scope filters, and diff what's on disk
//! against the index to decide what needs (re)indexing or removal.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, bail};
use globset::{Glob, GlobSet, GlobSetBuilder};
use walkdir::WalkDir;

use crate::config::SourceConfig;
use crate::extract::FileKind;
use crate::index::FileState;

/// One in-scope file found on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct ScannedFile {
    /// Path relative to the scan root — the file's identity in the index.
    /// For an eviction stub `.name.ext.icloud`, this is the logical
    /// `name.ext` path so the row survives materialization.
    pub rel_path: String,
    pub abs_path: PathBuf,
    pub kind: FileKind,
    pub size: i64,
    pub mtime_ms: i64,
    /// True when the file is an iCloud eviction stub (content not local).
    pub evicted: bool,
}

/// The work a scan decided on.
#[derive(Debug, Default)]
pub struct ScanPlan {
    pub to_index: Vec<ScannedFile>,
    /// rel_paths present in the index but gone from disk (or newly excluded).
    pub to_remove: Vec<String>,
    pub unchanged: usize,
}

/// Compiled exclusion globs, matched against rel_paths.
pub struct Excludes {
    set: GlobSet,
}

impl Excludes {
    pub fn compile(patterns: &[String]) -> Result<Excludes> {
        let mut builder = GlobSetBuilder::new();
        for p in patterns {
            builder.add(
                Glob::new(p).with_context(|| format!("invalid exclude glob {p:?}"))?,
            );
        }
        Ok(Excludes {
            set: builder.build().context("could not compile exclude globs")?,
        })
    }

    pub fn matches_file(&self, rel: &str) -> bool {
        self.set.is_match(rel)
    }

    /// A directory is pruned when the dir itself matches (pattern
    /// `Divorce`) or anything under it necessarily would (`Divorce/**`).
    pub fn matches_dir(&self, rel: &str) -> bool {
        self.set.is_match(rel) || self.set.is_match(format!("{rel}/x"))
    }
}

/// Walk `root` and return every in-scope file, filters applied.
pub fn scan_tree(root: &Path, source: &SourceConfig) -> Result<Vec<ScannedFile>> {
    if !root.is_dir() {
        bail!(
            "source root {} does not exist or is not a directory",
            root.display()
        );
    }
    let excludes = Excludes::compile(&source.exclude_globs)?;
    let max_bytes = source.max_file_mb.saturating_mul(1024 * 1024);
    let include: Vec<String> = source
        .include_extensions
        .iter()
        .map(|e| e.to_ascii_lowercase())
        .collect();

    let mut out = Vec::new();
    let walker = WalkDir::new(root).follow_links(false).into_iter();
    let it = walker.filter_entry(|entry| {
        let name = entry.file_name().to_string_lossy();
        if entry.file_type().is_dir() {
            if entry.depth() == 0 {
                return true;
            }
            // Hidden directories (.git, .Trash) and excluded subtrees are
            // never entered.
            if name.starts_with('.') {
                return false;
            }
            let rel = rel_path(root, entry.path());
            return !excludes.matches_dir(&rel);
        }
        true
    });

    for entry in it {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!("scan: skipping unreadable entry: {err}");
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let (logical_name, evicted) = match icloud_stub_target(&name) {
            Some(target) => (target.to_string(), true),
            None if name.starts_with('.') => continue, // hidden file
            None => (name.clone(), false),
        };

        let Some(ext) = Path::new(&logical_name)
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
        else {
            continue;
        };
        if !include.contains(&ext) {
            continue;
        }
        let Some(kind) = FileKind::from_extension(&ext) else {
            continue;
        };

        let logical_rel = {
            let parent = rel_path(root, entry.path().parent().unwrap_or(root));
            if parent.is_empty() {
                logical_name.clone()
            } else {
                format!("{parent}/{logical_name}")
            }
        };
        if excludes.matches_file(&logical_rel) {
            continue;
        }

        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(err) => {
                tracing::warn!("scan: could not stat {}: {err}", entry.path().display());
                continue;
            }
        };
        // Stub sizes reflect the placeholder, not the real content; keep
        // them anyway — a materialized file will differ and re-trigger.
        if !evicted && max_bytes > 0 && meta.len() > max_bytes {
            tracing::info!(
                "scan: skipping {} ({} MB > max_file_mb)",
                logical_rel,
                meta.len() / (1024 * 1024)
            );
            continue;
        }
        let mtime_ms = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        out.push(ScannedFile {
            rel_path: logical_rel,
            abs_path: entry.path().to_path_buf(),
            kind,
            size: meta.len() as i64,
            mtime_ms,
            evicted,
        });
    }
    // Deterministic order helps tests and makes progress output readable.
    out.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Ok(out)
}

/// Diff disk against index state: decide what to (re)index and remove.
pub fn plan(scanned: Vec<ScannedFile>, states: &HashMap<String, FileState>) -> ScanPlan {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut plan = ScanPlan::default();

    for file in scanned {
        // Two entries can map to one rel_path (a stub and its
        // materializing original mid-download); first one wins.
        if !seen.insert(file.rel_path.clone()) {
            continue;
        }
        match states.get(&file.rel_path) {
            None => plan.to_index.push(file),
            Some(state) => {
                let changed = state.size != file.size || state.mtime_ms != file.mtime_ms;
                let stalled = match state.status.as_str() {
                    // Retry errors and evictions on every scan; index
                    // pending files whose extractor has since shipped.
                    "error" | "evicted" => true,
                    "pending" => file.kind.extractable(),
                    _ => false,
                };
                if changed || stalled {
                    plan.to_index.push(file);
                } else {
                    plan.unchanged += 1;
                }
            }
        }
    }

    for rel_path in states.keys() {
        if !seen.contains(rel_path.as_str()) {
            plan.to_remove.push(rel_path.clone());
        }
    }
    plan.to_remove.sort();
    plan
}

fn rel_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// `.name.ext.icloud` → `name.ext` for iCloud eviction stubs.
fn icloud_stub_target(name: &str) -> Option<&str> {
    name.strip_prefix('.')?.strip_suffix(".icloud")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn source(exclude: &[&str]) -> SourceConfig {
        SourceConfig {
            exclude_globs: exclude.iter().map(|s| s.to_string()).collect(),
            ..SourceConfig::default()
        }
    }

    fn state(size: i64, mtime_ms: i64, status: &str) -> FileState {
        FileState {
            id: 1,
            size,
            mtime_ms,
            content_sha256: None,
            status: status.into(),
        }
    }

    fn scanned(rel: &str, size: i64, mtime_ms: i64) -> ScannedFile {
        ScannedFile {
            rel_path: rel.into(),
            abs_path: PathBuf::from(rel),
            kind: FileKind::Text,
            size,
            mtime_ms,
            evicted: false,
        }
    }

    #[test]
    fn scan_finds_included_files_and_skips_others() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "hello").unwrap();
        fs::write(dir.path().join("b.stl"), "mesh").unwrap();
        fs::write(dir.path().join(".hidden.txt"), "x").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/c.csv"), "x,y").unwrap();

        let files = scan_tree(dir.path(), &source(&[])).unwrap();
        let rels: Vec<&str> = files.iter().map(|f| f.rel_path.as_str()).collect();
        assert_eq!(rels, vec!["a.txt", "sub/c.csv"]);
    }

    #[test]
    fn excluded_globs_prune_files_and_whole_directories() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("Divorce")).unwrap();
        fs::write(dir.path().join("Divorce/settlement.txt"), "x").unwrap();
        fs::write(dir.path().join("keep.txt"), "x").unwrap();
        fs::write(dir.path().join("skipme.txt"), "x").unwrap();

        for pattern in ["Divorce/**", "Divorce"] {
            let files =
                scan_tree(dir.path(), &source(&[pattern, "skipme.txt"])).unwrap();
            let rels: Vec<&str> = files.iter().map(|f| f.rel_path.as_str()).collect();
            assert_eq!(rels, vec!["keep.txt"], "pattern {pattern}");
        }
    }

    #[test]
    fn hidden_directories_are_not_entered() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join(".git")).unwrap();
        fs::write(dir.path().join(".git/notes.txt"), "x").unwrap();
        let files = scan_tree(dir.path(), &source(&[])).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn icloud_stubs_map_to_their_logical_path() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(".report.pdf.icloud"), "stub").unwrap();
        let files = scan_tree(dir.path(), &source(&[])).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].rel_path, "report.pdf");
        assert!(files[0].evicted);
        assert_eq!(files[0].kind, FileKind::Pdf);
    }

    #[test]
    fn oversized_files_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("big.txt"), vec![b'x'; 3 * 1024 * 1024]).unwrap();
        let mut cfg = source(&[]);
        cfg.max_file_mb = 2;
        assert!(scan_tree(dir.path(), &cfg).unwrap().is_empty());
        cfg.max_file_mb = 4;
        assert_eq!(scan_tree(dir.path(), &cfg).unwrap().len(), 1);
    }

    #[test]
    fn missing_root_is_an_error() {
        assert!(scan_tree(Path::new("/nonexistent-root"), &source(&[])).is_err());
    }

    #[test]
    fn plan_indexes_new_and_changed_files_only() {
        let mut states = HashMap::new();
        states.insert("same.txt".to_string(), state(5, 100, "indexed"));
        states.insert("changed.txt".to_string(), state(5, 100, "indexed"));

        let p = plan(
            vec![
                scanned("same.txt", 5, 100),
                scanned("changed.txt", 6, 200),
                scanned("new.txt", 1, 1),
            ],
            &states,
        );
        let rels: Vec<&str> = p.to_index.iter().map(|f| f.rel_path.as_str()).collect();
        assert_eq!(rels, vec!["changed.txt", "new.txt"]);
        assert_eq!(p.unchanged, 1);
        assert!(p.to_remove.is_empty());
    }

    #[test]
    fn plan_removes_files_gone_from_disk() {
        let mut states = HashMap::new();
        states.insert("gone.txt".to_string(), state(5, 100, "indexed"));
        let p = plan(vec![], &states);
        assert_eq!(p.to_remove, vec!["gone.txt"]);
    }

    #[test]
    fn plan_retries_errored_and_evicted_files() {
        let mut states = HashMap::new();
        states.insert("err.txt".to_string(), state(5, 100, "error"));
        states.insert("ev.txt".to_string(), state(5, 100, "evicted"));
        let p = plan(
            vec![scanned("err.txt", 5, 100), scanned("ev.txt", 5, 100)],
            &states,
        );
        assert_eq!(p.to_index.len(), 2);
    }

    #[test]
    fn plan_retries_pending_files() {
        // Every kind now has an extractor, so pending files (e.g. media
        // recorded while transcription was disabled) retry each scan.
        let mut states = HashMap::new();
        states.insert("doc.txt".to_string(), state(5, 100, "pending"));
        states.insert("talk.mp3".to_string(), state(5, 100, "pending"));
        let mut audio = scanned("talk.mp3", 5, 100);
        audio.kind = FileKind::Audio;
        let p = plan(vec![scanned("doc.txt", 5, 100), audio], &states);
        assert_eq!(p.to_index.len(), 2);
        assert_eq!(p.unchanged, 0);
    }
}
