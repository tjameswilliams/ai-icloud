//! The watch daemon: FSEvents (via notify) on the source root triggers a
//! debounced incremental pass; a periodic tick reconciles anything
//! FSEvents dropped. Passes are cheap — the scanner walks and diffs in
//! milliseconds, so events carry no path information, only "something
//! changed".

use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use notify::{RecursiveMode, Watcher};

use crate::config::LoadedConfig;
use crate::embed;
use crate::index::IndexDb;
use crate::ingest;
use crate::scan;

pub fn run_watch(loaded: &LoadedConfig) -> Result<()> {
    let config = &loaded.config;
    let root = config.source_root()?;
    let debounce = Duration::from_millis(config.watch.debounce_ms.max(100));
    let full_scan_every =
        Duration::from_secs(u64::from(config.watch.full_scan_interval_minutes.max(1)) * 60);

    tracing::info!("watching {} (debounce {:?})", root.display(), debounce);
    run_pass(loaded);

    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        // Any event is just a wake-up; the scanner does the diffing.
        if res.is_ok() {
            let _ = tx.send(());
        }
    })
    .context("could not create the filesystem watcher")?;
    watcher
        .watch(&root, RecursiveMode::Recursive)
        .with_context(|| format!("could not watch {}", root.display()))?;

    loop {
        match wait_for_trigger(&rx, full_scan_every, debounce)? {
            Trigger::Events => tracing::debug!("change events settled; running a pass"),
            Trigger::Tick => tracing::debug!("periodic reconciliation pass"),
        }
        run_pass(loaded);
    }
}

#[derive(Debug, PartialEq)]
enum Trigger {
    Events,
    Tick,
}

/// Block until either an event arrives (then drain until the stream has
/// been quiet for `debounce`) or `tick_after` elapses with no events.
fn wait_for_trigger(
    rx: &Receiver<()>,
    tick_after: Duration,
    debounce: Duration,
) -> Result<Trigger> {
    match rx.recv_timeout(tick_after) {
        Ok(()) => {
            loop {
                match rx.recv_timeout(debounce) {
                    Ok(()) => continue, // still churning; keep waiting
                    Err(RecvTimeoutError::Timeout) => return Ok(Trigger::Events),
                    Err(RecvTimeoutError::Disconnected) => bail!("watcher stopped unexpectedly"),
                }
            }
        }
        Err(RecvTimeoutError::Timeout) => Ok(Trigger::Tick),
        Err(RecvTimeoutError::Disconnected) => bail!("watcher stopped unexpectedly"),
    }
}

/// One incremental pass: scan → ingest → enrich → embed. Failures are
/// logged, never fatal — the daemon must outlive a bad file or a
/// stopped LLM server.
fn run_pass(loaded: &LoadedConfig) {
    if let Err(err) = try_pass(loaded) {
        tracing::error!("pass failed: {err:#}");
    }
}

fn try_pass(loaded: &LoadedConfig) -> Result<()> {
    let config = &loaded.config;
    let root = config.source_root()?;
    let scanned = scan::scan_tree(&root, &config.source)?;
    let mut db = IndexDb::open(&config.index_db_path()?)?;
    let plan = scan::plan(scanned, &db.file_states()?);

    let had_work = !plan.to_index.is_empty() || !plan.to_remove.is_empty();
    let mut report = ingest::ingest(&mut db, config, plan, chrono::Utc::now().timestamp_millis());
    report.pruned_embeddings = db.prune_orphan_embeddings()?;

    match crate::enrich::enrich_pending(&mut db, config, false, u32::MAX) {
        Ok(er) if er.enriched + er.failed > 0 => {
            tracing::info!("enrichment: {}", er.summary());
        }
        Ok(_) => {}
        Err(err) => tracing::warn!("enrichment pass skipped: {err:#}"),
    }

    if db.missing_embeddings()?.is_empty() {
        // Nothing new to embed; skip loading the model.
    } else {
        let model_dir = config.index_dir()?.join("models");
        let mut embedder = embed::make_embedder(config, &model_dir)?;
        report.embedded = ingest::embed_missing(&mut db, embedder.as_mut())?;
    }
    db.set_last_scan_ms(chrono::Utc::now().timestamp_millis())?;

    if had_work {
        tracing::info!("{}", report.summary());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn quiet_stream_ticks() {
        let (_tx, rx) = channel::<()>();
        let t = wait_for_trigger(&rx, Duration::from_millis(10), Duration::from_millis(5)).unwrap();
        assert_eq!(t, Trigger::Tick);
    }

    #[test]
    fn events_settle_after_debounce() {
        let (tx, rx) = channel::<()>();
        tx.send(()).unwrap();
        tx.send(()).unwrap();
        let t = wait_for_trigger(&rx, Duration::from_secs(5), Duration::from_millis(10)).unwrap();
        assert_eq!(t, Trigger::Events);
    }

    #[test]
    fn disconnected_watcher_is_an_error() {
        let (tx, rx) = channel::<()>();
        drop(tx);
        assert!(wait_for_trigger(&rx, Duration::from_millis(10), Duration::from_millis(5)).is_err());
    }
}
