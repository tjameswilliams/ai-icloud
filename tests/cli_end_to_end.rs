//! End-to-end tests driving the real binary against a synthetic tree.
//! The debug-hash embedder keeps everything offline and deterministic.

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

struct Fixture {
    _dir: TempDir,
    tree: std::path::PathBuf,
    config_path: std::path::PathBuf,
}

fn fixture() -> Fixture {
    let dir = TempDir::new().unwrap();
    let tree = dir.path().join("tree");
    std::fs::create_dir_all(tree.join("House")).unwrap();
    std::fs::create_dir_all(tree.join("Private")).unwrap();
    std::fs::write(
        tree.join("House/closing-statement.txt"),
        "Closing statement for 423 R St. Final sale price: $487,500. \
         Payoff of existing mortgage: $210,000. Net proceeds to seller: $250,000.",
    )
    .unwrap();
    std::fs::write(
        tree.join("groceries.md"),
        "# Groceries\n\nmilk, eggs, coffee",
    )
    .unwrap();
    std::fs::write(tree.join("Private/secret.txt"), "do not index me").unwrap();
    std::fs::write(tree.join("voice-memo.mp3"), "fake audio bytes").unwrap();

    let index_path = dir.path().join("appdir").join("index.sqlite");
    let config_path = dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        format!(
            r#"
[source]
root = "{}"
exclude_globs = ["Private/**"]

[embeddings]
provider = "debug-hash"

[index]
database_path = "{}"

[llm]
base_url = "http://127.0.0.1:1/v1"
"#,
            tree.display(),
            index_path.display()
        ),
    )
    .unwrap();
    Fixture {
        _dir: dir,
        tree,
        config_path,
    }
}

fn cmd(fx: &Fixture) -> Command {
    let mut c = Command::cargo_bin("ai-icloud").unwrap();
    c.arg("--config").arg(&fx.config_path);
    c
}

#[test]
fn dry_run_reports_without_creating_an_index() {
    let fx = fixture();
    cmd(&fx)
        .args(["scan", "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains("would index 3 file(s)"))
        .stdout(predicate::str::contains("House/closing-statement.txt"))
        .stdout(predicate::str::contains("voice-memo.mp3 [audio]"))
        .stdout(predicate::str::contains("Private").not());
    // Dry run must leave no index behind.
    let index_exists = fx
        .config_path
        .parent()
        .unwrap()
        .join("appdir")
        .join("index.sqlite")
        .exists();
    assert!(!index_exists);
}

#[test]
fn scan_then_search_finds_documents_and_respects_exclusions() {
    let fx = fixture();
    cmd(&fx)
        .arg("scan")
        .assert()
        .success()
        .stdout(predicate::str::contains("2 indexed"))
        .stdout(predicate::str::contains("1 pending"));

    // Hybrid search (keyword + debug-hash vectors).
    cmd(&fx)
        .args(["search", "sale", "price"])
        .assert()
        .success()
        .stdout(predicate::str::contains("House/closing-statement.txt"))
        .stdout(predicate::str::contains("«sale»"));

    // Keyword-only and semantic-only paths both work.
    cmd(&fx)
        .args(["search", "--keyword", "groceries"])
        .assert()
        .success()
        .stdout(predicate::str::contains("groceries.md"));
    cmd(&fx)
        .args(["search", "--semantic", "mortgage", "payoff"])
        .assert()
        .success()
        .stdout(predicate::str::contains("House/closing-statement.txt"));

    // Excluded content is nowhere in the index.
    cmd(&fx)
        .args(["search", "secret"])
        .assert()
        .success()
        .stdout(predicate::str::contains("no results"));
}

#[test]
fn rescan_is_incremental_and_deletion_prunes() {
    let fx = fixture();
    cmd(&fx).arg("scan").assert().success();
    cmd(&fx)
        .arg("scan")
        .assert()
        .success()
        .stdout(predicate::str::contains("0 indexed"))
        // Two indexed text files plus the still-pending mp3.
        .stdout(predicate::str::contains("3 unchanged"));

    std::fs::remove_file(fx.tree.join("groceries.md")).unwrap();
    cmd(&fx)
        .arg("scan")
        .assert()
        .success()
        .stdout(predicate::str::contains("1 removed"));
    cmd(&fx)
        .args(["search", "groceries"])
        .assert()
        .success()
        .stdout(predicate::str::contains("no results"));
}

#[test]
fn search_without_an_index_gives_a_clear_error() {
    let fx = fixture();
    cmd(&fx)
        .args(["search", "anything"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("run `ai-icloud scan` first"));
}

#[test]
fn config_show_redacts_secrets_and_config_path_prints() {
    let fx = fixture();
    let raw = std::fs::read_to_string(&fx.config_path).unwrap();
    std::fs::write(
        &fx.config_path,
        format!("{raw}api_key = \"super-secret-key\"\n"),
    )
    .unwrap();

    cmd(&fx)
        .args(["config", "show"])
        .assert()
        .success()
        .stdout(predicate::str::contains("<redacted>"))
        .stdout(predicate::str::contains("super-secret-key").not());
    cmd(&fx)
        .args(["config", "path"])
        .assert()
        .success()
        .stdout(predicate::str::contains("config.toml"));
}

#[test]
fn doctor_passes_on_a_healthy_fixture() {
    let fx = fixture();
    // The unreachable LLM endpoint is a warning, not a failure.
    cmd(&fx)
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains("[ok  ] source root"))
        .stdout(predicate::str::contains("[ok  ] index"));
}

#[test]
fn doctor_fails_when_the_source_root_is_missing() {
    let fx = fixture();
    std::fs::remove_dir_all(&fx.tree).unwrap();
    cmd(&fx)
        .arg("doctor")
        .assert()
        .failure()
        .stdout(predicate::str::contains("[FAIL] source root"));
}
