//! `ai-icloud setup`: the interactive onboarding wizard.
//!
//! ai-icloud needs one thing the sister project does not: an
//! OpenAI-compatible LLM backend for the enrichment pass (and `ask`).
//! The wizard walks through that — LM Studio being the happy path —
//! verifies the endpoint, token, and model live, then writes config.toml
//! and offers the first scan.

use std::io::{BufRead, Write};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::config::{Config, LoadedConfig};

/// Suggested (never required) on Apple Silicon with 32 GB+ RAM:
/// multimodal (one model covers text and vision passes), fast, and
/// strong enough for structured extraction. Any instruction-tuned model
/// on any OpenAI-compatible server works.
const SUGGESTED_MODEL_FRAGMENT: &str = "gemma-4-12b";

pub fn run_setup(loaded: &LoadedConfig) -> Result<()> {
    let mut config = loaded.config.clone();
    let stdin = std::io::stdin();
    let mut input = stdin.lock();

    println!("ai-icloud setup");
    println!("===============\n");

    step_source_root(&mut config, &mut input)?;
    step_llm_backend(&mut config, &mut input)?;
    step_exclusions(&mut config, &mut input)?;
    step_transcription(&mut config, &mut input)?;
    write_config(&config, &loaded.path)?;

    println!("\nSetup complete. Next steps:");
    println!("  ai-icloud scan --dry-run   # preview what will be indexed");
    println!("  ai-icloud scan             # index, enrich, embed (first run downloads models)");
    println!("  ai-icloud service install  # background daemon: index changes continuously");
    println!("  ai-icloud connect          # MCP JSON to paste into any agent framework");
    if ask_yes_no(&mut input, "\nRun the first scan now?", false)? {
        crate::cli::first_scan(&LoadedConfig {
            config,
            path: loaded.path.clone(),
            from_file: true,
        })?;
    }
    Ok(())
}

fn step_source_root(config: &mut Config, input: &mut impl BufRead) -> Result<()> {
    println!("Step 1/4 — What to index");
    let root = prompt(input, "iCloud/source folder", &config.source.root)?;
    config.source.root = root;
    match config
        .source_root()
        .and_then(|r| crate::scan::scan_tree(&r, &config.source))
    {
        Ok(files) => println!("  ✓ readable — {} in-scope file(s) found\n", files.len()),
        Err(err) => println!(
            "  ! could not scan yet: {err:#}\n    (fix later; `ai-icloud doctor` will guide you)\n"
        ),
    }
    Ok(())
}

fn step_llm_backend(config: &mut Config, input: &mut impl BufRead) -> Result<()> {
    println!("Step 2/4 — LLM backend (used to summarize and extract facts from");
    println!("every document, and for the `ask` tool)\n");
    println!("Any OpenAI-compatible inference server works: LM Studio, Ollama,");
    println!("llama.cpp, vLLM, or a hosted provider — point the base URL at");
    println!("whichever serves you best.\n");
    println!("Happy path on macOS: LM Studio (https://lmstudio.ai)");
    println!("  1. Install LM Studio and download a model in its UI");
    println!("  2. Open the Developer tab → Start Server (default port 1234)");
    println!("  3. If the server requires an API token, copy it from the same tab\n");

    loop {
        let base_url = prompt(input, "LLM base URL", &config.llm.base_url)?;
        config.llm.base_url = base_url.trim_end_matches('/').to_string();

        // A remote provider is a legitimate choice, but sending document
        // content off-machine is opt-in, never a side effect.
        if !crate::embed::is_loopback_url(&config.llm.base_url)
            && !config.privacy.allow_remote_endpoints
        {
            println!(
                "  ! {} is not on this machine — enrichment sends document \
                 content there",
                config.llm.base_url
            );
            if ask_yes_no(input, "  Allow sending document content to this remote endpoint?", false)?
            {
                config.privacy.allow_remote_endpoints = true;
            } else {
                println!("  keeping loopback-only; enter a local URL instead\n");
                continue;
            }
        }

        match probe_models(&config.llm.base_url, config.llm.api_key.as_deref()) {
            Ok(models) => {
                println!("  ✓ endpoint answered with {} model(s)", models.len());
                step_pick_models(config, input, &models)?;
                verify_model(config);
                return Ok(());
            }
            Err(ProbeError::Unauthorized) => {
                println!("  ! the endpoint rejected our credentials (HTTP 401/403)");
                let key = prompt(input, "API token (LM Studio: Developer tab)", "")?;
                if key.is_empty() {
                    println!("  no token given; you can set [llm] api_key later\n");
                } else {
                    config.llm.api_key = Some(key);
                    continue; // re-probe with the token
                }
            }
            Err(ProbeError::Unreachable(err)) => {
                println!("  ! could not reach {}: {err}", config.llm.base_url);
                if !ask_yes_no(input, "  Retry (server started now)?", true)? {
                    println!(
                        "  Skipping — indexing still works; enrichment and `ask` \
                         will wait until the endpoint is up.\n"
                    );
                }
                if config.llm.api_key.is_none() {
                    continue;
                }
            }
        }
        return Ok(());
    }
}

fn step_pick_models(
    config: &mut Config,
    input: &mut impl BufRead,
    models: &[String],
) -> Result<()> {
    if models.is_empty() {
        println!(
            "  ! the server lists no models yet — download/load one in your \
             server's UI (LM Studio: the search tab), then set [llm] model later"
        );
    } else {
        println!("  available models:");
        for m in models.iter().take(15) {
            println!("    - {m}");
        }
        if models.len() > 15 {
            println!("    … and {} more", models.len() - 15);
        }
    }
    let capable = apple_silicon_32gb();
    if capable {
        println!(
            "  suggestion for this machine (Apple Silicon, 32 GB+): a \
             gemma-4-12b variant — multimodal, so one model covers text and \
             vision passes. Any instruction-tuned model works."
        );
    } else {
        println!(
            "  pick any instruction-tuned model your hardware runs well; \
             a multimodal one lets a single model cover text and vision \
             passes, otherwise set a separate vision model."
        );
    }
    let default = recommend_model(models, &config.llm.model, capable);
    let model = prompt(input, "enrichment model", &default)?;
    config.llm.model = model;
    let vision_default = if config.llm.vision_model.is_empty() {
        config.llm.model.clone()
    } else {
        config.llm.vision_model.clone()
    };
    let vision = prompt(
        input,
        "vision model (for PDF pages/images; same is fine if multimodal)",
        &vision_default,
    )?;
    config.llm.vision_model = vision;
    Ok(())
}

/// Existing choice wins if the server still has it; on capable hardware
/// a gemma-4-12b variant is suggested when present; otherwise the first
/// thing that is not an embedding model.
fn recommend_model(models: &[String], existing: &str, capable_hardware: bool) -> String {
    if !existing.is_empty() && models.iter().any(|m| m == existing) {
        return existing.to_string();
    }
    if capable_hardware
        && let Some(m) = models.iter().find(|m| m.contains(SUGGESTED_MODEL_FRAGMENT))
    {
        return m.clone();
    }
    models
        .iter()
        .find(|m| !m.contains("embed"))
        .cloned()
        .unwrap_or_else(|| existing.to_string())
}

/// Apple Silicon with 32 GB+ of unified memory — the tier where a 12B
/// multimodal model is a comfortable suggestion.
fn apple_silicon_32gb() -> bool {
    if std::env::consts::ARCH != "aarch64" {
        return false;
    }
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok())
        .is_some_and(|bytes| bytes >= 32 * 1024 * 1024 * 1024)
}

/// One tiny completion so a bad model id or a broken server surfaces now,
/// not on the first scan.
fn verify_model(config: &Config) {
    print!("  verifying the model answers (first call may load it — up to a minute)… ");
    let _ = std::io::stdout().flush();
    let mut req = ureq::post(&format!("{}/chat/completions", config.llm.base_url))
        .timeout(Duration::from_secs(300));
    if let Some(key) = &config.llm.api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    let mut body = serde_json::json!({
        "messages": [{ "role": "user", "content": "Reply with the single word OK." }],
        "max_tokens": 512,
    });
    if !config.llm.model.is_empty() {
        body["model"] = serde_json::json!(config.llm.model);
    }
    match req.send_json(body) {
        Ok(_) => println!("✓\n"),
        Err(err) => println!(
            "failed ({err})\n  enrichment will retry automatically; check the model id \
             and that the server is running\n"
        ),
    }
}

fn step_exclusions(config: &mut Config, input: &mut impl BufRead) -> Result<()> {
    println!("Step 3/4 — Privacy boundary");
    println!("Excluded folders are never read and never enter the database");
    println!("(anything indexed becomes readable by every MCP-connected agent).");
    let current = config.source.exclude_globs.join(", ");
    let raw = prompt(
        input,
        "exclude globs, comma-separated (e.g. Private/**, Divorce/**) — empty for none",
        &current,
    )?;
    config.source.exclude_globs = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    println!();
    Ok(())
}

fn step_transcription(config: &mut Config, input: &mut impl BufRead) -> Result<()> {
    println!("Step 4/4 — Audio/video transcription (whisper.cpp, fully local)");
    let has_ffmpeg = std::process::Command::new(crate::extract::media::ffmpeg_bin())
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !has_ffmpeg {
        println!("  ! ffmpeg not found — needed to decode media (`brew install ffmpeg`)");
    }
    config.transcription.enabled = ask_yes_no(
        input,
        "Transcribe audio/video files? (first media file downloads ~1.6 GB model)",
        config.transcription.enabled && has_ffmpeg,
    )?;
    println!();
    Ok(())
}

fn write_config(config: &Config, path: &std::path::Path) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if path.exists() {
        let backup = path.with_extension("toml.bak");
        std::fs::copy(path, &backup)
            .with_context(|| format!("could not back up {}", path.display()))?;
        println!("(previous config backed up to {})", backup.display());
    }
    std::fs::write(path, config.to_toml()?)
        .with_context(|| format!("could not write {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    println!("wrote {}", path.display());
    Ok(())
}

// ------------------------------------------------------------- prompting

/// Ask with a visible default; empty input (or EOF, for piped stdin)
/// keeps the default.
fn prompt(input: &mut impl BufRead, question: &str, default: &str) -> Result<String> {
    if default.is_empty() {
        print!("{question}: ");
    } else {
        print!("{question} [{default}]: ");
    }
    std::io::stdout().flush()?;
    let mut line = String::new();
    input.read_line(&mut line)?;
    let answer = line.trim();
    Ok(if answer.is_empty() {
        default.to_string()
    } else {
        answer.to_string()
    })
}

fn ask_yes_no(input: &mut impl BufRead, question: &str, default: bool) -> Result<bool> {
    let hint = if default { "Y/n" } else { "y/N" };
    let answer = prompt(input, &format!("{question} ({hint})"), "")?;
    Ok(match answer.to_lowercase().as_str() {
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default,
    })
}

// --------------------------------------------------------------- probing

enum ProbeError {
    Unauthorized,
    Unreachable(String),
}

fn probe_models(base_url: &str, api_key: Option<&str>) -> Result<Vec<String>, ProbeError> {
    let mut req = ureq::get(&format!("{base_url}/models")).timeout(Duration::from_secs(5));
    if let Some(key) = api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    match req.call() {
        Ok(resp) => {
            let body: serde_json::Value = resp.into_json().unwrap_or_default();
            Ok(body
                .get("data")
                .and_then(|d| d.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|m| m.get("id").and_then(|i| i.as_str()))
                        .map(String::from)
                        .collect()
                })
                .unwrap_or_default())
        }
        Err(ureq::Error::Status(401 | 403, _)) => Err(ProbeError::Unauthorized),
        Err(err) => Err(ProbeError::Unreachable(err.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn prompt_defaults_on_empty_or_eof() {
        let mut input = Cursor::new(b"\n".to_vec());
        assert_eq!(prompt(&mut input, "q", "dflt").unwrap(), "dflt");
        let mut eof = Cursor::new(Vec::new());
        assert_eq!(prompt(&mut eof, "q", "dflt").unwrap(), "dflt");
    }

    #[test]
    fn prompt_takes_the_typed_answer() {
        let mut input = Cursor::new(b"  custom  \n".to_vec());
        assert_eq!(prompt(&mut input, "q", "dflt").unwrap(), "custom");
    }

    #[test]
    fn yes_no_honors_defaults_and_answers() {
        let mut input = Cursor::new(b"\nn\ny\n".to_vec());
        assert!(ask_yes_no(&mut input, "q", true).unwrap());
        assert!(!ask_yes_no(&mut input, "q", true).unwrap());
        assert!(ask_yes_no(&mut input, "q", false).unwrap());
    }

    #[test]
    fn recommend_prefers_existing_then_suggestion_then_non_embedding() {
        let models = vec![
            "text-embedding-nomic-embed-text".to_string(),
            "google/gemma-4-12b-qat".to_string(),
            "qwen/qwen3.6-27b".to_string(),
        ];
        assert_eq!(
            recommend_model(&models, "qwen/qwen3.6-27b", true),
            "qwen/qwen3.6-27b"
        );
        assert_eq!(recommend_model(&models, "", true), "google/gemma-4-12b-qat");
        assert_eq!(
            recommend_model(&models, "gone-model", true),
            "google/gemma-4-12b-qat"
        );
        // The gemma suggestion is hardware-gated; smaller machines get
        // the first non-embedding model instead.
        assert_eq!(
            recommend_model(&models, "", false),
            "google/gemma-4-12b-qat" // still first non-embedding here
        );
        let no_gemma = vec![
            "text-embedding-nomic-embed-text".to_string(),
            "llama3.2".to_string(),
        ];
        assert_eq!(recommend_model(&no_gemma, "", true), "llama3.2");
        assert_eq!(recommend_model(&no_gemma, "", false), "llama3.2");
    }
}
