//! Per-user launchd agents: the watch daemon (KeepAlive) and, opt-in,
//! the MCP HTTP server.
//!
//! launchd runs the binary directly, so macOS attributes any TCC grants
//! to the *binary*, not the terminal that installed it — `install` says
//! so loudly. Logs go next to the index and contain only counts and
//! paths, never document content.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::config::LoadedConfig;

pub const WATCH_LABEL: &str = "com.ai-icloud.watch";
pub const SERVE_LABEL: &str = "com.ai-icloud.serve";
pub const DEFAULT_HTTP_ADDR: &str = "127.0.0.1:8788";

/// Test-only escape hatch: launchd's label namespace is machine-global
/// while plists are per-HOME, so a test suite operating on a sandboxed
/// HOME would otherwise bootout/bootstrap the developer's REAL agents.
/// With this env var set, every launchctl call is skipped and reported
/// as successful.
pub const NO_LAUNCHCTL_ENV: &str = "AI_ICLOUD_NO_LAUNCHCTL";

fn plist_path_for(label: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not determine home directory")?;
    Ok(home
        .join("Library/LaunchAgents")
        .join(format!("{label}.plist")))
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn build_plist(
    label: &str,
    binary: &Path,
    explicit_config: Option<&Path>,
    subcommand: &[&str],
    log: &Path,
) -> String {
    let mut args = vec![binary.display().to_string()];
    if let Some(cfg) = explicit_config {
        args.push("--config".into());
        args.push(cfg.display().to_string());
    }
    args.push("-v".into()); // info-level logs in the log file
    args.extend(subcommand.iter().map(|s| s.to_string()));
    let args_xml: String = args
        .iter()
        .map(|a| format!("    <string>{}</string>\n", xml_escape(a)))
        .collect();

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{label}</string>
  <key>ProgramArguments</key>
  <array>
{args_xml}  </array>
  <key>KeepAlive</key>
  <true/>
  <key>RunAtLoad</key>
  <true/>
  <key>ProcessType</key>
  <string>Background</string>
  <key>StandardOutPath</key>
  <string>{log}</string>
  <key>StandardErrorPath</key>
  <string>{log}</string>
</dict>
</plist>
"#,
        label = xml_escape(label),
        log = xml_escape(&log.display().to_string()),
    )
}

fn uid() -> Result<String> {
    let out = Command::new("id").arg("-u").output()?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn launchctl(args: &[&str]) -> Result<std::process::Output> {
    if std::env::var_os(NO_LAUNCHCTL_ENV).is_some() {
        use std::os::unix::process::ExitStatusExt;
        return Ok(std::process::Output {
            status: std::process::ExitStatus::from_raw(0),
            stdout: Vec::new(),
            stderr: Vec::new(),
        });
    }
    Command::new("launchctl")
        .args(args)
        .output()
        .context("could not run launchctl")
}

/// Bootstrap with retries: right after a bootout, launchd may not have
/// fully released the old instance (or its listening port), making an
/// immediate bootstrap fail transiently.
fn bootstrap_with_retry(uid: &str, label: &str, plist: &Path) -> Result<()> {
    let mut last = String::new();
    for attempt in 0..5 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        let out = launchctl(&[
            "bootstrap",
            &format!("gui/{uid}"),
            &plist.display().to_string(),
        ])?;
        if out.status.success() {
            return Ok(());
        }
        last = String::from_utf8_lossy(&out.stderr).trim().to_string();
    }
    bail!("launchctl bootstrap failed for {label}: {last}");
}

fn write_and_load(plist: &Path, content: &str, label: &str, no_load: bool) -> Result<()> {
    if let Some(dir) = plist.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(plist, content)?;
    println!("wrote {}", plist.display());
    if no_load {
        return Ok(());
    }
    let uid = uid()?;
    // Reload cleanly if an older agent is already bootstrapped.
    let _ = launchctl(&["bootout", &format!("gui/{uid}/{label}")]);
    bootstrap_with_retry(&uid, label, plist)?;
    println!("loaded {label}");
    Ok(())
}

/// Install the watch daemon and — only when `http` is given — the MCP
/// HTTP server agent. `no_load` writes plists without touching launchd.
pub fn install(
    loaded: &LoadedConfig,
    explicit_config: Option<&Path>,
    no_load: bool,
    http: Option<&str>,
) -> Result<()> {
    let binary = std::env::current_exe().context("could not determine the binary's own path")?;
    let logs = loaded.config.index_dir()?.join("logs");
    fs::create_dir_all(&logs)?;

    write_and_load(
        &plist_path_for(WATCH_LABEL)?,
        &build_plist(
            WATCH_LABEL,
            &binary,
            explicit_config,
            &["watch"],
            &logs.join("watch.log"),
        ),
        WATCH_LABEL,
        no_load,
    )?;
    println!("watch daemon indexes iCloud Drive changes continuously");

    if let Some(addr) = http {
        write_and_load(
            &plist_path_for(SERVE_LABEL)?,
            &build_plist(
                SERVE_LABEL,
                &binary,
                explicit_config,
                &["serve", "--http", addr],
                &logs.join("serve.log"),
            ),
            SERVE_LABEL,
            no_load,
        )?;
        println!(
            "MCP HTTP server kept running at http://{addr}/mcp\n\
             run `ai-icloud connect` for the bearer token"
        );
        if !addr.starts_with("127.0.0.1") && !addr.starts_with("localhost") {
            println!(
                "note: {addr} is not loopback — anyone who can reach it and \
                 holds the token can read your documents"
            );
        }
    }

    println!(
        "\nNOTE: launchd runs the binary directly. If the daemon logs \
         permission errors reading iCloud Drive, grant Full Disk Access to \
         the binary itself:\n  System Settings → Privacy & Security → Full \
         Disk Access → add {}\nCheck with: ai-icloud service status",
        binary.display()
    );
    Ok(())
}

fn installed_agents() -> Result<Vec<(&'static str, PathBuf)>> {
    Ok([
        (WATCH_LABEL, plist_path_for(WATCH_LABEL)?),
        (SERVE_LABEL, plist_path_for(SERVE_LABEL)?),
    ]
    .into_iter()
    .filter(|(_, p)| p.exists())
    .collect())
}

/// Load installed agents into launchd (without reinstalling anything).
pub fn start() -> Result<()> {
    let agents = installed_agents()?;
    if agents.is_empty() {
        bail!("nothing to start — run `ai-icloud service install` first");
    }
    let uid = uid()?;
    for (label, plist) in agents {
        let _ = launchctl(&["bootout", &format!("gui/{uid}/{label}")]);
        bootstrap_with_retry(&uid, label, &plist)?;
        println!("started {label}");
    }
    Ok(())
}

/// Unload agents from launchd but keep them installed; `service start`
/// resumes them, and they do not return at reboot until then.
pub fn stop() -> Result<()> {
    let agents = installed_agents()?;
    if agents.is_empty() {
        println!("nothing to stop — no agents installed");
        return Ok(());
    }
    let uid = uid()?;
    for (label, _) in agents {
        let _ = launchctl(&["bootout", &format!("gui/{uid}/{label}")]);
        println!("stopped {label} (still installed; `ai-icloud service start` resumes)");
    }
    Ok(())
}

/// Unload and delete the agents.
pub fn uninstall() -> Result<()> {
    let agents = installed_agents()?;
    if agents.is_empty() {
        println!("nothing installed");
        return Ok(());
    }
    let uid = uid()?;
    for (label, plist) in agents {
        let _ = launchctl(&["bootout", &format!("gui/{uid}/{label}")]);
        fs::remove_file(&plist)?;
        println!("uninstalled {label}");
    }
    Ok(())
}

/// The `--http ADDR` baked into the installed serve agent, if any.
pub fn installed_http_addr() -> Result<Option<String>> {
    let plist = plist_path_for(SERVE_LABEL)?;
    let Ok(content) = fs::read_to_string(&plist) else {
        return Ok(None);
    };
    Ok(parse_http_addr(&content))
}

fn parse_http_addr(plist: &str) -> Option<String> {
    let after = plist.split("<string>--http</string>").nth(1)?;
    let addr = after.split("<string>").nth(1)?.split("</string>").next()?;
    Some(addr.trim().to_string())
}

/// Report install + running state of both agents.
pub fn status(loaded: &LoadedConfig) -> Result<()> {
    let uid = uid()?;
    for label in [WATCH_LABEL, SERVE_LABEL] {
        let plist = plist_path_for(label)?;
        let installed = plist.exists();
        let running = installed
            && launchctl(&["print", &format!("gui/{uid}/{label}")])
                .map(|o| o.status.success())
                .unwrap_or(false);
        let state = match (installed, running) {
            (false, _) => "not installed",
            (true, false) => "installed, not running",
            (true, true) => "running",
        };
        println!("{label}: {state}");
    }
    println!(
        "logs: {}",
        loaded.config.index_dir()?.join("logs").display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plist_contains_binary_config_and_subcommand() {
        let plist = build_plist(
            WATCH_LABEL,
            Path::new("/usr/local/bin/ai-icloud"),
            Some(Path::new("/tmp/cfg & special.toml")),
            &["watch"],
            Path::new("/tmp/watch.log"),
        );
        assert!(plist.contains("<string>com.ai-icloud.watch</string>"));
        assert!(plist.contains("<string>/usr/local/bin/ai-icloud</string>"));
        assert!(plist.contains("<string>--config</string>"));
        // XML escaping of special characters in paths.
        assert!(plist.contains("cfg &amp; special.toml"));
        assert!(plist.contains("<string>watch</string>"));
        assert!(plist.contains("<key>KeepAlive</key>"));
    }

    #[test]
    fn http_addr_parses_out_of_the_plist() {
        let plist = build_plist(
            SERVE_LABEL,
            Path::new("/bin/ai-icloud"),
            None,
            &["serve", "--http", "127.0.0.1:9999"],
            Path::new("/tmp/serve.log"),
        );
        assert_eq!(parse_http_addr(&plist), Some("127.0.0.1:9999".to_string()));
        assert_eq!(parse_http_addr("<plist/>"), None);
    }

    #[test]
    fn serve_plist_bakes_in_the_http_addr() {
        let plist = build_plist(
            SERVE_LABEL,
            Path::new("/bin/ai-icloud"),
            None,
            &["serve", "--http", DEFAULT_HTTP_ADDR],
            Path::new("/tmp/serve.log"),
        );
        assert!(plist.contains("<string>--http</string>"));
        assert!(plist.contains(&format!("<string>{DEFAULT_HTTP_ADDR}</string>")));
        assert!(!plist.contains("--config"));
    }
}
