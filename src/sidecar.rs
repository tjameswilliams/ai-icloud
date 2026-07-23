//! The embedded Swift OCR sidecar (PDFKit + Vision).
//!
//! build.rs compiles `swift/ocr-helper.swift` and the resulting binary is
//! embedded here; at runtime it is materialized into the index directory
//! (refreshed whenever the embedded bytes change) and invoked per file.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const HELPER_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ocr-helper"));

#[derive(Debug, Deserialize)]
pub struct ProbeReply {
    pub ok: bool,
    pub version: u32,
}

#[derive(Debug, Deserialize)]
pub struct TextPage {
    pub index: usize,
    pub text: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextReply {
    pub page_count: usize,
    pub pages: Vec<TextPage>,
}

#[derive(Debug, Deserialize)]
pub struct OcrPage {
    pub index: usize,
    pub text: String,
    pub confidence: f64,
}

#[derive(Debug, Deserialize)]
pub struct OcrReply {
    pub pages: Vec<OcrPage>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPage {
    pub index: usize,
    pub png_base64: String,
}

#[derive(Debug, Deserialize)]
pub struct RenderReply {
    pub pages: Vec<RenderPage>,
}

#[derive(Debug)]
pub struct Sidecar {
    bin: PathBuf,
}

impl Sidecar {
    /// Materialize the embedded helper under `dir` (created if needed) and
    /// verify it runs.
    pub fn ensure(dir: &Path) -> Result<Sidecar> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("could not create {}", dir.display()))?;
        let bin = dir.join("ocr-helper");

        let embedded_hash = Sha256::digest(HELPER_BYTES);
        let current_matches = std::fs::read(&bin)
            .map(|bytes| Sha256::digest(&bytes) == embedded_hash)
            .unwrap_or(false);
        if !current_matches {
            // Unlink first: overwriting an executable in place poisons the
            // macOS code-signature cache and later execs die with SIGKILL.
            match std::fs::remove_file(&bin) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(e).with_context(|| format!("could not replace {}", bin.display()));
                }
            }
            std::fs::write(&bin, HELPER_BYTES)
                .with_context(|| format!("could not write {}", bin.display()))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755))
                    .with_context(|| format!("could not chmod {}", bin.display()))?;
            }
        }

        let sidecar = Sidecar { bin };
        let probe: ProbeReply = sidecar.run(&["probe"])?;
        if !probe.ok {
            bail!("OCR sidecar probe reported not-ok");
        }
        Ok(sidecar)
    }

    /// Text layer of every page (fast; no rendering).
    pub fn pdf_text(&self, path: &Path) -> Result<TextReply> {
        self.run(&["pdf-text", &path.to_string_lossy()])
    }

    /// Render + OCR the given (0-based) pages.
    pub fn ocr_pdf(&self, path: &Path, pages: &[usize], dpi: u32) -> Result<Vec<OcrPage>> {
        if pages.is_empty() {
            return Ok(Vec::new());
        }
        let csv = pages
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let reply: OcrReply = self.run(&[
            "ocr-pdf",
            &path.to_string_lossy(),
            &csv,
            &dpi.to_string(),
        ])?;
        Ok(reply.pages)
    }

    /// Render (0-based) PDF pages to downscaled base64 PNGs for the
    /// vision model.
    pub fn render_pdf(&self, path: &Path, pages: &[usize], dpi: u32) -> Result<Vec<RenderPage>> {
        if pages.is_empty() {
            return Ok(Vec::new());
        }
        let csv = pages
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let reply: RenderReply = self.run(&[
            "render-pdf",
            &path.to_string_lossy(),
            &csv,
            &dpi.to_string(),
        ])?;
        Ok(reply.pages)
    }

    /// Re-encode any readable image (incl. HEIC) as a downscaled base64
    /// PNG for the vision model.
    pub fn render_image(&self, path: &Path) -> Result<RenderPage> {
        let mut reply: RenderReply = self.run(&["render-image", &path.to_string_lossy()])?;
        reply
            .pages
            .pop()
            .context("sidecar returned no rendered image")
    }

    /// OCR a standalone image (png/jpg/heic).
    pub fn ocr_image(&self, path: &Path) -> Result<OcrPage> {
        let mut reply: OcrReply = self.run(&["ocr-image", &path.to_string_lossy()])?;
        reply
            .pages
            .pop()
            .context("sidecar returned no page for the image")
    }

    fn run<T: serde::de::DeserializeOwned>(&self, args: &[&str]) -> Result<T> {
        let output = Command::new(&self.bin)
            .args(args)
            .output()
            .with_context(|| format!("could not run {}", self.bin.display()))?;
        if !output.status.success() {
            bail!(
                "ocr-helper {} failed: {}",
                args.first().unwrap_or(&""),
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "ocr-helper {} returned unexpected JSON",
                args.first().unwrap_or(&"")
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn ensure_materializes_probes_and_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let sidecar = Sidecar::ensure(dir.path()).unwrap();
        assert!(sidecar.bin.exists());
        let mtime = std::fs::metadata(&sidecar.bin).unwrap().modified().unwrap();
        // Second ensure leaves the identical binary untouched.
        Sidecar::ensure(dir.path()).unwrap();
        assert_eq!(
            std::fs::metadata(&sidecar.bin).unwrap().modified().unwrap(),
            mtime
        );
    }

    #[test]
    fn corrupted_helper_is_replaced() {
        let dir = TempDir::new().unwrap();
        let sidecar = Sidecar::ensure(dir.path()).unwrap();
        std::fs::write(&sidecar.bin, b"garbage").unwrap();
        let sidecar = Sidecar::ensure(dir.path()).unwrap();
        let probe: ProbeReply = sidecar.run(&["probe"]).unwrap();
        assert!(probe.ok);
    }

    #[test]
    fn missing_pdf_is_a_clean_error() {
        let dir = TempDir::new().unwrap();
        let sidecar = Sidecar::ensure(dir.path()).unwrap();
        let err = sidecar
            .pdf_text(Path::new("/nonexistent.pdf"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("pdf-text"), "{err}");
    }
}
