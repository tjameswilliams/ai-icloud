//! Per-type content extractors: turn a file on disk into per-page text.
//!
//! The plain-text family (txt/md/html/csv) reads directly; PDFs and
//! images go through the embedded Vision sidecar (hybrid text-layer/OCR).
//! Audio/video extractors arrive in a later phase; their kinds are
//! recognized now so scanned files wait in `pending` state.

use std::path::Path;

use anyhow::Result;

use crate::sidecar::Sidecar;

pub mod image;
pub mod media;
pub mod pdf;
pub mod text;

/// What a file is, derived from its (lowercased) extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Text,
    Markdown,
    Html,
    Csv,
    Pdf,
    Image,
    Audio,
    Video,
}

impl FileKind {
    pub fn from_extension(ext: &str) -> Option<FileKind> {
        match ext.to_ascii_lowercase().as_str() {
            "txt" => Some(FileKind::Text),
            "md" | "markdown" => Some(FileKind::Markdown),
            "html" | "htm" => Some(FileKind::Html),
            "csv" => Some(FileKind::Csv),
            "pdf" => Some(FileKind::Pdf),
            "png" | "jpg" | "jpeg" | "heic" => Some(FileKind::Image),
            "mp3" | "m4a" | "wav" => Some(FileKind::Audio),
            "mp4" | "mov" => Some(FileKind::Video),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            FileKind::Text => "text",
            FileKind::Markdown => "markdown",
            FileKind::Html => "html",
            FileKind::Csv => "csv",
            FileKind::Pdf => "pdf",
            FileKind::Image => "image",
            FileKind::Audio => "audio",
            FileKind::Video => "video",
        }
    }

    /// Whether an extractor exists in this build. Kinds without one are
    /// recorded as `pending` and picked up when their phase ships.
    pub fn extractable(self) -> bool {
        true
    }

    /// Whether extraction goes through the OCR sidecar.
    pub fn needs_sidecar(self) -> bool {
        matches!(self, FileKind::Pdf | FileKind::Image)
    }

    /// Whether extraction needs the whisper transcriber.
    pub fn needs_transcriber(self) -> bool {
        matches!(self, FileKind::Audio | FileKind::Video)
    }
}

/// Extracted document content, one string per page, ready for chunking.
/// Single-page kinds (text family, images) produce exactly one entry.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedDoc {
    /// Best-effort title; the file stem until enrichment supplies better.
    pub title: Option<String>,
    pub pages: Vec<String>,
}

/// The heavyweight helpers extraction may need, both optional so the
/// text family works with neither.
#[derive(Default)]
pub struct Extractors<'a> {
    pub sidecar: Option<&'a Sidecar>,
    pub transcriber: Option<&'a media::Transcriber>,
}

/// Extract text from `path` according to its kind.
pub fn extract(path: &Path, kind: FileKind, ex: &Extractors<'_>) -> Result<ExtractedDoc> {
    let title = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty());
    let pages = match kind {
        FileKind::Text | FileKind::Markdown | FileKind::Csv => vec![text::read_plain(path)?],
        FileKind::Html => vec![text::read_html(path)?],
        FileKind::Pdf => pdf::extract_pdf(require_sidecar(ex.sidecar)?, path)?,
        FileKind::Image => image::extract_image(require_sidecar(ex.sidecar)?, path)?,
        FileKind::Audio | FileKind::Video => ex
            .transcriber
            .ok_or_else(|| anyhow::anyhow!(
                "transcriber unavailable (transcription disabled or model failed to load)"
            ))?
            .transcribe(path)?,
    };
    Ok(ExtractedDoc { title, pages })
}

fn require_sidecar(sidecar: Option<&Sidecar>) -> Result<&Sidecar> {
    sidecar.ok_or_else(|| anyhow::anyhow!("OCR sidecar unavailable (see `ai-icloud doctor`)"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions_map_to_kinds_case_insensitively() {
        assert_eq!(FileKind::from_extension("PDF"), Some(FileKind::Pdf));
        assert_eq!(FileKind::from_extension("Md"), Some(FileKind::Markdown));
        assert_eq!(FileKind::from_extension("heic"), Some(FileKind::Image));
        assert_eq!(FileKind::from_extension("exe"), None);
    }

    #[test]
    fn every_kind_is_extractable_and_helpers_are_flagged() {
        assert!(FileKind::Csv.extractable());
        assert!(FileKind::Pdf.extractable());
        assert!(FileKind::Audio.extractable());
        assert!(FileKind::Audio.needs_transcriber());
        assert!(FileKind::Video.needs_transcriber());
        assert!(!FileKind::Pdf.needs_transcriber());
    }

    #[test]
    fn sidecar_kinds_are_flagged() {
        assert!(FileKind::Pdf.needs_sidecar());
        assert!(FileKind::Image.needs_sidecar());
        assert!(!FileKind::Text.needs_sidecar());
    }

    #[test]
    fn extract_uses_file_stem_as_title() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("closing statement.txt");
        std::fs::write(&p, "sale price 487500").unwrap();
        let doc = extract(&p, FileKind::Text, &Extractors::default()).unwrap();
        assert_eq!(doc.title.as_deref(), Some("closing statement"));
        assert_eq!(doc.pages, vec!["sale price 487500".to_string()]);
    }

    #[test]
    fn sidecar_kinds_error_without_a_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.pdf");
        std::fs::write(&p, "%PDF").unwrap();
        let err = extract(&p, FileKind::Pdf, &Extractors::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("sidecar"), "{err}");
    }
}
