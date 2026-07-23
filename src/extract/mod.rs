//! Per-type content extractors: turn a file on disk into plain text.
//!
//! Phase 1 covers the plain-text family (txt/md/html/csv). PDF, image,
//! and audio/video extractors arrive in later phases; their kinds are
//! recognized now so scanned files can wait in `pending` state.

use std::path::Path;

use anyhow::Result;

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
        matches!(
            self,
            FileKind::Text | FileKind::Markdown | FileKind::Html | FileKind::Csv
        )
    }
}

/// Extracted document content, ready for chunking.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedDoc {
    /// Best-effort title; the file stem until enrichment supplies better.
    pub title: Option<String>,
    pub text: String,
}

/// Extract plain text from `path` according to its kind.
pub fn extract(path: &Path, kind: FileKind) -> Result<ExtractedDoc> {
    let title = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty());
    let text = match kind {
        FileKind::Text | FileKind::Markdown | FileKind::Csv => text::read_plain(path)?,
        FileKind::Html => text::read_html(path)?,
        other => anyhow::bail!("no extractor for {} files yet", other.as_str()),
    };
    Ok(ExtractedDoc { title, text })
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
    fn only_the_text_family_is_extractable_in_phase_one() {
        assert!(FileKind::Csv.extractable());
        assert!(FileKind::Html.extractable());
        assert!(!FileKind::Pdf.extractable());
        assert!(!FileKind::Audio.extractable());
    }

    #[test]
    fn extract_uses_file_stem_as_title() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("closing statement.txt");
        std::fs::write(&p, "sale price 487500").unwrap();
        let doc = extract(&p, FileKind::Text).unwrap();
        assert_eq!(doc.title.as_deref(), Some("closing statement"));
        assert_eq!(doc.text, "sale price 487500");
    }

    #[test]
    fn unextractable_kind_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.pdf");
        std::fs::write(&p, "%PDF").unwrap();
        assert!(extract(&p, FileKind::Pdf).is_err());
    }
}
