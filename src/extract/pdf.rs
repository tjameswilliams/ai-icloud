//! PDF extraction: hybrid text-layer / OCR.
//!
//! The embedded text layer is free and exact when present, so it wins per
//! page; pages with no usable layer (scans) are rendered and OCR'd by the
//! Vision sidecar. A page's provenance is invisible downstream — both
//! paths produce plain text per page.

use std::path::Path;

use anyhow::Result;

use crate::sidecar::Sidecar;

/// Render resolution for OCR'd pages. 300 dpi keeps small print in
/// closing statements legible to Vision.
const OCR_DPI: u32 = 300;

/// Upper bound on OCR'd pages per document; text-layer pages are free and
/// uncapped. Beyond this, remaining pages are indexed as empty.
const MAX_OCR_PAGES: usize = 100;

/// A page's text layer is usable when it has enough real characters and
/// is not dominated by replacement garbage from broken encodings.
fn text_layer_usable(text: &str) -> bool {
    let trimmed = text.trim();
    let chars = trimmed.chars().count();
    if chars < 50 {
        return false;
    }
    let garbage = trimmed.chars().filter(|&c| c == '\u{FFFD}').count();
    (garbage as f64) / (chars as f64) < 0.05
}

/// Extract one PDF into per-page text.
pub fn extract_pdf(sidecar: &Sidecar, path: &Path) -> Result<Vec<String>> {
    let layer = sidecar.pdf_text(path)?;
    let mut pages: Vec<String> = vec![String::new(); layer.page_count];
    for page in layer.pages {
        if page.index < pages.len() {
            pages[page.index] = page.text;
        }
    }

    let needs_ocr: Vec<usize> = pages
        .iter()
        .enumerate()
        .filter(|(_, text)| !text_layer_usable(text))
        .map(|(i, _)| i)
        .collect();

    if !needs_ocr.is_empty() {
        let capped: Vec<usize> = needs_ocr.iter().copied().take(MAX_OCR_PAGES).collect();
        if capped.len() < needs_ocr.len() {
            tracing::warn!(
                "{}: OCR capped at {MAX_OCR_PAGES} pages; {} pages left unindexed",
                path.display(),
                needs_ocr.len() - capped.len()
            );
        }
        tracing::info!(
            "{}: OCR for {}/{} pages",
            path.display(),
            capped.len(),
            pages.len()
        );
        for ocr_page in sidecar.ocr_pdf(path, &capped, OCR_DPI)? {
            if ocr_page.index < pages.len() {
                tracing::debug!(
                    "page {} OCR confidence {:.2}",
                    ocr_page.index,
                    ocr_page.confidence
                );
                pages[ocr_page.index] = ocr_page.text;
            }
        }
    }
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_or_empty_layers_are_unusable() {
        assert!(!text_layer_usable(""));
        assert!(!text_layer_usable("   \n  "));
        assert!(!text_layer_usable("Page 1"));
    }

    #[test]
    fn normal_prose_layer_is_usable() {
        assert!(text_layer_usable(
            "This closing statement covers the sale of the property at 423 R St, \
             including the final purchase price and seller credits."
        ));
    }

    #[test]
    fn replacement_char_garbage_is_unusable() {
        let garbage = "\u{FFFD}".repeat(30) + "some short real text here to pad the count";
        assert!(!text_layer_usable(&garbage));
    }
}
