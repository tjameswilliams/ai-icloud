//! Standalone image extraction: Vision OCR via the sidecar. Photographed
//! receipts, screenshots, and scanned single pages become searchable
//! text; captioning of non-text images arrives with the enrichment phase.

use std::path::Path;

use anyhow::Result;

use crate::sidecar::Sidecar;

pub fn extract_image(sidecar: &Sidecar, path: &Path) -> Result<Vec<String>> {
    let page = sidecar.ocr_image(path)?;
    tracing::debug!(
        "{}: image OCR confidence {:.2}",
        path.display(),
        page.confidence
    );
    Ok(vec![page.text])
}
