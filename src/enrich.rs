//! Document enrichment: one structured LLM call per document producing a
//! title, doc_type, summary, key-value facts, and tags — stored in the
//! index and (via the summary chunk) searchable. PDFs and images attach
//! rendered page images so the vision model sees layout the OCR text
//! flattens.

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::config::Config;
use crate::extract::FileKind;
use crate::index::{DocumentInfo, EnrichmentRecord, IndexDb};
use crate::llm::{LlmClient, Part};
use crate::sidecar::Sidecar;

/// Characters of extracted text offered to the model per document.
const TEXT_CAP: usize = 16_000;

/// Page images attached per PDF (first pages carry the identifying
/// content in statements and forms).
const MAX_PAGE_IMAGES: usize = 2;

/// Render DPI for enrichment images; layout matters, print quality not.
const RENDER_DPI: u32 = 150;

#[derive(Debug, Default, PartialEq)]
pub struct EnrichReport {
    pub enriched: usize,
    pub failed: usize,
}

impl EnrichReport {
    pub fn summary(&self) -> String {
        format!("{} enriched, {} failed", self.enriched, self.failed)
    }
}

/// Enrich every document still lacking a pass (or all, with `force`).
/// Individual failures are logged and skipped; the document stays
/// un-enriched and retries on the next run.
pub fn enrich_pending(
    db: &mut IndexDb,
    config: &Config,
    force: bool,
    limit: u32,
) -> Result<EnrichReport> {
    let docs = db.documents_needing_enrichment(force, limit)?;
    if docs.is_empty() {
        return Ok(EnrichReport::default());
    }
    let llm = LlmClient::from_config(config)?;
    let mut sidecar: Option<Sidecar> = None;
    let root = config.source_root()?;
    let mut report = EnrichReport::default();

    tracing::info!("enriching {} document(s)", docs.len());
    for doc in docs {
        let needs_images = matches!(
            FileKind::from_extension(doc.rel_path.rsplit('.').next().unwrap_or("")),
            Some(k) if k.needs_sidecar()
        );
        if needs_images && sidecar.is_none() {
            sidecar = Some(Sidecar::ensure(&config.index_dir()?.join("bin"))?);
        }
        match enrich_one(db, &llm, sidecar.as_ref(), &root, &doc) {
            Ok(()) => {
                report.enriched += 1;
                tracing::info!("enriched {}", doc.rel_path);
            }
            Err(err) => {
                report.failed += 1;
                tracing::warn!("enrichment failed for {}: {err:#}", doc.rel_path);
            }
        }
    }
    Ok(report)
}

fn enrich_one(
    db: &mut IndexDb,
    llm: &LlmClient,
    sidecar: Option<&Sidecar>,
    root: &std::path::Path,
    doc: &DocumentInfo,
) -> Result<()> {
    let chunks = db.document_chunks(doc.document_id)?;
    let mut text = String::new();
    for c in chunks.iter().filter(|c| !c.is_summary) {
        if text.chars().count() >= TEXT_CAP {
            break;
        }
        text.push_str(&c.text);
        text.push_str("\n\n");
    }
    let text: String = text.chars().take(TEXT_CAP).collect();

    let mut parts = vec![Part::Text(format!(
        "File path: {}\nFile kind: {}\n\nExtracted text (may be OCR, may be empty \
         for pure images):\n---\n{}\n---",
        doc.rel_path, doc.kind, text
    ))];
    parts.extend(page_images(sidecar, root, doc)?);

    let response = llm.chat_json(SYSTEM_PROMPT, &parts, "document_enrichment", &schema())?;
    let record = record_from_json(&response)?;
    db.apply_enrichment(
        doc.document_id,
        &record,
        "llm", // model identity comes from config; the string marks provenance
        chrono::Utc::now().timestamp_millis(),
    )?;
    Ok(())
}

/// Rendered page images for kinds where layout carries meaning. Render
/// failures degrade to text-only enrichment rather than failing the doc.
fn page_images(
    sidecar: Option<&Sidecar>,
    root: &std::path::Path,
    doc: &DocumentInfo,
) -> Result<Vec<Part>> {
    let Some(sidecar) = sidecar else {
        return Ok(Vec::new());
    };
    let abs = root.join(&doc.rel_path);
    let rendered = match doc.kind.as_str() {
        "pdf" => {
            let pages: Vec<usize> =
                (0..doc.page_count.unwrap_or(1).clamp(1, MAX_PAGE_IMAGES as i64) as usize)
                    .collect();
            sidecar
                .render_pdf(&abs, &pages, RENDER_DPI)
                .map(|ps| ps.into_iter().map(|p| p.png_base64).collect::<Vec<_>>())
        }
        "image" => sidecar.render_image(&abs).map(|p| vec![p.png_base64]),
        _ => return Ok(Vec::new()),
    };
    match rendered {
        Ok(images) => Ok(images.into_iter().map(Part::PngBase64).collect()),
        Err(err) => {
            tracing::warn!("could not render {} for enrichment: {err:#}", doc.rel_path);
            Ok(Vec::new())
        }
    }
}

const SYSTEM_PROMPT: &str = "You are indexing a personal document archive \
(household paperwork: closings, taxes, insurance, receipts, IDs, \
screenshots, photos). Given one document's text and, when available, its \
page images, produce metadata for retrieval.\n\
- title: a concise human title naming what the document IS (not its filename).\n\
- doc_type: snake_case category, e.g. closing_statement, tax_form, \
insurance_policy, bank_statement, receipt, invoice, id_document, letter, \
contract, ticket, screenshot, photo, spreadsheet, notes, other.\n\
- summary: 2-4 sentences stating what it is and the key figures, parties, \
dates, and addresses — write the numbers out, they must be searchable.\n\
- facts: every load-bearing value as key/value pairs. keys are snake_case \
and specific (sale_price, closing_date, policy_number, net_proceeds_to_seller). \
values verbatim from the document. kind is one of date, amount, party, \
address, other. page is the 1-based page it appears on, or null.\n\
- tags: 3-8 lowercase topical tags.\n\
Extract only what the document supports; never invent values.";

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "doc_type": { "type": "string" },
            "summary": { "type": "string" },
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": { "type": "string" },
                        "value": { "type": "string" },
                        "kind": { "type": "string", "enum": ["date", "amount", "party", "address", "other"] },
                        "page": { "type": ["integer", "null"] }
                    },
                    "required": ["key", "value", "kind", "page"],
                    "additionalProperties": false
                }
            },
            "tags": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["title", "doc_type", "summary", "facts", "tags"],
        "additionalProperties": false
    })
}

fn record_from_json(v: &Value) -> Result<EnrichmentRecord> {
    let title = v
        .get("title")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from);
    let doc_type = v
        .get("doc_type")
        .and_then(Value::as_str)
        .context("enrichment JSON lacks doc_type")?
        .trim()
        .to_lowercase()
        .replace([' ', '-'], "_");
    let summary = v
        .get("summary")
        .and_then(Value::as_str)
        .context("enrichment JSON lacks summary")?
        .trim()
        .to_string();
    let facts = v
        .get("facts")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|f| {
                    let key = f.get("key")?.as_str()?.trim().to_string();
                    let value = f.get("value")?.as_str()?.trim().to_string();
                    let kind = f.get("kind")?.as_str()?.trim().to_string();
                    if key.is_empty() || value.is_empty() {
                        return None;
                    }
                    let page = f.get("page").and_then(Value::as_i64).filter(|p| *p > 0);
                    Some((key, value, kind, page))
                })
                .collect()
        })
        .unwrap_or_default();
    let tags = v
        .get("tags")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|t| t.as_str())
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();
    Ok(EnrichmentRecord {
        title,
        doc_type,
        summary,
        facts,
        tags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_parses_a_full_response() {
        let v = json!({
            "title": "Closing Disclosure — 423 R St",
            "doc_type": "Closing Statement",
            "summary": "Sale of 423 R St for $485,000.",
            "facts": [
                { "key": "sale_price", "value": "$485,000", "kind": "amount", "page": 1 },
                { "key": "seller", "value": "Timothy Williams", "kind": "party", "page": null },
                { "key": "", "value": "dropped", "kind": "other", "page": null }
            ],
            "tags": ["Real Estate", "closing", ""]
        });
        let r = record_from_json(&v).unwrap();
        assert_eq!(r.doc_type, "closing_statement");
        assert_eq!(r.facts.len(), 2);
        assert_eq!(r.facts[0].3, Some(1));
        assert_eq!(r.facts[1].3, None);
        assert_eq!(r.tags, vec!["real estate", "closing"]);
    }

    #[test]
    fn missing_required_fields_are_errors() {
        assert!(record_from_json(&json!({ "summary": "x" })).is_err());
        assert!(record_from_json(&json!({ "doc_type": "x" })).is_err());
    }

    #[test]
    fn absent_optional_fields_default_cleanly() {
        let r = record_from_json(&json!({ "doc_type": "note", "summary": "s" })).unwrap();
        assert_eq!(r.title, None);
        assert!(r.facts.is_empty());
        assert!(r.tags.is_empty());
    }

    #[test]
    fn schema_is_valid_json_and_requires_the_core_fields() {
        let s = schema();
        let required: Vec<&str> = s["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"doc_type"));
        assert!(required.contains(&"summary"));
        assert!(required.contains(&"facts"));
    }
}
