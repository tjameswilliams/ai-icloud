//! The `ask` tool: a server-side research loop. The local LLM iterates —
//! search the index, read documents and facts, then synthesize an answer
//! citing the documents it used. Quality is capped by the local model;
//! callers wanting more take the retrieval tools and reason themselves.

use anyhow::Result;
use serde_json::{Value, json};

use crate::config::Config;
use crate::embed::Embedder;
use crate::index::IndexDb;
use crate::llm::{LlmClient, Part};
use crate::retrieve::{RetrievalParams, hybrid_search};

/// Per-action caps keeping one loop iteration's context bounded.
const SEARCH_LIMIT: u32 = 8;
const DOC_CHARS: usize = 12_000;
const TRANSCRIPT_CHARS: usize = 60_000;

pub fn ask(
    db: &IndexDb,
    config: &Config,
    embedder: Option<&mut Box<dyn Embedder>>,
    question: &str,
) -> Result<String> {
    let llm = LlmClient::from_config(config)?;
    let mut embedder = embedder;
    let mut transcript = format!("Question: {question}\n");
    let max_iterations = config.research.max_iterations.max(1);

    for iteration in 0..max_iterations {
        let last = iteration + 1 == max_iterations;
        let directive = if last {
            "This is your FINAL step: you must answer now with what you have."
        } else {
            "Choose your next action."
        };
        let parts = [Part::Text(format!(
            "{transcript}\n{directive}\n\
             Actions: search (query the document index), get_document \
             (read one document by id, with its facts), answer (final \
             answer to the user's question, citing rel_paths)."
        ))];
        let response = llm.chat_json(SYSTEM_PROMPT, &parts, "research_action", &schema())?;
        let action = response
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("answer");

        match action {
            "search" => {
                let query = response
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or(question);
                let query_vec = match embedder.as_deref_mut() {
                    Some(e) => Some(e.embed_query(query)?),
                    None => None,
                };
                let hits = hybrid_search(
                    db,
                    query,
                    query_vec.as_deref(),
                    &RetrievalParams {
                        fts_candidates: config.retrieval.fts_candidates,
                        vector_candidates: config.retrieval.vector_candidates,
                        limit: SEARCH_LIMIT,
                    },
                )?;
                transcript.push_str(&format!("\n[searched: \"{query}\"]\n"));
                if hits.is_empty() {
                    transcript.push_str("no results\n");
                }
                for h in &hits {
                    transcript.push_str(&format!(
                        "- doc {} ({}): {}\n",
                        h.document_id,
                        h.rel_path,
                        h.snippet.replace('\n', " ")
                    ));
                }
            }
            "get_document" => {
                let id = response.get("document_id").and_then(Value::as_i64);
                match id.and_then(|id| db.document_by_id(id).transpose()) {
                    Some(Ok(doc)) => {
                        let mut body = String::new();
                        if let Some(s) = &doc.summary {
                            body.push_str(&format!("summary: {s}\n"));
                        }
                        for f in db.facts_for_document(doc.document_id)? {
                            body.push_str(&format!("fact: {} = {} [{}]\n", f.key, f.value, f.kind));
                        }
                        let text: String = db
                            .document_chunks(doc.document_id)?
                            .iter()
                            .filter(|c| !c.is_summary)
                            .map(|c| c.text.as_str())
                            .collect::<Vec<_>>()
                            .join("\n")
                            .chars()
                            .take(DOC_CHARS)
                            .collect();
                        transcript.push_str(&format!(
                            "\n[read doc {} ({})]\n{body}{text}\n",
                            doc.document_id, doc.rel_path
                        ));
                    }
                    _ => transcript.push_str(&format!("\n[no document with id {id:?}]\n")),
                }
            }
            _ => {
                let answer = response
                    .get("answer")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                if !answer.is_empty() {
                    return Ok(answer.to_string());
                }
                transcript.push_str("\n[answer action carried no text; continue]\n");
            }
        }

        // Keep the rolling transcript bounded from the front (older
        // findings age out; the question is re-stated every loop).
        if transcript.chars().count() > TRANSCRIPT_CHARS {
            let tail: String = transcript
                .chars()
                .skip(transcript.chars().count() - TRANSCRIPT_CHARS)
                .collect();
            transcript = format!("Question: {question}\n…(earlier findings trimmed)…\n{tail}");
        }
    }
    anyhow::bail!("research loop ended without an answer — try search_documents directly")
}

const SYSTEM_PROMPT: &str = "You research questions against a personal \
document index (household paperwork: closings, taxes, insurance, receipts). \
Search with short keyword-style queries, read the most promising documents, \
and answer with concrete figures and dates, naming the source file paths. \
If the index cannot answer, say exactly what is missing. Never invent \
values.";

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": { "type": "string", "enum": ["search", "get_document", "answer"] },
            "query": { "type": ["string", "null"] },
            "document_id": { "type": ["integer", "null"] },
            "answer": { "type": ["string", "null"] }
        },
        "required": ["action", "query", "document_id", "answer"],
        "additionalProperties": false
    })
}
