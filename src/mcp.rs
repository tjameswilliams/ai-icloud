//! MCP (Model Context Protocol) server: newline-delimited JSON-RPC 2.0
//! over stdio, or stateless HTTP with a bearer token. Tools capability
//! only. Search and reads never mutate; `reindex_file` re-ingests one
//! path on request.
//!
//! The handler is pure — one JSON message in, at most one JSON message
//! out — so the protocol logic is testable without spawning a process.
//! stdout carries protocol frames only; all logging goes to stderr.

use anyhow::Result;
use serde_json::{Value, json};

use crate::config::Config;
use crate::embed::{self, Embedder};
use crate::index::{DocumentInfo, IndexDb};
use crate::ingest;
use crate::retrieve::{RetrievalParams, hybrid_search};

/// Protocol revisions this server knows; the client's choice is echoed
/// when we support it, otherwise we answer with our newest.
const SUPPORTED_PROTOCOLS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Cap on text returned by get_document in one call; continue with
/// `offset`.
const DOC_TEXT_CAP: usize = 40_000;

pub struct McpServer {
    index: IndexDb,
    config: Config,
    /// Lazily constructed on the first search so `initialize` stays fast.
    embedder: Option<Box<dyn Embedder>>,
}

impl McpServer {
    pub fn new(index: IndexDb, config: Config) -> Self {
        McpServer {
            index,
            config,
            embedder: None,
        }
    }

    /// Handle one incoming JSON-RPC message. `None` means nothing is sent
    /// back (notifications, and requests that malformed their own id).
    pub fn handle(&mut self, msg: &Value) -> Option<Value> {
        let id = msg.get("id").filter(|v| !v.is_null()).cloned();
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        // Notifications never get a response, whatever the method.
        let id = id?;

        let outcome = match method {
            "initialize" => Ok(self.initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tool_definitions() })),
            "tools/call" => return Some(self.tools_call(id, &params)),
            "" => Err((-32600, "missing method".to_string())),
            other => Err((-32601, format!("method not found: {other}"))),
        };
        Some(match outcome {
            Ok(result) => rpc_result(id, result),
            Err((code, message)) => rpc_error(id, code, &message),
        })
    }

    fn initialize(&self, params: &Value) -> Value {
        let requested = params
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let version = if SUPPORTED_PROTOCOLS.contains(&requested) {
            requested
        } else {
            SUPPORTED_PROTOCOLS[0]
        };
        json!({
            "protocolVersion": version,
            "capabilities": { "tools": {} },
            "serverInfo": {
                "name": "ai-icloud",
                "version": env!("CARGO_PKG_VERSION"),
            },
            "instructions": "Search over the user's local iCloud Drive \
                document index (OCR'd PDFs, images, text files). Start with \
                search_documents to find documents by topic; use get_document \
                to read a whole document once found, get_chunk to expand one \
                search hit with neighboring text, and list_documents to \
                browse by folder, type, or tag. search_facts queries \
                structured key-value facts (dates, amounts, parties) where \
                enrichment has extracted them. sync_status reports index \
                freshness; reindex_file forces one path to re-ingest.",
        })
    }

    fn tools_call(&mut self, id: Value, params: &Value) -> Value {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        let outcome = match name {
            "search_documents" => self.tool_search(&args),
            "get_document" => self.tool_get_document(&args),
            "get_chunk" => self.tool_get_chunk(&args),
            "list_documents" => self.tool_list_documents(&args),
            "search_facts" => self.tool_search_facts(&args),
            "sync_status" => self.tool_sync_status(),
            "reindex_file" => self.tool_reindex_file(&args),
            other => return rpc_error(id, -32602, &format!("unknown tool: {other}")),
        };
        // Tool execution failures are results with isError, not protocol
        // errors — the model is meant to read them.
        match outcome {
            Ok(text) => rpc_result(
                id,
                json!({ "content": [{ "type": "text", "text": text }], "isError": false }),
            ),
            Err(e) => rpc_result(
                id,
                json!({ "content": [{ "type": "text", "text": format!("error: {e:#}") }], "isError": true }),
            ),
        }
    }

    fn tool_search(&mut self, args: &Value) -> Result<String> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .filter(|q| !q.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("'query' (non-empty string) is required"))?;
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n.clamp(1, 50) as u32)
            .unwrap_or(self.config.retrieval.result_limit);
        let mode = args.get("mode").and_then(Value::as_str).unwrap_or("hybrid");

        let hits = match mode {
            "keyword" => self.index.search(query, limit)?,
            "semantic" => {
                let vec = self
                    .embedder()?
                    .ok_or_else(|| {
                        anyhow::anyhow!("no embeddings in the index; run `ai-icloud scan`")
                    })?
                    .embed_query(query)?;
                self.index.vector_search(&vec, limit)?
            }
            "hybrid" => {
                let query_vec = match self.embedder()? {
                    Some(e) => Some(e.embed_query(query)?),
                    None => None,
                };
                let params = RetrievalParams {
                    fts_candidates: self.config.retrieval.fts_candidates,
                    vector_candidates: self.config.retrieval.vector_candidates,
                    limit,
                };
                hybrid_search(&self.index, query, query_vec.as_deref(), &params)?
            }
            other => anyhow::bail!("unknown mode \"{other}\" (hybrid, keyword, or semantic)"),
        };

        if hits.is_empty() {
            return Ok(format!("No matches for \"{query}\"."));
        }
        let mut out = format!("{} result(s) for \"{query}\"\n", hits.len());
        for h in &hits {
            let text = self
                .index
                .chunk_text(h.chunk_id)?
                .unwrap_or_else(|| h.snippet.clone());
            let pages = self
                .index
                .chunk_window(h.chunk_id, 0, 0)?
                .and_then(|w| w.first().map(|c| format_pages(c.page_start, c.page_end)))
                .unwrap_or_default();
            out.push_str(&format!(
                "\n[chunk {} | doc {}] {}{}{}\n{}\n",
                h.chunk_id,
                h.document_id,
                h.rel_path,
                h.doc_type
                    .as_deref()
                    .map(|t| format!(" ({t})"))
                    .unwrap_or_default(),
                pages,
                text,
            ));
        }
        Ok(out)
    }

    fn tool_get_document(&mut self, args: &Value) -> Result<String> {
        let doc = self.resolve_document(args)?;
        let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;

        let mut out = format!("{} [{}]\n", doc.rel_path, doc.kind);
        if let Some(t) = &doc.title {
            out.push_str(&format!("title: {t}\n"));
        }
        if let Some(t) = &doc.doc_type {
            out.push_str(&format!("type: {t}\n"));
        }
        if !doc.tags.is_empty() {
            out.push_str(&format!("tags: {}\n", doc.tags.join(", ")));
        }
        if let Some(s) = &doc.summary {
            out.push_str(&format!("summary: {s}\n"));
        }
        let doc_facts = self.index.facts_for_document(doc.document_id)?;
        if !doc_facts.is_empty() {
            out.push_str("facts:\n");
            for f in &doc_facts {
                out.push_str(&format!("  {} = {} [{}]\n", f.key, f.value, f.kind));
            }
        }

        let chunks = self.index.document_chunks(doc.document_id)?;
        let full: String = chunks
            .iter()
            .filter(|c| !c.is_summary)
            .map(|c| {
                format!(
                    "{}{}\n",
                    format_pages_header(c.page_start, c.page_end),
                    c.text
                )
            })
            .collect();
        let total = full.chars().count();
        if total == 0 {
            out.push_str("\n(no extracted text)\n");
            return Ok(out);
        }
        if offset >= total {
            out.push_str(&format!("\n(offset {offset} is past the end; text is {total} chars)\n"));
            return Ok(out);
        }
        let slice: String = full.chars().skip(offset).take(DOC_TEXT_CAP).collect();
        let end = offset + slice.chars().count();
        out.push_str(&format!("\n--- text ({total} chars, showing {offset}..{end}) ---\n"));
        out.push_str(&slice);
        if end < total {
            out.push_str(&format!(
                "\n--- truncated; call again with offset={end} for more ---\n"
            ));
        }
        Ok(out)
    }

    fn tool_get_chunk(&mut self, args: &Value) -> Result<String> {
        let chunk_id = args
            .get("chunk_id")
            .and_then(Value::as_i64)
            .ok_or_else(|| anyhow::anyhow!("'chunk_id' (integer) is required"))?;
        let before = args
            .get("before")
            .and_then(Value::as_u64)
            .map(|n| n.min(20) as u32)
            .unwrap_or(1);
        let after = args
            .get("after")
            .and_then(Value::as_u64)
            .map(|n| n.min(20) as u32)
            .unwrap_or(1);

        let window = self
            .index
            .chunk_window(chunk_id, before, after)?
            .ok_or_else(|| anyhow::anyhow!("no chunk with id {chunk_id}"))?;
        let doc = window
            .first()
            .and_then(|c| self.index.document_by_id(c.document_id).transpose())
            .transpose()?;
        let mut out = match &doc {
            Some(d) => format!("{} — chunks around {chunk_id}:\n", d.rel_path),
            None => format!("chunks around {chunk_id}:\n"),
        };
        for c in &window {
            let marker = if c.id == chunk_id { "→" } else { " " };
            out.push_str(&format!(
                "\n{marker} [chunk {}]{}\n{}\n",
                c.id,
                format_pages(c.page_start, c.page_end),
                c.text
            ));
        }
        Ok(out)
    }

    fn tool_list_documents(&mut self, args: &Value) -> Result<String> {
        let filter = optional_str(args, "filter");
        let doc_type = optional_str(args, "doc_type");
        let tag = optional_str(args, "tag");
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n.clamp(1, 200) as u32)
            .unwrap_or(50);

        let docs = self
            .index
            .list_documents(filter.as_deref(), doc_type.as_deref(), tag.as_deref(), limit)?;
        if docs.is_empty() {
            return Ok("No documents match.".into());
        }
        let mut out = format!("{} document(s), most recently indexed first\n", docs.len());
        for d in &docs {
            out.push_str(&format!("- {}", format_doc_line(d)));
        }
        Ok(out)
    }

    fn tool_search_facts(&mut self, args: &Value) -> Result<String> {
        let key = optional_str(args, "key");
        let kind = optional_str(args, "kind");
        let value = optional_str(args, "value_contains");
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n.clamp(1, 500) as u32)
            .unwrap_or(100);
        let facts = self
            .index
            .search_facts(key.as_deref(), kind.as_deref(), value.as_deref(), limit)?;
        if facts.is_empty() {
            return Ok("No facts match. (Facts are populated by LLM enrichment; \
                       if the index is fresh, enrichment may not have run yet — \
                       fall back to search_documents.)"
                .into());
        }
        let mut out = format!("{} fact(s)\n", facts.len());
        for f in &facts {
            let page = f.page.map(|p| format!(" p{p}")).unwrap_or_default();
            out.push_str(&format!(
                "- [doc {}] {}{}: {} = {} [{}]\n",
                f.document_id, f.rel_path, page, f.key, f.value, f.kind
            ));
        }
        Ok(out)
    }

    fn tool_sync_status(&mut self) -> Result<String> {
        let statuses = self.index.status_counts()?;
        let mut out = String::from("Index status\n");
        out.push_str(&format!(
            "last scan: {}\n",
            self.index
                .last_scan_ms()?
                .map(format_ms)
                .unwrap_or_else(|| "never".into())
        ));
        for (status, n) in &statuses {
            out.push_str(&format!("{status}: {n}\n"));
        }
        out.push_str(&format!(
            "chunks: {}, embedded: {}\n",
            self.index.chunk_count()?,
            self.index.embedding_count()?
        ));
        let errors = self.index.error_files(10)?;
        if !errors.is_empty() {
            out.push_str("recent errors:\n");
            for (path, err) in errors {
                out.push_str(&format!("- {path}: {err}\n"));
            }
        }
        Ok(out)
    }

    fn tool_reindex_file(&mut self, args: &Value) -> Result<String> {
        let rel_path = args
            .get("rel_path")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("'rel_path' (non-empty string) is required"))?;
        ingest::reindex_path(&mut self.index, &self.config, rel_path)
    }

    fn resolve_document(&self, args: &Value) -> Result<DocumentInfo> {
        if let Some(id) = args.get("document_id").and_then(Value::as_i64) {
            return self
                .index
                .document_by_id(id)?
                .ok_or_else(|| anyhow::anyhow!("no document with id {id}"));
        }
        if let Some(path) = optional_str(args, "rel_path") {
            return self.index.document_by_path(&path)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "no indexed document at \"{path}\" — try list_documents or sync_status"
                )
            });
        }
        anyhow::bail!("'document_id' (integer) or 'rel_path' (string) is required")
    }

    fn embedder(&mut self) -> Result<Option<&mut Box<dyn Embedder>>> {
        if self.embedder.is_none() {
            if self.index.embedding_count()? == 0 {
                return Ok(None);
            }
            let cache = self.config.index_dir()?.join("models");
            self.embedder = Some(embed::make_embedder(&self.config, &cache)?);
        }
        Ok(self.embedder.as_mut())
    }
}

fn optional_str(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
}

fn format_doc_line(d: &DocumentInfo) -> String {
    let mut line = format!("[doc {}] {} [{}]", d.document_id, d.rel_path, d.kind);
    if let Some(t) = &d.doc_type {
        line.push_str(&format!(" ({t})"));
    }
    if !d.tags.is_empty() {
        line.push_str(&format!(" tags: {}", d.tags.join(",")));
    }
    line.push_str(&format!(
        " — {} chunk(s), modified {}\n",
        d.chunk_count,
        format_ms(d.mtime_ms)
    ));
    line
}

fn format_pages(start: Option<i64>, end: Option<i64>) -> String {
    match (start, end) {
        (Some(s), Some(e)) if s == e => format!(" p{s}"),
        (Some(s), Some(e)) => format!(" p{s}-{e}"),
        _ => String::new(),
    }
}

fn format_pages_header(start: Option<i64>, end: Option<i64>) -> String {
    let p = format_pages(start, end);
    if p.is_empty() {
        String::new()
    } else {
        format!("[page{p} ]\n")
    }
}

fn format_ms(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| "unknown time".into())
}

/// Serve MCP over streamable HTTP: a single `/mcp` endpoint accepting
/// POSTed JSON-RPC messages, answering with `application/json`. Stateless
/// (no session ids) and without server-initiated streams, both of which
/// the spec permits; GET therefore answers 405.
///
/// Every request must carry `Authorization: Bearer <token>` — this serves
/// personal documents, so there is no unauthenticated mode.
pub fn serve_http(server: &mut McpServer, addr: &str, token: &str) -> Result<()> {
    let http =
        tiny_http::Server::http(addr).map_err(|e| anyhow::anyhow!("could not bind {addr}: {e}"))?;
    // The OS picks the port when the caller asked for :0; always report
    // the resolved address so clients (and tests) know where to connect.
    eprintln!("MCP listening on http://{}/mcp", http.server_addr());
    if !addr.starts_with("127.0.0.1") && !addr.starts_with("localhost") && !addr.starts_with("[::1]")
    {
        eprintln!(
            "warning: binding a non-loopback address exposes your documents \
             to anyone on that network who has the token. Prefer 127.0.0.1 \
             or a tailnet address."
        );
    }

    const MAX_BODY_BYTES: u64 = 4 * 1024 * 1024;
    for mut request in http.incoming_requests() {
        let response = handle_http_request(server, &mut request, token, MAX_BODY_BYTES);
        let _ = request.respond(response);
    }
    Ok(())
}

fn handle_http_request(
    server: &mut McpServer,
    request: &mut tiny_http::Request,
    token: &str,
    max_body: u64,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    use tiny_http::{Header, Method, Response};

    let json_response = |status: u16, body: &Value| {
        Response::from_data(body.to_string().into_bytes())
            .with_status_code(status)
            .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
    };
    let empty = |status: u16| Response::from_data(Vec::new()).with_status_code(status);

    if request.url() != "/mcp" {
        return empty(404);
    }
    match request.method() {
        Method::Post => {}
        // No server-initiated streams and no sessions to delete.
        _ => return empty(405),
    }

    let authorized = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| h.value.as_str())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|presented| constant_time_eq(presented.as_bytes(), token.as_bytes()));
    if !authorized {
        return Response::from_data(Vec::new())
            .with_status_code(401)
            .with_header(Header::from_bytes("WWW-Authenticate", "Bearer").unwrap());
    }

    let mut body = String::new();
    use std::io::Read;
    if request
        .as_reader()
        .take(max_body)
        .read_to_string(&mut body)
        .is_err()
    {
        return empty(400);
    }
    let msg: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return json_response(
                400,
                &rpc_error(Value::Null, -32700, &format!("parse error: {e}")),
            );
        }
    };

    match server.handle(&msg) {
        Some(resp) => json_response(200, &resp),
        // Notifications and responses get 202 Accepted with no body.
        None => empty(202),
    }
}

/// Compare secrets without early exit; a timing oracle on a token guarding
/// personal documents is cheap to close.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn rpc_result(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

pub fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "search_documents",
            "description": "Search the local iCloud document index by topic. \
                Returns matching chunks with chunk/document ids, file path, \
                page range, and full chunk text, ranked by relevance. Default \
                mode fuses keyword and semantic ranking.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to search for" },
                    "limit": { "type": "integer", "description": "Max results (default from config, max 50)" },
                    "mode": { "type": "string", "enum": ["hybrid", "keyword", "semantic"],
                              "description": "Retrieval mode (default hybrid)" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_document",
            "description": "Read one whole document: metadata (title, type, \
                tags, summary, extracted facts) plus its full extracted text \
                with page markers. Large documents are paged; follow the \
                truncation note's offset to continue.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "document_id": { "type": "integer", "description": "Document id from search results" },
                    "rel_path": { "type": "string", "description": "Path relative to the iCloud root (alternative to document_id)" },
                    "offset": { "type": "integer", "description": "Character offset to continue a truncated read" }
                }
            }
        },
        {
            "name": "get_chunk",
            "description": "Expand a search hit: one chunk plus neighboring \
                chunks from the same document, in order.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_id": { "type": "integer", "description": "Chunk id from search_documents" },
                    "before": { "type": "integer", "description": "Neighbor chunks before (default 1, max 20)" },
                    "after": { "type": "integer", "description": "Neighbor chunks after (default 1, max 20)" }
                },
                "required": ["chunk_id"]
            }
        },
        {
            "name": "list_documents",
            "description": "Browse indexed documents, most recently indexed \
                first. Filter by path/title substring, doc_type, or tag.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filter": { "type": "string", "description": "Substring of the path or title (e.g. a folder name)" },
                    "doc_type": { "type": "string", "description": "Exact document type from enrichment" },
                    "tag": { "type": "string", "description": "Exact tag from enrichment" },
                    "limit": { "type": "integer", "description": "Max documents (default 50, max 200)" }
                }
            }
        },
        {
            "name": "search_facts",
            "description": "Query structured facts extracted from documents \
                (dates, dollar amounts, parties, addresses). Ideal for \
                questions like 'what was the sale price' once enrichment has \
                run; falls back gracefully when empty.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "Fact key substring (e.g. sale_price)" },
                    "kind": { "type": "string", "enum": ["date", "amount", "party", "address", "other"],
                              "description": "Fact kind" },
                    "value_contains": { "type": "string", "description": "Substring of the value" },
                    "limit": { "type": "integer", "description": "Max facts (default 100, max 500)" }
                }
            }
        },
        {
            "name": "sync_status",
            "description": "Index freshness: last scan time, file counts by \
                status (indexed/pending/evicted/error), chunk and embedding \
                counts, recent errors.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "reindex_file",
            "description": "Force one file to re-extract, re-chunk, and \
                re-embed now (e.g. after it changed or previously errored).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rel_path": { "type": "string", "description": "Path relative to the iCloud root" }
                },
                "required": ["rel_path"]
            }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkPiece;
    use tempfile::TempDir;

    fn server() -> (TempDir, McpServer) {
        let dir = TempDir::new().unwrap();
        let mut db = IndexDb::open(&dir.path().join("index.sqlite")).unwrap();
        let piece = |seq: i64, text: &str| ChunkPiece {
            seq,
            is_summary: false,
            page_start: Some(seq + 1),
            page_end: Some(seq + 1),
            text: text.to_string(),
            content_hash: crate::chunk::content_hash(text),
        };
        db.upsert_indexed_file(
            "House/closing.pdf",
            "pdf",
            10,
            1000,
            "sha1",
            Some("closing"),
            Some(2),
            &[
                piece(0, "Sale price of property 487500 dollars"),
                piece(1, "Net proceeds to seller 250000 dollars"),
            ],
            2000,
        )
        .unwrap();
        db.upsert_indexed_file(
            "notes.txt",
            "text",
            5,
            1000,
            "sha2",
            Some("notes"),
            Some(1),
            &[piece(0, "buy milk and coffee")],
            2000,
        )
        .unwrap();
        let mut config = Config::default();
        config.embeddings.provider = "debug-hash".into();
        config.index.database_path = dir
            .path()
            .join("index.sqlite")
            .to_string_lossy()
            .into_owned();
        (dir, McpServer::new(db, config))
    }

    fn call(server: &mut McpServer, name: &str, args: Value) -> (String, bool) {
        let msg = json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": { "name": name, "arguments": args }
        });
        let resp = server.handle(&msg).unwrap();
        let result = &resp["result"];
        (
            result["content"][0]["text"].as_str().unwrap().to_string(),
            result["isError"].as_bool().unwrap(),
        )
    }

    #[test]
    fn initialize_echoes_supported_protocol_and_names_server() {
        let (_d, mut s) = server();
        let resp = s
            .handle(&json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": { "protocolVersion": "2025-03-26" }
            }))
            .unwrap();
        assert_eq!(resp["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(resp["result"]["serverInfo"]["name"], "ai-icloud");
    }

    #[test]
    fn unknown_protocol_gets_our_newest() {
        let (_d, mut s) = server();
        let resp = s
            .handle(&json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": { "protocolVersion": "1999-01-01" }
            }))
            .unwrap();
        assert_eq!(resp["result"]["protocolVersion"], SUPPORTED_PROTOCOLS[0]);
    }

    #[test]
    fn notifications_get_no_response() {
        let (_d, mut s) = server();
        assert!(
            s.handle(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))
                .is_none()
        );
    }

    #[test]
    fn tools_list_names_all_seven_tools() {
        let (_d, mut s) = server();
        let resp = s
            .handle(&json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }))
            .unwrap();
        let names: Vec<&str> = resp["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec![
                "search_documents",
                "get_document",
                "get_chunk",
                "list_documents",
                "search_facts",
                "sync_status",
                "reindex_file"
            ]
        );
    }

    #[test]
    fn search_documents_finds_by_keyword() {
        let (_d, mut s) = server();
        let (text, is_err) =
            call(&mut s, "search_documents", json!({ "query": "sale price" }));
        assert!(!is_err);
        assert!(text.contains("House/closing.pdf"));
        assert!(text.contains("487500"));
    }

    #[test]
    fn search_documents_requires_a_query() {
        let (_d, mut s) = server();
        let (text, is_err) = call(&mut s, "search_documents", json!({}));
        assert!(is_err);
        assert!(text.contains("query"));
    }

    #[test]
    fn get_document_by_path_returns_metadata_and_paged_text() {
        let (_d, mut s) = server();
        let (text, is_err) = call(
            &mut s,
            "get_document",
            json!({ "rel_path": "House/closing.pdf" }),
        );
        assert!(!is_err);
        assert!(text.contains("title: closing"));
        assert!(text.contains("[page p1 ]"));
        assert!(text.contains("Net proceeds"));
    }

    #[test]
    fn get_document_unknown_path_is_a_tool_error() {
        let (_d, mut s) = server();
        let (text, is_err) = call(&mut s, "get_document", json!({ "rel_path": "nope.pdf" }));
        assert!(is_err);
        assert!(text.contains("nope.pdf"));
    }

    #[test]
    fn get_chunk_marks_the_target_and_includes_neighbors() {
        let (_d, mut s) = server();
        let (search, _) = call(&mut s, "search_documents", json!({ "query": "proceeds" }));
        let chunk_id: i64 = search
            .split("[chunk ")
            .nth(1)
            .unwrap()
            .split([' ', '|'])
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let (text, is_err) = call(&mut s, "get_chunk", json!({ "chunk_id": chunk_id }));
        assert!(!is_err);
        assert!(text.contains("→"));
        assert!(text.contains("Sale price"));
        assert!(text.contains("Net proceeds"));
    }

    #[test]
    fn list_documents_filters_by_path_substring() {
        let (_d, mut s) = server();
        let (text, is_err) = call(&mut s, "list_documents", json!({ "filter": "House" }));
        assert!(!is_err);
        assert!(text.contains("House/closing.pdf"));
        assert!(!text.contains("notes.txt"));
    }

    #[test]
    fn search_facts_explains_when_empty() {
        let (_d, mut s) = server();
        let (text, is_err) = call(&mut s, "search_facts", json!({ "key": "sale" }));
        assert!(!is_err);
        assert!(text.contains("enrichment"));
    }

    #[test]
    fn sync_status_reports_counts() {
        let (_d, mut s) = server();
        let (text, is_err) = call(&mut s, "sync_status", json!({}));
        assert!(!is_err);
        assert!(text.contains("indexed: 2"));
        assert!(text.contains("chunks: 3"));
    }

    #[test]
    fn unknown_tool_is_a_protocol_error() {
        let (_d, mut s) = server();
        let resp = s
            .handle(&json!({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": { "name": "steal_documents", "arguments": {} }
            }))
            .unwrap();
        assert_eq!(resp["error"]["code"], -32602);
    }

    #[test]
    fn unknown_method_is_method_not_found() {
        let (_d, mut s) = server();
        let resp = s
            .handle(&json!({ "jsonrpc": "2.0", "id": 7, "method": "resources/list" }))
            .unwrap();
        assert_eq!(resp["error"]["code"], -32601);
        assert_eq!(resp["id"], 7);
    }

    #[test]
    fn constant_time_eq_matches_only_equal_secrets() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
    }
}
