# ai-icloud — Local iCloud Documents RAG Index + MCP Server

Sister project to [ai-imessage](../ai-imessage) ("Local-first Apple Messages RAG index and MCP server").
We deliberately replicate its proven patterns and deviate only where the domain (files vs messages) demands it.

## Purpose

Watch the iCloud Drive folder (`~/Library/Mobile Documents/com~apple~CloudDocs`), OCR / transcribe /
interpret every in-scope file with local models, store text + embeddings + LLM-extracted structure in a
local SQLite database, and expose retrieval + research tools over MCP so any MCP-capable agent can answer
questions like *"when I sold my house, what was my profit?"* against personal documents.

Everything is local-first: OCR via Apple Vision, interpretation via a local OpenAI-compatible LLM endpoint,
embeddings on-device, index stored `0600` under the user's home.

## Decisions (interview, 2026-07-23)

| Decision | Choice |
|---|---|
| OCR strategy | **Hybrid** — Apple Vision framework for raw text; local vision-LLM pass for layout-aware interpretation, tables, structured extraction |
| LLM backend | **Pluggable OpenAI-style endpoint** (`base_url` + `api_key` + `model` in config). **Default: LM Studio** at `http://127.0.0.1:1234/v1` (requires API token). Ollama or anything OpenAI-compatible swaps in via config |
| Change detection | **launchd daemon + FSEvents** (notify crate) with debouncing, plus periodic full-scan reconciliation |
| Scope | Docs (pdf/txt/csv/html/md), images (png/jpg/jpeg/heic), audio/video (mp3/mp4/m4a via embedded whisper.cpp). **Exclusion config** for private folders — excluded material never enters the DB |
| MCP surface | Retrieval + structured facts + **deepResearch-style `ask` tool** (server-side iterative search + synthesis with the local LLM) |
| Enrichment depth | **Full**: per-document summary, doc-type classification, key-value facts (dates, amounts, parties, addresses), tags. Both raw chunks and the summary are embedded |
| Transcription | **whisper-rs** (whisper.cpp, Metal), model auto-downloaded; ffmpeg (doctor-checked) demuxes video audio |

## Environment facts (2026-07-23)

- Corpus: ~144 files, 64 PDF, 32 CSV, ~16 images, ~22 audio/video. Folders include House, Tax, Insurance,
  Credit, Divorce, 423 R St. Small corpus ⇒ brute-force cosine is plenty; the hard part is understanding, not scale.
- Hardware: M4 Max, 128 GB — 27B+ local VLMs are viable.
- LM Studio running on :1234 (token-authed), large model library incl. Qwen3.5-VL-9B, Gemma 3 12B/27B, Gemma 4.
- Ollama installed with only `nomic-embed-text`.
- Zero evicted iCloud files today, but eviction handling is still required (Optimize Mac Storage can change this).

## Architecture

Single binary `ai-icloud`, library + thin `main.rs`, mirroring ai-imessage's module split:

```
src/
  main.rs        // thin wrapper -> cli::run()
  lib.rs
  cli.rs         // clap derive: doctor, scan, watch, search, ask, serve, service, config
  config.rs      // TOML, serde(default, deny_unknown_fields), redaction, tilde paths
  paths.rs       // app-support dir, config path, icloud root discovery
  scan.rs        // full-scan reconciliation: walk tree, stat, hash, diff vs files table
  watch.rs       // FSEvents via notify + debouncer; feeds same ingest queue as scan
  icloud.rs      // eviction detection (".<name>.icloud" stubs / dataless), brctl download + wait
  ingest.rs      // per-file pipeline dispatcher by type; states: discovered -> extracted -> enriched -> embedded
  extract/
    mod.rs       // Extractor trait: (path) -> ExtractedDoc { pages/segments of text, render hints }
    pdf.rs       // pdfium-render: embedded text layer first; rasterize pages for OCR/VLM when needed
    ocr.rs       // Apple Vision RecognizeTextRequest via objc2/objc2-vision (fallback: tiny Swift sidecar)
    image.rs     // heic/png/jpg -> Vision OCR + VLM caption
    text.rs      // txt/md/html (strip tags), csv (header + schema + capped row text)
    media.rs     // whisper-rs transcription; ffmpeg demux for mp4/m4a
  enrich.rs      // local-LLM pass: summary, doc_type, facts (typed key-values), tags; VLM page pass for PDFs/images
  chunk.rs       // text chunking (target tokens + overlap), SHA-256 content hashes (hash-keyed like sister)
  embed.rs       // Embedder trait: fastembed (default, bge-small-en-v1.5), openai-compatible, debug-hash
  index.rs       // IndexDb: schema, migrations (PRAGMA user_version, additive), queries, Writer
  retrieve.rs    // hybrid FTS5 + brute-force cosine fused with RRF (RRF_K=60), same as sister
  llm.rs         // OpenAI-compatible chat client (ureq): /v1/chat/completions, vision via base64 image parts
  research.rs    // deepResearch loop: local LLM iteratively calls internal search/get tools, synthesizes answer
  mcp.rs         // pure JSON-RPC handler; stdio + tiny_http POST /mcp with Bearer token (constant-time compare)
  service.rs     // launchd plists: com.ai-icloud.watch (KeepAlive), com.ai-icloud.serve (opt-in HTTP)
  doctor.rs      // checks: icloud dir readable (TCC), LLM endpoint reachable + model loaded, ffmpeg, disk, eviction
  dryrun.rs      // scan --dry-run report: files by type, would-index/skip/excluded counts
```

### SQLite schema (index.sqlite, WAL, 0600, bundled rusqlite)

- `meta(key PK, value)` — schema bookkeeping, embedding model identity, last full-scan time.
- `files(id PK, rel_path UNIQUE, abs_path, kind, size, mtime_ms, content_sha256, status, error, indexed_at_ms)`
  — identity is `rel_path`; change detection by (size, mtime) fast-path then sha256; renames reconciled by hash.
- `documents(id PK, file_id -> files, title, doc_type, summary, lang, page_count, produced_by_model, enriched_at_ms)`
- `facts(id PK, document_id -> documents, key, value, value_norm, kind /* date|amount|party|address|other */, page, confidence)`
  — e.g. (`sale_price`, `$487,500`, `487500`, `amount`). Indexed on (key), (kind), (document_id).
- `tags(document_id, tag)` — indexed both ways.
- `chunks(id PK, document_id, seq, page_start, page_end, text, content_hash)` + `chunks_fts` (FTS5, manual sync)
- `embeddings(chunk_hash PK, model, vector BLOB /* raw f32 LE */)` — hash-keyed so embeddings survive re-chunking.
- Summary text is also chunked (seq = -1 convention or a `is_summary` flag) so doc-level meaning is searchable.

### Ingest pipeline

1. **Discover** — FSEvents event (debounced ~2s, ignores `.tmp`, dotfiles, partial-sync artifacts) or full scan.
   Exclusion globs from config are applied here; excluded paths never touch the DB.
2. **Materialize** — if the path is an eviction stub, `brctl download`, poll for materialization with timeout;
   on timeout mark `status=evicted-pending` and retry on next scan.
3. **Extract** — per-type extractor produces page/segment text. PDFs: embedded text layer via pdfium first;
   pages whose text layer is empty/garbage (heuristic: char count, replacement-char ratio) are rasterized and
   sent to Apple Vision OCR.
4. **Enrich** — local LLM: for PDFs/images a VLM pass per page image (tables, layout, handwriting) merged with
   OCR text; then one doc-level structured call returning `{title, doc_type, summary, facts[], tags[]}` (JSON
   schema-constrained). Stored with the producing model name for later re-enrichment.
5. **Chunk + embed** — token-target chunks with overlap, hash-keyed; missing-embedding batches committed per
   batch (resumable), model-change wipes vectors (sister's `ensure_embedding_model` pattern).

Each stage writes `files.status`, so a crash resumes where it left off; `scan` reconciles drift (deletes prune
documents/chunks/facts via FK cascade + orphan-embedding pruning).

### MCP tools

| Tool | Purpose |
|---|---|
| `search_documents` | hybrid/keyword/semantic over chunks + summaries; returns doc metadata + snippet + chunk ids |
| `get_document` | full extracted text + summary + facts + tags for a document id (paged) |
| `get_chunk` | chunk with surrounding context |
| `list_documents` | filter by folder, doc_type, tag, date range |
| `search_facts` | query the facts table (key/kind/value ranges) — e.g. all `amount` facts in doc_type `closing_statement` |
| `ask` | deepResearch: local LLM iteratively searches/reads server-side, returns synthesized answer + cited doc ids |
| `sync_status` | last scan/watch heartbeat, counts by status, pending/evicted/errored files |
| `reindex_file` | force re-ingest of one path (also exposed as CLI) |

Transports identical to sister: stdio default; `--http` via tiny_http, `POST /mcp`, Bearer token, loopback-guarded.

### Config (`~/Library/Application Support/ai-icloud/config.toml`, all defaults, deny_unknown_fields)

```toml
[source]
root = "~/Library/Mobile Documents/com~apple~CloudDocs"
include_extensions = ["pdf","txt","md","html","csv","png","jpg","jpeg","heic","mp3","mp4","m4a"]
exclude_globs = []            # e.g. ["Divorce/**", "**/*.stl"] — never indexed, never stored
max_file_mb = 200

[llm]                          # OpenAI-compatible; default LM Studio
base_url = "http://127.0.0.1:1234/v1"
api_key = ""                   # LM Studio API token
model = ""                     # chat/enrichment model (empty = server default)
vision_model = ""              # VLM for page/image passes (empty = model)

[embeddings]
provider = "embedded"          # fastembed bge-small-en-v1.5 | "openai-compatible" | "debug-hash"
model = "bge-small-en-v1.5"
batch_size = 32

[transcription]
enabled = true
whisper_model = "large-v3-turbo"   # auto-downloaded to <index_dir>/models/

[watch]
debounce_ms = 2000
full_scan_interval_minutes = 360   # reconciliation sweep

[index]
chunk_target_tokens = 750
chunk_overlap_tokens = 80

[retrieval]
fts_candidates = 30
vector_candidates = 30
result_limit = 10

[research]                     # `ask` tool budget
max_iterations = 6
max_context_chunks = 24

[service]
http_token = ""                # else generated 0600 like sister

[privacy]
allow_remote_endpoints = false # loopback-only for BOTH llm and embeddings unless flipped
```

## Known hardships & mitigations

1. **iCloud eviction / sync churn** — dataless stubs need `brctl download`; FSEvents fires storms during sync.
   Mitigation: debounce, stub detection, `evicted-pending` retry state, periodic reconciliation scan.
2. **TCC / Full Disk Access** — launchd runs the binary directly; FDA must be granted to the *binary*, and
   unsigned rebuilds invalidate the grant. Mitigation: sister's playbook — doctor check with actionable EPERM
   classification, codesigned releases (`scripts/release.sh` pattern).
3. **Apple Vision from Rust** — no first-party Rust API. Plan A: `objc2` + `objc2-vision`
   (`VNRecognizeTextRequest`). Plan B (fallback if bindings fight us): tiny Swift sidecar binary built by
   `build.rs`, spoken to over stdin/stdout. Decide in a spike before committing.
4. **PDF rasterization** — `pdfium-render` with a bundled/vendored pdfium dylib; doctor verifies load.
5. **LM Studio operational coupling** — needs the server running, a model loaded, and a valid token; JIT model
   loading can time out on first call. Mitigation: doctor checks `/v1/models`, generous first-call timeout,
   crisp error messages; endpoint fully swappable.
6. **Structured extraction reliability** — local models emit imperfect JSON. Mitigation: JSON-schema-constrained
   requests where supported, strict parse + one retry, then store summary-only and mark `facts_incomplete`.
7. **HEIC decode** — convert via `sips` (built into macOS) before OCR/VLM rather than pulling a heavy codec dep.
8. **ffmpeg dependency** for mp4/m4a demux — doctor-checked, clear install hint (`brew install ffmpeg`);
   mp3 can decode via symphonia without ffmpeg.
9. **Anything indexed is agent-readable** — exclusion config is the privacy boundary; `scan --dry-run` prints
   exactly what would be indexed before first run.

## Patterns inherited from ai-imessage (do not reinvent)

- Pure `handle(&Value) -> Option<Value>` JSON-RPC MCP handler; stdio + stateless HTTP; constant-time Bearer check.
- Watermark-free but hash-based incremental identity; hash-keyed chunk/embedding survival across re-chunks.
- Additive `PRAGMA user_version` migrations; refuse-if-newer.
- FTS5 + raw-f32-LE brute-force cosine + RRF (K=60); sanitize FTS queries by quoting tokens.
- Embedder provider trait incl. deterministic `debug-hash` for tests; loopback privacy gating.
- launchd plist rendering with `AI_ICLOUD_NO_LAUNCHCTL` test escape hatch; logs + tokens under 0700 index dir.
- Full-defaults TOML + `deny_unknown_fields` + secret redaction in `config show`.
- Synthetic-fixture integration tests (`assert_cmd` + `tempfile`): fake iCloud tree fixtures, no real user data;
  `debug-hash` embedder + mock LLM server (tiny_http) so tests need no models or network.

## Build phases

1. **Skeleton** — cargo init, cli/config/paths/doctor, index schema, scan (txt/md/csv/html only), chunk, embed
   (fastembed + debug-hash), FTS+vector+RRF search, `search` CLI. *End-to-end searchable on plain text.*
2. **PDF + OCR** — pdfium text layer, rasterization, Apple Vision spike (objc2 vs Swift sidecar), hybrid page
   pipeline. *All 64 PDFs searchable.*
3. **LLM enrichment** — llm.rs client (LM Studio default), VLM page pass, doc-level summary/facts/tags,
   `facts` table + `search_facts`. *"House profit" question answerable via retrieval + facts.*
4. **MCP server** — mcp.rs (stdio + HTTP), all tools except `ask`; register with Claude Code and validate the
   house-profit flow from an agent.
5. **Watcher + service** — watch.rs, icloud.rs eviction handling, service.rs launchd install, sync_status.
6. **Media + ask** — whisper-rs transcription, images/HEIC, `ask` deepResearch loop.
7. **Hardening** — doctor completeness, release script with codesigning, README.
