# ai-icloud

Local-first RAG index and MCP server for your iCloud Drive documents.

A background daemon watches `~/Library/Mobile Documents/com~apple~CloudDocs`,
extracts text from everything it finds — PDF text layers, Apple Vision OCR for
scans and images, whisper.cpp transcription for audio/video — enriches each
document with a local LLM (summary, type, key-value facts, tags), embeds
everything into a SQLite index, and serves it to any MCP-capable agent. Ask
your agent *"when I sold my house, what was my profit?"* and it finds the
closing statement and reads the numbers out of the facts table.

Everything runs on your machine. Document content only ever leaves the index
toward loopback endpoints unless you explicitly flip
`[privacy] allow_remote_endpoints`.

Sister project: [ai-imessage](https://github.com/tjameswilliams/ai-imessage) —
the same architecture over Apple Messages.

## Requirements

- macOS (Apple Vision OCR, PDFKit, FSEvents, launchd)
- Rust + Xcode Command Line Tools (`swiftc` builds the OCR sidecar at compile
  time; end users of a prebuilt binary don't need it)
- An OpenAI-compatible LLM server on loopback for enrichment and the `ask`
  tool — LM Studio by default (`http://127.0.0.1:1234/v1`); Ollama or anything
  else works via config
- `ffmpeg` for audio/video transcription (`brew install ffmpeg`), optional

## Quick start

```bash
cargo build --release

# What would be indexed?
./target/release/ai-icloud scan --dry-run

# Check the environment (iCloud access, index, OCR sidecar, LLM endpoint)
./target/release/ai-icloud doctor

# Index + enrich + embed
./target/release/ai-icloud scan

# Search from the CLI
./target/release/ai-icloud search closing statement sale price

# Register with Claude Code
claude mcp add --scope user ai-icloud -- $PWD/target/release/ai-icloud serve

# Install the launchd watch daemon (indexes changes continuously)
./target/release/ai-icloud service install
```

First scan downloads the embedding model (~130 MB); the first media file
downloads the whisper model (~1.6 GB). Both land under the index directory.

## Configuration

`~/Library/Application Support/ai-icloud/config.toml` — every key has a
default; the file is optional. The interesting ones:

```toml
[source]
root = "~/Library/Mobile Documents/com~apple~CloudDocs"
exclude_globs = ["Private/**"]   # never indexed, never stored

[llm]                            # OpenAI-compatible; LM Studio default
base_url = "http://127.0.0.1:1234/v1"
api_key = "sk-..."               # LM Studio: Developer → API token
model = "qwen/qwen3.6-27b"       # text enrichment + ask
vision_model = "qwen3.5-vl-9b-mlx-crack"  # page/image passes

[embeddings]
provider = "embedded"            # fastembed bge-small-en-v1.5, on-device

[transcription]
enabled = true
whisper_model = "large-v3-turbo"

[privacy]
allow_remote_endpoints = false   # loopback-only for llm + embeddings
```

`ai-icloud config show` prints the effective config with secrets redacted.

## Commands

| Command | Purpose |
|---|---|
| `scan [--dry-run] [--no-embed] [--no-enrich]` | one-shot index pass |
| `enrich [--force] [--limit N]` | run/redo LLM enrichment |
| `search <query> [--keyword\|--semantic]` | hybrid FTS5+vector search (RRF) |
| `watch` | foreground daemon (FSEvents + periodic reconciliation) |
| `service install\|start\|stop\|uninstall\|status` | launchd management |
| `serve [--http ADDR]` | MCP over stdio, or HTTP with a bearer token |
| `connect` | print (or mint) the HTTP bearer token |
| `doctor` | actionable environment checks |

## MCP tools

`search_documents`, `get_document` (paged full text + facts + tags),
`get_chunk` (neighbor context), `list_documents` (filter by folder / type /
tag), `search_facts` (structured dates, amounts, parties, addresses),
`sync_status`, `reindex_file`, and `ask` — a server-side research loop where
the local model searches, reads, and answers with cited file paths.

## How indexing works

1. **Scan** walks the tree (exclusion globs, extension scoping, size caps),
   detects iCloud eviction stubs (`.name.icloud`) and asks `brctl` to
   materialize them, and diffs size/mtime/sha256 against the index.
2. **Extract** per kind: pdfium-free PDF handling via a Swift sidecar
   (PDFKit text layer per page; pages without a usable layer are rendered at
   300 dpi and OCR'd by Apple Vision), image OCR (incl. HEIC), HTML tag
   stripping, whisper transcription with timestamps.
3. **Enrich**: one JSON-schema-constrained LLM call per document (with
   rendered page images for the vision model) produces title, doc_type,
   summary, facts, tags. The summary is embedded and searchable.
4. **Chunk + embed**: hash-keyed chunks (embeddings survive re-chunking and
   dedupe identical text), FTS5 mirror, raw-f32 vectors, brute-force cosine +
   BM25 fused with reciprocal rank fusion.

The watch daemon treats any FSEvent as a wake-up, debounces until the change
stream settles, and re-runs the (cheap) scan diff; a periodic tick reconciles
anything FSEvents dropped. Every stage is per-file/per-doc transactional, so
a crash or restart resumes cleanly.

## Full Disk Access

launchd runs the binary directly, so if the daemon logs permission errors on
`~/Library/Mobile Documents`, grant Full Disk Access to the **binary itself**
(System Settings → Privacy & Security → Full Disk Access). Rebuilding an
unsigned binary invalidates the grant — codesign release builds.

## License

MIT OR Apache-2.0
