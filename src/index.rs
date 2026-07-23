//! The destination index: schema, migrations, queries, and writes.
//!
//! One SQLite database (WAL, owner-only permissions) holding file sync
//! state, per-document metadata and LLM-extracted structure, hash-keyed
//! text chunks with an FTS5 mirror, and embedding vectors keyed by chunk
//! content hash so they survive re-chunking.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::{Connection, OpenFlags, params};

use crate::chunk::ChunkPiece;

pub const SCHEMA_VERSION: i32 = 1;

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  rel_path TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  size INTEGER NOT NULL,
  mtime_ms INTEGER NOT NULL,
  content_sha256 TEXT,
  status TEXT NOT NULL,
  error TEXT,
  indexed_at_ms INTEGER
);
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL UNIQUE REFERENCES files(id) ON DELETE CASCADE,
  title TEXT,
  doc_type TEXT,
  summary TEXT,
  lang TEXT,
  page_count INTEGER,
  produced_by_model TEXT,
  enriched_at_ms INTEGER
);
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  value_norm TEXT,
  kind TEXT NOT NULL,
  page INTEGER,
  confidence REAL
);
CREATE INDEX IF NOT EXISTS facts_document ON facts(document_id);
CREATE INDEX IF NOT EXISTS facts_key ON facts(key);
CREATE INDEX IF NOT EXISTS facts_kind ON facts(kind);
CREATE TABLE IF NOT EXISTS tags (
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  tag TEXT NOT NULL,
  PRIMARY KEY (document_id, tag)
);
CREATE INDEX IF NOT EXISTS tags_tag ON tags(tag);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  is_summary INTEGER NOT NULL DEFAULT 0,
  page_start INTEGER,
  page_end INTEGER,
  text TEXT NOT NULL,
  content_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id, seq);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(content_hash);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text);
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_hash TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  vector BLOB NOT NULL
);
";

const EMBEDDING_MODEL_KEY: &str = "embedding_model";
const LAST_SCAN_KEY: &str = "last_scan_ms";

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("could not create index directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("could not open index database {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: rusqlite::Error,
    },
    #[error(
        "index database {path} has schema version {found}, newer than this \
         binary supports ({supported}) — upgrade ai-icloud"
    )]
    NewerSchema {
        path: PathBuf,
        found: i32,
        supported: i32,
    },
    #[error("index database query failed: {0}")]
    Query(#[from] rusqlite::Error),
    #[error("could not restrict permissions on {path}: {source}")]
    Permissions {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// Lifecycle of a file in the index. Stored as text in `files.status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileStatus {
    /// In scope but its kind has no extractor yet (later phase).
    Pending,
    /// Extracted, chunked, and searchable.
    Indexed,
    /// Extraction failed; see `files.error`.
    Error,
    /// An iCloud eviction stub; content not on disk yet.
    Evicted,
}

impl FileStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            FileStatus::Pending => "pending",
            FileStatus::Indexed => "indexed",
            FileStatus::Error => "error",
            FileStatus::Evicted => "evicted",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(FileStatus::Pending),
            "indexed" => Some(FileStatus::Indexed),
            "error" => Some(FileStatus::Error),
            "evicted" => Some(FileStatus::Evicted),
            _ => None,
        }
    }
}

/// Sync-relevant state of one `files` row, keyed by rel_path elsewhere.
#[derive(Debug, Clone, PartialEq)]
pub struct FileState {
    pub id: i64,
    pub size: i64,
    pub mtime_ms: i64,
    pub content_sha256: Option<String>,
    pub status: String,
}

/// A document with its file metadata, as served by the MCP tools.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentInfo {
    pub document_id: i64,
    pub rel_path: String,
    pub kind: String,
    pub title: Option<String>,
    pub doc_type: Option<String>,
    pub summary: Option<String>,
    pub page_count: Option<i64>,
    pub size: i64,
    pub mtime_ms: i64,
    pub indexed_at_ms: Option<i64>,
    pub chunk_count: i64,
    pub tags: Vec<String>,
}

/// Enrichment output ready to store (see `apply_enrichment`).
#[derive(Debug, Clone, PartialEq)]
pub struct EnrichmentRecord {
    pub title: Option<String>,
    pub doc_type: String,
    pub summary: String,
    /// (key, value, kind, page)
    pub facts: Vec<(String, String, String, Option<i64>)>,
    pub tags: Vec<String>,
}

/// One stored chunk row (used for document text and context windows).
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRow {
    pub id: i64,
    pub document_id: i64,
    pub seq: i64,
    pub is_summary: bool,
    pub page_start: Option<i64>,
    pub page_end: Option<i64>,
    pub text: String,
}

/// One extracted fact (populated by the enrichment phase).
#[derive(Debug, Clone, PartialEq)]
pub struct FactRow {
    pub rel_path: String,
    pub document_id: i64,
    pub key: String,
    pub value: String,
    pub kind: String,
    pub page: Option<i64>,
}

/// One search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub chunk_id: i64,
    pub document_id: i64,
    /// Path relative to the indexed root — the human label for a hit.
    pub rel_path: String,
    pub title: Option<String>,
    pub doc_type: Option<String>,
    pub seq: i64,
    pub is_summary: bool,
    /// Keyword search: FTS5 snippet with matches wrapped in «guillemets».
    /// Vector search: the truncated start of the chunk.
    pub snippet: String,
    /// Cosine similarity, vector search only.
    pub score: Option<f32>,
}

fn truncate_snippet(text: &str) -> String {
    const MAX_CHARS: usize = 200;
    if text.chars().count() <= MAX_CHARS {
        return text.to_string();
    }
    let mut s: String = text.chars().take(MAX_CHARS).collect();
    s.push('…');
    s
}

/// Quote every whitespace-separated term so user input is always a valid
/// FTS5 query (terms AND together; operators lose their meaning).
fn sanitize_fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
}

/// A read-write connection to the index.
#[derive(Debug)]
pub struct IndexDb {
    conn: Connection,
    path: PathBuf,
}

impl IndexDb {
    /// Open (creating if needed) the index at `path`. The parent directory
    /// is created with owner-only permissions, as is the database file.
    pub fn open(path: &Path) -> Result<Self, IndexError> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            fs::create_dir_all(dir).map_err(|e| IndexError::CreateDir {
                path: dir.to_path_buf(),
                source: e,
            })?;
            restrict_permissions(dir, 0o700)?;
        }

        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|e| IndexError::Open {
            path: path.to_path_buf(),
            source: e,
        })?;
        restrict_permissions(path, 0o600)?;

        conn.busy_timeout(Duration::from_secs(10))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", true)?;

        let version: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version > SCHEMA_VERSION {
            return Err(IndexError::NewerSchema {
                path: path.to_path_buf(),
                found: version,
                supported: SCHEMA_VERSION,
            });
        }
        conn.execute_batch(SCHEMA_SQL)?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;

        Ok(IndexDb {
            conn,
            path: path.to_path_buf(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    // ------------------------------------------------------------- counts

    pub fn file_count(&self) -> Result<u64, IndexError> {
        self.count("files")
    }

    pub fn document_count(&self) -> Result<u64, IndexError> {
        self.count("documents")
    }

    pub fn chunk_count(&self) -> Result<u64, IndexError> {
        self.count("chunks")
    }

    pub fn embedding_count(&self) -> Result<u64, IndexError> {
        self.count("embeddings")
    }

    /// Files per status, for `sync_status` and doctor output.
    pub fn status_counts(&self) -> Result<Vec<(String, u64)>, IndexError> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT status, COUNT(*) FROM files GROUP BY status ORDER BY status")?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as u64)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    fn count(&self, table: &str) -> Result<u64, IndexError> {
        // Table names are compile-time constants, never user input.
        let n: i64 = self
            .conn
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))?;
        Ok(n.max(0) as u64)
    }

    // --------------------------------------------------------- sync state

    /// Everything the scanner needs to diff the tree against the index,
    /// keyed by rel_path.
    pub fn file_states(&self) -> Result<HashMap<String, FileState>, IndexError> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT rel_path, id, size, mtime_ms, content_sha256, status FROM files",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    FileState {
                        id: r.get(1)?,
                        size: r.get(2)?,
                        mtime_ms: r.get(3)?,
                        content_sha256: r.get(4)?,
                        status: r.get(5)?,
                    },
                ))
            })?
            .collect::<Result<HashMap<_, _>, _>>()?;
        Ok(rows)
    }

    pub fn set_last_scan_ms(&self, ms: i64) -> Result<(), IndexError> {
        meta_set(&self.conn, LAST_SCAN_KEY, &ms.to_string())
    }

    pub fn last_scan_ms(&self) -> Result<Option<i64>, IndexError> {
        Ok(meta_get(&self.conn, LAST_SCAN_KEY)?.and_then(|s| s.parse().ok()))
    }

    // ------------------------------------------------------------- writes

    /// Record a file that is in scope but not (yet) extracted: pending,
    /// evicted, or failed. Any previously indexed content for the path is
    /// removed — the file changed, so stale chunks must not serve queries.
    pub fn upsert_unextracted_file(
        &mut self,
        rel_path: &str,
        kind: &str,
        size: i64,
        mtime_ms: i64,
        status: FileStatus,
        error: Option<&str>,
    ) -> Result<(), IndexError> {
        debug_assert!(status != FileStatus::Indexed);
        let tx = self.conn.transaction()?;
        delete_document_content(&tx, rel_path)?;
        tx.execute(
            "INSERT INTO files (rel_path, kind, size, mtime_ms, content_sha256, status, error, indexed_at_ms)
             VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?6, NULL)
             ON CONFLICT(rel_path) DO UPDATE SET
               kind = excluded.kind, size = excluded.size, mtime_ms = excluded.mtime_ms,
               content_sha256 = NULL, status = excluded.status, error = excluded.error,
               indexed_at_ms = NULL",
            params![rel_path, kind, size, mtime_ms, status.as_str(), error],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Store one fully extracted file: the files row, its document row, and
    /// its chunks (with the FTS mirror) — atomically. Replaces any previous
    /// content for the path.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_indexed_file(
        &mut self,
        rel_path: &str,
        kind: &str,
        size: i64,
        mtime_ms: i64,
        content_sha256: &str,
        title: Option<&str>,
        page_count: Option<i64>,
        chunks: &[ChunkPiece],
        indexed_at_ms: i64,
    ) -> Result<(), IndexError> {
        let tx = self.conn.transaction()?;
        delete_document_content(&tx, rel_path)?;
        tx.execute(
            "INSERT INTO files (rel_path, kind, size, mtime_ms, content_sha256, status, error, indexed_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, 'indexed', NULL, ?6)
             ON CONFLICT(rel_path) DO UPDATE SET
               kind = excluded.kind, size = excluded.size, mtime_ms = excluded.mtime_ms,
               content_sha256 = excluded.content_sha256, status = 'indexed', error = NULL,
               indexed_at_ms = excluded.indexed_at_ms",
            params![rel_path, kind, size, mtime_ms, content_sha256, indexed_at_ms],
        )?;
        let file_id: i64 = tx.query_row(
            "SELECT id FROM files WHERE rel_path = ?1",
            params![rel_path],
            |r| r.get(0),
        )?;
        tx.execute(
            "INSERT INTO documents (file_id, title, page_count) VALUES (?1, ?2, ?3)",
            params![file_id, title, page_count],
        )?;
        let document_id = tx.last_insert_rowid();
        {
            let mut ins = tx.prepare_cached(
                "INSERT INTO chunks (document_id, seq, is_summary, page_start, page_end, text, content_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            let mut fts = tx.prepare_cached(
                "INSERT INTO chunks_fts (rowid, text) VALUES (?1, ?2)",
            )?;
            for piece in chunks {
                ins.execute(params![
                    document_id,
                    piece.seq,
                    piece.is_summary,
                    piece.page_start,
                    piece.page_end,
                    piece.text,
                    piece.content_hash,
                ])?;
                fts.execute(params![tx.last_insert_rowid(), piece.text])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Remove a file (deleted from disk or newly excluded) and all content
    /// derived from it. Returns true when a row existed.
    pub fn remove_file(&mut self, rel_path: &str) -> Result<bool, IndexError> {
        let tx = self.conn.transaction()?;
        delete_document_content(&tx, rel_path)?;
        let n = tx.execute("DELETE FROM files WHERE rel_path = ?1", params![rel_path])?;
        tx.commit()?;
        Ok(n > 0)
    }

    // --------------------------------------------------------- embeddings

    /// Chunks that have no stored embedding yet, as (content_hash, text).
    /// Hashes are distinct even if several chunk rows share content.
    pub fn missing_embeddings(&self) -> Result<Vec<(String, String)>, IndexError> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT content_hash, MIN(text) FROM chunks
             WHERE content_hash NOT IN (SELECT chunk_hash FROM embeddings)
             GROUP BY content_hash",
        )?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Store one batch of embeddings in its own transaction, so a long
    /// embedding run that is interrupted keeps everything finished so far.
    pub fn store_embeddings(
        &mut self,
        model: &str,
        items: &[(String, Vec<f32>)],
    ) -> Result<(), IndexError> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO embeddings (chunk_hash, model, vector) VALUES (?1, ?2, ?3)
                 ON CONFLICT(chunk_hash) DO UPDATE SET
                   model = excluded.model, vector = excluded.vector",
            )?;
            for (hash, vector) in items {
                stmt.execute(params![hash, model, crate::embed::vector_to_blob(vector)])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Record which embedding model the index was built with. A model
    /// change wipes all stored vectors (they are not comparable across
    /// models); returns true when that happened.
    pub fn ensure_embedding_model(&mut self, model: &str) -> Result<bool, IndexError> {
        let stored = meta_get(&self.conn, EMBEDDING_MODEL_KEY)?;
        match stored.as_deref() {
            Some(s) if s == model => Ok(false),
            Some(_) => {
                let tx = self.conn.transaction()?;
                tx.execute("DELETE FROM embeddings", [])?;
                meta_set(&tx, EMBEDDING_MODEL_KEY, model)?;
                tx.commit()?;
                Ok(true)
            }
            None => {
                meta_set(&self.conn, EMBEDDING_MODEL_KEY, model)?;
                Ok(false)
            }
        }
    }

    /// Drop embeddings whose chunk no longer exists (file removed or
    /// re-extracted). Returns how many were removed.
    pub fn prune_orphan_embeddings(&mut self) -> Result<u64, IndexError> {
        let n = self.conn.execute(
            "DELETE FROM embeddings
             WHERE chunk_hash NOT IN (SELECT content_hash FROM chunks)",
            [],
        )?;
        Ok(n as u64)
    }

    // ------------------------------------------------------------- search

    /// FTS5 keyword search over chunks, best match first.
    pub fn search(&self, query: &str, limit: u32) -> Result<Vec<SearchHit>, IndexError> {
        let fts_query = sanitize_fts_query(query);
        if fts_query.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare_cached(
            "SELECT
               c.id, c.document_id, f.rel_path, d.title, d.doc_type, c.seq, c.is_summary,
               snippet(chunks_fts, 0, '«', '»', ' … ', 24)
             FROM chunks_fts
             JOIN chunks c ON c.id = chunks_fts.rowid
             JOIN documents d ON d.id = c.document_id
             JOIN files f ON f.id = d.file_id
             WHERE chunks_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;
        let hits = stmt
            .query_map(params![fts_query, limit], |r| {
                Ok(SearchHit {
                    chunk_id: r.get(0)?,
                    document_id: r.get(1)?,
                    rel_path: r.get(2)?,
                    title: r.get(3)?,
                    doc_type: r.get(4)?,
                    seq: r.get(5)?,
                    is_summary: r.get::<_, i64>(6)? != 0,
                    snippet: r.get(7)?,
                    score: None,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(hits)
    }

    /// Brute-force cosine similarity over every embedded chunk, best first.
    /// At this corpus size (thousands of chunks) this is well under a
    /// millisecond; no vector index needed.
    pub fn vector_search(&self, query: &[f32], limit: u32) -> Result<Vec<SearchHit>, IndexError> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT
               c.id, c.document_id, f.rel_path, d.title, d.doc_type, c.seq, c.is_summary,
               c.text, e.vector
             FROM chunks c
             JOIN embeddings e ON e.chunk_hash = c.content_hash
             JOIN documents d ON d.id = c.document_id
             JOIN files f ON f.id = d.file_id",
        )?;
        let mut hits: Vec<(f32, SearchHit)> = stmt
            .query_map([], |r| {
                let text: String = r.get(7)?;
                let blob: Vec<u8> = r.get(8)?;
                Ok((
                    crate::embed::cosine(query, &crate::embed::blob_to_vector(&blob)),
                    SearchHit {
                        chunk_id: r.get(0)?,
                        document_id: r.get(1)?,
                        rel_path: r.get(2)?,
                        title: r.get(3)?,
                        doc_type: r.get(4)?,
                        seq: r.get(5)?,
                        is_summary: r.get::<_, i64>(6)? != 0,
                        snippet: truncate_snippet(&text),
                        score: None,
                    },
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Orthogonal or opposed vectors share no signal with the query;
        // "nearest" among those is noise, not a result.
        hits.retain(|(score, _)| *score > 0.0);
        hits.sort_by(|a, b| b.0.total_cmp(&a.0));
        hits.truncate(limit as usize);
        Ok(hits
            .into_iter()
            .map(|(score, mut h)| {
                h.score = Some(score);
                h
            })
            .collect())
    }

    /// Full text of one chunk, if it exists.
    pub fn chunk_text(&self, chunk_id: i64) -> Result<Option<String>, IndexError> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT text FROM chunks WHERE id = ?1")?;
        let mut rows = stmt.query(params![chunk_id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    // ---------------------------------------------------------- documents

    const DOCUMENT_SELECT: &'static str = "
        SELECT d.id, f.rel_path, f.kind, d.title, d.doc_type, d.summary, d.page_count,
               f.size, f.mtime_ms, f.indexed_at_ms,
               (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id)
        FROM documents d JOIN files f ON f.id = d.file_id";

    fn document_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DocumentInfo> {
        Ok(DocumentInfo {
            document_id: row.get(0)?,
            rel_path: row.get(1)?,
            kind: row.get(2)?,
            title: row.get(3)?,
            doc_type: row.get(4)?,
            summary: row.get(5)?,
            page_count: row.get(6)?,
            size: row.get(7)?,
            mtime_ms: row.get(8)?,
            indexed_at_ms: row.get(9)?,
            chunk_count: row.get(10)?,
            tags: Vec::new(),
        })
    }

    fn with_tags(&self, mut doc: DocumentInfo) -> Result<DocumentInfo, IndexError> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT tag FROM tags WHERE document_id = ?1 ORDER BY tag")?;
        doc.tags = stmt
            .query_map(params![doc.document_id], |r| r.get(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(doc)
    }

    pub fn document_by_id(&self, document_id: i64) -> Result<Option<DocumentInfo>, IndexError> {
        let sql = format!("{} WHERE d.id = ?1", Self::DOCUMENT_SELECT);
        let mut stmt = self.conn.prepare_cached(&sql)?;
        let doc = stmt
            .query_map(params![document_id], Self::document_from_row)?
            .next()
            .transpose()?;
        doc.map(|d| self.with_tags(d)).transpose()
    }

    pub fn document_by_path(&self, rel_path: &str) -> Result<Option<DocumentInfo>, IndexError> {
        let sql = format!("{} WHERE f.rel_path = ?1", Self::DOCUMENT_SELECT);
        let mut stmt = self.conn.prepare_cached(&sql)?;
        let doc = stmt
            .query_map(params![rel_path], Self::document_from_row)?
            .next()
            .transpose()?;
        doc.map(|d| self.with_tags(d)).transpose()
    }

    /// Documents matching optional filters, most recently indexed first.
    pub fn list_documents(
        &self,
        path_filter: Option<&str>,
        doc_type: Option<&str>,
        tag: Option<&str>,
        limit: u32,
    ) -> Result<Vec<DocumentInfo>, IndexError> {
        let mut sql = format!("{} WHERE 1=1", Self::DOCUMENT_SELECT);
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(f) = path_filter {
            sql.push_str(
                " AND (f.rel_path LIKE '%' || ? || '%' OR d.title LIKE '%' || ? || '%')",
            );
            args.push(Box::new(f.to_string()));
            args.push(Box::new(f.to_string()));
        }
        if let Some(t) = doc_type {
            sql.push_str(" AND d.doc_type = ?");
            args.push(Box::new(t.to_string()));
        }
        if let Some(t) = tag {
            sql.push_str(" AND d.id IN (SELECT document_id FROM tags WHERE tag = ?)");
            args.push(Box::new(t.to_string()));
        }
        sql.push_str(" ORDER BY f.indexed_at_ms DESC, f.rel_path LIMIT ?");
        args.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let docs = stmt
            .query_map(
                rusqlite::params_from_iter(args.iter().map(|a| a.as_ref())),
                Self::document_from_row,
            )?
            .collect::<Result<Vec<_>, _>>()?;
        docs.into_iter().map(|d| self.with_tags(d)).collect()
    }

    /// All chunks of a document in reading order (summary chunk first if
    /// one exists).
    pub fn document_chunks(&self, document_id: i64) -> Result<Vec<ChunkRow>, IndexError> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT id, document_id, seq, is_summary, page_start, page_end, text
             FROM chunks WHERE document_id = ?1
             ORDER BY is_summary DESC, seq",
        )?;
        let rows = stmt
            .query_map(params![document_id], chunk_row)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// One chunk plus up to `before`/`after` neighbors from the same
    /// document, in order. None when the chunk does not exist.
    pub fn chunk_window(
        &self,
        chunk_id: i64,
        before: u32,
        after: u32,
    ) -> Result<Option<Vec<ChunkRow>>, IndexError> {
        let target: Option<(i64, i64)> = {
            let mut stmt = self
                .conn
                .prepare_cached("SELECT document_id, seq FROM chunks WHERE id = ?1")?;
            let mut rows = stmt.query(params![chunk_id])?;
            match rows.next()? {
                Some(row) => Some((row.get(0)?, row.get(1)?)),
                None => None,
            }
        };
        let Some((document_id, seq)) = target else {
            return Ok(None);
        };
        let mut stmt = self.conn.prepare_cached(
            "SELECT id, document_id, seq, is_summary, page_start, page_end, text
             FROM chunks
             WHERE document_id = ?1 AND is_summary = 0 AND seq BETWEEN ?2 AND ?3
             ORDER BY seq",
        )?;
        let rows = stmt
            .query_map(
                params![document_id, seq - before as i64, seq + after as i64],
                chunk_row,
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(rows))
    }

    /// Search extracted facts (enrichment phase populates these).
    pub fn search_facts(
        &self,
        key: Option<&str>,
        kind: Option<&str>,
        value_contains: Option<&str>,
        limit: u32,
    ) -> Result<Vec<FactRow>, IndexError> {
        let mut sql = String::from(
            "SELECT f.rel_path, fa.document_id, fa.key, fa.value, fa.kind, fa.page
             FROM facts fa
             JOIN documents d ON d.id = fa.document_id
             JOIN files f ON f.id = d.file_id
             WHERE 1=1",
        );
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(k) = key {
            sql.push_str(" AND fa.key LIKE '%' || ? || '%'");
            args.push(Box::new(k.to_string()));
        }
        if let Some(k) = kind {
            sql.push_str(" AND fa.kind = ?");
            args.push(Box::new(k.to_string()));
        }
        if let Some(v) = value_contains {
            sql.push_str(" AND fa.value LIKE '%' || ? || '%'");
            args.push(Box::new(v.to_string()));
        }
        sql.push_str(" ORDER BY f.rel_path, fa.key LIMIT ?");
        args.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt
            .query_map(
                rusqlite::params_from_iter(args.iter().map(|a| a.as_ref())),
                |r| {
                    Ok(FactRow {
                        rel_path: r.get(0)?,
                        document_id: r.get(1)?,
                        key: r.get(2)?,
                        value: r.get(3)?,
                        kind: r.get(4)?,
                        page: r.get(5)?,
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Documents that still need (or, with `force`, should redo) an
    /// enrichment pass. Only successfully indexed files qualify.
    pub fn documents_needing_enrichment(
        &self,
        force: bool,
        limit: u32,
    ) -> Result<Vec<DocumentInfo>, IndexError> {
        let mut sql = format!("{} WHERE f.status = 'indexed'", Self::DOCUMENT_SELECT);
        if !force {
            sql.push_str(" AND d.enriched_at_ms IS NULL");
        }
        sql.push_str(" ORDER BY f.rel_path LIMIT ?");
        let mut stmt = self.conn.prepare(&sql)?;
        let docs = stmt
            .query_map(params![limit], Self::document_from_row)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(docs)
    }

    /// Store one document's enrichment atomically: metadata update, facts,
    /// tags, and a searchable summary chunk (replacing prior enrichment).
    pub fn apply_enrichment(
        &mut self,
        document_id: i64,
        record: &EnrichmentRecord,
        model: &str,
        now_ms: i64,
    ) -> Result<(), IndexError> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM chunks_fts WHERE rowid IN
               (SELECT id FROM chunks WHERE document_id = ?1 AND is_summary = 1)",
            params![document_id],
        )?;
        tx.execute(
            "DELETE FROM chunks WHERE document_id = ?1 AND is_summary = 1",
            params![document_id],
        )?;
        tx.execute("DELETE FROM facts WHERE document_id = ?1", params![document_id])?;
        tx.execute("DELETE FROM tags WHERE document_id = ?1", params![document_id])?;
        tx.execute(
            "UPDATE documents SET
               title = COALESCE(?2, title), doc_type = ?3, summary = ?4,
               produced_by_model = ?5, enriched_at_ms = ?6
             WHERE id = ?1",
            params![
                document_id,
                record.title,
                record.doc_type,
                record.summary,
                model,
                now_ms
            ],
        )?;
        {
            let mut fact_stmt = tx.prepare_cached(
                "INSERT INTO facts (document_id, key, value, value_norm, kind, page)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for (key, value, kind, page) in &record.facts {
                fact_stmt.execute(params![
                    document_id,
                    key,
                    value,
                    normalize_value(kind, value),
                    kind,
                    page
                ])?;
            }
            let mut tag_stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO tags (document_id, tag) VALUES (?1, ?2)",
            )?;
            for tag in &record.tags {
                let tag = tag.trim().to_lowercase();
                if !tag.is_empty() {
                    tag_stmt.execute(params![document_id, tag])?;
                }
            }
        }
        let summary = record.summary.trim();
        if !summary.is_empty() {
            tx.execute(
                "INSERT INTO chunks (document_id, seq, is_summary, page_start, page_end, text, content_hash)
                 VALUES (?1, -1, 1, NULL, NULL, ?2, ?3)",
                params![document_id, summary, crate::chunk::content_hash(summary)],
            )?;
            tx.execute(
                "INSERT INTO chunks_fts (rowid, text) VALUES (?1, ?2)",
                params![tx.last_insert_rowid(), summary],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Facts of one document, in insertion order.
    pub fn facts_for_document(&self, document_id: i64) -> Result<Vec<FactRow>, IndexError> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT f.rel_path, fa.document_id, fa.key, fa.value, fa.kind, fa.page
             FROM facts fa
             JOIN documents d ON d.id = fa.document_id
             JOIN files f ON f.id = d.file_id
             WHERE fa.document_id = ?1 ORDER BY fa.id",
        )?;
        let rows = stmt
            .query_map(params![document_id], |r| {
                Ok(FactRow {
                    rel_path: r.get(0)?,
                    document_id: r.get(1)?,
                    key: r.get(2)?,
                    value: r.get(3)?,
                    kind: r.get(4)?,
                    page: r.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Files currently in error state, with their messages.
    pub fn error_files(&self, limit: u32) -> Result<Vec<(String, String)>, IndexError> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT rel_path, COALESCE(error, '') FROM files
             WHERE status = 'error' ORDER BY rel_path LIMIT ?1",
        )?;
        let rows = stmt
            .query_map(params![limit], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

/// Best-effort machine-comparable form of a fact value: amounts lose
/// currency dressing, dates become ISO. None when no normalization
/// applies.
fn normalize_value(kind: &str, value: &str) -> Option<String> {
    match kind {
        "amount" => {
            let cleaned: String = value
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect();
            cleaned.parse::<f64>().ok().map(|n| {
                if n.fract() == 0.0 {
                    format!("{}", n as i64)
                } else {
                    format!("{n}")
                }
            })
        }
        "date" => {
            let v = value.trim();
            for fmt in ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"] {
                if let Ok(d) = chrono::NaiveDate::parse_from_str(v, fmt) {
                    return Some(d.format("%Y-%m-%d").to_string());
                }
            }
            None
        }
        _ => None,
    }
}

fn chunk_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChunkRow> {
    Ok(ChunkRow {
        id: row.get(0)?,
        document_id: row.get(1)?,
        seq: row.get(2)?,
        is_summary: row.get::<_, i64>(3)? != 0,
        page_start: row.get(4)?,
        page_end: row.get(5)?,
        text: row.get(6)?,
    })
}

/// Remove the document, chunks (incl. FTS mirror), facts, and tags derived
/// from `rel_path`, leaving the files row (if any) alone. FTS rows must go
/// first and explicitly: FK cascades do not touch virtual tables.
fn delete_document_content(
    tx: &rusqlite::Transaction<'_>,
    rel_path: &str,
) -> Result<(), IndexError> {
    tx.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (
           SELECT c.id FROM chunks c
           JOIN documents d ON d.id = c.document_id
           JOIN files f ON f.id = d.file_id
           WHERE f.rel_path = ?1
         )",
        params![rel_path],
    )?;
    tx.execute(
        "DELETE FROM documents WHERE file_id IN (SELECT id FROM files WHERE rel_path = ?1)",
        params![rel_path],
    )?;
    Ok(())
}

fn meta_get(conn: &Connection, key: &str) -> Result<Option<String>, IndexError> {
    let mut stmt = conn.prepare_cached("SELECT value FROM meta WHERE key = ?1")?;
    let mut rows = stmt.query(params![key])?;
    match rows.next()? {
        Some(row) => Ok(Some(row.get(0)?)),
        None => Ok(None),
    }
}

fn meta_set(conn: &Connection, key: &str, value: &str) -> Result<(), IndexError> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

#[cfg(unix)]
fn restrict_permissions(path: &Path, mode: u32) -> Result<(), IndexError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).map_err(|e| {
        IndexError::Permissions {
            path: path.to_path_buf(),
            source: e,
        }
    })
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path, _mode: u32) -> Result<(), IndexError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkPiece;
    use tempfile::TempDir;

    fn open_temp() -> (TempDir, IndexDb) {
        let dir = TempDir::new().unwrap();
        let db = IndexDb::open(&dir.path().join("nested").join("index.sqlite")).unwrap();
        (dir, db)
    }

    fn piece(seq: i64, text: &str) -> ChunkPiece {
        ChunkPiece {
            seq,
            is_summary: false,
            page_start: None,
            page_end: None,
            text: text.to_string(),
            content_hash: crate::chunk::content_hash(text),
        }
    }

    fn index_file(db: &mut IndexDb, rel_path: &str, chunks: &[ChunkPiece]) {
        db.upsert_indexed_file(rel_path, "text", 10, 1000, "sha", None, None, chunks, 2000)
            .unwrap();
    }

    #[test]
    fn open_creates_schema_and_parent_directory() {
        let (_dir, db) = open_temp();
        assert_eq!(db.file_count().unwrap(), 0);
        assert_eq!(db.chunk_count().unwrap(), 0);
    }

    #[test]
    fn newer_schema_is_refused() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index.sqlite");
        {
            let conn = Connection::open(&path).unwrap();
            conn.pragma_update(None, "user_version", SCHEMA_VERSION + 1)
                .unwrap();
        }
        match IndexDb::open(&path) {
            Err(IndexError::NewerSchema { found, .. }) => {
                assert_eq!(found, SCHEMA_VERSION + 1);
            }
            other => panic!("expected NewerSchema, got {other:?}"),
        }
    }

    #[test]
    fn indexed_file_is_searchable() {
        let (_dir, mut db) = open_temp();
        index_file(
            &mut db,
            "House/closing.txt",
            &[piece(0, "final sale price of the house was 487500 dollars")],
        );
        let hits = db.search("sale price", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rel_path, "House/closing.txt");
        assert!(hits[0].snippet.contains("«sale»"));
    }

    #[test]
    fn reindexing_replaces_old_chunks_and_fts_rows() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "a.txt", &[piece(0, "alpha original content")]);
        index_file(&mut db, "a.txt", &[piece(0, "beta replacement content")]);
        assert_eq!(db.file_count().unwrap(), 1);
        assert_eq!(db.document_count().unwrap(), 1);
        assert_eq!(db.chunk_count().unwrap(), 1);
        assert!(db.search("alpha", 10).unwrap().is_empty());
        assert_eq!(db.search("beta", 10).unwrap().len(), 1);
    }

    #[test]
    fn remove_file_drops_all_derived_content() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "a.txt", &[piece(0, "hello world")]);
        assert!(db.remove_file("a.txt").unwrap());
        assert!(!db.remove_file("a.txt").unwrap());
        assert_eq!(db.file_count().unwrap(), 0);
        assert_eq!(db.chunk_count().unwrap(), 0);
        assert!(db.search("hello", 10).unwrap().is_empty());
    }

    #[test]
    fn unextracted_upsert_clears_stale_content() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "a.pdf", &[piece(0, "old extracted text")]);
        db.upsert_unextracted_file("a.pdf", "pdf", 20, 3000, FileStatus::Pending, None)
            .unwrap();
        assert_eq!(db.file_count().unwrap(), 1);
        assert_eq!(db.chunk_count().unwrap(), 0);
        assert!(db.search("extracted", 10).unwrap().is_empty());
        let states = db.file_states().unwrap();
        assert_eq!(states["a.pdf"].status, "pending");
        assert_eq!(states["a.pdf"].content_sha256, None);
    }

    #[test]
    fn embeddings_roundtrip_and_vector_search() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "a.txt", &[piece(0, "pizza dinner tonight")]);
        index_file(&mut db, "b.txt", &[piece(0, "quarterly budget review")]);
        let missing = db.missing_embeddings().unwrap();
        assert_eq!(missing.len(), 2);

        let items: Vec<(String, Vec<f32>)> = missing
            .iter()
            .map(|(hash, text)| {
                let v = if text.contains("pizza") {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                };
                (hash.clone(), v)
            })
            .collect();
        db.store_embeddings("test-model", &items).unwrap();
        assert!(db.missing_embeddings().unwrap().is_empty());

        let hits = db.vector_search(&[1.0, 0.1], 10).unwrap();
        assert_eq!(hits[0].rel_path, "a.txt");
        assert!(hits[0].score.unwrap() > hits.last().unwrap().score.unwrap() || hits.len() == 1);
    }

    #[test]
    fn model_change_wipes_embeddings() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "a.txt", &[piece(0, "some text")]);
        let missing = db.missing_embeddings().unwrap();
        db.store_embeddings("model-a", &[(missing[0].0.clone(), vec![1.0])])
            .unwrap();
        assert!(!db.ensure_embedding_model("model-a").unwrap());
        assert_eq!(db.embedding_count().unwrap(), 1);
        assert!(db.ensure_embedding_model("model-b").unwrap());
        assert_eq!(db.embedding_count().unwrap(), 0);
    }

    #[test]
    fn orphan_embeddings_are_pruned() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "a.txt", &[piece(0, "text one")]);
        let missing = db.missing_embeddings().unwrap();
        db.store_embeddings("m", &[(missing[0].0.clone(), vec![1.0])])
            .unwrap();
        db.remove_file("a.txt").unwrap();
        assert_eq!(db.prune_orphan_embeddings().unwrap(), 1);
    }

    #[test]
    fn shared_content_hash_reuses_embeddings() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "a.txt", &[piece(0, "identical text")]);
        index_file(&mut db, "b.txt", &[piece(0, "identical text")]);
        // Two chunks, one distinct hash to embed.
        assert_eq!(db.chunk_count().unwrap(), 2);
        assert_eq!(db.missing_embeddings().unwrap().len(), 1);
    }

    #[test]
    fn fts_operators_cannot_break_search() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "a.txt", &[piece(0, "plain text here")]);
        for q in ["AND OR NOT", "\"unbalanced", "col:x", "(paren"] {
            db.search(q, 10).unwrap();
        }
    }

    #[test]
    fn status_counts_group_by_status() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "a.txt", &[piece(0, "x")]);
        db.upsert_unextracted_file("b.pdf", "pdf", 1, 1, FileStatus::Pending, None)
            .unwrap();
        db.upsert_unextracted_file("c.pdf", "pdf", 1, 1, FileStatus::Pending, None)
            .unwrap();
        let counts = db.status_counts().unwrap();
        assert!(counts.contains(&("indexed".to_string(), 1)));
        assert!(counts.contains(&("pending".to_string(), 2)));
    }

    #[test]
    fn enrichment_roundtrips_and_summary_is_searchable() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "House/closing.pdf", &[piece(0, "raw ocr text")]);
        let doc = db.document_by_path("House/closing.pdf").unwrap().unwrap();
        assert_eq!(db.documents_needing_enrichment(false, 10).unwrap().len(), 1);

        let record = EnrichmentRecord {
            title: Some("Closing Disclosure — 423 R St".into()),
            doc_type: "closing_statement".into(),
            summary: "Sale of 423 R St for $485,000, closing January 2026.".into(),
            facts: vec![
                (
                    "sale_price".into(),
                    "$485,000".into(),
                    "amount".into(),
                    Some(1),
                ),
                (
                    "closing_date".into(),
                    "01/13/2026".into(),
                    "date".into(),
                    None,
                ),
            ],
            tags: vec!["Real-Estate".into(), "closing".into()],
        };
        db.apply_enrichment(doc.document_id, &record, "test-model", 5000)
            .unwrap();

        assert!(db.documents_needing_enrichment(false, 10).unwrap().is_empty());
        let enriched = db.document_by_path("House/closing.pdf").unwrap().unwrap();
        assert_eq!(enriched.doc_type.as_deref(), Some("closing_statement"));
        assert_eq!(enriched.tags, vec!["closing", "real-estate"]);

        // The summary chunk is searchable and facts are normalized.
        assert!(!db.search("485,000", 10).unwrap().is_empty());
        let facts = db.facts_for_document(doc.document_id).unwrap();
        assert_eq!(facts.len(), 2);
        let by_key = db
            .search_facts(Some("sale_price"), None, None, 10)
            .unwrap();
        assert_eq!(by_key.len(), 1);

        // Re-applying replaces rather than duplicates.
        db.apply_enrichment(doc.document_id, &record, "test-model", 6000)
            .unwrap();
        assert_eq!(db.facts_for_document(doc.document_id).unwrap().len(), 2);
        assert_eq!(db.chunk_count().unwrap(), 2); // 1 content + 1 summary
    }

    #[test]
    fn reindexing_clears_enrichment_state() {
        let (_dir, mut db) = open_temp();
        index_file(&mut db, "a.txt", &[piece(0, "v1 text")]);
        let doc = db.document_by_path("a.txt").unwrap().unwrap();
        let record = EnrichmentRecord {
            title: None,
            doc_type: "note".into(),
            summary: "a note".into(),
            facts: vec![],
            tags: vec![],
        };
        db.apply_enrichment(doc.document_id, &record, "m", 5000).unwrap();
        // Content changed → re-ingest replaces the document row, so it
        // needs enrichment again and the stale summary chunk is gone.
        index_file(&mut db, "a.txt", &[piece(0, "v2 text")]);
        assert_eq!(db.documents_needing_enrichment(false, 10).unwrap().len(), 1);
        assert_eq!(db.chunk_count().unwrap(), 1);
    }

    #[test]
    fn normalize_value_handles_amounts_and_dates() {
        assert_eq!(normalize_value("amount", "$485,000"), Some("485000".into()));
        assert_eq!(normalize_value("amount", "$1,234.56"), Some("1234.56".into()));
        assert_eq!(normalize_value("amount", "n/a"), None);
        assert_eq!(
            normalize_value("date", "01/13/2026"),
            Some("2026-01-13".into())
        );
        assert_eq!(
            normalize_value("date", "January 13, 2026"),
            Some("2026-01-13".into())
        );
        assert_eq!(normalize_value("date", "sometime soon"), None);
        assert_eq!(normalize_value("party", "Timothy Williams"), None);
    }

    #[test]
    fn last_scan_roundtrips() {
        let (_dir, db) = open_temp();
        assert_eq!(db.last_scan_ms().unwrap(), None);
        db.set_last_scan_ms(12345).unwrap();
        assert_eq!(db.last_scan_ms().unwrap(), Some(12345));
    }
}
