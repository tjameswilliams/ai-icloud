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
            "INSERT INTO documents (file_id, title) VALUES (?1, ?2)",
            params![file_id, title],
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
        db.upsert_indexed_file(rel_path, "text", 10, 1000, "sha", None, chunks, 2000)
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
    fn last_scan_roundtrips() {
        let (_dir, db) = open_temp();
        assert_eq!(db.last_scan_ms().unwrap(), None);
        db.set_last_scan_ms(12345).unwrap();
        assert_eq!(db.last_scan_ms().unwrap(), Some(12345));
    }
}
