//! Split extracted document text into retrieval-sized chunks.
//!
//! Chunks are identified by a SHA-256 content hash; embeddings are keyed
//! on that hash, so re-extracting a file only re-embeds chunks whose text
//! actually changed, and identical text shared between files is embedded
//! once. Multi-page documents carry 1-based page ranges on each chunk.

use sha2::{Digest, Sha256};

/// One chunk ready to be stored.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkPiece {
    pub seq: i64,
    /// True for the document-summary chunk (enrichment phase).
    pub is_summary: bool,
    pub page_start: Option<i64>,
    pub page_end: Option<i64>,
    pub text: String,
    pub content_hash: String,
}

/// Hex SHA-256 of the chunk text — the embedding key.
pub fn content_hash(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Whitespace-based token estimate: ~4 characters per token holds well
/// enough for chunk sizing in English prose.
fn approx_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

/// One packable unit: a paragraph and the (1-based) page it came from.
struct Segment<'a> {
    page: Option<i64>,
    text: &'a str,
}

/// Chunk single-page text (no page numbers on the chunks).
pub fn chunk_text(text: &str, target_tokens: u32, overlap_tokens: u32) -> Vec<ChunkPiece> {
    let segments = split_segments(text, None, target_tokens);
    pack(segments, target_tokens, overlap_tokens)
}

/// Chunk a multi-page document; every chunk records the page range its
/// segments came from.
pub fn chunk_pages(pages: &[String], target_tokens: u32, overlap_tokens: u32) -> Vec<ChunkPiece> {
    let mut segments = Vec::new();
    for (i, page) in pages.iter().enumerate() {
        segments.extend(split_segments(page, Some(i as i64 + 1), target_tokens));
    }
    pack(segments, target_tokens, overlap_tokens)
}

fn split_segments(text: &str, page: Option<i64>, target_tokens: u32) -> Vec<Segment<'_>> {
    let max_segment_chars = (target_tokens.max(50) as usize) * 4;
    let mut segments = Vec::new();
    for para in text.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if para.chars().count() <= max_segment_chars {
            segments.push(Segment { page, text: para });
        } else {
            segments.extend(
                hard_split(para, max_segment_chars)
                    .into_iter()
                    .map(|text| Segment { page, text }),
            );
        }
    }
    segments
}

fn pack(segments: Vec<Segment<'_>>, target_tokens: u32, overlap_tokens: u32) -> Vec<ChunkPiece> {
    let target_tokens = target_tokens.max(50) as usize;
    let overlap_chars = (overlap_tokens as usize) * 4;

    let mut chunks: Vec<ChunkPiece> = Vec::new();
    let mut current = String::new();
    let mut page_range: Option<(i64, i64)> = None;

    let push_chunk = |buf: &mut String,
                      range: &mut Option<(i64, i64)>,
                      chunks: &mut Vec<ChunkPiece>| {
        let text = buf.trim().to_string();
        if !text.is_empty() {
            chunks.push(ChunkPiece {
                seq: chunks.len() as i64,
                is_summary: false,
                page_start: range.map(|(s, _)| s),
                page_end: range.map(|(_, e)| e),
                content_hash: content_hash(&text),
                text,
            });
        }
        buf.clear();
        *range = None;
    };

    for seg in segments {
        let would_be = approx_tokens(&current) + approx_tokens(seg.text);
        if !current.is_empty() && would_be > target_tokens {
            let tail = overlap_tail(&current, overlap_chars);
            // The overlap text comes from the end of the finished chunk,
            // so the new chunk's range starts at that chunk's last page.
            let tail_page = page_range.map(|(_, e)| e);
            push_chunk(&mut current, &mut page_range, &mut chunks);
            current = tail;
            page_range = tail_page.map(|p| (p, p));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(seg.text);
        if let Some(p) = seg.page {
            page_range = Some(match page_range {
                None => (p, p),
                Some((s, e)) => (s.min(p), e.max(p)),
            });
        }
    }
    push_chunk(&mut current, &mut page_range, &mut chunks);
    chunks
}

/// Split an oversized paragraph at whitespace near the size limit.
fn hard_split(para: &str, max_chars: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = para;
    while rest.chars().count() > max_chars {
        let byte_limit = rest
            .char_indices()
            .nth(max_chars)
            .map(|(i, _)| i)
            .unwrap_or(rest.len());
        // Prefer breaking at the last whitespace before the limit.
        let cut = rest[..byte_limit]
            .rfind(char::is_whitespace)
            .filter(|&i| i > 0)
            .unwrap_or(byte_limit);
        out.push(rest[..cut].trim());
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

/// The last ~`overlap_chars` of a chunk, snapped forward to a whitespace
/// boundary so the overlap never starts mid-word.
fn overlap_tail(text: &str, overlap_chars: usize) -> String {
    if overlap_chars == 0 {
        return String::new();
    }
    let total = text.chars().count();
    if total <= overlap_chars {
        return text.to_string();
    }
    let start_byte = text
        .char_indices()
        .nth(total - overlap_chars)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let tail = &text[start_byte..];
    match tail.find(char::is_whitespace) {
        Some(ws) => tail[ws..].trim_start().to_string(),
        None => tail.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_and_content_based() {
        assert_eq!(content_hash("abc"), content_hash("abc"));
        assert_ne!(content_hash("abc"), content_hash("abd"));
        assert_eq!(content_hash("abc").len(), 64);
    }

    #[test]
    fn short_text_becomes_one_chunk_without_pages() {
        let chunks = chunk_text("hello world", 750, 80);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
        assert_eq!(chunks[0].seq, 0);
        assert_eq!(chunks[0].page_start, None);
        assert_eq!(chunks[0].page_end, None);
    }

    #[test]
    fn empty_text_yields_no_chunks() {
        assert!(chunk_text("", 750, 80).is_empty());
        assert!(chunk_text("\n\n  \n\n", 750, 80).is_empty());
        assert!(chunk_pages(&[String::new(), "  ".into()], 750, 80).is_empty());
    }

    #[test]
    fn long_text_splits_into_multiple_chunks_with_sequential_seq() {
        let para = "lorem ipsum dolor sit amet ".repeat(40);
        let text = [para.as_str(); 6].join("\n\n");
        let chunks = chunk_text(&text, 200, 20);
        assert!(chunks.len() > 1, "expected a split, got {}", chunks.len());
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.seq, i as i64);
        }
    }

    #[test]
    fn oversized_single_paragraph_is_hard_split() {
        let text = "word ".repeat(2000);
        let chunks = chunk_text(&text, 100, 0);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(approx_tokens(&c.text) <= 150, "chunk too big: {}", c.text.len());
        }
    }

    #[test]
    fn consecutive_chunks_share_overlap_text() {
        let para = "alpha beta gamma delta epsilon zeta ".repeat(30);
        let text = [para.as_str(); 4].join("\n\n");
        let chunks = chunk_text(&text, 150, 40);
        assert!(chunks.len() > 1);
        // The second chunk starts with the tail of the first.
        let head: String = chunks[1].text.chars().take(20).collect();
        assert!(chunks[0].text.contains(head.trim()));
    }

    #[test]
    fn multibyte_text_does_not_panic() {
        let text = "héllo wörld émoji 🎉 ".repeat(500);
        let chunks = chunk_text(&text, 100, 20);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn small_pages_pack_together_with_a_page_range() {
        let pages = vec!["page one text".to_string(), "page two text".to_string()];
        let chunks = chunk_pages(&pages, 750, 80);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].page_start, Some(1));
        assert_eq!(chunks[0].page_end, Some(2));
    }

    #[test]
    fn large_pages_get_their_own_chunks_with_correct_pages() {
        let big = "content words repeated here ".repeat(60);
        let pages = vec![big.clone(), big.clone(), big];
        let chunks = chunk_pages(&pages, 200, 0);
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0].page_start, Some(1));
        assert_eq!(chunks.last().unwrap().page_end, Some(3));
        for c in &chunks {
            assert!(c.page_start.is_some() && c.page_end.is_some());
            assert!(c.page_start.unwrap() <= c.page_end.unwrap());
        }
    }

    #[test]
    fn empty_pages_between_content_pages_are_skipped() {
        let pages = vec![
            "first page".to_string(),
            String::new(),
            "third page".to_string(),
        ];
        let chunks = chunk_pages(&pages, 750, 80);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].page_start, Some(1));
        assert_eq!(chunks[0].page_end, Some(3));
    }
}
