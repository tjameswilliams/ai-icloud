//! Extractors for the plain-text family: txt, md, csv (verbatim, capped)
//! and html (tags stripped).

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

/// Cap on extracted text per file; beyond this, content is truncated at a
/// char boundary. Keeps a pathological file from bloating the index.
const MAX_TEXT_BYTES: usize = 2 * 1024 * 1024;

pub fn read_plain(path: &Path) -> Result<String> {
    let bytes =
        fs::read(path).with_context(|| format!("could not read {}", path.display()))?;
    Ok(cap(String::from_utf8_lossy(&bytes).into_owned()))
}

pub fn read_html(path: &Path) -> Result<String> {
    let raw = read_plain(path)?;
    Ok(cap(strip_html(&raw)))
}

fn cap(mut s: String) -> String {
    if s.len() > MAX_TEXT_BYTES {
        let mut end = MAX_TEXT_BYTES;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        s.truncate(end);
    }
    s
}

/// Strip tags, drop script/style bodies, decode the common entities, and
/// collapse runs of blank space. Deliberately not a full HTML parser —
/// good enough to make saved pages searchable.
fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut rest = html;
    let mut skip_until: Option<&str> = None;

    while let Some(lt) = rest.find('<') {
        if skip_until.is_none() {
            out.push_str(&rest[..lt]);
        }
        rest = &rest[lt..];
        let Some(gt) = rest.find('>') else { break };
        let tag = &rest[1..gt];
        let tag_name: String = tag
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();

        if let Some(closer) = skip_until {
            if tag.trim_start().starts_with('/') && tag_name == closer {
                skip_until = None;
            }
        } else if matches!(tag_name.as_str(), "script" | "style")
            && !tag.trim_start().starts_with('/')
            && !tag.ends_with('/')
        {
            skip_until = Some(if tag_name == "script" { "script" } else { "style" });
        } else if matches!(
            tag_name.as_str(),
            "p" | "br" | "div" | "tr" | "li" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "table"
        ) {
            out.push('\n');
        }
        rest = &rest[gt + 1..];
    }
    if skip_until.is_none() {
        out.push_str(rest);
    }

    let decoded = out
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");

    // Collapse horizontal whitespace and blank-line runs.
    let mut result = String::with_capacity(decoded.len());
    for line in decoded.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            if !result.ends_with("\n\n") && !result.is_empty() {
                result.push('\n');
            }
        } else {
            result.push_str(&line);
            result.push('\n');
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_tags_and_keeps_text() {
        let html = "<html><body><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>";
        let text = strip_html(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Hello world."));
        assert!(!text.contains('<'));
    }

    #[test]
    fn script_and_style_bodies_are_dropped() {
        let html = "<p>keep</p><script>var secret = 1;</script><style>.x{color:red}</style><p>also</p>";
        let text = strip_html(html);
        assert!(text.contains("keep"));
        assert!(text.contains("also"));
        assert!(!text.contains("secret"));
        assert!(!text.contains("color"));
    }

    #[test]
    fn entities_are_decoded() {
        assert_eq!(strip_html("a &amp; b &lt;c&gt;"), "a & b <c>");
    }

    #[test]
    fn block_tags_become_line_breaks() {
        let text = strip_html("<tr><td>a</td></tr><tr><td>b</td></tr>");
        assert!(text.contains('\n'));
    }

    #[test]
    fn unclosed_script_does_not_leak_content() {
        let text = strip_html("<p>visible</p><script>hidden forever");
        assert!(text.contains("visible"));
        assert!(!text.contains("hidden"));
    }

    #[test]
    fn plain_read_handles_non_utf8_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.txt");
        std::fs::write(&p, [b'h', b'i', 0xFF, b'!']).unwrap();
        let text = read_plain(&p).unwrap();
        assert!(text.starts_with("hi"));
    }

    #[test]
    fn cap_truncates_at_char_boundary() {
        let s = "é".repeat(MAX_TEXT_BYTES); // 2 bytes each
        let capped = cap(s);
        assert!(capped.len() <= MAX_TEXT_BYTES);
        assert!(capped.chars().all(|c| c == 'é'));
    }
}
