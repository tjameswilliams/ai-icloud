//! TOML configuration with full defaults — the tool must work with no
//! config file at all.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paths;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub source: SourceConfig,
    pub llm: LlmConfig,
    pub embeddings: EmbeddingsConfig,
    pub transcription: TranscriptionConfig,
    pub watch: WatchConfig,
    pub index: IndexConfig,
    pub retrieval: RetrievalConfig,
    pub research: ResearchConfig,
    pub service: ServiceConfig,
    pub privacy: PrivacyConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SourceConfig {
    /// Root of the tree to index. Defaults to the iCloud Drive folder.
    pub root: String,
    /// Lowercase extensions considered in-scope.
    pub include_extensions: Vec<String>,
    /// Glob patterns (relative to root) that are never indexed and never
    /// stored — the privacy boundary for sensitive folders.
    pub exclude_globs: Vec<String>,
    pub max_file_mb: u64,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            root: format!("~/{}", paths::ICLOUD_ROOT_SUFFIX),
            include_extensions: [
                "pdf", "txt", "md", "html", "csv", "png", "jpg", "jpeg", "heic", "mp3", "mp4",
                "m4a",
            ]
            .map(String::from)
            .to_vec(),
            exclude_globs: Vec::new(),
            max_file_mb: 200,
        }
    }
}

/// OpenAI-compatible endpoint used for enrichment and the `ask` tool.
/// Defaults to LM Studio on loopback.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    pub base_url: String,
    /// Redacted in `config show`.
    pub api_key: Option<String>,
    /// Chat/enrichment model; empty means "whatever the server default is".
    pub model: String,
    /// Vision model for page/image passes; empty means `model`.
    pub vision_model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:1234/v1".into(),
            api_key: None,
            model: String::new(),
            vision_model: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingsConfig {
    /// "embedded" (local ONNX model), "openai-compatible", or "debug-hash".
    pub provider: String,
    pub model: String,
    pub batch_size: u32,
    /// Only used by the openai-compatible provider.
    pub base_url: Option<String>,
    /// Only used by the openai-compatible provider. Redacted in `config show`.
    pub api_key: Option<String>,
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            provider: "embedded".into(),
            model: "bge-small-en-v1.5".into(),
            batch_size: 32,
            base_url: None,
            api_key: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TranscriptionConfig {
    pub enabled: bool,
    pub whisper_model: String,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            whisper_model: "large-v3-turbo".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WatchConfig {
    pub debounce_ms: u64,
    /// Reconciliation sweep interval for the watch daemon.
    pub full_scan_interval_minutes: u32,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 2000,
            full_scan_interval_minutes: 360,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IndexConfig {
    pub database_path: String,
    pub chunk_target_tokens: u32,
    pub chunk_overlap_tokens: u32,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            database_path: "~/Library/Application Support/ai-icloud/index.sqlite".into(),
            chunk_target_tokens: 750,
            chunk_overlap_tokens: 80,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetrievalConfig {
    pub fts_candidates: u32,
    pub vector_candidates: u32,
    pub result_limit: u32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            fts_candidates: 30,
            vector_candidates: 30,
            result_limit: 10,
        }
    }
}

/// Budget for the `ask` (deepResearch) tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResearchConfig {
    pub max_iterations: u32,
    pub max_context_chunks: u32,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            max_iterations: 6,
            max_context_chunks: 24,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ServiceConfig {
    /// Bearer token required by `serve --http`. When unset, a random token
    /// is generated on first use and stored next to the index. Redacted in
    /// `config show`.
    pub http_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct PrivacyConfig {
    /// Off by default: document content may only be sent to loopback
    /// endpoints (LLM and embeddings alike); anything else must be
    /// explicitly allowed.
    pub allow_remote_endpoints: bool,
}

impl Config {
    /// Absolute path to the indexed tree root, with `~` expanded.
    pub fn source_root(&self) -> Result<PathBuf> {
        paths::expand_tilde(&self.source.root)
    }

    /// Absolute path to the index database, with `~` expanded.
    pub fn index_db_path(&self) -> Result<PathBuf> {
        paths::expand_tilde(&self.index.database_path)
    }

    /// Directory that will hold the index (and model cache, logs, tokens).
    pub fn index_dir(&self) -> Result<PathBuf> {
        let p = self.index_db_path()?;
        Ok(p.parent().map(Path::to_path_buf).unwrap_or(p))
    }

    /// A copy safe for display: secrets are replaced, never printed.
    pub fn redacted(&self) -> Config {
        let mut c = self.clone();
        if c.llm.api_key.is_some() {
            c.llm.api_key = Some("<redacted>".into());
        }
        if c.embeddings.api_key.is_some() {
            c.embeddings.api_key = Some("<redacted>".into());
        }
        if c.service.http_token.is_some() {
            c.service.http_token = Some("<redacted>".into());
        }
        c
    }

    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("could not serialize configuration")
    }
}

/// A configuration plus where it came from.
#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub config: Config,
    pub path: PathBuf,
    pub from_file: bool,
}

/// Load configuration.
///
/// An explicitly passed path must exist. The default path is optional:
/// missing means "use built-in defaults".
pub fn load(explicit: Option<&Path>) -> Result<LoadedConfig> {
    match explicit {
        Some(p) => {
            let raw = fs::read_to_string(p)
                .with_context(|| format!("could not read config file {}", p.display()))?;
            Ok(LoadedConfig {
                config: parse(&raw).with_context(|| format!("in config file {}", p.display()))?,
                path: p.to_path_buf(),
                from_file: true,
            })
        }
        None => {
            let p = paths::default_config_path()?;
            if p.exists() {
                let raw = fs::read_to_string(&p)
                    .with_context(|| format!("could not read config file {}", p.display()))?;
                Ok(LoadedConfig {
                    config: parse(&raw)
                        .with_context(|| format!("in config file {}", p.display()))?,
                    path: p,
                    from_file: true,
                })
            } else {
                Ok(LoadedConfig {
                    config: Config::default(),
                    path: p,
                    from_file: false,
                })
            }
        }
    }
}

pub fn parse(raw: &str) -> Result<Config> {
    toml::from_str(raw).context("config file is not valid TOML for ai-icloud")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_spec() {
        let c = Config::default();
        assert!(c.source.root.ends_with("com~apple~CloudDocs"));
        assert!(c.source.include_extensions.contains(&"pdf".to_string()));
        assert!(c.source.exclude_globs.is_empty());
        assert_eq!(c.source.max_file_mb, 200);
        assert_eq!(c.llm.base_url, "http://127.0.0.1:1234/v1");
        assert_eq!(c.embeddings.provider, "embedded");
        assert_eq!(c.embeddings.model, "bge-small-en-v1.5");
        assert_eq!(c.watch.debounce_ms, 2000);
        assert_eq!(c.index.chunk_target_tokens, 750);
        assert_eq!(c.index.chunk_overlap_tokens, 80);
        assert_eq!(c.retrieval.result_limit, 10);
        assert_eq!(c.research.max_iterations, 6);
        assert!(!c.privacy.allow_remote_endpoints);
    }

    #[test]
    fn empty_file_yields_defaults() {
        assert_eq!(parse("").unwrap(), Config::default());
    }

    #[test]
    fn partial_file_overrides_only_named_keys() {
        let c = parse("[index]\nchunk_target_tokens = 500\n").unwrap();
        assert_eq!(c.index.chunk_target_tokens, 500);
        assert_eq!(c.index.chunk_overlap_tokens, 80);
        assert_eq!(c.retrieval.result_limit, 10);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        // Typos must fail loudly, not be silently ignored.
        assert!(parse("[index]\nchunk_target_tokenss = 1\n").is_err());
        assert!(parse("[indexx]\nchunk_target_tokens = 1\n").is_err());
    }

    #[test]
    fn serialization_roundtrips() {
        let mut c = Config::default();
        c.llm.api_key = Some("k".into());
        c.source.exclude_globs = vec!["Divorce/**".into()];
        let raw = c.to_toml().unwrap();
        assert_eq!(parse(&raw).unwrap(), c);
    }

    #[test]
    fn secrets_are_redacted() {
        let mut c = Config::default();
        c.llm.api_key = Some("llm-secret".into());
        c.embeddings.api_key = Some("emb-secret".into());
        c.service.http_token = Some("tok-secret".into());
        let shown = c.redacted().to_toml().unwrap();
        for secret in ["llm-secret", "emb-secret", "tok-secret"] {
            assert!(!shown.contains(secret));
        }
        assert!(shown.contains("<redacted>"));
    }

    #[test]
    fn source_root_expands_tilde() {
        let p = Config::default().source_root().unwrap();
        assert!(p.is_absolute());
        assert!(!p.to_string_lossy().contains('~') || p.to_string_lossy().contains("com~apple"));
        assert!(p.ends_with("Library/Mobile Documents/com~apple~CloudDocs"));
    }

    #[test]
    fn explicit_missing_path_is_an_error() {
        assert!(load(Some(Path::new("/nonexistent/config.toml"))).is_err());
    }
}
