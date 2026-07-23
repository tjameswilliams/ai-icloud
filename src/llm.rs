//! OpenAI-compatible chat client (LM Studio by default): JSON-schema-
//! constrained completions, with optional image parts routed to the
//! configured vision model. Reasoning models are handled by giving
//! generous token budgets — their thinking spends tokens before any
//! content appears.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::config::Config;
use crate::embed::is_loopback_url;

/// Generous ceiling: reasoning models (qwen3.x) burn budget on thinking
/// before emitting the constrained JSON.
const MAX_TOKENS: u32 = 8192;

/// First call may JIT-load a large model; later calls are fast.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

pub enum Part {
    Text(String),
    /// Base64 PNG, attached as a data URL image part.
    PngBase64(String),
}

#[derive(Debug)]
pub struct LlmClient {
    base_url: String,
    api_key: Option<String>,
    model: String,
    vision_model: String,
}

impl LlmClient {
    pub fn from_config(config: &Config) -> Result<LlmClient> {
        let base_url = config.llm.base_url.trim_end_matches('/').to_string();
        if !is_loopback_url(&base_url) && !config.privacy.allow_remote_endpoints {
            bail!(
                "llm.base_url {base_url} is not a loopback address; sending document \
                 content there requires privacy.allow_remote_endpoints = true"
            );
        }
        Ok(LlmClient {
            base_url,
            api_key: config.llm.api_key.clone(),
            model: config.llm.model.clone(),
            vision_model: config.llm.vision_model.clone(),
        })
    }

    /// One chat completion constrained to `schema`, returning the parsed
    /// JSON. Image parts route the request to the vision model. Retries
    /// once with a nudge when the model returns unparsable content.
    pub fn chat_json(
        &self,
        system: &str,
        parts: &[Part],
        schema_name: &str,
        schema: &Value,
    ) -> Result<Value> {
        let has_images = parts.iter().any(|p| matches!(p, Part::PngBase64(_)));
        let model = if has_images && !self.vision_model.is_empty() {
            &self.vision_model
        } else {
            &self.model
        };

        let content: Vec<Value> = parts
            .iter()
            .map(|p| match p {
                Part::Text(t) => json!({ "type": "text", "text": t }),
                Part::PngBase64(b64) => json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:image/png;base64,{b64}") }
                }),
            })
            .collect();

        let mut body = json!({
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": content },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": { "name": schema_name, "strict": true, "schema": schema }
            },
            "max_tokens": MAX_TOKENS,
            "temperature": 0.1,
        });
        if !model.is_empty() {
            body["model"] = json!(model);
        }

        let raw = self.post_completions(&body)?;
        match parse_json_content(&raw) {
            Ok(v) => Ok(v),
            Err(first_err) => {
                tracing::debug!("unparsable LLM content, retrying once: {first_err:#}");
                body["messages"].as_array_mut().unwrap().push(json!({
                    "role": "user",
                    "content": "Your previous answer was not valid JSON. Respond with \
                                ONLY a JSON object matching the schema."
                }));
                let raw = self.post_completions(&body)?;
                parse_json_content(&raw)
            }
        }
    }

    fn post_completions(&self, body: &Value) -> Result<String> {
        let mut req = ureq::post(&format!("{}/chat/completions", self.base_url))
            .timeout(REQUEST_TIMEOUT);
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let resp: Value = req
            .send_json(body.clone())
            .map_err(flatten_ureq_error)
            .with_context(|| format!("chat request to {} failed", self.base_url))?
            .into_json()
            .context("chat endpoint returned non-JSON")?;

        let choice = resp
            .get("choices")
            .and_then(|c| c.get(0))
            .context("chat endpoint returned no choices")?;
        let content = choice
            .pointer("/message/content")
            .and_then(Value::as_str)
            .unwrap_or("");
        if content.trim().is_empty() {
            let finish = choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .unwrap_or("?");
            bail!(
                "chat endpoint returned empty content (finish_reason: {finish}) — \
                 a reasoning model may have exhausted max_tokens while thinking"
            );
        }
        Ok(content.to_string())
    }
}

/// Body-preserving error: ureq's Display for HTTP errors drops the body,
/// which is where LM Studio explains itself.
fn flatten_ureq_error(err: ureq::Error) -> anyhow::Error {
    match err {
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            let brief: String = body.chars().take(400).collect();
            anyhow::anyhow!("HTTP {code}: {brief}")
        }
        other => anyhow::anyhow!(other),
    }
}

/// Parse model output as JSON, tolerating markdown code fences and stray
/// prose around the object.
pub fn parse_json_content(content: &str) -> Result<Value> {
    let trimmed = content.trim();
    if let Ok(v) = serde_json::from_str(trimmed) {
        return Ok(v);
    }
    // ```json ... ``` fences
    let unfenced = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .map(str::trim);
    if let Some(u) = unfenced
        && let Ok(v) = serde_json::from_str(u)
    {
        return Ok(v);
    }
    // Last resort: the outermost {...} span.
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}'))
        && start < end
        && let Ok(v) = serde_json::from_str(&trimmed[start..=end])
    {
        return Ok(v);
    }
    bail!(
        "model did not return valid JSON: {}",
        trimmed.chars().take(200).collect::<String>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_json_parses() {
        assert_eq!(parse_json_content(r#"{"a":1}"#).unwrap()["a"], 1);
    }

    #[test]
    fn fenced_json_parses() {
        let content = "```json\n{\"a\": 2}\n```";
        assert_eq!(parse_json_content(content).unwrap()["a"], 2);
    }

    #[test]
    fn json_with_surrounding_prose_parses() {
        let content = "Here is the answer:\n{\"a\": 3}\nHope that helps!";
        assert_eq!(parse_json_content(content).unwrap()["a"], 3);
    }

    #[test]
    fn garbage_is_an_error() {
        assert!(parse_json_content("no json here").is_err());
        assert!(parse_json_content("").is_err());
    }

    #[test]
    fn remote_endpoint_requires_the_privacy_flag() {
        let mut config = Config::default();
        config.llm.base_url = "https://api.example.com/v1".into();
        let err = LlmClient::from_config(&config).unwrap_err().to_string();
        assert!(err.contains("allow_remote_endpoints"));
        config.privacy.allow_remote_endpoints = true;
        assert!(LlmClient::from_config(&config).is_ok());
    }

    #[test]
    fn loopback_default_needs_no_flag() {
        assert!(LlmClient::from_config(&Config::default()).is_ok());
    }
}
