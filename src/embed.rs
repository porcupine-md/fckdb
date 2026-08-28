//! Native embedding: turning text into a vector at query or write time.
//!
//! The engine does not host a model. It calls one, through an
//! OpenAI-compatible `/v1/embeddings` endpoint, which is what almost every
//! provider and local server (vLLM, TEI, Ollama, LM Studio) speaks.
//!
//! Embedding is deliberately a trait rather than a concrete client. Two reasons:
//! tests must not make network calls, and a query that embeds is a query whose
//! latency and failure modes belong to someone else's service — which is worth
//! being able to swap.
//!
//! When no endpoint is configured, requests that need one are refused by name
//! rather than served with a zero vector. A zero vector would rank every
//! document equally and look like a working search returning nonsense.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub trait Embedder: Send + Sync {
    /// Embed a batch of texts, in order.
    ///
    /// A batch rather than one text at a time: embedding endpoints charge and
    /// rate-limit per request, and a write of a thousand documents should be one
    /// call, not a thousand.
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        model: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>>>;

    /// The model used when a request does not name one.
    fn default_model(&self) -> Option<&str>;
}

/// How an embedding endpoint is reached.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    pub url: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub timeout: std::time::Duration,
}

impl EmbedConfig {
    ///   FCKDB_EMBED_URL     e.g. https://api.openai.com/v1/embeddings
    ///   FCKDB_EMBED_MODEL   default model, when a request does not name one
    ///   FCKDB_EMBED_KEY     bearer token, if the endpoint needs one
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("FCKDB_EMBED_URL").ok()?;
        Some(Self {
            url,
            model: std::env::var("FCKDB_EMBED_MODEL").ok(),
            api_key: std::env::var("FCKDB_EMBED_KEY").ok(),
            timeout: std::time::Duration::from_secs(
                std::env::var("FCKDB_EMBED_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(30),
            ),
        })
    }
}

pub struct HttpEmbedder {
    client: reqwest::Client,
    config: EmbedConfig,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
    /// Providers are not required to return the batch in order, so the index is
    /// used rather than assumed.
    #[serde(default)]
    index: usize,
}

impl HttpEmbedder {
    pub fn new(config: EmbedConfig) -> Result<Self> {
        let client = reqwest::Client::builder().timeout(config.timeout).build()?;
        Ok(Self { client, config })
    }

    pub fn from_env() -> Result<Option<Arc<dyn Embedder>>> {
        match EmbedConfig::from_env() {
            None => Ok(None),
            Some(config) => Ok(Some(Arc::new(Self::new(config)?))),
        }
    }
}

impl Embedder for HttpEmbedder {
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        model: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>>> {
        Box::pin(async move {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            let model = model.or(self.config.model.as_deref());
            let mut request = self
                .client
                .post(&self.config.url)
                .json(&EmbedRequest { input: texts, model });
            if let Some(key) = &self.config.api_key {
                request = request.bearer_auth(key);
            }

            let response = request.send().await.context("calling the embedding endpoint")?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                // Include the provider's own message: "model not found" and
                // "rate limited" need very different responses from the caller.
                bail!("embedding endpoint returned {status}: {}", body.trim());
            }

            let parsed: EmbedResponse = serde_json::from_str(&body)
                .with_context(|| format!("decoding the embedding response: {}", truncate(&body)))?;
            if parsed.data.len() != texts.len() {
                bail!(
                    "embedding endpoint returned {} vectors for {} inputs",
                    parsed.data.len(),
                    texts.len()
                );
            }

            // Place by index rather than trusting arrival order: a mismatch would
            // attach one document's vector to another, which is the kind of bug
            // that shows up as slightly-wrong search results forever.
            let mut out = vec![Vec::new(); texts.len()];
            for datum in parsed.data {
                let slot = out.get_mut(datum.index).ok_or_else(|| {
                    anyhow::anyhow!("embedding response index {} is out of range", datum.index)
                })?;
                if !slot.is_empty() {
                    bail!("embedding response repeated index {}", datum.index);
                }
                if datum.embedding.is_empty() {
                    bail!("embedding endpoint returned an empty vector");
                }
                if datum.embedding.iter().any(|f| !f.is_finite()) {
                    bail!("embedding endpoint returned a non-finite component");
                }
                *slot = datum.embedding;
            }
            if let Some(missing) = out.iter().position(|v| v.is_empty()) {
                bail!("embedding response is missing index {missing}");
            }
            Ok(out)
        })
    }

    fn default_model(&self) -> Option<&str> {
        self.config.model.as_deref()
    }
}

fn truncate(s: &str) -> String {
    let s = s.trim();
    if s.len() <= 200 { s.to_string() } else { format!("{}…", &s[..200]) }
}

/// Embed one text, or explain why it cannot be done.
pub async fn embed_one(
    embedder: Option<&Arc<dyn Embedder>>,
    text: &str,
    model: Option<&str>,
) -> Result<Vec<f32>> {
    let Some(embedder) = embedder else {
        bail!(
            "native embedding is not configured; set FCKDB_EMBED_URL to an \
             OpenAI-compatible /v1/embeddings endpoint, or send a vector instead"
        );
    };
    let batch = vec![text.to_string()];
    let mut vectors = embedder.embed(&batch, model).await?;
    vectors.pop().ok_or_else(|| anyhow::anyhow!("embedding endpoint returned nothing"))
}

/// Embed a batch, or explain why it cannot be done.
pub async fn embed_many(
    embedder: Option<&Arc<dyn Embedder>>,
    texts: &[String],
    model: Option<&str>,
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    let Some(embedder) = embedder else {
        bail!(
            "native embedding is not configured; set FCKDB_EMBED_URL to an \
             OpenAI-compatible /v1/embeddings endpoint, or send vectors instead"
        );
    };
    embedder.embed(texts, model).await
}

#[cfg(test)]
pub mod testing {
    use super::*;
    use std::sync::Mutex;

    /// A stub embedder. Deterministic, offline, and deliberately confined to
    /// tests: a hash-based vector is useless for real retrieval, and shipping one
    /// would let a misconfigured deployment look like it was working.
    pub struct StubEmbedder {
        pub dim: usize,
        pub calls: Mutex<Vec<(Vec<String>, Option<String>)>>,
        pub fail: Option<String>,
    }

    impl StubEmbedder {
        pub fn new(dim: usize) -> Self {
            Self { dim, calls: Mutex::new(vec![]), fail: None }
        }

        pub fn failing(message: &str) -> Self {
            Self { dim: 4, calls: Mutex::new(vec![]), fail: Some(message.to_string()) }
        }

        pub fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    impl Embedder for StubEmbedder {
        fn embed<'a>(
            &'a self,
            texts: &'a [String],
            model: Option<&'a str>,
        ) -> BoxFuture<'a, Result<Vec<Vec<f32>>>> {
            Box::pin(async move {
                self.calls.lock().unwrap().push((texts.to_vec(), model.map(str::to_string)));
                if let Some(message) = &self.fail {
                    bail!("{message}");
                }
                Ok(texts
                    .iter()
                    .map(|t| {
                        // Same text always yields the same vector, and similar
                        // texts share leading tokens, which is enough for a test
                        // to assert that ranking followed the text.
                        let mut v = vec![0.0f32; self.dim];
                        for (i, token) in t.split_whitespace().enumerate() {
                            let slot = (token.len() + i) % self.dim;
                            v[slot] += 1.0;
                        }
                        if v.iter().all(|x| *x == 0.0) {
                            v[0] = 1.0;
                        }
                        v
                    })
                    .collect())
            })
        }

        fn default_model(&self) -> Option<&str> {
            Some("stub")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::StubEmbedder;
    use super::*;

    #[tokio::test]
    async fn an_unconfigured_embedder_refuses_by_name() {
        let err = embed_one(None, "hello", None).await.unwrap_err().to_string();
        assert!(err.contains("FCKDB_EMBED_URL"), "unhelpful error: {err}");
        // Never a zero vector: that would rank every document equally and look
        // like a working search returning nonsense.
        assert!(embed_many(None, &["a".to_string()], None).await.is_err());
        // An empty batch needs no endpoint at all.
        assert!(embed_many(None, &[], None).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn the_stub_is_deterministic_and_records_its_calls() {
        let stub: Arc<dyn Embedder> = Arc::new(StubEmbedder::new(8));
        let a = embed_one(Some(&stub), "quick brown fox", None).await.unwrap();
        let b = embed_one(Some(&stub), "quick brown fox", None).await.unwrap();
        assert_eq!(a, b, "the same text produced different vectors");
        assert_eq!(a.len(), 8);
        let c = embed_one(Some(&stub), "entirely different words here", None).await.unwrap();
        assert_ne!(a, c);
    }

    #[tokio::test]
    async fn a_batch_is_one_call() {
        let stub = Arc::new(StubEmbedder::new(4));
        let dynamic: Arc<dyn Embedder> = stub.clone();
        let texts: Vec<String> = (0..100).map(|i| format!("doc {i}")).collect();
        let vectors = embed_many(Some(&dynamic), &texts, Some("m")).await.unwrap();
        assert_eq!(vectors.len(), 100);
        assert_eq!(stub.call_count(), 1, "a batch should not become 100 requests");
        let calls = stub.calls.lock().unwrap();
        assert_eq!(calls[0].1.as_deref(), Some("m"), "the requested model was dropped");
    }

    #[tokio::test]
    async fn a_failing_endpoint_surfaces_its_message() {
        let stub: Arc<dyn Embedder> = Arc::new(StubEmbedder::failing("model not found"));
        let err = embed_one(Some(&stub), "x", None).await.unwrap_err().to_string();
        assert!(err.contains("model not found"), "the provider's message was lost: {err}");
    }

    #[test]
    fn config_comes_from_the_environment_or_not_at_all() {
        // Nothing set: no config, and therefore no embedding rather than a
        // default endpoint nobody asked for.
        assert!(EmbedConfig::from_env().is_none() || std::env::var("FCKDB_EMBED_URL").is_ok());
    }

    #[test]
    fn long_error_bodies_are_truncated() {
        let long = "x".repeat(1000);
        assert!(truncate(&long).len() < 250);
        assert_eq!(truncate("  short  "), "short");
    }
}
