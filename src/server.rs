//! HTTP service: auth, multi-tenancy, backpressure, background compaction.
//!
//! Three things here are load-bearing and easy to get wrong:
//!
//! 1. Exactly ONE `GroupCommit` per namespace, for the whole process lifetime.
//!    Two committers on one namespace reintroduce CAS contention and the request
//!    amplification that made the per-write design unusable. The registry exists
//!    to guarantee that invariant, not to cache objects.
//!
//! 2. Namespace names reach object storage paths. They are validated, not
//!    sanitized — a name containing `..` or `/` would let one tenant address
//!    another's prefix.
//!
//! 3. Writes are refused before the unindexed tail grows past the point where
//!    queries can still be answered consistently. turbopuffer's documented
//!    behaviour past that point is that writes stop being visible while the API
//!    keeps returning success; refusing loudly is the same backpressure with an
//!    error the caller can act on.

use crate::cache::RingCache;
use crate::doc::Record;
use crate::ops::{self, Pricing};
use crate::store::{GroupCommit, MAX_UNINDEXED_SCAN_BYTES, Namespace};
use crate::wire::{QueryRequest, WriteRequest, WriteResponse};
use anyhow::Result;
use axum::extract::{DefaultBodyLimit, Path as AxPath, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use object_store::ObjectStore;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Refuse writes at half the query scan cap, so the cliff where a strongly
/// consistent query becomes impossible is never actually reached in normal
/// operation. Backpressure should bite before correctness has to bend.
pub const WRITE_BACKPRESSURE_BYTES: u64 = MAX_UNINDEXED_SCAN_BYTES / 2;

/// Compact once the unindexed tail passes this, whichever comes first with the
/// entry count. Small enough that queries never scan much, large enough that
/// compaction stays amortized.
const COMPACT_TRIGGER_BYTES: u64 = 8 << 20;
const COMPACT_TRIGGER_ENTRIES: usize = 32;
const COMPACT_SWEEP_INTERVAL: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------- auth

#[derive(Clone, Debug)]
pub struct Org(pub String);

/// Bearer token to organisation. One token per tenant.
///
/// ponytail: static tokens from the environment, no rotation, no expiry, no
/// scopes. Correct for an internal service behind a private network. Anything
/// user-facing needs real key management, and that is a service of its own.
#[derive(Clone, Default)]
pub struct Auth {
    tokens: Vec<(String, String)>,
}

impl Auth {
    /// `FCKDB_TOKENS=tok1:org1,tok2:org2`, or `FCKDB_TOKEN=tok` for a single
    /// tenant. With neither set, auth is disabled and that is logged loudly —
    /// silent open access is how internal tools end up on the public internet.
    pub fn from_env() -> Self {
        if let Ok(multi) = std::env::var("FCKDB_TOKENS") {
            let tokens = multi
                .split(',')
                .filter_map(|pair| pair.split_once(':'))
                .map(|(t, o)| (t.trim().to_string(), o.trim().to_string()))
                .collect();
            return Self { tokens };
        }
        if let Ok(single) = std::env::var("FCKDB_TOKEN") {
            return Self { tokens: vec![(single, "default".into())] };
        }
        Self::default()
    }

    /// Explicit construction, for tests and embedded use. Avoids configuring a
    /// service by mutating process environment variables.
    pub fn from_pairs(pairs: Vec<(String, String)>) -> Self {
        Self { tokens: pairs }
    }

    pub fn disabled(&self) -> bool {
        self.tokens.is_empty()
    }

    fn resolve(&self, headers: &HeaderMap) -> Option<Org> {
        if self.disabled() {
            return Some(Org("default".into()));
        }
        let presented = headers
            .get(header::AUTHORIZATION)?
            .to_str()
            .ok()?
            .strip_prefix("Bearer ")?
            .to_string();

        // Compare against every token rather than a map lookup: a hash lookup
        // leaks which prefix matched through timing, and the token count here is
        // tiny.
        let mut found = None;
        for (token, org) in &self.tokens {
            if ct_eq(token.as_bytes(), presented.as_bytes()) {
                found = Some(Org(org.clone()));
            }
        }
        found
    }
}

/// Constant-time comparison. Length is allowed to leak, as is standard: it is
/// the byte contents that must not be discoverable one character at a time.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Namespace names become object storage paths, so this is a trust boundary.
/// Allowlist, never a blocklist: `..` and `/` are the obvious attacks, but so are
/// NUL bytes, unicode normalisation tricks and absolute paths.
fn valid_namespace(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

// ---------------------------------------------------------------- state

struct Resident {
    ns: Arc<Namespace>,
    commit: Arc<GroupCommit>,
}

#[derive(Clone)]
pub struct AppState {
    store: Arc<dyn ObjectStore>,
    cache: Option<Arc<RingCache>>,
    /// The one-committer-per-namespace invariant lives here.
    resident: Arc<Mutex<HashMap<String, Arc<Resident>>>>,
    auth: Auth,
    pricing: Pricing,
    started: Instant,
}

impl AppState {
    pub fn new(store: Arc<dyn ObjectStore>, cache: Option<Arc<RingCache>>) -> Self {
        Self {
            store,
            cache,
            resident: Arc::new(Mutex::new(HashMap::new())),
            auth: Auth::from_env(),
            pricing: Pricing::from_env(),
            started: Instant::now(),
        }
    }

    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    fn key(org: &str, name: &str) -> String {
        format!("ns/{org}/{name}")
    }

    /// Get or create the resident handle. Created under the lock so two
    /// concurrent first-requests cannot produce two committers.
    fn resident(&self, org: &str, name: &str) -> Arc<Resident> {
        let key = Self::key(org, name);
        let mut map = self.resident.lock().unwrap();
        if let Some(found) = map.get(&key) {
            return found.clone();
        }
        let mut ns = Namespace::new(self.store.clone(), key.clone());
        if let Some(cache) = &self.cache {
            ns = ns.with_cache(cache.clone());
        }
        let ns = Arc::new(ns);
        let handle = Arc::new(Resident { commit: Arc::new(GroupCommit::new(ns.clone())), ns });
        map.insert(key, handle.clone());
        handle
    }

    fn all_resident(&self) -> Vec<Arc<Resident>> {
        self.resident.lock().unwrap().values().cloned().collect()
    }

    fn forget(&self, org: &str, name: &str) {
        self.resident.lock().unwrap().remove(&Self::key(org, name));
    }
}

// ---------------------------------------------------------------- errors

pub struct AppError(StatusCode, String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        #[derive(Serialize)]
        struct Body {
            error: String,
        }
        if self.0.is_server_error() {
            tracing::error!(status = %self.0, "{}", self.1);
        }
        let mut res = (self.0, Json(Body { error: self.1 })).into_response();
        if self.0 == StatusCode::TOO_MANY_REQUESTS {
            res.headers_mut().insert(header::RETRY_AFTER, "5".parse().unwrap());
        }
        res
    }
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
    }
}

fn bad(msg: impl Into<String>) -> AppError {
    AppError(StatusCode::BAD_REQUEST, msg.into())
}

type ApiResult<T> = std::result::Result<T, AppError>;

// ---------------------------------------------------------------- middleware

async fn authenticate(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> std::result::Result<Response, AppError> {
    let Some(org) = state.auth.resolve(req.headers()) else {
        return Err(AppError(
            StatusCode::UNAUTHORIZED,
            "missing or invalid bearer token".into(),
        ));
    };
    req.extensions_mut().insert(org);
    Ok(next.run(req).await)
}

// ---------------------------------------------------------------- handlers

async fn write(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
    Json(body): Json<WriteRequest>,
) -> ApiResult<Json<WriteResponse>> {
    if !valid_namespace(&name) {
        return Err(bad("namespace must be 1-64 chars of [A-Za-z0-9_-]"));
    }
    if body.upsert.is_empty() && body.delete.is_empty() {
        return Err(bad("write requires at least one upsert or delete"));
    }
    // Reject ragged vectors at the boundary rather than letting compaction fail
    // later, when the caller is long gone.
    if let Some(first) = body.upsert.first() {
        let dim = first.vector.len();
        if dim == 0 {
            return Err(bad("vectors must have at least one dimension"));
        }
        if let Some(odd) = body.upsert.iter().find(|d| d.vector.len() != dim) {
            return Err(bad(format!(
                "inconsistent dimensions in batch: doc {} has {}, expected {dim}",
                odd.id,
                odd.vector.len()
            )));
        }
        if body.upsert.iter().any(|d| d.vector.iter().any(|f| !f.is_finite())) {
            return Err(bad("vectors must not contain NaN or infinity"));
        }
    }

    let handle = state.resident(&org, &name);

    // Backpressure. Consulted from the manifest, which records sizes inline, so
    // this costs one small GET and never a LIST or a data fetch.
    let (manifest, _) = handle.ns.load().await?;
    if manifest.unindexed_bytes() >= WRITE_BACKPRESSURE_BYTES {
        handle
            .ns
            .metrics
            .backpressure_rejects
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let ns = handle.ns.clone();
        tokio::spawn(async move {
            if let Err(e) = ns.compact(true).await {
                tracing::error!("backpressure compaction failed: {e:#}");
            }
        });
        return Err(AppError(
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "unindexed tail is {} bytes, at or over the {WRITE_BACKPRESSURE_BYTES} byte \
                 write limit; compaction has been triggered, retry shortly",
                manifest.unindexed_bytes()
            ),
        ));
    }

    let t = Instant::now();
    let mut records: Vec<Record> = body.upsert.into_iter().map(Record::Upsert).collect();
    records.extend(body.delete.into_iter().map(Record::Delete));
    let count = records.len();
    let seq = handle.commit.write_all(records).await?;

    Ok(Json(WriteResponse { seq, records: count, took_ms: t.elapsed().as_millis() as u64 }))
}

async fn query(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
    Json(req): Json<QueryRequest>,
) -> ApiResult<Response> {
    if !valid_namespace(&name) {
        return Err(bad("namespace must be 1-64 chars of [A-Za-z0-9_-]"));
    }
    if req.vector.is_empty() {
        return Err(bad("query vector must not be empty"));
    }
    if req.vector.iter().any(|f| !f.is_finite()) {
        return Err(bad("query vector must not contain NaN or infinity"));
    }
    if req.top_k == 0 || req.top_k > 1000 {
        return Err(bad("top_k must be between 1 and 1000"));
    }

    let handle = state.resident(&org, &name);
    match handle.ns.query(&req).await {
        Ok(res) => Ok(Json(res).into_response()),
        Err(e) => {
            // The one expected failure: a strongly consistent answer is
            // temporarily impossible. That is 503 with a retry, not 500.
            let msg = format!("{e:#}");
            if msg.contains("scan cap") {
                Err(AppError(StatusCode::SERVICE_UNAVAILABLE, msg))
            } else {
                Err(AppError(StatusCode::INTERNAL_SERVER_ERROR, msg))
            }
        }
    }
}

async fn metadata(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
) -> ApiResult<Response> {
    if !valid_namespace(&name) {
        return Err(bad("invalid namespace"));
    }
    let handle = state.resident(&org, &name);
    let (m, version) = handle.ns.load().await?;
    if version.is_none() {
        return Err(AppError(StatusCode::NOT_FOUND, format!("namespace {name} does not exist")));
    }
    Ok(Json(handle.ns.metadata_from(&m)).into_response())
}

async fn compact(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
) -> ApiResult<Response> {
    if !valid_namespace(&name) {
        return Err(bad("invalid namespace"));
    }
    Ok(Json(state.resident(&org, &name).ns.compact(true).await?).into_response())
}

async fn warm(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
) -> ApiResult<Response> {
    if !valid_namespace(&name) {
        return Err(bad("invalid namespace"));
    }
    Ok(Json(state.resident(&org, &name).ns.warm().await?).into_response())
}

async fn gc(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
) -> ApiResult<Response> {
    if !valid_namespace(&name) {
        return Err(bad("invalid namespace"));
    }
    // One hour of grace. Must comfortably exceed the longest possible single
    // commit, because an orphan younger than that may be a write in flight.
    let res = state.resident(&org, &name).ns.gc(Duration::from_secs(3600)).await?;
    Ok(Json(res).into_response())
}

async fn branch(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath((name, dest)): AxPath<(String, String)>,
) -> ApiResult<Response> {
    if !valid_namespace(&name) || !valid_namespace(&dest) {
        return Err(bad("invalid namespace name"));
    }
    // Branching stays inside the caller's org. A destination is a namespace name,
    // never a path.
    let dest_prefix = AppState::key(&org, &dest);
    let objects = state.resident(&org, &name).ns.branch(&dest_prefix).await.map_err(|e| {
        let msg = format!("{e:#}");
        // "Destination exists" is the caller's problem to resolve, and they need
        // to tell it apart from a server fault.
        if msg.contains("already exists") {
            AppError(StatusCode::CONFLICT, msg)
        } else {
            AppError(StatusCode::INTERNAL_SERVER_ERROR, msg)
        }
    })?;

    #[derive(Serialize)]
    struct Body {
        source: String,
        destination: String,
        objects_copied: usize,
    }
    Ok(Json(Body { source: name, destination: dest, objects_copied: objects }).into_response())
}

async fn destroy(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
) -> ApiResult<Response> {
    if !valid_namespace(&name) {
        return Err(bad("invalid namespace"));
    }
    let deleted = state.resident(&org, &name).ns.destroy().await?;
    state.forget(&org, &name);

    #[derive(Serialize)]
    struct Body {
        namespace: String,
        objects_deleted: usize,
    }
    Ok(Json(Body { namespace: name, objects_deleted: deleted }).into_response())
}

async fn list_namespaces(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
) -> ApiResult<Response> {
    let prefix = object_store::path::Path::from(format!("ns/{org}"));
    let listing = state
        .store
        .list_with_delimiter(Some(&prefix))
        .await
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let mut names: Vec<String> = listing
        .common_prefixes
        .iter()
        .filter_map(|p| p.as_ref().rsplit('/').next().map(str::to_string))
        .collect();
    names.sort();

    #[derive(Serialize)]
    struct Body {
        namespaces: Vec<String>,
    }
    Ok(Json(Body { namespaces: names }).into_response())
}

async fn healthz() -> &'static str {
    "ok\n"
}

/// Prometheus scrape. Aggregates every resident namespace, and reports the one
/// number that tells you whether the commit protocol is healthy.
async fn metrics(State(state): State<AppState>) -> Response {
    let residents = state.all_resident();
    let mut agg = crate::store::MetricsSnapshot::default();
    let mut bytes = 0u64;
    for r in &residents {
        let s = r.ns.metrics.snapshot();
        agg.gets += s.gets;
        agg.puts += s.puts;
        agg.deletes += s.deletes;
        agg.lists += s.lists;
        agg.bytes_get += s.bytes_get;
        agg.bytes_put += s.bytes_put;
        agg.cas_conflicts += s.cas_conflicts;
        agg.queries += s.queries;
        agg.writes += s.writes;
        agg.compactions += s.compactions;
        agg.backpressure_rejects += s.backpressure_rejects;
        bytes += r.ns.metadata().await.map(|m| m.total_bytes).unwrap_or(0);
    }
    let cost = ops::estimate(&agg, bytes, state.started.elapsed(), &state.pricing);
    let body = ops::prometheus(&agg, &cost, residents.len());
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body).into_response()
}

// ---------------------------------------------------------------- background

/// Compact namespaces whose unindexed tail has grown, before it reaches the
/// point where queries slow down or writes get refused.
///
/// One sweeper for the whole process rather than a task per namespace: with
/// thousands of namespaces, per-namespace timers become the scaling problem.
pub fn spawn_compactor(state: AppState) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(COMPACT_SWEEP_INTERVAL).await;
            for r in state.all_resident() {
                let Ok((m, version)) = r.ns.load().await else { continue };
                if version.is_none() {
                    continue;
                }
                let due = m.unindexed_bytes() >= COMPACT_TRIGGER_BYTES
                    || m.wal.len() >= COMPACT_TRIGGER_ENTRIES;
                if !due {
                    continue;
                }
                tracing::info!(
                    namespace = %r.ns.prefix,
                    wal_entries = m.wal.len(),
                    unindexed_bytes = m.unindexed_bytes(),
                    "compacting"
                );
                if let Err(e) = r.ns.compact(true).await {
                    // A concurrent compaction winning is expected, not an
                    // incident. Anything else is worth waking someone for.
                    tracing::warn!(namespace = %r.ns.prefix, "compaction skipped: {e:#}");
                }
            }
        }
    });
}

// ---------------------------------------------------------------- router

pub fn router(state: AppState) -> Router {
    let api = Router::new()
        .route("/v1/namespaces", get(list_namespaces))
        .route("/v1/namespaces/{ns}", get(metadata).delete(destroy))
        .route("/v1/namespaces/{ns}/write", post(write))
        .route("/v1/namespaces/{ns}/query", post(query))
        .route("/v1/namespaces/{ns}/compact", post(compact))
        .route("/v1/namespaces/{ns}/warm", post(warm))
        .route("/v1/namespaces/{ns}/gc", post(gc))
        .route("/v1/namespaces/{ns}/branch/{dest}", post(branch))
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate));

    Router::new()
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics))
        .merge(api)
        // A 512 MB cap mirrors turbopuffer's upsert batch limit and keeps one
        // request from exhausting memory.
        .layer(DefaultBodyLimit::max(512 << 20))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn serve(state: AppState, addr: &str) -> Result<()> {
    spawn_compactor(state.clone());
    let listener = tokio::net::TcpListener::bind(addr).await?;
    if state.auth.disabled() {
        tracing::warn!(
            "no FCKDB_TOKEN or FCKDB_TOKENS set: authentication is DISABLED and every \
             request is treated as org 'default'. Do not expose this port."
        );
    }
    tracing::info!(
        "listening on {} (auth {})",
        listener.local_addr()?,
        if state.auth.disabled() { "DISABLED" } else { "enabled" }
    );
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            // Nothing to flush. A write is durable the moment its commit
            // returns, so there is no in-memory state whose loss could lose an
            // acknowledged write. Requests in flight simply fail and are retried.
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_validation_blocks_path_traversal() {
        for good in ["docs", "a", "tenant-1", "A_b-9", &"x".repeat(64)] {
            assert!(valid_namespace(good), "rejected valid name {good:?}");
        }
        for bad in [
            "", "..", "../other", "a/b", "a b", "a.b", "/abs", "ns\0", "ünïcode",
            &"x".repeat(65), "a%2fb", "a\\b",
        ] {
            assert!(!valid_namespace(bad), "accepted dangerous name {bad:?}");
        }
    }

    #[test]
    fn constant_time_compare_is_correct() {
        assert!(ct_eq(b"secret", b"secret"));
        assert!(!ct_eq(b"secret", b"secreT"));
        assert!(!ct_eq(b"secret", b"secre"));
        assert!(!ct_eq(b"", b"x"));
        assert!(ct_eq(b"", b""));
    }

    fn headers(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(header::AUTHORIZATION, value.parse().unwrap());
        h
    }

    #[test]
    fn auth_maps_tokens_to_orgs_and_rejects_everything_else() {
        let auth = Auth::from_pairs(vec![
            ("t1".into(), "acme".into()),
            ("t2".into(), "globex".into()),
        ]);
        assert_eq!(auth.resolve(&headers("Bearer t1")).unwrap().0, "acme");
        assert_eq!(auth.resolve(&headers("Bearer t2")).unwrap().0, "globex");
        assert!(auth.resolve(&headers("Bearer t3")).is_none(), "unknown token accepted");
        assert!(auth.resolve(&headers("t1")).is_none(), "missing Bearer prefix accepted");
        assert!(auth.resolve(&headers("Basic t1")).is_none(), "wrong scheme accepted");
        assert!(auth.resolve(&HeaderMap::new()).is_none(), "missing header accepted");
        // A token that is a prefix of a real one must not pass.
        assert!(auth.resolve(&headers("Bearer t")).is_none());
    }

    #[test]
    fn disabled_auth_admits_everyone_as_default() {
        let auth = Auth::default();
        assert!(auth.disabled());
        assert_eq!(auth.resolve(&HeaderMap::new()).unwrap().0, "default");
    }

    #[test]
    fn namespace_keys_are_scoped_per_org() {
        assert_eq!(AppState::key("acme", "docs"), "ns/acme/docs");
        assert_ne!(AppState::key("acme", "docs"), AppState::key("globex", "docs"));
    }

    #[tokio::test]
    async fn registry_creates_exactly_one_committer_per_namespace() {
        let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let state = AppState::new(store, None);

        let a = state.resident("acme", "docs");
        let b = state.resident("acme", "docs");
        assert!(Arc::ptr_eq(&a, &b), "a second committer was created for one namespace");

        let other = state.resident("globex", "docs");
        assert!(!Arc::ptr_eq(&a, &other), "orgs must not share a namespace");
        assert_eq!(state.all_resident().len(), 2);

        state.forget("acme", "docs");
        assert_eq!(state.all_resident().len(), 1);
    }

    // ---------------------------------------------------------- http surface

    use axum::body::{Body, to_bytes};
    use axum::http::Request as HttpRequest;
    use serde_json::{Value, json};
    use object_store::ObjectStoreExt;
    use tower::ServiceExt;

    const TOKEN_A: &str = "token-acme";
    const TOKEN_B: &str = "token-globex";

    fn test_state() -> AppState {
        let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        AppState::new(store, None).with_auth(Auth::from_pairs(vec![
            (TOKEN_A.into(), "acme".into()),
            (TOKEN_B.into(), "globex".into()),
        ]))
    }

    async fn call(
        state: &AppState,
        method: &str,
        uri: &str,
        token: Option<&str>,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let mut req = HttpRequest::builder().method(method).uri(uri);
        if let Some(t) = token {
            req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        let req = match body {
            Some(v) => req
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&v).unwrap()))
                .unwrap(),
            None => req.body(Body::empty()).unwrap(),
        };
        let res = router(state.clone()).oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
        let value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| json!({ "raw": String::from_utf8_lossy(&bytes) }));
        (status, value)
    }

    fn doc(id: u64, v: Vec<f32>) -> Value {
        json!({ "id": id, "vector": v, "attrs": { "tenant": "a" } })
    }

    #[tokio::test]
    async fn unauthenticated_and_wrong_tokens_are_refused() {
        let state = test_state();
        let body = json!({ "upsert": [doc(1, vec![1.0, 0.0])] });

        for token in [None, Some("wrong"), Some(""), Some("token-acm")] {
            let (status, _) =
                call(&state, "POST", "/v1/namespaces/docs/write", token, Some(body.clone())).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED, "token {token:?} was accepted");
        }

        let (status, _) =
            call(&state, "POST", "/v1/namespaces/docs/write", Some(TOKEN_A), Some(body)).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn health_and_metrics_need_no_token() {
        let state = test_state();
        let (status, _) = call(&state, "GET", "/healthz", None, None).await;
        assert_eq!(status, StatusCode::OK);
        let (status, body) = call(&state, "GET", "/metrics", None, None).await;
        assert_eq!(status, StatusCode::OK);
        let text = body["raw"].as_str().unwrap_or_default();
        assert!(text.contains("fckdb_class_a_per_write"), "metrics missing: {text}");
    }

    #[tokio::test]
    async fn write_then_query_round_trip() {
        let state = test_state();
        let body = json!({
            "upsert": [
                doc(1, vec![1.0, 0.0]),
                doc(2, vec![0.9, 0.1]),
                doc(3, vec![0.0, 1.0]),
            ]
        });
        let (status, res) =
            call(&state, "POST", "/v1/namespaces/docs/write", Some(TOKEN_A), Some(body)).await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["records"], 3);

        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/docs/query",
            Some(TOKEN_A),
            Some(json!({ "vector": [1.0, 0.0], "top_k": 2 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["hits"][0]["id"], 1);
        assert_eq!(res["hits"].as_array().unwrap().len(), 2);
        assert_eq!(res["consistent"], true);
        assert_eq!(res["indexed"], false, "nothing has been compacted yet");

        // Compaction makes it indexed, and the answer must not change.
        let (status, res) =
            call(&state, "POST", "/v1/namespaces/docs/compact", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["docs_out"], 3);

        let (_, res) = call(
            &state,
            "POST",
            "/v1/namespaces/docs/query",
            Some(TOKEN_A),
            Some(json!({ "vector": [1.0, 0.0], "top_k": 2 })),
        )
        .await;
        assert_eq!(res["indexed"], true);
        assert_eq!(res["hits"][0]["id"], 1);
    }

    #[tokio::test]
    async fn typed_attributes_and_tuple_filters_over_http() {
        let state = test_state();
        let body = json!({ "upsert": [
            { "id": 1, "vector": [1.0, 0.0],
              "attrs": { "lang": "id", "rank": 10, "public": true,
                         "when": "2024-03-05T00:00:00Z", "tags": ["a", "b"] } },
            { "id": 2, "vector": [0.9, 0.1],
              "attrs": { "lang": "en", "rank": 20, "public": false,
                         "when": "2024-01-01T00:00:00Z", "tags": ["b"] } },
            { "id": 3, "vector": [0.8, 0.2],
              "attrs": { "lang": "id", "rank": 30, "public": true,
                         "when": "2024-06-01T00:00:00Z", "tags": ["c"] } },
        ]});
        let (status, res) =
            call(&state, "POST", "/v1/namespaces/typed/write", Some(TOKEN_A), Some(body)).await;
        assert_eq!(status, StatusCode::OK, "{res}");
        call(&state, "POST", "/v1/namespaces/typed/compact", Some(TOKEN_A), None).await;

        // turbopuffer's tuple filter grammar, straight off the wire.
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/typed/query",
            Some(TOKEN_A),
            Some(json!({
                "vector": [1.0, 0.0],
                "top_k": 10,
                "nprobe": 32,
                "filter": ["And", [["lang", "Eq", "id"], ["rank", "Gte", 20]]],
                "include_attributes": ["lang", "rank"],
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        let hits = res["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 1, "typed filter matched the wrong set: {res}");
        assert_eq!(hits[0]["id"], 3);
        // Projection: exactly what was asked for, and typed as sent.
        assert_eq!(hits[0]["attrs"]["lang"], "id");
        assert_eq!(hits[0]["attrs"]["rank"], 30);
        assert_eq!(hits[0]["attrs"].as_object().unwrap().len(), 2);

        // Or / Not / Glob / Contains all reachable from JSON.
        let cases: Vec<(Value, Vec<u64>)> = vec![
            (json!(["Or", [["rank", "Eq", 10], ["rank", "Eq", 30]]]), vec![1, 3]),
            (json!(["Not", ["public", "Eq", true]]), vec![2]),
            (json!(["lang", "Glob", "i*"]), vec![1, 3]),
            (json!(["tags", "Contains", "b"]), vec![1, 2]),
            (json!(["rank", "In", [10, 20]]), vec![1, 2]),
        ];
        for (filter, expected) in cases {
            let (status, res) = call(
                &state,
                "POST",
                "/v1/namespaces/typed/query",
                Some(TOKEN_A),
                Some(json!({ "vector": [1.0, 0.0], "top_k": 10, "nprobe": 32, "filter": filter })),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "filter {filter} -> {res}");
            let mut got: Vec<u64> =
                res["hits"].as_array().unwrap().iter().map(|h| h["id"].as_u64().unwrap()).collect();
            got.sort();
            assert_eq!(got, expected, "filter {filter} selected the wrong documents");
        }

        // include_attributes: true returns everything.
        let (_, res) = call(
            &state,
            "POST",
            "/v1/namespaces/typed/query",
            Some(TOKEN_A),
            Some(json!({ "vector": [1.0, 0.0], "top_k": 1, "include_attributes": true })),
        )
        .await;
        assert_eq!(res["hits"][0]["attrs"].as_object().unwrap().len(), 5);
        // A datetime sent as a string comes back as a string, unchanged, because
        // nothing has declared it a datetime yet.
        assert_eq!(res["hits"][0]["attrs"]["when"], "2024-03-05T00:00:00Z");
    }

    #[tokio::test]
    async fn malformed_filters_are_rejected_not_ignored() {
        let state = test_state();
        call(
            &state,
            "POST",
            "/v1/namespaces/badf/write",
            Some(TOKEN_A),
            Some(json!({ "upsert": [doc(1, vec![1.0, 0.0])] })),
        )
        .await;
        for filter in [
            json!("nonsense"),
            json!([]),
            json!(["And", []]),
            json!(["Frobnicate", [["a", "Eq", 1]]]),
            json!(["a", "Frobnicate", 1]),
            json!([1, "Eq", 1]),
        ] {
            let (status, res) = call(
                &state,
                "POST",
                "/v1/namespaces/badf/query",
                Some(TOKEN_A),
                Some(json!({ "vector": [1.0, 0.0], "filter": filter })),
            )
            .await;
            // A filter that cannot be parsed must fail the request. Silently
            // dropping it would return unfiltered data to a caller who believes
            // it was filtered — the worst possible outcome for a tenancy filter.
            assert!(
                status.is_client_error(),
                "filter {filter} was accepted with {status}: {res}"
            );
        }
    }

    #[tokio::test]
    async fn organisations_cannot_see_each_other() {
        let state = test_state();
        let (status, _) = call(
            &state,
            "POST",
            "/v1/namespaces/shared/write",
            Some(TOKEN_A),
            Some(json!({ "upsert": [doc(42, vec![1.0, 0.0])] })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // Same namespace NAME, different org: must be a different database.
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/shared/query",
            Some(TOKEN_B),
            Some(json!({ "vector": [1.0, 0.0] })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert!(
            res["hits"].as_array().unwrap().is_empty(),
            "org B read org A's data: {res}"
        );

        let (status, _) = call(&state, "GET", "/v1/namespaces/shared", Some(TOKEN_B), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "org B saw org A's namespace");

        let (_, res) = call(&state, "GET", "/v1/namespaces", Some(TOKEN_A), None).await;
        assert_eq!(res["namespaces"], json!(["shared"]));
        let (_, res) = call(&state, "GET", "/v1/namespaces", Some(TOKEN_B), None).await;
        assert_eq!(res["namespaces"], json!([]));
    }

    #[tokio::test]
    async fn malformed_input_is_rejected_at_the_boundary() {
        let state = test_state();
        let cases: Vec<(&str, Value)> = vec![
            ("empty write", json!({})),
            ("zero dimensions", json!({ "upsert": [{ "id": 1, "vector": [] }] })),
            (
                "ragged batch",
                json!({ "upsert": [
                    { "id": 1, "vector": [1.0, 2.0] },
                    { "id": 2, "vector": [1.0] },
                ] }),
            ),
        ];
        for (label, body) in cases {
            let (status, res) =
                call(&state, "POST", "/v1/namespaces/docs/write", Some(TOKEN_A), Some(body)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{label} was accepted: {res}");
        }

        // Path traversal through the namespace name.
        for name in ["..", "a.b", "a%2Fb", ""] {
            let (status, _) = call(
                &state,
                "POST",
                &format!("/v1/namespaces/{name}/write"),
                Some(TOKEN_A),
                Some(json!({ "upsert": [doc(1, vec![1.0])] })),
            )
            .await;
            assert!(
                status == StatusCode::BAD_REQUEST || status == StatusCode::NOT_FOUND,
                "namespace {name:?} produced {status}"
            );
        }

        // Bad queries.
        let bad_queries = vec![
            ("empty vector", json!({ "vector": [] })),
            ("top_k zero", json!({ "vector": [1.0], "top_k": 0 })),
            ("top_k too large", json!({ "vector": [1.0], "top_k": 10_000 })),
        ];
        for (label, body) in bad_queries {
            let (status, res) =
                call(&state, "POST", "/v1/namespaces/docs/query", Some(TOKEN_A), Some(body)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{label} was accepted: {res}");
        }
    }

    #[tokio::test]
    async fn non_finite_vectors_are_rejected() {
        let state = test_state();
        // NaN and infinity are not representable in JSON numbers, so they arrive
        // as strings or nulls and serde rejects them first. The guard exists for
        // any caller that finds a way through.
        let (status, _) = call(
            &state,
            "POST",
            "/v1/namespaces/docs/write",
            Some(TOKEN_A),
            Some(json!({ "upsert": [{ "id": 1, "vector": [1.0, null] }] })),
        )
        .await;
        assert!(status.is_client_error(), "a null dimension was accepted");
    }

    #[tokio::test]
    async fn metadata_reports_the_namespace_and_404s_when_absent() {
        let state = test_state();
        let (status, _) = call(&state, "GET", "/v1/namespaces/ghost", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        call(
            &state,
            "POST",
            "/v1/namespaces/real/write",
            Some(TOKEN_A),
            Some(json!({ "upsert": [doc(1, vec![1.0, 0.0])] })),
        )
        .await;
        let (status, res) = call(&state, "GET", "/v1/namespaces/real", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(res["namespace"], "ns/acme/real");
        assert_eq!(res["unindexed_records"], 1);
        assert_eq!(res["write_backpressure"], false);
    }

    #[tokio::test]
    async fn backpressure_returns_429_with_retry_after() {
        let state = test_state();
        let handle = state.resident("acme", "loaded");

        // Craft a manifest whose recorded tail is over the write limit. Sizes live
        // in the manifest exactly so this check needs no data fetch.
        let m = crate::store::Manifest {
            next_seq: 1,
            wal: vec![crate::store::WalEntry {
                name: "fake.bin".into(),
                bytes: WRITE_BACKPRESSURE_BYTES + 1,
                records: 1,
            }],
            segments: vec![],
            index: None,
        };
        handle
            .ns
            .store
            .put(
                &object_store::path::Path::from(format!("{}/manifest", handle.ns.prefix)),
                object_store::PutPayload::from(serde_json::to_vec(&m).unwrap()),
            )
            .await
            .unwrap();

        let req = HttpRequest::builder()
            .method("POST")
            .uri("/v1/namespaces/loaded/write")
            .header(header::AUTHORIZATION, format!("Bearer {TOKEN_A}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({ "upsert": [doc(1, vec![1.0, 0.0])] })).unwrap(),
            ))
            .unwrap();
        let res = router(state.clone()).oneshot(req).await.unwrap();

        assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            res.headers().get(header::RETRY_AFTER).map(|v| v.to_str().unwrap()),
            Some("5"),
            "429 without Retry-After leaves the client guessing"
        );
        assert_eq!(handle.ns.metrics.backpressure_rejects.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn branch_and_delete_over_http() {
        let state = test_state();
        call(
            &state,
            "POST",
            "/v1/namespaces/src/write",
            Some(TOKEN_A),
            Some(json!({ "upsert": [doc(1, vec![1.0, 0.0]), doc(2, vec![0.0, 1.0])] })),
        )
        .await;
        call(&state, "POST", "/v1/namespaces/src/compact", Some(TOKEN_A), None).await;

        let (status, res) =
            call(&state, "POST", "/v1/namespaces/src/branch/copy", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert!(res["objects_copied"].as_u64().unwrap() > 0);

        let (_, res) = call(
            &state,
            "POST",
            "/v1/namespaces/copy/query",
            Some(TOKEN_A),
            Some(json!({ "vector": [1.0, 0.0], "top_k": 1 })),
        )
        .await;
        assert_eq!(res["hits"][0]["id"], 1, "the branch is not readable");

        // A branch destination is a name, never a path.
        let (status, _) =
            call(&state, "POST", "/v1/namespaces/src/branch/..", Some(TOKEN_A), None).await;
        assert!(status.is_client_error() || status == StatusCode::NOT_FOUND);

        // Branching onto a name that already exists is a conflict the caller can
        // act on, not a server fault.
        let (status, _) =
            call(&state, "POST", "/v1/namespaces/src/branch/copy", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::CONFLICT);

        let (status, res) = call(&state, "DELETE", "/v1/namespaces/copy", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert!(res["objects_deleted"].as_u64().unwrap() > 0);
        let (status, _) = call(&state, "GET", "/v1/namespaces/copy", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "deleted namespace still resolves");
    }

    #[tokio::test]
    async fn warm_and_gc_over_http() {
        let state = test_state();
        call(
            &state,
            "POST",
            "/v1/namespaces/ops/write",
            Some(TOKEN_A),
            Some(json!({ "upsert": [doc(1, vec![1.0, 0.0]), doc(2, vec![0.0, 1.0])] })),
        )
        .await;
        call(&state, "POST", "/v1/namespaces/ops/compact", Some(TOKEN_A), None).await;

        let (status, res) = call(&state, "POST", "/v1/namespaces/ops/warm", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert!(res["objects_warmed"].as_u64().unwrap() > 0);

        let (status, res) = call(&state, "POST", "/v1/namespaces/ops/gc", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::OK, "{res}");
        // The one-hour grace window means a just-compacted namespace loses nothing.
        assert_eq!(res["deleted"], 0);
        assert!(res["spared_recent"].as_u64().unwrap() > 0);
    }

    #[test]
    fn backpressure_triggers_below_the_query_cap() {
        assert!(
            WRITE_BACKPRESSURE_BYTES < MAX_UNINDEXED_SCAN_BYTES,
            "writes must be refused before consistent queries become impossible"
        );
        assert!(COMPACT_TRIGGER_BYTES < WRITE_BACKPRESSURE_BYTES, "compaction must fire first");
    }
}
