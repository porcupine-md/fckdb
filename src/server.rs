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
use crate::compat::{
    V2Metadata, V2MultiResponse, V2Query, V2QueryResponse, V2Write, V2WriteResponse,
};
use crate::store::{
    GroupCommit, MAX_DELETE_BY_FILTER_ROWS, MAX_PATCH_BY_FILTER_ROWS,
    MAX_UNINDEXED_SCAN_BYTES, Namespace, WriteConfig,
};
use crate::wire::{QueryRequest, WriteRequest, WriteResponse};
use crate::doc::Doc;
use crate::embed::{Embedder, HttpEmbedder};
use anyhow::Result;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, FromRequest, Path as AxPath, Request, State};
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
    /// One registry for the whole process. Counters have to outlive the
    /// namespaces they describe, or destroying one makes the totals fall and
    /// every rate built on them goes wrong.
    metrics: Arc<crate::store::Metrics>,
    /// Absent when no embedding endpoint is configured, in which case requests
    /// that need one are refused by name rather than served a zero vector.
    embedder: Option<Arc<dyn Embedder>>,
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
            metrics: Arc::new(crate::store::Metrics::default()),
            embedder: HttpEmbedder::from_env().unwrap_or_else(|e| {
                tracing::warn!("embedding endpoint is configured but unusable: {e:#}");
                None
            }),
            started: Instant::now(),
        }
    }

    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
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
        let mut ns = Namespace::new(self.store.clone(), key.clone())
            .with_metrics(self.metrics.clone());
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
        // 501 is a refusal, not an incident: the caller asked for something this
        // build does not do. Logging it at ERROR alongside real faults trains
        // operators to ignore the level that matters.
        match self.0 {
            StatusCode::NOT_IMPLEMENTED => tracing::debug!(status = %self.0, "{}", self.1),
            s if s.is_server_error() => tracing::error!(status = %s, "{}", self.1),
            _ => {}
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

/// `Json`, but a malformed body is a 400 with a JSON error body.
///
/// Axum's own rejection is a 422 with a plain-text body, which is a second error
/// shape for clients to handle and a status they are unlikely to be checking for.
/// One shape for every failure is worth fifteen lines.
pub struct AppJson<T>(pub T);

impl<S, T> FromRequest<S> for AppJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request(req: Request, state: &S) -> std::result::Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(req, state)
            .await
            .map_err(|e| AppError(StatusCode::BAD_REQUEST, e.body_text()))?;
        Ok(AppJson(value))
    }
}

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
    AppJson(body): AppJson<WriteRequest>,
) -> ApiResult<Json<WriteResponse>> {
    if !valid_namespace(&name) {
        return Err(bad("namespace must be 1-64 chars of [A-Za-z0-9_-]"));
    }
    if body.is_empty() {
        return Err(bad(
            "write requires at least one of: upsert, patch, delete, upsert_columns, \
             patch_columns, delete_by_filter, patch_by_filter",
        ));
    }


    let handle = state.resident(&org, &name);

    // Backpressure. Consulted from the manifest, which records sizes inline, so
    // this costs one small GET and never a LIST or a data fetch.
    let (manifest, _) = handle.ns.load().await?;
    if manifest.unindexed_bytes() >= WRITE_BACKPRESSURE_BYTES {
        return Err(backpressure(&handle, manifest.unindexed_bytes()));
    }

    let t = Instant::now();
    let config = WriteConfig { metric: body.distance_metric, ..Default::default() };
    let plan = assemble(&handle.ns, body).await?;
    let count = plan.records.len();
    let seq = handle
        .commit
        .write_all(plan.records, config)
        .await
        // A schema conflict is the caller's mistake, not a server fault: they
        // sent a type that disagrees with what the namespace already holds.
        .map_err(|e| {
            let msg = format!("{e:#}");
            if msg.contains("declared")
                || msg.contains("dimensions")
                || msg.contains("distance_metric")
                || msg.contains("cannot interpret")
            {
                AppError(StatusCode::BAD_REQUEST, msg)
            } else {
                AppError(StatusCode::INTERNAL_SERVER_ERROR, msg)
            }
        })?;

    Ok(Json(WriteResponse {
        seq,
        records: count,
        rows_upserted: plan.upserted,
        rows_patched: plan.patched,
        rows_deleted: plan.deleted,
        rows_remaining: plan.remaining,
        took_ms: t.elapsed().as_millis() as u64,
    }))
}

/// Refuse a write and kick compaction, so the caller gets an actionable error
/// rather than a write that silently stops being visible.
fn backpressure(handle: &Resident, unindexed: u64) -> AppError {
    handle.ns.metrics.backpressure_rejects.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let ns = handle.ns.clone();
    tokio::spawn(async move {
        if let Err(e) = ns.compact(true).await {
            tracing::error!("backpressure compaction failed: {e:#}");
        }
    });
    AppError(
        StatusCode::TOO_MANY_REQUESTS,
        format!(
            "unindexed tail is {unindexed} bytes, at or over the {WRITE_BACKPRESSURE_BYTES} \
             byte write limit; compaction has been triggered, retry shortly"
        ),
    )
}

/// Default clusters probed when a compatibility query does not say.
const DEFAULT_NPROBE: usize = 8;

/// What a write request expands to.
struct Plan {
    records: Vec<Record>,
    upserted: usize,
    patched: usize,
    deleted: usize,
    remaining: bool,
}

/// Expand every write form into records.
///
/// Order matters and follows turbopuffer's: `delete_by_filter` first, then
/// `patch_by_filter`, then the explicit row and column operations. A patch that
/// arrives in the same request as the filter-delete that would have removed its
/// target must lose, or the outcome depends on map iteration order.
async fn assemble(ns: &Namespace, body: WriteRequest) -> ApiResult<Plan> {
    let mut records = Vec::new();
    let (mut upserted, mut patched, mut deleted) = (0usize, 0usize, 0usize);
    let mut remaining = false;

    if let Some(filter) = &body.delete_by_filter {
        let (ids, more) = ns
            .ids_matching(filter, MAX_DELETE_BY_FILTER_ROWS)
            .await
            .map_err(AppError::from)?;
        remaining |= more;
        deleted += ids.len();
        records.extend(ids.into_iter().map(Record::Delete));
    }

    if let Some(fp) = &body.patch_by_filter {
        if fp.patch.is_empty() {
            return Err(bad("patch_by_filter requires a non-empty patch"));
        }
        let (ids, more) =
            ns.ids_matching(&fp.filters, MAX_PATCH_BY_FILTER_ROWS).await.map_err(AppError::from)?;
        remaining |= more;
        patched += ids.len();
        records.extend(
            ids.into_iter().map(|id| Record::Patch { id, attrs: fp.patch.clone() }),
        );
    }

    if let Some(cols) = &body.upsert_columns {
        for row in cols.transpose().map_err(|e| bad(format!("upsert_columns: {e}")))? {
            let Some(vector) = row.vector else {
                return Err(bad(format!("upsert_columns: document {} has no vector", row.id)));
            };
            upserted += 1;
            records.push(Record::Upsert(Doc { id: row.id, vector, attrs: row.attrs }));
        }
    }

    if let Some(cols) = &body.patch_columns {
        if cols.0.contains_key("vector") {
            // Same restriction turbopuffer documents: a patch cannot touch a
            // vector, because the index would have to be updated in place.
            return Err(bad("vectors cannot be patched; upsert the whole document"));
        }
        for row in cols.transpose().map_err(|e| bad(format!("patch_columns: {e}")))? {
            patched += 1;
            records.push(Record::Patch { id: row.id, attrs: row.attrs });
        }
    }

    for doc in body.upsert {
        if doc.vector.is_empty() {
            return Err(bad(format!("document {} has no vector", doc.id)));
        }
        upserted += 1;
        records.push(Record::Upsert(doc));
    }
    for p in body.patch {
        patched += 1;
        records.push(Record::Patch { id: p.id, attrs: p.attrs });
    }
    for id in body.delete {
        deleted += 1;
        records.push(Record::Delete(id));
    }

    if records.is_empty() {
        return Err(bad("write matched no documents and contained no rows"));
    }
    Ok(Plan { records, upserted, patched, deleted, remaining })
}

async fn query(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
    AppJson(req): AppJson<QueryRequest>,
) -> ApiResult<Response> {
    if !valid_namespace(&name) {
        return Err(bad("namespace must be 1-64 chars of [A-Za-z0-9_-]"));
    }
    if req.order_by.is_none()
        && req.text.is_none()
        && req.sparse.is_none()
        && req.aggregate_by.is_none()
        && req.vector.is_empty()
    {
        return Err(bad("query requires a vector, order_by, text, sparse, or aggregate_by"));
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

// ---------------------------------------------------------------- /v2 compat

/// Map a request-translation failure to a status.
///
/// Every error from `into_native` describes something wrong with the request, so
/// none of them can be a 500. Keyword-matching those against a list of phrases —
/// as `classify` must do for engine errors — would silently turn each new
/// validation message into an internal server error until someone noticed.
fn classify_request(e: anyhow::Error) -> AppError {
    let msg = format!("{e:#}");
    if msg.contains("not implemented") {
        AppError(StatusCode::NOT_IMPLEMENTED, msg)
    } else {
        AppError(StatusCode::BAD_REQUEST, msg)
    }
}

/// Map an engine error to the right status. Shared by both surfaces so a client
/// mistake never reads as a server fault on one and not the other.
fn classify(e: anyhow::Error) -> AppError {
    let msg = format!("{e:#}");
    // ponytail: matching on message text. Fragile in exactly the way it looks —
    // a new engine error defaults to 500 until someone adds its phrase here, and
    // this list has already had to grow twice. The fix is a typed error enum in
    // the engine that carries its own kind; worth doing when the list next grows.
    let client = [
        "declared",
        "needs a numeric",
        "not enabled for full-text",
        "is not a sparse vector",
        "num_shards",
        "has not been compacted",
        "dimensions",
        "distance_metric",
        "cannot interpret",
        "cannot be changed",
        "is not a",
        "no vector",
        "not implemented",
        "disagree",
        "must be",
        "requires",
    ];
    if msg.contains("scan cap") {
        AppError(StatusCode::SERVICE_UNAVAILABLE, msg)
    } else if msg.contains("not implemented") {
        AppError(StatusCode::NOT_IMPLEMENTED, msg)
    } else if client.iter().any(|c| msg.contains(c)) {
        AppError(StatusCode::BAD_REQUEST, msg)
    } else {
        AppError(StatusCode::INTERNAL_SERVER_ERROR, msg)
    }
}

async fn v2_write(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
    AppJson(body): AppJson<V2Write>,
) -> ApiResult<Response> {
    if !valid_namespace(&name) {
        return Err(bad("namespace must be 1-64 chars of [A-Za-z0-9_-]"));
    }
    if body.is_empty() {
        return Err(bad("write requires at least one row, column, or filter operation"));
    }

    let handle = state.resident(&org, &name);
    let (manifest, _) = handle.ns.load().await?;
    if manifest.unindexed_bytes() >= WRITE_BACKPRESSURE_BYTES {
        return Err(backpressure(&handle, manifest.unindexed_bytes()));
    }

    let config = WriteConfig {
        metric: body.distance_metric,
        declared_types: body.declared_types(),
        declared_fts: body.declared_fts(),
        num_shards: body.sharding.map(|s| s.num_shards),
        declared_embed: body.declared_embed(),
    };

    // Auto-embedding: rows without a vector get one from the attribute the
    // schema names. One batched call, because embedding endpoints charge and
    // rate-limit per request.
    let mut body = body;
    let embed_spec = if config.declared_embed.is_empty() {
        manifest.schema.embed.clone()
    } else {
        config.declared_embed.clone()
    };
    let pending = body.rows_needing_embedding(&embed_spec);
    if !pending.is_empty() {
        let model = embed_spec.values().next().and_then(|s| s.model.clone());
        let texts: Vec<String> = pending.iter().map(|(_, t)| t.clone()).collect();
        let vectors = crate::embed::embed_many(state.embedder.as_ref(), &texts, model.as_deref())
            .await
            .map_err(classify_embed)?;
        body.attach_vectors(
            pending.iter().map(|(i, _)| *i).zip(vectors).collect(),
        );
    }
    // Which counters to report is decided by what was asked for, before
    // translation loses that information.
    let asked_upsert = !body.upsert_rows.is_empty() || body.upsert_columns.is_some();
    let asked_patch = !body.patch_rows.is_empty()
        || body.patch_columns.is_some()
        || body.patch_by_filter.is_some();
    let asked_delete = !body.deletes.is_empty() || body.delete_by_filter.is_some();
    let asked_filter = body.delete_by_filter.is_some() || body.patch_by_filter.is_some();

    let native = body.into_native().map_err(classify_request)?;
    let plan = assemble(&handle.ns, native).await?;
    let affected = plan.upserted + plan.patched + plan.deleted;

    handle.commit.write_all(plan.records, config).await.map_err(classify)?;

    Ok(Json(V2WriteResponse {
        rows_affected: affected,
        rows_upserted: asked_upsert.then_some(plan.upserted),
        rows_patched: asked_patch.then_some(plan.patched),
        rows_deleted: asked_delete.then_some(plan.deleted),
        rows_remaining: asked_filter.then_some(plan.remaining),
    })
    .into_response())
}

async fn v2_query(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
    AppJson(body): AppJson<V2Query>,
) -> ApiResult<Response> {
    if !valid_namespace(&name) {
        return Err(bad("namespace must be 1-64 chars of [A-Za-z0-9_-]"));
    }

    // A multi-query runs each sub-query against the same namespace, then either
    // reports them separately or fuses them into one list.
    let consistency = body.consistency;
    if let Some((subqueries, rerank, limit)) =
        body.clone().split_multi().map_err(classify_request)?
    {
        // Every sub-query that embeds is resolved in ONE call to the endpoint.
        // A hybrid query typically embeds the same text it also searches for
        // lexically, and paying per sub-query would be gratuitous.
        let mut subqueries = subqueries;
        let pending: Vec<(usize, String, Option<String>)> = subqueries
            .iter()
            .enumerate()
            .filter_map(|(i, q)| q.pending_embed().map(|(t, m)| (i, t, m)))
            .collect();
        if !pending.is_empty() {
            let model = pending[0].2.clone();
            if pending.iter().any(|(_, _, m)| *m != model) {
                return Err(bad("sub-queries must request the same embedding model"));
            }
            let texts: Vec<String> = pending.iter().map(|(_, t, _)| t.clone()).collect();
            let vectors =
                crate::embed::embed_many(state.embedder.as_ref(), &texts, model.as_deref())
                    .await
                    .map_err(classify_embed)?;
            for ((index, _, _), vector) in pending.into_iter().zip(vectors) {
                subqueries[index].resolve_embed(vector);
            }
        }
        let handle = state.resident(&org, &name);
        let (manifest, _) = handle.ns.load().await?;
        let metric = manifest.schema.distance_metric;

        let mut results = Vec::with_capacity(subqueries.len());
        for mut sub in subqueries {
            // Consistency is a property of the request, not of each sub-query.
            sub.consistency = consistency;
            let one = run_v2_query(&handle, sub, metric).await?;
            results.push(one);
        }

        return match rerank {
            None => Ok(Json(V2MultiResponse { results }).into_response()),
            Some(crate::compat::RerankBy::Rrf { k }) => {
                let lists: Vec<Vec<crate::compat::V2Row>> =
                    results.into_iter().filter_map(|r| r.rows).collect();
                if lists.is_empty() {
                    return Err(bad("rerank_by needs sub-queries that return rows"));
                }
                let rows = crate::compat::rrf(lists, k, limit);
                Ok(Json(V2QueryResponse { rows: Some(rows), ..Default::default() })
                    .into_response())
            }
        };
    }

    let mut body = body;
    if let Some((text, model)) = body.pending_embed() {
        let vector = crate::embed::embed_one(state.embedder.as_ref(), &text, model.as_deref())
            .await
            .map_err(classify_embed)?;
        body.resolve_embed(vector);
    }

    let handle = state.resident(&org, &name);
    let (manifest, _) = handle.ns.load().await?;
    let metric = manifest.schema.distance_metric;
    let response = run_v2_query(&handle, body, metric).await?;
    Ok(Json(response).into_response())
}

/// An embedding failure is somebody else's service failing, which is neither the
/// caller's fault nor this server's: 501 when unconfigured, 502 when the
/// endpoint itself refused.
fn classify_embed(e: anyhow::Error) -> AppError {
    let msg = format!("{e:#}");
    if msg.contains("not configured") {
        AppError(StatusCode::NOT_IMPLEMENTED, msg)
    } else {
        AppError(StatusCode::BAD_GATEWAY, msg)
    }
}

/// Execute one compatibility query and shape its response.
///
/// Shared by the single and multi-query paths so a sub-query behaves exactly
/// like the same query sent on its own.
async fn run_v2_query(
    handle: &Resident,
    body: V2Query,
    metric: crate::doc::DistanceMetric,
) -> ApiResult<V2QueryResponse> {
    let (req, exact) = body.into_native(DEFAULT_NPROBE).map_err(classify_request)?;
    let aggregating = req.aggregate_by.is_some();
    let grouped = !req.group_by.is_empty();
    let ranking = !req.vector.is_empty()
        || req.text.is_some()
        || req.sparse.is_some()
        || req.order_by.is_some();

    if !ranking && !aggregating {
        return Err(bad("rank_by vector must be non-empty and finite"));
    }
    if !req.vector.is_empty() && req.vector.iter().any(|f| !f.is_finite()) {
        return Err(bad("rank_by vector must be non-empty and finite"));
    }

    // A pure aggregation ranks nothing and reports no rows.
    if aggregating && !ranking {
        let res = handle.ns.query(&req).await.map_err(classify)?;
        return Ok(crate::compat::to_v2_aggregations(
            res.aggregations,
            res.aggregation_groups,
            grouped,
        ));
    }

    let (hits, aggregated) = if exact {
        // kNN means exact, so bypass the index entirely rather than probing more.
        let hits = handle
            .ns
            .query_brute(&req.vector, req.top_k, req.filter.as_ref())
            .await
            .map_err(classify)?;
        (hits, None)
    } else {
        let res = handle.ns.query(&req).await.map_err(classify)?;
        let agg = aggregating.then(|| (res.aggregations, res.aggregation_groups));
        (res.hits, agg)
    };

    let rows = if req.order_by.is_some() {
        // Ordering by attribute reports no distance at all.
        crate::compat::to_v2_ordered_rows(hits)
    } else if req.text.is_some() || req.sparse.is_some() {
        // BM25 and sparse similarity both report a score in $dist where HIGHER
        // is better — unlike a vector distance. Passing either through the
        // vector conversion would invert the ranking.
        crate::compat::to_v2_score_rows(hits)
    } else {
        crate::compat::to_v2_rows(hits, metric)
    };

    // Ranking and aggregating in one request yields both, which is how a faceted
    // result page is fetched in a single round trip.
    let mut response = V2QueryResponse { rows: Some(rows), ..Default::default() };
    if let Some((aggregations, groups)) = aggregated {
        let shaped = crate::compat::to_v2_aggregations(aggregations, groups, grouped);
        response.aggregations = shaped.aggregations;
        response.aggregation_groups = shaped.aggregation_groups;
    }
    Ok(response)
}

async fn v2_metadata(
    State(state): State<AppState>,
    Extension(Org(org)): Extension<Org>,
    AxPath(name): AxPath<String>,
) -> ApiResult<Json<V2Metadata>> {
    if !valid_namespace(&name) {
        return Err(bad("invalid namespace"));
    }
    let handle = state.resident(&org, &name);
    let (m, version) = handle.ns.load().await?;
    if version.is_none() {
        return Err(AppError(StatusCode::NOT_FOUND, format!("namespace {name} does not exist")));
    }
    Ok(Json(crate::compat::to_v2_metadata(&name, &m)))
}

async fn healthz() -> &'static str {
    "ok\n"
}

/// Prometheus scrape. Aggregates every resident namespace, and reports the one
/// number that tells you whether the commit protocol is healthy.
async fn metrics(State(state): State<AppState>) -> Response {
    // Deliberately touches nothing but memory.
    //
    // This used to call metadata() per resident namespace, one object storage
    // GET each, sequentially. Measured at 140ms per namespace: 4.2s at thirty,
    // past Prometheus' ten-second default timeout at about seventy. Monitoring
    // that slows down as the system grows stops working exactly when it is
    // needed, and it fails silently — the server answers, just too late.
    let residents = state.all_resident();
    let bytes: u64 = residents.iter().map(|r| r.ns.known_bytes()).sum();
    let agg = state.metrics.snapshot();
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
            // Deciding whether each namespace is due costs one GET. Checked
            // concurrently, because sequentially it takes longer than the sweep
            // interval itself once there are more than a handful, and the loop
            // never rests.
            let residents = state.all_resident();
            let due = futures::future::join_all(residents.iter().map(|r| async move {
                let (m, version) = r.ns.load().await.ok()?;
                version?;
                let due = m.unindexed_bytes() >= COMPACT_TRIGGER_BYTES
                    || m.wal.len() >= COMPACT_TRIGGER_ENTRIES;
                due.then(|| (m.wal.len(), m.unindexed_bytes()))
            }))
            .await;

            for (r, stats) in residents.iter().zip(due) {
                let Some((wal_entries, unindexed_bytes)) = stats else { continue };
                tracing::info!(
                    namespace = %r.ns.prefix, wal_entries, unindexed_bytes, "compacting"
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
        // turbopuffer-compatible surface.
        .route("/v1/namespaces/{ns}/metadata", get(v2_metadata))
        .route("/v2/namespaces/{ns}", post(v2_write))
        .route("/v2/namespaces/{ns}/query", post(v2_query))
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
    async fn schema_conflicts_are_client_errors_over_http() {
        let state = test_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/sch/write",
            Some(TOKEN_A),
            Some(json!({ "upsert": [
                { "id": 1, "vector": [1.0, 0.0], "attrs": { "count": 5, "name": "a" } }
            ]})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");

        // The namespace now declares count:uint. A string is a client error.
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/sch/write",
            Some(TOKEN_A),
            Some(json!({ "upsert": [
                { "id": 2, "vector": [1.0, 0.0], "attrs": { "count": "five" } }
            ]})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "type conflict accepted: {res}");

        // So is a mismatched dimension.
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/sch/write",
            Some(TOKEN_A),
            Some(json!({ "upsert": [{ "id": 3, "vector": [1.0, 0.0, 0.0] }] })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "dimension conflict accepted: {res}");

        // Metadata reports the inferred schema.
        let (_, md) = call(&state, "GET", "/v1/namespaces/sch", Some(TOKEN_A), None).await;
        assert_eq!(md["schema"]["count"], "uint");
        assert_eq!(md["schema"]["name"], "string");
        assert_eq!(md["id_type"], "uint");
        assert_eq!(md["distance_metric"], "cosine_distance");
        assert_eq!(md["dim"], 2);
    }

    #[tokio::test]
    async fn string_ids_patches_and_metric_over_http() {
        let state = test_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/mixed/write",
            Some(TOKEN_A),
            Some(json!({
                "distance_metric": "euclidean_squared",
                "upsert": [
                    { "id": "doc-a", "vector": [1.0, 0.0], "attrs": { "n": 1 } },
                    { "id": "doc-b", "vector": [0.0, 1.0], "attrs": { "n": 2 } },
                ]
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");

        let (_, md) = call(&state, "GET", "/v1/namespaces/mixed", Some(TOKEN_A), None).await;
        assert_eq!(md["id_type"], "string");
        assert_eq!(md["distance_metric"], "euclidean_squared");

        // Patch merges without disturbing the vector.
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/mixed/write",
            Some(TOKEN_A),
            Some(json!({ "patch": [{ "id": "doc-a", "n": 99 }] })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");

        let (_, res) = call(
            &state,
            "POST",
            "/v1/namespaces/mixed/query",
            Some(TOKEN_A),
            Some(json!({ "vector": [1.0, 0.0], "top_k": 1, "include_attributes": true })),
        )
        .await;
        assert_eq!(res["hits"][0]["id"], "doc-a", "string id did not round-trip");
        assert_eq!(res["hits"][0]["attrs"]["n"], 99, "patch did not apply");

        // Changing the metric afterwards is refused.
        let (status, _) = call(
            &state,
            "POST",
            "/v1/namespaces/mixed/write",
            Some(TOKEN_A),
            Some(json!({
                "distance_metric": "cosine_distance",
                "upsert": [{ "id": "doc-c", "vector": [1.0, 1.0] }]
            })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "metric was changed in place");
    }

    // ---------------------------------------------------------- /v2 surface

    #[tokio::test]
    async fn v2_write_and_query_round_trip() {
        let state = test_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tp",
            Some(TOKEN_A),
            Some(json!({
                "upsert_rows": [
                    { "id": 1, "vector": [1.0, 0.0], "name": "one", "rank": 10 },
                    { "id": 2, "vector": [0.0, 1.0], "name": "two", "rank": 20 },
                ],
                "distance_metric": "cosine_distance"
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows_affected"], 2);
        assert_eq!(res["rows_upserted"], 2);
        // Counters for operations that were not requested are absent, not zero.
        assert!(res.get("rows_deleted").is_none(), "reported a delete count for no deletes");
        assert!(res.get("rows_remaining").is_none());

        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tp/query",
            Some(TOKEN_A),
            Some(json!({
                "rank_by": ["vector", "ANN", [1.0, 0.0]],
                "top_k": 2,
                "include_attributes": ["name"],
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        let rows = res["rows"].as_array().unwrap();
        assert_eq!(rows[0]["id"], 1);
        // Attributes flattened next to id and $dist.
        assert_eq!(rows[0]["name"], "one");
        assert!(rows[0].get("attrs").is_none(), "attributes were nested");
        assert!(rows[0].get("rank").is_none(), "returned an unrequested attribute");

        // $dist is a distance: the best match has the SMALLEST value, and an
        // exact match is ~0 under cosine.
        let d0 = rows[0]["$dist"].as_f64().unwrap();
        let d1 = rows[1]["$dist"].as_f64().unwrap();
        assert!(d0 < d1, "$dist did not order ascending: {d0} then {d1}");
        assert!(d0.abs() < 1e-5, "an exact cosine match should have $dist ~0, got {d0}");
    }

    #[tokio::test]
    async fn v2_filters_limit_and_consistency() {
        let state = test_state();
        call(
            &state,
            "POST",
            "/v2/namespaces/tpf",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [
                { "id": 1, "vector": [1.0, 0.0], "lang": "id", "rank": 10 },
                { "id": 2, "vector": [0.9, 0.1], "lang": "en", "rank": 20 },
                { "id": 3, "vector": [0.8, 0.2], "lang": "id", "rank": 30 },
            ]})),
        )
        .await;

        // `filters` (plural), `limit.total`, and object-shaped consistency.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpf/query",
            Some(TOKEN_A),
            Some(json!({
                "rank_by": ["vector", "ANN", [1.0, 0.0]],
                "limit": { "total": 5 },
                "filters": ["And", [["lang", "Eq", "id"], ["rank", "Gte", 20]]],
                "consistency": { "level": "eventual" },
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        let rows = res["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["id"], 3);

        // kNN must produce the same ranking as ANN on a namespace this small.
        let (status, knn) = call(
            &state,
            "POST",
            "/v2/namespaces/tpf/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["vector", "kNN", [1.0, 0.0]], "top_k": 3 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{knn}");
        let knn_ids: Vec<u64> =
            knn["rows"].as_array().unwrap().iter().map(|r| r["id"].as_u64().unwrap()).collect();
        assert_eq!(knn_ids, vec![1, 2, 3], "kNN ranking disagreed with the vectors");
    }

    #[tokio::test]
    async fn v2_reports_unimplemented_features_as_501() {
        let state = test_state();
        call(
            &state,
            "POST",
            "/v2/namespaces/tpu",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [{ "id": 1, "vector": [1.0, 0.0] }] })),
        )
        .await;

        for body in [
            json!({ "rank_by": ["vector", "ANN", [1.0, 0.0]], "vector_encoding": "base64" }),
        ] {
            let (status, res) =
                call(&state, "POST", "/v2/namespaces/tpu/query", Some(TOKEN_A), Some(body.clone()))
                    .await;
            // Not implemented is honest; silently ignoring the field and
            // returning plain vector results would be a wrong answer.
            assert_eq!(
                status,
                StatusCode::NOT_IMPLEMENTED,
                "{body} returned {status}: {res}"
            );
        }
    }

    #[tokio::test]
    async fn v2_schema_declaration_types_the_untypeable() {
        let state = test_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tps",
            Some(TOKEN_A),
            Some(json!({
                "upsert_rows": [
                    { "id": 1, "vector": [1.0, 0.0],
                      "when": "2024-03-05T00:00:00Z",
                      "who": "550e8400-e29b-41d4-a716-446655440000" },
                    { "id": 2, "vector": [0.9, 0.1],
                      "when": "2023-01-01T00:00:00Z",
                      "who": "550e8400-e29b-41d4-a716-446655440001" },
                ],
                "schema": {
                    "when": { "type": "datetime" },
                    "who": { "type": "uuid", "filterable": true },
                }
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");

        // The declared datetime makes a range filter work; without it these would
        // be strings and Gte would compare lexicographically.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tps/query",
            Some(TOKEN_A),
            Some(json!({
                "rank_by": ["vector", "ANN", [1.0, 0.0]],
                "top_k": 10,
                "filters": ["when", "Gte", "2024-01-01T00:00:00Z"],
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        let rows = res["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 1, "datetime range filter selected wrongly: {res}");
        assert_eq!(rows[0]["id"], 1);

        // The metadata endpoint reports the declared schema in their shape.
        let (status, md) =
            call(&state, "GET", "/v1/namespaces/tps/metadata", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::OK, "{md}");
        assert_eq!(md["id"], "tps");
        assert_eq!(md["schema"]["when"]["type"], "datetime");
        assert_eq!(md["schema"]["who"]["type"], "uuid");
        assert_eq!(md["approx_row_count"], 2);
        assert_eq!(md["distance_metric"], "cosine_distance");
        assert_eq!(md["index"]["unindexed_bytes"].as_u64().unwrap() > 0, true);
        assert!(md["last_write_at"].as_str().is_some());
        assert_eq!(md["encryption"]["mode"], "default");

        // Changing a declared type is refused.
        let (status, _) = call(
            &state,
            "POST",
            "/v2/namespaces/tps",
            Some(TOKEN_A),
            Some(json!({
                "upsert_rows": [{ "id": 3, "vector": [1.0, 0.0] }],
                "schema": { "when": { "type": "string" } }
            })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "a declared type was changed in place");
    }

    #[tokio::test]
    async fn v2_column_and_filter_writes() {
        let state = test_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpc",
            Some(TOKEN_A),
            Some(json!({ "upsert_columns": {
                "id": [1, 2, 3, 4],
                "vector": [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
                "tier": ["gold", "silver", "gold", "silver"],
            }})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows_upserted"], 4);

        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpc",
            Some(TOKEN_A),
            Some(json!({ "delete_by_filter": ["tier", "Eq", "silver"] })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows_deleted"], 2);
        assert_eq!(res["rows_affected"], 2);
        assert_eq!(res["rows_remaining"], false);

        let (_, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpc/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["vector", "ANN", [1.0, 0.0]], "top_k": 10 })),
        )
        .await;
        let ids: Vec<u64> =
            res["rows"].as_array().unwrap().iter().map(|r| r["id"].as_u64().unwrap()).collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[tokio::test]
    async fn v2_bm25_full_text_search() {
        let state = test_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpb",
            Some(TOKEN_A),
            Some(json!({
                "upsert_rows": [
                    { "id": 1, "vector": [1.0, 0.0],
                      "body": "the quick brown fox jumps over the lazy dog" },
                    { "id": 2, "vector": [0.0, 1.0], "body": "a quick brown dog" },
                    { "id": 3, "vector": [0.5, 0.5], "body": "unrelated notes about databases" },
                ],
                "schema": { "body": { "type": "string", "full_text_search": true } }
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");

        // BM25 needs the term index, which compaction builds.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpb/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["body", "BM25", "quick fox"], "top_k": 5 })),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "an unbuilt text index should be an actionable error, not a 500: {res}"
        );

        let (status, _) =
            call(&state, "POST", "/v1/namespaces/tpb/compact", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::OK, "compaction is a native operation");
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpb/query",
            Some(TOKEN_A),
            Some(json!({
                "rank_by": ["body", "BM25", "quick fox"],
                "top_k": 5,
                "include_attributes": ["body"],
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        let rows = res["rows"].as_array().unwrap();
        // Document 1 has both terms; 2 has only "quick"; 3 has neither and must
        // be absent rather than scored zero.
        assert_eq!(rows.len(), 2, "BM25 returned the wrong set: {res}");
        assert_eq!(rows[0]["id"], 1);
        assert_eq!(rows[1]["id"], 2);
        // For BM25, $dist is a relevance score: HIGHER is better, the opposite of
        // a vector distance. Passing it through the distance conversion would
        // invert the ranking.
        let d0 = rows[0]["$dist"].as_f64().unwrap();
        let d1 = rows[1]["$dist"].as_f64().unwrap();
        assert!(d0 > d1, "BM25 $dist was inverted: {d0} then {d1}");
        assert!(rows[0]["body"].as_str().unwrap().contains("fox"));

        // Stemming: "foxes" finds "fox".
        let (_, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpb/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["body", "BM25", "foxes jumping"], "top_k": 5 })),
        )
        .await;
        assert_eq!(res["rows"][0]["id"], 1, "stemming did not match foxes to fox");

        // A term nobody has returns nothing, not a page of zeroes.
        let (_, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpb/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["body", "BM25", "zebra"], "top_k": 5 })),
        )
        .await;
        assert!(res["rows"].as_array().unwrap().is_empty());

        // An attribute without full_text_search is refused by name.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpb/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["nope", "BM25", "quick"], "top_k": 5 })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{res}");
    }

    #[tokio::test]
    async fn v2_indonesian_full_text() {
        let state = test_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpid",
            Some(TOKEN_A),
            Some(json!({
                "upsert_rows": [
                    { "id": 1, "vector": [1.0, 0.0], "isi": "buku yang ada di rak itu" },
                    { "id": 2, "vector": [0.0, 1.0], "isi": "tulisan puisi untuk sekolah" },
                    { "id": 3, "vector": [0.5, 0.5], "isi": "makanan pedas dari warung" },
                ],
                "schema": { "isi": { "type": "string", "full_text_search": {
                    "tokenizer": { "language": "indonesian" }
                }}}
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        let (status, _) =
            call(&state, "POST", "/v1/namespaces/tpid/compact", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::OK);

        // The tokenizer choice round-trips through the schema.
        let (_, md) = call(&state, "GET", "/v1/namespaces/tpid/metadata", Some(TOKEN_A), None).await;
        assert_eq!(md["schema"]["isi"]["full_text_search"]["tokenizer"]["language"], "indonesian");

        // Suffix stemming: "makanan" in the document is found by "makan".
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpid/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["isi", "BM25", "makan"], "top_k": 5 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows"].as_array().unwrap().len(), 1, "{res}");
        assert_eq!(res["rows"][0]["id"], 3);

        // "tulis" finds "tulisan".
        let (_, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpid/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["isi", "BM25", "tulis"], "top_k": 5 })),
        )
        .await;
        assert_eq!(res["rows"][0]["id"], 2, "{res}");

        // Stopwords carry no signal, so a query of nothing but them matches
        // nothing rather than everything.
        let (_, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpid/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["isi", "BM25", "yang di untuk dari"], "top_k": 5 })),
        )
        .await;
        assert!(res["rows"].as_array().unwrap().is_empty(), "stopwords matched documents: {res}");
    }

    #[tokio::test]
    async fn v2_full_text_filters() {
        let state = test_state();
        call(
            &state,
            "POST",
            "/v2/namespaces/tpft",
            Some(TOKEN_A),
            Some(json!({
                "upsert_rows": [
                    { "id": 1, "vector": [1.0, 0.0], "body": "the king of spain visited" },
                    { "id": 2, "vector": [0.9, 0.1], "body": "spain beat the king in chess" },
                    { "id": 3, "vector": [0.8, 0.2], "body": "database internals" },
                ],
                "schema": { "body": { "type": "string", "full_text_search": true } }
            })),
        )
        .await;
        call(&state, "POST", "/v1/namespaces/tpft/compact", Some(TOKEN_A), None).await;

        let query = |filter: Value| {
            let state = state.clone();
            async move {
                let (status, res) = call(
                    &state,
                    "POST",
                    "/v2/namespaces/tpft/query",
                    Some(TOKEN_A),
                    Some(json!({
                        "rank_by": ["vector", "ANN", [1.0, 0.0]],
                        "top_k": 10,
                        "filters": filter,
                    })),
                )
                .await;
                assert_eq!(status, StatusCode::OK, "{res}");
                let mut ids: Vec<u64> = res["rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| r["id"].as_u64().unwrap())
                    .collect();
                ids.sort();
                ids
            }
        };

        // Both mention king and spain somewhere.
        assert_eq!(query(json!(["body", "ContainsAllTokens", "king spain"])).await, vec![1, 2]);
        // Only one has them as a phrase — and "of" being a stopword must not
        // break the adjacency.
        assert_eq!(
            query(json!(["body", "ContainsTokenSequence", "king of spain"])).await,
            vec![1]
        );
        // Reversed, the phrase appears nowhere.
        assert!(query(json!(["body", "ContainsTokenSequence", "spain king"])).await.is_empty());
        // Fuzzy tolerates a typo on a long enough token.
        assert_eq!(query(json!(["body", "Fuzzy", "databse"])).await, vec![3]);
        assert!(query(json!(["body", "Fuzzy", "zzzzzzzz"])).await.is_empty());
    }

    #[tokio::test]
    async fn v2_aggregations_and_group_by() {
        let state = test_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpa",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [
                { "id": 1, "vector": [1.0, 0.0], "color": "red",  "size": "L",  "price": 10,
                  "tags": ["a", "b"] },
                { "id": 2, "vector": [0.9, 0.1], "color": "red",  "size": "L",  "price": 20,
                  "tags": ["b"] },
                { "id": 3, "vector": [0.0, 1.0], "color": "blue", "size": "XL", "price": 5,
                  "tags": [] },
            ]})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");

        // A pure aggregation: no rank_by, and therefore no rows.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpa/query",
            Some(TOKEN_A),
            Some(json!({ "aggregate_by": {
                "n": ["Count"], "total": ["Sum", "price"],
                "cheapest": ["Min", "price"], "mean": ["Avg", "price"],
            }})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["aggregations"]["n"], 3);
        assert_eq!(res["aggregations"]["total"], 35);
        assert_eq!(res["aggregations"]["cheapest"], 5);
        assert!((res["aggregations"]["mean"].as_f64().unwrap() - 35.0 / 3.0).abs() < 1e-9);
        assert!(res.get("rows").is_none(), "a pure aggregation returned rows");
        assert!(res.get("aggregation_groups").is_none());

        // Filters select what is aggregated.
        let (_, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpa/query",
            Some(TOKEN_A),
            Some(json!({
                "aggregate_by": { "n": ["Count"], "total": ["Sum", "price"] },
                "filters": ["color", "Eq", "red"],
            })),
        )
        .await;
        assert_eq!(res["aggregations"]["n"], 2);
        assert_eq!(res["aggregations"]["total"], 30);

        // group_by reports aggregation_groups instead, with the key flattened in.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpa/query",
            Some(TOKEN_A),
            Some(json!({
                "aggregate_by": { "n": ["Count"] },
                "group_by": ["color", "size"],
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert!(res.get("aggregations").is_none(), "a grouped result also reported a total");
        let groups = res["aggregation_groups"].as_array().unwrap();
        assert_eq!(groups.len(), 2);
        let red = groups.iter().find(|g| g["color"] == "red").unwrap();
        assert_eq!(red["size"], "L");
        assert_eq!(red["n"], 2);

        // ForEachUnique explodes an array attribute into one group per element.
        let (_, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpa/query",
            Some(TOKEN_A),
            Some(json!({
                "aggregate_by": { "n": ["Count"] },
                "group_by": [["ForEachUnique", "tags"]],
            })),
        )
        .await;
        let groups = res["aggregation_groups"].as_array().unwrap();
        let by_tag = |t: &str| {
            groups.iter().find(|g| g["tags"] == t).map(|g| g["n"].as_u64().unwrap())
        };
        assert_eq!(by_tag("a"), Some(1));
        assert_eq!(by_tag("b"), Some(2), "a document tagged twice should count in both groups");

        // Ranking and aggregating together yields BOTH, so a faceted page is one
        // round trip rather than two.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpa/query",
            Some(TOKEN_A),
            Some(json!({
                "rank_by": ["vector", "ANN", [1.0, 0.0]],
                "top_k": 2,
                "aggregate_by": { "n": ["Count"] },
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows"].as_array().unwrap().len(), 2);
        assert_eq!(res["rows"][0]["id"], 1);
        assert_eq!(res["aggregations"]["n"], 3, "aggregation was dropped alongside ranking");
    }

    #[tokio::test]
    async fn v2_malformed_aggregations_are_rejected() {
        let state = test_state();
        call(
            &state,
            "POST",
            "/v2/namespaces/tpae",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [
                { "id": 1, "vector": [1.0, 0.0], "color": "red", "price": 10 }
            ]})),
        )
        .await;

        for body in [
            // Sum without an attribute.
            json!({ "aggregate_by": { "s": ["Sum"] } }),
            // Unknown function.
            json!({ "aggregate_by": { "s": ["Frobnicate", "price"] } }),
            // Grouping with nothing to compute.
            json!({ "group_by": ["color"] }),
            // Neither ranking nor aggregating.
            json!({ "top_k": 5 }),
            // Malformed group key.
            json!({ "aggregate_by": { "n": ["Count"] }, "group_by": [["Explode", "tags"]] }),
        ] {
            let (status, res) =
                call(&state, "POST", "/v2/namespaces/tpae/query", Some(TOKEN_A), Some(body.clone()))
                    .await;
            assert!(status.is_client_error(), "{body} was accepted with {status}: {res}");
        }

        // Summing a string is a type error the caller can act on, not a 500.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpae/query",
            Some(TOKEN_A),
            Some(json!({ "aggregate_by": { "s": ["Sum", "color"] } })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{res}");
    }

    #[tokio::test]
    async fn metrics_touch_no_object_storage_and_survive_deletion() {
        let state = test_state();
        for i in 0..4 {
            call(
                &state,
                "POST",
                &format!("/v2/namespaces/m{i}"),
                Some(TOKEN_A),
                Some(json!({ "upsert_rows": [{ "id": 1, "vector": [1.0, 0.0] }] })),
            )
            .await;
        }

        // A scrape must cost nothing. It used to cost one GET per namespace,
        // sequentially, which broke Prometheus' timeout at around seventy.
        let before = state.metrics.snapshot().gets;
        let (status, _) = call(&state, "GET", "/metrics", None, None).await;
        assert_eq!(status, StatusCode::OK);
        let (_, body) = call(&state, "GET", "/metrics", None, None).await;
        assert_eq!(
            state.metrics.snapshot().gets,
            before,
            "a metrics scrape read from object storage"
        );

        let sample = |text: &str, name: &str| -> f64 {
            text.lines()
                .find(|l| l.starts_with(&format!("{name} ")))
                .and_then(|l| l.split(' ').nth(1))
                .and_then(|v| v.parse().ok())
                .unwrap_or(-1.0)
        };
        let text = body["raw"].as_str().unwrap_or_default();
        assert_eq!(sample(text, "fckdb_namespaces"), 4.0);
        assert!(sample(text, "fckdb_bytes_stored") > 0.0, "stored bytes were not reported");

        // Counters must not fall when a namespace goes away. Prometheus reads a
        // falling counter as a restart and produces a garbage rate.
        let gets_before = state.metrics.snapshot().gets;
        let writes_before = state.metrics.snapshot().writes;
        assert!(gets_before > 0 && writes_before > 0);
        for i in 0..3 {
            call(&state, "DELETE", &format!("/v1/namespaces/m{i}"), Some(TOKEN_A), None).await;
        }
        let after = state.metrics.snapshot();
        assert!(
            after.gets >= gets_before && after.writes >= writes_before,
            "counters went backwards when namespaces were deleted: \
             gets {gets_before} -> {}, writes {writes_before} -> {}",
            after.gets,
            after.writes
        );
    }

    #[tokio::test]
    async fn aggregation_streams_rather_than_gathering() {
        // Sharded, so the streaming path actually iterates more than once, and
        // the answer must match the unsharded one exactly.
        let state = test_state();
        let rows: Vec<Value> = (0..120)
            .map(|i| json!({ "id": i, "vector": [1.0, i as f64 * 0.01],
                             "n": i, "tier": if i % 4 == 0 { "gold" } else { "grey" } }))
            .collect();
        for (ns, shards) in [("agg1", 1), ("agg8", 8)] {
            call(
                &state,
                "POST",
                &format!("/v2/namespaces/{ns}"),
                Some(TOKEN_A),
                Some(json!({ "sharding": { "num_shards": shards }, "upsert_rows": rows })),
            )
            .await;
            call(&state, "POST", &format!("/v1/namespaces/{ns}/compact"), Some(TOKEN_A), None).await;
        }

        for body in [
            json!({ "aggregate_by": { "n": ["Count"], "s": ["Sum","n"], "a": ["Avg","n"],
                                      "lo": ["Min","n"], "hi": ["Max","n"] } }),
            json!({ "aggregate_by": { "n": ["Count"] }, "group_by": ["tier"] }),
            json!({ "aggregate_by": { "n": ["Count"], "s": ["Sum","n"] },
                    "filters": ["tier","Eq","gold"] }),
        ] {
            let (_, one) =
                call(&state, "POST", "/v2/namespaces/agg1/query", Some(TOKEN_A), Some(body.clone()))
                    .await;
            let (_, eight) =
                call(&state, "POST", "/v2/namespaces/agg8/query", Some(TOKEN_A), Some(body.clone()))
                    .await;
            assert_eq!(one, eight, "sharded aggregation disagreed for {body}");
        }

        // Spot-check the arithmetic, so "they agree" cannot mean "both wrong".
        let (_, res) = call(
            &state,
            "POST",
            "/v2/namespaces/agg8/query",
            Some(TOKEN_A),
            Some(json!({ "aggregate_by": { "n": ["Count"], "s": ["Sum","n"], "hi": ["Max","n"] } })),
        )
        .await;
        assert_eq!(res["aggregations"]["n"], 120);
        assert_eq!(res["aggregations"]["s"], 119 * 120 / 2);
        assert_eq!(res["aggregations"]["hi"], 119);
    }

    #[tokio::test]
    async fn v2_sparse_vector_search() {
        let state = test_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpsp",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [
                { "id": 1, "vector": [1.0, 0.0], "terms": { "cat": 1.0, "pet": 0.5 } },
                { "id": 2, "vector": [0.0, 1.0], "terms": { "cat": 0.2, "dog": 1.0 } },
                { "id": 3, "vector": [0.5, 0.5], "terms": { "car": 2.0 } },
            ]})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");

        // The attribute's type is inferred from the object shape.
        let (_, md) = call(&state, "GET", "/v1/namespaces/tpsp/metadata", Some(TOKEN_A), None).await;
        assert_eq!(md["schema"]["terms"]["type"], "{}f16");

        let (status, _) =
            call(&state, "POST", "/v1/namespaces/tpsp/compact", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::OK);

        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpsp/query",
            Some(TOKEN_A),
            Some(json!({
                "rank_by": ["terms", "SparseKNN", { "cat": 1.0, "pet": 1.0 }],
                "top_k": 5
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        let rows = res["rows"].as_array().unwrap();
        // doc1 = 1.0 + 0.5 = 1.5, doc2 = 0.2. doc3 shares no dimension and is
        // absent rather than scored zero.
        assert_eq!(rows.len(), 2, "{res}");
        assert_eq!(rows[0]["id"], 1);
        assert_eq!(rows[1]["id"], 2);
        // Sparse similarity is a score: higher is better, like BM25.
        assert!(rows[0]["$dist"].as_f64().unwrap() > rows[1]["$dist"].as_f64().unwrap());
        assert!((rows[0]["$dist"].as_f64().unwrap() - 1.5).abs() < 1e-5);

        // A dimension nobody has returns nothing.
        let (_, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpsp/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["terms", "SparseKNN", { "zebra": 1.0 }], "top_k": 5 })),
        )
        .await;
        assert!(res["rows"].as_array().unwrap().is_empty());

        // An attribute that is not sparse is refused by name.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpsp/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["nope", "SparseKNN", { "cat": 1.0 }], "top_k": 5 })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{res}");
    }

    #[tokio::test]
    async fn v2_multi_query_and_rrf_hybrid_search() {
        let state = test_state();
        call(
            &state,
            "POST",
            "/v2/namespaces/tphy",
            Some(TOKEN_A),
            Some(json!({
                "upsert_rows": [
                    { "id": 1, "vector": [1.0, 0.0], "body": "quick brown fox", "n": 1 },
                    { "id": 2, "vector": [0.0, 1.0], "body": "lazy dog sleeping", "n": 2 },
                    { "id": 3, "vector": [0.7, 0.7], "body": "quick lazy afternoon", "n": 3 },
                ],
                "schema": { "body": { "type": "string", "full_text_search": true } }
            })),
        )
        .await;
        call(&state, "POST", "/v1/namespaces/tphy/compact", Some(TOKEN_A), None).await;

        // Without rerank_by, each sub-query reports separately and in order.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tphy/query",
            Some(TOKEN_A),
            Some(json!({ "queries": [
                { "rank_by": ["vector", "ANN", [0.0, 1.0]], "top_k": 3 },
                { "rank_by": ["body", "BM25", "quick"], "top_k": 3 },
                { "aggregate_by": { "n": ["Count"] } },
            ]})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        let results = res["results"].as_array().unwrap();
        assert_eq!(results.len(), 3);
        // Vector query: nearest to [0,1] is doc 2.
        assert_eq!(results[0]["rows"][0]["id"], 2);
        // BM25 query: "quick" is in docs 1 and 3.
        let bm25: Vec<u64> = results[1]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["id"].as_u64().unwrap())
            .collect();
        assert_eq!(bm25.len(), 2);
        // An aggregating sub-query reports aggregations, not rows.
        assert_eq!(results[2]["aggregations"]["n"], 3);
        assert!(results[2].get("rows").is_none());

        // With rerank_by RRF, the lists fuse into one. This is hybrid search: a
        // BM25 relevance score and a cosine distance are on incomparable scales,
        // so only the RANKS are combined.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tphy/query",
            Some(TOKEN_A),
            Some(json!({
                "top_k": 3,
                "rerank_by": ["RRF"],
                "queries": [
                    { "rank_by": ["vector", "ANN", [1.0, 0.0]], "top_k": 3 },
                    { "rank_by": ["body", "BM25", "quick"], "top_k": 3 },
                ]
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert!(res.get("results").is_none(), "a fused query also reported per-list results");
        let rows = res["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        // Document 1 is first in both lists, so it must fuse to the top.
        assert_eq!(rows[0]["id"], 1);
        // Fused scores descend, and are RRF contributions rather than either
        // original scale.
        let scores: Vec<f64> = rows.iter().map(|r| r["$dist"].as_f64().unwrap()).collect();
        assert!(scores.windows(2).all(|w| w[0] >= w[1]), "fused scores not ordered: {scores:?}");
        assert!(scores[0] < 1.0, "fused score looks like a raw BM25 score: {scores:?}");
    }

    #[tokio::test]
    async fn v2_multi_query_validation() {
        let state = test_state();
        call(
            &state,
            "POST",
            "/v2/namespaces/tpmv",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [{ "id": 1, "vector": [1.0, 0.0] }] })),
        )
        .await;

        let cases: Vec<(&str, Value)> = vec![
            ("empty queries", json!({ "queries": [] })),
            (
                "mixed with ordinary fields",
                json!({ "queries": [{ "rank_by": ["vector", "ANN", [1.0, 0.0]] }],
                        "rank_by": ["vector", "ANN", [1.0, 0.0]] }),
            ),
            (
                "consistency inside a sub-query",
                json!({ "queries": [{ "rank_by": ["vector", "ANN", [1.0, 0.0]],
                                      "consistency": { "level": "eventual" } }] }),
            ),
            (
                "nested multi-query",
                json!({ "queries": [{ "queries": [
                    { "rank_by": ["vector", "ANN", [1.0, 0.0]] }] }] }),
            ),
            (
                "unknown rerank function",
                json!({ "rerank_by": ["Magic"],
                        "queries": [{ "rank_by": ["vector", "ANN", [1.0, 0.0]] }] }),
            ),
            (
                "rrf over aggregations only",
                json!({ "rerank_by": ["RRF"], "queries": [{ "aggregate_by": { "n": ["Count"] } }] }),
            ),
        ];
        for (label, body) in cases {
            let (status, res) =
                call(&state, "POST", "/v2/namespaces/tpmv/query", Some(TOKEN_A), Some(body)).await;
            assert!(status.is_client_error(), "{label} accepted with {status}: {res}");
        }

        // Seventeen sub-queries is one too many.
        let too_many: Vec<Value> = (0..17)
            .map(|_| json!({ "rank_by": ["vector", "ANN", [1.0, 0.0]] }))
            .collect();
        let (status, _) = call(
            &state,
            "POST",
            "/v2/namespaces/tpmv/query",
            Some(TOKEN_A),
            Some(json!({ "queries": too_many })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    fn embedding_state() -> (AppState, Arc<crate::embed::testing::StubEmbedder>) {
        let stub = Arc::new(crate::embed::testing::StubEmbedder::new(8));
        let dynamic: Arc<dyn crate::embed::Embedder> = stub.clone();
        (test_state().with_embedder(dynamic), stub)
    }

    #[tokio::test]
    async fn v2_embed_at_query_time() {
        let (state, stub) = embedding_state();
        // Write with explicit vectors so only the query embeds.
        let a = json!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = json!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        call(
            &state,
            "POST",
            "/v2/namespaces/tpe",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [
                { "id": 1, "vector": a, "body": "one" },
                { "id": 2, "vector": b, "body": "two" },
            ]})),
        )
        .await;

        // The stub embeds "aaaa" to a vector with weight in slot 4 (len 4 + i 0).
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpe/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["vector", "ANN", ["Embed", "aaaa"]], "top_k": 2 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows"].as_array().unwrap().len(), 2);
        assert!(stub.call_count() >= 1, "the query did not embed");

        // The model option reaches the endpoint.
        let before = stub.call_count();
        call(
            &state,
            "POST",
            "/v2/namespaces/tpe/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["vector", "ANN", ["Embed", "aaaa", { "model": "m1" }]] })),
        )
        .await;
        let calls = stub.calls.lock().unwrap();
        assert!(calls.len() > before);
        assert_eq!(calls.last().unwrap().1.as_deref(), Some("m1"));
    }

    #[tokio::test]
    async fn embedding_is_refused_by_name_when_unconfigured() {
        // No embedder: a request needing one must say so, never quietly rank on
        // a zero vector, which would look like a working search.
        let state = test_state();
        call(
            &state,
            "POST",
            "/v2/namespaces/tpne",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [{ "id": 1, "vector": [1.0, 0.0] }] })),
        )
        .await;
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpne/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["vector", "ANN", ["Embed", "anything"]] })),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{res}");
        assert!(res["error"].as_str().unwrap().contains("FCKDB_EMBED_URL"));
    }

    #[tokio::test]
    async fn a_failing_embedding_endpoint_is_a_bad_gateway() {
        let stub: Arc<dyn crate::embed::Embedder> =
            Arc::new(crate::embed::testing::StubEmbedder::failing("rate limited"));
        let state = test_state().with_embedder(stub);
        call(
            &state,
            "POST",
            "/v2/namespaces/tpfe",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [{ "id": 1, "vector": [1.0, 0.0] }] })),
        )
        .await;
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpfe/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["vector", "ANN", ["Embed", "x"]] })),
        )
        .await;
        // Somebody else's service failing is neither the caller's fault nor ours.
        assert_eq!(status, StatusCode::BAD_GATEWAY, "{res}");
        assert!(res["error"].as_str().unwrap().contains("rate limited"));
    }

    #[tokio::test]
    async fn v2_embed_at_write_time() {
        let (state, stub) = embedding_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpwe",
            Some(TOKEN_A),
            Some(json!({
                "upsert_rows": [
                    { "id": 1, "body": "quick brown fox" },
                    { "id": 2, "body": "lazy sleeping dog" },
                ],
                "schema": { "body": { "type": "string", "embed": { "model": "m1" } } }
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows_upserted"], 2);
        // One call for the whole batch, not one per document.
        assert_eq!(stub.call_count(), 1, "the batch became several requests");

        // The documents are queryable, so a vector really was attached.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpwe/query",
            Some(TOKEN_A),
            Some(json!({ "rank_by": ["vector", "ANN", ["Embed", "quick brown fox"]], "top_k": 2 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows"][0]["id"], 1, "the matching document did not rank first");

        // An explicit vector always wins: a caller correcting a document must not
        // have it silently overwritten by a re-embedding.
        let before = stub.call_count();
        let (status, _) = call(
            &state,
            "POST",
            "/v2/namespaces/tpwe",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [
                { "id": 3, "vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "body": "text" }
            ]})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(stub.call_count(), before, "an explicit vector was re-embedded anyway");

        // The metadata reports the embedding configuration.
        let (_, md) = call(&state, "GET", "/v1/namespaces/tpwe/metadata", Some(TOKEN_A), None).await;
        assert_eq!(md["schema"]["body"]["embed"]["model"], "m1");
    }

    #[tokio::test]
    async fn a_hybrid_query_embeds_once_for_all_sub_queries() {
        let (state, stub) = embedding_state();
        call(
            &state,
            "POST",
            "/v2/namespaces/tphe",
            Some(TOKEN_A),
            Some(json!({
                "upsert_rows": [
                    { "id": 1, "body": "quick brown fox" },
                    { "id": 2, "body": "lazy sleeping dog" },
                ],
                "schema": { "body": { "type": "string", "full_text_search": true,
                                      "embed": {} } }
            })),
        )
        .await;
        call(&state, "POST", "/v1/namespaces/tphe/compact", Some(TOKEN_A), None).await;
        let before = stub.call_count();

        // Two sub-queries embedding the same text should cost one call, not two:
        // a hybrid query almost always embeds the text it also searches for
        // lexically.
        let (status, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tphe/query",
            Some(TOKEN_A),
            Some(json!({
                "top_k": 2,
                "rerank_by": ["RRF"],
                "queries": [
                    { "rank_by": ["vector", "ANN", ["Embed", "quick fox"]], "top_k": 2 },
                    { "rank_by": ["body", "BM25", "quick fox"], "top_k": 2 },
                ]
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(stub.call_count(), before + 1, "the sub-queries embedded separately");
        assert!(!res["rows"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn v2_order_by_attribute() {
        let state = test_state();
        call(
            &state,
            "POST",
            "/v2/namespaces/tpo",
            Some(TOKEN_A),
            Some(json!({ "upsert_rows": [
                { "id": 1, "vector": [1.0, 0.0], "rank": 30 },
                { "id": 2, "vector": [0.0, 1.0], "rank": 10 },
                { "id": 3, "vector": [0.5, 0.5], "rank": 20 },
                { "id": 4, "vector": [0.1, 0.9] },
            ]})),
        )
        .await;

        for (direction, expected) in
            [("asc", vec![2, 3, 1, 4]), ("desc", vec![1, 3, 2, 4])]
        {
            let (status, res) = call(
                &state,
                "POST",
                "/v2/namespaces/tpo/query",
                Some(TOKEN_A),
                Some(json!({ "rank_by": ["rank", direction], "top_k": 10 })),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{direction}: {res}");
            let ids: Vec<u64> =
                res["rows"].as_array().unwrap().iter().map(|r| r["id"].as_u64().unwrap()).collect();
            assert_eq!(ids, expected, "{direction} ordering wrong");
            // No vector was ranked, so no distance is reported.
            assert!(
                res["rows"][0].get("$dist").is_none(),
                "an ordered row carried a distance"
            );
        }

        // Ordering combines with filters and attribute projection.
        let (_, res) = call(
            &state,
            "POST",
            "/v2/namespaces/tpo/query",
            Some(TOKEN_A),
            Some(json!({
                "rank_by": ["rank", "asc"],
                "top_k": 10,
                "filters": ["rank", "Gte", 20],
                "include_attributes": ["rank"],
            })),
        )
        .await;
        let rows = res["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["rank"], 20);
        assert_eq!(rows[1]["rank"], 30);
    }

    #[tokio::test]
    async fn v2_requires_auth_and_validates_namespaces() {
        let state = test_state();
        let body = json!({ "upsert_rows": [{ "id": 1, "vector": [1.0, 0.0] }] });
        let (status, _) = call(&state, "POST", "/v2/namespaces/x", None, Some(body.clone())).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, _) =
            call(&state, "POST", "/v2/namespaces/a.b", Some(TOKEN_A), Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) =
            call(&state, "GET", "/v1/namespaces/ghost/metadata", Some(TOKEN_A), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
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
    async fn column_writes_transpose_into_documents() {
        let state = test_state();
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/cols/write",
            Some(TOKEN_A),
            Some(json!({ "upsert_columns": {
                "id": [1, 2, 3],
                "vector": [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
                "name": ["a", "b", null],
                "rank": [10, 20, 30],
            }})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows_upserted"], 3);

        let (_, res) = call(
            &state,
            "POST",
            "/v1/namespaces/cols/query",
            Some(TOKEN_A),
            Some(json!({ "vector": [1.0, 0.0], "top_k": 3, "include_attributes": true })),
        )
        .await;
        assert_eq!(res["hits"][0]["id"], 1);
        assert_eq!(res["hits"][0]["attrs"]["name"], "a");
        assert_eq!(res["hits"][0]["attrs"]["rank"], 10);
        // A null column entry means this document has no value, not a null value.
        let third = res["hits"].as_array().unwrap().iter().find(|h| h["id"] == 3).unwrap();
        assert!(third["attrs"].get("name").is_none(), "null column became an attribute");
        assert_eq!(third["attrs"]["rank"], 30);

        // patch_columns merges without touching vectors.
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/cols/write",
            Some(TOKEN_A),
            Some(json!({ "patch_columns": { "id": [1], "rank": [99] } })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows_patched"], 1);
        let (_, res) = call(
            &state,
            "POST",
            "/v1/namespaces/cols/query",
            Some(TOKEN_A),
            Some(json!({ "vector": [1.0, 0.0], "top_k": 1, "include_attributes": true })),
        )
        .await;
        assert_eq!(res["hits"][0]["attrs"]["rank"], 99);
    }

    #[tokio::test]
    async fn ragged_and_invalid_column_writes_are_rejected() {
        let state = test_state();
        let cases: Vec<(&str, Value)> = vec![
            ("missing id column", json!({ "upsert_columns": { "vector": [[1.0]] } })),
            (
                "ragged columns",
                json!({ "upsert_columns": {
                    "id": [1, 2],
                    "vector": [[1.0, 0.0], [0.0, 1.0]],
                    "name": ["only-one"],
                }}),
            ),
            (
                "missing vector",
                json!({ "upsert_columns": { "id": [1], "name": ["a"] } }),
            ),
            (
                "vector in patch_columns",
                json!({ "patch_columns": { "id": [1], "vector": [[1.0, 0.0]] } }),
            ),
        ];
        for (label, body) in cases {
            let (status, res) =
                call(&state, "POST", "/v1/namespaces/badcols/write", Some(TOKEN_A), Some(body))
                    .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{label} was accepted: {res}");
        }
    }

    #[tokio::test]
    async fn filter_based_writes_delete_and_patch_in_bulk() {
        let state = test_state();
        let ids: Vec<u64> = (1..=20).collect();
        let body = json!({ "upsert_columns": {
            "id": ids,
            "vector": ids.iter().map(|i| vec![1.0, *i as f64 * 0.01]).collect::<Vec<_>>(),
            "tier": ids.iter().map(|i| if i % 2 == 0 { "gold" } else { "silver" }).collect::<Vec<_>>(),
            "rank": ids.clone(),
        }});
        let (status, res) =
            call(&state, "POST", "/v1/namespaces/bulk/write", Some(TOKEN_A), Some(body)).await;
        assert_eq!(status, StatusCode::OK, "{res}");

        // Patch every gold document.
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/bulk/write",
            Some(TOKEN_A),
            Some(json!({ "patch_by_filter": {
                "filters": ["tier", "Eq", "gold"],
                "patch": { "tier": "platinum" },
            }})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows_patched"], 10);
        assert_eq!(res["rows_remaining"], false);

        let (_, res) = call(
            &state,
            "POST",
            "/v1/namespaces/bulk/query",
            Some(TOKEN_A),
            Some(json!({ "vector": [1.0, 0.0], "top_k": 100,
                         "filter": ["tier", "Eq", "platinum"] })),
        )
        .await;
        assert_eq!(res["hits"].as_array().unwrap().len(), 10, "patch_by_filter missed rows");

        // Delete every silver document.
        let (status, res) = call(
            &state,
            "POST",
            "/v1/namespaces/bulk/write",
            Some(TOKEN_A),
            Some(json!({ "delete_by_filter": ["tier", "Eq", "silver"] })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{res}");
        assert_eq!(res["rows_deleted"], 10);

        let (_, res) = call(
            &state,
            "POST",
            "/v1/namespaces/bulk/query",
            Some(TOKEN_A),
            Some(json!({ "vector": [1.0, 0.0], "top_k": 100 })),
        )
        .await;
        assert_eq!(res["hits"].as_array().unwrap().len(), 10, "delete_by_filter left rows behind");

        // An empty patch is refused rather than committing nothing.
        let (status, _) = call(
            &state,
            "POST",
            "/v1/namespaces/bulk/write",
            Some(TOKEN_A),
            Some(json!({ "patch_by_filter": { "filters": ["tier", "Eq", "gold"], "patch": {} } })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // A filter matching nothing is an error, not a silent no-op write.
        let (status, _) = call(
            &state,
            "POST",
            "/v1/namespaces/bulk/write",
            Some(TOKEN_A),
            Some(json!({ "delete_by_filter": ["tier", "Eq", "nonexistent"] })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
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
            shards: vec![],
            schema: Default::default(),
            created_at: None,
            last_write_at: None,
            updated_at: None,
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
