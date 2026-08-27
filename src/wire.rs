//! Wire types shared by the storage engine and the HTTP surface.
//!
//! These are the contract. Keeping them in one place means the engine and the
//! server cannot drift, and the e2e driver exercises the same shapes real
//! clients send.

use crate::doc::{DistanceMetric, Doc, Filter, Id, IdType, Include};
use crate::value::Type;
pub use crate::doc::Hit;
use serde::{Deserialize, Serialize};

fn default_top_k() -> usize {
    10
}
fn default_nprobe() -> usize {
    8
}

/// How fresh a query's view of the namespace must be.
///
/// `Strong` costs exactly one extra object storage roundtrip — the commit point
/// check — which on a distant bucket is the entire warm-query latency. `Eventual`
/// spends that roundtrip only when its snapshot has aged out.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum Consistency {
    #[default]
    Strong,
    /// Serve from a manifest snapshot no older than `max_age_ms`.
    Eventual { max_age_ms: u64 },
}

#[derive(Debug, Clone, Deserialize)]
pub struct QueryRequest {
    pub vector: Vec<f32>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub filter: Option<Filter>,
    /// Clusters to probe. Higher trades latency and object GETs for recall.
    #[serde(default = "default_nprobe")]
    pub nprobe: usize,
    #[serde(default)]
    pub consistency: Consistency,
    /// Which attributes to return: `true`, `false`, or a list of names.
    #[serde(default)]
    pub include_attributes: Include,
}

impl QueryRequest {
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            top_k: default_top_k(),
            filter: None,
            nprobe: default_nprobe(),
            consistency: Consistency::Strong,
            include_attributes: Include::None,
        }
    }
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }
    pub fn nprobe(mut self, n: usize) -> Self {
        self.nprobe = n;
        self
    }
    pub fn filter(mut self, f: Filter) -> Self {
        self.filter = Some(f);
        self
    }
    pub fn consistency(mut self, c: Consistency) -> Self {
        self.consistency = c;
        self
    }
    pub fn include(mut self, i: Include) -> Self {
        self.include_attributes = i;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub hits: Vec<Hit>,
    /// False when the answer may not reflect every committed write. Always true
    /// for `Consistency::Strong`, which errors rather than lie.
    pub consistent: bool,
    /// False means the vector index was not used and this was an exhaustive scan.
    pub indexed: bool,
    /// Records read from the unindexed WAL tail for this query.
    pub unindexed_records: usize,
    pub unindexed_bytes: u64,
    /// Object storage GETs actually issued, cache hits excluded.
    pub object_gets: usize,
    pub cache_hits: u64,
    pub took_ms: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WriteRequest {
    #[serde(default)]
    pub upsert: Vec<Doc>,
    /// Merge attributes into existing documents. A null value removes one.
    #[serde(default)]
    pub patch: Vec<PatchRow>,
    #[serde(default)]
    pub delete: Vec<Id>,
    /// Settable on a namespace's first write only; a later change is rejected
    /// because the index's cluster assignment depends on it.
    #[serde(default)]
    pub distance_metric: Option<DistanceMetric>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PatchRow {
    pub id: Id,
    #[serde(flatten)]
    pub attrs: std::collections::BTreeMap<String, crate::value::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteResponse {
    pub seq: u64,
    pub records: usize,
    pub took_ms: u64,
}

/// Everything answerable from the manifest alone — no LIST, no data fetch.
/// That is deliberate: backpressure and billing must be cheap enough to consult
/// on every request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NamespaceMetadata {
    pub namespace: String,
    pub indexed_docs: usize,
    pub clusters: usize,
    pub segments: usize,
    pub segment_bytes: u64,
    pub wal_entries: usize,
    pub unindexed_records: usize,
    pub unindexed_bytes: u64,
    pub index_bytes: u64,
    pub total_bytes: u64,
    pub dim: Option<usize>,
    /// True when the unindexed tail is large enough that writes are being
    /// refused until compaction catches up.
    pub write_backpressure: bool,
    /// Declared attribute types, inferred from the first write carrying each.
    #[serde(default)]
    pub schema: std::collections::BTreeMap<String, Type>,
    #[serde(default)]
    pub id_type: Option<IdType>,
    #[serde(default)]
    pub distance_metric: DistanceMetric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactResponse {
    pub records_in: usize,
    pub docs_out: usize,
    pub wal_consumed: usize,
    pub clusters: usize,
    pub cas_attempts: usize,
    pub took_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcResponse {
    pub scanned: usize,
    pub referenced: usize,
    pub deleted: usize,
    /// Unreferenced objects left alone because they are younger than the grace
    /// window and might belong to a write still in flight.
    pub spared_recent: usize,
    pub took_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmResponse {
    pub objects_warmed: usize,
    pub bytes: u64,
    pub already_cached: usize,
    pub took_ms: u64,
}
