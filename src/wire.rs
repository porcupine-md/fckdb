//! Wire types shared by the storage engine and the HTTP surface.
//!
//! These are the contract. Keeping them in one place means the engine and the
//! server cannot drift, and the e2e driver exercises the same shapes real
//! clients send.

use crate::doc::{Attrs, DistanceMetric, Doc, Filter, Id, IdType, Include};
use anyhow::{Result, bail};
use std::collections::BTreeMap;
use crate::value::{Type, Value};
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
    /// Column-oriented form of `upsert`. Each key is a column, each value the
    /// per-document values in the same order.
    #[serde(default)]
    pub upsert_columns: Option<Columns>,
    #[serde(default)]
    pub patch_columns: Option<Columns>,
    /// Delete or patch every document matching a filter, up to a cap.
    #[serde(default)]
    pub delete_by_filter: Option<Filter>,
    #[serde(default)]
    pub patch_by_filter: Option<FilterPatch>,
    /// Settable on a namespace's first write only; a later change is rejected
    /// because the index's cluster assignment depends on it.
    #[serde(default)]
    pub distance_metric: Option<DistanceMetric>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PatchRow {
    pub id: Id,
    #[serde(flatten)]
    pub attrs: Attrs,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FilterPatch {
    pub filters: Filter,
    pub patch: Attrs,
}

impl WriteRequest {
    pub fn is_empty(&self) -> bool {
        self.upsert.is_empty()
            && self.patch.is_empty()
            && self.delete.is_empty()
            && self.upsert_columns.is_none()
            && self.patch_columns.is_none()
            && self.delete_by_filter.is_none()
            && self.patch_by_filter.is_none()
    }
}

/// Column-oriented documents: `{"id":[1,2], "vector":[[…],[…]], "name":["a","b"]}`
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct Columns(pub BTreeMap<String, Vec<serde_json::Value>>);

/// One document pulled out of a column layout.
pub struct Row {
    pub id: Id,
    pub vector: Option<Vec<f32>>,
    pub attrs: Attrs,
}

impl Columns {
    /// Transpose columns into rows.
    ///
    /// Every column must be the same length, because the Nth entry of each is by
    /// definition the same document. A ragged batch is a client bug that would
    /// otherwise silently attach one document's attribute to another.
    pub fn transpose(&self) -> Result<Vec<Row>> {
        let Some(ids) = self.0.get("id") else {
            bail!("column writes require an `id` column");
        };
        let n = ids.len();
        for (name, col) in &self.0 {
            if col.len() != n {
                bail!(
                    "column {name:?} has {} values but `id` has {n}; every column must be \
                     the same length",
                    col.len()
                );
            }
        }

        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let id: Id = serde_json::from_value(ids[i].clone())
                .map_err(|e| anyhow::anyhow!("row {i}: {e}"))?;

            let vector = match self.0.get("vector").map(|c| &c[i]) {
                None | Some(serde_json::Value::Null) => None,
                Some(raw) => Some(
                    serde_json::from_value::<Vec<f32>>(raw.clone())
                        .map_err(|e| anyhow::anyhow!("row {i} vector: {e}"))?,
                ),
            };

            let mut attrs = Attrs::new();
            for (name, col) in &self.0 {
                if name == "id" || name == "vector" {
                    continue;
                }
                // A null in a column means this document has no value for it,
                // which is different from the attribute not existing at all.
                let value = crate::value::from_json(col[i].clone())
                    .map_err(|e| anyhow::anyhow!("row {i}, column {name:?}: {e}"))?;
                if !matches!(value, Value::Null) {
                    attrs.insert(name.clone(), value);
                }
            }
            rows.push(Row { id, vector, attrs });
        }
        Ok(rows)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WriteResponse {
    pub seq: u64,
    pub records: usize,
    pub rows_upserted: usize,
    pub rows_patched: usize,
    pub rows_deleted: usize,
    /// True when a filter-based write hit its cap and more documents still
    /// match. Reissue the same request to continue.
    pub rows_remaining: bool,
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
