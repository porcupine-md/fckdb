//! The storage engine: WAL, commit protocol, compaction, query path, operations.
//!
//! Object storage layout for one namespace:
//!
//!   {prefix}/manifest              commit point. CAS here. The ONLY mutable object.
//!   {prefix}/wal/{seq}-{uuid}.bin  framed Records, uncompacted
//!   {prefix}/data/{uuid}.bin       framed Records, compacted segment
//!   {prefix}/data/{uuid}.cen       centroid blob
//!   {prefix}/data/{uuid}.clu       framed Records, one cluster
//!
//! Everything except the manifest is immutable and uniquely named. That single
//! property buys three things for free: the read cache needs no invalidation, a
//! failed CAS leaves garbage rather than corruption, and GC is a set difference.

use crate::attrindex::{self, AttrIndex};
use crate::fts::FtsIndex;
use crate::cache::RingCache;
use crate::doc::{DistanceMetric, Doc, Filter, Id, Include, Record, Schema};
use crate::value::Type;
use std::collections::BTreeMap;
use crate::index::{self, IndexMeta, IvfParams};
use crate::wire::{
    CompactResponse, Consistency, GcResponse, Hit, NamespaceMetadata, QueryRequest, QueryResponse,
    WarmResponse,
};
use anyhow::{Context, Result, bail};
use bytes::{BufMut, Bytes, BytesMut};
use futures::StreamExt;
use object_store::{
    Error as OsError, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload, UpdateVersion,
    path::Path,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

const MAX_CAS_ATTEMPTS: usize = 100;

/// Caps on one coalesced batch. Bounds memory and keeps a single WAL object a
/// sane size; leftovers ride the next batch. This also sets the per-namespace
/// write throughput ceiling: MAX_BATCH_LEN / commit_latency writes per second.
pub const MAX_BATCH_LEN: usize = 1024;
pub const MAX_BATCH_BYTES: usize = 8 << 20;

/// How much unindexed WAL a query will read before it stops.
///
/// Mirrors turbopuffer's documented 128 MiB. Past this point exhaustive scanning
/// of the tail costs more than the query is worth, so the choice is between a
/// stale answer and no answer. `Eventual` takes stale; `Strong` refuses, because
/// silently violating the consistency it promised is the one option that is
/// always wrong.
pub const MAX_UNINDEXED_SCAN_BYTES: u64 = 128 << 20;

/// Caps on filter-based writes, matching turbopuffer's documented limits.
///
/// These exist so indexing and consistent reads can keep up: an unbounded
/// `delete_by_filter` on a large namespace would produce a WAL entry bigger than
/// the query scan cap, making the namespace unqueryable until compaction caught
/// up. Callers reissue the request until `rows_remaining` is false.
pub const MAX_DELETE_BY_FILTER_ROWS: usize = 5_000_000;
pub const MAX_PATCH_BY_FILTER_ROWS: usize = 50_000;

/// Absolute ceiling on an exact filtered scan, so a pathological namespace
/// cannot turn one query into an unbounded amount of work.
const PREFILTER_MAX_CANDIDATES: usize = 8192;

/// Should a filtered query scan its candidates exactly, or probe clusters and
/// filter after?
///
/// Compare what each path actually reads. The exact path reads `candidates`
/// documents. The cluster path reads roughly `docs * nprobe / clusters` of them.
/// So the exact path wins precisely when the filter is more selective than the
/// fraction of the index a probe would touch — which makes the decision
/// self-tuning as `nprobe` changes, rather than a constant that happens to fit
/// one dataset size.
///
/// It is also the path with no recall loss: probing and then filtering can return
/// fewer than `top_k` results, or none, when every surviving document lives in a
/// cluster the query never probed.
fn should_prefilter(candidates: usize, docs: usize, clusters: usize, nprobe: usize) -> bool {
    if candidates > PREFILTER_MAX_CANDIDATES {
        return false;
    }
    if clusters == 0 {
        return true;
    }
    let probed_docs = docs.saturating_mul(nprobe.max(1)) / clusters;
    candidates <= probed_docs.max(1)
}

// ---------------------------------------------------------------- framing

/// Length-prefixed framing so one object can hold many records.
pub fn frame(entries: &[Bytes]) -> Bytes {
    let total = entries.iter().map(|e| e.len() + 4).sum();
    let mut buf = BytesMut::with_capacity(total);
    for e in entries {
        buf.put_u32_le(e.len() as u32);
        buf.put_slice(e);
    }
    buf.freeze()
}

pub fn unframe(blob: &Bytes) -> Result<Vec<Bytes>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < blob.len() {
        let Some(hdr) = blob.get(pos..pos + 4) else {
            bail!("truncated frame header at offset {pos}");
        };
        let len = u32::from_le_bytes(hdr.try_into().unwrap()) as usize;
        pos += 4;
        if pos + len > blob.len() {
            bail!("truncated frame body at {pos}: want {len}, have {}", blob.len() - pos);
        }
        out.push(blob.slice(pos..pos + len));
        pos += len;
    }
    Ok(out)
}

fn frame_records(records: &[Record]) -> Bytes {
    frame(&records.iter().map(|r| r.encode()).collect::<Vec<_>>())
}

// ---------------------------------------------------------------- manifest

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WalEntry {
    pub name: String,
    pub bytes: u64,
    pub records: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentEntry {
    pub name: String,
    pub bytes: u64,
    pub docs: u32,
}

/// A namespace's commit point.
///
/// Sizes are recorded inline so backpressure, billing and the query planner can
/// all be answered from this one small object — no LIST, no data fetch. Ordering
/// is defined by the order of `wal`, never by filename.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub next_seq: u64,
    pub wal: Vec<WalEntry>,
    pub segments: Vec<SegmentEntry>,
    pub index: Option<IndexMeta>,
    #[serde(default)]
    pub schema: Schema,
    /// Nanoseconds since the epoch. Reported by the compatibility metadata
    /// endpoint, which returns ISO 8601.
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub last_write_at: Option<i64>,
    #[serde(default)]
    pub updated_at: Option<i64>,
}

impl Manifest {
    /// Whether anything has ever been written. Distinguishes "not configured
    /// yet" from "configured and immutable" for namespace-level settings.
    pub fn has_data(&self) -> bool {
        !self.wal.is_empty() || !self.segments.is_empty()
    }

    fn stamp(&mut self, wrote_data: bool) {
        let now = chrono::Utc::now().timestamp_nanos_opt();
        self.created_at = self.created_at.or(now);
        self.updated_at = now;
        if wrote_data {
            self.last_write_at = now;
        }
    }
}

impl Manifest {
    pub fn unindexed_bytes(&self) -> u64 {
        self.wal.iter().map(|e| e.bytes).sum()
    }
    pub fn unindexed_records(&self) -> usize {
        self.wal.iter().map(|e| e.records as usize).sum()
    }
    pub fn segment_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.bytes).sum()
    }
    pub fn index_bytes(&self) -> u64 {
        self.index.as_ref().map_or(0, |i| i.bytes)
    }
    pub fn total_bytes(&self) -> u64 {
        self.unindexed_bytes() + self.segment_bytes() + self.index_bytes()
    }

    /// Every object this manifest keeps alive, as `wal/...` or `data/...`
    /// suffixes. GC is the set difference against what storage actually holds.
    pub fn referenced(&self) -> HashSet<String> {
        let mut out = HashSet::new();
        for e in &self.wal {
            out.insert(format!("wal/{}", e.name));
        }
        for s in &self.segments {
            out.insert(format!("data/{}", s.name));
        }
        if let Some(i) = &self.index {
            out.insert(format!("data/{}", i.centroids));
            for c in &i.clusters {
                out.insert(format!("data/{c}"));
            }
        }
        out
    }

    /// The longest PREFIX of the WAL that fits in `cap` bytes.
    ///
    /// A prefix, never a subset: WAL entries are ordered mutations, so dropping
    /// from the middle or the front would apply a later upsert without the
    /// earlier one it supersedes. Truncating the tail yields a consistent view of
    /// an earlier moment, which is exactly what "eventual" means.
    fn wal_prefix_within(&self, cap: u64) -> (&[WalEntry], bool) {
        let mut acc = 0u64;
        for (i, e) in self.wal.iter().enumerate() {
            if acc + e.bytes > cap {
                return (&self.wal[..i], true);
            }
            acc += e.bytes;
        }
        (&self.wal[..], false)
    }
}

// ---------------------------------------------------------------- metrics

#[derive(Debug, Default)]
pub struct Metrics {
    pub gets: AtomicUsize,
    pub puts: AtomicUsize,
    pub deletes: AtomicUsize,
    pub lists: AtomicUsize,
    pub bytes_get: AtomicU64,
    pub bytes_put: AtomicU64,
    pub cas_conflicts: AtomicUsize,
    pub queries: AtomicUsize,
    pub writes: AtomicUsize,
    pub compactions: AtomicUsize,
    pub backpressure_rejects: AtomicUsize,
}

impl Metrics {
    fn get(&self, bytes: u64) {
        self.gets.fetch_add(1, Ordering::Relaxed);
        self.bytes_get.fetch_add(bytes, Ordering::Relaxed);
    }
    fn put(&self, bytes: u64) {
        self.puts.fetch_add(1, Ordering::Relaxed);
        self.bytes_put.fetch_add(bytes, Ordering::Relaxed);
    }
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            gets: self.gets.load(Ordering::Relaxed),
            puts: self.puts.load(Ordering::Relaxed),
            deletes: self.deletes.load(Ordering::Relaxed),
            lists: self.lists.load(Ordering::Relaxed),
            bytes_get: self.bytes_get.load(Ordering::Relaxed),
            bytes_put: self.bytes_put.load(Ordering::Relaxed),
            cas_conflicts: self.cas_conflicts.load(Ordering::Relaxed),
            queries: self.queries.load(Ordering::Relaxed),
            writes: self.writes.load(Ordering::Relaxed),
            compactions: self.compactions.load(Ordering::Relaxed),
            backpressure_rejects: self.backpressure_rejects.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub gets: usize,
    pub puts: usize,
    pub deletes: usize,
    pub lists: usize,
    pub bytes_get: u64,
    pub bytes_put: u64,
    pub cas_conflicts: usize,
    pub queries: usize,
    pub writes: usize,
    pub compactions: usize,
    pub backpressure_rejects: usize,
}

// ---------------------------------------------------------------- namespace

struct Snapshot {
    manifest: Manifest,
    version: Option<UpdateVersion>,
    at: Instant,
}

pub struct Namespace {
    pub store: Arc<dyn ObjectStore>,
    pub prefix: String,
    cache: Option<Arc<RingCache>>,
    /// Last known commit point, for `Consistency::Eventual`. Never consulted by
    /// writes: a stale manifest there would mean a lost write, not a stale read.
    snapshot: Mutex<Option<Snapshot>>,
    pub metrics: Metrics,
}

impl Namespace {
    pub fn new(store: Arc<dyn ObjectStore>, prefix: impl Into<String>) -> Self {
        Self {
            store,
            prefix: prefix.into(),
            cache: None,
            snapshot: Mutex::new(None),
            metrics: Metrics::default(),
        }
    }

    pub fn with_cache(mut self, cache: Arc<RingCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn cache(&self) -> Option<&Arc<RingCache>> {
        self.cache.as_ref()
    }

    fn manifest_path(&self) -> Path {
        Path::from(format!("{}/manifest", self.prefix))
    }
    fn wal_path(&self, name: &str) -> Path {
        Path::from(format!("{}/wal/{}", self.prefix, name))
    }
    fn data_path(&self, name: &str) -> Path {
        Path::from(format!("{}/data/{}", self.prefix, name))
    }
    fn sub_path(&self, suffix: &str) -> Path {
        Path::from(format!("{}/{}", self.prefix, suffix))
    }

    // ------------------------------------------------------------ manifest io

    /// Fetch the commit point. Always a real roundtrip.
    ///
    /// Never cached, and never served from the snapshot: this is the one mutable
    /// object, and reusing a stale copy here is precisely what eventual
    /// consistency is — a decision that belongs to the caller, not to the loader.
    pub async fn load(&self) -> Result<(Manifest, Option<UpdateVersion>)> {
        match self.store.get(&self.manifest_path()).await {
            Ok(res) => {
                let version = UpdateVersion {
                    e_tag: res.meta.e_tag.clone(),
                    version: res.meta.version.clone(),
                };
                let bytes = res.bytes().await?;
                self.metrics.get(bytes.len() as u64);
                let manifest: Manifest = serde_json::from_slice(&bytes)
                    .context("decoding manifest; format change without migration?")?;
                self.remember(manifest.clone(), Some(version.clone()));
                Ok((manifest, Some(version)))
            }
            Err(OsError::NotFound { .. }) => {
                self.metrics.get(0);
                Ok((Manifest::default(), None))
            }
            Err(e) => Err(e.into()),
        }
    }

    fn remember(&self, manifest: Manifest, version: Option<UpdateVersion>) {
        *self.snapshot.lock().unwrap() = Some(Snapshot { manifest, version, at: Instant::now() });
    }

    fn forget_snapshot(&self) {
        *self.snapshot.lock().unwrap() = None;
    }

    /// The remembered commit point, if any, without touching the network.
    ///
    /// Safe to commit against precisely because CAS validates it. If the
    /// remembered version is stale the conditional PUT fails and the caller
    /// reloads — the guess costs a retry, never correctness.
    fn snapshot_view(&self) -> Option<(Manifest, Option<UpdateVersion>)> {
        let guard = self.snapshot.lock().unwrap();
        let s = guard.as_ref()?;
        // A remembered "namespace does not exist yet" is not useful to commit
        // against: PutMode::Create would be attempted forever if another writer
        // created it. Force a real read in that case.
        s.version.as_ref().map(|v| (s.manifest.clone(), Some(v.clone())))
    }

    /// Resolve a manifest under the requested consistency.
    /// Returns (manifest, served_from_snapshot).
    async fn load_with(&self, consistency: Consistency) -> Result<(Manifest, bool)> {
        if let Consistency::Eventual { max_age_ms } = consistency {
            let cached = {
                let guard = self.snapshot.lock().unwrap();
                guard.as_ref().and_then(|s| {
                    (s.at.elapsed() <= Duration::from_millis(max_age_ms))
                        .then(|| s.manifest.clone())
                })
            };
            if let Some(m) = cached {
                return Ok((m, true));
            }
        }
        let (m, _) = self.load().await?;
        Ok((m, false))
    }

    // ------------------------------------------------------------ object io

    /// Fetch an immutable object, through the cache when one is attached.
    async fn read_immutable(&self, path: &Path) -> Result<Bytes> {
        let key = path.as_ref();
        if let Some(cache) = &self.cache
            && let Some(hit) = cache.get(key)
        {
            return Ok(hit);
        }
        let bytes = self.store.get(path).await?.bytes().await?;
        self.metrics.get(bytes.len() as u64);
        if let Some(cache) = &self.cache {
            cache.put(key, &bytes);
        }
        Ok(bytes)
    }

    async fn put_object(&self, path: &Path, body: Bytes) -> Result<()> {
        let n = body.len() as u64;
        self.store.put(path, PutPayload::from(body)).await?;
        self.metrics.put(n);
        Ok(())
    }

    async fn read_records(&self, path: &Path) -> Result<Vec<Record>> {
        unframe(&self.read_immutable(path).await?)
            .with_context(|| format!("unframing {path}"))?
            .iter()
            .map(|b| Record::decode(b))
            .collect()
    }

    /// Fetch many objects concurrently. One "roundtrip" in the sense that
    /// matters: wall clock is the slowest fetch, not their sum.
    async fn read_records_parallel(&self, paths: Vec<Path>) -> Result<Vec<Record>> {
        let mut out = Vec::new();
        for r in futures::future::join_all(paths.iter().map(|p| self.read_records(p))).await {
            out.extend(r?);
        }
        Ok(out)
    }

    /// Load the attribute indexes a filter actually reads. Nothing else is
    /// fetched, and these objects hold no vectors, so they are far smaller than
    /// the segment they describe.
    async fn load_attr_indexes(
        &self,
        idx: &IndexMeta,
        filter: &Filter,
    ) -> Result<BTreeMap<String, AttrIndex>> {
        let mut wanted = Vec::new();
        filter.keys(&mut wanted);
        wanted.sort();
        wanted.dedup();

        let targets: Vec<(String, Path)> = wanted
            .into_iter()
            .filter_map(|k| idx.attributes.get(&k).map(|n| (k, self.data_path(n))))
            .collect();

        let blobs =
            futures::future::join_all(targets.iter().map(|(_, p)| self.read_immutable(p))).await;

        let mut out = BTreeMap::new();
        for ((key, _), blob) in targets.into_iter().zip(blobs) {
            out.insert(key, AttrIndex::decode(&blob?)?);
        }
        Ok(out)
    }

    async fn load_ids(&self, idx: &IndexMeta) -> Result<Vec<Id>> {
        let Some(name) = &idx.ids else { return Ok(vec![]) };
        let blob = self.read_immutable(&self.data_path(name)).await?;
        let mut pos = 0usize;
        let mut out = Vec::with_capacity(idx.docs);
        while pos < blob.len() {
            out.push(Id::decode(&blob, &mut pos)?);
        }
        Ok(out)
    }

    // ------------------------------------------------------------ writes

    /// Write one WAL object holding every entry in `batch`, then advance the
    /// commit point via CAS. One object, one CAS, regardless of batch size.
    pub async fn commit_batch(&self, batch: &[Bytes]) -> Result<(u64, usize)> {
        let blob = frame(batch);
        let entry_bytes = blob.len() as u64;
        let entry_records = batch.len() as u32;

        // First attempt commits against the remembered version, skipping the
        // manifest GET. With one committer per namespace that guess is right
        // essentially always, which drops a commit from three requests
        // (GET manifest, PUT wal, CAS manifest) to two — and removes a full
        // object storage roundtrip from write latency. A wrong guess is caught by
        // the CAS and costs one retry.
        let mut optimistic = true;

        for attempt in 1..=MAX_CAS_ATTEMPTS {
            let (mut manifest, version) = match optimistic.then(|| self.snapshot_view()).flatten() {
                Some(v) => v,
                None => self.load().await?,
            };
            let seq = manifest.next_seq;

            // Unique name per attempt. Not decoration: if the name were derived
            // purely from `seq`, two writers that both observe seq=N would write
            // the SAME path and the later would silently overwrite the earlier.
            // The CAS winner then claims a file holding the loser's bytes, and a
            // write vanishes with no error.
            let name = format!("{seq:010}-{}.bin", uuid::Uuid::new_v4());

            // Data is written BEFORE the commit. A failed CAS leaves an orphan —
            // garbage, not corruption, because readers only follow the manifest.
            // Reclaimed by `gc`.
            self.put_object(&self.wal_path(&name), blob.clone()).await?;

            manifest.next_seq += 1;
            manifest.wal.push(WalEntry {
                name,
                bytes: entry_bytes,
                records: entry_records,
            });
            manifest.stamp(true);

            let mode = match version {
                Some(v) => PutMode::Update(v),
                None => PutMode::Create,
            };
            let body = serde_json::to_vec(&manifest)?;
            let n = body.len() as u64;
            match self
                .store
                .put_opts(
                    &self.manifest_path(),
                    PutPayload::from(body),
                    PutOptions { mode, ..Default::default() },
                )
                .await
            {
                Ok(res) => {
                    self.metrics.put(n);
                    self.metrics.writes.fetch_add(entry_records as usize, Ordering::Relaxed);
                    // Remember the manifest we just wrote, with the version the
                    // store handed back. An eventual-consistency read on this
                    // node is then immediately fresh rather than one age-out
                    // behind, which is why "most updates are visible
                    // immediately" holds in practice.
                    self.remember(
                        manifest,
                        Some(UpdateVersion { e_tag: res.e_tag, version: res.version }),
                    );
                    return Ok((seq, attempt));
                }
                Err(OsError::Precondition { .. }) | Err(OsError::AlreadyExists { .. }) => {
                    self.metrics.cas_conflicts.fetch_add(1, Ordering::Relaxed);
                    // Whatever we held is now known stale. Drop it and read for
                    // real, or the retry repeats the same losing guess.
                    self.forget_snapshot();
                    optimistic = false;
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
        bail!("CAS contention: failed to commit after {MAX_CAS_ATTEMPTS} attempts")
    }

    pub async fn append(&self, data: Bytes) -> Result<(u64, usize)> {
        self.commit_batch(&[data]).await
    }

    pub async fn write_records(&self, records: &[Record]) -> Result<(u64, usize)> {
        self.commit_records(records, &WriteConfig::default()).await
    }

    /// Commit records with schema inference and enforcement.
    ///
    /// Coercion happens inside the CAS loop, not before it: a lost race means
    /// another writer may have declared a type we have not seen, so the batch has
    /// to be re-checked against the manifest we actually commit onto.
    pub async fn commit_records(
        &self,
        records: &[Record],
        config: &WriteConfig,
    ) -> Result<(u64, usize)> {
        let mut optimistic = true;

        for attempt in 1..=MAX_CAS_ATTEMPTS {
            let (mut manifest, version) = match optimistic.then(|| self.snapshot_view()).flatten() {
                Some(v) => v,
                None => self.load().await?,
            };

            let mut batch = records.to_vec();
            let mut schema = manifest.schema.clone();
            if let Some(m) = config.metric {
                schema.set_metric(m, manifest.has_data())?;
            }
            // Declared types go in before inference runs, so `absorb` coerces
            // against them instead of inferring something looser. One mechanism,
            // not a second coercion pass in the adapter.
            schema.declare(&config.declared_types)?;
            schema.declare_fts(&config.declared_fts)?;
            schema.absorb(&mut batch)?;

            let blob = frame_records(&batch);
            let seq = manifest.next_seq;
            let name = format!("{seq:010}-{}.bin", uuid::Uuid::new_v4());
            self.put_object(&self.wal_path(&name), blob.clone()).await?;

            manifest.next_seq += 1;
            manifest.wal.push(WalEntry {
                name,
                bytes: blob.len() as u64,
                records: batch.len() as u32,
            });
            manifest.schema = schema;
            manifest.stamp(true);

            let mode = match version {
                Some(v) => PutMode::Update(v),
                None => PutMode::Create,
            };
            let body = serde_json::to_vec(&manifest)?;
            let n = body.len() as u64;
            match self
                .store
                .put_opts(
                    &self.manifest_path(),
                    PutPayload::from(body),
                    PutOptions { mode, ..Default::default() },
                )
                .await
            {
                Ok(res) => {
                    self.metrics.put(n);
                    self.metrics.writes.fetch_add(batch.len(), Ordering::Relaxed);
                    self.remember(
                        manifest,
                        Some(UpdateVersion { e_tag: res.e_tag, version: res.version }),
                    );
                    return Ok((seq, attempt));
                }
                Err(OsError::Precondition { .. }) | Err(OsError::AlreadyExists { .. }) => {
                    self.metrics.cas_conflicts.fetch_add(1, Ordering::Relaxed);
                    self.forget_snapshot();
                    optimistic = false;
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
        bail!("CAS contention: failed to commit after {MAX_CAS_ATTEMPTS} attempts")
    }

    // ------------------------------------------------------------ reads

    /// Every COMMITTED record, in commit order: segments first, then WAL.
    pub async fn all_records(&self) -> Result<Vec<Record>> {
        let (m, _) = self.load().await?;
        let mut out = Vec::new();
        for s in &m.segments {
            out.extend(self.read_records(&self.data_path(&s.name)).await?);
        }
        for e in &m.wal {
            out.extend(self.read_records(&self.wal_path(&e.name)).await?);
        }
        Ok(out)
    }

    /// Apply records in order: later upserts win, tombstones remove.
    pub fn materialize(records: Vec<Record>) -> HashMap<Id, Doc> {
        let mut live: HashMap<Id, Doc> = HashMap::new();
        for r in records {
            apply(&mut live, r);
        }
        live
    }

    /// Exhaustive scan. Correct by construction, and the oracle the index is
    /// measured against.
    pub async fn query_brute(
        &self,
        vector: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Hit>> {
        let (m, _) = self.load().await?;
        let mut owned;
        let filter = match filter {
            Some(f) => {
                owned = f.clone();
                owned.coerce(&m.schema.attributes)?;
                Some(&owned)
            }
            None => None,
        };
        let live = Self::materialize(self.all_records().await?);
        Ok(crate::doc::top_k(
            live.values(),
            vector,
            k,
            filter,
            &Include::None,
            m.schema.distance_metric,
            &m.schema.fts,
        ))
    }

    /// The real query path.
    ///
    /// Roundtrip shape on a cold cache, mirroring turbopuffer's:
    ///   1. manifest                      (skipped entirely under Eventual)
    ///   2. centroids + unindexed WAL     (concurrent)
    ///   3. the probed cluster objects    (concurrent)
    pub async fn query(&self, req: &QueryRequest) -> Result<QueryResponse> {
        let t = Instant::now();
        self.metrics.queries.fetch_add(1, Ordering::Relaxed);
        let gets_before = self.metrics.gets.load(Ordering::Relaxed);
        let hits_before = self.cache.as_ref().map_or(0, |c| c.stats().0);

        let (m, from_snapshot) = self.load_with(req.consistency).await?;

        // Filter literals arrive as JSON, which cannot express datetime or uuid.
        // Coercing them against the schema is what makes a typed filter match
        // anything at all.
        let mut filter = req.filter.clone();
        if let Some(f) = filter.as_mut() {
            f.coerce(&m.schema.attributes)?;
        }

        // Decide how much of the unindexed tail to read.
        let unindexed_total = m.unindexed_bytes();
        let (wal_entries, truncated) = m.wal_prefix_within(MAX_UNINDEXED_SCAN_BYTES);
        if truncated && req.consistency == Consistency::Strong {
            bail!(
                "unindexed WAL is {unindexed_total} bytes, over the {MAX_UNINDEXED_SCAN_BYTES} \
                 byte scan cap; a strongly consistent answer is not available until compaction \
                 catches up. Retry with eventual consistency or compact this namespace."
            );
        }
        let wal_paths: Vec<Path> =
            wal_entries.iter().map(|e| self.wal_path(&e.name)).collect();
        let unindexed_records: usize = wal_entries.iter().map(|e| e.records as usize).sum();
        let unindexed_bytes: u64 = wal_entries.iter().map(|e| e.bytes).sum();

        // Aggregations scan the documents matching the filter. They compose with
        // ranking rather than replacing it: a request carrying both gets rows AND
        // totals, which is how a faceted search page is built in one round trip.
        let (aggregations, aggregation_groups) = match &req.aggregate_by {
            None => (BTreeMap::new(), vec![]),
            Some(aggs) => {
                let mut paths: Vec<Path> =
                    m.segments.iter().map(|s| self.data_path(&s.name)).collect();
                paths.extend(wal_entries.iter().map(|e| self.wal_path(&e.name)));
                let live = Self::materialize(self.read_records_parallel(paths).await?);
                let matching: Vec<&Doc> = live
                    .values()
                    .filter(|d| filter.as_ref().is_none_or(|f| f.matches(&d.attrs, &m.schema.fts)))
                    .collect();
                crate::aggregate::aggregate(matching.into_iter(), aggs, &req.group_by)?
            }
        };

        // A request that only aggregates has nothing to rank.
        let ranking_requested =
            !req.vector.is_empty() || req.text.is_some() || req.order_by.is_some();
        if req.aggregate_by.is_some() && !ranking_requested {
            return Ok(QueryResponse {
                hits: vec![],
                consistent: !from_snapshot && !truncated,
                indexed: false,
                prefiltered: false,
                ordered: false,
                aggregations,
                aggregation_groups,
                unindexed_records,
                unindexed_bytes,
                object_gets: self.metrics.gets.load(Ordering::Relaxed) - gets_before,
                cache_hits: self.cache.as_ref().map_or(0, |c| c.stats().0) - hits_before,
                took_ms: t.elapsed().as_millis() as u64,
            });
        }

        // Full-text ranking: score from the term index, then score the tail on
        // the same scale so recent writes remain rankable.
        if let Some(text) = &req.text {
            let Some(idx) = &m.index else {
                bail!(
                    "full-text search needs an index; namespace has not been compacted yet"
                );
            };
            let Some(object) = idx.fts.get(&text.attribute) else {
                bail!(
                    "attribute {:?} is not enabled for full-text search",
                    text.attribute
                );
            };
            let fts = FtsIndex::decode(&self.read_immutable(&self.data_path(object)).await?)?;

            let seg_paths: Vec<Path> =
                m.segments.iter().map(|s| self.data_path(&s.name)).collect();
            let mut wal_paths: Vec<Path> =
                wal_entries.iter().map(|e| self.wal_path(&e.name)).collect();
            let seg_records = self.read_records_parallel(seg_paths).await?;
            let tail = self.read_records_parallel(std::mem::take(&mut wal_paths)).await?;

            // Current state of every document, so a tombstoned or rewritten
            // document is never scored from what the index remembers.
            let mut live: HashMap<Id, Doc> = HashMap::new();
            for r in &seg_records {
                apply(&mut live, r.clone());
            }
            for r in &tail {
                apply(&mut live, r.clone());
            }

            // Documents the tail touched: their text may differ from what the
            // index saw, so their stored postings are stale. Rescored below.
            let touched: std::collections::HashSet<Id> =
                tail.iter().map(|r| r.id().clone()).collect();

            let mut hits: Vec<Hit> = Vec::new();
            let matches = |doc: &Doc| {
                filter.as_ref().is_none_or(|f| f.matches(&doc.attrs, &m.schema.fts))
            };

            // Indexed documents the tail left alone: use the stored score.
            for (ordinal, score) in fts.score(&text.query) {
                let Some(Record::Upsert(indexed)) = seg_records.get(ordinal as usize) else {
                    continue;
                };
                if touched.contains(&indexed.id) {
                    continue;
                }
                let Some(doc) = live.get(&indexed.id) else { continue };
                if matches(doc) {
                    hits.push(Hit {
                        id: doc.id.clone(),
                        score,
                        attrs: req.include_attributes.project(&doc.attrs),
                    });
                }
            }

            // Everything the tail touched, scored against the index's corpus
            // statistics so it lands on the same scale.
            for id in &touched {
                let Some(doc) = live.get(id) else { continue };
                let Some(t) =
                    doc.attrs.get(&text.attribute).and_then(crate::fts::attribute_text)
                else {
                    continue;
                };
                let score = fts.score_text(&t, &text.query);
                if score > 0.0 && matches(doc) {
                    hits.push(Hit {
                        id: doc.id.clone(),
                        score,
                        attrs: req.include_attributes.project(&doc.attrs),
                    });
                }
            }

            hits.sort_unstable_by(|a, b| {
                b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id))
            });
            hits.truncate(req.top_k);

            return Ok(QueryResponse {
                hits,
                consistent: !from_snapshot && !truncated,
                indexed: true,
                prefiltered: false,
                ordered: false,
                aggregations: aggregations.clone(),
                aggregation_groups: aggregation_groups.clone(),
                unindexed_records,
                unindexed_bytes,
                object_gets: self.metrics.gets.load(Ordering::Relaxed) - gets_before,
                cache_hits: self.cache.as_ref().map_or(0, |c| c.stats().0) - hits_before,
                took_ms: t.elapsed().as_millis() as u64,
            });
        }

        // Ordering by attribute has no vector to probe with, so it is always a
        // scan of the candidate set.
        if let Some(order) = &req.order_by {
            let mut paths: Vec<Path> =
                m.segments.iter().map(|s| self.data_path(&s.name)).collect();
            paths.extend(wal_entries.iter().map(|e| self.wal_path(&e.name)));
            let live = Self::materialize(self.read_records_parallel(paths).await?);
            let hits = crate::doc::order_by(
                live.values(),
                order,
                req.top_k,
                filter.as_ref(),
                &req.include_attributes,
                &m.schema.fts,
            );
            return Ok(QueryResponse {
                hits,
                consistent: !from_snapshot && !truncated,
                indexed: false,
                prefiltered: false,
                ordered: true,
                aggregations: aggregations.clone(),
                aggregation_groups: aggregation_groups.clone(),
                unindexed_records,
                unindexed_bytes,
                object_gets: self.metrics.gets.load(Ordering::Relaxed) - gets_before,
                cache_hits: self.cache.as_ref().map_or(0, |c| c.stats().0) - hits_before,
                took_ms: t.elapsed().as_millis() as u64,
            });
        }

        // Selective filters take an exact path. Probing clusters and then
        // filtering can return fewer than top_k results, or none, when every
        // surviving document sits in a cluster the query never probed — so for a
        // filter the index can bound, scan the bounded set instead.
        let mut prefiltered = false;
        if let (Some(idx), Some(f)) = (&m.index, filter.as_ref())
            && idx.ids.is_some()
        {
            let indexes = self.load_attr_indexes(idx, f).await?;
            if let Some(sel) = attrindex::evaluate(f, &indexes, idx.docs as u32)
                && should_prefilter(sel.ordinals.len(), idx.docs, idx.clusters.len(), req.nprobe)
            {
                prefiltered = true;
                let seg_paths: Vec<Path> =
                    m.segments.iter().map(|s| self.data_path(&s.name)).collect();
                let seg_records = self.read_records_parallel(seg_paths).await?;

                // Segment order is ordinal order: compaction sorts by id.
                let mut candidates = HashMap::new();
                for ordinal in &sel.ordinals {
                    if let Some(Record::Upsert(d)) = seg_records.get(*ordinal as usize) {
                        candidates.insert(d.id.clone(), d.clone());
                    }
                }
                // The tail still overrides: it may add matches the index never
                // saw, or remove ones it did.
                for r in wal_records_for_overlay(&wal_entries, self, &m).await? {
                    apply(&mut candidates, r);
                }

                let hits = crate::doc::top_k(
                    candidates.values(),
                    &req.vector,
                    req.top_k,
                    filter.as_ref(),
                    &req.include_attributes,
                    m.schema.distance_metric,
                    &m.schema.fts,
                );
                return Ok(QueryResponse {
                    hits,
                    consistent: !from_snapshot && !truncated,
                    indexed: true,
                    prefiltered,
                    ordered: false,
                    aggregations: aggregations.clone(),
                    aggregation_groups: aggregation_groups.clone(),
                    unindexed_records,
                    unindexed_bytes,
                    object_gets: self.metrics.gets.load(Ordering::Relaxed) - gets_before,
                    cache_hits: self.cache.as_ref().map_or(0, |c| c.stats().0) - hits_before,
                    took_ms: t.elapsed().as_millis() as u64,
                });
            }
        }

        let (hits, indexed) = match &m.index {
            None => {
                // No index: fall back to scanning every segment plus the tail.
                let mut paths: Vec<Path> =
                    m.segments.iter().map(|s| self.data_path(&s.name)).collect();
                paths.extend(wal_paths);
                let live = Self::materialize(self.read_records_parallel(paths).await?);
                (
                    crate::doc::top_k(
                        live.values(),
                        &req.vector,
                        req.top_k,
                        filter.as_ref(),
                        &req.include_attributes,
                        m.schema.distance_metric,
                        &m.schema.fts,
                    ),
                    false,
                )
            }
            Some(idx) => {
                if idx.dim != req.vector.len() {
                    bail!("query has {} dims, index has {}", req.vector.len(), idx.dim);
                }
                let centroid_path = self.data_path(&idx.centroids);
                let (centroid_blob, wal_records) = tokio::try_join!(
                    self.read_immutable(&centroid_path),
                    self.read_records_parallel(wal_paths),
                )?;
                let centroids = index::decode_centroids(&centroid_blob, idx.dim)?;

                let probed = index::probe(&centroids, &req.vector, req.nprobe, m.schema.distance_metric);
                let cluster_paths: Vec<Path> = probed
                    .iter()
                    .filter_map(|i| idx.clusters.get(*i))
                    .map(|n| self.data_path(n))
                    .collect();
                let cluster_records = self.read_records_parallel(cluster_paths).await?;

                // Indexed candidates first, then the WAL overlay so recent
                // upserts and tombstones win over whatever the index believes.
                let mut candidates = Self::materialize(cluster_records);
                for r in wal_records {
                    apply(&mut candidates, r);
                }
                (
                    crate::doc::top_k(
                        candidates.values(),
                        &req.vector,
                        req.top_k,
                        filter.as_ref(),
                        &req.include_attributes,
                        m.schema.distance_metric,
                        &m.schema.fts,
                    ),
                    true,
                )
            }
        };

        Ok(QueryResponse {
            hits,
            // Honest reporting: an answer is consistent only if it came from a
            // fresh commit point AND read the whole tail.
            consistent: !from_snapshot && !truncated,
            indexed,
            prefiltered,
            ordered: false,
            aggregations,
            aggregation_groups,
            unindexed_records,
            unindexed_bytes,
            object_gets: self.metrics.gets.load(Ordering::Relaxed) - gets_before,
            cache_hits: self.cache.as_ref().map_or(0, |c| c.stats().0) - hits_before,
            took_ms: t.elapsed().as_millis() as u64,
        })
    }

    /// Document ids matching `filter`, capped, plus whether more remain.
    ///
    /// ponytail: full scan of the live set. Selection is resolved against a
    /// snapshot taken now, so a document written concurrently may be missed — the
    /// same bound any capped filter-write has, and why callers reissue until
    /// `rows_remaining` is false. The attribute index (phase 17) is what makes
    /// this cheap rather than what makes it correct.
    pub async fn ids_matching(&self, filter: &Filter, cap: usize) -> Result<(Vec<Id>, bool)> {
        let (m, _) = self.load().await?;
        let mut filter = filter.clone();
        filter.coerce(&m.schema.attributes)?;

        // The fast path needs both an exact answer AND an empty tail: the index
        // describes the segment only, so an unindexed WAL entry could have
        // patched a document into or out of the match set. When the tail is
        // empty the index is the whole truth, and this touches no vectors at all.
        if m.wal.is_empty()
            && let Some(idx) = &m.index
            && idx.ids.is_some()
        {
            let indexes = self.load_attr_indexes(idx, &filter).await?;
            if let Some(sel) = attrindex::evaluate(&filter, &indexes, idx.docs as u32)
                && sel.exact
            {
                let ids = self.load_ids(idx).await?;
                let mut matched: Vec<Id> = sel
                    .ordinals
                    .iter()
                    .filter_map(|o| ids.get(*o as usize).cloned())
                    .collect();
                matched.sort();
                let remaining = matched.len() > cap;
                matched.truncate(cap);
                return Ok((matched, remaining));
            }
        }

        let live = Self::materialize(self.all_records().await?);
        let mut matched: Vec<Id> = live
            .values()
            .filter(|d| filter.matches(&d.attrs, &m.schema.fts))
            .map(|d| d.id.clone())
            .collect();
        // Deterministic order, so a reissued request makes progress through the
        // set rather than re-picking an arbitrary subset.
        matched.sort();
        let remaining = matched.len() > cap;
        matched.truncate(cap);
        Ok((matched, remaining))
    }

    // ------------------------------------------------------------ compaction

    /// Fold every WAL object into one segment, rebuild the index, and commit.
    ///
    /// ponytail: full rewrite, not leveled compaction, and the index is rebuilt
    /// wholesale rather than maintained incrementally. Cost is O(iters*n*k*dim)
    /// and it needs the live set in memory. Measured at 60s for 20k docs, so it
    /// is minutes at a million. Leveled compaction plus SPFresh-style LIRE
    /// split/merge is the upgrade path, and it is a large one.
    pub async fn compact(&self, build_index: bool) -> Result<CompactResponse> {
        let t = Instant::now();
        self.metrics.compactions.fetch_add(1, Ordering::Relaxed);

        let (m, _) = self.load().await?;
        if m.wal.is_empty() && (m.index.is_some() || !build_index) {
            return Ok(CompactResponse {
                records_in: 0,
                docs_out: m.segments.iter().map(|s| s.docs as usize).sum(),
                wal_consumed: 0,
                clusters: m.index.as_ref().map_or(0, |i| i.clusters.len()),
                cas_attempts: 0,
                took_ms: t.elapsed().as_millis() as u64,
            });
        }

        let consumed: Vec<String> = m.wal.iter().map(|e| e.name.clone()).collect();
        let consumed_segments = m.segments.clone();
        let records = self.all_records().await?;
        let records_in = records.len();
        let live = Self::materialize(records);

        let mut docs: Vec<Doc> = live.into_values().collect();
        // Deterministic segment and index contents for a given live set.
        docs.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        let docs_out = docs.len();

        let mut new_segments = Vec::new();
        let mut new_index = None;
        let mut clusters_written = 0usize;

        if !docs.is_empty() {
            let seg_name = format!("{}.bin", uuid::Uuid::new_v4());
            let seg_records: Vec<Record> = docs.iter().cloned().map(Record::Upsert).collect();
            let seg_blob = frame_records(&seg_records);
            let seg_bytes = seg_blob.len() as u64;
            self.put_object(&self.data_path(&seg_name), seg_blob).await?;
            new_segments.push(SegmentEntry {
                name: seg_name,
                bytes: seg_bytes,
                docs: docs_out as u32,
            });

            if build_index {
                let params = IvfParams::for_docs(docs.len()).with_metric(m.schema.distance_metric);
                let (flat, groups) = index::build(&docs, params)?;

                let cen_name = format!("{}.cen", uuid::Uuid::new_v4());
                let cen_blob = Bytes::from(index::encode_centroids(&flat));
                let mut index_bytes = cen_blob.len() as u64;
                self.put_object(&self.data_path(&cen_name), cen_blob).await?;

                let cluster_names: Vec<String> =
                    groups.iter().map(|_| format!("{}.clu", uuid::Uuid::new_v4())).collect();
                let cluster_paths: Vec<Path> =
                    cluster_names.iter().map(|n| self.data_path(n)).collect();
                let blobs: Vec<Bytes> = groups
                    .iter()
                    .map(|g| {
                        frame_records(
                            &g.iter().cloned().map(Record::Upsert).collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                index_bytes += blobs.iter().map(|b| b.len() as u64).sum::<u64>();

                // Concurrent: dozens of small PUTs at ~300ms each would otherwise
                // make compaction latency the dominant cost.
                let writes = cluster_paths
                    .iter()
                    .zip(&blobs)
                    .map(|(p, b)| self.put_object(p, b.clone()));
                for r in futures::future::join_all(writes).await {
                    r?;
                }

                // Ordinal -> id, so attribute indexes can address documents
                // compactly and still be resolved back to real ids.
                let ids_name = format!("{}.ids", uuid::Uuid::new_v4());
                let ids_blob = {
                    let mut b = BytesMut::new();
                    for doc in &docs {
                        doc.id.encode(&mut b);
                    }
                    b.freeze()
                };
                index_bytes += ids_blob.len() as u64;
                self.put_object(&self.data_path(&ids_name), ids_blob).await?;

                // One inverted index per declared attribute.
                let mut attr_names = std::collections::BTreeMap::new();
                let mut attr_writes = Vec::new();
                for attribute in m.schema.attributes.keys() {
                    let built = AttrIndex::build(docs.iter().enumerate().filter_map(|(o, d)| {
                        d.attrs.get(attribute).map(|v| (o as u32, v.clone()))
                    }));
                    if built.is_empty() {
                        continue;
                    }
                    let name = format!("{}.att", uuid::Uuid::new_v4());
                    let blob = built.encode();
                    index_bytes += blob.len() as u64;
                    attr_names.insert(attribute.clone(), name.clone());
                    attr_writes.push((self.data_path(&name), blob));
                }
                for r in futures::future::join_all(
                    attr_writes.iter().map(|(p, b)| self.put_object(p, b.clone())),
                )
                .await
                {
                    r?;
                }

                // One full-text index per attribute the schema enables.
                let mut fts_names = std::collections::BTreeMap::new();
                let mut fts_writes = Vec::new();
                for (attribute, config) in &m.schema.fts {
                    let built = FtsIndex::build(
                        docs.iter().enumerate().filter_map(|(o, d)| {
                            d.attrs
                                .get(attribute)
                                .and_then(crate::fts::attribute_text)
                                .map(|t| (o as u32, t))
                        }).collect::<Vec<_>>().iter().map(|(o, t)| (*o, t.as_str())),
                        docs.len(),
                        config,
                    );
                    if built.is_empty() {
                        continue;
                    }
                    let name = format!("{}.fts", uuid::Uuid::new_v4());
                    let blob = built.encode();
                    index_bytes += blob.len() as u64;
                    fts_names.insert(attribute.clone(), name.clone());
                    fts_writes.push((self.data_path(&name), blob));
                }
                for r in futures::future::join_all(
                    fts_writes.iter().map(|(p, b)| self.put_object(p, b.clone())),
                )
                .await
                {
                    r?;
                }

                clusters_written = cluster_names.len();
                new_index = Some(IndexMeta {
                    dim: docs[0].vector.len(),
                    centroids: cen_name,
                    clusters: cluster_names,
                    docs: docs.len(),
                    bytes: index_bytes,
                    ids: Some(ids_name),
                    attributes: attr_names,
                    fts: fts_names,
                });
            }
        }

        // Commit. The objects above are already durable, so a lost CAS only means
        // recomputing the manifest delta, not redoing the work.
        for attempt in 1..=MAX_CAS_ATTEMPTS {
            let (mut current, version) = self.load().await?;
            if current.segments != consumed_segments {
                bail!("another compaction committed concurrently; discarding this one");
            }
            // Keep WAL entries that arrived while we were compacting.
            current.wal.retain(|e| !consumed.contains(&e.name));
            current.segments = new_segments.clone();
            current.index = new_index.clone();
            current.stamp(false);

            let mode = match version {
                Some(v) => PutMode::Update(v),
                None => PutMode::Create,
            };
            let body = serde_json::to_vec(&current)?;
            let n = body.len() as u64;
            match self
                .store
                .put_opts(
                    &self.manifest_path(),
                    PutPayload::from(body),
                    PutOptions { mode, ..Default::default() },
                )
                .await
            {
                Ok(res) => {
                    self.metrics.put(n);
                    self.remember(
                        current,
                        Some(UpdateVersion { e_tag: res.e_tag, version: res.version }),
                    );
                    return Ok(CompactResponse {
                        records_in,
                        docs_out,
                        wal_consumed: consumed.len(),
                        clusters: clusters_written,
                        cas_attempts: attempt,
                        took_ms: t.elapsed().as_millis() as u64,
                    });
                }
                Err(OsError::Precondition { .. }) | Err(OsError::AlreadyExists { .. }) => {
                    self.metrics.cas_conflicts.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
        bail!("compaction could not commit after {MAX_CAS_ATTEMPTS} attempts")
    }

    // ------------------------------------------------------------ operations

    pub async fn metadata(&self) -> Result<NamespaceMetadata> {
        let (m, _) = self.load().await?;
        Ok(self.metadata_from(&m))
    }

    pub fn metadata_from(&self, m: &Manifest) -> NamespaceMetadata {
        NamespaceMetadata {
            namespace: self.prefix.clone(),
            indexed_docs: m.index.as_ref().map_or(0, |i| i.docs),
            clusters: m.index.as_ref().map_or(0, |i| i.clusters.len()),
            segments: m.segments.len(),
            segment_bytes: m.segment_bytes(),
            wal_entries: m.wal.len(),
            unindexed_records: m.unindexed_records(),
            unindexed_bytes: m.unindexed_bytes(),
            index_bytes: m.index_bytes(),
            total_bytes: m.total_bytes(),
            dim: m.schema.dim.or_else(|| m.index.as_ref().map(|i| i.dim)),
            write_backpressure: m.unindexed_bytes() >= MAX_UNINDEXED_SCAN_BYTES,
            schema: m.schema.attributes.clone(),
            id_type: m.schema.id_type,
            distance_metric: m.schema.distance_metric,
        }
    }

    /// Pull the index into the local cache so the next query does not pay for it.
    ///
    /// This is the primitive behind turbopuffer's cache-warm hint: an application
    /// that knows a user is about to search (search box focused, dialog opened)
    /// can spend the cold latency before the user notices it.
    pub async fn warm(&self) -> Result<WarmResponse> {
        let t = Instant::now();
        let (m, _) = self.load().await?;
        let mut paths = Vec::new();
        if let Some(idx) = &m.index {
            paths.push(self.data_path(&idx.centroids));
            paths.extend(idx.clusters.iter().map(|c| self.data_path(c)));
        } else {
            paths.extend(m.segments.iter().map(|s| self.data_path(&s.name)));
        }

        let already = self
            .cache
            .as_ref()
            .map_or(0, |c| paths.iter().filter(|p| c.get(p.as_ref()).is_some()).count());

        let mut bytes = 0u64;
        for r in futures::future::join_all(paths.iter().map(|p| self.read_immutable(p))).await {
            bytes += r?.len() as u64;
        }
        Ok(WarmResponse {
            objects_warmed: paths.len(),
            bytes,
            already_cached: already,
            took_ms: t.elapsed().as_millis() as u64,
        })
    }

    /// Delete unreferenced objects.
    ///
    /// GC is a set difference because every data object is immutable and named
    /// uniquely: anything storage holds that the manifest does not name is dead.
    ///
    /// Two things produce orphans, and the second is the common one:
    ///   - a writer that died between its PUT and its CAS
    ///   - compaction, which retires WAL objects from the manifest without
    ///     deleting them
    /// So this is not an error-recovery path that rarely runs. It is the second
    /// half of compaction, and a namespace that never runs it grows forever while
    /// its manifest stays small.
    ///
    /// The grace window is the one subtlety and it is not optional. A freshly
    /// written WAL object is unreferenced for the moment between its PUT and its
    /// CAS. Deleting inside that window would destroy a write that is about to be
    /// committed — GC racing the writer. Anything younger than `grace` is left
    /// alone, so the window only has to exceed a single commit's duration.
    pub async fn gc(&self, grace: Duration) -> Result<GcResponse> {
        let t = Instant::now();
        let (m, _) = self.load().await?;
        let referenced = m.referenced();

        let cutoff = chrono::Utc::now()
            - chrono::Duration::from_std(grace).unwrap_or(chrono::Duration::zero());

        let mut scanned = 0usize;
        let mut spared_recent = 0usize;
        let mut doomed = Vec::new();

        for sub in ["wal", "data"] {
            let prefix = self.sub_path(sub);
            self.metrics.lists.fetch_add(1, Ordering::Relaxed);
            let mut listing = self.store.list(Some(&prefix));
            while let Some(meta) = listing.next().await {
                let meta = meta?;
                scanned += 1;
                let Some(name) = meta.location.as_ref().strip_prefix(&format!("{}/", self.prefix))
                else {
                    continue;
                };
                if referenced.contains(name) {
                    continue;
                }
                if meta.last_modified > cutoff {
                    spared_recent += 1;
                    continue;
                }
                doomed.push(meta.location);
            }
        }

        let deleted = doomed.len();
        for r in futures::future::join_all(doomed.iter().map(|p| self.store.delete(p))).await {
            r?;
        }
        self.metrics.deletes.fetch_add(deleted, Ordering::Relaxed);

        Ok(GcResponse {
            scanned,
            referenced: referenced.len(),
            deleted,
            spared_recent,
            took_ms: t.elapsed().as_millis() as u64,
        })
    }

    /// Copy this namespace to `dest_prefix` as an independent point-in-time snapshot.
    ///
    /// ponytail: server-side copies every referenced object, so cost is O(bytes).
    /// An O(1) branch is possible — the objects are immutable, so two manifests
    /// could simply name the same ones — but that makes GC a reference-counting
    /// problem across namespaces, and a wrong refcount deletes live data. Copying
    /// keeps GC a local set difference. Revisit if branching large namespaces
    /// becomes a common operation.
    pub async fn branch(&self, dest_prefix: &str) -> Result<usize> {
        let (m, _) = self.load().await?;
        if self.store.head(&Path::from(format!("{dest_prefix}/manifest"))).await.is_ok() {
            bail!("destination namespace {dest_prefix} already exists");
        }

        let objects: Vec<String> = m.referenced().into_iter().collect();
        let copies = objects.iter().map(|name| {
            let from = self.sub_path(name);
            let to = Path::from(format!("{dest_prefix}/{name}"));
            async move { self.store.copy(&from, &to).await }
        });
        for r in futures::future::join_all(copies).await {
            r?;
        }

        // Manifest last: until it exists the destination is not a namespace, so a
        // failure part-way leaves garbage rather than a half-readable database.
        let body = serde_json::to_vec(&m)?;
        self.store
            .put_opts(
                &Path::from(format!("{dest_prefix}/manifest")),
                PutPayload::from(body),
                PutOptions { mode: PutMode::Create, ..Default::default() },
            )
            .await?;
        Ok(objects.len())
    }

    /// Delete the namespace and everything under it.
    ///
    /// Manifest first: it is the entry point, so removing it makes the namespace
    /// gone from a reader's perspective immediately, and whatever remains is
    /// unreferenced garbage that `gc` can finish off if this call is interrupted.
    pub async fn destroy(&self) -> Result<usize> {
        let _ = self.store.delete(&self.manifest_path()).await;
        let mut doomed = Vec::new();
        let prefix = Path::from(self.prefix.clone());
        let mut listing = self.store.list(Some(&prefix));
        while let Some(meta) = listing.next().await {
            doomed.push(meta?.location);
        }
        let n = doomed.len();
        for r in futures::future::join_all(doomed.iter().map(|p| self.store.delete(p))).await {
            r?;
        }
        self.metrics.deletes.fetch_add(n, Ordering::Relaxed);
        *self.snapshot.lock().unwrap() = None;
        Ok(n)
    }

    // ------------------------------------------------------------ probes

    /// Negative control: does this backend actually ENFORCE compare-and-swap?
    ///
    /// Run before trusting any store. A backend that accepts a stale If-Match
    /// silently degrades the commit protocol to last-write-wins and loses
    /// concurrent writes with no error anywhere.
    pub async fn verify_cas(&self) -> Result<()> {
        let path = self.manifest_path();
        // Distinct content per write. This matters: on S3-compatible stores the
        // ETag is derived from content, so re-PUTting identical bytes yields the
        // SAME ETag and a "stale" version still compares equal — the probe would
        // pass against a backend that enforces nothing.
        let body = |seq: u64| {
            PutPayload::from(
                serde_json::to_vec(&Manifest { next_seq: seq, ..Default::default() }).unwrap(),
            )
        };

        self.store.put(&path, body(1)).await?;
        let (_, version) = self.load().await?;
        let version = version.context("object vanished immediately after PUT")?;

        self.store
            .put_opts(
                &path,
                body(2),
                PutOptions { mode: PutMode::Update(version.clone()), ..Default::default() },
            )
            .await
            .context("CAS against a fresh version was rejected")?;

        match self
            .store
            .put_opts(
                &path,
                body(3),
                PutOptions { mode: PutMode::Update(version), ..Default::default() },
            )
            .await
        {
            Err(OsError::Precondition { .. }) => Ok(()),
            Ok(_) => bail!(
                "BACKEND DOES NOT ENFORCE CAS: a stale If-Match was accepted. \
                 The commit protocol is unsafe here — do not build on it."
            ),
            Err(e) => Err(e.into()),
        }
    }

    pub async fn count_objects(&self, sub: &str) -> Result<usize> {
        let prefix = self.sub_path(sub);
        self.metrics.lists.fetch_add(1, Ordering::Relaxed);
        Ok(self.store.list(Some(&prefix)).filter_map(|r| async { r.ok() }).count().await)
    }

    pub fn fetch_count(&self) -> usize {
        self.metrics.gets.load(Ordering::Relaxed)
    }
}

/// Read the WAL entries a query is allowed to scan.
async fn wal_records_for_overlay(
    entries: &[WalEntry],
    ns: &Namespace,
    _m: &Manifest,
) -> Result<Vec<Record>> {
    let paths: Vec<Path> = entries.iter().map(|e| ns.wal_path(&e.name)).collect();
    ns.read_records_parallel(paths).await
}

/// Apply one record to a live set. The single definition of mutation semantics,
/// shared by compaction and by the query-time WAL overlay — two copies of this
/// would eventually disagree about what a patch means.
fn apply(live: &mut HashMap<Id, Doc>, r: Record) {
    match r {
        Record::Upsert(d) => {
            live.insert(d.id.clone(), d);
        }
        Record::Delete(id) => {
            live.remove(&id);
        }
        Record::Patch { id, attrs } => {
            // A patch on an absent document is a no-op, not a resurrection: there
            // is no vector to attach, so the document would be unqueryable.
            if let Some(doc) = live.get_mut(&id) {
                for (k, v) in attrs {
                    if v.is_null() {
                        doc.attrs.remove(&k);
                    } else {
                        doc.attrs.insert(k, v);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- group commit

/// Namespace-level settings a write may carry.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WriteConfig {
    /// Settable only on a namespace's first write.
    pub metric: Option<DistanceMetric>,
    /// Client-declared attribute types, for the types JSON cannot express.
    pub declared_types: BTreeMap<String, Type>,
    /// Attributes to enable for full-text search.
    pub declared_fts: crate::fts::FtsSchema,
}

impl WriteConfig {
    pub fn is_empty(&self) -> bool {
        self.metric.is_none() && self.declared_types.is_empty() && self.declared_fts.is_empty()
    }
}

/// One queued write: the record, any namespace-level settings, and
/// where to report the commit.
///
/// Records rather than encoded bytes, because schema inference has to see the
/// values before they are framed — and it has to see the whole coalesced batch,
/// so two documents in one commit cannot disagree about a type.
type Req = (Record, WriteConfig, oneshot::Sender<Result<u64, String>>);

/// Funnels every write for one namespace through a single committer.
///
/// There is deliberately no timer. The loop blocks for one request, drains
/// whatever else is already queued, and commits the lot. While a commit is in
/// flight arrivals pile up and ride the next one, so batch size self-tunes to the
/// backend's latency with nothing to configure. A fixed window would be more code
/// and worse: it adds latency when idle and still under-coalesces when busy.
///
/// The deeper point: a single committer never races itself, so CAS is never
/// contested. "One writer per namespace" is not a limitation of this design — it
/// is the thing that makes CAS-instead-of-consensus work.
pub struct GroupCommit {
    tx: mpsc::UnboundedSender<Req>,
    batches: Arc<AtomicUsize>,
    attempts: Arc<AtomicUsize>,
}

impl GroupCommit {
    pub fn new(ns: Arc<Namespace>) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<Req>();
        let batches = Arc::new(AtomicUsize::new(0));
        let attempts = Arc::new(AtomicUsize::new(0));
        let (b, a) = (batches.clone(), attempts.clone());

        tokio::spawn(async move {
            while let Some(first) = rx.recv().await {
                let mut waiters = vec![first];
                let mut bytes = waiters[0].0.encoded_len();
                while waiters.len() < MAX_BATCH_LEN && bytes < MAX_BATCH_BYTES {
                    match rx.try_recv() {
                        Ok(item) => {
                            bytes += item.0.encoded_len();
                            waiters.push(item);
                        }
                        Err(_) => break,
                    }
                }

                let records: Vec<Record> = waiters.iter().map(|(r, _, _)| r.clone()).collect();
                // A namespace-level setting only has to be requested once; the
                // first request in the batch carries it and the rest agree by
                // construction, since a conflicting value is rejected by the
                // schema rather than silently applied.
                let config = waiters
                    .iter()
                    .map(|(_, c, _)| c)
                    .find(|c| !c.is_empty())
                    .cloned()
                    .unwrap_or_default();
                b.fetch_add(1, Ordering::Relaxed);
                let result = ns.commit_records(&records, &config).await;
                if let Ok((_, n)) = &result {
                    a.fetch_add(*n, Ordering::Relaxed);
                }

                let reply = result.map(|(seq, _)| seq).map_err(|e| e.to_string());
                for (_, _, done) in waiters {
                    let _ = done.send(reply.clone());
                }
            }
        });

        Self { tx, batches, attempts }
    }

    pub async fn submit(&self, record: Record, config: WriteConfig) -> Result<u64> {
        let (done, wait) = oneshot::channel();
        self.tx
            .send((record, config, done))
            .map_err(|_| anyhow::anyhow!("committer stopped"))?;
        wait.await.context("committer dropped the request")?.map_err(|e| anyhow::anyhow!(e))
    }

    pub async fn upsert(&self, doc: Doc) -> Result<u64> {
        self.submit(Record::Upsert(doc), WriteConfig::default()).await
    }

    pub async fn delete(&self, id: Id) -> Result<u64> {
        self.submit(Record::Delete(id), WriteConfig::default()).await
    }

    /// Submit many records and wait for all of them, so a caller's whole batch
    /// rides one commit instead of one commit each.
    pub async fn write_all(
        &self,
        records: Vec<Record>,
        config: WriteConfig,
    ) -> Result<u64> {
        let waits: Vec<_> = records
            .into_iter()
            .map(|r| {
                let (done, wait) = oneshot::channel();
                self.tx
                    .send((r, config.clone(), done))
                    .map_err(|_| anyhow::anyhow!("committer stopped"))?;
                Ok::<_, anyhow::Error>(wait)
            })
            .collect::<Result<_>>()?;

        let mut seq = 0;
        for w in waits {
            seq = w.await.context("committer dropped a request")?.map_err(|e| anyhow::anyhow!(e))?;
        }
        Ok(seq)
    }

    /// (batches, CAS attempts)
    pub fn stats(&self) -> (usize, usize) {
        (self.batches.load(Ordering::Relaxed), self.attempts.load(Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------- backend

/// Pick a backend from the environment. One code path for MinIO / R2 / S3.
///
///   FCKDB_BUCKET            required; falls back to InMemory when unset
///   FCKDB_ENDPOINT          MinIO: http://127.0.0.1:9000
///                           R2:    https://{account_id}.r2.cloudflarestorage.com
///   AWS_ACCESS_KEY_ID       read by from_env()
///   AWS_SECRET_ACCESS_KEY   read by from_env()
///
/// There is deliberately no local-filesystem option: object_store's
/// LocalFileSystem does not implement `PutMode::Update`, so the commit protocol
/// cannot work on it. Verified — `verify_cas` rejects it outright. Use InMemory
/// for tests, or MinIO for a local backend that does enforce CAS.
pub fn open_store() -> Result<(Arc<dyn ObjectStore>, String)> {
    let Ok(bucket) = std::env::var("FCKDB_BUCKET") else {
        return Ok((Arc::new(object_store::memory::InMemory::new()), "InMemory".into()));
    };

    use object_store::aws::{AmazonS3Builder, S3ConditionalPut};
    let mut builder = AmazonS3Builder::from_env()
        .with_bucket_name(&bucket)
        .with_region("auto")
        // Must be explicit: this tells object_store the backend supports CAS via
        // If-Match on ETag. Without it, PutMode::Update is not used and the whole
        // commit protocol degrades to last-write-wins.
        .with_conditional_put(S3ConditionalPut::ETagMatch);

    let mut label = format!("S3 bucket={bucket}");
    if let Ok(endpoint) = std::env::var("FCKDB_ENDPOINT") {
        let http = endpoint.starts_with("http://");
        label = format!("{endpoint} bucket={bucket}");
        builder = builder.with_endpoint(endpoint).with_allow_http(http);
    }
    Ok((Arc::new(builder.build()?), label))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::Filter;
    use crate::doc::Id;
    use crate::value::Value;
    use crate::wire::Consistency;

    fn mem() -> Arc<dyn ObjectStore> {
        Arc::new(object_store::memory::InMemory::new())
    }

    fn ns(prefix: &str) -> Arc<Namespace> {
        Arc::new(Namespace::new(mem(), prefix))
    }

    fn tmp_cache() -> Arc<RingCache> {
        let p = std::env::temp_dir().join(format!("fckdb-t-{}", uuid::Uuid::new_v4()));
        Arc::new(RingCache::open(p, 64 << 20).unwrap())
    }

    /// Deterministic clustered vectors, the way real embeddings sit: grouped, not
    /// uniform. Uniform random data makes any IVF index look bad and tells you
    /// nothing about production recall.
    fn synth(n: usize, dim: usize, clusters: usize) -> Vec<Doc> {
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };
        let centers: Vec<Vec<f32>> =
            (0..clusters).map(|_| (0..dim).map(|_| rng() * 10.0).collect()).collect();
        (0..n)
            .map(|i| {
                let c = &centers[i % clusters];
                let v = c.iter().map(|x| x + rng() * 0.5).collect();
                Doc::new(i as u64, v).with_attr("tenant", if i % 3 == 0 { "a" } else { "b" })
            })
            .collect()
    }

    /// Integer ids, so assertions read as `vec![1, 2]`. Every test namespace here
    /// uses integer ids.
    fn ids(hits: &[Hit]) -> Vec<u64> {
        hits.iter().map(|h| h.id.as_uint().expect("test ids are integers")).collect()
    }

    // -------------------------------------------------------- phase 3: queries

    #[tokio::test]
    async fn upsert_and_query_brute() {
        let ns = ns("t/basic");
        ns.write_records(&[
            Record::Upsert(Doc::new(1u64, vec![1.0, 0.0])),
            Record::Upsert(Doc::new(2u64, vec![0.9, 0.1])),
            Record::Upsert(Doc::new(3u64, vec![0.0, 1.0])),
        ])
        .await
        .unwrap();
        let got = ns.query_brute(&[1.0, 0.0], 2, None).await.unwrap();
        assert_eq!(ids(&got), vec![1, 2]);
    }

    #[tokio::test]
    async fn later_upsert_wins_and_tombstone_removes() {
        let ns = ns("t/mutate");
        ns.write_records(&[Record::Upsert(Doc::new(1u64, vec![1.0, 0.0]))]).await.unwrap();
        ns.write_records(&[Record::Upsert(Doc::new(2u64, vec![0.0, 1.0]))]).await.unwrap();
        ns.write_records(&[Record::Upsert(Doc::new(1u64, vec![-1.0, 0.0]))]).await.unwrap();

        let got = ns.query_brute(&[1.0, 0.0], 3, None).await.unwrap();
        assert_eq!(got[0].id.as_uint().unwrap(), 2, "the newer version of doc 1 was not applied");

        ns.write_records(&[Record::Delete(Id::Uint(2))]).await.unwrap();
        let got = ns.query_brute(&[1.0, 0.0], 3, None).await.unwrap();
        assert!(!ids(&got).contains(&2), "tombstone did not remove doc 2");
        assert_eq!(got.len(), 1);
    }

    // -------------------------------------------------------- phase 4: compaction

    #[tokio::test]
    async fn compaction_preserves_the_live_set() {
        let ns = ns("t/compact");
        for d in synth(30, 8, 3) {
            ns.write_records(&[Record::Upsert(d)]).await.unwrap();
        }
        for i in 0..15u64 {
            ns.write_records(&[Record::Upsert(Doc::new(i, vec![9.0; 8]))]).await.unwrap();
        }
        for i in 20..30u64 {
            ns.write_records(&[Record::Delete(Id::Uint(i))]).await.unwrap();
        }

        let before = ns.query_brute(&[9.0; 8], 5, None).await.unwrap();
        let stats = ns.compact(false).await.unwrap();

        assert_eq!(stats.docs_out, 20, "live set should be 30 minus 10 deleted");
        assert!(stats.records_in > stats.docs_out, "compaction should drop dead records");
        assert_eq!(stats.wal_consumed, 55);

        let (m, _) = ns.load().await.unwrap();
        assert!(m.wal.is_empty(), "WAL should be empty after compaction");
        assert_eq!(m.segments.len(), 1);
        assert_eq!(m.unindexed_bytes(), 0);

        let after = ns.query_brute(&[9.0; 8], 5, None).await.unwrap();
        assert_eq!(before, after, "compaction changed query results");
    }

    #[tokio::test]
    async fn writes_arriving_during_compaction_survive() {
        let ns = ns("t/concurrent-compact");
        for d in synth(10, 4, 2) {
            ns.write_records(&[Record::Upsert(d)]).await.unwrap();
        }
        let (m, _) = ns.load().await.unwrap();
        let before = m.wal.len();
        ns.write_records(&[Record::Upsert(Doc::new(999u64, vec![1.0; 4]))]).await.unwrap();

        let stats = ns.compact(false).await.unwrap();
        assert_eq!(stats.wal_consumed, before + 1);

        let live = Namespace::materialize(ns.all_records().await.unwrap());
        assert!(live.contains_key(&Id::Uint(999)), "the concurrent write was lost");
        assert_eq!(live.len(), 11);
    }

    // -------------------------------------------------------- phase 5-6: index

    #[tokio::test]
    async fn indexed_query_matches_brute_force() {
        let ns = ns("t/index");
        let docs = synth(600, 16, 12);
        ns.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        let stats = ns.compact(true).await.unwrap();
        assert!(stats.clusters > 1, "index should produce multiple clusters");

        let (m, _) = ns.load().await.unwrap();
        let idx = m.index.expect("index should exist after compact(true)");
        assert_eq!(idx.docs, 600);
        assert!(idx.bytes > 0, "index byte accounting missing");

        let mut total = 0.0;
        for q in docs.iter().take(20) {
            let exact = ns.query_brute(&q.vector, 10, None).await.unwrap();
            let res = ns.query(&QueryRequest::new(q.vector.clone())).await.unwrap();
            assert!(res.indexed, "index was not used");
            assert!(res.consistent, "a strong query on a fresh manifest must be consistent");
            assert_eq!(res.hits.len(), 10);
            assert_eq!(res.hits[0].id, q.id, "a doc must be its own nearest neighbour");
            total += index::recall(&exact, &res.hits);
        }
        let recall = total / 20.0;
        assert!(recall > 0.9, "recall@10 was {recall}, expected > 0.9 on clustered data");
    }

    #[tokio::test]
    async fn indexed_query_sees_the_unindexed_wal() {
        let ns = ns("t/overlay");
        let docs = synth(200, 8, 4);
        ns.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        ns.compact(true).await.unwrap();
        let target = docs[0].vector.clone();

        ns.write_records(&[Record::Upsert(Doc::new(9999u64, target.clone()))]).await.unwrap();
        let res = ns.query(&QueryRequest::new(target.clone()).top_k(3).nprobe(4)).await.unwrap();
        assert!(ids(&res.hits).contains(&9999), "indexed query missed a doc still in the WAL");
        assert_eq!(res.unindexed_records, 1);
        assert!(res.unindexed_bytes > 0);

        ns.write_records(&[Record::Delete(docs[0].id.clone())]).await.unwrap();
        let res = ns.query(&QueryRequest::new(target).top_k(3).nprobe(4)).await.unwrap();
        assert!(
            !ids(&res.hits).contains(&docs[0].id.as_uint().unwrap()),
            "WAL tombstone did not suppress an indexed doc"
        );
    }

    #[tokio::test]
    async fn filters_apply_to_indexed_queries() {
        let ns = ns("t/filter");
        let docs = synth(300, 8, 6);
        ns.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        ns.compact(true).await.unwrap();

        let res = ns
            .query(&QueryRequest::new(docs[0].vector.clone()).filter(Filter::eq("tenant", "a")))
            .await
            .unwrap();
        assert!(!res.hits.is_empty());
        let live = Namespace::materialize(ns.all_records().await.unwrap());
        for h in &res.hits {
            assert_eq!(
                live[&h.id].attrs["tenant"],
                Value::from("a"),
                "filter leaked a foreign tenant"
            );
        }
    }

    #[tokio::test]
    async fn patch_merges_attributes_and_null_removes_them() {
        let ns = ns("t/patch");
        ns.write_records(&[Record::Upsert(
            Doc::new(1u64, vec![1.0, 0.0]).with_attr("tenant", "acme").with_attr("count", 1u64),
        )])
        .await
        .unwrap();

        ns.write_records(&[Record::Patch {
            id: Id::Uint(1),
            attrs: std::collections::BTreeMap::from([
                ("count".to_string(), Value::Uint(9)),
                ("added".to_string(), Value::Bool(true)),
            ]),
        }])
        .await
        .unwrap();

        let live = Namespace::materialize(ns.all_records().await.unwrap());
        let d = &live[&Id::Uint(1)];
        assert_eq!(d.attrs["count"], Value::Uint(9), "patch did not overwrite");
        assert_eq!(d.attrs["added"], Value::Bool(true), "patch did not add");
        assert_eq!(d.attrs["tenant"], Value::from("acme"), "patch clobbered an untouched attribute");
        assert_eq!(d.vector, vec![1.0, 0.0], "patch disturbed the vector");

        // Null removes.
        ns.write_records(&[Record::Patch {
            id: Id::Uint(1),
            attrs: std::collections::BTreeMap::from([("tenant".to_string(), Value::Null)]),
        }])
        .await
        .unwrap();
        let live = Namespace::materialize(ns.all_records().await.unwrap());
        assert!(!live[&Id::Uint(1)].attrs.contains_key("tenant"), "null did not remove the attribute");

        // A patch against a document that does not exist is a no-op, never a
        // resurrection — there would be no vector to attach.
        ns.write_records(&[Record::Patch {
            id: Id::Uint(999),
            attrs: std::collections::BTreeMap::from([("x".to_string(), Value::Uint(1))]),
        }])
        .await
        .unwrap();
        let live = Namespace::materialize(ns.all_records().await.unwrap());
        assert!(!live.contains_key(&Id::Uint(999)), "patch resurrected a missing document");
    }

    #[tokio::test]
    async fn patch_survives_compaction_and_applies_over_the_index() {
        let ns = ns("t/patch-compact");
        let docs = synth(100, 8, 4);
        ns.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        ns.compact(true).await.unwrap();

        // Patch lands only in the WAL, so it must be applied as an overlay on top
        // of what the index still believes.
        let target = docs[0].clone();
        ns.write_records(&[Record::Patch {
            id: target.id.clone(),
            attrs: std::collections::BTreeMap::from([("tenant".to_string(), Value::from("z"))]),
        }])
        .await
        .unwrap();

        let res = ns
            .query(
                &QueryRequest::new(target.vector.clone())
                    .top_k(1)
                    .include(crate::doc::Include::All),
            )
            .await
            .unwrap();
        assert_eq!(res.hits[0].id, target.id);
        assert_eq!(res.hits[0].attrs["tenant"], Value::from("z"), "patch not visible over index");

        // After compaction the patch is folded in and the result is unchanged.
        ns.compact(true).await.unwrap();
        let after = ns
            .query(
                &QueryRequest::new(target.vector.clone())
                    .top_k(1)
                    .include(crate::doc::Include::All),
            )
            .await
            .unwrap();
        assert_eq!(after.hits[0].attrs["tenant"], Value::from("z"), "compaction dropped the patch");
        let (m, _) = ns.load().await.unwrap();
        assert!(m.wal.is_empty());
    }

    #[tokio::test]
    async fn typed_range_filters_work_through_the_index() {
        let ns = ns("t/typed-filter");
        let base = crate::value::parse_datetime("2024-03-01T00:00:00Z").unwrap();
        let day = 86_400_000_000_000i64;
        let docs: Vec<Doc> = (0..60)
            .map(|i| {
                Doc::new(i as u64, vec![1.0 + i as f32 * 0.001, 0.0])
                    .with_attr("when", Value::Datetime(base + i as i64 * day))
                    .with_attr("rank", Value::Uint(i as u64))
                    .with_attr("tags", Value::StringArray(vec![format!("t{}", i % 3)]))
            })
            .collect();
        ns.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        ns.compact(true).await.unwrap();

        let q = vec![1.0f32, 0.0];
        let cutoff = Value::Datetime(base + 30 * day);

        let res = ns
            .query(
                &QueryRequest::new(q.clone())
                    .top_k(100)
                    .nprobe(64)
                    .filter(Filter::cmp("when", crate::doc::Op::Gte, cutoff.clone())),
            )
            .await
            .unwrap();
        assert!(!res.hits.is_empty());
        assert!(
            res.hits.iter().all(|h| h.id.as_uint().unwrap() >= 30),
            "Gte on datetime leaked older docs"
        );

        let res = ns
            .query(
                &QueryRequest::new(q.clone()).top_k(100).nprobe(64).filter(Filter::And(vec![
                    Filter::cmp("rank", crate::doc::Op::Gte, 10u64),
                    Filter::cmp("rank", crate::doc::Op::Lt, 20u64),
                    Filter::cmp("tags", crate::doc::Op::Contains, "t1"),
                ])),
            )
            .await
            .unwrap();
        assert!(!res.hits.is_empty());
        assert!(
            res.hits.iter().all(|h| {
                let n = h.id.as_uint().unwrap();
                (10..20).contains(&n) && n % 3 == 1
            }),
            "compound typed filter admitted the wrong documents: {:?}",
            ids(&res.hits)
        );
    }

    #[tokio::test]
    async fn include_attributes_projects_through_the_query_path() {
        let ns = ns("t/include");
        ns.write_records(&[Record::Upsert(
            Doc::new(1u64, vec![1.0, 0.0])
                .with_attr("a", 1u64)
                .with_attr("b", "two")
                .with_attr("c", true),
        )])
        .await
        .unwrap();
        ns.compact(true).await.unwrap();
        let q = vec![1.0f32, 0.0];

        let none = ns.query(&QueryRequest::new(q.clone())).await.unwrap();
        assert!(none.hits[0].attrs.is_empty(), "attributes returned without being asked for");

        let all = ns
            .query(&QueryRequest::new(q.clone()).include(crate::doc::Include::All))
            .await
            .unwrap();
        assert_eq!(all.hits[0].attrs.len(), 3);
        assert_eq!(all.hits[0].attrs["b"], Value::from("two"));

        let some = ns
            .query(
                &QueryRequest::new(q)
                    .include(crate::doc::Include::Only(vec!["a".into(), "nope".into()])),
            )
            .await
            .unwrap();
        assert_eq!(some.hits[0].attrs.len(), 1, "projection leaked or invented an attribute");
        assert_eq!(some.hits[0].attrs["a"], Value::Uint(1));
    }

    #[tokio::test]
    async fn query_without_index_falls_back_to_a_scan() {
        let ns = ns("t/noindex");
        let docs = synth(50, 4, 2);
        ns.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        ns.compact(false).await.unwrap();

        let res = ns.query(&QueryRequest::new(docs[0].vector.clone())).await.unwrap();
        assert!(!res.indexed, "reported an index that was never built");
        assert_eq!(res.hits[0].id, docs[0].id);
    }

    #[tokio::test]
    async fn dimension_mismatch_is_rejected_not_silently_wrong() {
        let ns = ns("t/dims");
        ns.write_records(&[Record::Upsert(Doc::new(1u64, vec![1.0; 8]))]).await.unwrap();
        ns.compact(true).await.unwrap();
        assert!(ns.query(&QueryRequest::new(vec![1.0; 4])).await.is_err());
    }

    // -------------------------------------------------------- phase 17: attr index

    /// The failure this exists to prevent: a selective filter whose surviving
    /// documents are spread across clusters the query never probes. Post-filtering
    /// returns almost nothing; the attribute index makes the answer exact.
    #[tokio::test]
    async fn a_selective_filter_no_longer_loses_recall() {
        let ns = ns("t/prefilter-recall");
        // 800 documents across 20 well-separated clusters. One in forty is
        // "rare", and those are deliberately spread across every cluster.
        let docs: Vec<Record> = synth(800, 16, 20)
            .into_iter()
            .enumerate()
            .map(|(i, mut d)| {
                d.attrs.insert(
                    "tier".into(),
                    Value::from(if i % 40 == 0 { "rare" } else { "common" }),
                );
                Record::Upsert(d)
            })
            .collect();
        ns.write_records(&docs).await.unwrap();
        ns.compact(true).await.unwrap();

        let (m, _) = ns.load().await.unwrap();
        let idx = m.index.clone().unwrap();
        assert!(idx.attributes.contains_key("tier"), "no attribute index was built");
        assert!(idx.ids.is_some(), "no ordinal-to-id map was written");

        let q = match &docs[0] {
            Record::Upsert(d) => d.vector.clone(),
            _ => unreachable!(),
        };

        // nprobe=1 probes a single cluster, so post-filtering would find at most
        // the one or two rare documents that happen to live in it.
        let res = ns
            .query(
                &QueryRequest::new(q.clone())
                    .top_k(10)
                    .nprobe(1)
                    .filter(Filter::eq("tier", "rare")),
            )
            .await
            .unwrap();
        assert!(res.prefiltered, "the selective filter did not take the exact path");
        assert_eq!(res.hits.len(), 10, "prefiltered query returned short: {:?}", res.hits.len());

        // And it agrees exactly with brute force, which is the definition of no
        // recall loss.
        let exact = ns.query_brute(&q, 10, Some(&Filter::eq("tier", "rare"))).await.unwrap();
        assert_eq!(res.hits, exact, "prefiltered result disagreed with brute force");
        assert_eq!(crate::index::recall(&exact, &res.hits), 1.0);

        // A non-selective filter is left to the cluster path, which is what the
        // threshold is for.
        let broad = ns
            .query(
                &QueryRequest::new(q)
                    .top_k(10)
                    .nprobe(8)
                    .filter(Filter::eq("tier", "common")),
            )
            .await
            .unwrap();
        assert!(!broad.prefiltered, "a broad filter should not force an exact scan");
        assert!(!broad.hits.is_empty());
    }

    #[tokio::test]
    async fn ids_matching_uses_the_index_and_touches_no_vectors() {
        let store = mem();
        let seed = Namespace::new(store.clone(), "t/ids-index");
        let docs: Vec<Record> = synth(400, 32, 8)
            .into_iter()
            .enumerate()
            .map(|(i, mut d)| {
                d.attrs.insert("tier".into(), Value::from(if i % 4 == 0 { "gold" } else { "grey" }));
                Record::Upsert(d)
            })
            .collect();
        seed.write_records(&docs).await.unwrap();
        seed.compact(true).await.unwrap();

        // Fresh namespace handle so byte counters start clean.
        let ns = Namespace::new(store.clone(), "t/ids-index");
        let (ids, remaining) = ns.ids_matching(&Filter::eq("tier", "gold"), 1000).await.unwrap();
        assert_eq!(ids.len(), 100);
        assert!(!remaining);
        let indexed_bytes = ns.metrics.snapshot().bytes_get;

        // Compare against the scan path, which has to pull every vector.
        let scanner = Namespace::new(store, "t/ids-index");
        let live = Namespace::materialize(scanner.all_records().await.unwrap());
        let scanned: Vec<Id> = {
            let mut v: Vec<Id> = live
                .values()
                .filter(|d| Filter::eq("tier", "gold").matches(&d.attrs, &crate::fts::FtsSchema::new()))
                .map(|d| d.id.clone())
                .collect();
            v.sort();
            v
        };
        let scan_bytes = scanner.metrics.snapshot().bytes_get;

        assert_eq!(ids, scanned, "the index disagreed with the scan");
        assert!(
            indexed_bytes * 4 < scan_bytes,
            "index path read {indexed_bytes} bytes vs scan {scan_bytes}; it should avoid vectors"
        );
    }

    #[tokio::test]
    async fn an_unindexed_tail_forces_the_scan_path() {
        let ns = ns("t/ids-tail");
        let docs: Vec<Record> = (0..40u64)
            .map(|i| {
                Record::Upsert(
                    Doc::new(i, vec![1.0, i as f32])
                        .with_attr("tier", if i < 20 { "gold" } else { "grey" }),
                )
            })
            .collect();
        ns.write_records(&docs).await.unwrap();
        ns.compact(true).await.unwrap();

        // A patch in the tail moves a document out of the match set. The index
        // describes the segment only, so trusting it here would return a stale id.
        ns.write_records(&[Record::Patch {
            id: Id::Uint(0),
            attrs: std::collections::BTreeMap::from([("tier".to_string(), Value::from("grey"))]),
        }])
        .await
        .unwrap();

        let (ids, _) = ns.ids_matching(&Filter::eq("tier", "gold"), 100).await.unwrap();
        assert_eq!(ids.len(), 19, "the index was trusted over an unindexed patch");
        assert!(!ids.contains(&Id::Uint(0)), "returned a document the tail had moved out");

        // And a document patched INTO the set is found.
        ns.write_records(&[Record::Patch {
            id: Id::Uint(25),
            attrs: std::collections::BTreeMap::from([("tier".to_string(), Value::from("gold"))]),
        }])
        .await
        .unwrap();
        let (ids, _) = ns.ids_matching(&Filter::eq("tier", "gold"), 100).await.unwrap();
        assert!(ids.contains(&Id::Uint(25)), "missed a document the tail moved in");
        assert_eq!(ids.len(), 20);
    }

    #[tokio::test]
    async fn prefiltered_queries_still_honour_the_tail() {
        let ns = ns("t/prefilter-tail");
        let docs: Vec<Record> = synth(200, 8, 4)
            .into_iter()
            .enumerate()
            .map(|(i, mut d)| {
                d.attrs.insert("tier".into(), Value::from(if i % 50 == 0 { "rare" } else { "x" }));
                Record::Upsert(d)
            })
            .collect();
        ns.write_records(&docs).await.unwrap();
        ns.compact(true).await.unwrap();
        let q = match &docs[0] {
            Record::Upsert(d) => d.vector.clone(),
            _ => unreachable!(),
        };

        // A brand-new rare document lives only in the tail, so the attribute
        // index has never seen it.
        ns.write_records(&[Record::Upsert(
            Doc::new(999_999u64, q.clone()).with_attr("tier", "rare"),
        )])
        .await
        .unwrap();

        let res = ns
            .query(&QueryRequest::new(q.clone()).top_k(5).nprobe(1).filter(Filter::eq("tier", "rare")))
            .await
            .unwrap();
        assert!(res.prefiltered);
        assert!(
            ids(&res.hits).contains(&999_999),
            "prefiltered query missed a document that is only in the tail"
        );

        // A tombstone in the tail must suppress an indexed match.
        let victim = match &docs[0] {
            Record::Upsert(d) => d.id.clone(),
            _ => unreachable!(),
        };
        ns.write_records(&[Record::Delete(victim.clone())]).await.unwrap();
        let res = ns
            .query(&QueryRequest::new(q).top_k(5).nprobe(1).filter(Filter::eq("tier", "rare")))
            .await
            .unwrap();
        assert!(
            !res.hits.iter().any(|h| h.id == victim),
            "prefiltered query returned a tombstoned document"
        );
    }

    // -------------------------------------------------------- phase 19: BM25

    async fn text_ns(prefix: &str) -> Arc<Namespace> {
        let ns = ns(prefix);
        let cfg = WriteConfig {
            declared_fts: crate::fts::FtsSchema::from([(
                "body".to_string(),
                crate::fts::FtsConfig::default(),
            )]),
            ..Default::default()
        };
        let docs = vec![
            Record::Upsert(
                Doc::new(1u64, vec![1.0, 0.0])
                    .with_attr("body", "the quick brown fox jumps over the lazy dog")
                    .with_attr("tier", "gold"),
            ),
            Record::Upsert(
                Doc::new(2u64, vec![0.9, 0.1])
                    .with_attr("body", "a quick brown dog")
                    .with_attr("tier", "grey"),
            ),
            Record::Upsert(
                Doc::new(3u64, vec![0.0, 1.0])
                    .with_attr("body", "unrelated notes about databases")
                    .with_attr("tier", "gold"),
            ),
        ];
        ns.commit_records(&docs, &cfg).await.unwrap();
        ns.compact(true).await.unwrap();
        ns
    }

    #[tokio::test]
    async fn bm25_ranks_and_excludes_non_matches() {
        let ns = text_ns("t/bm25").await;
        let (m, _) = ns.load().await.unwrap();
        assert!(
            m.index.as_ref().unwrap().fts.contains_key("body"),
            "compaction did not build a full-text index"
        );

        let res = ns.query(&QueryRequest::new(vec![]).text("body", "quick fox").top_k(10)).await.unwrap();
        assert_eq!(ids(&res.hits), vec![1, 2], "BM25 returned the wrong set");
        assert!(res.hits[0].score > res.hits[1].score, "scores not ordered");
        // A document with neither term is absent, not scored zero.
        assert!(!ids(&res.hits).contains(&3));

        // Filters compose with text ranking.
        let res = ns
            .query(
                &QueryRequest::new(vec![])
                    .text("body", "quick")
                    .top_k(10)
                    .filter(Filter::eq("tier", "gold")),
            )
            .await
            .unwrap();
        assert_eq!(ids(&res.hits), vec![1]);
    }

    #[tokio::test]
    async fn bm25_ranks_documents_that_are_only_in_the_tail() {
        let ns = text_ns("t/bm25-tail").await;

        // A new document, more relevant than anything indexed, exists only in the
        // WAL. If it were unrankable it would be invisible to every text query
        // until the next compaction.
        ns.write_records(&[Record::Upsert(
            Doc::new(99u64, vec![1.0, 0.0])
                .with_attr("body", "quick fox quick fox")
                .with_attr("tier", "gold"),
        )])
        .await
        .unwrap();

        let res = ns.query(&QueryRequest::new(vec![]).text("body", "quick fox").top_k(10)).await.unwrap();
        assert!(ids(&res.hits).contains(&99), "a tail-only document was not ranked");
        assert_eq!(res.hits[0].id, Id::Uint(99), "the most relevant document did not rank first");

        // A patch that removes the terms must drop the document, not leave it
        // ranked on what the index remembers.
        ns.write_records(&[Record::Patch {
            id: Id::Uint(1),
            attrs: std::collections::BTreeMap::from([(
                "body".to_string(),
                Value::from("nothing relevant here"),
            )]),
        }])
        .await
        .unwrap();
        let res = ns.query(&QueryRequest::new(vec![]).text("body", "quick fox").top_k(10)).await.unwrap();
        assert!(
            !ids(&res.hits).contains(&1),
            "a document rewritten by the tail was still scored from stale postings"
        );

        // And a tombstone removes it entirely.
        ns.write_records(&[Record::Delete(Id::Uint(2))]).await.unwrap();
        let res = ns.query(&QueryRequest::new(vec![]).text("body", "quick").top_k(10)).await.unwrap();
        assert!(!ids(&res.hits).contains(&2), "a tombstoned document was still ranked");
    }

    #[tokio::test]
    async fn full_text_search_needs_configuration_and_an_index() {
        let ns = ns("t/bm25-unconfigured");
        ns.write_records(&[Record::Upsert(
            Doc::new(1u64, vec![1.0]).with_attr("body", "quick fox"),
        )])
        .await
        .unwrap();

        // Before compaction there is no term index at all.
        let err = ns
            .query(&QueryRequest::new(vec![]).text("body", "quick"))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("compacted"), "unhelpful error: {err}");

        // After compaction, an attribute nobody enabled is still refused by name.
        ns.compact(true).await.unwrap();
        let err = ns
            .query(&QueryRequest::new(vec![]).text("body", "quick"))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("full-text"), "unhelpful error: {err}");
    }

    #[tokio::test]
    async fn a_tokenizer_cannot_be_changed_under_an_existing_index() {
        let ns = text_ns("t/bm25-tokenizer").await;
        let different = WriteConfig {
            declared_fts: crate::fts::FtsSchema::from([(
                "body".to_string(),
                crate::fts::FtsConfig {
                    tokenizer: crate::fts::Tokenizer { stemming: false, ..Default::default() },
                    ..Default::default()
                },
            )]),
            ..Default::default()
        };
        // Stored postings were produced by the old tokenizer, so a query would
        // tokenize differently from the index and match the wrong documents.
        let err = ns
            .commit_records(&[Record::Upsert(Doc::new(9u64, vec![1.0, 0.0]))], &different)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("full-text"), "unhelpful error: {err}");
    }

    // -------------------------------------------------------- phase 12: schema

    #[tokio::test]
    async fn schema_is_inferred_from_the_first_write_then_enforced() {
        let ns = ns("t/schema");
        ns.write_records(&[Record::Upsert(
            Doc::new(1u64, vec![1.0, 0.0])
                .with_attr("name", "a")
                .with_attr("count", 1u64)
                .with_attr("live", true),
        )])
        .await
        .unwrap();

        let (m, _) = ns.load().await.unwrap();
        assert_eq!(m.schema.attributes["name"], crate::value::Type::String);
        assert_eq!(m.schema.attributes["count"], crate::value::Type::Uint);
        assert_eq!(m.schema.attributes["live"], crate::value::Type::Bool);
        assert_eq!(m.schema.dim, Some(2));
        assert_eq!(m.schema.id_type, Some(crate::doc::IdType::Uint));

        // A later document that disagrees is rejected, not silently reinterpreted.
        let err = ns
            .write_records(&[Record::Upsert(
                Doc::new(2u64, vec![1.0, 0.0]).with_attr("count", "not a number"),
            )])
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("count"), "unhelpful schema error: {err}");

        // Widening within the numeric family is allowed and stores the declared type.
        ns.write_records(&[Record::Upsert(
            Doc::new(3u64, vec![1.0, 0.0]).with_attr("count", 7u64),
        )])
        .await
        .unwrap();
        let live = Namespace::materialize(ns.all_records().await.unwrap());
        assert_eq!(live[&Id::Uint(3)].attrs["count"], Value::Uint(7));
    }

    #[tokio::test]
    async fn a_null_never_declares_a_type() {
        let ns = ns("t/schema-null");
        ns.write_records(&[Record::Upsert(
            Doc::new(1u64, vec![1.0]).with_attr("maybe", Value::Null),
        )])
        .await
        .unwrap();
        let (m, _) = ns.load().await.unwrap();
        assert!(
            !m.schema.attributes.contains_key("maybe"),
            "a null poisoned the attribute's type"
        );

        // So a later real value is still free to declare it.
        ns.write_records(&[Record::Upsert(Doc::new(2u64, vec![1.0]).with_attr("maybe", 5u64))])
            .await
            .unwrap();
        let (m, _) = ns.load().await.unwrap();
        assert_eq!(m.schema.attributes["maybe"], crate::value::Type::Uint);
    }

    #[tokio::test]
    async fn declared_types_promote_the_strings_json_cannot_type() {
        let ns = ns("t/schema-coerce");
        // Declare by writing the typed values first.
        ns.write_records(&[Record::Upsert(
            Doc::new(1u64, vec![1.0])
                .with_attr("when", Value::Datetime(0))
                .with_attr("who", Value::Uuid(uuid::Uuid::nil())),
        )])
        .await
        .unwrap();

        // Now a client sends them as strings, as JSON forces it to.
        ns.write_records(&[Record::Upsert(
            Doc::new(2u64, vec![1.0])
                .with_attr("when", "2024-03-01T00:00:00Z")
                .with_attr("who", "550e8400-e29b-41d4-a716-446655440000"),
        )])
        .await
        .unwrap();

        let live = Namespace::materialize(ns.all_records().await.unwrap());
        let d = &live[&Id::Uint(2)];
        assert!(matches!(d.attrs["when"], Value::Datetime(_)), "datetime not promoted");
        assert!(matches!(d.attrs["who"], Value::Uuid(_)), "uuid not promoted");

        // And a range filter over the promoted values works.
        let cutoff = Value::Datetime(crate::value::parse_datetime("2020-01-01").unwrap());
        let res = ns
            .query(
                &QueryRequest::new(vec![1.0])
                    .filter(Filter::cmp("when", crate::doc::Op::Gte, cutoff)),
            )
            .await
            .unwrap();
        assert_eq!(ids(&res.hits), vec![2], "range filter missed the promoted datetime");
    }

    #[tokio::test]
    async fn vector_dimensions_are_enforced_across_writes() {
        let ns = ns("t/dim-enforce");
        ns.write_records(&[Record::Upsert(Doc::new(1u64, vec![1.0, 2.0]))]).await.unwrap();
        let err = ns
            .write_records(&[Record::Upsert(Doc::new(2u64, vec![1.0, 2.0, 3.0]))])
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("dimensions"), "unhelpful dimension error: {err}");

        // A vectorless document in a vector namespace is also refused.
        assert!(ns.write_records(&[Record::Upsert(Doc::new(3u64, vec![]))]).await.is_err());
        // As is a non-finite component.
        assert!(
            ns.write_records(&[Record::Upsert(Doc::new(4u64, vec![1.0, f32::NAN]))]).await.is_err()
        );
    }

    // -------------------------------------------------------- phase 13: ids

    #[tokio::test]
    async fn string_ids_work_end_to_end() {
        let ns = ns("t/string-ids");
        let docs: Vec<Record> = ["alpha", "beta", "gamma"]
            .iter()
            .enumerate()
            .map(|(i, name)| {
                Record::Upsert(Doc::new(*name, vec![1.0 - i as f32 * 0.1, i as f32 * 0.1]))
            })
            .collect();
        ns.write_records(&docs).await.unwrap();
        ns.compact(true).await.unwrap();

        let (m, _) = ns.load().await.unwrap();
        assert_eq!(m.schema.id_type, Some(crate::doc::IdType::String));

        let res = ns.query(&QueryRequest::new(vec![1.0, 0.0]).top_k(2)).await.unwrap();
        assert_eq!(res.hits[0].id, Id::String("alpha".into()));

        // Delete and tombstone by string id.
        ns.write_records(&[Record::Delete(Id::String("alpha".into()))]).await.unwrap();
        let res = ns.query(&QueryRequest::new(vec![1.0, 0.0]).top_k(3)).await.unwrap();
        assert!(!res.hits.iter().any(|h| h.id == Id::String("alpha".into())));

        // A numeric id in a string namespace is coerced, not rejected.
        ns.write_records(&[Record::Upsert(Doc::new(42u64, vec![1.0, 0.0]))]).await.unwrap();
        let live = Namespace::materialize(ns.all_records().await.unwrap());
        assert!(live.contains_key(&Id::String("42".into())), "numeric id was not coerced");
    }

    #[tokio::test]
    async fn uuid_ids_survive_a_roundtrip() {
        let ns = ns("t/uuid-ids");
        let u = uuid::Uuid::from_u128(0x1234_5678);
        ns.write_records(&[Record::Upsert(Doc::new(u, vec![1.0, 0.0]))]).await.unwrap();
        ns.compact(true).await.unwrap();

        let (m, _) = ns.load().await.unwrap();
        assert_eq!(m.schema.id_type, Some(crate::doc::IdType::Uuid));
        let res = ns.query(&QueryRequest::new(vec![1.0, 0.0]).top_k(1)).await.unwrap();
        assert_eq!(res.hits[0].id, Id::Uuid(u));

        // A string form of the same uuid addresses the same document.
        ns.write_records(&[Record::Delete(Id::String(u.to_string()))]).await.unwrap();
        let res = ns.query(&QueryRequest::new(vec![1.0, 0.0]).top_k(1)).await.unwrap();
        assert!(res.hits.is_empty(), "uuid string did not resolve to the same document");
    }

    // -------------------------------------------------------- phase 14: metric

    #[tokio::test]
    async fn distance_metric_changes_ranking_and_is_immutable() {
        let store = mem();

        // Two documents: one closer by angle, the other closer by magnitude.
        let docs = vec![
            Record::Upsert(Doc::new(1u64, vec![10.0, 0.0])),
            Record::Upsert(Doc::new(2u64, vec![1.0, 0.1])),
        ];
        let q = vec![1.0f32, 0.0];

        let cos = Namespace::new(store.clone(), "t/metric-cos");
        cos.commit_records(&docs, &WriteConfig { metric: Some(DistanceMetric::CosineDistance), ..Default::default() }).await.unwrap();
        cos.compact(true).await.unwrap();
        let by_angle = cos.query(&QueryRequest::new(q.clone()).top_k(2)).await.unwrap();

        let euc = Namespace::new(store.clone(), "t/metric-euc");
        euc.commit_records(&docs, &WriteConfig { metric: Some(DistanceMetric::EuclideanSquared), ..Default::default() }).await.unwrap();
        euc.compact(true).await.unwrap();
        let by_distance = euc.query(&QueryRequest::new(q.clone()).top_k(2)).await.unwrap();

        // Cosine ignores magnitude, so the collinear vector wins. Euclidean does
        // not, so the nearby one wins. If the metric were not honoured these
        // would agree.
        assert_eq!(by_angle.hits[0].id, Id::Uint(1), "cosine did not rank by angle");
        assert_eq!(by_distance.hits[0].id, Id::Uint(2), "euclidean did not rank by distance");

        let (m, _) = euc.load().await.unwrap();
        assert_eq!(m.schema.distance_metric, DistanceMetric::EuclideanSquared);

        // Changing it later is refused: the index's clusters were assigned with it.
        let err = euc
            .commit_records(
                &[Record::Upsert(Doc::new(3u64, vec![1.0, 0.0]))],
                &WriteConfig { metric: Some(DistanceMetric::CosineDistance), ..Default::default() },
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("distance_metric"), "unhelpful metric error: {err}");
    }

    // -------------------------------------------------------- phase 7: cache

    #[tokio::test]
    async fn cache_removes_repeat_fetches() {
        let store = mem();
        let warm = Namespace::new(store.clone(), "t/cache");
        let docs = synth(200, 8, 4);
        warm.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        warm.compact(true).await.unwrap();

        let cache = tmp_cache();
        let ns = Namespace::new(store, "t/cache").with_cache(cache.clone());
        let cold = ns.query(&QueryRequest::new(docs[0].vector.clone())).await.unwrap();
        let warm_res = ns.query(&QueryRequest::new(docs[0].vector.clone())).await.unwrap();

        assert_eq!(cold.hits, warm_res.hits, "cached query returned a different answer");
        assert!(warm_res.object_gets < cold.object_gets, "cache did not reduce fetches");
        // The manifest is deliberately never cached, so exactly one GET remains.
        assert_eq!(
            warm_res.object_gets, 1,
            "only the commit-point read should survive the cache"
        );
        assert!(warm_res.cache_hits > 0, "no cache hits recorded");
    }

    #[tokio::test]
    async fn warm_prefetches_the_index() {
        let store = mem();
        let seed = Namespace::new(store.clone(), "t/warm");
        let docs = synth(200, 8, 4);
        seed.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        seed.compact(true).await.unwrap();

        let ns = Namespace::new(store, "t/warm").with_cache(tmp_cache());
        let first = ns.warm().await.unwrap();
        assert!(first.objects_warmed > 1, "warm should pull centroids and clusters");
        assert!(first.bytes > 0);
        assert_eq!(first.already_cached, 0);

        let second = ns.warm().await.unwrap();
        assert_eq!(second.already_cached, second.objects_warmed, "warm did not populate the cache");

        // A query after warming pays only the commit-point read.
        let res = ns.query(&QueryRequest::new(docs[0].vector.clone())).await.unwrap();
        assert_eq!(res.object_gets, 1);
    }

    // -------------------------------------------------------- phase 8: consistency

    #[tokio::test]
    async fn eventual_consistency_skips_the_commit_point_read() {
        let store = mem();
        let seed = Namespace::new(store.clone(), "t/consistency");
        let docs = synth(100, 8, 3);
        seed.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        seed.compact(true).await.unwrap();

        let ns = Namespace::new(store, "t/consistency").with_cache(tmp_cache());
        let q = docs[0].vector.clone();

        // Strong: one GET, the manifest, every time.
        let strong = ns.query(&QueryRequest::new(q.clone())).await.unwrap();
        assert!(strong.consistent);
        let strong2 = ns.query(&QueryRequest::new(q.clone())).await.unwrap();
        assert_eq!(strong2.object_gets, 1, "strong consistency must always re-read the manifest");

        // Eventual with a live snapshot: zero GETs. This is the entire latency
        // difference between the two modes.
        let eventual = ns
            .query(
                &QueryRequest::new(q.clone())
                    .consistency(Consistency::Eventual { max_age_ms: 60_000 }),
            )
            .await
            .unwrap();
        assert_eq!(eventual.object_gets, 0, "eventual query still read the manifest");
        assert!(!eventual.consistent, "a snapshot-served answer must not claim consistency");
        assert_eq!(eventual.hits, strong.hits, "eventual returned a different answer");

        // max_age 0 ages out immediately and degrades to a strong read.
        let expired = ns
            .query(&QueryRequest::new(q).consistency(Consistency::Eventual { max_age_ms: 0 }))
            .await
            .unwrap();
        assert_eq!(expired.object_gets, 1, "an expired snapshot must be refetched");
        assert!(expired.consistent);
    }

    #[tokio::test]
    async fn a_committed_write_is_immediately_visible_to_eventual_reads() {
        let ns = ns("t/read-own-write");
        let docs = synth(50, 4, 2);
        ns.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        ns.compact(true).await.unwrap();

        let target = docs[0].vector.clone();
        ns.write_records(&[Record::Upsert(Doc::new(4242u64, target.clone()))]).await.unwrap();

        // The committer remembers the manifest it just wrote, so even a stale-
        // tolerant read on this node sees the write rather than lagging an
        // age-out behind.
        let res = ns
            .query(
                &QueryRequest::new(target)
                    .consistency(Consistency::Eventual { max_age_ms: 60_000 }),
            )
            .await
            .unwrap();
        assert!(ids(&res.hits).contains(&4242), "eventual read missed this node's own write");
    }

    #[test]
    fn prefilter_decision_compares_what_each_path_reads() {
        // 1000 docs in 40 clusters: one probe touches about 25 of them.
        // A filter selecting fewer than that is cheaper to scan exactly.
        assert!(should_prefilter(20, 1000, 40, 1), "a highly selective filter should scan exactly");
        assert!(!should_prefilter(500, 1000, 40, 1), "a broad filter should probe clusters");

        // Raising nprobe widens the cluster path, so exact scanning wins for
        // larger candidate sets too. The decision self-tunes.
        assert!(should_prefilter(200, 1000, 40, 8), "nprobe=8 touches ~200 docs");
        assert!(!should_prefilter(300, 1000, 40, 8));

        // The absolute ceiling still applies, however selective the filter looks.
        assert!(!should_prefilter(PREFILTER_MAX_CANDIDATES + 1, 100_000_000, 10, 1));
        // No clusters means nothing to probe, so scanning is the only option.
        assert!(should_prefilter(5, 5, 0, 8));
        // Degenerate inputs must not divide by zero or panic.
        assert!(should_prefilter(0, 0, 0, 0));
    }

    #[test]
    fn wal_truncation_keeps_a_prefix_never_a_subset() {
        let e = |name: &str, bytes: u64| WalEntry { name: name.into(), bytes, records: 1 };
        let m = Manifest {
            wal: vec![e("a", 40), e("b", 40), e("c", 40)],
            ..Default::default()
        };
        assert_eq!(m.unindexed_bytes(), 120);

        let (kept, truncated) = m.wal_prefix_within(1000);
        assert_eq!(kept.len(), 3);
        assert!(!truncated);

        // Ordered mutations: dropping the tail yields an earlier consistent view.
        // Dropping the head would apply a later upsert without the one it
        // supersedes, and resurrect deleted documents.
        let (kept, truncated) = m.wal_prefix_within(90);
        assert!(truncated);
        assert_eq!(kept.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);

        let (kept, truncated) = m.wal_prefix_within(10);
        assert!(truncated);
        assert!(kept.is_empty(), "an over-cap first entry must yield an empty prefix");
    }

    #[tokio::test]
    async fn strong_consistency_refuses_rather_than_lies() {
        let ns = ns("t/cliff");
        // Craft a manifest whose recorded tail exceeds the scan cap. Sizes live in
        // the manifest precisely so this decision needs no data fetch.
        let (_, version) = ns.load().await.unwrap();
        let m = Manifest {
            next_seq: 1,
            wal: vec![WalEntry {
                name: "fake.bin".into(),
                bytes: MAX_UNINDEXED_SCAN_BYTES + 1,
                records: 1,
            }],
            segments: vec![],
            index: None,
            schema: Default::default(),
            created_at: None,
            last_write_at: None,
            updated_at: None,
        };
        let mode = match version {
            Some(v) => PutMode::Update(v),
            None => PutMode::Create,
        };
        ns.store
            .put_opts(
                &ns.manifest_path(),
                PutPayload::from(serde_json::to_vec(&m).unwrap()),
                PutOptions { mode, ..Default::default() },
            )
            .await
            .unwrap();

        let err = ns.query(&QueryRequest::new(vec![1.0; 4])).await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("scan cap"), "unexpected error: {msg}");

        // Eventual tolerates it: the tail is dropped and the answer says so,
        // rather than the query failing outright.
        let res = ns
            .query(
                &QueryRequest::new(vec![1.0; 4])
                    .consistency(Consistency::Eventual { max_age_ms: 60_000 }),
            )
            .await
            .unwrap();
        assert!(!res.consistent, "a truncated scan must not claim consistency");
        assert_eq!(res.unindexed_records, 0);
    }

    // -------------------------------------------------------- phase 10: ops

    #[tokio::test]
    async fn metadata_is_answerable_from_the_manifest_alone() {
        let ns = ns("t/meta");
        let docs = synth(120, 8, 4);
        ns.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        ns.compact(true).await.unwrap();
        ns.write_records(&[Record::Upsert(Doc::new(7777u64, vec![1.0; 8]))]).await.unwrap();

        let before = ns.metrics.gets.load(Ordering::Relaxed);
        let md = ns.metadata().await.unwrap();
        let gets = ns.metrics.gets.load(Ordering::Relaxed) - before;
        assert_eq!(gets, 1, "metadata should cost exactly one manifest read");

        assert_eq!(md.indexed_docs, 120);
        assert!(md.clusters > 1);
        assert_eq!(md.segments, 1);
        assert_eq!(md.wal_entries, 1);
        assert_eq!(md.unindexed_records, 1);
        assert_eq!(md.dim, Some(8));
        assert_eq!(md.total_bytes, md.segment_bytes + md.index_bytes + md.unindexed_bytes);
        assert!(!md.write_backpressure);
    }

    #[tokio::test]
    async fn gc_deletes_orphans_spares_recent_and_never_touches_referenced() {
        let ns = ns("t/gc");
        let docs = synth(60, 8, 3);
        ns.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        ns.compact(true).await.unwrap();

        let live_before = ns.query_brute(&docs[0].vector, 5, None).await.unwrap();
        let (m, _) = ns.load().await.unwrap();
        assert!(!m.referenced().is_empty());

        // Compaction already left orphans behind: the WAL objects it folded into
        // the segment are still in storage but no longer named by the manifest.
        // GC is the second half of compaction, not just failed-write cleanup.
        let baseline = ns.gc(Duration::from_secs(3600)).await.unwrap();
        assert!(
            baseline.spared_recent >= 1,
            "compaction should have left its consumed WAL object in storage"
        );
        let orphans_from_compaction = baseline.spared_recent;

        // Two more orphans, of the other kind: what a writer that died between
        // its PUT and its CAS leaves behind.
        ns.put_object(&ns.wal_path("9999999999-orphan.bin"), Bytes::from_static(b"junk"))
            .await
            .unwrap();
        ns.put_object(&ns.data_path("stale.clu"), Bytes::from_static(b"junk")).await.unwrap();

        // Generous grace: everything is younger, so nothing may be deleted. This
        // is the guard that stops GC from racing an in-flight write.
        let spared = ns.gc(Duration::from_secs(3600)).await.unwrap();
        assert_eq!(spared.deleted, 0, "GC deleted inside the grace window");
        assert_eq!(spared.spared_recent, orphans_from_compaction + 2);

        // Zero grace: the orphans go, the referenced objects stay.
        let swept = ns.gc(Duration::ZERO).await.unwrap();
        assert_eq!(swept.deleted, orphans_from_compaction + 2, "GC missed the orphans");
        assert_eq!(swept.spared_recent, 0);

        let live_after = ns.query_brute(&docs[0].vector, 5, None).await.unwrap();
        assert_eq!(live_before, live_after, "GC destroyed live data");

        // Idempotent.
        assert_eq!(ns.gc(Duration::ZERO).await.unwrap().deleted, 0);
    }

    #[tokio::test]
    async fn branch_is_an_independent_snapshot() {
        let store = mem();
        let src = Namespace::new(store.clone(), "t/branch-src");
        let docs = synth(80, 8, 4);
        src.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        src.compact(true).await.unwrap();

        let copied = src.branch("t/branch-dst").await.unwrap();
        assert!(copied > 1, "branch copied nothing");

        let dst = Namespace::new(store.clone(), "t/branch-dst");
        let q = docs[0].vector.clone();
        assert_eq!(
            src.query(&QueryRequest::new(q.clone())).await.unwrap().hits,
            dst.query(&QueryRequest::new(q.clone())).await.unwrap().hits,
            "the branch is not a faithful copy"
        );

        // Writes must not cross in either direction.
        dst.write_records(&[Record::Upsert(Doc::new(11111u64, q.clone()))]).await.unwrap();
        src.write_records(&[Record::Upsert(Doc::new(22222u64, q.clone()))]).await.unwrap();

        let src_ids = ids(&src.query(&QueryRequest::new(q.clone())).await.unwrap().hits);
        let dst_ids = ids(&dst.query(&QueryRequest::new(q)).await.unwrap().hits);
        assert!(src_ids.contains(&22222) && !src_ids.contains(&11111), "dst leaked into src");
        assert!(dst_ids.contains(&11111) && !dst_ids.contains(&22222), "src leaked into dst");

        // Re-branching onto an existing namespace must refuse, not clobber.
        assert!(src.branch("t/branch-dst").await.is_err());
    }

    #[tokio::test]
    async fn destroy_removes_everything() {
        let ns = ns("t/destroy");
        let docs = synth(40, 4, 2);
        ns.write_records(&docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>())
            .await
            .unwrap();
        ns.compact(true).await.unwrap();

        let deleted = ns.destroy().await.unwrap();
        assert!(deleted > 0);
        let (m, version) = ns.load().await.unwrap();
        assert!(version.is_none(), "manifest survived destroy");
        assert_eq!(m, Manifest::default());
        assert_eq!(ns.count_objects("wal").await.unwrap(), 0);
        assert_eq!(ns.count_objects("data").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn group_commit_coalesces_without_losing_writes() {
        const N: usize = 64;
        let ns = ns("t/gc-batch");
        let commit = Arc::new(GroupCommit::new(ns.clone()));

        let tasks: Vec<_> = (0..N)
            .map(|i| {
                let c = commit.clone();
                tokio::spawn(async move { c.upsert(Doc::new(i as u64, vec![i as f32, 0.0])).await })
            })
            .collect();
        for t in tasks {
            t.await.unwrap().unwrap();
        }

        let live = Namespace::materialize(ns.all_records().await.unwrap());
        assert_eq!(live.len(), N, "group commit lost a write");

        let (batches, attempts) = commit.stats();
        assert!(batches <= N);
        // A single committer never races itself, so CAS is never contested.
        assert_eq!(attempts, batches, "single committer should never retry CAS");
        assert_eq!(ns.metrics.cas_conflicts.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn write_all_puts_a_whole_batch_in_one_commit() {
        let ns = ns("t/write-all");
        let commit = GroupCommit::new(ns.clone());
        let records: Vec<Record> =
            (0..50u64).map(|i| Record::Upsert(Doc::new(i, vec![i as f32]))).collect();
        commit.write_all(records, WriteConfig::default()).await.unwrap();

        let (batches, _) = commit.stats();
        assert_eq!(batches, 1, "a single caller batch should cost one commit");
        let (m, _) = ns.load().await.unwrap();
        assert_eq!(m.wal.len(), 1);
        assert_eq!(m.wal[0].records, 50);
        assert_eq!(Namespace::materialize(ns.all_records().await.unwrap()).len(), 50);
    }

    #[tokio::test]
    async fn committing_against_the_remembered_version_skips_the_manifest_read() {
        let ns = ns("t/optimistic");
        // First commit has nothing remembered: GET manifest, PUT wal, CAS.
        ns.write_records(&[Record::Upsert(Doc::new(1u64, vec![1.0]))]).await.unwrap();
        let first = ns.metrics.snapshot();
        assert_eq!(first.puts, 2);

        // Subsequent commits skip the GET entirely.
        let gets_before = first.gets;
        for i in 2..=5u64 {
            ns.write_records(&[Record::Upsert(Doc::new(i, vec![i as f32]))]).await.unwrap();
        }
        let after = ns.metrics.snapshot();
        assert_eq!(after.gets, gets_before, "optimistic commit still read the manifest");
        assert_eq!(after.puts, 2 + 4 * 2, "each commit should cost exactly two PUTs");
        assert_eq!(after.cas_conflicts, 0);

        assert_eq!(Namespace::materialize(ns.all_records().await.unwrap()).len(), 5);
    }

    #[tokio::test]
    async fn a_stale_remembered_version_loses_the_cas_and_recovers() {
        let store = mem();
        let a = Namespace::new(store.clone(), "t/stale");
        let b = Namespace::new(store.clone(), "t/stale");

        // Both learn the same commit point.
        a.write_records(&[Record::Upsert(Doc::new(1u64, vec![1.0]))]).await.unwrap();
        b.load().await.unwrap();

        // A moves the manifest forward. B's remembered version is now stale.
        a.write_records(&[Record::Upsert(Doc::new(2u64, vec![2.0]))]).await.unwrap();

        // B commits anyway: the CAS must reject the stale guess, and B must
        // recover rather than lose the write or clobber A's.
        b.write_records(&[Record::Upsert(Doc::new(3u64, vec![3.0]))]).await.unwrap();
        assert_eq!(b.metrics.cas_conflicts.load(Ordering::Relaxed), 1, "stale guess was accepted");

        let live = Namespace::materialize(a.all_records().await.unwrap());
        assert_eq!(live.len(), 3, "a write was lost recovering from a stale version");
        for id in 1..=3 {
            assert!(live.contains_key(&Id::Uint(id)), "doc {id} vanished");
        }
    }

    #[tokio::test]
    async fn metrics_count_real_object_operations() {
        let ns = ns("t/metrics");
        ns.write_records(&[Record::Upsert(Doc::new(1u64, vec![1.0]))]).await.unwrap();
        let s = ns.metrics.snapshot();
        // One WAL object plus one manifest CAS.
        assert_eq!(s.puts, 2);
        assert_eq!(s.writes, 1);
        assert!(s.bytes_put > 0);
        assert!(s.gets >= 1);
        assert_eq!(s.cas_conflicts, 0);
    }
}
