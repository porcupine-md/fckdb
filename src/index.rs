//! IVF vector index: centroid clustering with posting lists as separate objects.
//!
//! HONEST SCOPING: this is IVF-Flat, not SPFresh. It matches SPFresh's *shape* —
//! centroid-based, so a cold query is one small centroid fetch plus one parallel
//! burst of posting-list fetches, which is what makes it object-storage friendly
//! (few roundtrips, low write amplification) versus HNSW or DiskANN.
//!
//! What it does NOT have is SPFresh's LIRE protocol: incremental cluster
//! split/merge that keeps recall stable under continuous updates. Here, the
//! index is rebuilt wholesale by compaction. That is a real ceiling, not a
//! detail — see the ponytail note on `build`.
//!
//! This module does no IO. It takes documents and returns bytes and groupings;
//! the storage layer decides where they land.

use crate::doc::{DistanceMetric, Doc, Hit};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Where a namespace's index lives on object storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexMeta {
    pub dim: usize,
    /// Object holding `k * dim` little-endian f32 centroids.
    pub centroids: String,
    /// One object per cluster, framed `Record`s. Position matches centroid order.
    pub clusters: Vec<String>,
    /// Documents covered. Anything in the WAL beyond this is unindexed and must
    /// be scanned exhaustively — the same tradeoff turbopuffer makes behind its
    /// index cursor.
    pub docs: usize,
    /// Total bytes of the centroid and cluster objects, so storage cost is
    /// answerable from the manifest without a LIST.
    #[serde(default)]
    pub bytes: u64,
    /// Object holding the segment's document ids in ordinal order. Attribute
    /// indexes address documents by ordinal, and this resolves them back.
    #[serde(default)]
    pub ids: Option<String>,
    /// Attribute name to its inverted-index object.
    #[serde(default)]
    pub attributes: std::collections::BTreeMap<String, String>,
    /// Attribute name to its full-text index object.
    #[serde(default)]
    pub fts: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy)]
pub struct IvfParams {
    pub k: usize,
    pub iters: usize,
    /// Cluster assignment must use the namespace's own geometry. Building with
    /// Euclidean and querying with cosine puts a document's nearest neighbours
    /// in a cluster the query never probes.
    pub metric: DistanceMetric,
}

impl IvfParams {
    /// sqrt(n) clusters is the standard IVF rule of thumb: it balances centroid
    /// scan cost against posting list size.
    pub fn for_docs(n: usize) -> Self {
        Self {
            k: (n as f64).sqrt().ceil().max(1.0) as usize,
            iters: 10,
            metric: DistanceMetric::default(),
        }
    }

    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// Cluster `docs` and return (centroids, docs grouped by cluster).
///
/// ponytail: full rebuild, no incremental maintenance. Every compaction reruns
/// k-means over everything. Cost is O(iters * n * k * dim), which is fine while
/// a namespace fits in memory and compaction is infrequent. Implement LIRE-style
/// incremental split/merge when rebuild time, not query time, becomes the
/// bottleneck.
pub fn build(docs: &[Doc], params: IvfParams) -> Result<(Vec<f32>, Vec<Vec<Doc>>)> {
    if docs.is_empty() {
        bail!("cannot build an index over zero documents");
    }
    let dim = docs[0].vector.len();
    if let Some(bad) = docs.iter().find(|d| d.vector.len() != dim) {
        bail!("dimension mismatch: doc {} has {} dims, expected {dim}", bad.id, bad.vector.len());
    }

    let vectors: Vec<&[f32]> = docs.iter().map(|d| d.vector.as_slice()).collect();
    let centroids = kmeans(&vectors, params.k.min(docs.len()), params.iters, params.metric);

    let mut groups: Vec<Vec<Doc>> = vec![Vec::new(); centroids.len()];
    for doc in docs {
        groups[nearest(&centroids, &doc.vector, params.metric)].push(doc.clone());
    }

    // Empty clusters would cost a centroid comparison forever and never return a
    // candidate. Drop them, keeping centroid/cluster positions aligned.
    let mut flat = Vec::with_capacity(centroids.len() * dim);
    let mut kept = Vec::with_capacity(groups.len());
    for (c, g) in centroids.iter().zip(groups) {
        if g.is_empty() {
            continue;
        }
        flat.extend_from_slice(c);
        kept.push(g);
    }
    Ok((flat, kept))
}

/// Lloyd's algorithm. Deterministic init by strided sampling: no RNG dependency,
/// and identical input gives an identical index, which makes tests and the recall
/// harness reproducible.
pub fn kmeans(
    vectors: &[&[f32]],
    k: usize,
    iters: usize,
    metric: DistanceMetric,
) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let k = k.clamp(1, n);
    let dim = vectors[0].len();

    let mut centroids: Vec<Vec<f32>> =
        (0..k).map(|i| vectors[i * n / k].to_vec()).collect();

    for _ in 0..iters {
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for v in vectors {
            let c = nearest(&centroids, v, metric);
            counts[c] += 1;
            for (s, x) in sums[c].iter_mut().zip(v.iter()) {
                *s += x;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                continue; // keep the old position rather than collapsing it
            }
            for (target, sum) in centroids[c].iter_mut().zip(&sums[c]) {
                *target = sum / counts[c] as f32;
            }
        }
    }
    centroids
}

fn nearest(centroids: &[Vec<f32>], v: &[f32], metric: DistanceMetric) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = metric.distance(c, v);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Indices of the `nprobe` centroids closest to `q`, nearest first.
pub fn probe(
    centroids: &[Vec<f32>],
    q: &[f32],
    nprobe: usize,
    metric: DistanceMetric,
) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> =
        centroids.iter().enumerate().map(|(i, c)| (metric.distance(c, q), i)).collect();
    scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    scored.truncate(nprobe.max(1));
    scored.into_iter().map(|(_, i)| i).collect()
}

pub fn encode_centroids(flat: &[f32]) -> Vec<u8> {
    flat.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn decode_centroids(buf: &[u8], dim: usize) -> Result<Vec<Vec<f32>>> {
    if dim == 0 || buf.len() % (dim * 4) != 0 {
        bail!("centroid blob of {} bytes is not a multiple of dim {dim}", buf.len());
    }
    Ok(buf
        .chunks_exact(dim * 4)
        .map(|c| c.chunks_exact(4).map(|f| f32::from_le_bytes(f.try_into().unwrap())).collect())
        .collect())
}

/// Fraction of the exact top-k that the approximate result also found.
///
/// Without this number an ANN index is unfalsifiable: it will always return
/// *something*, and you cannot tell a tuning change from a correctness bug.
pub fn recall(exact: &[Hit], approx: &[Hit]) -> f32 {
    if exact.is_empty() {
        return 1.0;
    }
    let found: std::collections::HashSet<&crate::doc::Id> = approx.iter().map(|h| &h.id).collect();
    exact.iter().filter(|h| found.contains(&h.id)).count() as f32 / exact.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three tight, well-separated blobs. k-means must recover them.
    fn blobs() -> Vec<Doc> {
        let mut docs = Vec::new();
        for (c, center) in [[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]].iter().enumerate() {
            for i in 0..10 {
                let jitter = i as f32 * 0.01;
                docs.push(Doc::new(
                    (c * 100 + i) as u64,
                    vec![center[0] + jitter, center[1] + jitter],
                ));
            }
        }
        docs
    }

    #[test]
    fn build_recovers_separated_clusters() {
        let docs = blobs();
        let (flat, groups) = build(&docs, IvfParams { k: 3, iters: 20, metric: DistanceMetric::default() }).unwrap();
        assert_eq!(groups.len(), 3, "expected 3 non-empty clusters");
        assert_eq!(flat.len(), 3 * 2);
        for g in &groups {
            assert_eq!(g.len(), 10, "blobs should split evenly");
            // Every member of a cluster came from the same original blob.
            let band = g[0].id.as_uint().unwrap() / 100;
            assert!(g.iter().all(|d| d.id.as_uint().unwrap() / 100 == band), "cluster mixed blobs");
        }
    }

    #[test]
    fn probe_finds_the_owning_cluster() {
        let docs = blobs();
        let (flat, groups) = build(&docs, IvfParams { k: 3, iters: 20, metric: DistanceMetric::default() }).unwrap();
        let centroids = decode_centroids(&encode_centroids(&flat), 2).unwrap();

        // A query sitting on top of one blob must probe that blob's cluster first.
        let hit = probe(&centroids, &[10.0, 0.0], 1, DistanceMetric::default())[0];
        assert_eq!(groups[hit][0].id.as_uint().unwrap() / 100, 0);
    }

    #[test]
    fn centroids_roundtrip() {
        let flat = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let got = decode_centroids(&encode_centroids(&flat), 3).unwrap();
        assert_eq!(got, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert!(decode_centroids(&encode_centroids(&flat), 4).is_err(), "ragged blob accepted");
        assert!(decode_centroids(&[1, 2, 3], 1).is_err(), "partial float accepted");
    }

    #[test]
    fn build_rejects_ragged_dimensions() {
        let docs = vec![Doc::new(1u64, vec![1.0, 2.0]), Doc::new(2u64, vec![1.0])];
        assert!(build(&docs, IvfParams::for_docs(2)).is_err());
        assert!(build(&[], IvfParams::for_docs(0)).is_err());
    }

    #[test]
    fn recall_counts_overlap() {
        let h = |id: u64, score| Hit::new(id, score);
        let exact = vec![h(1, 1.0), h(2, 0.9), h(3, 0.8), h(4, 0.7)];
        assert_eq!(recall(&exact, &exact), 1.0);
        assert_eq!(recall(&exact, &exact[..2]), 0.5);
        assert_eq!(recall(&exact, &[h(99, 1.0)]), 0.0);
        assert_eq!(recall(&[], &[]), 1.0);
    }

    #[test]
    fn k_larger_than_docs_is_clamped() {
        let docs = vec![Doc::new(1u64, vec![1.0]), Doc::new(2u64, vec![2.0])];
        let (_, groups) = build(&docs, IvfParams { k: 100, iters: 5, metric: DistanceMetric::default() }).unwrap();
        assert!(groups.len() <= 2);
        assert_eq!(groups.iter().map(|g| g.len()).sum::<usize>(), 2, "lost a doc");
    }
}
