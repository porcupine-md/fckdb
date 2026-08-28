//! Sparse vector search.
//!
//! A sparse vector is named dimensions to weights — what SPLADE-style learned
//! sparse retrieval produces. Similarity is the dot product over shared
//! dimensions, so a document with no dimension in common with the query scores
//! zero and is excluded.
//!
//! The index is an inverted list per dimension, the same shape as the full-text
//! term index. That is not a coincidence: a sparse vector query is a weighted
//! term query, and the access pattern object storage rewards is identical —
//! touch only the dimensions the query mentions, never every document.
//!
//! IVF clustering is deliberately absent. Centroids over a space with tens of
//! thousands of mostly-empty dimensions carry almost no information, and the
//! inverted list already restricts candidates to documents that share a
//! dimension.

use anyhow::{Result, bail};
use bytes::{BufMut, Bytes, BytesMut};
use std::collections::{BTreeMap, HashMap};

pub type SparseVector = BTreeMap<String, f32>;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SparseIndex {
    /// Sorted by dimension so a lookup is a binary search.
    dims: Vec<(String, Vec<(u32, f32)>)>,
    doc_count: usize,
}

impl SparseIndex {
    pub fn build<'a>(
        docs: impl Iterator<Item = (u32, &'a SparseVector)>,
        doc_count: usize,
    ) -> Self {
        let mut postings: HashMap<&str, Vec<(u32, f32)>> = HashMap::new();
        for (ordinal, vector) in docs {
            for (dim, weight) in vector {
                // A zero weight contributes nothing to any dot product, so
                // storing it would only cost space and candidate scans.
                if *weight == 0.0 {
                    continue;
                }
                postings.entry(dim).or_default().push((ordinal, *weight));
            }
        }
        let mut dims: Vec<(String, Vec<(u32, f32)>)> =
            postings.into_iter().map(|(d, p)| (d.to_string(), p)).collect();
        dims.sort_by(|a, b| a.0.cmp(&b.0));
        for (_, p) in dims.iter_mut() {
            p.sort_unstable_by_key(|(o, _)| *o);
        }
        Self { dims, doc_count }
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn dimensions(&self) -> usize {
        self.dims.len()
    }

    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    fn postings(&self, dim: &str) -> Option<&[(u32, f32)]> {
        self.dims
            .binary_search_by(|(d, _)| d.as_str().cmp(dim))
            .ok()
            .map(|i| self.dims[i].1.as_slice())
    }

    /// Dot product against every document sharing a dimension with `query`,
    /// descending. Documents scoring zero are excluded.
    pub fn score(&self, query: &SparseVector) -> Vec<(u32, f32)> {
        let mut acc: HashMap<u32, f32> = HashMap::new();
        for (dim, qw) in query {
            if *qw == 0.0 {
                continue;
            }
            let Some(list) = self.postings(dim) else { continue };
            for (ordinal, dw) in list {
                *acc.entry(*ordinal).or_insert(0.0) += qw * dw;
            }
        }
        let mut out: Vec<(u32, f32)> = acc.into_iter().filter(|(_, s)| *s != 0.0).collect();
        out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    pub fn encode(&self) -> Bytes {
        let mut b = BytesMut::new();
        b.put_u32_le(self.doc_count as u32);
        b.put_u32_le(self.dims.len() as u32);
        for (dim, postings) in &self.dims {
            b.put_u32_le(dim.len() as u32);
            b.put_slice(dim.as_bytes());
            b.put_u32_le(postings.len() as u32);
            for (ordinal, weight) in postings {
                b.put_u32_le(*ordinal);
                b.put_f32_le(*weight);
            }
        }
        b.freeze()
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let end = pos.checked_add(n).unwrap_or(usize::MAX);
            let Some(s) = buf.get(pos..end) else {
                bail!("truncated sparse index: want {n} bytes at {pos}, have {}", buf.len());
            };
            pos += n;
            Ok(s)
        };
        let u32le = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());

        let doc_count = u32le(take(4)?) as usize;
        let n_dims = u32le(take(4)?) as usize;
        let mut dims = Vec::with_capacity(n_dims.min(1 << 20));
        for _ in 0..n_dims {
            let dl = u32le(take(4)?) as usize;
            let dim = String::from_utf8(take(dl)?.to_vec())?;
            let np = u32le(take(4)?) as usize;
            let mut postings = Vec::with_capacity(np.min(1 << 20));
            for _ in 0..np {
                let ordinal = u32le(take(4)?);
                let weight = f32::from_le_bytes(take(4)?.try_into().unwrap());
                postings.push((ordinal, weight));
            }
            dims.push((dim, postings));
        }
        Ok(Self { dims, doc_count })
    }
}

/// Dot product of two sparse vectors, for scoring documents the index has not
/// seen yet — the unindexed tail.
///
/// The same measure as the index computes, so a freshly written document is
/// directly comparable rather than unrankable.
pub fn dot(a: &SparseVector, b: &SparseVector) -> f32 {
    // Iterate the shorter side: a query has a handful of dimensions while a
    // document may have thousands.
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    short.iter().filter_map(|(dim, w)| long.get(dim).map(|o| w * o)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(pairs: &[(&str, f32)]) -> SparseVector {
        pairs.iter().map(|(d, w)| (d.to_string(), *w)).collect()
    }

    fn corpus() -> (Vec<SparseVector>, SparseIndex) {
        let docs = vec![
            sv(&[("cat", 1.0), ("pet", 0.5)]),
            sv(&[("cat", 0.2), ("dog", 1.0)]),
            sv(&[("car", 2.0)]),
            sv(&[]),
        ];
        let index = SparseIndex::build(docs.iter().enumerate().map(|(i, v)| (i as u32, v)), 4);
        (docs, index)
    }

    #[test]
    fn scores_by_dot_product_over_shared_dimensions() {
        let (_, i) = corpus();
        let got = i.score(&sv(&[("cat", 1.0), ("pet", 1.0)]));
        // doc0: 1.0*1.0 + 0.5*1.0 = 1.5 ; doc1: 0.2*1.0 = 0.2
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].0, 0);
        assert!((got[0].1 - 1.5).abs() < 1e-6, "got {}", got[0].1);
        assert_eq!(got[1].0, 1);
        assert!((got[1].1 - 0.2).abs() < 1e-6);
    }

    #[test]
    fn documents_sharing_no_dimension_are_excluded() {
        let (_, i) = corpus();
        // "car" only touches doc2; nothing else shares a dimension.
        let got = i.score(&sv(&[("car", 1.0)]));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 2);
        // A query with no known dimension returns nothing, not every document.
        assert!(i.score(&sv(&[("zebra", 1.0)])).is_empty());
        assert!(i.score(&sv(&[])).is_empty());
    }

    #[test]
    fn zero_weights_are_ignored_on_both_sides() {
        // A zero weight contributes nothing to any dot product, so it should not
        // create a candidate.
        let docs = vec![sv(&[("a", 0.0), ("b", 1.0)])];
        let i = SparseIndex::build(docs.iter().enumerate().map(|(n, v)| (n as u32, v)), 1);
        assert_eq!(i.dimensions(), 1, "a zero-weight dimension was indexed");
        assert!(i.score(&sv(&[("a", 5.0)])).is_empty());
        assert!(i.score(&sv(&[("b", 0.0)])).is_empty());
        assert_eq!(i.score(&sv(&[("b", 2.0)]))[0].1, 2.0);
    }

    #[test]
    fn negative_weights_rank_below_positive_ones() {
        // Learned sparse models emit only positive weights, but nothing in the
        // format forbids negatives and the ordering must still be sane.
        let docs = vec![sv(&[("a", 1.0)]), sv(&[("a", -1.0)])];
        let i = SparseIndex::build(docs.iter().enumerate().map(|(n, v)| (n as u32, v)), 2);
        let got = i.score(&sv(&[("a", 1.0)]));
        assert_eq!(got[0].0, 0);
        assert_eq!(got[1].0, 1);
        assert!(got[1].1 < 0.0);
    }

    #[test]
    fn ties_break_by_ordinal_for_stable_results() {
        let docs = vec![sv(&[("a", 1.0)]), sv(&[("a", 1.0)]), sv(&[("a", 1.0)])];
        let i = SparseIndex::build(docs.iter().enumerate().map(|(n, v)| (n as u32, v)), 3);
        let got = i.score(&sv(&[("a", 1.0)]));
        assert_eq!(got.iter().map(|(o, _)| *o).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn dot_matches_what_the_index_computes() {
        let (docs, i) = corpus();
        let query = sv(&[("cat", 1.0), ("pet", 1.0)]);
        let from_index = i.score(&query);
        for (ordinal, score) in from_index {
            let direct = dot(&docs[ordinal as usize], &query);
            assert!(
                (direct - score).abs() < 1e-6,
                "tail scoring diverged from the index: {direct} vs {score}"
            );
        }
        // Symmetric, and empty on either side is zero.
        assert_eq!(dot(&query, &docs[0]), dot(&docs[0], &query));
        assert_eq!(dot(&sv(&[]), &query), 0.0);
    }

    #[test]
    fn roundtrip_through_bytes() {
        let (_, i) = corpus();
        let encoded = i.encode();
        assert_eq!(SparseIndex::decode(&encoded).unwrap(), i);
        assert_eq!(
            SparseIndex::decode(&encoded).unwrap().score(&sv(&[("cat", 1.0)])),
            i.score(&sv(&[("cat", 1.0)]))
        );
        for cut in 0..encoded.len() {
            // Truncation errors rather than panicking or yielding a short index.
            let _ = SparseIndex::decode(&encoded[..cut]);
        }

        let empty = SparseIndex::default();
        assert_eq!(SparseIndex::decode(&empty.encode()).unwrap(), empty);
        assert!(empty.score(&sv(&[("a", 1.0)])).is_empty());
    }
}
