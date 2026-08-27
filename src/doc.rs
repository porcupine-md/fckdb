//! Document model, binary codec, distance, filters.
//!
//! One codec, used for every data object in the system: WAL entries, compacted
//! segments, and index cluster objects are all just framed `Record`s. Uniform
//! format means one encoder, one decoder, one place for bugs.

use anyhow::{Result, bail};
use bytes::{BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type Id = u64;

/// One ranked result. The single hit shape used by the scan, the index, the
/// recall harness and the HTTP surface — so nothing has to convert between two
/// representations of the same idea.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    pub id: Id,
    pub score: f32,
}

/// A stored document.
///
/// ponytail: attribute values are strings only. Add typed values (numbers,
/// bools, arrays) when a query actually needs range or set filters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Doc {
    pub id: Id,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub attrs: BTreeMap<String, String>,
}

impl Doc {
    pub fn new(id: Id, vector: Vec<f32>) -> Self {
        Self { id, vector, attrs: BTreeMap::new() }
    }

    pub fn with_attr(mut self, k: &str, v: &str) -> Self {
        self.attrs.insert(k.into(), v.into());
        self
    }
}

/// A mutation. Append-only: object storage cannot update in place, so an update
/// is an upsert and a delete is a tombstone.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Upsert(Doc),
    Delete(Id),
}

const TAG_UPSERT: u8 = 0;
const TAG_DELETE: u8 = 1;

impl Record {
    pub fn id(&self) -> Id {
        match self {
            Record::Upsert(d) => d.id,
            Record::Delete(id) => *id,
        }
    }

    pub fn encode(&self) -> Bytes {
        let mut b = BytesMut::new();
        match self {
            Record::Delete(id) => {
                b.put_u8(TAG_DELETE);
                b.put_u64_le(*id);
            }
            Record::Upsert(d) => {
                b.put_u8(TAG_UPSERT);
                b.put_u64_le(d.id);
                b.put_u32_le(d.vector.len() as u32);
                for f in &d.vector {
                    b.put_f32_le(*f);
                }
                b.put_u16_le(d.attrs.len() as u16);
                for (k, v) in &d.attrs {
                    b.put_u16_le(k.len() as u16);
                    b.put_slice(k.as_bytes());
                    b.put_u16_le(v.len() as u16);
                    b.put_slice(v.as_bytes());
                }
            }
        }
        b.freeze()
    }

    pub fn decode(buf: &[u8]) -> Result<Record> {
        let mut r = Cursor { buf, pos: 0 };
        let out = match r.u8()? {
            TAG_DELETE => Record::Delete(r.u64()?),
            TAG_UPSERT => {
                let id = r.u64()?;
                let dim = r.u32()? as usize;
                let mut vector = Vec::with_capacity(dim);
                for _ in 0..dim {
                    vector.push(r.f32()?);
                }
                let n = r.u16()? as usize;
                let mut attrs = BTreeMap::new();
                for _ in 0..n {
                    let k = r.string()?;
                    let v = r.string()?;
                    attrs.insert(k, v);
                }
                Record::Upsert(Doc { id, vector, attrs })
            }
            t => bail!("unknown record tag {t}"),
        };
        Ok(out)
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let Some(s) = self.buf.get(self.pos..self.pos + n) else {
            bail!("truncated record: want {n} bytes at {}, have {}", self.pos, self.buf.len());
        };
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
}

// ---------------------------------------------------------------- distance

/// Plain f32 loop. LLVM autovectorizes this when the slices are equal length;
/// build with `-C target-cpu=native` to get the wide registers.
/// ponytail: no explicit SIMD intrinsics. Add them only if a profile shows this
/// loop dominating and autovectorization failing.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Cosine similarity. Higher is better, so it sorts descending everywhere.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = norm(a) * norm(b);
    if d == 0.0 { 0.0 } else { dot(a, b) / d }
}

/// Squared L2. Used for centroid assignment, where only ordering matters so the
/// sqrt is wasted work.
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ---------------------------------------------------------------- filters

/// ponytail: evaluated per candidate during the scan, no separate index
/// structure. That is fine while filters are cheap and selectivity is low. Build
/// an inverted attribute index when a highly selective filter forces scanning
/// far more candidates than it keeps.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    Eq(String, String),
    Ne(String, String),
    And(Vec<Filter>),
    Or(Vec<Filter>),
}

impl Filter {
    pub fn eq(k: &str, v: &str) -> Self {
        Filter::Eq(k.into(), v.into())
    }

    pub fn matches(&self, d: &Doc) -> bool {
        match self {
            Filter::Eq(k, v) => d.attrs.get(k).is_some_and(|x| x == v),
            Filter::Ne(k, v) => !d.attrs.get(k).is_some_and(|x| x == v),
            Filter::And(fs) => fs.iter().all(|f| f.matches(d)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(d)),
        }
    }
}

/// Rank `docs` against `vector` and return the best `top_k` as (score, id),
/// descending.
pub fn top_k<'a>(
    docs: impl Iterator<Item = &'a Doc>,
    vector: &[f32],
    top_k: usize,
    filter: Option<&Filter>,
) -> Vec<Hit> {
    let mut scored: Vec<Hit> = docs
        .filter(|d| filter.is_none_or(|f| f.matches(d)))
        .map(|d| Hit { id: d.id, score: cosine(vector, &d.vector) })
        .collect();
    // Ties break by id, never by iteration order. Candidates arrive from a
    // HashMap, so without this an identical query returns a different top-k on
    // every run once any two documents score equally — which silently breaks
    // pagination, result caching, and any attempt to debug a ranking change.
    // ponytail: full sort. Switch to select_nth_unstable_by when candidate sets
    // get large enough that O(n log n) shows up next to the distance math.
    scored.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    scored.truncate(top_k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_roundtrip() {
        for r in [
            Record::Delete(42),
            Record::Upsert(Doc::new(7, vec![1.0, -2.5, 3.25])),
            Record::Upsert(Doc::new(9, vec![0.5; 8]).with_attr("tenant", "acme").with_attr("k", "")),
        ] {
            let enc = r.encode();
            assert_eq!(Record::decode(&enc).unwrap(), r, "roundtrip changed the record");
        }
    }

    #[test]
    fn truncated_record_errors_not_panics() {
        let enc = Record::Upsert(Doc::new(1, vec![1.0, 2.0])).encode();
        for cut in 1..enc.len() {
            assert!(Record::decode(&enc[..cut]).is_err(), "accepted a truncated record at {cut}");
        }
    }

    #[test]
    fn cosine_is_scale_invariant_and_ordered() {
        let q = [1.0, 0.0];
        assert!((cosine(&q, &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&q, &[1.0, 0.0]) > cosine(&q, &[1.0, 1.0]));
        assert!(cosine(&q, &[1.0, 1.0]) > cosine(&q, &[0.0, 1.0]));
        assert_eq!(cosine(&q, &[0.0, 0.0]), 0.0, "zero vector must not produce NaN");
    }

    #[test]
    fn filters_compose() {
        let d = Doc::new(1, vec![1.0]).with_attr("t", "a").with_attr("k", "x");
        assert!(Filter::eq("t", "a").matches(&d));
        assert!(!Filter::eq("t", "b").matches(&d));
        assert!(Filter::Ne("t".into(), "b".into()).matches(&d));
        assert!(Filter::And(vec![Filter::eq("t", "a"), Filter::eq("k", "x")]).matches(&d));
        assert!(!Filter::And(vec![Filter::eq("t", "a"), Filter::eq("k", "y")]).matches(&d));
        assert!(Filter::Or(vec![Filter::eq("t", "z"), Filter::eq("k", "x")]).matches(&d));
        // Missing key must not match Eq, and must match Ne.
        assert!(!Filter::eq("missing", "").matches(&d));
        assert!(Filter::Ne("missing".into(), "".into()).matches(&d));
    }

    #[test]
    fn top_k_breaks_ties_deterministically() {
        // Every doc scores identically, so only the tie-break decides the result.
        let docs: Vec<Doc> = [5u64, 1, 9, 3].iter().map(|i| Doc::new(*i, vec![1.0, 0.0])).collect();
        let got = top_k(docs.iter(), &[1.0, 0.0], 3, None);
        assert_eq!(
            got.iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![1, 3, 5],
            "ties must resolve by id, not by iteration order"
        );
    }

    #[test]
    fn top_k_respects_filter_and_order() {
        let docs = vec![
            Doc::new(1, vec![1.0, 0.0]).with_attr("t", "a"),
            Doc::new(2, vec![0.9, 0.1]).with_attr("t", "b"),
            Doc::new(3, vec![0.0, 1.0]).with_attr("t", "a"),
        ];
        let got = top_k(docs.iter(), &[1.0, 0.0], 2, None);
        assert_eq!(got.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 2]);

        let got = top_k(docs.iter(), &[1.0, 0.0], 2, Some(&Filter::eq("t", "a")));
        assert_eq!(got.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 3]);
    }
}
