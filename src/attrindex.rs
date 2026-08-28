//! Inverted attribute indexes.
//!
//! One object per attribute, holding `value -> document ordinals` sorted by
//! value so range predicates are a contiguous slice. Ordinals index into the
//! segment's document order, which compaction fixes by sorting on id.
//!
//! The contract is deliberately loose: `select` may return a SUPERSET of the
//! matching documents, or `None` when it cannot bound the answer at all. Callers
//! re-apply the real filter while scoring, so a superset costs a little wasted
//! work and never a wrong answer. Only `ids_matching` cares about exactness, and
//! it checks.
//!
//! Two things this buys beyond speed:
//!
//!   - `ids_matching` for filter-based writes stops materializing the whole live
//!     set when the filter is exactly answerable
//!   - **filtered vector search stops losing recall**. Probing clusters and then
//!     filtering can return fewer than `top_k` results, or none, when the
//!     surviving documents all live in clusters the query never probed. A
//!     selective filter is answered exactly instead.

use crate::doc::{Filter, Op};
use crate::value::Value;
use anyhow::{Result, bail};
use bytes::{BufMut, Bytes, BytesMut};
use std::collections::{BTreeMap, BTreeSet};

/// What the index could work out about a filter.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub ordinals: BTreeSet<u32>,
    /// False when `ordinals` is only a superset and the filter must still be
    /// re-applied to each candidate.
    pub exact: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AttrIndex {
    /// Sorted by value. For array attributes the keys are ELEMENTS, so a
    /// document appears under each of its values.
    entries: Vec<(Value, Vec<u32>)>,
    /// True when the source attribute was an array, which changes which
    /// operators the index can answer.
    multi_valued: bool,
}

impl AttrIndex {
    pub fn build(values: impl Iterator<Item = (u32, Value)>) -> Self {
        let mut map: Vec<(Value, Vec<u32>)> = Vec::new();
        let mut index: BTreeMap<String, usize> = BTreeMap::new();
        let mut multi_valued = false;

        for (ordinal, value) in values {
            if value.is_null() {
                continue;
            }
            let parts = match &value {
                Value::StringArray(_) | Value::UintArray(_) | Value::IntArray(_) => {
                    multi_valued = true;
                    value.elements()
                }
                _ => vec![value.clone()],
            };
            for part in parts {
                // A stable string key, only to deduplicate while building; the
                // stored key is the Value itself.
                let key = format!("{:?}", part);
                match index.get(&key) {
                    Some(&i) => map[i].1.push(ordinal),
                    None => {
                        index.insert(key, map.len());
                        map.push((part, vec![ordinal]));
                    }
                }
            }
        }

        map.sort_by(|a, b| a.0.compare(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, ords) in map.iter_mut() {
            ords.sort_unstable();
            ords.dedup();
        }
        Self { entries: map, multi_valued }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Ordinals whose value satisfies `op` against `operand`.
    ///
    /// `None` means "cannot answer" and the caller must fall back — never
    /// "matches nothing", which would silently drop documents.
    pub fn select(&self, op: Op, operand: &Value) -> Option<Selection> {
        let all = |it: Box<dyn Iterator<Item = &(Value, Vec<u32>)> + '_>| -> BTreeSet<u32> {
            it.flat_map(|(_, ords)| ords.iter().copied()).collect()
        };

        let exact_scalar = !self.multi_valued;
        match op {
            Op::Eq => {
                // On an array attribute the keys are elements, so equality
                // against the whole array is not answerable here.
                if self.multi_valued || operand.is_null() {
                    return None;
                }
                let ords = self.lookup(operand);
                Some(Selection { ordinals: ords, exact: true })
            }
            Op::Gt | Op::Gte | Op::Lt | Op::Lte => {
                if operand.is_null() {
                    return None;
                }
                let picked = self.entries.iter().filter(|(v, _)| {
                    use std::cmp::Ordering::*;
                    match v.compare(operand) {
                        Some(Greater) => matches!(op, Op::Gt | Op::Gte),
                        Some(Less) => matches!(op, Op::Lt | Op::Lte),
                        Some(Equal) => matches!(op, Op::Gte | Op::Lte),
                        None => false,
                    }
                });
                Some(Selection { ordinals: all(Box::new(picked)), exact: exact_scalar })
            }
            Op::In | Op::ContainsAny => {
                let wanted = operand.elements();
                let mut ordinals = BTreeSet::new();
                for w in &wanted {
                    ordinals.extend(self.lookup(w));
                }
                // On an array attribute, "contains any" is exactly this union.
                // On a scalar, "In" is too.
                Some(Selection { ordinals, exact: true })
            }
            Op::Contains => {
                if self.multi_valued {
                    // Membership: the element index answers it exactly.
                    Some(Selection { ordinals: self.lookup(operand), exact: true })
                } else {
                    // Substring over text: every distinct value must be tested,
                    // but that is still far cheaper than every document.
                    let picked = self.entries.iter().filter(|(v, _)| {
                        matches!((v.as_text(), operand.as_text()), (Some(h), Some(n)) if h.contains(&n))
                    });
                    Some(Selection { ordinals: all(Box::new(picked)), exact: true })
                }
            }
            Op::Glob | Op::IGlob => {
                let pattern = operand.as_text()?;
                let re = crate::doc::glob_to_regex(&pattern, op == Op::Glob).ok()?;
                let picked = self
                    .entries
                    .iter()
                    .filter(|(v, _)| v.as_text().is_some_and(|t| re.is_match(&t)));
                Some(Selection { ordinals: all(Box::new(picked)), exact: true })
            }
            Op::Regex => {
                let pattern = operand.as_text()?;
                let re = regex::Regex::new(&pattern).ok()?;
                let picked = self
                    .entries
                    .iter()
                    .filter(|(v, _)| v.as_text().is_some_and(|t| re.is_match(&t)));
                Some(Selection { ordinals: all(Box::new(picked)), exact: true })
            }
            // Negations would need the universe of ordinals, which the index does
            // not hold: a document with no value for this attribute appears
            // nowhere here, yet satisfies NotEq. Left to the scan.
            Op::NotEq
            | Op::NotIn
            | Op::NotContains
            | Op::NotContainsAny
            | Op::NotGlob
            | Op::NotIGlob
            | Op::NotRegex => None,
        }
    }

    fn lookup(&self, needle: &Value) -> BTreeSet<u32> {
        self.entries
            .binary_search_by(|(v, _)| v.compare(needle).unwrap_or(std::cmp::Ordering::Equal))
            .map(|i| self.entries[i].1.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn encode(&self) -> Bytes {
        let mut b = BytesMut::new();
        b.put_u8(self.multi_valued as u8);
        b.put_u32_le(self.entries.len() as u32);
        for (value, ords) in &self.entries {
            value.encode(&mut b);
            b.put_u32_le(ords.len() as u32);
            for o in ords {
                b.put_u32_le(*o);
            }
        }
        b.freeze()
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let need = |pos: usize, n: usize, len: usize| -> Result<()> {
            if pos + n > len {
                bail!("truncated attribute index at {pos}");
            }
            Ok(())
        };
        need(pos, 5, buf.len())?;
        let multi_valued = buf[pos] != 0;
        pos += 1;
        let count = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut entries = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            let value = Value::decode(buf, &mut pos)?;
            need(pos, 4, buf.len())?;
            let n = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            need(pos, n * 4, buf.len())?;
            let mut ords = Vec::with_capacity(n.min(4096));
            for _ in 0..n {
                ords.push(u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()));
                pos += 4;
            }
            entries.push((value, ords));
        }
        Ok(Self { entries, multi_valued })
    }
}

/// Bound a filter using whatever attribute indexes exist.
///
/// Returns `None` when nothing useful can be said. A `Some` result is always at
/// least a superset of the true match set, so callers must still apply the
/// filter while scoring unless `exact` is set.
pub fn evaluate(
    filter: &Filter,
    indexes: &BTreeMap<String, AttrIndex>,
    total: u32,
) -> Option<Selection> {
    match filter {
        Filter::Cmp { key, op, value } => indexes.get(key)?.select(*op, value),

        Filter::And(parts) => {
            // An operand the index cannot answer simply drops out: intersecting
            // fewer constraints still yields a superset.
            let mut acc: Option<Selection> = None;
            for p in parts {
                let Some(sel) = evaluate(p, indexes, total) else { continue };
                acc = Some(match acc {
                    None => sel,
                    Some(prev) => Selection {
                        ordinals: prev.ordinals.intersection(&sel.ordinals).copied().collect(),
                        exact: prev.exact && sel.exact,
                    },
                });
            }
            // Exact only if every operand was answerable AND exact.
            acc.map(|mut s| {
                s.exact = s.exact && parts.iter().all(|p| {
                    evaluate(p, indexes, total).is_some_and(|x| x.exact)
                });
                s
            })
        }

        Filter::Or(parts) => {
            // Unlike And, one unanswerable operand poisons the union: its matches
            // would be missing entirely rather than merely under-constrained.
            let mut ordinals = BTreeSet::new();
            let mut exact = true;
            for p in parts {
                let sel = evaluate(p, indexes, total)?;
                exact &= sel.exact;
                ordinals.extend(sel.ordinals);
            }
            Some(Selection { ordinals, exact })
        }

        Filter::Not(inner) => {
            // Complement is only meaningful against an exact set.
            let sel = evaluate(inner, indexes, total)?;
            if !sel.exact {
                return None;
            }
            Some(Selection {
                ordinals: (0..total).filter(|o| !sel.ordinals.contains(o)).collect(),
                exact: true,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::Filter;

    fn idx(values: Vec<(u32, Value)>) -> AttrIndex {
        AttrIndex::build(values.into_iter())
    }

    fn scalars() -> AttrIndex {
        idx(vec![
            (0, Value::Uint(10)),
            (1, Value::Uint(20)),
            (2, Value::Uint(30)),
            (3, Value::Uint(20)),
            (4, Value::Null),
        ])
    }

    fn set(v: &[u32]) -> BTreeSet<u32> {
        v.iter().copied().collect()
    }

    #[test]
    fn equality_and_ranges() {
        let i = scalars();
        assert_eq!(i.select(Op::Eq, &Value::Uint(20)).unwrap().ordinals, set(&[1, 3]));
        assert_eq!(i.select(Op::Eq, &Value::Uint(99)).unwrap().ordinals, set(&[]));
        assert_eq!(i.select(Op::Gte, &Value::Uint(20)).unwrap().ordinals, set(&[1, 2, 3]));
        assert_eq!(i.select(Op::Gt, &Value::Uint(20)).unwrap().ordinals, set(&[2]));
        assert_eq!(i.select(Op::Lt, &Value::Uint(20)).unwrap().ordinals, set(&[0]));
        assert_eq!(i.select(Op::Lte, &Value::Uint(20)).unwrap().ordinals, set(&[0, 1, 3]));
        assert!(i.select(Op::Eq, &Value::Uint(20)).unwrap().exact);
        // A null-valued document is absent from the index entirely.
        assert!(!i.select(Op::Gte, &Value::Uint(0)).unwrap().ordinals.contains(&4));
    }

    #[test]
    fn negations_are_not_answerable_from_an_inverted_index() {
        let i = scalars();
        // Ordinal 4 has no value for this attribute, so it appears nowhere here,
        // yet it satisfies NotEq. Answering from the index would drop it.
        for op in [Op::NotEq, Op::NotIn, Op::NotContains, Op::NotGlob, Op::NotRegex] {
            assert!(i.select(op, &Value::Uint(20)).is_none(), "{op:?} claimed an answer");
        }
    }

    #[test]
    fn array_attributes_index_their_elements() {
        let i = idx(vec![
            (0, Value::StringArray(vec!["red".into(), "blue".into()])),
            (1, Value::StringArray(vec!["blue".into()])),
            (2, Value::StringArray(vec![])),
        ]);
        assert_eq!(i.select(Op::Contains, &Value::from("blue")).unwrap().ordinals, set(&[0, 1]));
        assert_eq!(i.select(Op::Contains, &Value::from("red")).unwrap().ordinals, set(&[0]));
        assert_eq!(
            i.select(Op::ContainsAny, &Value::StringArray(vec!["red".into(), "green".into()]))
                .unwrap()
                .ordinals,
            set(&[0])
        );
        // Equality against a whole array is not what the element index holds.
        assert!(i.select(Op::Eq, &Value::StringArray(vec!["blue".into()])).is_none());
    }

    #[test]
    fn text_operators_scan_distinct_values_not_documents() {
        let i = idx(vec![
            (0, Value::from("docs/intro.md")),
            (1, Value::from("docs/guide.md")),
            (2, Value::from("blog/post.md")),
        ]);
        assert_eq!(i.select(Op::Glob, &Value::from("docs/*")).unwrap().ordinals, set(&[0, 1]));
        assert_eq!(i.select(Op::Regex, &Value::from(r"\.md$")).unwrap().ordinals, set(&[0, 1, 2]));
        assert_eq!(i.select(Op::Contains, &Value::from("intro")).unwrap().ordinals, set(&[0]));
        // A malformed pattern is unanswerable, not empty.
        assert!(i.select(Op::Regex, &Value::from("[unclosed")).is_none());
    }

    #[test]
    fn roundtrip_through_bytes() {
        for i in [scalars(), idx(vec![(0, Value::StringArray(vec!["a".into()]))]), AttrIndex::default()] {
            let encoded = i.encode();
            assert_eq!(AttrIndex::decode(&encoded).unwrap(), i);
            // Truncation is an error, never a panic or a silently short index.
            for cut in 0..encoded.len() {
                let _ = AttrIndex::decode(&encoded[..cut]);
            }
        }
    }

    fn indexes() -> BTreeMap<String, AttrIndex> {
        BTreeMap::from([
            ("rank".to_string(), scalars()),
            (
                "tier".to_string(),
                idx(vec![
                    (0, Value::from("gold")),
                    (1, Value::from("silver")),
                    (2, Value::from("gold")),
                    (3, Value::from("silver")),
                    (4, Value::from("gold")),
                ]),
            ),
        ])
    }

    #[test]
    fn and_intersects_and_tolerates_unanswerable_operands() {
        let ix = indexes();
        let f = Filter::And(vec![
            Filter::cmp("tier", Op::Eq, "gold"),
            Filter::cmp("rank", Op::Gte, 20u64),
        ]);
        let sel = evaluate(&f, &ix, 5).unwrap();
        assert_eq!(sel.ordinals, set(&[2]));
        assert!(sel.exact);

        // A NotEq operand cannot be answered, so it drops out and the result is a
        // superset — still safe, because the caller re-applies the filter.
        let f = Filter::And(vec![
            Filter::cmp("tier", Op::Eq, "gold"),
            Filter::cmp("rank", Op::NotEq, 10u64),
        ]);
        let sel = evaluate(&f, &ix, 5).unwrap();
        assert_eq!(sel.ordinals, set(&[0, 2, 4]));
        assert!(!sel.exact, "a superset must not claim to be exact");

        // An unknown attribute likewise drops out.
        let f = Filter::And(vec![
            Filter::cmp("tier", Op::Eq, "gold"),
            Filter::cmp("nosuch", Op::Eq, 1u64),
        ]);
        assert_eq!(evaluate(&f, &ix, 5).unwrap().ordinals, set(&[0, 2, 4]));
    }

    #[test]
    fn or_is_poisoned_by_an_unanswerable_operand() {
        let ix = indexes();
        let ok = Filter::Or(vec![
            Filter::cmp("tier", Op::Eq, "silver"),
            Filter::cmp("rank", Op::Eq, 30u64),
        ]);
        let sel = evaluate(&ok, &ix, 5).unwrap();
        assert_eq!(sel.ordinals, set(&[1, 2, 3]));

        // Dropping an unanswerable branch of an Or would LOSE its matches, which
        // is a wrong answer rather than extra work.
        let bad = Filter::Or(vec![
            Filter::cmp("tier", Op::Eq, "silver"),
            Filter::cmp("rank", Op::NotEq, 10u64),
        ]);
        assert!(evaluate(&bad, &ix, 5).is_none(), "Or silently dropped a branch");
    }

    #[test]
    fn not_complements_only_an_exact_set() {
        let ix = indexes();
        let f = Filter::Not(Box::new(Filter::cmp("tier", Op::Eq, "gold")));
        let sel = evaluate(&f, &ix, 5).unwrap();
        assert_eq!(sel.ordinals, set(&[1, 3]));
        assert!(sel.exact);

        // Complementing a superset would exclude documents that do match.
        let f = Filter::Not(Box::new(Filter::And(vec![
            Filter::cmp("tier", Op::Eq, "gold"),
            Filter::cmp("rank", Op::NotEq, 10u64),
        ])));
        assert!(evaluate(&f, &ix, 5).is_none(), "complemented an inexact set");
    }

    #[test]
    fn selection_is_always_a_superset_of_the_truth() {
        // The property that makes the loose contract safe: whatever the index
        // returns, no document that really matches is ever missing from it.
        let ix = indexes();
        let docs: Vec<(u32, crate::doc::Attrs)> = (0..5u32)
            .map(|o| {
                let mut a = crate::doc::Attrs::new();
                if o != 4 {
                    a.insert("rank".into(), Value::Uint([10, 20, 30, 20][o as usize]));
                }
                a.insert(
                    "tier".into(),
                    Value::from(["gold", "silver", "gold", "silver", "gold"][o as usize]),
                );
                (o, a)
            })
            .collect();

        let filters = vec![
            Filter::cmp("tier", Op::Eq, "gold"),
            Filter::And(vec![
                Filter::cmp("tier", Op::Eq, "gold"),
                Filter::cmp("rank", Op::Gte, 20u64),
            ]),
            Filter::Or(vec![
                Filter::cmp("tier", Op::Eq, "silver"),
                Filter::cmp("rank", Op::Eq, 30u64),
            ]),
            Filter::Not(Box::new(Filter::cmp("tier", Op::Eq, "gold"))),
        ];

        for f in filters {
            let Some(sel) = evaluate(&f, &ix, 5) else { continue };
            let truth: BTreeSet<u32> =
                docs.iter().filter(|(_, a)| f.matches(a)).map(|(o, _)| *o).collect();
            assert!(
                truth.is_subset(&sel.ordinals),
                "index dropped real matches for {f:?}: truth {truth:?}, selection {:?}",
                sel.ordinals
            );
            if sel.exact {
                assert_eq!(sel.ordinals, truth, "an exact selection disagreed with the truth");
            }
        }
    }
}
