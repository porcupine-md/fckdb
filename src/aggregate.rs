//! Aggregations: `Count`, `Sum`, `Min`, `Max`, `Avg`, with optional grouping.
//!
//! Wire shapes follow turbopuffer's:
//!
//! ```json
//! { "aggregate_by": { "total": ["Sum", "amount"], "n": ["Count"] },
//!   "group_by": ["color", ["ForEachUnique", "tags"]] }
//! ```
//!
//! Ungrouped results come back under `aggregations`; grouped ones under
//! `aggregation_groups`, where each row carries its group key alongside the
//! computed values.

use crate::doc::{Attrs, Doc};
use crate::value::Value;
use anyhow::{Result, bail};
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Agg {
    /// Documents in the group. With an attribute, documents whose value for it
    /// is present — the usual SQL distinction between `count(*)` and
    /// `count(col)`.
    Count(Option<String>),
    Sum(String),
    Min(String),
    Max(String),
    Avg(String),
}

impl Agg {
    fn attribute(&self) -> Option<&str> {
        match self {
            Agg::Count(a) => a.as_deref(),
            Agg::Sum(a) | Agg::Min(a) | Agg::Max(a) | Agg::Avg(a) => Some(a),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Agg::Count(_) => "Count",
            Agg::Sum(_) => "Sum",
            Agg::Min(_) => "Min",
            Agg::Max(_) => "Max",
            Agg::Avg(_) => "Avg",
        }
    }
}

impl<'de> Deserialize<'de> for Agg {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Agg, D::Error> {
        let raw = serde_json::Value::deserialize(d)?;
        parse_agg(&raw).map_err(de::Error::custom)
    }
}

impl Serialize for Agg {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(None)?;
        seq.serialize_element(self.name())?;
        if let Some(a) = self.attribute() {
            seq.serialize_element(a)?;
        }
        seq.end()
    }
}

fn parse_agg(raw: &serde_json::Value) -> Result<Agg> {
    let Some(arr) = raw.as_array() else {
        bail!("an aggregation must be an array like [\"Count\"] or [\"Sum\", \"attr\"], got {raw}");
    };
    let func = arr
        .first()
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("aggregation function must be a string"))?;
    let attr = match arr.get(1) {
        None => None,
        Some(v) => Some(
            v.as_str()
                .ok_or_else(|| anyhow::anyhow!("aggregation attribute must be a string"))?
                .to_string(),
        ),
    };
    if arr.len() > 2 {
        bail!("aggregation {func} takes at most one attribute, got {} elements", arr.len());
    }

    let need = |a: Option<String>| -> Result<String> {
        a.ok_or_else(|| anyhow::anyhow!("aggregation {func} requires an attribute"))
    };
    Ok(match func {
        "Count" => Agg::Count(attr),
        "Sum" => Agg::Sum(need(attr)?),
        "Min" => Agg::Min(need(attr)?),
        "Max" => Agg::Max(need(attr)?),
        "Avg" => Agg::Avg(need(attr)?),
        other => bail!("unknown aggregation function {other:?}"),
    })
}

/// One grouping key.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupKey {
    Attribute(String),
    /// Explode an array attribute: a document joins one group per distinct
    /// element, so a document tagged `["a","b"]` is counted under both.
    ForEachUnique(String),
}

impl GroupKey {
    pub fn attribute(&self) -> &str {
        match self {
            GroupKey::Attribute(a) | GroupKey::ForEachUnique(a) => a,
        }
    }

    /// The values this document contributes for this key. More than one only
    /// for `ForEachUnique`.
    fn values(&self, attrs: &Attrs) -> Vec<Value> {
        let raw = attrs.get(self.attribute()).cloned().unwrap_or(Value::Null);
        match self {
            GroupKey::Attribute(_) => vec![raw],
            GroupKey::ForEachUnique(_) => {
                if raw.is_null() {
                    return vec![Value::Null];
                }
                let mut elements = raw.elements();
                // Distinct, so a document listing a tag twice is not counted
                // twice in that tag's group.
                elements.sort_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal));
                elements.dedup_by(|a, b| a.equals(b));
                if elements.is_empty() { vec![Value::Null] } else { elements }
            }
        }
    }
}

impl<'de> Deserialize<'de> for GroupKey {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<GroupKey, D::Error> {
        let raw = serde_json::Value::deserialize(d)?;
        match &raw {
            serde_json::Value::String(s) => Ok(GroupKey::Attribute(s.clone())),
            serde_json::Value::Array(items) => {
                let op = items.first().and_then(|v| v.as_str());
                let attr = items.get(1).and_then(|v| v.as_str());
                match (op, attr, items.len()) {
                    (Some("ForEachUnique"), Some(a), 2) => {
                        Ok(GroupKey::ForEachUnique(a.to_string()))
                    }
                    _ => Err(de::Error::custom(format!(
                        "a group_by entry is an attribute name or [\"ForEachUnique\", \"attr\"], \
                         got {raw}"
                    ))),
                }
            }
            other => Err(de::Error::custom(format!("group_by entry must be a string, got {other}"))),
        }
    }
}

impl Serialize for GroupKey {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            GroupKey::Attribute(a) => s.serialize_str(a),
            GroupKey::ForEachUnique(a) => {
                use serde::ser::SerializeSeq;
                let mut seq = s.serialize_seq(Some(2))?;
                seq.serialize_element("ForEachUnique")?;
                seq.serialize_element(a)?;
                seq.end()
            }
        }
    }
}

/// Running state for one aggregation.
#[derive(Debug, Clone, Default)]
struct Accumulator {
    count: u64,
    sum: f64,
    /// False once a non-integer value is seen, so a sum of integers stays an
    /// integer instead of coming back as `10.0`.
    integral: bool,
    /// True once any value has been summed, distinguishing "no rows" from "sum
    /// of zero".
    seen: bool,
    min: Option<Value>,
    max: Option<Value>,
}

impl Accumulator {
    fn new() -> Self {
        Self { integral: true, ..Default::default() }
    }
}

/// Aggregate `docs`, optionally grouped.
///
/// Returns (ungrouped results, grouped results). Exactly one is populated:
/// turbopuffer reports `aggregations` when there is no grouping and
/// `aggregation_groups` when there is, and a caller must be able to tell an
/// empty grouped result from an ungrouped one.
pub fn aggregate<'a>(
    docs: impl Iterator<Item = &'a Doc>,
    aggs: &BTreeMap<String, Agg>,
    group_by: &[GroupKey],
) -> Result<(BTreeMap<String, Value>, Vec<AggregationGroup>)> {
    if aggs.is_empty() {
        bail!("aggregate_by requires at least one aggregation");
    }

    // Group key -> per-label accumulators. BTreeMap so groups come out in a
    // deterministic order rather than hash order.
    let mut groups: BTreeMap<Vec<OrderedValue>, BTreeMap<String, Accumulator>> = BTreeMap::new();
    let fresh =
        || -> BTreeMap<String, Accumulator> { aggs.keys().map(|k| (k.clone(), Accumulator::new())).collect() };

    for doc in docs {
        for key in expand_keys(group_by, &doc.attrs) {
            let slot = groups.entry(key).or_insert_with(fresh);
            for (label, agg) in aggs {
                accumulate(slot.get_mut(label).expect("label present"), agg, &doc.attrs)?;
            }
        }
    }

    if group_by.is_empty() {
        // One implicit group. An empty namespace still reports a zero count,
        // which is different from reporting nothing.
        let state = groups.into_values().next().unwrap_or_else(fresh);
        let mut out = BTreeMap::new();
        for (label, agg) in aggs {
            out.insert(label.clone(), finalize(&state[label], agg));
        }
        return Ok((out, vec![]));
    }

    let rows = groups
        .into_iter()
        .map(|(key, state)| AggregationGroup {
            key: group_by
                .iter()
                .zip(key)
                .map(|(g, v)| (g.attribute().to_string(), v.0))
                .collect(),
            values: aggs.iter().map(|(l, a)| (l.clone(), finalize(&state[l], a))).collect(),
        })
        .collect();
    Ok((BTreeMap::new(), rows))
}

/// The cartesian product of each key's values, so a document with two exploded
/// array attributes lands in every combination.
fn expand_keys(group_by: &[GroupKey], attrs: &Attrs) -> Vec<Vec<OrderedValue>> {
    if group_by.is_empty() {
        return vec![vec![]];
    }
    let mut combos: Vec<Vec<OrderedValue>> = vec![vec![]];
    for key in group_by {
        let values = key.values(attrs);
        let mut next = Vec::with_capacity(combos.len() * values.len());
        for base in &combos {
            for v in &values {
                let mut extended = base.clone();
                extended.push(OrderedValue(v.clone()));
                next.push(extended);
            }
        }
        combos = next;
    }
    combos
}

fn accumulate(acc: &mut Accumulator, agg: &Agg, attrs: &Attrs) -> Result<()> {
    let value = agg.attribute().map(|a| attrs.get(a).cloned().unwrap_or(Value::Null));

    match agg {
        // count(*) counts rows; count(col) counts present values.
        Agg::Count(None) => acc.count += 1,
        Agg::Count(Some(_)) => {
            if value.as_ref().is_some_and(|v| !v.is_null()) {
                acc.count += 1;
            }
        }
        Agg::Sum(a) | Agg::Avg(a) => {
            let Some(v) = value else { return Ok(()) };
            if v.is_null() {
                return Ok(());
            }
            let n = numeric(&v).ok_or_else(|| {
                anyhow::anyhow!(
                    "{} needs a numeric attribute; {a:?} holds a {}",
                    agg.name(),
                    v.type_of().map_or("null", |t| t.name())
                )
            })?;
            if !matches!(v, Value::Uint(_) | Value::Int(_)) {
                acc.integral = false;
            }
            acc.sum += n;
            acc.count += 1;
            acc.seen = true;
        }
        Agg::Min(_) | Agg::Max(_) => {
            let Some(v) = value else { return Ok(()) };
            if v.is_null() {
                // Nulls are skipped, not treated as the smallest value: a
                // missing measurement is not a measurement of zero.
                return Ok(());
            }
            let target = if matches!(agg, Agg::Min(_)) { &mut acc.min } else { &mut acc.max };
            let wins = match target {
                None => true,
                Some(current) => match v.compare(current) {
                    Some(std::cmp::Ordering::Less) => matches!(agg, Agg::Min(_)),
                    Some(std::cmp::Ordering::Greater) => matches!(agg, Agg::Max(_)),
                    _ => false,
                },
            };
            if wins {
                *target = Some(v);
            }
            acc.count += 1;
        }
    }
    Ok(())
}

fn finalize(acc: &Accumulator, agg: &Agg) -> Value {
    match agg {
        Agg::Count(_) => Value::Uint(acc.count),
        Agg::Sum(_) => {
            if !acc.seen {
                // No values at all: zero, not null. Summing nothing is zero, and
                // a caller charting totals wants a number.
                return Value::Uint(0);
            }
            if acc.integral && acc.sum.fract() == 0.0 && acc.sum.abs() < 9e15 {
                if acc.sum < 0.0 {
                    Value::Int(acc.sum as i64)
                } else {
                    Value::Uint(acc.sum as u64)
                }
            } else {
                Value::F64(acc.sum)
            }
        }
        // An average of nothing is undefined, so null rather than zero.
        Agg::Avg(_) => {
            if acc.count == 0 { Value::Null } else { Value::F64(acc.sum / acc.count as f64) }
        }
        Agg::Min(_) => acc.min.clone().unwrap_or(Value::Null),
        Agg::Max(_) => acc.max.clone().unwrap_or(Value::Null),
    }
}

fn numeric(v: &Value) -> Option<f64> {
    match v {
        Value::Uint(u) => Some(*u as f64),
        Value::Int(i) => Some(*i as f64),
        Value::F64(f) => Some(*f),
        Value::Datetime(n) => Some(*n as f64),
        _ => None,
    }
}

/// A `Value` with a total order, so it can key a `BTreeMap`.
#[derive(Debug, Clone, PartialEq)]
struct OrderedValue(Value);

impl Eq for OrderedValue {}
impl PartialOrd for OrderedValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Values of different types never compare; fall back to their type name
        // so grouping stays deterministic instead of collapsing distinct keys.
        self.0.compare(&other.0).unwrap_or_else(|| {
            let name = |v: &Value| v.type_of().map_or("", |t| t.name());
            name(&self.0).cmp(name(&other.0))
        })
    }
}

/// One grouped result: the group's key values, plus each aggregation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregationGroup {
    /// Attribute name to the group's value for it.
    pub key: BTreeMap<String, Value>,
    /// Aggregation label to computed value.
    pub values: BTreeMap<String, Value>,
}

impl AggregationGroup {
    /// Flatten key and values into one object, which is how turbopuffer returns
    /// a group row.
    pub fn flatten(&self) -> BTreeMap<String, Value> {
        self.key.iter().chain(self.values.iter()).map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs() -> Vec<Doc> {
        vec![
            Doc::new(1u64, vec![1.0])
                .with_attr("color", "red")
                .with_attr("size", "L")
                .with_attr("price", 10u64)
                .with_attr("tags", Value::StringArray(vec!["a".into(), "b".into()])),
            Doc::new(2u64, vec![1.0])
                .with_attr("color", "red")
                .with_attr("size", "L")
                .with_attr("price", 20u64)
                .with_attr("tags", Value::StringArray(vec!["b".into()])),
            Doc::new(3u64, vec![1.0])
                .with_attr("color", "blue")
                .with_attr("size", "XL")
                .with_attr("price", 5u64)
                .with_attr("tags", Value::StringArray(vec![])),
            // No price and no color at all.
            Doc::new(4u64, vec![1.0]).with_attr("size", "L"),
        ]
    }

    fn agg(spec: &str) -> BTreeMap<String, Agg> {
        serde_json::from_str(spec).unwrap()
    }

    fn ungrouped(spec: &str) -> BTreeMap<String, Value> {
        aggregate(docs().iter(), &agg(spec), &[]).unwrap().0
    }

    #[test]
    fn count_star_counts_rows_count_column_counts_values() {
        assert_eq!(ungrouped(r#"{"n":["Count"]}"#)["n"], Value::Uint(4));
        // Document 4 has no price, so count(price) is one lower.
        assert_eq!(ungrouped(r#"{"n":["Count","price"]}"#)["n"], Value::Uint(3));
        assert_eq!(ungrouped(r#"{"n":["Count","nosuch"]}"#)["n"], Value::Uint(0));
    }

    #[test]
    fn sum_of_integers_stays_an_integer() {
        assert_eq!(ungrouped(r#"{"s":["Sum","price"]}"#)["s"], Value::Uint(35));
        // A float anywhere makes the whole sum a float.
        let mut d = docs();
        d.push(Doc::new(5u64, vec![1.0]).with_attr("price", 0.5f64));
        let got = aggregate(d.iter(), &agg(r#"{"s":["Sum","price"]}"#), &[]).unwrap().0;
        assert_eq!(got["s"], Value::F64(35.5));
    }

    #[test]
    fn summing_nothing_is_zero_but_averaging_nothing_is_null() {
        // A total of no rows is zero — a caller charting totals wants a number.
        assert_eq!(ungrouped(r#"{"s":["Sum","nosuch"]}"#)["s"], Value::Uint(0));
        // An average of no rows is undefined, and reporting zero would be a lie.
        assert_eq!(ungrouped(r#"{"a":["Avg","nosuch"]}"#)["a"], Value::Null);
        assert_eq!(ungrouped(r#"{"m":["Min","nosuch"]}"#)["m"], Value::Null);
        assert_eq!(ungrouped(r#"{"m":["Max","nosuch"]}"#)["m"], Value::Null);
    }

    #[test]
    fn min_max_avg_skip_nulls() {
        assert_eq!(ungrouped(r#"{"m":["Min","price"]}"#)["m"], Value::Uint(5));
        assert_eq!(ungrouped(r#"{"m":["Max","price"]}"#)["m"], Value::Uint(20));
        // 35/3, not 35/4: the document with no price is not a price of zero.
        match ungrouped(r#"{"a":["Avg","price"]}"#)["a"] {
            Value::F64(v) => assert!((v - 35.0 / 3.0).abs() < 1e-9, "got {v}"),
            ref other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn min_max_work_on_strings_and_datetimes() {
        assert_eq!(ungrouped(r#"{"m":["Min","color"]}"#)["m"], Value::from("blue"));
        assert_eq!(ungrouped(r#"{"m":["Max","color"]}"#)["m"], Value::from("red"));

        let d = vec![
            Doc::new(1u64, vec![1.0]).with_attr("t", Value::Datetime(100)),
            Doc::new(2u64, vec![1.0]).with_attr("t", Value::Datetime(50)),
        ];
        let got = aggregate(d.iter(), &agg(r#"{"m":["Min","t"]}"#), &[]).unwrap().0;
        assert_eq!(got["m"], Value::Datetime(50));
    }

    #[test]
    fn summing_a_non_numeric_attribute_is_an_error_not_zero() {
        let err = aggregate(docs().iter(), &agg(r#"{"s":["Sum","color"]}"#), &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("color") && err.contains("numeric"), "unhelpful error: {err}");
    }

    #[test]
    fn grouping_splits_the_aggregate() {
        let (ungrouped, groups) = aggregate(
            docs().iter(),
            &agg(r#"{"n":["Count"],"s":["Sum","price"]}"#),
            &[GroupKey::Attribute("color".into())],
        )
        .unwrap();
        assert!(ungrouped.is_empty(), "grouped results must not also report an ungrouped total");
        assert_eq!(groups.len(), 3, "expected red, blue, and the missing-colour group");

        let by_color: BTreeMap<String, &AggregationGroup> = groups
            .iter()
            .map(|g| (format!("{:?}", g.key["color"]), g))
            .collect();
        let red = by_color.values().find(|g| g.key["color"] == Value::from("red")).unwrap();
        assert_eq!(red.values["n"], Value::Uint(2));
        assert_eq!(red.values["s"], Value::Uint(30));

        // A document missing the attribute forms its own null group rather than
        // being dropped, so counts across groups still add up to the total.
        let missing = groups.iter().find(|g| g.key["color"].is_null()).unwrap();
        assert_eq!(missing.values["n"], Value::Uint(1));
        let total: u64 = groups
            .iter()
            .map(|g| match g.values["n"] {
                Value::Uint(n) => n,
                _ => 0,
            })
            .sum();
        assert_eq!(total, 4, "group counts did not sum to the document count");
    }

    #[test]
    fn grouping_by_several_attributes_uses_the_combination() {
        let (_, groups) = aggregate(
            docs().iter(),
            &agg(r#"{"n":["Count"]}"#),
            &[GroupKey::Attribute("color".into()), GroupKey::Attribute("size".into())],
        )
        .unwrap();
        // (red,L) x2, (blue,XL) x1, (null,L) x1
        assert_eq!(groups.len(), 3);
        let red_l = groups
            .iter()
            .find(|g| g.key["color"] == Value::from("red") && g.key["size"] == Value::from("L"))
            .unwrap();
        assert_eq!(red_l.values["n"], Value::Uint(2));
        assert_eq!(red_l.flatten()["n"], Value::Uint(2));
        assert_eq!(red_l.flatten()["color"], Value::from("red"));
    }

    #[test]
    fn for_each_unique_explodes_array_attributes() {
        let (_, groups) = aggregate(
            docs().iter(),
            &agg(r#"{"n":["Count"]}"#),
            &[GroupKey::ForEachUnique("tags".into())],
        )
        .unwrap();

        let get = |tag: &str| {
            groups
                .iter()
                .find(|g| g.key["tags"] == Value::from(tag))
                .map(|g| g.values["n"].clone())
        };
        // Document 1 is tagged a and b; document 2 only b.
        assert_eq!(get("a"), Some(Value::Uint(1)));
        assert_eq!(get("b"), Some(Value::Uint(2)));
        // An empty array and a missing attribute both land in the null group.
        let null_group = groups.iter().find(|g| g.key["tags"].is_null()).unwrap();
        assert_eq!(null_group.values["n"], Value::Uint(2));

        // Exploding means group counts may EXCEED the document count, which is
        // the intended behaviour and worth being explicit about.
        let total: u64 = groups
            .iter()
            .map(|g| match g.values["n"] {
                Value::Uint(n) => n,
                _ => 0,
            })
            .sum();
        assert_eq!(total, 5, "document 1 should be counted under both of its tags");
    }

    #[test]
    fn a_repeated_array_element_is_counted_once() {
        let d = vec![
            Doc::new(1u64, vec![1.0])
                .with_attr("tags", Value::StringArray(vec!["a".into(), "a".into()])),
        ];
        let (_, groups) =
            aggregate(d.iter(), &agg(r#"{"n":["Count"]}"#), &[GroupKey::ForEachUnique("tags".into())])
                .unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].values["n"], Value::Uint(1), "a duplicate tag double-counted");
    }

    #[test]
    fn groups_come_back_in_a_deterministic_order() {
        let a = aggregate(docs().iter(), &agg(r#"{"n":["Count"]}"#), &[GroupKey::Attribute("color".into())]).unwrap().1;
        let b = aggregate(docs().iter(), &agg(r#"{"n":["Count"]}"#), &[GroupKey::Attribute("color".into())]).unwrap().1;
        assert_eq!(a, b, "group order changed between identical calls");
    }

    #[test]
    fn empty_input_still_reports_a_zero_count() {
        let (ungrouped, groups) =
            aggregate(std::iter::empty(), &agg(r#"{"n":["Count"],"s":["Sum","x"]}"#), &[]).unwrap();
        assert_eq!(ungrouped["n"], Value::Uint(0));
        assert_eq!(ungrouped["s"], Value::Uint(0));
        assert!(groups.is_empty());

        // Grouped over nothing yields no groups, not one empty group.
        let (_, groups) = aggregate(
            std::iter::empty(),
            &agg(r#"{"n":["Count"]}"#),
            &[GroupKey::Attribute("color".into())],
        )
        .unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn aggregation_specs_parse_and_reject() {
        assert_eq!(parse_agg(&serde_json::json!(["Count"])).unwrap(), Agg::Count(None));
        assert_eq!(
            parse_agg(&serde_json::json!(["Count", "id"])).unwrap(),
            Agg::Count(Some("id".into()))
        );
        assert_eq!(parse_agg(&serde_json::json!(["Sum", "n"])).unwrap(), Agg::Sum("n".into()));

        for bad in [
            serde_json::json!("Count"),
            serde_json::json!([]),
            serde_json::json!(["Sum"]),
            serde_json::json!(["Avg"]),
            serde_json::json!(["Frobnicate", "n"]),
            serde_json::json!(["Sum", "a", "b"]),
            serde_json::json!(["Sum", 5]),
        ] {
            assert!(parse_agg(&bad).is_err(), "accepted {bad}");
        }

        // Round-trips through JSON.
        let spec = agg(r#"{"a":["Count"],"b":["Sum","n"]}"#);
        let back: BTreeMap<String, Agg> =
            serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(back, spec);
    }

    #[test]
    fn group_keys_parse_and_reject() {
        let keys: Vec<GroupKey> =
            serde_json::from_str(r#"["color", ["ForEachUnique", "tags"]]"#).unwrap();
        assert_eq!(
            keys,
            vec![GroupKey::Attribute("color".into()), GroupKey::ForEachUnique("tags".into())]
        );
        let back: Vec<GroupKey> =
            serde_json::from_str(&serde_json::to_string(&keys).unwrap()).unwrap();
        assert_eq!(back, keys);

        for bad in [r#"[5]"#, r#"[["Explode","tags"]]"#, r#"[["ForEachUnique"]]"#, r#"[[]]"#] {
            assert!(serde_json::from_str::<Vec<GroupKey>>(bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn an_empty_aggregate_by_is_refused() {
        let empty: BTreeMap<String, Agg> = BTreeMap::new();
        assert!(aggregate(docs().iter(), &empty, &[]).is_err());
    }
}
