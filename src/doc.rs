//! Document model, binary codec, distance, filters.
//!
//! One codec for every data object in the system: WAL entries, compacted
//! segments, and index cluster objects are all framed `Record`s. Uniform format
//! means one encoder, one decoder, one place for bugs.

use crate::value::{Type, Value};
use anyhow::{Result, bail};
use bytes::{BufMut, Bytes, BytesMut};
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type Id = u64;
pub type Attrs = BTreeMap<String, Value>;

/// One ranked result. The single hit shape used by the scan, the index, the
/// recall harness and the HTTP surface — so nothing converts between two
/// representations of the same idea.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    pub id: Id,
    pub score: f32,
    /// Populated only for the attributes a query asked for.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: Attrs,
}

impl Hit {
    pub fn new(id: Id, score: f32) -> Self {
        Self { id, score, attrs: Attrs::new() }
    }
}

/// A stored document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Doc {
    pub id: Id,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub attrs: Attrs,
}

impl Doc {
    pub fn new(id: Id, vector: Vec<f32>) -> Self {
        Self { id, vector, attrs: Attrs::new() }
    }

    pub fn with_attr(mut self, k: &str, v: impl Into<Value>) -> Self {
        self.attrs.insert(k.into(), v.into());
        self
    }

    /// Reinterpret JSON-inferred attributes according to a schema.
    pub fn coerce(&mut self, schema: &BTreeMap<String, Type>) -> Result<()> {
        for (k, v) in self.attrs.iter_mut() {
            if let Some(ty) = schema.get(k) {
                let taken = std::mem::replace(v, Value::Null);
                *v = taken
                    .coerce(*ty)
                    .map_err(|e| anyhow::anyhow!("attribute {k:?} on doc {}: {e}", self.id))?;
            }
        }
        Ok(())
    }
}

/// A mutation. Append-only: object storage cannot update in place, so an update
/// is an upsert, a partial update is a patch, and a delete is a tombstone.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Upsert(Doc),
    /// Merge these attributes into an existing document, leaving the rest — and
    /// the vector — alone. A `Null` value removes the attribute.
    Patch { id: Id, attrs: Attrs },
    Delete(Id),
}

const TAG_UPSERT: u8 = 0;
const TAG_DELETE: u8 = 1;
const TAG_PATCH: u8 = 2;

impl Record {
    pub fn id(&self) -> Id {
        match self {
            Record::Upsert(d) => d.id,
            Record::Patch { id, .. } | Record::Delete(id) => *id,
        }
    }

    pub fn encode(&self) -> Bytes {
        let mut b = BytesMut::new();
        match self {
            Record::Delete(id) => {
                b.put_u8(TAG_DELETE);
                b.put_u64_le(*id);
            }
            Record::Patch { id, attrs } => {
                b.put_u8(TAG_PATCH);
                b.put_u64_le(*id);
                put_attrs(&mut b, attrs);
            }
            Record::Upsert(d) => {
                b.put_u8(TAG_UPSERT);
                b.put_u64_le(d.id);
                b.put_u32_le(d.vector.len() as u32);
                for f in &d.vector {
                    b.put_f32_le(*f);
                }
                put_attrs(&mut b, &d.attrs);
            }
        }
        b.freeze()
    }

    pub fn decode(buf: &[u8]) -> Result<Record> {
        let mut pos = 0usize;
        let tag = take(buf, &mut pos, 1)?[0];
        Ok(match tag {
            TAG_DELETE => Record::Delete(get_u64(buf, &mut pos)?),
            TAG_PATCH => {
                let id = get_u64(buf, &mut pos)?;
                Record::Patch { id, attrs: get_attrs(buf, &mut pos)? }
            }
            TAG_UPSERT => {
                let id = get_u64(buf, &mut pos)?;
                let dim = u32::from_le_bytes(take(buf, &mut pos, 4)?.try_into().unwrap()) as usize;
                let raw = take(buf, &mut pos, dim.checked_mul(4).unwrap_or(usize::MAX))?;
                let vector = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Record::Upsert(Doc { id, vector, attrs: get_attrs(buf, &mut pos)? })
            }
            t => bail!("unknown record tag {t}"),
        })
    }
}

fn put_attrs(b: &mut BytesMut, attrs: &Attrs) {
    b.put_u16_le(attrs.len() as u16);
    for (k, v) in attrs {
        b.put_u16_le(k.len() as u16);
        b.put_slice(k.as_bytes());
        v.encode(b);
    }
}

fn get_attrs(buf: &[u8], pos: &mut usize) -> Result<Attrs> {
    let n = u16::from_le_bytes(take(buf, pos, 2)?.try_into().unwrap()) as usize;
    let mut out = Attrs::new();
    for _ in 0..n {
        let klen = u16::from_le_bytes(take(buf, pos, 2)?.try_into().unwrap()) as usize;
        let key = String::from_utf8(take(buf, pos, klen)?.to_vec())?;
        out.insert(key, Value::decode(buf, pos)?);
    }
    Ok(out)
}

fn take<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8]> {
    let Some(s) = buf.get(*pos..pos.checked_add(n).unwrap_or(usize::MAX)) else {
        bail!("truncated record: want {n} bytes at {pos}, have {}", buf.len());
    };
    *pos += n;
    Ok(s)
}

fn get_u64(buf: &[u8], pos: &mut usize) -> Result<u64> {
    Ok(u64::from_le_bytes(take(buf, pos, 8)?.try_into().unwrap()))
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

/// Comparison operators, spelled as turbopuffer spells them on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Eq,
    NotEq,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    NotIn,
    /// Shell-style wildcards: `*` any run, `?` one character.
    Glob,
    NotGlob,
    /// Case-insensitive glob.
    IGlob,
    NotIGlob,
    /// Substring for text, membership for array attributes.
    Contains,
    NotContains,
    Regex,
    NotRegex,
}

/// A filter expression.
///
/// Wire format matches turbopuffer's tuple syntax, so one grammar serves both
/// the native and the compatibility surface:
///
/// ```json
/// ["And", [["timestamp", "Gte", "2024-03-01T00:00:00Z"],
///          ["public", "Eq", true]]]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
    Cmp { key: String, op: Op, value: Value },
}

impl Filter {
    pub fn eq(key: &str, value: impl Into<Value>) -> Self {
        Filter::Cmp { key: key.into(), op: Op::Eq, value: value.into() }
    }
    pub fn cmp(key: &str, op: Op, value: impl Into<Value>) -> Self {
        Filter::Cmp { key: key.into(), op, value: value.into() }
    }

    /// Attributes this filter reads. Needed so a query can fetch what it must
    /// evaluate even when the caller did not ask for those attributes back.
    pub fn keys(&self, out: &mut Vec<String>) {
        match self {
            Filter::And(fs) | Filter::Or(fs) => fs.iter().for_each(|f| f.keys(out)),
            Filter::Not(f) => f.keys(out),
            Filter::Cmp { key, .. } => out.push(key.clone()),
        }
    }

    pub fn matches(&self, attrs: &Attrs) -> bool {
        match self {
            Filter::And(fs) => fs.iter().all(|f| f.matches(attrs)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(attrs)),
            Filter::Not(f) => !f.matches(attrs),
            Filter::Cmp { key, op, value } => {
                let actual = attrs.get(key).unwrap_or(&Value::Null);
                eval(actual, *op, value)
            }
        }
    }
}

fn eval(actual: &Value, op: Op, expected: &Value) -> bool {
    use std::cmp::Ordering::*;
    match op {
        Op::Eq => actual.equals(expected),
        Op::NotEq => !actual.equals(expected),
        // A missing attribute never satisfies an ordering comparison. Without
        // this, `Null` sorting low would make every absent value match `Lt`.
        Op::Gt | Op::Gte | Op::Lt | Op::Lte if actual.is_null() || expected.is_null() => false,
        Op::Gt => actual.compare(expected) == Some(Greater),
        Op::Gte => matches!(actual.compare(expected), Some(Greater | Equal)),
        Op::Lt => actual.compare(expected) == Some(Less),
        Op::Lte => matches!(actual.compare(expected), Some(Less | Equal)),
        Op::In => expected.elements().iter().any(|e| actual.equals(e)),
        Op::NotIn => !expected.elements().iter().any(|e| actual.equals(e)),
        Op::Contains => contains(actual, expected),
        Op::NotContains => !contains(actual, expected),
        Op::Glob => glob(actual, expected, true),
        Op::NotGlob => !glob(actual, expected, true),
        Op::IGlob => glob(actual, expected, false),
        Op::NotIGlob => !glob(actual, expected, false),
        Op::Regex => regex(actual, expected),
        Op::NotRegex => !regex(actual, expected),
    }
}

/// For an array attribute this is membership; for text it is substring.
fn contains(actual: &Value, expected: &Value) -> bool {
    match actual {
        Value::StringArray(_) | Value::UintArray(_) => {
            actual.elements().iter().any(|e| e.equals(expected))
        }
        _ => match (actual.as_text(), expected.as_text()) {
            (Some(h), Some(n)) => h.contains(&n),
            _ => false,
        },
    }
}

fn glob(actual: &Value, pattern: &Value, case_sensitive: bool) -> bool {
    let (Some(text), Some(pat)) = (actual.as_text(), pattern.as_text()) else {
        return false;
    };
    match glob_to_regex(&pat, case_sensitive) {
        Ok(re) => re.is_match(&text),
        Err(_) => false,
    }
}

fn regex(actual: &Value, pattern: &Value) -> bool {
    let (Some(text), Some(pat)) = (actual.as_text(), pattern.as_text()) else {
        return false;
    };
    match regex::Regex::new(&pat) {
        Ok(re) => re.is_match(&text),
        Err(_) => false,
    }
}

/// Translate a glob to an anchored regex.
///
/// Every character that is not a wildcard is escaped, so a pattern containing
/// regex metacharacters (`.`, `+`, `(`) matches them literally rather than being
/// silently reinterpreted as a regex.
pub fn glob_to_regex(pattern: &str, case_sensitive: bool) -> Result<regex::Regex> {
    let mut re = String::with_capacity(pattern.len() * 2 + 4);
    re.push('^');
    for c in pattern.chars() {
        match c {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            other => re.push_str(&regex::escape(&other.to_string())),
        }
    }
    re.push('$');
    Ok(regex::RegexBuilder::new(&re).case_insensitive(!case_sensitive).build()?)
}

// ---------------------------------------------------------------- filter JSON

impl Serialize for Filter {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        match self {
            Filter::And(fs) => {
                let mut t = s.serialize_tuple(2)?;
                t.serialize_element("And")?;
                t.serialize_element(fs)?;
                t.end()
            }
            Filter::Or(fs) => {
                let mut t = s.serialize_tuple(2)?;
                t.serialize_element("Or")?;
                t.serialize_element(fs)?;
                t.end()
            }
            Filter::Not(f) => {
                let mut t = s.serialize_tuple(2)?;
                t.serialize_element("Not")?;
                t.serialize_element(f)?;
                t.end()
            }
            Filter::Cmp { key, op, value } => {
                let mut t = s.serialize_tuple(3)?;
                t.serialize_element(key)?;
                t.serialize_element(op)?;
                t.serialize_element(value)?;
                t.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Filter {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Filter, D::Error> {
        let raw = serde_json::Value::deserialize(d)?;
        from_json_filter(&raw).map_err(de::Error::custom)
    }
}

fn from_json_filter(raw: &serde_json::Value) -> Result<Filter> {
    let Some(arr) = raw.as_array() else {
        bail!("a filter must be an array, got {raw}");
    };
    let head = arr.first().and_then(|v| v.as_str()).unwrap_or_default();

    // Connectives are recognised by their leading keyword. An attribute is
    // allowed to be named "And": a 3-element array is always a comparison,
    // which is why the arity is checked before the keyword.
    if arr.len() == 2 {
        let operands = &arr[1];
        return match head {
            "And" | "Or" => {
                let Some(items) = operands.as_array() else {
                    bail!("{head} expects an array of filters, got {operands}");
                };
                let parsed: Result<Vec<Filter>> = items.iter().map(from_json_filter).collect();
                let parsed = parsed?;
                if parsed.is_empty() {
                    bail!("{head} requires at least one operand");
                }
                Ok(if head == "And" { Filter::And(parsed) } else { Filter::Or(parsed) })
            }
            "Not" => Ok(Filter::Not(Box::new(from_json_filter(operands)?))),
            other => bail!("unknown filter connective {other:?}"),
        };
    }

    if arr.len() == 3 {
        let key = arr[0]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("filter attribute name must be a string"))?;
        let op: Op = serde_json::from_value(arr[1].clone())
            .map_err(|_| anyhow::anyhow!("unknown filter operator {}", arr[1]))?;
        let value = crate::value::from_json(arr[2].clone())?;
        return Ok(Filter::Cmp { key: key.into(), op, value });
    }

    bail!("a filter is [key, op, value] or [connective, operands], got {} elements", arr.len())
}

// ---------------------------------------------------------------- projection

/// Which attributes a query should return.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Include {
    #[default]
    None,
    All,
    Only(Vec<String>),
}

impl Include {
    fn project(&self, attrs: &Attrs) -> Attrs {
        match self {
            Include::None => Attrs::new(),
            Include::All => attrs.clone(),
            Include::Only(keys) => keys
                .iter()
                .filter_map(|k| attrs.get(k).map(|v| (k.clone(), v.clone())))
                .collect(),
        }
    }
}

impl Serialize for Include {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Include::None => s.serialize_bool(false),
            Include::All => s.serialize_bool(true),
            Include::Only(keys) => keys.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for Include {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Include, D::Error> {
        // `true` for everything, `false` for nothing, or an explicit list —
        // matching turbopuffer's include_attributes.
        let raw = serde_json::Value::deserialize(d)?;
        match raw {
            serde_json::Value::Bool(true) => Ok(Include::All),
            serde_json::Value::Bool(false) | serde_json::Value::Null => Ok(Include::None),
            serde_json::Value::Array(items) => {
                let keys: std::result::Result<Vec<String>, _> = items
                    .into_iter()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_string)
                            .ok_or_else(|| de::Error::custom("attribute names must be strings"))
                    })
                    .collect();
                Ok(Include::Only(keys?))
            }
            other => Err(de::Error::custom(format!(
                "include_attributes must be true, false, or a list of names, got {other}"
            ))),
        }
    }
}

/// Rank `docs` against `vector` and return the best `top_k`, descending.
pub fn top_k<'a>(
    docs: impl Iterator<Item = &'a Doc>,
    vector: &[f32],
    top_k: usize,
    filter: Option<&Filter>,
    include: &Include,
) -> Vec<Hit> {
    let mut scored: Vec<Hit> = docs
        .filter(|d| filter.is_none_or(|f| f.matches(&d.attrs)))
        .map(|d| Hit { id: d.id, score: cosine(vector, &d.vector), attrs: include.project(&d.attrs) })
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
            Record::Upsert(Doc::new(0, vec![])),
            Record::Upsert(
                Doc::new(9, vec![0.5; 8])
                    .with_attr("tenant", "acme")
                    .with_attr("count", 12u64)
                    .with_attr("ratio", -1.5f64)
                    .with_attr("public", true)
                    .with_attr("empty", "")
                    .with_attr("tags", Value::StringArray(vec!["a".into(), "b".into()]))
                    .with_attr("when", Value::Datetime(1_709_251_200_000_000_000))
                    .with_attr("who", Value::Uuid(uuid::Uuid::nil()))
                    .with_attr("missing", Value::Null),
            ),
            Record::Patch {
                id: 5,
                attrs: BTreeMap::from([
                    ("a".to_string(), Value::Uint(1)),
                    ("gone".to_string(), Value::Null),
                ]),
            },
        ] {
            let enc = r.encode();
            assert_eq!(Record::decode(&enc).unwrap(), r, "roundtrip changed the record");
        }
    }

    #[test]
    fn truncated_record_errors_not_panics() {
        let enc = Record::Upsert(Doc::new(1, vec![1.0, 2.0]).with_attr("k", "v")).encode();
        for cut in 0..enc.len() {
            assert!(Record::decode(&enc[..cut]).is_err(), "accepted a truncated record at {cut}");
        }
        assert!(Record::decode(&[99u8]).is_err(), "accepted an unknown tag");
    }

    #[test]
    fn cosine_is_scale_invariant_and_ordered() {
        let q = [1.0, 0.0];
        assert!((cosine(&q, &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&q, &[1.0, 0.0]) > cosine(&q, &[1.0, 1.0]));
        assert!(cosine(&q, &[1.0, 1.0]) > cosine(&q, &[0.0, 1.0]));
        assert_eq!(cosine(&q, &[0.0, 0.0]), 0.0, "zero vector must not produce NaN");
    }

    fn attrs() -> Attrs {
        BTreeMap::from([
            ("tenant".into(), Value::String("acme".into())),
            ("count".into(), Value::Uint(12)),
            ("public".into(), Value::Bool(true)),
            ("when".into(), Value::Datetime(crate::value::parse_datetime("2024-03-05").unwrap())),
            ("tags".into(), Value::StringArray(vec!["red".into(), "blue".into()])),
            ("path".into(), Value::String("docs/intro.md".into())),
        ])
    }

    #[test]
    fn equality_and_membership() {
        let a = attrs();
        assert!(Filter::eq("tenant", "acme").matches(&a));
        assert!(!Filter::eq("tenant", "globex").matches(&a));
        assert!(Filter::cmp("tenant", Op::NotEq, "globex").matches(&a));
        assert!(Filter::eq("count", 12u64).matches(&a));
        assert!(Filter::eq("public", true).matches(&a));

        assert!(
            Filter::cmp("tenant", Op::In, Value::StringArray(vec!["acme".into(), "x".into()]))
                .matches(&a)
        );
        assert!(
            Filter::cmp("tenant", Op::NotIn, Value::StringArray(vec!["x".into()])).matches(&a)
        );
        // Array attribute + Contains is membership.
        assert!(Filter::cmp("tags", Op::Contains, "red").matches(&a));
        assert!(!Filter::cmp("tags", Op::Contains, "green").matches(&a));
        // Text + Contains is substring.
        assert!(Filter::cmp("path", Op::Contains, "intro").matches(&a));
    }

    #[test]
    fn range_comparisons_need_comparable_types() {
        let a = attrs();
        assert!(Filter::cmp("count", Op::Gte, 12u64).matches(&a));
        assert!(Filter::cmp("count", Op::Gt, 11u64).matches(&a));
        assert!(!Filter::cmp("count", Op::Gt, 12u64).matches(&a));
        assert!(Filter::cmp("count", Op::Lte, 12u64).matches(&a));
        assert!(Filter::cmp("count", Op::Lt, 13u64).matches(&a));

        let march1 = Value::Datetime(crate::value::parse_datetime("2024-03-01").unwrap());
        assert!(Filter::cmp("when", Op::Gte, march1.clone()).matches(&a));
        assert!(!Filter::cmp("when", Op::Lt, march1).matches(&a));

        // A type mismatch must not match. If `compare` returned Equal or Less for
        // incomparable values, this would admit every document.
        assert!(!Filter::cmp("count", Op::Gte, "12").matches(&a));
        assert!(!Filter::cmp("tenant", Op::Gt, 1u64).matches(&a));
    }

    #[test]
    fn a_missing_attribute_matches_nothing_ordered() {
        let a = attrs();
        for op in [Op::Gt, Op::Gte, Op::Lt, Op::Lte] {
            assert!(
                !Filter::cmp("absent", op, 5u64).matches(&a),
                "{op:?} matched a missing attribute"
            );
        }
        assert!(!Filter::eq("absent", 5u64).matches(&a));
        // NotEq on a missing attribute is true: it is definitely not 5.
        assert!(Filter::cmp("absent", Op::NotEq, 5u64).matches(&a));
    }

    #[test]
    fn glob_escapes_regex_metacharacters() {
        let a = attrs();
        assert!(Filter::cmp("path", Op::Glob, "docs/*").matches(&a));
        assert!(Filter::cmp("path", Op::Glob, "docs/intro.md").matches(&a));
        assert!(Filter::cmp("path", Op::Glob, "docs/intro?md").matches(&a));
        assert!(!Filter::cmp("path", Op::Glob, "docs/*.txt").matches(&a));
        // Anchored: a bare prefix must not match.
        assert!(!Filter::cmp("path", Op::Glob, "docs").matches(&a));
        // The dot is literal, not "any character".
        let dotted = BTreeMap::from([("path".to_string(), Value::String("docsXintro".into()))]);
        assert!(
            !Filter::cmp("path", Op::Glob, "docs.intro").matches(&dotted),
            "glob treated '.' as a regex wildcard"
        );
        // Case sensitivity.
        assert!(!Filter::cmp("path", Op::Glob, "DOCS/*").matches(&a));
        assert!(Filter::cmp("path", Op::IGlob, "DOCS/*").matches(&a));
        assert!(Filter::cmp("path", Op::NotGlob, "other/*").matches(&a));
    }

    #[test]
    fn regex_operator_and_invalid_patterns() {
        let a = attrs();
        assert!(Filter::cmp("path", Op::Regex, "^docs/").matches(&a));
        assert!(Filter::cmp("path", Op::Regex, r"\.md$").matches(&a));
        assert!(Filter::cmp("path", Op::NotRegex, "^other/").matches(&a));
        // A malformed pattern must not match and must not panic.
        assert!(!Filter::cmp("path", Op::Regex, "[unclosed").matches(&a));
    }

    #[test]
    fn connectives_nest() {
        let a = attrs();
        let f = Filter::And(vec![
            Filter::cmp("when", Op::Gte, Value::Datetime(crate::value::parse_datetime("2024-03-01").unwrap())),
            Filter::eq("public", true),
            Filter::Or(vec![Filter::eq("tenant", "acme"), Filter::eq("tenant", "globex")]),
        ]);
        assert!(f.matches(&a));
        assert!(!Filter::Not(Box::new(f.clone())).matches(&a));
        assert!(!Filter::And(vec![f, Filter::eq("tenant", "nope")]).matches(&a));
    }

    #[test]
    fn filter_parses_turbopuffer_tuple_syntax() {
        let json = r#"["And", [["timestamp","Gte","2024-03-01T00:00:00Z"],["public","Eq",true]]]"#;
        let f: Filter = serde_json::from_str(json).unwrap();
        assert_eq!(
            f,
            Filter::And(vec![
                Filter::Cmp {
                    key: "timestamp".into(),
                    op: Op::Gte,
                    // Still a String: only a schema promotes it to Datetime.
                    value: Value::String("2024-03-01T00:00:00Z".into()),
                },
                Filter::Cmp { key: "public".into(), op: Op::Eq, value: Value::Bool(true) },
            ])
        );
        // And it serializes back to the same shape.
        let back: Filter = serde_json::from_str(&serde_json::to_string(&f).unwrap()).unwrap();
        assert_eq!(back, f);
    }

    #[test]
    fn filter_parse_errors_are_rejected() {
        for bad in [
            r#""just a string""#,
            r#"[]"#,
            r#"["And"]"#,
            r#"["And", []]"#,
            r#"["Nope", [["a","Eq",1]]]"#,
            r#"["a","Frobnicate",1]"#,
            r#"[1,"Eq",1]"#,
            r#"["a","Eq",1,"extra"]"#,
            r#"["And", "not an array"]"#,
        ] {
            assert!(serde_json::from_str::<Filter>(bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn an_attribute_may_be_named_like_a_connective() {
        // Three elements is always a comparison, so a column called "And" works.
        let f: Filter = serde_json::from_str(r#"["And","Eq","x"]"#).unwrap();
        assert_eq!(f, Filter::Cmp { key: "And".into(), op: Op::Eq, value: "x".into() });
    }

    #[test]
    fn filter_reports_the_keys_it_reads() {
        let f: Filter =
            serde_json::from_str(r#"["And",[["a","Eq",1],["Not",["b","Gt",2]],["Or",[["c","Eq",3]]]]]"#)
                .unwrap();
        let mut keys = vec![];
        f.keys(&mut keys);
        keys.sort();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn include_controls_the_projection() {
        let a = attrs();
        assert!(Include::None.project(&a).is_empty());
        assert_eq!(Include::All.project(&a).len(), a.len());
        let only = Include::Only(vec!["tenant".into(), "absent".into()]);
        let got = only.project(&a);
        assert_eq!(got.len(), 1, "a requested-but-absent attribute must be omitted, not null");
        assert_eq!(got["tenant"], Value::String("acme".into()));
    }

    #[test]
    fn include_parses_bool_or_list() {
        assert_eq!(serde_json::from_str::<Include>("true").unwrap(), Include::All);
        assert_eq!(serde_json::from_str::<Include>("false").unwrap(), Include::None);
        assert_eq!(serde_json::from_str::<Include>("null").unwrap(), Include::None);
        assert_eq!(
            serde_json::from_str::<Include>(r#"["a","b"]"#).unwrap(),
            Include::Only(vec!["a".into(), "b".into()])
        );
        assert!(serde_json::from_str::<Include>("5").is_err());
        assert!(serde_json::from_str::<Include>(r#"[1]"#).is_err());
    }

    #[test]
    fn top_k_breaks_ties_deterministically() {
        let docs: Vec<Doc> = [5u64, 1, 9, 3].iter().map(|i| Doc::new(*i, vec![1.0, 0.0])).collect();
        let got = top_k(docs.iter(), &[1.0, 0.0], 3, None, &Include::None);
        assert_eq!(
            got.iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![1, 3, 5],
            "ties must resolve by id, not by iteration order"
        );
    }

    #[test]
    fn top_k_respects_filter_order_and_projection() {
        let docs = vec![
            Doc::new(1, vec![1.0, 0.0]).with_attr("t", "a").with_attr("n", 1u64),
            Doc::new(2, vec![0.9, 0.1]).with_attr("t", "b").with_attr("n", 2u64),
            Doc::new(3, vec![0.0, 1.0]).with_attr("t", "a").with_attr("n", 3u64),
        ];
        let got = top_k(docs.iter(), &[1.0, 0.0], 2, None, &Include::None);
        assert_eq!(got.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 2]);
        assert!(got[0].attrs.is_empty());

        let got = top_k(
            docs.iter(),
            &[1.0, 0.0],
            2,
            Some(&Filter::eq("t", "a")),
            &Include::Only(vec!["n".into()]),
        );
        assert_eq!(got.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(got[0].attrs["n"], Value::Uint(1));
        assert_eq!(got[0].attrs.len(), 1, "projection leaked an unrequested attribute");
    }

    #[test]
    fn schema_coercion_promotes_strings_to_declared_types() {
        let mut d = Doc::new(1, vec![1.0])
            .with_attr("when", "2024-03-01T00:00:00Z")
            .with_attr("who", "550e8400-e29b-41d4-a716-446655440000")
            .with_attr("plain", "left alone");
        let schema = BTreeMap::from([
            ("when".to_string(), Type::Datetime),
            ("who".to_string(), Type::Uuid),
        ]);
        d.coerce(&schema).unwrap();
        assert!(matches!(d.attrs["when"], Value::Datetime(_)));
        assert!(matches!(d.attrs["who"], Value::Uuid(_)));
        assert_eq!(d.attrs["plain"], Value::String("left alone".into()));

        // A value that cannot be coerced names the attribute and the document.
        let mut bad = Doc::new(77, vec![1.0]).with_attr("when", "not a date");
        let err = bad.coerce(&schema).unwrap_err().to_string();
        assert!(err.contains("when") && err.contains("77"), "unhelpful error: {err}");
    }
}
