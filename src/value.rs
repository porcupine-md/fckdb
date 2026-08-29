//! Typed attribute values.
//!
//! The engine started with `String -> String` attributes, which is enough for
//! equality filters and nothing else. Range filters (`Gte` on a timestamp),
//! ordering by attribute, and aggregations all need to know what a value *is*.
//! This module is that foundation.
//!
//! Two representations, deliberately separate:
//!
//!   - **JSON**, what clients send. Types are inferred where JSON carries them
//!     (number, bool, string, array) and coerced via the namespace schema where
//!     it does not (`uuid` and `datetime` both arrive as strings).
//!   - **binary**, what goes to object storage. Tag byte plus payload, stable
//!     across versions because a tag is never reused.

use anyhow::{Result, bail};
use bytes::{BufMut, BytesMut};
use serde::de::{self, Deserializer};
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// Wire tags. Append only — changing one silently reinterprets stored bytes.
const T_NULL: u8 = 0;
const T_BOOL: u8 = 1;
const T_UINT: u8 = 2;
const T_INT: u8 = 3;
const T_F64: u8 = 4;
const T_STRING: u8 = 5;
const T_DATETIME: u8 = 6;
const T_UUID: u8 = 7;
const T_STRING_ARRAY: u8 = 8;
const T_UINT_ARRAY: u8 = 9;
const T_INT_ARRAY: u8 = 10;
const T_SPARSE: u8 = 11;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Uint(u64),
    Int(i64),
    F64(f64),
    String(String),
    /// Nanoseconds since the Unix epoch, UTC. Nanoseconds because that is what
    /// turbopuffer's ISO 8601 output carries, and truncating to millis would make
    /// round-trips lossy.
    Datetime(i64),
    Uuid(uuid::Uuid),
    StringArray(Vec<String>),
    UintArray(Vec<u64>),
    IntArray(Vec<i64>),
    /// A sparse vector: named dimensions to weights.
    ///
    /// turbopuffer spells the type `{}f16`. Weights are stored as f32 here, which
    /// is a lossless superset of f16 at twice the size — worth revisiting if
    /// sparse namespaces get large.
    Sparse(BTreeMap<String, f32>),
}

/// A declared attribute type, as it appears in a namespace schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Type {
    Bool,
    Uint,
    Int,
    /// Spelled `float` on the wire, matching turbopuffer's schema vocabulary.
    #[serde(rename = "float")]
    F64,
    String,
    Datetime,
    Uuid,
    #[serde(rename = "[]string")]
    StringArray,
    #[serde(rename = "[]uint")]
    UintArray,
    #[serde(rename = "[]int")]
    IntArray,
    #[serde(rename = "{}f16")]
    SparseF16,
}

impl Type {
    pub fn name(self) -> &'static str {
        match self {
            Type::Bool => "bool",
            Type::Uint => "uint",
            Type::Int => "int",
            Type::F64 => "float",
            Type::String => "string",
            Type::Datetime => "datetime",
            Type::Uuid => "uuid",
            Type::StringArray => "[]string",
            Type::UintArray => "[]uint",
            Type::IntArray => "[]int",
            Type::SparseF16 => "{}f16",
        }
    }
}

impl Value {
    /// The declared type of this value. `Null` has none: it is the absence of a
    /// value, not a type, which is why a null never conflicts with a schema.
    pub fn type_of(&self) -> Option<Type> {
        Some(match self {
            Value::Null => return None,
            Value::Bool(_) => Type::Bool,
            Value::Uint(_) => Type::Uint,
            Value::Int(_) => Type::Int,
            Value::F64(_) => Type::F64,
            Value::String(_) => Type::String,
            Value::Datetime(_) => Type::Datetime,
            Value::Uuid(_) => Type::Uuid,
            Value::StringArray(_) => Type::StringArray,
            Value::UintArray(_) => Type::UintArray,
            Value::IntArray(_) => Type::IntArray,
            Value::Sparse(_) => Type::SparseF16,
        })
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Reinterpret a JSON-inferred value as a declared type.
    ///
    /// JSON cannot express a uuid or a timestamp, so both arrive as strings and
    /// only the schema knows better. Widening between numeric types is allowed
    /// where it is lossless; everything else is an error rather than a silent
    /// reinterpretation.
    pub fn coerce(self, ty: Type) -> Result<Value> {
        if self.is_null() {
            return Ok(Value::Null);
        }
        if self.type_of() == Some(ty) {
            return Ok(self);
        }
        Ok(match (self, ty) {
            (Value::String(s), Type::Uuid) => Value::Uuid(uuid::Uuid::parse_str(&s)?),
            (Value::String(s), Type::Datetime) => Value::Datetime(parse_datetime(&s)?),
            (Value::Uint(u), Type::Int) => Value::Int(i64::try_from(u)?),
            (Value::Uint(u), Type::F64) => Value::F64(u as f64),
            (Value::Int(i), Type::F64) => Value::F64(i as f64),
            (Value::Int(i), Type::Uint) => Value::Uint(u64::try_from(i)?),
            (Value::Uint(u), Type::Datetime) => Value::Datetime(i64::try_from(u)?),
            (Value::UintArray(a), Type::StringArray) => {
                Value::StringArray(a.iter().map(|u| u.to_string()).collect())
            }
            // An empty JSON array is ambiguous; accept it as either array type.
            (Value::StringArray(a), Type::UintArray) if a.is_empty() => Value::UintArray(vec![]),
            (Value::StringArray(a), Type::IntArray) if a.is_empty() => Value::IntArray(vec![]),
            (Value::UintArray(a), Type::IntArray) => Value::IntArray(
                a.iter().map(|u| i64::try_from(*u)).collect::<std::result::Result<_, _>>()?,
            ),
            (v, ty) => bail!(
                "cannot interpret {} as {}",
                v.type_of().map_or("null", |t| t.name()),
                ty.name()
            ),
        })
    }

    /// Ordering for range filters and sorting.
    ///
    /// `None` means the two are not comparable, which callers must treat as "does
    /// not match" rather than as equal — otherwise a `Gte` against the wrong type
    /// silently admits every document.
    pub fn compare(&self, other: &Value) -> Option<Ordering> {
        use Value::*;
        match (self, other) {
            (Null, Null) => Some(Ordering::Equal),
            // Null orders below everything, so sorts are total, but it is never
            // *equal* to a real value.
            (Null, _) => Some(Ordering::Less),
            (_, Null) => Some(Ordering::Greater),
            (Bool(a), Bool(b)) => Some(a.cmp(b)),
            (String(a), String(b)) => Some(a.cmp(b)),
            (Uuid(a), Uuid(b)) => Some(a.cmp(b)),
            (Datetime(a), Datetime(b)) => Some(a.cmp(b)),
            (StringArray(a), StringArray(b)) => Some(a.cmp(b)),
            (UintArray(a), UintArray(b)) => Some(a.cmp(b)),
            (IntArray(a), IntArray(b)) => Some(a.cmp(b)),
            // Sparse vectors have no meaningful order; they are ranked, not
            // compared. Returning an ordering would let a range filter on one
            // silently succeed.
            (Sparse(_), Sparse(_)) => None,
            // Numeric comparison across integer widths goes through i128 so no
            // value is lost; only a float operand falls back to f64.
            (F64(_), _) | (_, F64(_)) => {
                Some(as_f64(self)?.total_cmp(&as_f64(other)?))
            }
            (Uint(_) | Int(_), Uint(_) | Int(_)) => Some(as_i128(self)?.cmp(&as_i128(other)?)),
            _ => None,
        }
    }

    /// Membership, for `In` / `NotIn` and for array attributes with `Contains`.
    pub fn equals(&self, other: &Value) -> bool {
        if self.is_null() || other.is_null() {
            // NULL is not equal to anything, including NULL, for filter purposes.
            // Use an explicit `Eq(key, Null)` only via `Value::is_null` checks.
            return self.is_null() && other.is_null();
        }
        self.compare(other) == Some(Ordering::Equal)
    }

    /// Elements, when this value is an array. Scalars yield themselves, so
    /// `Contains` works uniformly.
    pub fn elements(&self) -> Vec<Value> {
        match self {
            Value::StringArray(a) => a.iter().cloned().map(Value::String).collect(),
            Value::UintArray(a) => a.iter().copied().map(Value::Uint).collect(),
            Value::IntArray(a) => a.iter().copied().map(Value::Int).collect(),
            other => vec![other.clone()],
        }
    }

    /// The value as a string, for `Glob` and `Regex` which are text operators.
    pub fn as_text(&self) -> Option<String> {
        Some(match self {
            Value::String(s) => s.clone(),
            Value::Uuid(u) => u.to_string(),
            Value::Datetime(n) => format_datetime(*n),
            _ => return None,
        })
    }

    // ------------------------------------------------------------ binary codec

    pub fn encode(&self, b: &mut BytesMut) {
        match self {
            Value::Null => b.put_u8(T_NULL),
            Value::Bool(v) => {
                b.put_u8(T_BOOL);
                b.put_u8(*v as u8);
            }
            Value::Uint(v) => {
                b.put_u8(T_UINT);
                b.put_u64_le(*v);
            }
            Value::Int(v) => {
                b.put_u8(T_INT);
                b.put_i64_le(*v);
            }
            Value::F64(v) => {
                b.put_u8(T_F64);
                b.put_f64_le(*v);
            }
            Value::String(s) => {
                b.put_u8(T_STRING);
                put_str(b, s);
            }
            Value::Datetime(v) => {
                b.put_u8(T_DATETIME);
                b.put_i64_le(*v);
            }
            Value::Uuid(u) => {
                b.put_u8(T_UUID);
                b.put_slice(u.as_bytes());
            }
            Value::StringArray(a) => {
                b.put_u8(T_STRING_ARRAY);
                b.put_u32_le(a.len() as u32);
                for s in a {
                    put_str(b, s);
                }
            }
            Value::UintArray(a) => {
                b.put_u8(T_UINT_ARRAY);
                b.put_u32_le(a.len() as u32);
                for v in a {
                    b.put_u64_le(*v);
                }
            }
            Value::IntArray(a) => {
                b.put_u8(T_INT_ARRAY);
                b.put_u32_le(a.len() as u32);
                for v in a {
                    b.put_i64_le(*v);
                }
            }
            Value::Sparse(m) => {
                b.put_u8(T_SPARSE);
                b.put_u32_le(m.len() as u32);
                for (dim, weight) in m {
                    put_str(b, dim);
                    b.put_f32_le(*weight);
                }
            }
        }
    }

    pub fn decode(buf: &[u8], pos: &mut usize) -> Result<Value> {
        let tag = *take(buf, pos, 1)?.first().unwrap();
        Ok(match tag {
            T_NULL => Value::Null,
            T_BOOL => Value::Bool(take(buf, pos, 1)?[0] != 0),
            T_UINT => Value::Uint(u64::from_le_bytes(take(buf, pos, 8)?.try_into().unwrap())),
            T_INT => Value::Int(i64::from_le_bytes(take(buf, pos, 8)?.try_into().unwrap())),
            T_F64 => Value::F64(f64::from_le_bytes(take(buf, pos, 8)?.try_into().unwrap())),
            T_STRING => Value::String(get_str(buf, pos)?),
            T_DATETIME => {
                Value::Datetime(i64::from_le_bytes(take(buf, pos, 8)?.try_into().unwrap()))
            }
            T_UUID => {
                let bytes: [u8; 16] = take(buf, pos, 16)?.try_into().unwrap();
                Value::Uuid(uuid::Uuid::from_bytes(bytes))
            }
            T_STRING_ARRAY => {
                let n = u32::from_le_bytes(take(buf, pos, 4)?.try_into().unwrap()) as usize;
                let mut out = Vec::with_capacity(n.min(1024));
                for _ in 0..n {
                    out.push(get_str(buf, pos)?);
                }
                Value::StringArray(out)
            }
            T_UINT_ARRAY => {
                let n = u32::from_le_bytes(take(buf, pos, 4)?.try_into().unwrap()) as usize;
                let mut out = Vec::with_capacity(n.min(1024));
                for _ in 0..n {
                    out.push(u64::from_le_bytes(take(buf, pos, 8)?.try_into().unwrap()));
                }
                Value::UintArray(out)
            }
            T_INT_ARRAY => {
                let n = u32::from_le_bytes(take(buf, pos, 4)?.try_into().unwrap()) as usize;
                let mut out = Vec::with_capacity(n.min(1024));
                for _ in 0..n {
                    out.push(i64::from_le_bytes(take(buf, pos, 8)?.try_into().unwrap()));
                }
                Value::IntArray(out)
            }
            T_SPARSE => {
                let n = u32::from_le_bytes(take(buf, pos, 4)?.try_into().unwrap()) as usize;
                let mut out = BTreeMap::new();
                for _ in 0..n {
                    let dim = get_str(buf, pos)?;
                    let weight = f32::from_le_bytes(take(buf, pos, 4)?.try_into().unwrap());
                    out.insert(dim, weight);
                }
                Value::Sparse(out)
            }
            other => bail!("unknown value tag {other} at offset {}", *pos - 1),
        })
    }
}

fn put_str(b: &mut BytesMut, s: &str) {
    b.put_u32_le(s.len() as u32);
    b.put_slice(s.as_bytes());
}

fn get_str(buf: &[u8], pos: &mut usize) -> Result<String> {
    let n = u32::from_le_bytes(take(buf, pos, 4)?.try_into().unwrap()) as usize;
    Ok(String::from_utf8(take(buf, pos, n)?.to_vec())?)
}

fn take<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8]> {
    let Some(s) = buf.get(*pos..*pos + n) else {
        bail!("truncated value: want {n} bytes at {}, have {}", pos, buf.len());
    };
    *pos += n;
    Ok(s)
}

fn as_f64(v: &Value) -> Option<f64> {
    Some(match v {
        Value::Uint(u) => *u as f64,
        Value::Int(i) => *i as f64,
        Value::F64(f) => *f,
        Value::Datetime(n) => *n as f64,
        _ => return None,
    })
}

fn as_i128(v: &Value) -> Option<i128> {
    Some(match v {
        Value::Uint(u) => *u as i128,
        Value::Int(i) => *i as i128,
        Value::Datetime(n) => *n as i128,
        _ => return None,
    })
}

/// ISO 8601 / RFC 3339 to nanoseconds since the epoch.
pub fn parse_datetime(s: &str) -> Result<i64> {
    use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return dt
            .timestamp_nanos_opt()
            .ok_or_else(|| anyhow::anyhow!("timestamp {s} is out of nanosecond range"));
    }
    // Accept the two forms turbopuffer's own examples use that are not strict
    // RFC 3339: a naive datetime, and a bare date.
    if let Ok(naive) = s.parse::<NaiveDateTime>() {
        return naive
            .and_utc()
            .timestamp_nanos_opt()
            .ok_or_else(|| anyhow::anyhow!("timestamp {s} is out of nanosecond range"));
    }
    if let Ok(date) = s.parse::<NaiveDate>() {
        let dt: DateTime<Utc> = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        return dt
            .timestamp_nanos_opt()
            .ok_or_else(|| anyhow::anyhow!("date {s} is out of nanosecond range"));
    }
    bail!("cannot parse {s:?} as a datetime")
}

pub fn format_datetime(nanos: i64) -> String {
    chrono::DateTime::from_timestamp_nanos(nanos)
        .format("%Y-%m-%dT%H:%M:%S%.9fZ")
        .to_string()
}

// ---------------------------------------------------------------- JSON

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Value::Null => s.serialize_none(),
            Value::Bool(v) => s.serialize_bool(*v),
            Value::Uint(v) => s.serialize_u64(*v),
            Value::Int(v) => s.serialize_i64(*v),
            Value::F64(v) => s.serialize_f64(*v),
            Value::String(v) => s.serialize_str(v),
            // Both leave as strings, which is how they arrived and how
            // turbopuffer returns them.
            Value::Datetime(n) => s.serialize_str(&format_datetime(*n)),
            Value::Uuid(u) => s.serialize_str(&u.to_string()),
            Value::StringArray(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for v in a {
                    seq.serialize_element(v)?;
                }
                seq.end()
            }
            Value::UintArray(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for v in a {
                    seq.serialize_element(v)?;
                }
                seq.end()
            }
            Value::IntArray(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for v in a {
                    seq.serialize_element(v)?;
                }
                seq.end()
            }
            Value::Sparse(m) => {
                use serde::ser::SerializeMap;
                let mut map = s.serialize_map(Some(m.len()))?;
                for (dim, weight) in m {
                    map.serialize_entry(dim, weight)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Value, D::Error> {
        let raw = serde_json::Value::deserialize(d)?;
        from_json(raw).map_err(de::Error::custom)
    }
}

/// Infer a value from JSON.
///
/// `uuid` and `datetime` are NOT inferred: a string that happens to parse as a
/// UUID is still a string until a schema says otherwise. Guessing here would make
/// a document's type depend on its content, so two documents with the same
/// attribute could disagree about its type.
pub fn from_json(raw: serde_json::Value) -> Result<Value> {
    use serde_json::Value as J;
    Ok(match raw {
        J::Null => Value::Null,
        J::Bool(b) => Value::Bool(b),
        J::Number(n) => {
            if let Some(u) = n.as_u64() {
                Value::Uint(u)
            } else if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::F64(f)
            } else {
                bail!("number {n} is not representable")
            }
        }
        J::String(s) => Value::String(s),
        J::Array(items) => {
            if items.iter().all(|v| v.is_u64()) {
                Value::UintArray(items.iter().map(|v| v.as_u64().unwrap()).collect())
            } else if items.iter().all(|v| v.is_i64()) {
                Value::IntArray(items.iter().map(|v| v.as_i64().unwrap()).collect())
            } else if items.iter().all(|v| v.is_string()) {
                Value::StringArray(
                    items.iter().map(|v| v.as_str().unwrap().to_string()).collect(),
                )
            } else if items.is_empty() {
                Value::StringArray(vec![])
            } else {
                bail!("arrays must be all strings or all unsigned integers")
            }
        }
        // The one object-shaped type is a sparse vector, so an object is read as
        // one. Anything non-numeric inside is a nested object, which is not a
        // supported attribute value.
        J::Object(fields) => {
            let mut out = BTreeMap::new();
            for (dim, weight) in fields {
                let Some(w) = weight.as_f64() else {
                    bail!(
                        "attribute values may be a sparse vector of numbers, but {dim:?} holds \
                         {weight}"
                    );
                };
                out.insert(dim, w as f32);
            }
            Value::Sparse(out)
        }
    })
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.into())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}
impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::Uint(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::F64(v)
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: &Value) -> Value {
        let mut b = BytesMut::new();
        v.encode(&mut b);
        let bytes = b.freeze();
        let mut pos = 0;
        let out = Value::decode(&bytes, &mut pos).unwrap();
        assert_eq!(pos, bytes.len(), "decoder did not consume the whole value");
        out
    }

    fn sample() -> Vec<Value> {
        vec![
            Value::Null,
            Value::Bool(true),
            Value::Bool(false),
            Value::Uint(0),
            Value::Uint(u64::MAX),
            Value::Int(i64::MIN),
            Value::Int(-1),
            Value::F64(-1.5),
            Value::F64(f64::MAX),
            Value::String(String::new()),
            Value::String("héllo ✨".into()),
            Value::Datetime(0),
            Value::Datetime(i64::MIN),
            Value::Datetime(i64::MAX),
            Value::Uuid(uuid::Uuid::nil()),
            Value::Uuid(uuid::Uuid::from_u128(0xdead_beef)),
            Value::StringArray(vec![]),
            Value::StringArray(vec!["a".into(), "".into(), "ü".into()]),
            Value::UintArray(vec![]),
            Value::UintArray(vec![0, u64::MAX]),
            Value::IntArray(vec![]),
            Value::IntArray(vec![i64::MIN, 0, i64::MAX]),
            Value::Sparse(BTreeMap::new()),
            Value::Sparse(BTreeMap::from([
                ("dim0".to_string(), 0.5f32),
                ("dim1".to_string(), -1.25f32),
            ])),
        ]
    }

    #[test]
    fn binary_roundtrip_every_variant() {
        for v in sample() {
            assert_eq!(roundtrip(&v), v, "binary roundtrip changed {v:?}");
        }
    }

    #[test]
    fn truncated_bytes_error_not_panic() {
        for v in sample() {
            let mut b = BytesMut::new();
            v.encode(&mut b);
            let bytes = b.freeze();
            for cut in 0..bytes.len() {
                let mut pos = 0;
                // Must not panic; a short buffer is an error.
                let _ = Value::decode(&bytes[..cut], &mut pos);
            }
        }
    }

    #[test]
    fn unknown_tag_is_rejected() {
        let mut pos = 0;
        assert!(Value::decode(&[200u8], &mut pos).is_err());
    }

    #[test]
    fn json_infers_the_types_json_can_express() {
        let cases = vec![
            ("null", Value::Null),
            ("true", Value::Bool(true)),
            ("12", Value::Uint(12)),
            ("-5", Value::Int(-5)),
            ("1.5", Value::F64(1.5)),
            (r#""foo""#, Value::String("foo".into())),
            (r#"["a","b"]"#, Value::StringArray(vec!["a".into(), "b".into()])),
            ("[1,2]", Value::UintArray(vec![1, 2])),
            ("[-1,2]", Value::IntArray(vec![-1, 2])),
        ];
        for (json, expected) in cases {
            let got: Value = serde_json::from_str(json).unwrap();
            assert_eq!(got, expected, "parsing {json}");
            // And it must serialize back to the same JSON shape.
            let back = serde_json::to_string(&got).unwrap();
            let reparsed: Value = serde_json::from_str(&back).unwrap();
            assert_eq!(reparsed, expected, "JSON roundtrip changed {json}");
        }
    }

    #[test]
    fn a_uuid_shaped_string_stays_a_string_until_the_schema_says_otherwise() {
        let s = "550e8400-e29b-41d4-a716-446655440000";
        let v: Value = serde_json::from_str(&format!("\"{s}\"")).unwrap();
        assert_eq!(v, Value::String(s.into()), "inferred a uuid from content");
        // Only coercion promotes it, so an attribute's type never depends on the
        // particular string a document happens to carry.
        assert_eq!(
            v.coerce(Type::Uuid).unwrap(),
            Value::Uuid(uuid::Uuid::parse_str(s).unwrap())
        );
    }

    #[test]
    fn mixed_arrays_are_rejected() {
        assert!(serde_json::from_str::<Value>(r#"[1,"a"]"#).is_err());
        assert!(serde_json::from_str::<Value>(r#"[{"a":1}]"#).is_err());
    }

    #[test]
    fn objects_read_as_sparse_vectors() {
        let v: Value = serde_json::from_str(r#"{"dim0":0.5,"dim1":2}"#).unwrap();
        assert_eq!(
            v,
            Value::Sparse(BTreeMap::from([
                ("dim0".to_string(), 0.5f32),
                ("dim1".to_string(), 2.0f32),
            ]))
        );
        assert_eq!(v.type_of(), Some(Type::SparseF16));
        // Round-trips as an object.
        let back: Value = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
        assert_eq!(back, v);
        assert_eq!(serde_json::from_str::<Value>("{}").unwrap(), Value::Sparse(BTreeMap::new()));
        // A nested object is not a weight.
        assert!(serde_json::from_str::<Value>(r#"{"a":{"b":1}}"#).is_err());
        assert!(serde_json::from_str::<Value>(r#"{"a":"x"}"#).is_err());
    }

    #[test]
    fn sparse_vectors_have_no_order() {
        // Ranked, never compared. An ordering here would let a Gte filter on a
        // sparse attribute silently succeed.
        let a = Value::Sparse(BTreeMap::from([("d".to_string(), 1.0f32)]));
        let b = Value::Sparse(BTreeMap::from([("d".to_string(), 2.0f32)]));
        assert_eq!(a.compare(&b), None);
        assert!(!a.equals(&b));
        assert!(a.as_text().is_none());
    }

    #[test]
    fn coercion_widens_losslessly_and_refuses_the_rest() {
        assert_eq!(Value::Uint(5).coerce(Type::Int).unwrap(), Value::Int(5));
        assert_eq!(Value::Uint(5).coerce(Type::F64).unwrap(), Value::F64(5.0));
        assert_eq!(Value::Int(-5).coerce(Type::F64).unwrap(), Value::F64(-5.0));
        assert_eq!(Value::Int(5).coerce(Type::Uint).unwrap(), Value::Uint(5));
        assert_eq!(Value::Null.coerce(Type::Uuid).unwrap(), Value::Null);

        // Lossy or nonsensical conversions must fail loudly.
        assert!(Value::Int(-5).coerce(Type::Uint).is_err(), "negative became unsigned");
        assert!(Value::Uint(u64::MAX).coerce(Type::Int).is_err(), "overflowed into i64");
        assert!(Value::String("nope".into()).coerce(Type::Uuid).is_err());
        assert!(Value::String("nope".into()).coerce(Type::Datetime).is_err());
        assert!(Value::Bool(true).coerce(Type::Uint).is_err());
    }

    #[test]
    fn datetime_parses_the_forms_clients_actually_send() {
        // RFC 3339 with zone, the documented form.
        let z = parse_datetime("2024-03-01T00:00:00Z").unwrap();
        assert_eq!(format_datetime(z), "2024-03-01T00:00:00.000000000Z");
        // Offsets normalise to UTC.
        assert_eq!(parse_datetime("2024-03-01T07:00:00+07:00").unwrap(), z);
        // Naive datetime and bare date, both of which appear in their examples.
        assert_eq!(parse_datetime("2024-03-01T00:00:00").unwrap(), z);
        assert_eq!(parse_datetime("2024-03-01").unwrap(), z);
        // Fractional seconds survive.
        assert_eq!(parse_datetime("1970-01-01T00:00:00.000000123Z").unwrap(), 123);
        assert!(parse_datetime("last tuesday").is_err());
    }

    #[test]
    fn comparison_orders_within_a_type() {
        assert_eq!(Value::Uint(1).compare(&Value::Uint(2)), Some(Ordering::Less));
        assert_eq!(
            Value::String("a".into()).compare(&Value::String("b".into())),
            Some(Ordering::Less)
        );
        assert_eq!(Value::Bool(false).compare(&Value::Bool(true)), Some(Ordering::Less));
        assert_eq!(Value::Datetime(1).compare(&Value::Datetime(2)), Some(Ordering::Less));
    }

    #[test]
    fn numeric_comparison_crosses_widths_without_losing_precision() {
        // The classic f64 trap: these two differ by 1 but collapse when cast.
        let big = Value::Uint(u64::MAX);
        let smaller = Value::Uint(u64::MAX - 1);
        assert_eq!(big.compare(&smaller), Some(Ordering::Greater));

        assert_eq!(Value::Int(-1).compare(&Value::Uint(0)), Some(Ordering::Less));
        assert_eq!(Value::Uint(3).compare(&Value::Int(3)), Some(Ordering::Equal));
        assert_eq!(Value::Uint(3).compare(&Value::F64(3.5)), Some(Ordering::Less));
        assert_eq!(Value::F64(2.0).compare(&Value::Int(3)), Some(Ordering::Less));
    }

    #[test]
    fn incomparable_types_return_none_not_equal() {
        // This is the load-bearing case: if a mismatched comparison returned
        // Equal or Less, a Gte filter would silently admit every document.
        assert_eq!(Value::String("1".into()).compare(&Value::Uint(1)), None);
        assert_eq!(Value::Bool(true).compare(&Value::Uint(1)), None);
        assert_eq!(
            Value::Uuid(uuid::Uuid::nil()).compare(&Value::String("x".into())),
            None
        );
        assert!(!Value::String("1".into()).equals(&Value::Uint(1)));
    }

    #[test]
    fn null_sorts_low_but_equals_nothing_real() {
        assert_eq!(Value::Null.compare(&Value::Uint(0)), Some(Ordering::Less));
        assert_eq!(Value::Uint(0).compare(&Value::Null), Some(Ordering::Greater));
        assert_eq!(Value::Null.compare(&Value::Null), Some(Ordering::Equal));
        assert!(Value::Null.equals(&Value::Null));
        assert!(!Value::Null.equals(&Value::Uint(0)), "null matched a real value");
    }

    #[test]
    fn elements_flattens_arrays_and_passes_scalars_through() {
        assert_eq!(
            Value::StringArray(vec!["a".into(), "b".into()]).elements(),
            vec![Value::String("a".into()), Value::String("b".into())]
        );
        assert_eq!(Value::UintArray(vec![1, 2]).elements(), vec![Value::Uint(1), Value::Uint(2)]);
        assert_eq!(Value::Uint(7).elements(), vec![Value::Uint(7)]);
        assert_eq!(Value::StringArray(vec![]).elements(), vec![]);
    }

    #[test]
    fn as_text_covers_the_string_like_types_only() {
        assert_eq!(Value::String("x".into()).as_text().as_deref(), Some("x"));
        assert!(Value::Uuid(uuid::Uuid::nil()).as_text().is_some());
        assert!(Value::Datetime(0).as_text().is_some());
        assert!(Value::Uint(1).as_text().is_none(), "a number is not text");
        assert!(Value::Null.as_text().is_none());
    }

    #[test]
    fn type_names_match_the_documented_schema_spelling() {
        assert_eq!(Type::StringArray.name(), "[]string");
        assert_eq!(Type::F64.name(), "float");
        assert_eq!(serde_json::to_string(&Type::F64).unwrap(), r#""float""#);
        assert_eq!(serde_json::to_string(&Type::IntArray).unwrap(), r#""[]int""#);
        assert_eq!(serde_json::to_string(&Type::SparseF16).unwrap(), r#""{}f16""#);
        assert_eq!(Type::UintArray.name(), "[]uint");
        assert_eq!(serde_json::to_string(&Type::StringArray).unwrap(), r#""[]string""#);
        assert_eq!(serde_json::to_string(&Type::Datetime).unwrap(), r#""datetime""#);
        assert_eq!(Value::Null.type_of(), None, "null must not claim a type");
    }
}
