//! turbopuffer-compatible `/v2` surface.
//!
//! A translation layer, deliberately thin: it converts wire shapes and delegates
//! to the same engine the native API uses. Nothing here should contain search
//! logic — if it starts to, the two surfaces have diverged and one of them is
//! lying about what the engine does.
//!
//! The differences are not cosmetic. Four of them will silently corrupt results
//! if translated carelessly:
//!
//!   1. `$dist` is a DISTANCE (lower is better); the native `score` is a
//!      SIMILARITY (higher is better). The conversion inverts ordering.
//!   2. Attributes are flattened into the row next to `id` and `$dist`, rather
//!      than nested under `attrs`.
//!   3. `filters` is plural, and `top_k` is an alias for `limit.total`.
//!   4. `consistency` is an object, `{"level": "eventual"}`, not a string.

use crate::doc::{Attrs, DistanceMetric, Doc, Filter, Id, Include};
use crate::store::Manifest;
use crate::value::Type;
use crate::fts::{FtsConfig, FtsSchema};
use crate::wire::{Columns, Consistency, FilterPatch, QueryRequest, WriteRequest};
use anyhow::{Result, bail};
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------- write

/// A row in turbopuffer's row format: `id` and `vector` are named fields, every
/// other key is an attribute at the same level.
#[derive(Debug, Clone, Deserialize)]
pub struct Row {
    pub id: Id,
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
    #[serde(flatten)]
    pub attrs: Attrs,
}

/// One attribute's declared schema. `type` is the part that changes behaviour
/// here; the indexing flags are accepted and stored so a client's schema round
/// trips, but only `type` is acted on until the attribute index lands.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SchemaAttr {
    #[serde(rename = "type")]
    pub ty: Type,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filterable: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glob: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_text_search: Option<FullText>,
}

/// `full_text_search` is either a flag or a tokenizer configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum FullText {
    Enabled(bool),
    Config(Box<FtsConfig>),
}

impl FullText {
    fn config(&self) -> Option<FtsConfig> {
        match self {
            FullText::Enabled(true) => Some(FtsConfig::default()),
            FullText::Enabled(false) => None,
            FullText::Config(c) => Some((**c).clone()),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct V2Write {
    #[serde(default)]
    pub upsert_rows: Vec<Row>,
    #[serde(default)]
    pub upsert_columns: Option<Columns>,
    #[serde(default)]
    pub patch_rows: Vec<Row>,
    #[serde(default)]
    pub patch_columns: Option<Columns>,
    #[serde(default)]
    pub deletes: Vec<Id>,
    #[serde(default)]
    pub delete_by_filter: Option<Filter>,
    #[serde(default)]
    pub patch_by_filter: Option<FilterPatch>,
    #[serde(default)]
    pub distance_metric: Option<DistanceMetric>,
    #[serde(default)]
    pub schema: Option<BTreeMap<String, SchemaAttr>>,
}

impl V2Write {
    pub fn is_empty(&self) -> bool {
        self.upsert_rows.is_empty()
            && self.patch_rows.is_empty()
            && self.deletes.is_empty()
            && self.upsert_columns.is_none()
            && self.patch_columns.is_none()
            && self.delete_by_filter.is_none()
            && self.patch_by_filter.is_none()
    }

    /// Declared attribute types, for the engine's schema.
    pub fn declared_types(&self) -> BTreeMap<String, Type> {
        self.schema
            .as_ref()
            .map(|s| s.iter().map(|(k, v)| (k.clone(), v.ty)).collect())
            .unwrap_or_default()
    }

    /// Attributes the client enabled for full-text search, with their tokenizer.
    pub fn declared_fts(&self) -> FtsSchema {
        self.schema
            .as_ref()
            .map(|s| {
                s.iter()
                    .filter_map(|(k, v)| {
                        v.full_text_search.as_ref()?.config().map(|c| (k.clone(), c))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Translate into the native write request the engine already understands.
    pub fn into_native(self) -> Result<WriteRequest> {
        let mut upsert = Vec::with_capacity(self.upsert_rows.len());
        for row in self.upsert_rows {
            let Some(vector) = row.vector else {
                bail!("upsert_rows: document {} has no vector", row.id);
            };
            upsert.push(Doc { id: row.id, vector, attrs: row.attrs });
        }

        Ok(WriteRequest {
            upsert,
            patch: self
                .patch_rows
                .into_iter()
                .map(|r| crate::wire::PatchRow { id: r.id, attrs: r.attrs })
                .collect(),
            delete: self.deletes,
            upsert_columns: self.upsert_columns,
            patch_columns: self.patch_columns,
            delete_by_filter: self.delete_by_filter,
            patch_by_filter: self.patch_by_filter,
            distance_metric: self.distance_metric,
        })
    }
}

/// Counts, with each field present only when the corresponding operation was
/// used — matching turbopuffer, where a caller distinguishes "zero rows patched"
/// from "no patch was requested".
#[derive(Debug, Clone, Default, Serialize)]
pub struct V2WriteResponse {
    pub rows_affected: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows_upserted: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows_patched: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows_deleted: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows_remaining: Option<bool>,
}

// ---------------------------------------------------------------- query

/// How a query ranks. `["vector", "ANN", [0.1, 0.2]]` and friends.
#[derive(Debug, Clone, PartialEq)]
pub enum RankBy {
    /// Approximate nearest neighbour over a vector attribute.
    Ann { attribute: String, vector: Vec<f32> },
    /// Exact nearest neighbour: the same ranking, without the index.
    Knn { attribute: String, vector: Vec<f32> },
    /// Rank by an attribute value.
    Order { attribute: String, descending: bool },
    /// BM25 full-text ranking over a text attribute.
    Bm25 { attribute: String, query: String },
    /// Recognised but not implemented yet, so it can be reported as such rather
    /// than mis-parsed as something else.
    Unsupported(String),
}

impl<'de> Deserialize<'de> for RankBy {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<RankBy, D::Error> {
        let raw = serde_json::Value::deserialize(d)?;
        parse_rank_by(&raw).map_err(de::Error::custom)
    }
}

fn parse_rank_by(raw: &serde_json::Value) -> Result<RankBy> {
    let Some(arr) = raw.as_array() else {
        bail!("rank_by must be an array, got {raw}");
    };
    // Ordering by attribute is the one two-element form: ["created_at", "desc"].
    if arr.len() == 2 {
        let attribute = arr[0]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("rank_by attribute must be a string"))?
            .to_string();
        return match arr[1].as_str() {
            Some("asc") => Ok(RankBy::Order { attribute, descending: false }),
            Some("desc") => Ok(RankBy::Order { attribute, descending: true }),
            // A ranking function that takes an argument is missing it, which is a
            // shape error. Reporting "not implemented" for a function that IS
            // implemented would send the caller looking in the wrong place.
            Some(f @ ("ANN" | "kNN" | "BM25" | "SparseKNN")) => {
                bail!("rank_by {f} requires an argument: [attribute, \"{f}\", …]")
            }
            Some(other) => Ok(RankBy::Unsupported(other.to_string())),
            None => bail!("rank_by direction must be \"asc\" or \"desc\""),
        };
    }
    if arr.len() != 3 {
        bail!("rank_by must be [attribute, function, argument], got {} elements", arr.len());
    }
    let attribute = arr[0]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("rank_by attribute must be a string"))?
        .to_string();
    let func = arr[1]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("rank_by function must be a string"))?;

    match func {
        "BM25" => {
            let query = arr[2]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("rank_by BM25 expects a query string"))?
                .to_string();
            Ok(RankBy::Bm25 { attribute, query })
        }
        "ANN" | "kNN" => {
            let vector: Vec<f32> = serde_json::from_value(arr[2].clone()).map_err(|_| {
                // Embed takes text instead of a vector; say so precisely rather
                // than reporting a generic parse failure.
                if arr[2].get(0).and_then(|v| v.as_str()) == Some("Embed") {
                    anyhow::anyhow!("native embedding (Embed) is not implemented")
                } else {
                    anyhow::anyhow!("rank_by {func} expects an array of numbers")
                }
            })?;
            Ok(if func == "ANN" {
                RankBy::Ann { attribute, vector }
            } else {
                RankBy::Knn { attribute, vector }
            })
        }
        other => Ok(RankBy::Unsupported(other.to_string())),
    }
}

/// `limit` is an object; `top_k` is an alias for `limit.total`.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Limit {
    pub total: usize,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct V2Consistency {
    pub level: ConsistencyLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConsistencyLevel {
    Strong,
    Eventual,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct V2Query {
    #[serde(default)]
    pub rank_by: Option<RankBy>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub limit: Option<Limit>,
    #[serde(default)]
    pub filters: Option<Filter>,
    #[serde(default)]
    pub include_attributes: Include,
    #[serde(default)]
    pub consistency: Option<V2Consistency>,
    /// Recognised so an unsupported value is reported rather than ignored.
    #[serde(default)]
    pub vector_encoding: Option<String>,
    /// Multi-query. Recognised and reported as unimplemented.
    #[serde(default)]
    pub queries: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub aggregate_by: Option<serde_json::Value>,
}

/// Default eventual-consistency staleness window, matching turbopuffer's
/// documented "up to about one hour".
const EVENTUAL_MAX_AGE_MS: u64 = 3_600_000;

impl V2Query {
    /// Translate into the native request. Returns the request and whether the
    /// index should be bypassed (`kNN` means exact).
    pub fn into_native(self, nprobe: usize) -> Result<(QueryRequest, bool)> {
        if self.queries.is_some() {
            bail!("multi-queries are not implemented");
        }
        if self.aggregate_by.is_some() {
            bail!("aggregations are not implemented");
        }
        if let Some(enc) = &self.vector_encoding
            && enc != "float"
        {
            bail!("vector_encoding {enc:?} is not implemented; only \"float\" is supported");
        }

        let Some(rank_by) = self.rank_by else {
            bail!("rank_by is required");
        };
        let (vector, exact, order_by, text) = match rank_by {
            RankBy::Ann { vector, .. } => (vector, false, None, None),
            RankBy::Knn { vector, .. } => (vector, true, None, None),
            RankBy::Order { attribute, descending } => {
                (vec![], false, Some(crate::doc::OrderBy { attribute, descending }), None)
            }
            RankBy::Bm25 { attribute, query } => (
                vec![],
                false,
                None,
                Some(crate::wire::TextSearch { attribute, query }),
            ),
            RankBy::Unsupported(f) => bail!("rank_by function {f:?} is not implemented"),
        };

        // top_k is documented as an alias for limit.total; if both arrive, they
        // must not silently disagree.
        let top_k = match (self.top_k, self.limit) {
            (Some(a), Some(l)) if a != l.total => {
                bail!("top_k ({a}) and limit.total ({}) disagree", l.total)
            }
            (Some(a), _) => a,
            (None, Some(l)) => l.total,
            (None, None) => 10,
        };
        if top_k == 0 || top_k > 10_000 {
            bail!("top_k must be between 1 and 10,000");
        }

        let consistency = match self.consistency.map(|c| c.level) {
            None | Some(ConsistencyLevel::Strong) => Consistency::Strong,
            Some(ConsistencyLevel::Eventual) => {
                Consistency::Eventual { max_age_ms: EVENTUAL_MAX_AGE_MS }
            }
        };

        Ok((
            QueryRequest {
                vector,
                top_k,
                filter: self.filters,
                nprobe,
                consistency,
                include_attributes: self.include_attributes,
                order_by,
                text,
            },
            exact,
        ))
    }
}

/// One result row: `id`, `$dist`, and attributes at the same level.
#[derive(Debug, Clone, Serialize)]
pub struct V2Row {
    pub id: Id,
    /// Absent when ordering by an attribute: there is no distance to report, and
    /// emitting a zero would read as a perfect match.
    #[serde(rename = "$dist", skip_serializing_if = "Option::is_none")]
    pub dist: Option<f32>,
    #[serde(flatten)]
    pub attrs: Attrs,
}

#[derive(Debug, Clone, Serialize)]
pub struct V2QueryResponse {
    pub rows: Vec<V2Row>,
}

/// Convert native hits into compatibility rows.
///
/// The sign flip lives here and nowhere else. `score` is a similarity and `$dist`
/// is a distance, so the conversion must invert the ordering — a client that
/// received similarities under the name `$dist` would rank every result exactly
/// backwards and see no error at all.
pub fn to_v2_rows(hits: Vec<crate::doc::Hit>, metric: DistanceMetric) -> Vec<V2Row> {
    hits.into_iter()
        .map(|h| V2Row {
            id: h.id,
            dist: Some(match metric {
                // score is cosine similarity in [-1, 1]; the distance is 1 - it.
                DistanceMetric::CosineDistance => 1.0 - h.score,
                // score is the negated squared distance.
                DistanceMetric::EuclideanSquared => -h.score,
                // score is the dot product; larger is nearer, so negate.
                DistanceMetric::DotProduct => -h.score,
            }),
            attrs: h.attrs,
        })
        .collect()
}

/// Rows for a BM25 query.
///
/// `$dist` carries the relevance score unchanged, because for BM25 HIGHER is
/// better — the opposite of a vector distance. Running these through the vector
/// conversion would invert the ranking, which is the same trap as the score/dist
/// flip, in the other direction.
pub fn to_v2_score_rows(hits: Vec<crate::doc::Hit>) -> Vec<V2Row> {
    hits.into_iter().map(|h| V2Row { id: h.id, dist: Some(h.score), attrs: h.attrs }).collect()
}

/// Rows for an attribute-ordered query, which carries no distance.
pub fn to_v2_ordered_rows(hits: Vec<crate::doc::Hit>) -> Vec<V2Row> {
    hits.into_iter().map(|h| V2Row { id: h.id, dist: None, attrs: h.attrs }).collect()
}

// ---------------------------------------------------------------- metadata

#[derive(Debug, Clone, Serialize)]
pub struct V2IndexInfo {
    pub unindexed_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct V2Metadata {
    pub id: String,
    pub approx_row_count: usize,
    pub approx_logical_bytes: u64,
    pub schema: BTreeMap<String, SchemaAttr>,
    pub index: V2IndexInfo,
    pub distance_metric: DistanceMetric,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_write_at: Option<String>,
    pub encryption: serde_json::Value,
}

pub fn to_v2_metadata(name: &str, m: &Manifest) -> V2Metadata {
    let iso = |n: Option<i64>| n.map(crate::value::format_datetime);
    V2Metadata {
        id: name.to_string(),
        // Approximate on purpose: the indexed count is exact but the unindexed
        // tail may contain updates and tombstones for documents already counted.
        approx_row_count: m.index.as_ref().map_or(0, |i| i.docs) + m.unindexed_records(),
        approx_logical_bytes: m.total_bytes(),
        schema: m
            .schema
            .attributes
            .iter()
            .map(|(k, ty)| {
                (
                    k.clone(),
                    SchemaAttr {
                        ty: *ty,
                        filterable: Some(true),
                        glob: None,
                        full_text_search: m
                            .schema
                            .fts
                            .get(k)
                            .map(|c| FullText::Config(Box::new(c.clone()))),
                    },
                )
            })
            .collect(),
        index: V2IndexInfo { unindexed_bytes: m.unindexed_bytes() },
        distance_metric: m.schema.distance_metric,
        created_at: iso(m.created_at),
        updated_at: iso(m.updated_at),
        last_write_at: iso(m.last_write_at),
        encryption: serde_json::json!({ "mode": "default" }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Hit, Record};
    use crate::value::Value;

    #[test]
    fn rank_by_parses_ann_and_knn() {
        let ann: RankBy = serde_json::from_str(r#"["vector","ANN",[0.1,0.2]]"#).unwrap();
        assert_eq!(ann, RankBy::Ann { attribute: "vector".into(), vector: vec![0.1, 0.2] });
        let knn: RankBy = serde_json::from_str(r#"["vector","kNN",[0.1,0.2]]"#).unwrap();
        assert_eq!(knn, RankBy::Knn { attribute: "vector".into(), vector: vec![0.1, 0.2] });
    }

    #[test]
    fn rank_by_parses_bm25() {
        let got: RankBy = serde_json::from_str(r#"["content","BM25","quick fox"]"#).unwrap();
        assert_eq!(
            got,
            RankBy::Bm25 { attribute: "content".into(), query: "quick fox".into() }
        );
        let (req, exact) = V2Query { rank_by: Some(got), ..Default::default() }
            .into_native(8)
            .unwrap();
        assert!(!exact);
        assert!(req.vector.is_empty(), "a text query needs no vector");
        let t = req.text.unwrap();
        assert_eq!(t.attribute, "content");
        assert_eq!(t.query, "quick fox");

        // A non-string argument is a shape error, not a mis-parse.
        assert!(serde_json::from_str::<RankBy>(r#"["content","BM25",[1.0]]"#).is_err());
    }

    #[test]
    fn rank_by_parses_attribute_ordering() {
        let asc: RankBy = serde_json::from_str(r#"["created_at","asc"]"#).unwrap();
        assert_eq!(asc, RankBy::Order { attribute: "created_at".into(), descending: false });
        let desc: RankBy = serde_json::from_str(r#"["created_at","desc"]"#).unwrap();
        assert_eq!(desc, RankBy::Order { attribute: "created_at".into(), descending: true });

        // An ordered query needs no vector, and reports no distance.
        let (req, exact) = serde_json::from_str::<V2Query>(r#"{"rank_by":["n","desc"]}"#)
            .unwrap()
            .into_native(8)
            .unwrap();
        assert!(!exact);
        assert!(req.vector.is_empty());
        assert_eq!(req.order_by.unwrap().descending, true);

        let rows = to_v2_ordered_rows(vec![Hit::new(1u64, 0.0)]);
        let json = serde_json::to_value(V2QueryResponse { rows }).unwrap();
        assert!(
            json["rows"][0].get("$dist").is_none(),
            "an ordered row reported a distance; zero would read as a perfect match"
        );
    }

    #[test]
    fn unimplemented_rank_functions_are_named_not_swallowed() {
        for f in ["SparseKNN"] {
            let got: RankBy =
                serde_json::from_str(&format!(r#"["a","{f}",[1.0]]"#)).unwrap();
            assert_eq!(got, RankBy::Unsupported(f.into()));
            let err = V2Query { rank_by: Some(got), ..Default::default() }
                .into_native(8)
                .unwrap_err()
                .to_string();
            assert!(err.contains(f), "error did not name the function: {err}");
        }
    }

    #[test]
    fn embed_is_reported_precisely() {
        let err = serde_json::from_str::<RankBy>(r#"["content","ANN",["Embed","foxes"]]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Embed"), "unhelpful error: {err}");
    }

    #[test]
    fn rank_by_shape_errors() {
        for bad in [
            r#""vector""#,
            r#"[]"#,
            // A function that takes an argument, missing it: a shape error, not
            // an "unimplemented function" error.
            r#"["vector","ANN"]"#,
            r#"["text","BM25"]"#,
            r#"[1,"ANN",[1.0]]"#,
            r#"["n",5]"#,
        ] {
            assert!(serde_json::from_str::<RankBy>(bad).is_err(), "accepted {bad}");
        }
        // An unknown two-element direction is reported as unimplemented, not as
        // malformed: the shape is right, the function is not supported.
        let got: RankBy = serde_json::from_str(r#"["n","sideways"]"#).unwrap();
        assert_eq!(got, RankBy::Unsupported("sideways".into()));
    }

    #[test]
    fn top_k_and_limit_are_aliases_that_must_agree() {
        let q = |json: &str| serde_json::from_str::<V2Query>(json).unwrap().into_native(8);

        let (r, _) = q(r#"{"rank_by":["v","ANN",[1.0]],"top_k":5}"#).unwrap();
        assert_eq!(r.top_k, 5);
        let (r, _) = q(r#"{"rank_by":["v","ANN",[1.0]],"limit":{"total":7}}"#).unwrap();
        assert_eq!(r.top_k, 7);
        let (r, _) = q(r#"{"rank_by":["v","ANN",[1.0]],"top_k":3,"limit":{"total":3}}"#).unwrap();
        assert_eq!(r.top_k, 3);
        // Disagreeing is a client bug; guessing which one they meant is worse.
        assert!(q(r#"{"rank_by":["v","ANN",[1.0]],"top_k":3,"limit":{"total":9}}"#).is_err());
        // Default, and the documented ceiling.
        let (r, _) = q(r#"{"rank_by":["v","ANN",[1.0]]}"#).unwrap();
        assert_eq!(r.top_k, 10);
        assert!(q(r#"{"rank_by":["v","ANN",[1.0]],"top_k":0}"#).is_err());
        assert!(q(r#"{"rank_by":["v","ANN",[1.0]],"top_k":10001}"#).is_err());
    }

    #[test]
    fn consistency_is_an_object_with_a_level() {
        let q = |json: &str| serde_json::from_str::<V2Query>(json).unwrap().into_native(8).unwrap().0;
        assert_eq!(q(r#"{"rank_by":["v","ANN",[1.0]]}"#).consistency, Consistency::Strong);
        assert_eq!(
            q(r#"{"rank_by":["v","ANN",[1.0]],"consistency":{"level":"strong"}}"#).consistency,
            Consistency::Strong
        );
        assert_eq!(
            q(r#"{"rank_by":["v","ANN",[1.0]],"consistency":{"level":"eventual"}}"#).consistency,
            Consistency::Eventual { max_age_ms: EVENTUAL_MAX_AGE_MS }
        );
        // A bare string is their older/other spelling and must not silently
        // become strong.
        assert!(
            serde_json::from_str::<V2Query>(
                r#"{"rank_by":["v","ANN",[1.0]],"consistency":"eventual"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn knn_requests_bypass_the_index() {
        let (_, exact) =
            serde_json::from_str::<V2Query>(r#"{"rank_by":["v","kNN",[1.0]]}"#)
                .unwrap()
                .into_native(8)
                .unwrap();
        assert!(exact, "kNN must not be served from the approximate index");
        let (_, exact) =
            serde_json::from_str::<V2Query>(r#"{"rank_by":["v","ANN",[1.0]]}"#)
                .unwrap()
                .into_native(8)
                .unwrap();
        assert!(!exact);
    }

    #[test]
    fn unimplemented_query_features_are_refused_not_ignored() {
        for json in [
            r#"{"queries":[{"rank_by":["v","ANN",[1.0]]}]}"#,
            r#"{"rank_by":["v","ANN",[1.0]],"aggregate_by":{"n":["Count"]}}"#,
            r#"{"rank_by":["v","ANN",[1.0]],"vector_encoding":"base64"}"#,
            r#"{"top_k":5}"#,
        ] {
            let q: V2Query = serde_json::from_str(json).unwrap();
            assert!(q.into_native(8).is_err(), "silently accepted {json}");
        }
        // float encoding is the default and must be accepted.
        let q: V2Query =
            serde_json::from_str(r#"{"rank_by":["v","ANN",[1.0]],"vector_encoding":"float"}"#)
                .unwrap();
        assert!(q.into_native(8).is_ok());
    }

    /// The trap: `$dist` and `score` order oppositely. A conversion that only
    /// renamed the field would rank every result backwards, with no error.
    #[test]
    fn dist_inverts_the_ordering_of_score() {
        let hits = vec![Hit::new(1u64, 1.0), Hit::new(2u64, 0.5), Hit::new(3u64, -0.2)];
        for metric in [
            DistanceMetric::CosineDistance,
            DistanceMetric::EuclideanSquared,
            DistanceMetric::DotProduct,
        ] {
            let d: Vec<f32> =
                to_v2_rows(hits.clone(), metric).iter().map(|r| r.dist.unwrap()).collect();
            assert_eq!(d.len(), 3);
            // Best score must become the smallest distance.
            assert!(
                d[0] < d[1] && d[1] < d[2],
                "{metric:?} did not invert the ordering: {d:?}"
            );
        }
    }

    #[test]
    fn cosine_distance_matches_the_conventional_definition() {
        let dist = |score| {
            to_v2_rows(vec![Hit::new(1u64, score)], DistanceMetric::CosineDistance)[0]
                .dist
                .unwrap()
        };
        // Identical vectors: similarity 1, distance 0.
        assert!((dist(1.0) - 0.0).abs() < 1e-6, "got {}", dist(1.0));
        // Orthogonal: similarity 0, distance 1.
        assert!((dist(0.0) - 1.0).abs() < 1e-6);
        // Opposed: similarity -1, distance 2.
        assert!((dist(-1.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn rows_serialize_with_dist_and_flattened_attributes() {
        let mut attrs = Attrs::new();
        attrs.insert("title".into(), Value::from("puffer"));
        let rows = vec![V2Row { id: Id::Uint(8), dist: Some(1.7), attrs }];
        let json = serde_json::to_value(V2QueryResponse { rows }).unwrap();
        let row = &json["rows"][0];
        assert_eq!(row["id"], 8);
        // f32 widened to f64, so compare with tolerance rather than exactly:
        // `$dist` carries f32 precision, which is what the engine computes in.
        assert!((row["$dist"].as_f64().unwrap() - 1.7).abs() < 1e-6);
        // Flattened alongside id, not nested under "attrs".
        assert_eq!(row["title"], "puffer");
        assert!(row.get("attrs").is_none(), "attributes were nested instead of flattened");
    }

    #[test]
    fn write_rows_flatten_attributes() {
        let w: V2Write = serde_json::from_str(
            r#"{"upsert_rows":[{"id":1,"vector":[1.0,2.0],"name":"foo","n":3}]}"#,
        )
        .unwrap();
        assert_eq!(w.upsert_rows.len(), 1);
        let row = &w.upsert_rows[0];
        assert_eq!(row.id, Id::Uint(1));
        assert_eq!(row.vector.as_deref(), Some(&[1.0f32, 2.0][..]));
        assert_eq!(row.attrs["name"], Value::from("foo"));
        assert_eq!(row.attrs["n"], Value::Uint(3));

        let native = w.into_native().unwrap();
        assert_eq!(native.upsert.len(), 1);
        assert_eq!(native.upsert[0].attrs["name"], Value::from("foo"));
    }

    #[test]
    fn a_vectorless_upsert_row_is_refused() {
        let w: V2Write = serde_json::from_str(r#"{"upsert_rows":[{"id":1,"name":"x"}]}"#).unwrap();
        assert!(w.into_native().is_err());
    }

    #[test]
    fn schema_declaration_is_idempotent_but_refuses_a_type_change() {
        let mut schema = crate::doc::Schema::default();
        let declared = BTreeMap::from([("n".to_string(), Type::Uint)]);
        schema.declare(&declared).unwrap();
        assert_eq!(schema.attributes["n"], Type::Uint);

        // Clients may resend the schema on every write, so re-declaring the same
        // type must be a no-op.
        schema.declare(&declared).unwrap();
        let err = schema
            .declare(&BTreeMap::from([("n".to_string(), Type::String)]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("uint") && err.contains("string"), "unhelpful error: {err}");
    }

    #[test]
    fn a_declared_type_promotes_values_the_engine_would_have_inferred_loosely() {
        // Declaring datetime makes the engine coerce an incoming string, instead
        // of inferring `string` and locking that in forever.
        let mut schema = crate::doc::Schema::default();
        schema.declare(&BTreeMap::from([("when".to_string(), Type::Datetime)])).unwrap();
        let mut records =
            vec![Record::Upsert(Doc::new(1u64, vec![1.0]).with_attr("when", "2024-03-01"))];
        schema.absorb(&mut records).unwrap();
        match &records[0] {
            Record::Upsert(d) => assert!(matches!(d.attrs["when"], Value::Datetime(_))),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn metadata_maps_to_their_field_names() {
        let mut m = Manifest::default();
        m.schema.attributes.insert("n".into(), Type::Uint);
        m.schema.distance_metric = DistanceMetric::EuclideanSquared;
        m.created_at = Some(0);
        m.last_write_at = Some(1_709_251_200_000_000_000);
        let md = to_v2_metadata("docs", &m);
        let json = serde_json::to_value(&md).unwrap();

        assert_eq!(json["id"], "docs");
        assert_eq!(json["schema"]["n"]["type"], "uint");
        assert_eq!(json["distance_metric"], "euclidean_squared");
        assert_eq!(json["index"]["unindexed_bytes"], 0);
        assert_eq!(json["encryption"]["mode"], "default");
        assert!(json["created_at"].as_str().unwrap().starts_with("1970-01-01"));
        assert!(json["last_write_at"].as_str().unwrap().starts_with("2024-03-01"));
        // Never written: the field is absent rather than null.
        assert!(json.get("updated_at").is_none());
    }
}
