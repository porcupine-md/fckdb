# Usage

Every request here is runnable curl, and every response shown was captured from a
live server rather than written by hand.

```bash
export U=http://127.0.0.1:8080
export T='Authorization: Bearer secret-token'
export J='Content-Type: application/json'
```

---

## Starting a server

```bash
FCKDB_BUCKET=my-bucket \
FCKDB_ENDPOINT=https://<account>.r2.cloudflarestorage.com \
AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… \
FCKDB_TOKENS='secret-token:my-org' \
FCKDB_CACHE_PATH=/var/cache/fckdb.bin FCKDB_CACHE_BYTES=$((4*1024*1024*1024)) \
cargo run --release -- serve
```

With no `FCKDB_BUCKET` it runs entirely in memory — useful for trying things out,
and it needs no credentials.

| variable | meaning |
|---|---|
| `FCKDB_BUCKET` | bucket name. Empty or unset means in-memory |
| `FCKDB_ENDPOINT` | `https://{account}.r2.cloudflarestorage.com`, or a MinIO URL |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | credentials |
| `FCKDB_TOKENS` | `tok1:org1,tok2:org2`. Or `FCKDB_TOKEN=tok` for one tenant |
| `FCKDB_ADDR` | listen address, default `127.0.0.1:8080` |
| `FCKDB_CACHE_PATH` / `FCKDB_CACHE_BYTES` | NVMe read cache. Unset means every query pays cold latency |
| `FCKDB_EMBED_URL` / `_MODEL` / `_KEY` | OpenAI-compatible embeddings endpoint |
| `FCKDB_PRICE_*` | override the R2 list prices used by `/metrics` |

**Leaving both token variables unset disables authentication**, and the server
logs a warning saying so. Do not expose that port.

### Health

```bash
curl -s $U/healthz
# ok
```

---

## Two surfaces

| | path | shape |
|---|---|---|
| **`/v2`** | `POST /v2/namespaces/{ns}` | turbopuffer-compatible. Attributes flattened, `$dist`, `rank_by` |
| **`/v1`** | `POST /v1/namespaces/{ns}/write` | native. Attributes nested under `attrs`, `score`, plain `vector` |

Both drive the same engine. Use `/v2` if you have turbopuffer clients; the
examples below use it. Operational endpoints (`compact`, `gc`, `warm`, `branch`)
are `/v1` only — turbopuffer has no equivalent.

Namespaces are created by writing to them, are scoped to the token's
organisation, and must match `[A-Za-z0-9_-]{1,64}`.

---

## Writing

### Rows

```bash
curl -s -X POST $U/v2/namespaces/articles -H "$T" -H "$J" -d '{
  "distance_metric": "cosine_distance",
  "schema": {
    "published": { "type": "datetime" },
    "body":      { "type": "string", "full_text_search": true }
  },
  "upsert_rows": [
    { "id": 1, "vector": [1,0,0], "title": "Rust for beginners",
      "topic": "tech", "score": 90, "published": "2024-06-01T00:00:00Z",
      "tags": ["rust","programming"],
      "body": "a complete guide to the rust programming language" },
    { "id": 2, "vector": [0.9,0.1,0], "title": "Databases on object storage",
      "topic": "tech", "score": 85, "published": "2025-01-15T00:00:00Z",
      "tags": ["database","architecture"],
      "body": "notes on building a database that keeps everything in object storage" },
    { "id": 3, "vector": [0,1,0], "title": "Rendang recipe",
      "topic": "food", "score": 95, "published": "2023-03-20T00:00:00Z",
      "tags": ["cooking"],
      "body": "a spicy beef recipe from padang with all the spices" }
  ]
}'
```

```json
{ "rows_affected": 3, "rows_upserted": 3 }
```

Counters for operations you did not request are **absent**, not zero — so you can
tell "no rows deleted" from "no delete requested".

Ids may be integers, strings, or UUIDs. The first write fixes which.

### Columns

Same documents, transposed. Every column must be the same length.

```bash
curl -s -X POST $U/v2/namespaces/articles -H "$T" -H "$J" -d '{
  "upsert_columns": {
    "id":     [4, 5],
    "vector": [[0,0,1], [0.2,0,0.9]],
    "title":  ["Marathon training", "Workout tracker apps"],
    "topic":  ["sport", "sport"],
    "score":  [60, 78]
  }
}'
```

A `null` in a column means *this document has no value for it*, which is not the
same as a null value.

### Patch

Merges attributes, leaves the vector alone. A `null` removes an attribute.

```bash
curl -s -X POST $U/v2/namespaces/articles -H "$T" -H "$J" \
  -d '{ "patch_rows": [{ "id": 1, "score": 99, "draft": null }] }'
```

Vectors cannot be patched — upsert the whole document instead.

### Delete

```bash
curl -s -X POST $U/v2/namespaces/articles -H "$T" -H "$J" -d '{ "deletes": [5] }'

curl -s -X POST $U/v2/namespaces/articles -H "$T" -H "$J" \
  -d '{ "delete_by_filter": ["topic","Eq","sport"] }'
```

```json
{ "rows_affected": 1, "rows_deleted": 1, "rows_remaining": false }
```

`rows_remaining: true` means the request hit its cap (5M for `delete_by_filter`,
50k for `patch_by_filter`) and more documents still match. **Reissue the same
request until it is false.**

### Patch by filter

```bash
curl -s -X POST $U/v2/namespaces/articles -H "$T" -H "$J" -d '{
  "patch_by_filter": {
    "filters": ["topic","Eq","tech"],
    "patch": { "reviewed": true }
  }
}'
```

Within one request, operations apply in a fixed order: `delete_by_filter`, then
`patch_by_filter`, then explicit rows and columns.

---

## Schema

Types are **inferred** from the first write that carries each value, then
enforced. You only need to declare the two things JSON cannot express:

| type | declare it because |
|---|---|
| `datetime` | JSON has no date type; a string would compare lexicographically |
| `uuid` | a UUID-shaped string stays a string unless you say otherwise |

```json
"schema": {
  "published": { "type": "datetime" },
  "owner":     { "type": "uuid" },
  "body":      { "type": "string", "full_text_search": true }
}
```

Supported: `string`, `int`, `uint`, `float`, `bool`, `datetime`, `uuid`,
`[]string`, `[]int`, `[]uint`, `{}f16` (sparse vector).

Changing an attribute's type later is refused — every stored value was encoded
under the old one.

```json
{ "error": "document 5, attribute \"score\": cannot interpret string as uint" }
```

### Full-text options

```json
"body": { "type": "string", "full_text_search": {
  "tokenizer": {
    "language": "indonesian",
    "stemming": true,
    "remove_stopwords": true,
    "ascii_folding": false,
    "case_sensitive": false,
    "max_token_length": 39
  },
  "k1": 1.2, "b": 0.75
}}
```

Languages with a stemmer: english, french, german, spanish, italian, portuguese,
dutch, swedish, norwegian, danish, russian. Indonesian has stopwords and
**suffix-only** stemming — see [ARCHITECTURE.md](ARCHITECTURE.md) for why prefixes
are deliberately left alone.

---

## Indexing

Writes land in a WAL and are searchable immediately by exhaustive scan. Building
the indexes is a separate step:

```bash
curl -s -X POST $U/v1/namespaces/articles/compact -H "$T"
```

```json
{ "records_in": 5, "docs_out": 5, "wal_consumed": 2,
  "clusters": 3, "cas_attempts": 1, "took_ms": 1977 }
```

A background sweeper does this automatically at 8 MiB or 32 WAL entries. Call it
by hand when you have just bulk-loaded and want to query indexed immediately —
**BM25 and sparse search require it**, and say so if you skip it.

---

## Querying

### Vector search

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" -d '{
  "rank_by": ["vector","ANN",[1,0,0]],
  "top_k": 2,
  "include_attributes": ["title","topic"]
}'
```

```json
{ "rows": [
  { "id": 1, "$dist": 0,           "title": "Rust for beginners",          "topic": "tech" },
  { "id": 2, "$dist": 0.006116271, "title": "Databases on object storage", "topic": "tech" }
]}
```

**`$dist` is a distance: lower is better.** An exact cosine match is 0. Attributes
come back flattened alongside `id`, not nested.

`ANN` uses the index. `kNN` is the same ranking without it — exact, and slower on
anything large:

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" \
  -d '{ "rank_by": ["vector","kNN",[1,0,0]], "top_k": 3 }'
```

`top_k` and `limit` are aliases; `{"limit": {"total": 10}}` is the same thing.
Sending both with different values is an error rather than a guess.

### Recall and `nprobe`

`nprobe` controls how many clusters are searched. Default 8, which reached 100%
recall in [the benchmark](BENCHMARK.md). Lower is faster and less accurate:

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" \
  -d '{ "rank_by": ["vector","ANN",[1,0,0]], "top_k": 5, "nprobe": 1 }'
```

### Order by attribute

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" \
  -d '{ "rank_by": ["score","desc"], "top_k": 3, "include_attributes": ["title","score"] }'
```

```json
{ "rows": [
  { "id": 3, "score": 95, "title": "Rendang recipe" },
  { "id": 1, "score": 90, "title": "Rust for beginners" },
  { "id": 2, "score": 85, "title": "Databases on object storage" }
]}
```

No vector is ranked, so **`$dist` is omitted** rather than reported as zero.
Documents missing the attribute sort last in both directions.

### Full-text search

Requires `full_text_search` on the attribute and a compaction.

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" \
  -d '{ "rank_by": ["body","BM25","rust guide"], "top_k": 3, "include_attributes": ["title"] }'
```

**For BM25, `$dist` is a relevance score: higher is better** — the opposite of a
vector distance. Documents matching no query term are absent, not scored zero.

Stemming means `"guide"` finds `"guides"`, and in Indonesian `"makan"` finds
`"makanan"`.

### Sparse vectors

An object attribute is a sparse vector (`{}f16`):

```bash
curl -s -X POST $U/v2/namespaces/articles -H "$T" -H "$J" -d '{
  "upsert_rows": [{ "id": 6, "vector": [0,0,1], "terms": { "cat": 1.0, "pet": 0.5 } }]
}'
curl -s -X POST $U/v1/namespaces/articles/compact -H "$T"

curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" \
  -d '{ "rank_by": ["terms","SparseKNN",{"cat":1.0,"pet":1.0}], "top_k": 5 }'
```

Scored by dot product over shared dimensions. Higher is better, like BM25.
Documents sharing no dimension are excluded.

---

## Filters

Tuple syntax, identical to turbopuffer's:

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" -d '{
  "rank_by": ["vector","ANN",[1,0,0]],
  "top_k": 10,
  "filters": ["And", [
    ["published","Gte","2024-01-01T00:00:00Z"],
    ["topic","Eq","tech"]
  ]],
  "include_attributes": ["title","published"]
}'
```

| operator | on |
|---|---|
| `Eq` `NotEq` | anything. `Eq` with `null` matches documents missing the attribute |
| `Gt` `Gte` `Lt` `Lte` | numbers, strings, datetimes |
| `In` `NotIn` | a list of candidate values |
| `Contains` `NotContains` | array membership, or substring for text |
| `ContainsAny` `NotContainsAny` | array intersection |
| `Glob` `NotGlob` `IGlob` `NotIGlob` | `*` and `?` wildcards; `I` is case-insensitive |
| `Regex` `NotRegex` | full regex |
| `ContainsAllTokens` | full-text: all tokens present anywhere |
| `ContainsTokenSequence` | full-text: tokens adjacent and in order — a phrase |
| `Fuzzy` | full-text: a token within one edit |
| `And` `Or` `Not` | nesting |

```bash
# array membership
-d '{ "rank_by":["vector","ANN",[1,0,0]], "filters":["tags","Contains","rust"] }'

# phrase, not just the words
-d '{ "rank_by":["vector","ANN",[1,0,0]], "filters":["body","ContainsTokenSequence","object storage"] }'
#   -> [{ "id": 2, "$dist": 0.006116271, "title": "Databases on object storage" }]

# typo tolerance
-d '{ "rank_by":["vector","ANN",[1,0,0]], "filters":["body","Fuzzy","databse"] }'

# nesting
-d '{ "rank_by":["vector","ANN",[1,0,0]],
      "filters":["Or",[["topic","Eq","tech"],["score","Gte",90]]] }'
```

A filter value is coerced to the attribute's declared type, so an ISO string
compares correctly against a `datetime`. A malformed filter is **rejected**, never
silently ignored — silently dropping a tenant filter would return other people's
data.

### Filters and recall

A selective filter is answered exactly from the attribute index rather than by
probing clusters and discarding most of what was read. You do not have to ask for
this; the planner decides. `"prefiltered": true` appears in the native `/v1`
response when it happened.

---

## `include_attributes`

```json
"include_attributes": ["title","score"]   // just these
"include_attributes": true                // everything
"include_attributes": false               // nothing (default)
```

Requesting an attribute a document lacks omits it rather than returning null.

---

## Consistency

```bash
# default: sees every committed write
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" \
  -d '{ "rank_by":["vector","ANN",[1,0,0]], "consistency":{"level":"strong"} }'

# skips the commit-point read
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" \
  -d '{ "rank_by":["vector","ANN",[1,0,0]], "consistency":{"level":"eventual"} }'
```

**This is the largest latency lever available: measured at 145 ms strong versus
0.5 ms eventual** on a distant bucket. Strong consistency costs exactly one
uncacheable round trip to check the commit point.

Strong consistency **refuses rather than lies**: if the unindexed tail is too
large to scan, it returns 503 instead of answering from part of the data.

---

## Aggregations

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" -d '{
  "aggregate_by": {
    "count":   ["Count"],
    "total":   ["Sum","score"],
    "lowest":  ["Min","score"],
    "average": ["Avg","score"]
  },
  "filters": ["topic","Eq","tech"]
}'
```

```json
{ "aggregations": { "average": 87.5, "count": 2, "lowest": 85, "total": 175 } }
```

`Sum` of integers stays an integer; `Avg` returns a float. Summing nothing is `0`,
but averaging nothing is `null` — an average of no rows is undefined, and
reporting zero would be a lie.

No `rank_by`, so no rows. `["Count"]` counts documents; `["Count","score"]` counts
documents having that attribute.

### Grouping

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" \
  -d '{ "aggregate_by": {"count":["Count"],"total":["Sum","score"]}, "group_by": ["topic"] }'
```

```json
{ "aggregation_groups": [
  { "count": 1, "topic": "food",  "total": 95  },
  { "count": 1, "topic": "sport", "total": 60  },
  { "count": 2, "topic": "tech",  "total": 175 }
]}
```

Grouped results come back under `aggregation_groups` and ungrouped ones under
`aggregations` — never both, so you can tell an empty grouped result from an
ungrouped one. Documents missing the attribute form their own `null` group rather
than disappearing, so the counts still add up.

`ForEachUnique` explodes an array into one group per element:

```bash
-d '{ "aggregate_by": {"count":["Count"]}, "group_by": [["ForEachUnique","tags"]] }'
```

A document with two tags counts in both groups, so group totals may exceed the
document count. That is intended.

### With ranking

Send `rank_by` and `aggregate_by` together to get rows **and** totals in one round
trip — a faceted result page:

```bash
-d '{ "rank_by":["vector","ANN",[1,0,0]], "top_k":10, "aggregate_by":{"count":["Count"]} }'
```

---

## Multi-query and hybrid search

Up to 16 sub-queries, executed together:

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" -d '{
  "queries": [
    { "rank_by": ["vector","ANN",[1,0,0]], "top_k": 1 },
    { "aggregate_by": { "n": ["Count"] } }
  ]
}'
```

```json
{ "results": [
  { "rows": [{ "id": 1, "$dist": 0 }] },
  { "aggregations": { "n": 4 } }
]}
```

Add `rerank_by` to fuse them into one list instead:

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" -d '{
  "top_k": 5,
  "rerank_by": ["RRF"],
  "queries": [
    { "rank_by": ["vector","ANN",[1,0,0]], "top_k": 5 },
    { "rank_by": ["body","BM25","object storage"], "top_k": 5 }
  ]
}'
```

```json
{ "rows": [
  { "id": 2, "$dist": 0.032522473 },
  { "id": 1, "$dist": 0.016393442 },
  { "id": 3, "$dist": 0.015873017 }
]}
```

`$dist` here is the fused RRF contribution, on neither input's scale.

**This is hybrid search.** RRF combines only the *ranks*, never the scores,
because a BM25 relevance score and a cosine distance live on incomparable scales —
blending the numbers would let whichever happens to be larger dominate.

`consistency` belongs on the root object, not inside a sub-query; putting it
inside is an error rather than being ignored.

---

## Sharding

Set on a namespace's **first write only**:

```bash
curl -s -X POST $U/v2/namespaces/big -H "$T" -H "$J" -d '{
  "sharding": { "num_shards": 4 },
  "upsert_rows": [ … ]
}'
```

Queries fan out and merge automatically; nothing else changes. Up to 256 shards.

**Shard because a namespace has outgrown a single index, never for speed** — every
shard must answer before the query can, so more shards means higher latency. Going
from 1 to 8 shards measured 602 ms → 1007 ms.

The count cannot be changed afterwards; copy into a new namespace instead.

---

## Native embedding

Configure an OpenAI-compatible endpoint:

```bash
FCKDB_EMBED_URL=https://api.openai.com/v1/embeddings \
FCKDB_EMBED_MODEL=text-embedding-3-small \
FCKDB_EMBED_KEY=sk-… \
cargo run --release -- serve
```

Then send text where a vector would go:

```bash
curl -s -X POST $U/v2/namespaces/articles/query -H "$T" -H "$J" -d '{
  "rank_by": ["vector","ANN",["Embed","articles about systems programming"]],
  "top_k": 5
}'
```

Or embed on write, by declaring which attribute holds the text:

```bash
curl -s -X POST $U/v2/namespaces/articles -H "$T" -H "$J" -d '{
  "schema": { "body": { "type": "string", "embed": { "model": "text-embedding-3-small" } } },
  "upsert_rows": [{ "id": 7, "body": "documents without a vector get one" }]
}'
```

A whole batch is one call to the endpoint. **An explicit vector always wins** over
re-embedding, so correcting a document is never silently overwritten.

Unconfigured, these return 501 naming the variable to set:

```json
{ "error": "native embedding is not configured; set FCKDB_EMBED_URL to an OpenAI-compatible /v1/embeddings endpoint, or send a vector instead" }
```

### A warning about cross-lingual search

Measured against `text-embedding-3-small` on a corpus of six English and six
Indonesian documents: **the model clusters by language more strongly than by
topic** for short queries. The same question gets different answers depending on
which language you ask it in.

```
"coral reefs turning white because the sea is too warm"
  -> #11 How coral reefs bleach            ✅

"terumbu karang yang memutih karena air laut menghangat"   (same question)
  -> #5 Membuat sambal terasi              ❌ a sambal recipe
     #4 Rahasia rendang yang empuk
     #8 Menjaga pola tidur                 ← all three Indonesian, all unrelated
     #11 How coral reefs bleach            ← the right answer, fourth
```

Across five Indonesian queries, 11 of 15 results were Indonesian documents against
a chance baseline of about half.

This is the model, not the engine: ranking and distances were verified against
OpenAI's own embeddings computed independently, and agreed to 3.3e-4 — f32
rounding on a 1536-dimension dot product.

**Hybrid search does not fix it.** RRF combines ranks, so it can only help when at
least one input has the answer somewhere. Here both are wrong:

| approach | result |
|---|---|
| vector only, Indonesian query | ✗ not in top 3 |
| BM25 only, Indonesian query | ✗ not in top 3 |
| hybrid RRF | ✗ **still not in top 3** |
| query with a few terms in the corpus language | ✅ first |
| query translated | ✅ first |

So if your corpus and your queries are in different languages, **translate the
query** before embedding, or append a handful of corpus-language terms to it. A
model with stronger cross-lingual alignment (`text-embedding-3-large`, or a
multilingual-specific model) is the other option, and worth measuring on your own
data rather than assuming.

---

## Operations

### Metadata

```bash
curl -s $U/v1/namespaces/articles/metadata -H "$T"
```

```json
{
  "id": "articles",
  "approx_row_count": 4,
  "approx_logical_bytes": 2691,
  "schema": {
    "body": { "type": "string", "filterable": true, "full_text_search": { … } },
    "published": { "type": "datetime", "filterable": true },
    "score": { "type": "uint", "filterable": true },
    "tags": { "type": "[]string", "filterable": true }
  },
  "index": { "unindexed_bytes": 512 },
  "distance_metric": "cosine_distance",
  "created_at": "2026-08-29T06:42:24.000000000Z",
  "last_write_at": "2026-08-29T06:44:01.000000000Z",
  "encryption": { "mode": "default" }
}
```

Costs exactly **one GET** — every number is recorded in the manifest.
`index.unindexed_bytes` is how far behind indexing is.

### Cache warming

```bash
curl -s -X POST $U/v1/namespaces/articles/warm -H "$T"
```

```json
{ "objects_warmed": 3, "bytes": 794, "already_cached": 0, "took_ms": 434 }
```

Pulls the index into the local cache before a user searches, so they never see
cold latency. Call it when a user opens a search box, not when they type.

### Garbage collection

```bash
curl -s -X POST $U/v1/namespaces/articles/gc -H "$T"
```

```json
{ "scanned": 13, "referenced": 12, "deleted": 0, "spared_recent": 1, "took_ms": 667 }
```

**Compaction is the main producer of garbage** — it writes a fresh segment and
fresh indexes, leaving the entire previous set unreferenced, on top of the WAL
objects it retired. Measured: four compactions left 70% of stored objects
orphaned, invisible to `metadata`, which counts only what is referenced.

A background sweeper does this every 10 minutes, so you do not normally need to
call it. Objects younger than one hour are spared, because an unreferenced object
may be a write still in flight.

### Branching

```bash
curl -s -X POST $U/v1/namespaces/articles/branch/articles_backup -H "$T"
```

```json
{ "source": "articles", "destination": "articles_backup", "objects_copied": 12 }
```

A point-in-time copy, fully independent afterwards. Branching onto a name that
already exists returns 409.

### Listing and deleting

```bash
curl -s $U/v1/namespaces -H "$T"
curl -s -X DELETE $U/v1/namespaces/articles -H "$T"
```

Listing is scoped to your organisation.

### Metrics

```bash
curl -s $U/metrics | grep -v '^#'
```

Prometheus format, no auth. The one to watch:

```
fckdb_class_a_per_write 0.0134
```

**Class-A operations per document written.** Above ~1 the commit path is broken;
below ~0.1 group commit is doing its job. It is the number that decides the bill,
and no latency graph would show it.

---

## Errors

| status | meaning |
|---|---|
| `400` | malformed request, invalid namespace, schema conflict, bad filter |
| `401` | missing or wrong bearer token |
| `404` | no such namespace, or no such route |
| `409` | branch destination already exists |
| `429` | write backpressure — `Retry-After: 5`, compaction was triggered |
| `500` | genuine server fault |
| `501` | recognised but not implemented — `compute_attributes`, `Highlight`, unconfigured embedding |
| `502` | the embedding endpoint failed; its message is passed through |
| `503` | too much unindexed data for a strongly consistent answer; retry or use eventual |

Errors are always JSON: `{"error": "…"}`.

**Unimplemented features return 501 by name.** They are never silently ignored,
because ignoring `aggregate_by` and returning plain vector results is a wrong
answer that looks like a working one.

---

## Limits

| | |
|---|---|
| request body | 512 MB |
| `top_k` | 1–10,000 |
| sub-queries per multi-query | 16 |
| shards per namespace | 256 |
| `delete_by_filter` per request | 5,000,000 |
| `patch_by_filter` per request | 50,000 |
| unindexed tail before writes are refused | 64 MiB |
| unindexed tail a query will scan | 128 MiB |
| namespace name | `[A-Za-z0-9_-]{1,64}` |

---

## What this is not for

Writes cost hundreds of milliseconds because they go to object storage. That is
the trade the architecture makes, and it is not tunable. **Do not put a checkout
flow behind it.** It is built for read-heavy search over data written in batches.

For under a million vectors, Postgres with pgvector will be simpler and faster.
The economics here only start to bite at tens of millions.
