# fckdb

A vector + attribute search engine with **object storage as the only source of
truth**. No consensus layer, no replication, no stateful nodes. Built to test
whether turbopuffer's architecture holds up when you build it yourself — with
every claim measured on real Cloudflare R2 rather than taken from a blog post.

Nodes hold nothing but cache. Losing one costs latency, never data.

## The one idea

A compare-and-swap on a single manifest object replaces the entire consensus
layer. That works only because of a deliberate constraint:

> **One writer per namespace, and no transactions across namespaces.**

A single committer never races itself, so CAS is never contested. That is not a
limitation this design tolerates — it is the thing that makes the design
possible. If your workload cannot accept it, this architecture does not apply
and no amount of object-storage cleverness fixes it.

Everything except the manifest is immutable and uniquely named. That single
property buys three things for free:

- the read cache needs **no invalidation logic**
- a failed CAS leaves **garbage, never corruption**
- garbage collection is a **set difference**

## Layout

```
{prefix}/manifest              commit point. CAS here. The ONLY mutable object.
{prefix}/wal/{seq}-{uuid}.bin  framed Records, uncompacted
{prefix}/data/{uuid}.bin       framed Records, compacted segment
{prefix}/data/{uuid}.cen       centroid blob
{prefix}/data/{uuid}.clu       framed Records, one IVF cluster
```

WAL entries, segments and cluster objects are all just framed `Record`s. One
codec, one place for bugs.

The manifest records the **size and record count of every object inline**, so
backpressure, billing, and the query planner are all answerable from one small
GET — no LIST, no data fetch.

## Write path

```
mem buffer ──group commit (no timer)──▶ PUT wal/xxx.bin ──▶ CAS manifest
                                        └─ orphan if the CAS loses
```

Two pointers, not one:

- **CAS commit point** — advances via compare-and-swap. Defines durability.
- **index cursor** — advances asynchronously. Defines speed.

Data between them is durable but unindexed, and still searchable by exhaustive
scan. That is how strong consistency is delivered without waiting for indexing.

Group commit has **no timer**. The committer blocks for one request, drains
whatever else is queued, and commits the lot. While a commit is in flight,
arrivals pile up and ride the next one — so batch size self-tunes to the
backend's latency with nothing to configure.

The first CAS attempt uses the *remembered* manifest version, skipping the read
entirely. With one committer that guess is right essentially always, so a commit
costs **two requests, not three**. A wrong guess is caught by the CAS.

## Query path

```
Roundtrip 1      │ Roundtrip 2          │ Roundtrip 3
─────────────────┼──────────────────────┼──────────────
manifest         │ centroid index       │ probed clusters
(skipped under   │ unindexed WAL tail   │
 eventual)       │ (concurrent)         │
```

Indexed candidates are overlaid with the WAL tail, so recent upserts and
tombstones always win over what the index still believes.

## Measured on Cloudflare R2

Bucket in the default (distant) region; edge RTT 25 ms.

| | 2,000 docs | 20,000 docs |
|---|---|---|
| Group-commit ingest | 2.1 s, **6 class-A PUTs** | 77.9 s, **42 PUTs** |
| Brute-force query | 771 ms | 27 s |
| Indexed query, cold | 811 ms | **528 ms** |
| Cached query (strong) | 176 ms | 162 ms |
| Cached query (eventual) | **2 ms** | — |
| Recall@10 | 100% | 100% |

### Four results worth knowing

**1. Per-write CAS is unusable; group commit is 67× cheaper.** 16 concurrent
writers cost 270 class-A PUTs and 13.0 s per-write, versus 4 PUTs and 1.7 s
grouped. At 1k writes/sec that is ~$197k/month against ~$28/month. Nothing in a
latency graph would have shown that, which is why `fckdb_class_a_per_write` is a
first-class metric.

**2. The index is a pessimization below ~10k documents.** At 2k docs the whole
dataset is one 1 MB object — 2 GETs to scan everything beats 10 GETs to be
selective (0.8×). At 20k docs the index wins 50.6×. Below the crossover, use
Postgres and pgvector.

**3. Warm query latency is *entirely* the commit-point read.** 9 of 10 GETs hit
cache; the one that cannot is the consistency check. Strong 176 ms → eventual
**2 ms**, an 88× difference from one skipped roundtrip. This is turbopuffer's
"~10 ms consistency floor", reproduced — ours is 176 ms only because the bucket
is far away.

**4. Compaction is O(n) and rebuilds the index wholesale.** 60 s at 20k docs, so
minutes at a million. This is the honest boundary of the index work below.

## What the index is, precisely

**IVF-Flat, not SPFresh.** It has SPFresh's *shape* — centroid-based, so a cold
query is one small centroid fetch plus one parallel burst of posting-list
fetches, which is why it beats HNSW and DiskANN on object storage (HNSW needs
the whole graph resident, defeating the point, and its write amplification
multiplies request cost).

What it lacks is SPFresh's **LIRE protocol**: incremental cluster split/merge
that holds recall under continuous updates. Here the index is rebuilt by
compaction. That is result 4 above, and it is the largest single gap between this
and turbopuffer.

**Recall of 100% is on synthetic clustered data**, which is the easy case. Real
embeddings are messier. The harness is `index::recall` — point it at your own
vectors before believing that number.

## Consistency

| Mode | Cost | Guarantee |
|---|---|---|
| `strong` (default) | one manifest GET per query | sees every committed write, or **errors** |
| `eventual` | zero GETs while the snapshot is fresh | may lag; reports `consistent: false` |

Strong consistency **refuses rather than lies**. If the unindexed tail exceeds
the 128 MiB scan cap, a strong query returns 503 instead of silently answering
from part of the data. Eventual truncates the tail — always keeping a *prefix*,
never a subset, since WAL entries are ordered mutations and dropping from the
middle would resurrect deleted documents.

A committed write is immediately visible to eventual reads on the same node,
because the committer remembers the manifest it just wrote.

## Backpressure

Writes are refused with **429 + Retry-After** at 64 MiB of unindexed tail —
half the query scan cap — so the point where consistent queries become
impossible is never reached in normal operation. Compaction is triggered
automatically on rejection, and a background sweeper compacts at 8 MiB or 32 WAL
entries, whichever comes first.

turbopuffer's documented behaviour past its cliff is that writes stop being
visible while the API keeps returning success. Refusing loudly is the same
backpressure with an error the caller can act on.

## HTTP API

Two surfaces over one engine. `/v1` is native; `/v2` is turbopuffer-compatible.

```
GET    /healthz
GET    /metrics                                  Prometheus, no auth
GET    /v1/namespaces                            list (scoped to your org)
GET    /v1/namespaces/{ns}                       metadata
DELETE /v1/namespaces/{ns}
POST   /v1/namespaces/{ns}/write                 { upsert: [Doc], delete: [id] }
POST   /v1/namespaces/{ns}/query                 { vector, top_k, filter, nprobe, consistency }
POST   /v1/namespaces/{ns}/compact
POST   /v1/namespaces/{ns}/warm                  pull the index into cache
POST   /v1/namespaces/{ns}/gc
POST   /v1/namespaces/{ns}/branch/{dest}         point-in-time copy

POST   /v2/namespaces/{ns}                       turbopuffer-compatible write
POST   /v2/namespaces/{ns}/query                 turbopuffer-compatible query
GET    /v1/namespaces/{ns}/metadata              turbopuffer-compatible metadata
```

Bearer token per tenant, compared in constant time against every configured
token (a map lookup leaks which prefix matched). Namespaces are keyed
`ns/{org}/{name}`, and names are **allowlisted** to `[A-Za-z0-9_-]{1,64}` —
they become object storage paths, so `..` and `/` would let one tenant address
another's data.

```bash
curl -s -X POST localhost:8080/v1/namespaces/docs/query \
  -H 'authorization: Bearer $TOKEN' -H 'content-type: application/json' \
  -d '{"vector":[1,0],"top_k":2,
       "filter":["And",[["lang","Eq","id"],["rank","Gte",20]]],
       "include_attributes":["lang","rank"],
       "consistency":{"mode":"eventual","max_age_ms":60000}}'
```

## Running

```bash
cargo run --release -- serve     # HTTP service
cargo run --release -- e2e       # 15-stage end-to-end exercise
cargo test                       # 131 tests
```

With no `FCKDB_BUCKET`, everything runs against an in-memory store — tests need
no credentials and no network.

| Variable | Meaning |
|---|---|
| `FCKDB_BUCKET` | bucket name; unset means in-memory |
| `FCKDB_ENDPOINT` | `https://{account}.r2.cloudflarestorage.com`, or MinIO |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | credentials |
| `FCKDB_TOKEN` / `FCKDB_TOKENS` | `tok` or `tok1:org1,tok2:org2`. Unset = **auth disabled**, logged loudly |
| `FCKDB_ADDR` | listen address, default `127.0.0.1:8080` |
| `FCKDB_CACHE_PATH` / `FCKDB_CACHE_BYTES` | NVMe ring buffer; unset means no cache |
| `FCKDB_PRICE_*` | override the R2 list prices in the cost model |
| `FCKDB_DOCS` / `FCKDB_DIM` / `FCKDB_NPROBE` | e2e dataset shape |
| `FCKDB_KEEP` | keep e2e objects instead of deleting them |

## Backend compatibility

The gate is one question: **does the store enforce `If-Match` on PUT?** Without
it the commit protocol degrades to last-write-wins. `verify_cas` answers it.

| Backend | CAS | Tested | Result |
|---|---|---|---|
| Cloudflare R2 | yes | **full phases 0–10** | ✅ |
| MinIO `RELEASE.2025-09-07` | yes | **full phases 0–10** | ✅ |
| `object_store` InMemory | yes | **full phases 0–10** | ✅ |
| `object_store` LocalFileSystem | **no** | verify_cas | ❌ `PutMode::Update` not implemented — refused at stage 0 |
| AWS S3 | yes (`If-Match`, Nov 2024) | — | untested; same code path as R2 |
| Tigris, Ceph RGW, Backblaze B2 | claimed | — | untested; run stage 0 first |
| GCS, Azure Blob | yes | — | **not wired.** `open_store` builds only `AmazonS3Builder`; needs the `gcp`/`azure` features and a builder branch |

Anything not marked tested: run `cargo run --release -- e2e` against it. Stage 0
fails loudly rather than corrupting data quietly.

## turbopuffer compatibility

**None, in either direction.** The on-disk format is turbopuffer's own and
undocumented, so data cannot be read across. The HTTP API differs at every
level — see the table below. This is a codebase that *implements the same
architecture*, not a drop-in replacement.

| | turbopuffer | fckdb |
|---|---|---|
| Write | `POST /v2/namespaces/{ns}` | `POST /v1/namespaces/{ns}/write` |
| Write body | `upsert_rows`, `upsert_columns`, `patch_rows`, `patch_by_filter`, `deletes`, `delete_by_filter` | `upsert`, `delete` |
| Query rank | `rank_by: ["vector","ANN",[…]]`; also kNN, BM25, SparseKNN, order-by-attr, `Embed` | `vector: […]`, cosine ANN only |
| Result limit | `limit` | `top_k` |
| Filters | `("And", (("ts","Gte",…),("public","Eq",true)))`; Gt/Lt/In/Glob/Regex | ✅ **same tuple grammar**: Eq/NotEq/Gt/Gte/Lt/Lte/In/NotIn/Glob/IGlob/Contains/Regex + And/Or/Not |
| Score | `$dist` — a **distance**, lower is better | ✅ `$dist` on `/v2` (native `/v1` keeps `score`, a similarity) |
| Attributes | typed + schema, `include_attributes` | ✅ **typed** (bool, uint, int, f64, string, datetime, uuid, []string, []uint), inferred schema, `include_attributes` |
| Document ids | uint / string / uuid | ✅ all three |
| `distance_metric` | cosine, euclidean, … | ✅ cosine, euclidean-squared, dot product |
| Partial update | `patch_rows`, `patch_columns` | ✅ `Record::Patch` — merge, with null removing an attribute |
| Aggregations | `aggregate_by`, group-by | none — returns **501** |
| Multi-vector, sharding, CMEK | yes | no |

The common subset now works: a client doing row or column upserts, deletes,
patches, filter-based writes, ANN or kNN queries with typed filters and
`include_attributes` can point at `/v2` unchanged. What remains is BM25, sparse
vectors, aggregations, multi-query and sharding — each new engine work rather
than adapter work, and each returning **501 Not Implemented** today rather than a
plausible wrong answer.

### Parity progress

Work toward turbopuffer API parity lives on `feat/turbopuffer-parity`.

**Done**

- Typed attribute values (`value.rs`): bool, uint, int, f64, string, datetime,
  uuid, `[]string`, `[]uint` — with a binary codec, JSON inference, and ordering
  that returns *incomparable* rather than equal across mismatched types
- turbopuffer's tuple filter grammar, all operators, `And`/`Or`/`Not` nesting
- `include_attributes`: `true`, `false`, or a list
- `Record::Patch` — partial update, applied identically by compaction and by the
  query-time WAL overlay

- **Namespace schema** in the manifest: attribute types inferred from the first
  write that carries each value, then enforced. A null never declares a type.
  Vector dimension and id type fixed the same way
- **Document ids** as uint, string, or UUID, with coercion to the declared type
- **`distance_metric`** per namespace (cosine, euclidean-squared, dot product),
  honoured by centroid assignment as well as ranking — building with one geometry
  and querying with another puts a document's neighbours in a cluster the query
  never probes

- **Column-oriented and filter-based writes**: `upsert_columns`,
  `patch_columns`, `delete_by_filter`, `patch_by_filter`, with the documented
  caps (5M / 50k) and `rows_remaining`
- **The `/v2` compatibility surface** (`src/compat.rs`): `upsert_rows`,
  `rank_by` ANN/kNN, `filters`, `limit.total`/`top_k`, object-shaped
  `consistency`, flattened result rows with `$dist`, `rows_affected` counters,
  client-declared `schema`, and `GET /v1/namespaces/{ns}/metadata`

**Next, in dependency order**

1. Inverted attribute index (filters are still evaluated per candidate)
2. Order by attribute
3. **BM25** full-text, **aggregations**, sparse vectors, multi-query, sharding —
   each its own project

Two traps to keep in mind while doing this. `$dist` is a distance and `score` is
a similarity, so the conversion inverts ordering — that needs a test asserting
the inversion, not just the field name. And a string that happens to look like a
UUID stays a string until a schema says otherwise; inferring from content would
make an attribute's type depend on which document you looked at first.

### Verify your backend first

`Namespace::verify_cas` is a negative control that runs as stage 0 of the e2e. A
store that accepts a stale `If-Match` degrades the whole commit protocol to
last-write-wins and loses writes **with no error anywhere**.

It must write *distinct* content each time: on S3-compatible stores the ETag is
derived from content, so re-PUTting identical bytes yields the same ETag and a
"stale" version still compares equal — the probe would pass against a backend
that enforces nothing. Verified against R2, MinIO, and the in-memory store.

## Known ceilings

Each is marked with a `ponytail:` comment at the code that owns it.

| Ceiling | Where | Upgrade path |
|---|---|---|
| Index rebuilt wholesale, O(iters·n·k·dim) | `index::build` | SPFresh LIRE incremental split/merge |
| Full-rewrite compaction, needs live set in RAM | `Namespace::compact` | leveled/tiered compaction |
| Write throughput = `MAX_BATCH_LEN` ÷ commit latency | `store` | raise the cap; ~257 docs/s observed |
| Filters evaluated per candidate during the scan | `doc::Filter` | inverted attribute index |
| `ids_matching` full-scans the live set for filter-based writes | `Namespace::ids_matching` | attribute index |
| Glob is `*`/`?` only, not full globset (`**`, `{a,b}`, ranges) | `doc::glob_to_regex` | the `globset` crate |
| Branch copies every object, O(bytes) | `Namespace::branch` | refcounting, at the cost of cross-namespace GC |
| Blocking `pread` on the async runtime | `cache::RingCache` | `spawn_blocking` or io_uring |
| Static tokens, no rotation or scopes | `server::Auth` | real key management |
| Single region, no replication beyond the bucket's own | — | — |

### Operational notes

**GC is the second half of compaction, not error recovery.** Compaction retires
WAL objects from the manifest without deleting them. A namespace that never runs
`gc` grows forever while its manifest stays small.

The **grace window is not optional**. A freshly written WAL object is
unreferenced for the moment between its PUT and its CAS; deleting inside that
window destroys a write that is about to commit. Default is one hour, which must
comfortably exceed any single commit.

**Nothing needs flushing on shutdown.** A write is durable the moment its commit
returns, so no in-memory state exists whose loss could lose an acknowledged
write. Requests in flight simply fail and are retried.

**Set a bucket location hint.** The 176 ms consistency floor above is dominated
by distance to the bucket, not by protocol. It cannot be changed after bucket
creation, and it is the single highest-leverage tuning available.
