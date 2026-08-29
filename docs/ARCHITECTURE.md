# Architecture

How fckdb works, and why each piece is shaped the way it is. Measured figures
live in [BENCHMARK.md](BENCHMARK.md); this document is the reasoning.

---

## 1. The one idea

A compare-and-swap on a single object replaces the entire consensus layer.

That works only because of a constraint accepted up front:

> **One writer per namespace, and no transactions across namespaces.**

A single committer never races itself, so CAS is never contested. This is not a
limitation the design tolerates — it is the thing that makes the design possible.
A workload that cannot accept it needs Raft, and with Raft the cost advantage
disappears.

Everything else follows from one property:

> **Every object except the manifest is immutable and uniquely named.**

Which buys, for free:

| | because |
|---|---|
| the read cache needs no invalidation | a name never refers to different bytes |
| a failed CAS leaves garbage, never corruption | readers only follow the manifest |
| GC is a set difference | anything unnamed is dead |
| branching is a manifest copy | the data it points at cannot change |

---

## 2. Layout on object storage

```
{prefix}/manifest                the commit point. CAS here. The ONLY mutable object.
{prefix}/wal/{seq}-{uuid}.bin    framed Records, uncompacted
{prefix}/data/{uuid}.bin         framed Records — a compacted segment
{prefix}/data/{uuid}.cen         centroids
{prefix}/data/{uuid}.clu         framed Records — one IVF cluster
{prefix}/data/{uuid}.ids         document ids in ordinal order
{prefix}/data/{uuid}.att         inverted attribute index
{prefix}/data/{uuid}.fts         positional term index
{prefix}/data/{uuid}.spx         sparse-vector inverted list
```

WAL entries, segments and cluster objects are all just framed `Record`s. **One
codec, one place for bugs.**

### The manifest carries sizes

Every entry records its own byte count and record count. So backpressure,
billing, the query planner and the metadata endpoint are all answerable from
**one small GET** — no LIST, no data fetch. That is why `metadata` costs exactly
one request, and why a write can decide to refuse itself without reading
anything.

---

## 3. Write path

```
                                        mem buffer
        UPSERT / PATCH / DELETE          ┌──────┐
        ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ─▶   │░░░░░░│
                                        └──┬───┘
WAL                                        │ group commit (no timer)
s3://…/{ns}/wal                            ▼
╔═══════════════════════════════════════════════════════════╗
║ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐              ║
║ │██████│ │██████│ │▞▞▞▞▞▞│ │▞▞▞▞▞▞│ │▞▞▞▞▞▞│   (░░░░░░)   ║
║ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘              ║
║   01       02   ▲   03       04       05   ▲     06       ║
╚═════════════════│══════════════════════════│══════════════╝
             index cursor              CAS commit point

█ indexed + committed   ▞ committed, unindexed   ░ written, not committed
```

**Two pointers, not one.**

- The **CAS commit point** advances by compare-and-swap. It defines durability.
- The **index cursor** advances asynchronously. It defines speed.

Data between them is durable but unindexed, and still searchable by exhaustive
scan. That is how strong consistency is delivered without waiting for indexing.

### Group commit has no timer

The committer blocks for one request, drains whatever else is already queued, and
commits the lot. While a commit is in flight, arrivals pile up and ride the next
one — so batch size **self-tunes to the backend's latency** with nothing to
configure. A fixed window would be more code and strictly worse: it adds latency
when idle and still under-coalesces when busy.

The consequence is the opposite of the usual one: **cost per write falls as load
rises**, because commits are bounded by round trips rather than by document count.
Measured at 0.0080 → 0.0022 class-A ops per document as batches grow.

### The optimistic commit

The first CAS attempt uses the *remembered* manifest version, skipping the read
entirely. With one committer that guess is right essentially always, so a commit
costs **two requests, not three** — and one fewer round trip of latency. A wrong
guess is caught by the CAS and costs one retry.

### Unique names are load-bearing

If a WAL object's name were derived from its sequence number alone, two writers
that both observed `seq=N` would write the **same path**, and the later would
silently overwrite the earlier. The CAS winner would then claim a file holding the
loser's bytes: a write vanishes with no error anywhere. Names carry a UUID for
exactly this reason, and `concurrent_writers_lose_nothing` is the test that fails
if it is removed.

---

## 4. Read path

```
Roundtrip 1        │ Roundtrip 2           │ Roundtrip 3
───────────────────┼───────────────────────┼──────────────────
manifest           │ centroid index        │ probed clusters
(skipped entirely  │ unindexed WAL tail    │
 under eventual)   │ (concurrent)          │
```

### Two paths for a filtered query

Chosen by comparing what each would actually read, and what each would return:

1. **Cost.** The exact path reads `candidates` documents; the cluster path reads
   roughly `docs × nprobe / clusters`. Exact is cheaper precisely when the filter
   is more selective than the fraction of the index a probe touches.
2. **Recall.** Filtering *after* probing discards most of what it read. If the
   probed clusters would leave fewer than a few survivors per requested result,
   the ranked list is drawn from almost nothing. Measured at **30% recall** for a
   5%-selective filter at `nprobe=1` before this guard existed.

The second reason is the one that matters, and it was found by benchmarking, not
by reasoning. A cheap wrong answer is not a bargain.

### The tail always wins

Every ranked path overlays the unindexed WAL on top of whatever the index
believes. A document the tail touched is **rescored from its current state**
rather than trusted — otherwise a patch that removes the query terms would leave
the document ranked on stale postings.

---

## 5. Indexes

| index | object | answers |
|---|---|---|
| IVF centroids + clusters | `.cen`, `.clu` | vector similarity |
| inverted attribute | `.att` | filters, and the exact filtered path |
| positional term | `.fts` | BM25, phrase, fuzzy |
| sparse inverted list | `.spx` | `SparseKNN` |
| ordinal → id | `.ids` | resolving the above back to documents |

### Why IVF and not HNSW

Centroid-based indexes minimise **roundtrips** and **write amplification**, which
are the two things object storage charges for. HNSW needs the whole graph
resident, which defeats the point of putting data in a bucket, and each update
touches many graph nodes, multiplying PUTs.

**This is IVF-Flat, not SPFresh.** It has the right shape but rebuilds wholesale
on compaction rather than maintaining clusters incrementally. That is the largest
single gap between this and turbopuffer.

### The attribute index has a loose contract

`select` may return a **superset**, or `None` when it cannot bound the answer.
Callers re-apply the real filter while scoring, so a superset costs wasted work
and never a wrong answer. Three cases needed care:

- **`And`** drops unanswerable operands — intersecting fewer constraints still
  yields a superset.
- **`Or`** must not — dropping a branch loses its matches entirely.
- **`Not`** requires an *exact* operand, since complementing a superset excludes
  documents that do match. And negations are unanswerable from an inverted index
  at all: a document with no value for the attribute appears nowhere in it, yet
  satisfies `NotEq`.

---

## 6. Consistency

| mode | cost | guarantee |
|---|---|---|
| `strong` (default) | one manifest GET per query | sees every committed write, or **errors** |
| `eventual` | zero GETs while the snapshot is fresh | may lag; reports `consistent: false` |

Strong consistency **refuses rather than lies**. If the unindexed tail exceeds the
128 MiB scan cap, a strong query returns 503 instead of quietly answering from
part of the data.

Eventual truncates the tail — always keeping a **prefix**, never a subset. WAL
entries are ordered mutations, so dropping from the middle would apply a later
upsert without the earlier one it supersedes, and resurrect deleted documents.

A committed write is immediately visible to eventual reads *on the same node*,
because the committer remembers the manifest it just wrote.

**Measured: strong 145 ms, eventual 0.5 ms.** A 300× difference from one skipped
round trip — the single largest lever in the system.

---

## 7. Sharding

Shards share the WAL, so a write still commits once. They do **not** share
indexes: each indexes only its own documents, which is what lets a namespace grow
past the size a single index can serve.

Assignment is **FNV-1a, not `DefaultHasher`**. The assignment is persisted, and a
hash whose output can change between Rust releases would silently relocate every
document, leaving each shard's index describing rows it no longer owns.

- **Ranked queries** fan out concurrently and merge. The merge is exact because
  the global top-k is a subset of the union of per-shard top-k.
- **Aggregations and ordering** gather instead — a partial average cannot be
  reconstructed from finalized per-shard averages.
- **The tail is partitioned** the same way documents are. Applied whole to every
  shard, one document would appear in several candidate sets and survive the
  merge more than once.

**Sharding costs latency; it does not buy it.** Every shard must answer before the
query can, so the slowest sets the tail: 602 → 1007 ms from one shard to eight.
Shard because a namespace has outgrown one index, never for speed.

---

## 8. Durability, and what a node actually holds

A node holds exactly two things, and neither is authoritative:

1. **The read cache** — immutable objects only, on a local ring buffer. Entries
   are checksummed, so a failing disk costs a refetch, not a wrong answer.
2. **In-flight requests** — records queued in the committer that have not
   committed yet.

A write is acknowledged **only after its commit returns**. So a node dying
mid-flight fails the request; it never loses a write the caller believes
succeeded. Nothing else is at risk, because nothing else is local.

There is nothing to flush on shutdown, for the same reason.

### Running more than one node

The single-committer-per-namespace invariant is **process-local**. Two processes
writing the same namespace remain *correct* — CAS is what enforces correctness,
not the committer — but they contend on the manifest and lose the batching that
makes writes cheap. Measured on the per-write CAS path this replaced: 16.9
class-A operations per document instead of 0.013.

So the deployment shape that works is: **route a namespace to one writer**. Reads
scale freely across any number of nodes, since they hold nothing.

---

## 9. Garbage collection

GC is a set difference, because every data object is immutable and uniquely
named: anything storage holds that the manifest does not name is dead.

Two things produce orphans, and **the second is the common one**:

- a writer that died between its PUT and its CAS
- **compaction**, which retires WAL objects from the manifest without deleting
  them

So GC is not an error-recovery path that rarely runs. It is the second half of
compaction, and a namespace that never runs it grows forever while its manifest
stays small. Measured: four compactions of a 200-document namespace left 109
objects of which 33 were live — **70% garbage**, and the manifest reported only
the live portion.

A background sweeper runs every 10 minutes and collects anything unreferenced and
older than the grace window. It skips namespaces that have not compacted since
their last sweep, and deliberately does *not* mark a namespace done while orphans
were spared for being too young — doing so would mean never returning for them,
which is the leak it exists to close.

The **grace window is not optional**. A freshly written WAL object is unreferenced
for the moment between its PUT and its CAS; deleting inside that window destroys a
write that is about to commit. Default is one hour, which must comfortably exceed
any single commit.

---

## 10. Backpressure

Writes are refused with **429 + Retry-After** at 64 MiB of unindexed tail — half
the query scan cap — so the point where consistent queries become impossible is
never reached in normal operation. Compaction is triggered on rejection, and a
background sweeper compacts at 8 MiB or 32 WAL entries.

turbopuffer's documented behaviour past its own cliff is that writes stop being
visible while the API keeps returning success. Refusing loudly is the same
backpressure with an error the caller can act on.

---

## 11. Module map

| module | holds |
|---|---|
| `value` | typed attribute values, binary codec, JSON inference, ordering |
| `doc` | documents, records, ids, filters, schema, distance metrics |
| `store` | WAL, CAS commit, group commit, compaction, query planner, GC, branching |
| `index` | IVF build, probe, recall — no IO |
| `attrindex` | inverted attribute indexes |
| `fts` | tokenizer, positional term index, BM25 |
| `sparse` | sparse-vector inverted lists |
| `aggregate` | Count/Sum/Min/Max/Avg, grouping |
| `embed` | the `Embedder` trait and an OpenAI-compatible client |
| `cache` | NVMe ring buffer |
| `wire` | request and response types shared by engine and server |
| `compat` | the turbopuffer `/v2` translation layer — no search logic |
| `server` | HTTP, auth, tenancy, backpressure, background compaction |
| `ops` | cost accounting and Prometheus output |
| `bench` | the benchmark harness |

Metrics counters live on the process, not on a namespace: they have to outlive
the namespace they describe, or deleting one makes the totals fall and Prometheus
reads that as a restart. A scrape reads only atomics and the last-known size each
namespace recorded when it last loaded its manifest — **monitoring must not cost
requests**, or it bills you for observing and gets slower as the system grows.

`compat` deliberately contains **no search logic**. If it starts to, the two
surfaces have diverged and one of them is lying about what the engine does.

---

## 12. Known ceilings

Each is marked with a `ponytail:` comment at the code that owns it; the README
tabulates all of them. The four that matter most:

1. **Compaction rebuilds indexes wholesale.** O(iters·n·k·dim), needs the live set
   in memory. 7.6 s at 8k documents means minutes at a million. Leveled compaction
   plus SPFresh-style incremental cluster maintenance is the upgrade, and it is a
   large one.
2. **The IVF index is not SPFresh.** Right shape, no incremental split/merge.
3. **Ordering by attribute still materializes every shard**, because a top-k by
   attribute compares documents against each other. Aggregation no longer does:
   it accumulates into fixed-size state and streams one shard at a time.
4. **A full-text query fetches the whole term index object.** A term dictionary
   with offsets at the tail would allow range-requesting only the query's terms.
5. **HTTP status is chosen by matching engine error text.** Fragile in exactly the
   way it looks; a typed error enum is the fix.
