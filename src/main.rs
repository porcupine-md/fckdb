//! fckdb — two modes:
//!
//!   cargo run -- serve      HTTP service on FCKDB_ADDR (default 127.0.0.1:8080)
//!   cargo run -- e2e        end-to-end exercise of phases 0-10 (default)
//!
//! Both run against whatever `open_store` resolves: InMemory with no env, MinIO
//! or R2 with FCKDB_* set.

use anyhow::Result;
use fckdb::cache::RingCache;
use fckdb::doc::{Doc, Filter, Hit, Include, Op, Record};
use fckdb::ops::{self, Pricing};
use fckdb::server::{AppState, Auth, router, serve};
use fckdb::store::{GroupCommit, Namespace, open_store};
use fckdb::wire::{Consistency, QueryRequest};
use fckdb::index;
use fckdb::value::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_flag(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| v != "0" && v != "false")
}

/// Deterministic clustered vectors. Real embeddings cluster; uniform random data
/// makes any IVF index look bad and predicts nothing about production recall.
fn synth(n: usize, dim: usize, clusters: usize) -> Vec<Doc> {
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut rng = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32 - 0.5
    };
    let centers: Vec<Vec<f32>> =
        (0..clusters).map(|_| (0..dim).map(|_| rng() * 10.0).collect()).collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % clusters];
            let v = c.iter().map(|x| x + rng() * 0.8).collect();
            Doc::new(i as u64, v).with_attr("tenant", if i % 3 == 0 { "a" } else { "b" })
        })
        .collect()
}

fn ids(hits: &[Hit]) -> Vec<u64> {
    hits.iter().filter_map(|h| h.id.as_uint()).collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "fckdb=info,tower_http=warn".into()),
        )
        .init();

    match std::env::args().nth(1).unwrap_or_else(|| "e2e".into()).as_str() {
        "serve" => run_serve().await,
        "e2e" => run_e2e().await,
        other => {
            eprintln!("unknown mode {other:?}; expected `serve` or `e2e`");
            std::process::exit(2);
        }
    }
}

async fn run_serve() -> Result<()> {
    let (store, backend) = open_store()?;
    tracing::info!("backend {backend}");

    let cache = match std::env::var("FCKDB_CACHE_PATH") {
        Ok(path) => {
            let bytes = env_usize("FCKDB_CACHE_BYTES", 4 << 30) as u64;
            tracing::info!("NVMe cache {path} ({} MiB)", bytes / (1 << 20));
            Some(Arc::new(RingCache::open(path, bytes)?))
        }
        Err(_) => {
            tracing::warn!("FCKDB_CACHE_PATH unset: no read cache, every query pays cold latency");
            None
        }
    };

    let addr = std::env::var("FCKDB_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    serve(AppState::new(store, cache), &addr).await
}

async fn run_e2e() -> Result<()> {
    let n = env_usize("FCKDB_DOCS", 2000);
    let dim = env_usize("FCKDB_DIM", 128);
    let nprobe = env_usize("FCKDB_NPROBE", 8);
    let top_k = 10;
    let started = Instant::now();

    let (store, backend) = open_store()?;
    let run = uuid::Uuid::new_v4();
    let ns = Arc::new(Namespace::new(store.clone(), format!("ns/e2e-{run}")));

    println!("backend : {backend}");
    println!("dataset : {n} docs x {dim} dims, nprobe={nprobe}, top_k={top_k}");
    println!("prefix  : {}\n", ns.prefix);

    // -------------------------------------------------- phase 1-2: commit protocol
    print!("[0]  CAS enforcement .......... ");
    let probe = Namespace::new(store.clone(), format!("ns/e2e-{run}-casprobe"));
    let t = Instant::now();
    probe.verify_cas().await?;
    println!("enforced ({:.0?})", t.elapsed());

    let docs = synth(n, dim, 20);
    let commit = Arc::new(GroupCommit::new(ns.clone()));
    print!("[1]  group-commit ingest ...... ");
    let t = Instant::now();
    let tasks: Vec<_> = docs
        .iter()
        .cloned()
        .map(|d| {
            let c = commit.clone();
            tokio::spawn(async move { c.upsert(d).await })
        })
        .collect();
    for task in tasks {
        task.await??;
    }
    let ingest = t.elapsed();
    let (batches, attempts) = commit.stats();
    println!(
        "{n} docs in {:.1?}  ({batches} batches, {attempts} CAS, {} class-A PUTs)",
        ingest,
        attempts * 2
    );

    // -------------------------------------------------- phase 3: query path
    print!("[2]  brute-force query ........ ");
    let q = docs[n / 2].vector.clone();
    let t = Instant::now();
    let exact = ns.query_brute(&q, top_k, None).await?;
    let brute = t.elapsed();
    println!("{:.0?}  top id={} score={:.4}", brute, exact[0].id, exact[0].score);
    assert_eq!(exact[0].id, docs[n / 2].id, "a doc must be its own nearest neighbour");

    // -------------------------------------------------- phase 4-5: compact + index
    print!("[3]  compact + build index .... ");
    let c = ns.compact(true).await?;
    println!(
        "{:.1?}  {} records -> {} docs, {} clusters, {} WAL retired",
        Duration::from_millis(c.took_ms),
        c.records_in,
        c.docs_out,
        c.clusters,
        c.wal_consumed
    );

    print!("[4]  indexed query (cold) ..... ");
    let cold_ns = Namespace::new(store.clone(), ns.prefix.clone());
    let t = Instant::now();
    let res = cold_ns.query(&QueryRequest::new(q.clone()).top_k(top_k).nprobe(nprobe)).await?;
    let cold = t.elapsed();
    println!(
        "{:.0?}  {} GETs  indexed={}  recall@{top_k}={:.0}%",
        cold,
        res.object_gets,
        res.indexed,
        index::recall(&exact, &res.hits) * 100.0
    );
    let approx = res.hits.clone();

    print!("[5]  recall over 25 queries ... ");
    let mut total = 0.0;
    for d in docs.iter().step_by((n / 25).max(1)).take(25) {
        let e = ns.query_brute(&d.vector, top_k, None).await?;
        let a = ns.query(&QueryRequest::new(d.vector.clone()).top_k(top_k).nprobe(nprobe)).await?;
        total += index::recall(&e, &a.hits);
    }
    println!("mean recall@{top_k} = {:.1}%", total / 25.0 * 100.0);

    // -------------------------------------------------- phase 7: cache
    let cache_path = std::env::temp_dir().join(format!("fckdb-cache-{run}"));
    let cache = Arc::new(RingCache::open(&cache_path, 512 << 20)?);
    let warm_ns = Namespace::new(store.clone(), ns.prefix.clone()).with_cache(cache.clone());

    print!("[6]  cache warm hint .......... ");
    let w = warm_ns.warm().await?;
    println!(
        "{:.0?}  {} objects, {} KiB",
        Duration::from_millis(w.took_ms),
        w.objects_warmed,
        w.bytes / 1024
    );

    print!("[7]  cached query ............. ");
    let t = Instant::now();
    let res = warm_ns.query(&QueryRequest::new(q.clone()).top_k(top_k).nprobe(nprobe)).await?;
    let warm = t.elapsed();
    let (hits, misses, evictions) = cache.stats();
    println!(
        "{:.0?}  {} GETs  (cache {hits} hit / {misses} miss / {evictions} evict)",
        warm, res.object_gets
    );
    assert_eq!(res.hits, approx, "cached query returned a different answer");

    // -------------------------------------------------- phase 8: consistency
    print!("[8]  strong vs eventual ....... ");
    let strong = warm_ns.query(&QueryRequest::new(q.clone()).top_k(top_k).nprobe(nprobe)).await?;
    let eventual = warm_ns
        .query(
            &QueryRequest::new(q.clone())
                .top_k(top_k)
                .nprobe(nprobe)
                .consistency(Consistency::Eventual { max_age_ms: 60_000 }),
        )
        .await?;
    println!(
        "strong {}ms/{} GETs (consistent={})  eventual {}ms/{} GETs (consistent={})",
        strong.took_ms,
        strong.object_gets,
        strong.consistent,
        eventual.took_ms,
        eventual.object_gets,
        eventual.consistent
    );
    assert_eq!(eventual.hits, strong.hits, "eventual returned a different answer");
    assert_eq!(eventual.object_gets, 0, "eventual query still read the commit point");

    // -------------------------------------------------- overlay + filter
    print!("[9]  WAL overlay .............. ");
    ns.write_records(&[Record::Upsert(Doc::new(u64::MAX, q.clone()))]).await?;
    let seen = ns.query(&QueryRequest::new(q.clone()).top_k(top_k).nprobe(nprobe)).await?;
    let saw_new = ids(&seen.hits).contains(&u64::MAX);
    ns.write_records(&[Record::Delete(docs[n / 2].id.clone())]).await?;
    let after = ns.query(&QueryRequest::new(q.clone()).top_k(top_k).nprobe(nprobe)).await?;
    let hid_old = !after.hits.iter().any(|h| h.id == docs[n / 2].id);
    println!("unindexed upsert visible={saw_new}, tombstone suppresses indexed doc={hid_old}");
    assert!(saw_new && hid_old, "WAL overlay is broken");

    print!("[10] filtered query ........... ");
    let f = ns
        .query(
            &QueryRequest::new(q.clone())
                .top_k(top_k)
                .nprobe(nprobe)
                .filter(Filter::eq("tenant", "a")),
        )
        .await?;
    let live = Namespace::materialize(ns.all_records().await?);
    let clean = f.hits.iter().all(|h| live[&h.id].attrs["tenant"] == Value::from("a"));
    println!("{} hits, all tenant=a: {clean}", f.hits.len());
    assert!(clean, "filter leaked a foreign tenant");

    // -------------------------------------------------- parity: typed attrs
    print!("[11] typed filter + patch ..... ");
    let base = fckdb::value::parse_datetime("2024-03-01T00:00:00Z")?;
    let day = 86_400_000_000_000i64;
    let typed: Vec<Record> = (0..40u64)
        .map(|i| {
            Record::Upsert(
                Doc::new(1_000_000 + i, vec![1.0; dim])
                    .with_attr("when", Value::Datetime(base + i as i64 * day))
                    .with_attr("rank", Value::Uint(i))
                    .with_attr("tags", Value::StringArray(vec![format!("t{}", i % 3)])),
            )
        })
        .collect();
    ns.write_records(&typed).await?;
    let cutoff = Value::Datetime(base + 20 * day);
    let tf = ns
        .query(
            &QueryRequest::new(vec![1.0; dim])
                .top_k(100)
                .nprobe(nprobe)
                .include(Include::Only(vec!["rank".into()]))
                .filter(Filter::And(vec![
                    Filter::cmp("when", Op::Gte, cutoff),
                    Filter::cmp("tags", Op::Contains, "t1"),
                ])),
        )
        .await?;
    let ranks: Vec<u64> = tf
        .hits
        .iter()
        .filter_map(|h| match h.attrs.get("rank") {
            Some(Value::Uint(r)) => Some(*r),
            _ => None,
        })
        .collect();
    let ok_typed = !ranks.is_empty() && ranks.iter().all(|r| *r >= 20 && r % 3 == 1);

    // Patch one of them and confirm it applies over the index.
    ns.write_records(&[Record::Patch {
        id: fckdb::doc::Id::Uint(1_000_000),
        attrs: std::collections::BTreeMap::from([("rank".to_string(), Value::Uint(999))]),
    }])
    .await?;
    let patched = ns
        .query(
            &QueryRequest::new(vec![1.0; dim])
                .top_k(100)
                .nprobe(nprobe)
                .include(Include::All)
                .filter(Filter::eq("rank", 999u64)),
        )
        .await?;
    let ok_patch = patched.hits.iter().any(|h| h.id == fckdb::doc::Id::Uint(1_000_000));
    println!(
        "{} typed hits (all rank>=20 and rank%3==1: {ok_typed}), patch visible over index: {ok_patch}",
        tf.hits.len()
    );
    assert!(ok_typed, "typed range/array filter selected the wrong documents");
    assert!(ok_patch, "patch was not visible through the indexed query path");

    // -------------------------------------------------- phase 9: HTTP surface
    print!("[12] HTTP surface ............. ");
    let http = http_smoke(store.clone()).await?;
    println!("{http}");

    // -------------------------------------------------- phase 10: operations
    print!("[13] GC ....................... ");
    let spared = ns.gc(Duration::from_secs(3600)).await?;
    let swept = ns.gc(Duration::ZERO).await?;
    println!(
        "{} scanned, {} referenced | grace 1h: {} deleted / {} spared | grace 0: {} deleted",
        swept.scanned, swept.referenced, spared.deleted, spared.spared_recent, swept.deleted
    );
    let post_gc = ns.query(&QueryRequest::new(q.clone()).top_k(top_k).nprobe(nprobe)).await?;
    assert_eq!(post_gc.hits.len(), top_k, "GC destroyed live data");

    print!("[14] branch ................... ");
    let dest = format!("ns/e2e-{run}-branch");
    let copied = ns.branch(&dest).await?;
    let branched = Namespace::new(store.clone(), dest.clone());
    let bres = branched.query(&QueryRequest::new(q.clone()).top_k(top_k).nprobe(nprobe)).await?;
    println!("{copied} objects copied, branch returns {} hits", bres.hits.len());
    assert_eq!(bres.hits, post_gc.hits, "branch is not a faithful copy");

    print!("[15] metadata ................. ");
    let md = ns.metadata().await?;
    println!(
        "{} indexed docs, {} clusters, {} unindexed, {} KiB total, backpressure={}",
        md.indexed_docs,
        md.clusters,
        md.unindexed_records,
        md.total_bytes / 1024,
        md.write_backpressure
    );

    // -------------------------------------------------- summary
    let snap = ns.metrics.snapshot();
    let cost = ops::estimate(&snap, md.total_bytes, started.elapsed(), &Pricing::from_env());
    println!("\nlatency  brute={:.0?}  cold-indexed={:.0?}  warm-cached={:.0?}", brute, cold, warm);
    // Ratios are only meaningful once the measurements are above timer noise; an
    // in-memory backend finishes in microseconds and would report nonsense.
    let ratio = |a: Duration, b: Duration| {
        if b.as_micros() < 200 || a.as_micros() < 200 {
            "n/a (below timer noise)".to_string()
        } else {
            format!("{:.1}x", a.as_secs_f64() / b.as_secs_f64())
        }
    };
    println!(
        "speedup  index {} over brute, cache {} over cold",
        ratio(brute, cold),
        ratio(cold, warm)
    );
    println!(
        "requests {} GET / {} PUT / {} DELETE / {} LIST, {} CAS conflicts",
        snap.gets, snap.puts, snap.deletes, snap.lists, snap.cas_conflicts
    );
    println!(
        "cost     {:.4} class-A ops per record written  |  extrapolated ${:.2}/month at this rate",
        cost.class_a_per_write, cost.monthly_total_usd
    );

    let _ = std::fs::remove_file(&cache_path);
    if env_flag("FCKDB_KEEP") {
        println!("\nkept {} and {dest} (FCKDB_KEEP set)", ns.prefix);
    } else {
        let a = ns.destroy().await?;
        let b = branched.destroy().await?;
        let c = probe.destroy().await?;
        println!("\ncleanup  {} objects removed", a + b + c);
    }
    println!("\nOK: phases 0-10 green");
    Ok(())
}

/// Drive the real router in-process: middleware, auth, validation, handlers.
/// No socket, but every layer the service layer adds is exercised against the
/// same backend as everything above.
async fn http_smoke(store: Arc<dyn object_store::ObjectStore>) -> Result<String> {
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode, header};
    use tower::ServiceExt;

    let state = AppState::new(store, None)
        .with_auth(Auth::from_pairs(vec![("e2e-token".into(), "e2e-org".into())]));
    let ns = format!("smoke{}", uuid::Uuid::new_v4().simple());

    let send = |method: &str, uri: String, token: Option<&str>, body: Option<serde_json::Value>| {
        let app = router(state.clone());
        let method = method.to_string();
        let token = token.map(str::to_string);
        async move {
            let mut req = Request::builder().method(method.as_str()).uri(uri);
            if let Some(t) = token {
                req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
            }
            let req = match body {
                Some(v) => req
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&v).unwrap()))
                    .unwrap(),
                None => req.body(Body::empty()).unwrap(),
            };
            let res = app.oneshot(req).await.unwrap();
            let status = res.status();
            let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
            (status, String::from_utf8_lossy(&bytes).to_string())
        }
    };

    // Unauthenticated write must be refused.
    let (status, _) = send(
        "POST",
        format!("/v1/namespaces/{ns}/write"),
        None,
        Some(serde_json::json!({ "upsert": [{ "id": 1, "vector": [1.0, 0.0] }] })),
    )
    .await;
    if status != StatusCode::UNAUTHORIZED {
        anyhow::bail!("unauthenticated write returned {status}, expected 401");
    }

    let (status, body) = send(
        "POST",
        format!("/v1/namespaces/{ns}/write"),
        Some("e2e-token"),
        Some(serde_json::json!({ "upsert": [
            { "id": 1, "vector": [1.0, 0.0], "attrs": { "t": "a" } },
            { "id": 2, "vector": [0.0, 1.0], "attrs": { "t": "b" } },
        ] })),
    )
    .await;
    if status != StatusCode::OK {
        anyhow::bail!("write returned {status}: {body}");
    }

    let (status, body) = send(
        "POST",
        format!("/v1/namespaces/{ns}/query"),
        Some("e2e-token"),
        Some(serde_json::json!({ "vector": [1.0, 0.0], "top_k": 1 })),
    )
    .await;
    if status != StatusCode::OK {
        anyhow::bail!("query returned {status}: {body}");
    }
    let parsed: serde_json::Value = serde_json::from_str(&body)?;
    if parsed["hits"][0]["id"] != 1 {
        anyhow::bail!("query returned the wrong hit: {body}");
    }

    // The turbopuffer-compatible surface, against the same backend.
    let (status, body) = send(
        "POST",
        format!("/v2/namespaces/{ns}"),
        Some("e2e-token"),
        Some(serde_json::json!({
            "upsert_rows": [
                { "id": 10, "vector": [1.0, 0.0], "lang": "id", "rank": 3,
                  "when": "2024-03-05T00:00:00Z" },
                { "id": 11, "vector": [0.0, 1.0], "lang": "en", "rank": 4,
                  "when": "2023-01-01T00:00:00Z" },
            ],
            "schema": { "when": { "type": "datetime" } }
        })),
    )
    .await;
    if status != StatusCode::OK {
        anyhow::bail!("v2 write returned {status}: {body}");
    }
    let parsed: serde_json::Value = serde_json::from_str(&body)?;
    if parsed["rows_upserted"] != 2 {
        anyhow::bail!("v2 write reported the wrong count: {body}");
    }

    let (status, body) = send(
        "POST",
        format!("/v2/namespaces/{ns}/query"),
        Some("e2e-token"),
        Some(serde_json::json!({
            "rank_by": ["vector", "ANN", [1.0, 0.0]],
            "limit": { "total": 5 },
            "filters": ["when", "Gte", "2024-01-01T00:00:00Z"],
            "include_attributes": ["lang"],
            "consistency": { "level": "strong" }
        })),
    )
    .await;
    if status != StatusCode::OK {
        anyhow::bail!("v2 query returned {status}: {body}");
    }
    let parsed: serde_json::Value = serde_json::from_str(&body)?;
    let rows = parsed["rows"].as_array().cloned().unwrap_or_default();
    // The declared datetime made the range filter select exactly one row, and
    // $dist must be a distance: an exact cosine match is ~0.
    let dist = rows.first().and_then(|r| r["$dist"].as_f64()).unwrap_or(f64::NAN);
    if rows.len() != 1 || rows[0]["id"] != 10 || rows[0]["lang"] != "id" || dist.abs() > 1e-4 {
        anyhow::bail!("v2 query returned the wrong shape: {body}");
    }

    let (status, md) =
        send("GET", format!("/v1/namespaces/{ns}/metadata"), Some("e2e-token"), None).await;
    if status != StatusCode::OK {
        anyhow::bail!("v2 metadata returned {status}: {md}");
    }
    let parsed: serde_json::Value = serde_json::from_str(&md)?;
    if parsed["schema"]["when"]["type"] != "datetime" {
        anyhow::bail!("v2 metadata lost the declared schema: {md}");
    }

    // BM25 full-text, through the compatibility surface, against the same backend.
    let fts_ns = format!("fts{}", uuid::Uuid::new_v4().simple());
    let (status, body) = send(
        "POST",
        format!("/v2/namespaces/{fts_ns}"),
        Some("e2e-token"),
        Some(serde_json::json!({
            "upsert_rows": [
                { "id": 1, "vector": [1.0, 0.0], "body": "the quick brown fox jumps over the lazy dog" },
                { "id": 2, "vector": [0.0, 1.0], "body": "a quick brown dog" },
                { "id": 3, "vector": [0.5, 0.5], "body": "unrelated notes about databases" },
            ],
            "schema": { "body": { "type": "string", "full_text_search": true } }
        })),
    )
    .await;
    if status != StatusCode::OK {
        anyhow::bail!("fts write returned {status}: {body}");
    }
    let (status, body) =
        send("POST", format!("/v1/namespaces/{fts_ns}/compact"), Some("e2e-token"), None).await;
    if status != StatusCode::OK {
        anyhow::bail!("fts compact returned {status}: {body}");
    }
    let (status, body) = send(
        "POST",
        format!("/v2/namespaces/{fts_ns}/query"),
        Some("e2e-token"),
        Some(serde_json::json!({
            "rank_by": ["body", "BM25", "foxes jumping"],
            "top_k": 5
        })),
    )
    .await;
    if status != StatusCode::OK {
        anyhow::bail!("bm25 query returned {status}: {body}");
    }
    let parsed: serde_json::Value = serde_json::from_str(&body)?;
    let rows = parsed["rows"].as_array().cloned().unwrap_or_default();
    // Stemming maps "foxes"->"fox" and "jumping"->"jump", so only document 1
    // matches; a document with neither term must be absent, not scored zero.
    if rows.len() != 1 || rows[0]["id"] != 1 {
        anyhow::bail!("bm25 ranking wrong: {body}");
    }
    let (status, body) = send(
        "POST",
        format!("/v2/namespaces/{fts_ns}/query"),
        Some("e2e-token"),
        Some(serde_json::json!({
            "rank_by": ["vector", "ANN", [1.0, 0.0]],
            "top_k": 5,
            "filters": ["body", "ContainsTokenSequence", "quick brown fox"]
        })),
    )
    .await;
    if status != StatusCode::OK {
        anyhow::bail!("phrase filter returned {status}: {body}");
    }
    let parsed: serde_json::Value = serde_json::from_str(&body)?;
    if parsed["rows"].as_array().map(|r| r.len()) != Some(1) {
        anyhow::bail!("phrase filter matched the wrong set: {body}");
    }
    let _ = send("DELETE", format!("/v1/namespaces/{fts_ns}"), Some("e2e-token"), None).await;

    // Aggregations and grouping, through /v2 against the same backend.
    let (status, body) = send(
        "POST",
        format!("/v2/namespaces/{ns}/query"),
        Some("e2e-token"),
        Some(serde_json::json!({
            "aggregate_by": { "n": ["Count"], "total": ["Sum", "rank"] },
            "group_by": ["lang"]
        })),
    )
    .await;
    if status != StatusCode::OK {
        anyhow::bail!("aggregation returned {status}: {body}");
    }
    let parsed: serde_json::Value = serde_json::from_str(&body)?;
    let groups = parsed["aggregation_groups"].as_array().cloned().unwrap_or_default();
    // A grouped result must not also report an ungrouped total. Documents
    // without the attribute form their own null group rather than vanishing, so
    // the group counts still add up to the document count.
    if parsed.get("aggregations").is_some() {
        anyhow::bail!("a grouped result also reported a total: {body}");
    }
    let find = |lang: &str| {
        groups
            .iter()
            .find(|g| g["lang"] == lang)
            .map(|g| (g["n"].as_u64().unwrap_or(0), g["total"].as_u64().unwrap_or(0)))
    };
    let has_null_group = groups.iter().any(|g| g["lang"].is_null());
    if find("id") != Some((1, 3)) || find("en") != Some((1, 4)) || !has_null_group {
        anyhow::bail!("group_by computed the wrong values: {body}");
    }

    let (_, metrics) = send("GET", "/metrics".into(), None, None).await;
    let scraped = metrics.lines().filter(|l| !l.starts_with('#')).count();

    // Clean up the namespace this smoke test created.
    let (status, _) =
        send("DELETE", format!("/v1/namespaces/{ns}"), Some("e2e-token"), None).await;
    if status != StatusCode::OK {
        anyhow::bail!("namespace delete returned {status}");
    }

    Ok(format!(
        "401 on no token, v1+v2 write/query/delete OK, BM25+phrase OK, \
         group_by OK, {scraped} metrics exposed"
    ))
}
