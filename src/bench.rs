//! Benchmark harness. Emits markdown tables from measured runs.
//!
//! Every latency figure is a p50 and p90 over repeated samples, never a single
//! timing: object storage latency has a long tail, and one sample of a cold
//! query says more about that request's luck than about the system.
//!
//! Cold measurements use a FRESH namespace handle each time. Reusing one would
//! measure the manifest snapshot and the read cache instead of the cold path,
//! which is the number people actually care about.

use crate::doc::{Doc, Filter, Id, Record};
use crate::store::{GroupCommit, Namespace, WriteConfig};
use crate::wire::{Consistency, QueryRequest};
use anyhow::Result;
use object_store::ObjectStore;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_list(key: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(key) {
        Ok(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        Err(_) => default.to_vec(),
    }
}

/// Deterministic clustered vectors, as elsewhere: uniform random data makes any
/// IVF index look bad and predicts nothing about production recall.
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
            Doc::new(i as u64, v)
                .with_attr("tier", if i % 20 == 0 { "rare" } else { "common" })
                .with_attr("rank", i as u64)
                .with_attr(
                    "body",
                    if i % 3 == 0 { "the quick brown fox jumps" } else { "a lazy sleeping dog" },
                )
        })
        .collect()
}

struct Stats {
    p50: Duration,
    p90: Duration,
}

impl Stats {
    fn of(mut samples: Vec<Duration>) -> Self {
        samples.sort();
        let at = |q: f64| samples[((samples.len() as f64 - 1.0) * q).round() as usize];
        Self { p50: at(0.5), p90: at(0.9) }
    }
}

fn ms(d: Duration) -> String {
    let micros = d.as_micros();
    if micros < 1000 { format!("{micros}µs") } else { format!("{:.0}ms", d.as_secs_f64() * 1000.0) }
}

/// Time an async closure `reps` times.
macro_rules! sample {
    ($reps:expr, $body:expr) => {{
        let mut out = Vec::with_capacity($reps);
        for _ in 0..$reps {
            let t = Instant::now();
            let _ = $body;
            out.push(t.elapsed());
        }
        Stats::of(out)
    }};
}

pub async fn run(store: Arc<dyn ObjectStore>, backend: &str) -> Result<()> {
    let dim = env_usize("FCKDB_BENCH_DIM", 128);
    let reps = env_usize("FCKDB_BENCH_REPS", 9);
    let sizes = env_list("FCKDB_BENCH_SIZES", &[500, 2000, 8000]);
    let top_k = 10;
    let run_id = uuid::Uuid::new_v4();

    println!("<!-- backend: {backend} -->");
    println!("<!-- dim={dim} reps={reps} sizes={sizes:?} -->\n");

    // ---------------------------------------------------------------- ingest
    println!("### Ingest\n");
    println!("| docs | wall | docs/sec | commits | class-A PUTs | PUTs/doc |");
    println!("|---:|---:|---:|---:|---:|---:|");

    let mut prepared = Vec::new();
    for &n in &sizes {
        let docs = synth(n, dim, 20);
        let ns = Arc::new(Namespace::new(store.clone(), format!("bench/{run_id}-n{n}")));
        let commit = Arc::new(GroupCommit::new(ns.clone()));

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
        let wall = t.elapsed();
        let (batches, attempts) = commit.stats();
        let puts = attempts * 2;
        println!(
            "| {n} | {} | {:.0} | {batches} | {puts} | {:.4} |",
            ms(wall),
            n as f64 / wall.as_secs_f64(),
            puts as f64 / n as f64
        );
        prepared.push((n, docs, ns));
    }
    println!();

    // ---------------------------------------------------------------- compaction
    println!("### Compaction and index build\n");
    println!("| docs | wall | clusters | index bytes | bytes/doc |");
    println!("|---:|---:|---:|---:|---:|");
    // Captured here, before any querying, so the cost table below reflects
    // ingest and compaction only rather than every benchmark that follows.
    let mut write_cost = Vec::new();
    for (n, _, ns) in &prepared {
        let c = ns.compact(true).await?;
        let (m, _) = ns.load().await?;
        let bytes = m.index_bytes();
        println!(
            "| {n} | {} | {} | {} KiB | {:.0} B |",
            ms(Duration::from_millis(c.took_ms)),
            c.clusters,
            bytes / 1024,
            bytes as f64 / *n as f64
        );
        let snap = ns.metrics.snapshot();
        write_cost.push((*n, snap, m.total_bytes()));
    }
    println!();

    // ---------------------------------------------------------------- crossover
    println!("### Query latency: where the index starts paying\n");
    println!("| docs | brute p50 | cold indexed p50 | cold p90 | warm p50 | eventual p50 | index speedup | GETs cold/warm |");
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|");

    let mut warm_holder = Vec::new();
    for (n, docs, ns) in &prepared {
        let q = docs[n / 2].vector.clone();
        let req = || QueryRequest::new(q.clone()).top_k(top_k).nprobe(8);

        let brute = sample!(reps, ns.query_brute(&q, top_k, None).await?);

        // A fresh handle per sample: reusing one would measure the snapshot and
        // the read cache rather than the cold path.
        let cold = {
            let mut out = Vec::with_capacity(reps);
            for _ in 0..reps {
                let fresh = Namespace::new(store.clone(), ns.prefix.clone());
                let t = Instant::now();
                let _ = fresh.query(&req()).await?;
                out.push(t.elapsed());
            }
            Stats::of(out)
        };
        let cold_gets = {
            let fresh = Namespace::new(store.clone(), ns.prefix.clone());
            fresh.query(&req()).await?.object_gets
        };

        let cache_path = std::env::temp_dir().join(format!("fckdb-bench-{run_id}-{n}"));
        let cache = Arc::new(crate::cache::RingCache::open(&cache_path, 512 << 20)?);
        let warm_ns = Namespace::new(store.clone(), ns.prefix.clone()).with_cache(cache.clone());
        warm_ns.warm().await?;
        let warm = sample!(reps, warm_ns.query(&req()).await?);
        let warm_gets = warm_ns.query(&req()).await?.object_gets;

        let eventual = sample!(
            reps,
            warm_ns
                .query(&req().consistency(Consistency::Eventual { max_age_ms: 60_000 }))
                .await?
        );

        println!(
            "| {n} | {} | {} | {} | {} | {} | {:.1}x | {cold_gets}/{warm_gets} |",
            ms(brute.p50),
            ms(cold.p50),
            ms(cold.p90),
            ms(warm.p50),
            ms(eventual.p50),
            brute.p50.as_secs_f64() / cold.p50.as_secs_f64()
        );
        warm_holder.push((*n, cache_path));
    }
    println!();

    // ---------------------------------------------------------------- recall
    let (biggest, docs, ns) = prepared.last().unwrap();
    println!("### Recall against nprobe, at {biggest} docs\n");
    println!("| nprobe | recall@{top_k} | p50 | GETs |");
    println!("|---:|---:|---:|---:|");
    for nprobe in [1usize, 2, 4, 8, 16, 32] {
        let mut total = 0.0;
        let queries = 20.min(*biggest);
        let step = (biggest / queries).max(1);
        for d in docs.iter().step_by(step).take(queries) {
            let exact = ns.query_brute(&d.vector, top_k, None).await?;
            let got = ns
                .query(&QueryRequest::new(d.vector.clone()).top_k(top_k).nprobe(nprobe))
                .await?;
            total += crate::index::recall(&exact, &got.hits);
        }
        let q = docs[biggest / 2].vector.clone();
        let lat = sample!(
            reps,
            ns.query(&QueryRequest::new(q.clone()).top_k(top_k).nprobe(nprobe)).await?
        );
        let gets = ns
            .query(&QueryRequest::new(q).top_k(top_k).nprobe(nprobe))
            .await?
            .object_gets;
        println!(
            "| {nprobe} | {:.1}% | {} | {gets} |",
            total / queries as f32 * 100.0,
            ms(lat.p50)
        );
    }
    println!();

    // ---------------------------------------------------------------- filters
    println!("### Filtered search: the two paths\n");
    println!("| filter | selectivity | path | p50 | recall vs exact |");
    println!("|---|---:|---|---:|---:|");
    let q = docs[biggest / 2].vector.clone();
    for (label, filter, expected) in [
        ("tier = rare", Filter::eq("tier", "rare"), 0.05),
        ("tier = common", Filter::eq("tier", "common"), 0.95),
    ] {
        let req =
            || QueryRequest::new(q.clone()).top_k(top_k).nprobe(1).filter(filter.clone());
        let res = ns.query(&req()).await?;
        let exact = ns.query_brute(&q, top_k, Some(&filter)).await?;
        let lat = sample!(reps, ns.query(&req()).await?);
        println!(
            "| `{label}` | {:.0}% | {} | {} | {:.0}% |",
            expected * 100.0,
            if res.prefiltered { "exact prefilter" } else { "cluster probe" },
            ms(lat.p50),
            crate::index::recall(&exact, &res.hits) * 100.0
        );
    }
    println!();

    // ---------------------------------------------------------------- modalities
    println!("### Ranking modalities, at {biggest} docs\n");
    println!("| modality | p50 | p90 | GETs |");
    println!("|---|---:|---:|---:|");
    let modalities: Vec<(&str, QueryRequest)> = vec![
        ("vector ANN (nprobe 8)", QueryRequest::new(q.clone()).top_k(top_k).nprobe(8)),
        ("order by attribute", QueryRequest::new(vec![]).top_k(top_k).order_by("rank", true)),
        (
            "aggregate Count+Sum",
            QueryRequest::new(vec![]).aggregate(std::collections::BTreeMap::from([
                ("n".to_string(), serde_json::from_str(r#"["Count"]"#)?),
                ("s".to_string(), serde_json::from_str(r#"["Sum","rank"]"#)?),
            ])),
        ),
        (
            "group by tier",
            QueryRequest::new(vec![])
                .aggregate(std::collections::BTreeMap::from([(
                    "n".to_string(),
                    serde_json::from_str(r#"["Count"]"#)?,
                )]))
                .group_by(vec![crate::aggregate::GroupKey::Attribute("tier".into())]),
        ),
    ];
    for (label, req) in modalities {
        let lat = sample!(reps, ns.query(&req).await?);
        let gets = ns.query(&req).await?.object_gets;
        println!("| {label} | {} | {} | {gets} |", ms(lat.p50), ms(lat.p90));
    }
    // BM25 needs its own namespace with the attribute enabled.
    {
        let fts = Arc::new(Namespace::new(store.clone(), format!("bench/{run_id}-fts")));
        let cfg = WriteConfig {
            declared_fts: crate::fts::FtsSchema::from([(
                "body".to_string(),
                crate::fts::FtsConfig::default(),
            )]),
            ..Default::default()
        };
        fts.commit_records(
            &docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>(),
            &cfg,
        )
        .await?;
        fts.compact(true).await?;
        let req = QueryRequest::new(vec![]).text("body", "quick fox").top_k(top_k);
        let lat = sample!(reps, fts.query(&req).await?);
        let r = fts.query(&req).await?;
        println!("| BM25 full-text | {} | {} | {} |", ms(lat.p50), ms(lat.p90), r.object_gets);
        fts.destroy().await?;
    }
    println!();

    // ---------------------------------------------------------------- sharding
    println!("### Sharding, at {biggest} docs\n");
    println!("| shards | compact | query p50 | query p90 | GETs | agrees with 1 shard |");
    println!("|---:|---:|---:|---:|---:|---:|");
    let baseline_ids: Vec<Id> = ns
        .query(&QueryRequest::new(q.clone()).top_k(top_k).nprobe(10_000))
        .await?
        .hits
        .iter()
        .map(|h| h.id.clone())
        .collect();
    for shards in [1usize, 2, 4, 8] {
        let sh = Arc::new(Namespace::new(
            store.clone(),
            format!("bench/{run_id}-s{shards}"),
        ));
        sh.commit_records(
            &docs.iter().cloned().map(Record::Upsert).collect::<Vec<_>>(),
            &WriteConfig { num_shards: Some(shards), ..Default::default() },
        )
        .await?;
        let t = Instant::now();
        sh.compact(true).await?;
        let compact_wall = t.elapsed();

        let req = || QueryRequest::new(q.clone()).top_k(top_k).nprobe(8);
        let lat = sample!(reps, sh.query(&req()).await?);
        let res = sh.query(&req()).await?;
        let ids: Vec<Id> = res.hits.iter().map(|h| h.id.clone()).collect();
        // Compared at high nprobe so both are exhaustive and must agree exactly.
        let exhaustive: Vec<Id> = sh
            .query(&QueryRequest::new(q.clone()).top_k(top_k).nprobe(10_000))
            .await?
            .hits
            .iter()
            .map(|h| h.id.clone())
            .collect();
        let _ = ids;
        println!(
            "| {shards} | {} | {} | {} | {} | {} |",
            ms(compact_wall),
            ms(lat.p50),
            ms(lat.p90),
            res.object_gets,
            if exhaustive == baseline_ids { "yes" } else { "NO" }
        );
        sh.destroy().await?;
    }
    println!();

    // ---------------------------------------------------------------- cost
    // Ingest and compaction only. A monthly figure from a few seconds of
    // benchmarking would be meaningless, so this reports the cost of writing a
    // fixed amount of data instead — which is comparable across backends and
    // does not depend on how long the benchmark happened to run.
    println!("### Cost of writing, ingest and compaction only\n");
    println!("| docs | GET | class-A ops | class-A/doc | stored | $ per 1M docs written |");
    println!("|---:|---:|---:|---:|---:|---:|");
    let pricing = crate::ops::Pricing::from_env();
    for (n, snap, bytes) in &write_cost {
        let class_a = snap.puts + snap.lists + snap.deletes;
        let per_doc = class_a as f64 / *n as f64;
        let per_million = per_doc * 1_000_000.0 / 1_000_000.0 * pricing.class_a_per_million;
        println!(
            "| {n} | {} | {class_a} | {per_doc:.4} | {} KiB | ${per_million:.2} |",
            snap.gets,
            bytes / 1024
        );
    }
    println!();

    // ---------------------------------------------------------------- teardown
    let mut removed = 0usize;
    for (_, _, ns) in &prepared {
        removed += ns.destroy().await?;
    }
    for (_, path) in warm_holder {
        let _ = std::fs::remove_file(path);
    }
    println!("<!-- cleanup: {removed} objects removed -->");
    Ok(())
}
