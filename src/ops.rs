//! Cost accounting.
//!
//! This exists because cost is the entire reason this architecture is worth
//! building. An object-storage database that cannot tell you what it spends has
//! given up its only advantage — and the trap is specific: storage is trivially
//! cheap here while REQUESTS are not, so the bill is driven by a number most
//! databases never surface.
//!
//! Measured on this codebase: per-write CAS cost 16.9 class-A operations per
//! document; group commit brought that to 0.002. Same durability, same
//! correctness, 8000x apart on the invoice. Nothing in a latency graph would
//! have shown that.

use crate::store::MetricsSnapshot;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Provider list prices. Defaults are Cloudflare R2.
///
/// ponytail: hardcoded defaults, overridable by env. Verify against the current
/// price sheet before quoting these to anyone — they move, and egress terms
/// differ sharply between providers (R2 is free, S3 is not).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Pricing {
    pub storage_gb_month: f64,
    /// Writes, lists, copies.
    pub class_a_per_million: f64,
    /// Reads.
    pub class_b_per_million: f64,
    pub egress_gb: f64,
}

impl Default for Pricing {
    fn default() -> Self {
        // Cloudflare R2, standard storage.
        Self {
            storage_gb_month: 0.015,
            class_a_per_million: 4.50,
            class_b_per_million: 0.36,
            egress_gb: 0.0,
        }
    }
}

impl Pricing {
    pub fn from_env() -> Self {
        let f = |k: &str, d: f64| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let d = Self::default();
        Self {
            storage_gb_month: f("FCKDB_PRICE_STORAGE_GB_MONTH", d.storage_gb_month),
            class_a_per_million: f("FCKDB_PRICE_CLASS_A_PER_M", d.class_a_per_million),
            class_b_per_million: f("FCKDB_PRICE_CLASS_B_PER_M", d.class_b_per_million),
            egress_gb: f("FCKDB_PRICE_EGRESS_GB", d.egress_gb),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostEstimate {
    pub window_secs: f64,
    pub class_a_ops: usize,
    pub class_b_ops: usize,
    pub bytes_stored: u64,
    /// What the observed request rate extrapolates to over 30 days.
    pub monthly_class_a_usd: f64,
    pub monthly_class_b_usd: f64,
    pub monthly_storage_usd: f64,
    pub monthly_total_usd: f64,
    /// The number that actually decides the architecture. Class-A operations per
    /// document written: above ~1 the commit path is broken, below ~0.01 group
    /// commit is doing its job.
    pub class_a_per_write: f64,
}

pub fn estimate(
    m: &MetricsSnapshot,
    bytes_stored: u64,
    window: Duration,
    p: &Pricing,
) -> CostEstimate {
    // PUT, LIST, COPY and DELETE bill as class A; GET as class B. Providers
    // disagree on DELETE (R2 bills it as class A), so counting it here is the
    // conservative reading.
    let class_a = m.puts + m.lists + m.deletes;
    let class_b = m.gets;

    let secs = window.as_secs_f64().max(0.001);
    let month = 30.0 * 24.0 * 3600.0;
    let scale = month / secs;

    let a_usd = (class_a as f64 * scale / 1_000_000.0) * p.class_a_per_million;
    let b_usd = (class_b as f64 * scale / 1_000_000.0) * p.class_b_per_million;
    let s_usd = (bytes_stored as f64 / 1e9) * p.storage_gb_month;

    CostEstimate {
        window_secs: secs,
        class_a_ops: class_a,
        class_b_ops: class_b,
        bytes_stored,
        monthly_class_a_usd: a_usd,
        monthly_class_b_usd: b_usd,
        monthly_storage_usd: s_usd,
        monthly_total_usd: a_usd + b_usd + s_usd,
        class_a_per_write: if m.writes == 0 { 0.0 } else { class_a as f64 / m.writes as f64 },
    }
}

/// Prometheus text exposition. Hand-rolled because the format is four lines of
/// rules and a client library would be a dependency for string formatting.
pub fn prometheus(m: &MetricsSnapshot, cost: &CostEstimate, namespaces: usize) -> String {
    let mut out = String::new();
    let mut g = |name: &str, help: &str, kind: &str, value: String| {
        out.push_str(&format!("# HELP fckdb_{name} {help}\n# TYPE fckdb_{name} {kind}\nfckdb_{name} {value}\n"));
    };

    g("object_gets_total", "Object storage GET requests (class B).", "counter", m.gets.to_string());
    g("object_puts_total", "Object storage PUT requests (class A).", "counter", m.puts.to_string());
    g("object_deletes_total", "Object storage DELETE requests.", "counter", m.deletes.to_string());
    g("object_lists_total", "Object storage LIST requests (class A).", "counter", m.lists.to_string());
    g("bytes_read_total", "Bytes read from object storage.", "counter", m.bytes_get.to_string());
    g("bytes_written_total", "Bytes written to object storage.", "counter", m.bytes_put.to_string());
    g("cas_conflicts_total", "Lost compare-and-swap races on the manifest.", "counter", m.cas_conflicts.to_string());
    g("queries_total", "Queries served.", "counter", m.queries.to_string());
    g("records_written_total", "Records committed.", "counter", m.writes.to_string());
    g("compactions_total", "Compactions run.", "counter", m.compactions.to_string());
    g("backpressure_rejects_total", "Writes refused because the unindexed tail was too large.", "counter", m.backpressure_rejects.to_string());
    g("namespaces", "Namespaces currently resident.", "gauge", namespaces.to_string());
    g("bytes_stored", "Bytes stored across resident namespaces.", "gauge", cost.bytes_stored.to_string());
    g("class_a_per_write", "Class-A operations per record written. The commit protocol's health in one number.", "gauge", format!("{:.6}", cost.class_a_per_write));
    g("estimated_monthly_usd", "Extrapolated monthly spend at the observed request rate.", "gauge", format!("{:.4}", cost.monthly_total_usd));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(puts: usize, gets: usize, writes: usize) -> MetricsSnapshot {
        MetricsSnapshot { puts, gets, writes, ..Default::default() }
    }

    #[test]
    fn class_a_per_write_exposes_the_commit_protocol() {
        // The per-write CAS path we measured: 270 PUTs for 16 documents.
        let bad = estimate(&snap(270, 133, 16), 0, Duration::from_secs(13), &Pricing::default());
        assert!(bad.class_a_per_write > 16.0, "got {}", bad.class_a_per_write);

        // Group commit: 4 PUTs for the same 16 documents.
        let good = estimate(&snap(4, 3, 16), 0, Duration::from_secs(2), &Pricing::default());
        assert!(good.class_a_per_write < 0.3, "got {}", good.class_a_per_write);
        assert!(bad.class_a_per_write / good.class_a_per_write > 50.0);
    }

    #[test]
    fn extrapolates_to_a_month() {
        // 1 PUT/sec for a month at $4.50/million.
        let e = estimate(&snap(10, 0, 10), 0, Duration::from_secs(10), &Pricing::default());
        let expected = 2_592_000.0 / 1e6 * 4.50;
        assert!((e.monthly_class_a_usd - expected).abs() < 0.01, "got {}", e.monthly_class_a_usd);
    }

    #[test]
    fn storage_is_priced_per_gb_not_extrapolated() {
        let e = estimate(&snap(0, 0, 0), 10_000_000_000, Duration::from_secs(1), &Pricing::default());
        assert!((e.monthly_storage_usd - 0.15).abs() < 1e-9, "got {}", e.monthly_storage_usd);
    }

    #[test]
    fn zero_window_and_zero_writes_do_not_produce_nan() {
        let e = estimate(&snap(0, 0, 0), 0, Duration::ZERO, &Pricing::default());
        assert!(e.monthly_total_usd.is_finite());
        assert_eq!(e.class_a_per_write, 0.0);
    }

    #[test]
    fn prometheus_output_is_well_formed() {
        let e = estimate(&snap(4, 3, 16), 1024, Duration::from_secs(2), &Pricing::default());
        let text = prometheus(&snap(4, 3, 16), &e, 2);
        for name in ["fckdb_object_puts_total", "fckdb_class_a_per_write", "fckdb_namespaces"] {
            assert!(text.contains(&format!("# TYPE {name} ")), "missing TYPE for {name}");
            assert!(text.lines().any(|l| l.starts_with(&format!("{name} "))), "missing value for {name}");
        }
        // Every non-comment line must be exactly "name value".
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            assert_eq!(line.split(' ').count(), 2, "malformed sample: {line}");
        }
    }
}
