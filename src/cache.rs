//! NVMe read cache: one preallocated file treated as a ring buffer.
//!
//! Deliberately simple, and it can be: the cache holds only IMMUTABLE objects.
//! WAL entries, segments, centroids and cluster files are written once under a
//! unique name and never modified. The manifest is the sole mutable object in the
//! system and is never cached. So there is no invalidation problem at all — the
//! worst a stale or missing entry can do is cost one extra fetch.
//!
//! That is the whole reason a cache this crude is safe. A wrong answer is
//! impossible; only a slow one is.

use anyhow::{Context, Result};
use bytes::Bytes;
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::Path as FsPath;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Where an object sits in the ring, and what it should hash to.
#[derive(Clone, Copy)]
struct Entry {
    offset: u64,
    len: u32,
    /// FNV-1a of the bytes as written.
    ///
    /// Eviction already stops a reused region being served, but nothing else
    /// checks that what comes back is what went in. Without this a torn write or
    /// a bad sector is returned as a valid object, and the engine's guarantee
    /// that object storage is the only source of truth quietly stops being true.
    /// A mismatch costs one refetch; serving it costs a wrong answer.
    checksum: u64,
}

struct Ring {
    cursor: u64,
    map: HashMap<String, Entry>,
    /// Allocation order, so eviction follows the ring.
    order: VecDeque<String>,
}

pub struct RingCache {
    file: File,
    capacity: u64,
    ring: Mutex<Ring>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    corrupt: AtomicU64,
}

impl RingCache {
    /// Allocate a cache file of exactly `capacity` bytes.
    pub fn open(path: impl AsRef<FsPath>, capacity: u64) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())
            .with_context(|| format!("opening cache file {}", path.as_ref().display()))?;
        file.set_len(capacity)?;
        Ok(Self {
            file,
            capacity,
            ring: Mutex::new(Ring { cursor: 0, map: HashMap::new(), order: VecDeque::new() }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            corrupt: AtomicU64::new(0),
        })
    }

    pub fn get(&self, key: &str) -> Option<Bytes> {
        let entry = {
            let ring = self.ring.lock().unwrap();
            ring.map.get(key).copied()
        };
        let Some(entry) = entry else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let mut buf = vec![0u8; entry.len as usize];
        // ponytail: blocking pread on the async thread. An NVMe read of a few
        // hundred KB is tens of microseconds; move to spawn_blocking or io_uring
        // only if a profile shows the runtime stalling.
        match self.file.read_exact_at(&mut buf, entry.offset) {
            Ok(()) if fnv1a(&buf) == entry.checksum => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(Bytes::from(buf))
            }
            // Corrupt on disk. Drop the entry and report a miss: the caller
            // refetches from object storage, which is the only source of truth.
            Ok(()) => {
                self.corrupt.fetch_add(1, Ordering::Relaxed);
                self.misses.fetch_add(1, Ordering::Relaxed);
                self.ring.lock().unwrap().map.remove(key);
                None
            }
            // A short read means the region was reused.
            Err(_) => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    pub fn put(&self, key: &str, data: &[u8]) {
        let len = data.len() as u64;
        if len > self.capacity {
            return; // never worth evicting the whole cache for one object
        }
        let mut ring = self.ring.lock().unwrap();

        // Wrap to the start when this object would run off the end.
        if ring.cursor + len > self.capacity {
            ring.cursor = 0;
        }
        let (start, end) = (ring.cursor, ring.cursor + len);

        // Evict every entry whose bytes this write is about to overwrite.
        while let Some(front) = ring.order.front().cloned() {
            let Some(&Entry { offset: off, len: l, .. }) = ring.map.get(&front) else {
                ring.order.pop_front();
                continue;
            };
            if off < end && start < off + l as u64 {
                ring.map.remove(&front);
                ring.order.pop_front();
                self.evictions.fetch_add(1, Ordering::Relaxed);
            } else {
                break;
            }
        }

        if self.file.write_all_at(data, start).is_err() {
            return; // a cache write failure must never fail a query
        }
        ring.map.insert(
            key.to_string(),
            Entry { offset: start, len: data.len() as u32, checksum: fnv1a(data) },
        );
        ring.order.push_back(key.to_string());
        ring.cursor = end;
    }

    /// (hits, misses, evictions)
    pub fn stats(&self) -> (u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed),
        )
    }

    /// Entries dropped because their bytes did not match their checksum. Should
    /// be zero; anything else means the cache disk is failing.
    pub fn corrupt(&self) -> u64 {
        self.corrupt.load(Ordering::Relaxed)
    }

    pub fn len(&self) -> usize {
        self.ring.lock().unwrap().map.len()
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("fckdb-cache-test-{name}"))
    }

    #[test]
    fn stores_and_returns_bytes() {
        let c = RingCache::open(tmp("basic"), 1024).unwrap();
        assert!(c.get("a").is_none());
        c.put("a", b"hello");
        assert_eq!(c.get("a").unwrap(), Bytes::from_static(b"hello"));
        let (hits, misses, _) = c.stats();
        assert_eq!((hits, misses), (1, 1));
    }

    #[test]
    fn wraps_and_evicts_the_oldest() {
        // Capacity holds exactly 4 x 25 bytes.
        let c = RingCache::open(tmp("wrap"), 100).unwrap();
        for i in 0..4 {
            c.put(&format!("k{i}"), &vec![i as u8; 25]);
        }
        assert_eq!(c.len(), 4);
        for i in 0..4 {
            assert!(c.get(&format!("k{i}")).is_some(), "k{i} should still be cached");
        }

        // The fifth write wraps to offset 0 and must overwrite k0.
        c.put("k4", &vec![4u8; 25]);
        assert!(c.get("k0").is_none(), "k0 should have been evicted by the wrap");
        assert_eq!(c.get("k4").unwrap(), Bytes::from(vec![4u8; 25]));
        for i in 1..4 {
            assert!(c.get(&format!("k{i}")).is_some(), "k{i} evicted too early");
        }
        assert!(c.stats().2 > 0, "eviction was not counted");
    }

    #[test]
    fn evicted_entries_never_return_stale_bytes() {
        let c = RingCache::open(tmp("stale"), 60).unwrap();
        c.put("old", &vec![0xAAu8; 30]);
        c.put("mid", &vec![0xBBu8; 30]);
        // Wraps to 0, overwriting "old" with different bytes.
        c.put("new", &vec![0xCCu8; 30]);
        assert!(c.get("old").is_none(), "returned bytes for an overwritten region");
        assert_eq!(c.get("new").unwrap(), Bytes::from(vec![0xCCu8; 30]));
    }

    #[test]
    fn corrupted_bytes_are_a_miss_not_an_answer() {
        use std::os::unix::fs::FileExt;
        let path = tmp("corrupt");
        let c = RingCache::open(&path, 1024).unwrap();
        c.put("a", b"the original bytes");
        assert_eq!(c.get("a").unwrap(), Bytes::from_static(b"the original bytes"));

        // Flip a byte underneath the cache, as a bad sector or a torn write
        // would. Serving this would be a wrong answer with no error anywhere.
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all_at(b"X", 3).unwrap();
        drop(f);

        assert!(c.get("a").is_none(), "corrupted bytes were served as valid");
        assert_eq!(c.corrupt(), 1, "corruption was not counted");
        // The entry is dropped, so the caller refetches and can repopulate.
        assert!(c.get("a").is_none());
        c.put("a", b"the original bytes");
        assert_eq!(c.get("a").unwrap(), Bytes::from_static(b"the original bytes"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn oversized_object_is_skipped_not_fatal() {
        let c = RingCache::open(tmp("big"), 16).unwrap();
        c.put("huge", &vec![1u8; 64]);
        assert!(c.get("huge").is_none());
        assert_eq!(c.len(), 0);
        // Cache still usable afterwards.
        c.put("small", b"ok");
        assert_eq!(c.get("small").unwrap(), Bytes::from_static(b"ok"));
    }
}
