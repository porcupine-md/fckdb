# Turbopuffer — Verifikasi Arsitektur dari Sumber Primer

Sumber: dokumentasi resmi turbopuffer (`/docs/architecture`, `/docs/concepts`,
`/docs/guarantees`, `/docs/tradeoffs`, `/docs/limits`, `/docs/sharding`,
`/docs/pinning`), blog resmi, dan catatan implementasi ulang pihak ketiga.
Tanggal verifikasi: 2026-08-27.

Dokumen ini **mengoreksi** rangkuman wawancara, bukan mengulanginya.

---

## 1. Koreksi terhadap rangkuman wawancara

### ❌ "CAS di S3 (Desember 2023)" — SALAH

S3 tidak punya CAS di Desember 2023. Timeline sebenarnya:

| Tanggal | Fitur |
|---|---|
| Des 2020 | S3 strong read-after-write consistency |
| Agu 2024 | S3 conditional write `If-None-Match` (create-if-absent) |
| Nov 2024 | S3 conditional write `If-Match` (**CAS sesungguhnya**) |
| Okt 2025 | Conditional write diperluas ke operasi `CopyObject` |

Bukti langsung dari Simon sendiri (Maret 2024, di X):

> "This is why @turbopuffer started on GCS. Every object store has
> preconditions (R2, Azure, Tigris, GCS), except S3."

**Artinya: turbopuffer dibangun di atas GCS justru KARENA S3 belum punya CAS.**
GCS, Azure Blob, R2, dan Tigris sudah punya precondition ETag sejak lama.
Ini bukan detail sepele — ini mengubah cerita "tiga primitif yang baru
tersedia" menjadi "dua primitif, dan satu vendor yang sudah punya."

### ⚠️ Tiga tingkatan cache "10ms / 100ms / 500-1000ms" — bukan angka resmi

Angka yang benar-benar dipublikasikan turbopuffer:

| Kondisi | Angka resmi |
|---|---|
| Query panas (cached), 1M dokumen | **p50 = 14 ms** |
| Query dingin (first query), 1M dokumen | **p50 = 874 ms** |
| Floor konsistensi kuat | **~10 ms** (biaya cek metadata object storage) |
| Satu roundtrip ke object storage | **~100 ms** |
| Cold query total | **3–4 roundtrip, sering ~400 ms** |
| Write, payload 500 kB | **p50 = 165 ms** |
| Metadata GET S3 | p50 = 10 ms, p90 = 17 ms |
| Metadata GET GCS | p50 = 12–18 ms, p90 = 15–25 ms |

Yang "10 ms" itu bukan tier DRAM. Itu **floor konsistensi**: setiap query
konsisten wajib satu kali cek ke object storage untuk memastikan tidak ada
write baru. Tidak bisa ditawar kecuali kamu pindah ke eventual consistency.
Tidak ada tier "DRAM vs NVMe" yang dipublikasikan terpisah.

### ⚠️ "V1 2 roundtrip, V2 3 roundtrip"

Dokumen resmi menyebut cold query butuh **3–4 roundtrip**, masing-masing
~100 ms, total sering ~400 ms. Pembagiannya:

```
Roundtrip 1        │ Roundtrip 2          │ Roundtrip 3
───────────────────┼──────────────────────┼──────────────
Metadata storage   │ Filter index         │ Clusters
engine             │ Centroid index       │ (posting list)
                   │ Unindexed WAL        │
```

Roundtrip 2 menggabungkan tiga hal dalam satu request — inilah optimasinya.
Dan ada **query planner** yang memutuskan trade-off: tambah roundtrip, atau
ambil lebih banyak data di roundtrip yang sudah ada. Roundtrip 1 (metadata)
yang di-cache hampir permanen, sehingga praktis jadi 2 roundtrip.

### ❌ "Node melakukan compaction sendiri secara opportunistic"

Salah. Ada **pemisahan peran biner** yang eksplisit:

- `./tpuf query` — punya Memory Cache + NVMe Cache
- `./tpuf indexer` — node terpisah, khusus indexing/compaction
- **Indexing Queue yang hidup DI object storage**, bukan di message broker

Jadi bukan "semua node boleh compact dan bertarung lewat CAS". Ada queue
persisten di S3/GCS, dan indexer node menariknya.

---

## 2. Bagian yang rangkuman video lewatkan sepenuhnya (dan ini yang paling penting)

Rangkuman itu menggambarkan write path sebagai "PUT file, lalu CAS metadata".
Yang sebenarnya jauh lebih spesifik dan jauh lebih menentukan:

```
                                              mem buffer
                                              ┌──────┐
            UPSERT/PATCH/DELETE               │░░░░░░│
            ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ─▶ │░░░░░░│
                                              └──┬───┘
WAL                                              │ group commit (≤ 1/detik)
s3://tpuf/{namespace_id}/wal                     ▼
╔═════════════════════════════════════════════════════════╗
║ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            ║
║ │██████│ │██████│ │▞▞▞▞▞▞│ │▞▞▞▞▞▞│ │▞▞▞▞▞▞│            ║
║ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘            ║
║  01.bin   02.bin ▲ 03.bin   04.bin   05.bin ▲ (06.bin)  ║
╚══════════════════│══════════════════════════│═══════════╝
                   │                          │
            index cursor                 CAS commit point

█ indexed + committed   ▞ committed, unindexed   ░ written, not committed
```

Tiga hal krusial di gambar ini:

1. **Group commit ≤ 1 WAL entry per detik per namespace.** Semua write
   konkuren ke namespace yang sama digabung ke satu WAL entry. Ini bukan
   bug, ini desain: request S3 mahal, jadi jangan pernah satu write = satu PUT.

2. **Dua pointer yang berbeda: CAS commit point dan index cursor.**
   Commit point maju lewat CAS — itu yang menentukan durabilitas.
   Index cursor maju secara asinkron — itu yang menentukan kecepatan.
   Data di antara keduanya (`▞`) **sudah durable tapi belum terindeks**, dan
   tetap bisa dicari lewat **exhaustive scan**. Inilah cara mereka memberi
   strong consistency tanpa menunggu indexing.

3. **`░` = sudah tertulis ke S3 tapi belum committed.** File boleh ada di S3
   tanpa dianggap valid. Kebenaran ditentukan CAS, bukan keberadaan file.
   Ini yang membuat retry dan zombie writer aman.

---

## 3. Jebakan yang tidak ada di rangkuman mana pun

### 🔴 Cliff 128 MiB — write jadi INVISIBLE

Dari `/docs/guarantees`, kutipan langsung:

> "once a namespace has more than 128MiB of outstanding writes, further writes
> are not visible until they are indexed and loaded into cache. For small
> namespaces indexing and cache warming takes tens of seconds; for large
> namespaces this process can take **tens of minutes**."

Artinya: kalau kamu backfill besar, datamu bisa **tidak terlihat selama
puluhan menit** — padahal API sudah balas sukses. Ini backpressure paling
keras di arsitektur ini, dan turbopuffer sendiri tidak lolos darinya.
Kalau kamu bangun sendiri, kamu akan menabrak dinding yang sama.

### 🔴 Biaya request, bukan biaya storage

Catatan dari orang yang membangun kloningnya (VecPuff):

> "Storage is $0.023/GB, but 1M GETs/day is $12/month. My first benchmarks
> were doing **50+ GETs per query**. That math doesn't work at scale.
> Batching isn't optional."

Hitung sendiri: 100 QPS × 4 GET/query × 86400 detik ≈ 34M GET/hari ≈
**~$415/bulan hanya untuk request**, sebelum storage sepeser pun. Ini sebabnya
`hint_cachewarm` bukan cuma fitur latensi — itu alat kontrol biaya. Skema
tagihannya sendiri mengakui ini: *"if it's already in cache it's free, and if
not we charge you for one query."*

### 🟡 Sharding = fanout, dan fanout = tail latency

- Max 1 TB / shard, 500M dokumen / shard, 256 shard, efektif 256 TB
- Semua shard **berbagi satu WAL** → write commit sekali, atomik ke semua shard
- **Setiap query menunggu SEMUA shard.** Satu shard lambat = seluruh query lambat
- `num_shards` **tidak bisa diubah** — harus copy ke namespace baru
- Query eventual-consistent bisa mengembalikan hasil dari **titik waktu berbeda
  per shard**

### 🟡 Serialisasi harus zero-copy

Dari VecPuff: saat menarik posting list 10 MB dari S3, deserialisasi JSON jadi
bottleneck. Mereka pindah ke `rkyv` supaya data siap dipakai SIMD begitu paket
jaringan sampai. JSON/Protobuf akan membunuhmu di layer ini.

---

## 4. Angka batas resmi (untuk kalibrasi ekspektasi)

| Limit | Nilai |
|---|---|
| Write throughput per namespace | 10k writes/s @ 32 MB/s |
| Write throughput global (terbukti) | 10M+ writes/s @ 32 GB/s |
| WAL entry per namespace | **1 per detik** |
| Max dokumen per namespace | 128B @ 256TB (terbukti 100B @ 200TB) |
| Max dokumen per shard | 500M @ 1TB |
| Max vector column per namespace | 8 |
| Max dimensi dense vector | 10.752 |
| Max upsert batch | 512 MB |
| Jumlah namespace | Unlimited (terbukti 250M+) |
| Max waktu idle di cache | "hours" |
| Eventual consistency: unindexed scan | ≤ 128 MiB |
| Eventual consistency: staleness worst case | ~1 jam |
| Eventual consistency: tetap konsisten | >99.8% query |

**Pinning** (reserved compute + NVMe untuk satu namespace): tagihan berubah
dari per-query jadi GB-hours (`size × replicas × hours`), floor 128 GB dan
10 menit. Break-even vs per-query sekitar **10 QPS**. Satu replica melayani
100–1000 QPS. Contoh dari kalkulator resmi: namespace 256 GB @ 50 QPS →
multi-tenant $7.498/bln vs pinned $2.573/bln (2.9x lebih murah).

Perhatikan: begitu kamu butuh latensi predictable, kamu **pinning**, dan
begitu kamu pinning, kamu kembali membayar per-kapasitas — bukan per-query
lagi. Model "serverless murni" itu punya batas.

---

## 5. Algoritma index: SPFresh, dan kenapa HNSW adalah jebakan

Turbopuffer memakai **SPFresh** ([SOSP '23](https://dl.acm.org/doi/10.1145/3600006.3613166)) —
ANN berbasis centroid dengan update cluster inkremental (protokol LIRE).
Alasan resminya:

> "A centroid-based index works well for object storage as it minimizes
> roundtrips **and write-amplification**, compared to graph-based indexes
> like HNSW or DiskANN."

Dua kata kunci: **roundtrip** dan **write amplification**.

- HNSW butuh seluruh graph di RAM → membatalkan seluruh alasan pakai S3
- HNSW punya write amplification tinggi → setiap update menyentuh banyak node
  graph → banyak PUT → tagihan request meledak
- Centroid: query = 1 lookup centroid kecil + N fetch posting list paralel.
  Update = sentuh 1 posting list.

Selain vektor, mereka juga punya **inverted index BM25** yang dioptimasi untuk
object storage, dan **exact index** (bukan approximate) untuk metadata filtering.

---

## 6. Yang berubah sejak wawancara itu — ini yang menentukan keputusan build/buy

Rangkuman video menggambarkan lanskap 2023–2024. Sekarang Agustus 2026, dan
tiga hal sudah berubah drastis:

### S3 CAS sudah universal (Nov 2024)
Moat "kami bisa karena kami di GCS" sudah hilang. Semua object store punya
precondition sekarang. Primitifnya jadi komoditas.

### Amazon S3 Vectors — AWS mengkomoditisasi layer ini langsung
- Preview Juli 2025, **GA Desember 2025** (40x skala preview),
  ekspansi ke 17 region tambahan Maret 2026
- Storage **$0.06/GB** (turbopuffer ~$0.33/GB) → ~5.5x lebih murah di storage
- Cold query sub-1 detik, query sering ~100 ms
- Contoh workload 400M vektor + 10M query/bulan ≈ **$1.217/bulan**

Ini persis kategori yang mau kamu bangun, dijual langsung oleh pemilik S3.

### Building block open source sudah ada
- **SlateDB** — LSM embedded di atas object storage. Ini layer storage engine
  yang tadinya harus kamu tulis sendiri.
- **LanceDB / format Lance** — kolumnar + vector index + FTS + SQL di object
  storage, open source. Realistis ini ~70% dari turbopuffer.
- **mini-lsm** (skyzh) — kursus membangun LSM storage engine dari nol.
- **VecPuff** — kloning turbopuffer open source oleh satu orang, lengkap
  dengan catatan semua kesalahannya.

---

## 7. Verdict

**Bagian sulitnya bukan S3.** Plumbing object storage (WAL, CAS, manifest,
ring buffer cache) itu bagian yang gampang — buktinya satu orang bisa
menyelesaikannya dalam hitungan minggu. Yang sulit dan yang sebenarnya jadi
produk:

1. ANN index kelas SPFresh yang menjaga recall sambil cluster terus
   split/merge di bawah update kontinu
2. Cache scheduler multi-tenant yang adil (ribuan namespace, satu pool NVMe)
3. Query planner yang menawar antara "tambah roundtrip" vs "ambil lebih banyak"
4. Disiplin biaya request di setiap jalur kode

Pertanyaan yang harus dijawab sebelum menulis satu baris kode:

> **Apa unit partisi kamu, dan apakah kamu sanggup menolak transaksi
> lintas-partisi serta menerima ≤1 commit/detik per partisi?**

Turbopuffer menjawab: unit = namespace, dan ya, mereka sanggup menolak
keduanya. Itulah satu-satunya alasan mereka tidak butuh konsensus. Kalau
jawabanmu "tidak sanggup", seluruh arsitektur ini gugur dan kamu kembali
butuh Raft — dan saat itu terjadi, keunggulan biayanya hilang.

---

## Sumber

- [turbopuffer Architecture](https://turbopuffer.com/docs/architecture)
- [turbopuffer Concepts](https://turbopuffer.com/docs/concepts)
- [turbopuffer Guarantees](https://turbopuffer.com/docs/guarantees)
- [turbopuffer Tradeoffs](https://turbopuffer.com/docs/tradeoffs)
- [turbopuffer Limits](https://turbopuffer.com/docs/limits)
- [turbopuffer Sharding](https://turbopuffer.com/docs/sharding)
- [turbopuffer Pinning](https://turbopuffer.com/docs/pinning)
- [turbopuffer: fast search on object storage (blog)](https://turbopuffer.com/blog/turbopuffer)
- [Jason Liu — TurboPuffer: Object Storage-First Vector Database Architecture](https://jxnl.co/writing/2025/09/11/turbopuffer-object-storage-first-vector-database-architecture/)
- [VecPuff — What I learned building a vector database on object storage](https://blog.karanjanthe.me/posts/vecpuff/)
- [Amazon S3 adds new functionality for conditional writes](https://aws.amazon.com/about-aws/whats-new/2024/11/amazon-s3-functionality-conditional-writes)
- [Amazon S3 conditional write for copy operations (Okt 2025)](https://aws.amazon.com/about-aws/whats-new/2025/10/amazon-s3-conditional-write-functionality-copy-operations)
- [Simon Eskildsen di X — kenapa turbopuffer mulai di GCS](https://x.com/Sirupsen/status/1767349402727772379)
- [Amazon S3 Vectors GA (Des 2025)](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-s3-vectors-generally-available/)
- [Amazon S3 Vectors ekspansi 17 region (Mar 2026)](https://aws.amazon.com/about-aws/whats-new/2026/03/s3-vectors-expands-17-regions)
- [Zilliz — Will Amazon S3 Vectors Kill Vector Databases?](https://zilliz.com/blog/will-amazon-s3-vectors-kill-vector-databases-or-save-them)
- [SPFresh (SOSP '23)](https://dl.acm.org/doi/10.1145/3600006.3613166)
- [LanceDB](https://github.com/lancedb/lancedb)
- [mini-lsm](https://skyzh.github.io/mini-lsm/)
