# HNSW native traversal plane — design

Origin: this library was designed and extracted from the Harper vector-index engine
(HarperFast/harper, branch kris/hnsw-native-plane); "the JS implementation" and issue numbers
below refer to that codebase, and the measured JS baselines come from its
`benchmarks/hnsw-scale.js`. The design moves HNSW graph storage and traversal into a native
(Rust/napi-rs) module over a memory-mapped fixed-slot file, replacing a KV-store column
family as the home of graph nodes.

## 1. Motivation — measured, not estimated

Per-visit cost decomposition at 5M nodes / ef 512 (768-d int8, `benchmarks/hnsw-scale.js`
corpus, 22.18 ms p50 / 5,107 visits):

| Component                                   | Cost    | Share of a warm visit |
| ------------------------------------------- | ------- | --------------------- |
| Total per visited node                      | 4.34 µs | 100%                  |
| int8 asymmetric cosine, 768-d, JS           | 0.43 µs | 10%                   |
| msgpackr decode of one node (VT-cache miss) | 5.57 µs | +128% when cold       |
| Neighbour iteration + visited-set ops       | 0.21 µs | 5%                    |

~85% of a warm visit is JS object bookkeeping — candidate heap, visited `Set`, property access,
allocation, GC — not distance math and not I/O. Three consequences:

1. **A native distance kernel is worth ~nothing.** Distance is 10% of the visit; a NAPI crossing
   costs 0.1–0.5 µs. The win requires the whole search loop native, over a native data layout,
   with one boundary crossing per query.
2. **The fetch path decides the ceiling.** A warm RocksDB `Get` is ~1–2 µs even called natively
   (block-cache lookup, block parse, value memcpy) — 20–40× the SIMD distance it feeds. Direct
   slot addressing (`base + id × SLOT_SIZE`) into a resident mapping is ~100–200 ns. Traversal
   over RocksDB caps at ~3–5× improvement; traversal over a fixed-slot mapping reaches the
   full ceiling.
3. **Estimated native budget: ~0.25–0.4 µs/visit** (SIMD int8 dot ~50 ns + streaming 768
   contiguous bytes ~150 ns + bitset/heap ops ~50 ns) → **~10–15× on the search path**
   (22 ms → ~1.5–2 ms at 5M/ef 512), with the JS event loop untouched.

This is also the enabling dependency for same-node index slicing (parallel slice searches need
off-loop execution) and changes cluster QPS arithmetic by the same factor.

## 2. Goals / non-goals

Goals:

- Search traversal fully native, off the JS event loop, one NAPI crossing per query.
- Graph nodes in a memory-mapped fixed-slot file — **the file is the index**: the maintained
  primary of the derived data, updated in place on every commit, not a cache of RocksDB.
- Incremental maintenance preserved: insert/update/delete keep working exactly as today from
  the application's view.
- Relaxed transactional adherence (deliberate): HNSW results are approximate by contract, and
  the existing post-load exact rescore + MVCC record lookup already filter stale/wrong
  candidates. No cross-slot atomicity.
- Node-id reuse via an in-file freelist — structurally fixes the #2182 lifetime high-water
  ef over-provisioning.
- Slicing-ready: one file per slice; native merge of per-slice top-k (C2 hook).

Non-goals (this phase):

- Binary quantization / Matryoshka truncation (benchmark-gated per the Reflex study; the format
  reserves a quantization-mode field so a binary plane is a format v2, not a redesign).
- Native insert loop (phase 3; insert logic stays in JS initially, persisting through the
  native slot-write API).
- Cross-node ANN protocol. Out of scope entirely.
- Lexical/BM25 anything.

## 3. Architecture

```
                     JS (worker threads)                    native (Rust, napi-rs)
  ┌─────────────────────────────────────────┐   ┌─────────────────────────────────────┐
  │ HierarchicalNavigableSmallWorld.ts      │   │ hnsw-plane                          │
  │  • pk→nodeId mapping   (stays RocksDB)  │   │  • mmap'd slot file (per index/slice)│
  │  • insert/update/delete logic (phase 1) ├──►│  • slot read/write API (seqlocked)   │
  │  • commit callback → slot writes        │   │  • search(query, k, ef, filter) →    │
  │  • record load + exact rescore (as-is)  │◄──┤    top-k ids, own thread pool        │
  │  • runIndexing replay from watermark    │   │  • TSFN batch filter callback        │
  └─────────────────────────────────────────┘   └─────────────────────────────────────┘
```

What stays in RocksDB: the pk→nodeId mapping (transactional with record writes — it is the
authority on which node id a record owns), records themselves, and all other indexes. What
moves to the file: node vectors, per-layer adjacency, entry point, id allocator, freelist.

## 4. File format (v1)

One file per index (per slice, once C2 lands): `<index-path>.hnsw`.

**Header (4 KB page):**

| Field                             | Type       | Notes                                                        |
| --------------------------------- | ---------- | ------------------------------------------------------------ |
| magic + format version            | u32 + u32  | rebuild required on version mismatch (accepted contract)     |
| dims, quantization mode           | u16 + u8   | 0 = int8, 2 = int16; 1 reserved for the unimplemented f32 mode |
| slot_size, layer0_cap, upper_cap  | u16 ×3     | derived from M/optimizeRouting at creation                   |
| entry_point_id, entry_point_level | u32 + u8   | atomically updated                                           |
| id_high_water                     | u64 atomic | replaces the shared Atomics BigInt64Array incrementer        |
| freelist_head                     | u64 atomic | CAS push/pop; ABA-guarded with a 32-bit tag                  |
| txn_watermark                     | u64        | last durably indexed transaction; advanced by msync cadence  |
| clean_shutdown flag               | u8         | torn-state detection on open                                 |
| invalidated latch                 | u8         | one-way (v7): watermark reads 0 on every handle, open refuses |
| write_epoch                       | u64 atomic | bumped by every node write; re-arms the read-side repair probe |

**Main region — layer-0 slots**, addressed `4096 + id × slot_size`:

| Field                           | Size (768-d int8, cap 64)           |
| ------------------------------- | ----------------------------------- |
| seq (seqlock)                   | 4 B                                 |
| flags (valid/deleted) + level   | 2 B                                 |
| scale (f32) + invMag (f32)      | 8 B                                 |
| degree                          | 2 B                                 |
| vector (dims × elem_size)       | 768 B int8 / 1,536 B int16 (padded to a 4-byte boundary) |
| neighbor ids (u32 × layer0_cap) | 256 B                               |
| **total, padded**               | **1,040 B → 1 KB-aligned 1,088 B**  |

The vector's trailing pad keeps the neighbor array 4-aligned for every `dims`, so the search
hot path reads each neighbor id as one aligned volatile `u32`. Upper-layer id lists are padded
the same way (`degree u16 + pad u16 + ids`).

**The codec is carried by the file version, not by the header byte alone.** Released readers
ignore `H_QUANT` and validate geometry only against the *rounded* slot size, which collides
between the widths — at dims 16 / cap 16 both round to 128 B. So an int8 plane writes a v8
header and an int16 plane writes v9, and `open` accepts exactly `(v8, int8)` and
`(v9, int16)`; every other pair, the reserved f32 byte included, is a descriptive refusal.
That way an older binary is turned away at the version check instead of opening an int16 file
and writing neighbor ids over its vector. `PlaneFile` carries the parsed codec and a derived
`vector_bytes`, and exposes `neighbor_offset()` / `key_offset()` as methods — every slot
offset comes from the byte length, and no call site can pass `dims` where bytes are meant.

At 100M nodes: ~109 GB (int8); int16 adds one byte per dimension per slot, +18% at 128 dims
and +73% at 1536, which is why it is a per-index choice and not the default. A binary-code v2 slot (96 B codes + ids) is ~384 B → ~38 GB.
For comparison, today's encoding averages 1,425 B/node _plus_ RocksDB overhead — so v1 is
already ~25% smaller while being fixed-offset addressable, because per-edge cached float64
distances are dropped (recomputing a distance costs ~50 ns native; storing it costs 8 B and
~40% of today's node bytes).

**Upper-layer region** (append-allocated, compacted on rebuild): only ~6% of nodes have
level > 0, and upper layers hold neighbor id lists only (vectors live in the main slot). Each
entry: `node_id, level, [degree, ids × upper_cap] × level`. Kept fully resident; a few hundred
MB at 100M nodes.

**Host keys (format v8).** Each slot ends with the host's key for the node: a `u16` length,
a pad to 4 bytes, then `key_cap` bytes (`key_cap` is a create-time header field; 0 = no keys).
A key that fits is stored inline; a longer one is copied into a **key overflow arena** after the
upper region (sparse; `key_arena_bytes_per_node` at create, default max(128, 4 × key_cap); CAS bump
allocation) and the payload holds its offset as two
aligned `u32` halves (every field read on the search path stays a naturally aligned volatile
load). The key is stored under the slot's write lock, first, so an exhausted arena leaves the
slot's previous state intact. Ranges are reserved in 64-byte classes, so a range's capacity
follows from the key length stored with it and no separate capacity word can be torn away from
the offset by a kill mid-write; a rewrite reuses the range while the key fits that class, so only
growth past it allocates. Ranges of deleted or outgrown keys, and of keys that shrank below
their class, are not reclaimed until a rebuild; a record torn by a dead writer is never reused
(the lock takeover zeroes its length). `key_cap` is at least 8, the size of the offset. Raw
mirroring without a key keeps the stored one. Searches and predicate batches return the keys with the hits, so the host
resolves a hit to its record without a lookup by node id: in Harper that lookup was one RocksDB
read per candidate, about half the per-query CPU once traversal went native. For 128-d/768-d
int8 slots at cap 128 a 40-byte inline key fits inside the existing 64 B padding (704 B / 1,344 B
slots, unchanged). A key read happens under the slot's seqlock at result time; a slot deleted or
reused since the traversal yields an empty key (host skips) or the new occupant's key (the
host's exact rescore drops it), the same relaxed contract as the mapping race it replaces. A
predicated search returns each admitted hit with the key its predicate batch carried, so the
verdict and the returned key always describe the same record.

**Degree cap decision.** Today layer-0 caps at `M<<1` then `<<2` under `optimizeRouting` = 128,
with transient overshoot to 160 before pruning; measured mean degree is ~37. Sizing slots at
cap 128 doubles the file for a tail. v1 policy: **hard prune-to-cap-64 on write** — the insert
path's in-memory candidate selection can overshoot as today, but what is written is pruned to
64 by the same routing-aware selection that currently prunes at 160→128. Transient overshoot
never touches the file. Recall impact must be measured in the validation phase (§9); the cap is
a header field, so revising it is a rebuild, not a format change.

## 5. Concurrency

- **Per-slot seqlock.** Writer: fetch_add seq to odd → write slot → fetch_add to even. Reader
  (traversal): read seq, copy the ≤1 KB slot (or read fields in place), re-check seq; retry on
  change. Retries are rare (writes touch ~40 slots per insert out of millions) and cheap.
- **No cross-slot atomicity.** An insert updates the new node's slot plus ~M neighbors'
  back-edge lists, each independently. A traversal may observe the half-linked state: an edge
  to a slot whose valid flag is not yet set → skip (HNSW tolerates missing edges); a
  just-deleted neighbor → skip via flags. Wrong-candidate leakage is filtered by the existing
  exact rescore + MVCC record load, which is why relaxed adherence is safe _here_ and not a
  general storage pattern.
- **Writers.** Multiple worker threads insert concurrently today (distinct records); the same
  holds: id allocation is one atomic fetch_add on the header, freelist pop is CAS, slot writes
  are seqlocked. Two inserts updating the same neighbor's edge list serialize on that slot's
  seqlock (a Rust-side per-slot spinlock on the odd state).
- **Batch insert.** `insert_batch` (napi `insertBatch`) is the bulk driver for those writer
  primitives: one crossing per chunk, one worker per scratch pulling records off a shared
  counter, so a hub-heavy or cold region slows one worker rather than the chunk. Record faults
  (non-finite component, unstorable key, wrong dims, an overflow key once the arena is full)
  skip that record and are reported by index; a plane fault (full, wedged) stops the dispatch
  and fails the batch once in-flight inserts finish, reporting the ids that landed — the call
  never returns with an insert still running, which is what lets a host's barrier cover
  exactly what it applied. The napi surface runs a plane's batches in order on one worker
  thread, and each worker's visited set is 4 B per allocated id, so a batch's working memory
  is `threads × 4 B × id_high_water`. Records land in thread order, so a batch-built graph is
  one of many equally valid shapes for the same input (§10: concurrency only shuffles the
  insertion permutation). Scaling in §11.
- **Id reuse & ABA.** Delete pushes the id onto the freelist; a traversal holding the old id may
  read the reused slot and score the wrong vector — acceptable under the relaxed contract
  (rescore/record-load rejects it). The freelist head itself is tag-guarded against ABA.

## 6. Durability & crash recovery

The file is `msync`'d on a cadence (default: every N seconds or M mutated slots, configurable),
**not** per commit. The header watermark records the last transaction whose index mutations are
known durable; it advances only after a completed msync barrier.

On open:

- Clean-shutdown flag set → map and serve.
- Torn state → replay records from `txn_watermark` through the existing `runIndexing` re-feed
  path (which already treats a re-fed already-indexed record as an update — the exact semantics
  needed). This anchors today's heuristic crash re-feed to a precise watermark.
- Format-version mismatch or corruption (header checksum) → full rebuild from records. Explicit
  contract: **format upgrades require reindex** (accepted).

Note the asymmetry with today: RocksDB gave the graph per-commit durability; the file gives it
bounded-lag durability with deterministic catch-up. For an approximate index whose source of
truth (records + pk→nodeId) remains fully transactional, bounded lag is the right trade — it
buys the entire performance model.

**Invalidation (a plane the host cannot delete).** Disabling a plane deletes its file; when the
unlink fails (Windows sharing violation while another process maps it) the file must not be
adopted later at its nonzero watermark, or it silently serves searches missing every mutation
made while mirroring was off. `invalidate_plane(path)` / `invalidate_file(&handle)` leave two
markers, always attempting both: in band — `PlaneFile::invalidate` sets a one-way header latch,
zeroes the watermark, and msyncs the header page alone (a whole-mapping flush cannot run inline
on a multi-GB plane, and lowering the watermark is the safe direction) — then a `<path>.stale`
sidecar, created with create-new semantics (a planted symlink is never followed) and fsync'd
together with its directory entry (the directory fsync is skipped on Windows, where `std` has no
directory handle and `FlushFileBuffers` on the marker covers its creation). The package enforces
both markers: `open` refuses a file carrying either, `create` refuses a path with a leftover
sidecar, and `watermark()` reads 0 on every handle while the latch is set — so a flush already
in flight on another handle, which still stamps the word, cannot revive the plane. In band
first: the sidecar is what a process that cannot map the file checks, the latch is what covers a
plane whose sidecar a crash lost. A temporary handle opened for the in-band mark is unmapped
and closed before the sidecar step — its own mapping would keep the file undeletable — and the
call fails only when neither marker is durable, leaving the file exactly as found.

**Backup/copy-db/reseed:** the file is node-local derived state. Backup either includes it
(consistent-enough after an msync barrier) or marks the index rebuild-on-restore. Replica
reseed = rebuild from records (C5 bulk construction makes this fast; until then, the existing
per-row path).

## 7. Search path & NAPI surface

```ts
// one crossing per query; executes on the module's own thread pool
search(sliceHandles, queryVector: Float32Array, k, ef, filter?): Promise<{ids, distances, keys, keyEnds}>
```

- Asymmetric distance as today: float query × int8 stored, cached invMag, SIMD (AVX2/VNNI on
  x86, NEON on ARM; `std::arch` intrinsics with a scalar fallback).
- Visited set: one bit per node id plus a journal of the words a sweep set, cleared per sweep
  by walking the journal (one per pool thread, reused across queries — no allocation per
  query). Per scratch: `8 × ceil(nodes / 64)` bytes of bitmap plus at most 256 KB of journal,
  so 200M nodes is ~25 MB; worst case per process is `(in-flight searches + 1 insert scratch)
  × that`, and the pool retains at most 64 idle scratches. The u32 epoch stamp it replaced was
  4 B/node materialized per scratch — 800 MB at 200M, 205 GB across a 256-thread libuv pool
  (issue #8). A sweep that sets more than 65 536 distinct words overflows the journal and the
  next clear walks the whole bitmap, amortized against the ≥ 65 536 visits that caused it. Candidate heap: fixed-capacity binary heap of (dist, id) pairs.
- Auto-ef / auto-efC read the node count from the header high-water minus freelist length —
  same semantics as today, minus the #2182 inflation (freed ids return to the pool).

**Upper-layer descent is a beam, not hill climbing** (`beam_descend`, `DESCENT_EF = 16`). The
textbook width-1 descent halts at the first upper-layer node no neighbor improves on. On a
clustered corpus that local minimum can sit in the wrong basin, and layer-0 adjacency is
intra-basin, so the layer-0 beam has no uphill edge with which to leave — the query's true
nearest neighbor is then unreachable at *any* ef, and raising ef only expands the wrong basin.
Measured on the `tests/concurrent.rs` corpus (8 000 nodes, 64-d, self-query every node at
ef 256, insertion order fixed by seed): width 1 loses 125 nodes over 200 builds, width 4 loses
20 over 200, width 8 loses 8 over 700, width 16 loses 0 over 700. Cost at 50 000 × 768-d:
visits/query +17 % to +27 %, p50 +0.06 ms flat (0.15 → 0.21 ms at ef 16, 0.21 → 0.28 at ef 64,
0.46 → 0.47 at ef 512 — the descent is a fixed cost, so it hurts most where ef is small), build
throughput -20 %. recall@10 improves below ef 128 (0.844 → 0.903 at ef 16, 0.983 → 1.000 at
ef 64) and is unchanged above.

`beam_descend` is shared by the read and write paths deliberately: insert must route through
the same graph its queries will, or nodes get their neighbors chosen from a basin searches
never reach. The width is a compile-time constant rather than a parameter because it is a
correctness floor, not a recall/latency dial — `ef` is the dial.

Three facts worth keeping when working on this.

The trap is a property of graph *shape*, not of concurrency: it reproduces single-threaded from
a fixed insertion permutation, and concurrency only shuffles that permutation. It also needs the
full corpus — no seed reproduces it at 32 dims, or at 2 000 / 4 000 nodes, so a shrunken repro
is not evidence of a fix. `descent_width_sweep` in `tests/concurrent.rs` (ignored by default) is
the harness behind the table above.

Read and write descent widths must match. Measured over 200 builds per cell: width 1 both sides
loses 125 nodes, read-only widening loses 37, **write-only widening loses 245 — worse than
either**, and both sides widened loses 0. `insert` seeds each level's `search_layer` from the
descent's landing point, so a graph wired under one routing policy and queried under another is
less navigable than one where they agree. This is also the upgrade story: an existing plane file
read by a new binary is the read-only row, improved but not repaired until its nodes are
re-inserted.

`HNSW_SWEEP_READ_EF` sets the sweep's query-side width; the build side is whatever `DESCENT_EF`
is compiled as, so the four cells are two runs per value of the constant:

```text
HNSW_SWEEP_SEEDS=200 HNSW_SWEEP_READ_EF=1  cargo test --release --test concurrent \
    descent_width_sweep -- --ignored --nocapture
HNSW_SWEEP_SEEDS=200 HNSW_SWEEP_READ_EF=16 cargo test --release --test concurrent \
    descent_width_sweep -- --ignored --nocapture
```

Do not add a per-level visit cap to the descent without re-measuring. The obvious ceiling,
`DESCENT_EF * UPPER_CAP` = 1024, is already exceeded by ordinary queries: the worst of 3 000
random queries visits 788 nodes at level 1 on a 50 000-node graph and 1 044 on a 500 000-node
one. A cap that binds silently degrades recall, which is the defect this exists to fix. What
bounds the pathological case instead is `search_layer`'s strict `d < worst`: with every distance
tied — a zero query ties them all at exactly 1.0 — a full result set never admits another
candidate, so the descent drains after `ef` expansions per level (measured 608 visits at 50 000
nodes, 990 at 500 000). `a_tied_distance_descent_stops_at_its_visit_cap` fails if that `<` is
ever relaxed.

**Filtering** (predicate-aware / ACORN, `filteredSearch = true` today):

1. **Bitset fast path.** RBAC allow-lists and companion-condition candidate sets are computed
   before the query and passed as a roaring/plain bitset over node ids. Zero callbacks. This
   covers the dominant production filter shapes.
2. **Pipelined TSFN batch path** for arbitrary JS predicates. Traversal batches candidate ids
   (64–256) through a ThreadsafeFunction to a JS evaluator and **continues expanding in
   distance order while verdicts are in flight**; verdicts merge in to steer selection and
   gate results. The existing `filterExpansion` visit budget bounds speculative overshoot.
   Traversal never blocks on the event loop — that would re-import the p99 problem this
   design exists to remove. Worst case (loop saturated): budget exhausts, return what passed —
   the same contract as today's budget-bound filtered search.
3. TSFN lifecycle: shutdown-while-query-in-flight is a first-class test (see rocksdb-js #665's
   TSFN teardown SIGSEGV). napi-rs `ThreadsafeFunction` + explicit abort on env teardown.

**Prefetch has two tiers, and the kernel tier is gated** (`unvisited_prefetched` in
`search.rs`, `prefetch.rs`). Each expansion gathers its unvisited neighbour ids and hints their
slots before the distance loop. The CPU hint (`_mm_prefetch`) overlaps cache misses on resident
pages but is dropped on a non-resident page — no fault, no I/O — so once the plane exceeds page
cache every neighbour became a synchronous, queue-depth-1 major fault (issue #9: p95 5.6 → 68.8 ms
and p99 7.0 → 161 ms from 4M to 8M with p50 flat, ~18k faults/s on the builder). The kernel tier
issues one `process_madvise(MADV_WILLNEED)` over the batch's slot pages (vector and adjacency
only, never the key field), so the k reads start together: measured on NVMe, 32 random pages
cost 2,540 µs as serial faults and 420 µs (submit + touch) batched.

It cannot be always on. The vectored call costs ~0.5 µs per range when the pages are already
resident (1.8 µs per `madvise` on the per-range fallback), against a 0.13 µs resident visit at
128-d, so at ef 1448 (~1,400 expansions) it would add ~20 ms to a ~5 ms in-cache query. And a
static switch cannot be right either: one query walks a hot region near the entry point and a
cold tail. So the gate is per expansion, from an in-process signal: a cycle-counter read
(`rdtsc` where the TSC is invariant, `cntvct_el0` on arm64, `Instant` otherwise; ~14 ns here,
calibrated once at the backend probe) every 4th expansion while unarmed and every expansion while
armed, and a window is fault-scale when it exceeds `kept × (1 µs + vector_bytes ns) + 16 µs` —
~8–10× the resident visit at every supported width plus a term below one NVMe fault (~80 µs). A fault-scale expansion arms a hold of 16
expansions; each fast one decrements it; the kernel prefetch is issued while the hold is armed.
A prefetch-assisted expansion under pressure still waits one device round trip, so it stays
fault-scale and the hold does not oscillate once the faults are parallel; page-cache-hit minor
faults (~1 µs) do not trip it, correctly, since WILLNEED cannot help them. Cost accounting: a
resident plane pays the clock read and no syscall (`SearchStats.willneed_batches` reads 0.0–0.1
per query in the tables below, the pre-emption cases); a spurious arm costs ≤ 16 × k × 0.5 µs
≈ 240 µs at k = 30, about one serial fault, which is also what the first, undetected expansion
of a cold region costs.

This does not conflict with `MADV_RANDOM` (§4, `format.rs`): that is about the readahead window
around a random fault polluting co-tenants' page cache; the targeted advice fetches exactly the
pages the next distance reads need.

Backends are probed at first use, not by kernel version: Linux `process_madvise` (one syscall per
batch; unprivileged self-advice needs Linux ≥ 6.13, so Ubuntu 24.04's 6.8 gets `EPERM`) → per-range
`madvise(MADV_WILLNEED)` (older Linux, macOS; k syscalls, still asynchronous readahead) → off
(Windows; `PrefetchVirtualMemory` is the vectored equivalent, not implemented). A backend that
fails with a permanent error latches the next one down for the process. `HNSW_KERNEL_PREFETCH=0`
forces off — a kill switch and A/B control, read once; there is no "on" value because the gate
decides.

Measured 2026-09-19 on a 4M × 128-d int8 plane (cap 32, 1.3 GB, `/home` NVMe), `bench` built
from `main` vs this branch on the same file and queries, CPU-pinned, on a 20-thread Alder Lake
host that other agents' benchmarks kept at load 20+ and the NVMe under ~70 MB/s of random reads
throughout (a serial fault cost 80 µs on the idle device and ~250 µs during these runs).

*Resident* (file warm, 200 queries, mean µs, 4 alternating rounds after a warm-up pair):

| ef | base per round | branch per round | median Δ | `willneed_batches`/query |
|---|---|---|---|---|
| 128 | 248 · 248 · 317 · 257 | 251 · 288 · 277 · 250 | −0.9 % | 0.0 |
| 512 | 447 · 418 · 465 · 419 | 426 · 520 · 478 · 440 | +4.0 % | 0.0 |
| 1448 | 1726 · 1691 · 1965 · 1754 | 1742 · 1876 · 1951 · 1753 | +0.4 % | 0.1 |

Run-to-run spread on this host was ±15 %, wider than the effect, so the resident cost was also
measured as retired user instructions (`perf stat -e instructions:u`, exact under contention):
gate on vs `HNSW_KERNEL_PREFETCH=0` in the same binary differs by 13.8 M instructions over the
1,200 queries of a run (two "on" runs agree to 11 k), 11.5 k per query or 0.5 % of the query
phase, and the `main` binary sits within 2 M of either (noise). With `rdtsc` at 14 ns on this
CPU and one sample per 4 expansions, the gate's time is 0.4–0.8 % of a query at every ef. An earlier build that read `clock_gettime`
(32 ns) on every expansion measured 3–8 % on the same runs, which is what set the window and the
counter: a high-ef expansion scores only ~3–9 unvisited slots, so per-expansion overhead is paid
~700 times in a 400 µs query.

*Exceeds page cache* (same file under `systemd-run --scope -p MemoryMax=…` after
`fadvise(DONTNEED)`; cgroup v2 charges the file pages, so the plane refaults from the device):

`MemoryMax=256M` (≈ 20 % of the plane resident; base/branch back to back per ef, 100 queries,
50 at ef 1448; ms):

| ef | round | base p50 / p95 / p99 | branch p50 / p95 / p99 | speed-up | batches · major faults per query |
|---|---|---|---|---|---|
| 128 | 1 | 102 / 277 / 587 | 60 / 183 / 222 | 1.7× / 1.5× / 2.6× | 94 · 350 |
| 128 | 2 | 103 / 168 / 453 | 56 / 74 / 146 | 1.8× / 2.3× / 3.1× | 94 · 352 |
| 512 | 1 | 186 / 406 / 885 | 80 / 288 / 320 | 2.3× / 1.4× / 2.8× | 137 · 421 |
| 512 | 2 | 152 / 306 / 542 | 74 / 272 / 370 | 2.0× / 1.1× / 1.5× | 137 · 420 |
| 1448 | 1 | 2527 / 9064 / 10527 | 897 / 2615 / 4079 | 2.8× / 3.5× / 2.6× | 820 · 644 |
| 1448 | 2 | 974 / 2273 / 3031 | 253 / 598 / 979 | 3.8× / 3.8× / 3.1× | 820 · 643 |

The batch and fault counts are the mechanism: at ef 128 the branch takes ~350 major faults for
~1,470 non-resident pages per query, the rest arriving through the ~94 kernel batches, and a
probe on the same loaded device put a 12-page WILLNEED batch at 1.1 ms against 3.1 ms of serial
faults (0.4 ms against 2.5 ms on the idle device), which is where the remaining cost sits. An
earlier single-run round with all three efs in one process showed the branch 2× *worse* at
ef 128/512 and 10× better at ef 1448; the per-ef back-to-back pairs above are what the
minute-scale swings in the other tenant's I/O allow to be compared.

`MemoryMax=768M` (≈ 60 % resident; after the warm-up pass most of a 100-query set's pages are
in cache, so this is the issue's own regime — median on trend, tail from the few queries that
walk into cold pages; ms):

| ef | round | base p50 / p95 / p99 (mean) | branch p50 / p95 / p99 (mean) | batches · major faults per query |
|---|---|---|---|---|
| 128 | 1 | 0.32 / 0.39 / 0.45 | 0.34 / 0.42 / 0.47 | 0.0 · 0.0 |
| 128 | 2 | 0.28 / 0.42 / 55.1 (1.87) | 0.25 / 0.30 / 0.35 (0.26) | 0.0 · 0.0 |
| 512 | 1 | 0.71 / 1.23 / 884 (22.5) | 0.76 / 4.90 / 5.47 (1.62) | 2.1 · 0.0 |
| 512 | 2 | 0.40 / 0.54 / 0.79 | 0.59 / 0.77 / 1.11 | 0.0 · 0.0 |
| 1448 | 1 | 1475 / 5244 / 8327 | 667 / 1788 / 3268 | 818 · 373 |
| 1448 | 2 | 559 / 881 / 1684 | 138 / 242 / 254 | 812 · 372 |

The ef 512 round-1 pair is the issue's shape: base mean 22.5 ms from two or three queries near
one second, branch mean 1.6 ms with those queries at ~5 ms. Which queries land on cold pages
differs run to run (round 2's pairs came out warm on both sides), and pairs with zero batches on
both sides differ only by CPU noise: right after this series, ef 512 fully resident and back to
back at load 27 gave base 399 / 399 µs and branch 408 / 391 / 424 µs mean.

A batch on the loaded device never lost to the serial faults it replaced, measured with the
probe on mixed batches of cold and resident pages: 1 cold page 382 vs 372 µs, 2 cold 900 vs
1518, 3 cold 405 vs 1017, 6 cold 747 vs 2670, 12 cold 2272 vs 9622.

## 8. Write path phasing

- **Phase 1 — dual-write, search cutover.** Insert/update/delete logic stays in JS
  (`HierarchicalNavigableSmallWorld.ts` unchanged algorithmically); mutations persist to BOTH
  the index CF (as today) and the file via native slot-write calls. Search runs native from the
  file. Validation = compare native results against the JS path on the same graph; rollback =
  flip search back to JS, drop the file. The double-write cost is bounded (index writes are
  a fraction of insert cost) and temporary.

  _Integrated_ behind the opt-in `nativePlane: true` index option (search-only: toggling never
  reindexes; int8 + cosine indexes only — the flag no-ops elsewhere). Mutations mirror at the
  exact `indexStore.put/remove` sites via `writeNodeRaw`/`clearNode`/`setEntryPoint` with
  host-allocated ids; the plane file (`<store path>/<table>.<attr>.hnsw`, layer0 cap 128,
  16M-node sparse reservation) is created lazily with a full mirror of the existing CF graph on
  first enable, reopened on restart, deleted on drop/clear/reindex. The compiled module is
  optional (`npm run build:hnsw-plane`); absence falls back to the JS path with one warning.
  Parity, predicate, restart, and lifecycle coverage in `unitTests/resources/vectorIndexPlane.test.js`.
  Watermark/replay wiring, slicing, and msync-cadence flushes are not wired yet (open items).

- **Phase 2 — file-primary.** Drop the CF writes; the file is the only graph store. JS insert
  reads nodes through a native `getNode(id)` (one NAPI crossing per read, ~1 µs — comparable to
  today's decode path). Migration for existing indexes: reindex (accepted contract), or a
  one-shot CF→file bulk conversion since it is a pure format transform.
- **Phase 3 — native insert.** Move the insert search + neighbor selection native (same
  traversal core), leaving JS a thin `index(pk, vector)` call. Unlocks bulk build (C5) at
  native speed and removes the ~tens-of-ms event-loop pin per insert (#895).

## 9. Validation plan

Baselines exist in `benchmarks/hnsw-scale.js` output (1M/2M/5M anchors, e.g. 1M efC-200:
p50 7.2 ms / recall@10-set 0.997 @ ef 512). Acceptance for phase 1:

1. **Parity:** native search over a dual-written graph returns identical candidate sets to the
   JS path at equal ef (modulo seqlock-retry races under concurrent write load — measured as a
   bounded divergence rate, not exact equality under churn).
2. **Recall:** cap-64 prune vs cap-128 measured at 1M and 5M; accept if recall@10 delta ≤ 0.5 pt
   at equal ef, else revisit the cap (header field — rebuild, not redesign).
3. **Latency:** ≥8× p50 improvement at 5M/ef 512 (22.2 ms → ≤2.8 ms), p99 within 2× p50 under
   concurrent insert load (the metric that motivates off-loop execution).
4. **Crash:** kill -9 during sustained ingest → reopen → watermark replay → graph passes
   connectivity + recall checks (extend the #1712 repair test harness).
5. **Churn:** delete/reinsert cycles hold node count stable (freelist reuse; #2182 regression
   test).

## 10. Decisions & open questions

Decided (Kris, 2026-08-31):

- **Degree cap: 128 for the int8 plane** (revised 2026-08-31 after measurement). The original
  cap-64 preference assumed 128 doubles the file; it does not for int8 slots — the 768 B vector
  dominates, so 128 costs +23.5% (1,344 vs 1,088 B slots). Measured at 1M: cap-64 loses 2.2 pts
  of recall (0.975 vs 0.996, where JS = 0.997) at equal ef and equal latency. +24% bytes for
  full recall parity is the right trade. The cap stays a header field; the **binary-code v2
  plane reopens the question** (cap-64 ≈ 352 B vs cap-128 ≈ 608 B slots, +73% — there a
  diversity-preserving prune at lower cap is worth engineering).
- **Platform policy.** Performance is a Linux target only. macOS must work (mmap/msync semantics
  differ slightly — `F_FULLFSYNC` for real durability barriers, no sparse-file guarantees on all
  filesystems — both handled, neither optimized). Windows may fall back to the JS implementation
  entirely; the native plane is allowed to be absent there.
- **Packaging: independent open-source package.** The core has zero Harper coupling — the crate
  compiles standalone and its NAPI surface is generic (create/open plane, insert(id, vector),
  remove(id), search(query, k, ef, filter), watermark get/set). Harper-specific glue — the
  pk→nodeId mapping, commit-callback integration, txnlog-anchored replay, auto-ef policy
  constants — stays in Harper regardless of packaging. Plan: develop in-repo under
  `native/hnsw-plane/` until the NAPI surface stabilizes (end of phase 1), then split to its own
  repo in the symphony/lmdb-js mold and consume via npm. The pitch as a community package: a
  persistent, incrementally-maintained, concurrently-searchable HNSW for Node — hnswlib-node has
  no durable incremental persistence, no off-loop batched filtering, no seqlock concurrency.

Open:

- **Atomic slot payloads.** Fields a concurrent reader acts on (flags, level, degree, scale,
  invMag, neighbor and upper ids) are read through aligned `read_volatile`, which forbids the
  reload/split/sink across the seqlock's validating fence that `lto = true, codegen-units = 1`
  otherwise licenses. That is not the same as being race-free under Rust's memory model: only
  making those fields `AtomicU8`/`AtomicU16`/`AtomicU32` in the slot layout would be, and that
  is a format change deferred past phase 1. The stored vector stays an ordinary load on
  purpose — `cosine_raw` must keep autovectorizing, and a torn vector only perturbs a
  distance the generation check discards. Element width does not change that: a torn 2-byte
  element is as discardable as a torn 1-byte one.
- **msync cadence default** — bounded-lag durability window vs write amplification; needs a
  workload measurement, not a guess.
- **f32 (quantization:"none") slot variant** — 3,072 B vectors → 3.4 KB slots; `H_QUANT = 1` is
  reserved for it, but nothing implements it and `open` refuses the byte.
- **int16 storage precision** — done. `Plane.create(..., precision)` fixes the codec for the
  life of the file; int8 stays the default. int16 exists for accuracy, not speed: at
  `max|c|/32767` the quantization error is ~0.003% of the largest component against int8's
  ~0.8%. Whether that clears a host's bar for ranking on plane distances instead of reranking
  against exact vectors is that host's call against its own corpus; what this repo measures is
  a 128-d ordering test (`tests/precision.rs`), not a production embedding set. The kernel is `_mm256_madd_epi16` over an i16-quantized query, and the
  operand domain is ±32767 — **-32768 is refused at every writer**, because a single pair of
  them sums to exactly 2^31 inside one madd lane, before any accumulator width can help.
  Above the madd, each result is widened to i64 before accumulating (the safe i32 interval is
  one iteration, not several), which also makes the AVX2 kernel bit-identical to the scalar
  reference. Measured on a 12th-gen i7-12700H: the int16 search kernel is 0.76–0.92x the cost
  of the f32 x int8 one, but an int16 plane still searches ~6–11% slower end to end, because
  the extra byte per dimension costs more in scan bandwidth than the kernel saves. The
  f32-accumulating kernel variant is 1.5–1.7x faster than the i64 one and stays in
  `distance.rs` for the benchmark that measured it; it was not shipped because it is accurate
  only to ~1e-5 relative, and exactness is worth more than a fraction of a kernel that is
  under 3% of process CPU.
- ~~Upper-layer region persistence~~ — done (format v2): fixed-entry region in the same file,
  per-entry seqlocks, reserved for max_nodes/8. Upper entries leak on delete (bounded by the
  2x-headroom reserve); an upper freelist is the remaining nicety.
- **Reservation growth** — max_nodes is fixed at create; production needs either a generous
  sparse reservation (Linux-fine; strict-overcommit hosts need care) or mremap-based growth.

## 11. Prototype measurements (kzyp Linux box, 768-d int8, ef 512, cap 64)

Gaussian-mixture corpus matching `benchmarks/hnsw-scale.js` calibration (intra-cos 0.75,
clusters = N/500). JS baseline for scale: 4.34 µs/visit; 1M efC-200 anchor: p50 7.2 ms,
recall@10-set 0.997, ~3,110 visits.

| N                   | cap | p50     | p95     | visits/query | µs/visit | recall@10 (set) | build rate      |
| ------------------- | --- | ------- | ------- | ------------ | -------- | --------------- | --------------- |
| 100K                | 64  | 0.28 ms | 0.46 ms | 1,395        | 0.201    | 1.000           | 5,583 inserts/s |
| 1M                  | 64  | 0.81 ms | 1.60 ms | 2,279        | 0.353    | 0.975           | 1,670 inserts/s |
| 1M                  | 128 | 0.75 ms | 1.48 ms | 2,309        | 0.324    | **0.996**       | 1,242 inserts/s |
| 1M (fmt v2)         | 128 | 0.83 ms | 1.61 ms | 2,309        | 0.359    | 0.996           | 1,346 inserts/s |
| 1M (coverage prune) | 128 | 0.75 ms | 1.52 ms | 2,279        | 0.327    | **0.999**       | 1,359 inserts/s |

Concurrency (same 1M graph): **6,345 QPS aggregate** across 8 searcher threads (p50 1.03 ms,
worst-thread p99 3.84 ms) while a background writer sustained **1,102 inserts/s** — the QPS
input §9 of the Reflex study lacked. Reverse-edge overflow eviction is coverage-aware
(evict the far member provably reachable via a kept nearer one; bounded 16×16 checks): the
concurrent torture test caught closest-keep eviction orphaning nodes in near-duplicate
clusters (~1-in-4 runs), and the fix also raised 1M recall from 0.996 to 0.999 at equal
build cost.
| 1M JS anchor | 128 | 7.2 ms | 12.0 ms | ~3,110 | 4.34 | 0.997 | ~263 inserts/s |

At the 1M anchor with cap 128: **9.6× p50, 12.9× per-visit, 4.7× build rate, at JS-equal
recall.** The µs/visit rise from 100K (0.20) to 1M (0.32–0.35) is the working set leaving L3 —
the memory-hierarchy term; it is the number that holds at 60–100M. An ef-1024 sweep on a
reopened cap-64 plane without its hierarchy (pre-sidecar) still reached 0.985 at p50 2.47 ms —
layer-0 beam is robust to a missing hierarchy, at ~3.4× the visits.

### Parallel batch build (hnsw#6)

`insert_batch` scaling on the same box, measured 2026-09-19 on NVMe (`/home`, not tmpfs):
128-d int8 rows from a 16M float32 pool, layer-0 cap 128, efConstruction 200, 4,096-record
chunks, 200 held-out queries. The box was shared with three other multi-GB benchmark sessions
throughout (load average 10–31 on 20 hardware threads, IO pressure 20–40%), so absolute rates
are conservative and the single-thread reference spent long stretches IO-stalled; the shape
of the curve is the finding. recall@10 is against brute-force truth at ef 256.

| N  | build threads | inserts/s | recall@10 |
| -- | ------------- | --------- | --------- |
| 4M | 1 (serial `insert_with_key`, 3.76 h) | 296 avg; 2,200 → 160 per 200k segment as IO pressure rose | 0.9995 |
| 4M | 2  | 2,065 | 0.9995 |
| 4M | 4  | 4,169 | 0.9995 |
| 4M | 8  | 5,225 | 0.9995 |
| 4M | 16 | 8,461 | 0.9995 |
| 4M | 20 | 7,792 (box oversubscribed: load 24) | 0.9995 |
| 8M | 1 (serial, stopped at 3.95M after 3.7 h) | 293 avg | — |
| 8M | 4  | 2,625 | 1.0000 |
| 8M | 8  | 4,182 | 1.0000 |
| 8M | 16 | 9,105 | 1.0000 |
| 8M | 20 | 8,555 (load 22) | 1.0000 |

On a quiet box the serial path ran 3,114 inserts/s at 200k (SIFT1M rows) and 8 threads
19,695/s — 6.3×. Two things the curve says: recall is unchanged by parallel insertion at
every point (§10's "concurrency only shuffles the insertion permutation" holds at 4M and 8M),
and under IO pressure the single writer collapses (hundreds of inserts/s, `D` state at
~4,300 major faults/s) while 4–16 workers keep 2,000–8,500/s — the fault-overlap effect
hnsw#6 predicted for out-of-cache builds is larger than the CPU term. The 8 → 16 → 20 step
flattens where the box ran out of idle cores; the hub-contention ceiling on a quiet machine
is still to be measured.

Milestones: zero-copy seqlock reads + AVX2 kernels took per-visit cost from 0.440 µs (first
scalar prototype) to ~0.1–0.35 µs, beating the 0.25–0.4 µs design budget. The
optimizeRouting-parity insert (including the recomputed neighbor↔neighbor distances) restored
recall from 0.49 (placeholder insert) to JS parity. Uniform-random 768-d corpora produce
meaningless recall numbers (the JS benchmark's own calibration note: a corpus "no ANN can
index") — all comparisons use the mixture corpus.
