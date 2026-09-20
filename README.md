# @harperfast/hnsw

Persistent, incrementally-maintained, concurrently-searchable HNSW vector index for Node.js —
a native (Rust) traversal engine over a memory-mapped fixed-slot graph file.

Most HNSW libraries for Node either keep the graph in JS objects (slow per-visit cost, GC
pressure) or wrap an in-memory C++ index with no durable incremental persistence. This one is
built around a different contract:

- **The file is the index.** One memory-mapped file per index: fixed-size node slots
  (quantized vector + neighbor ids, page-grouped so slots never straddle page
  boundaries), an in-file upper-layer region, an id freelist, and a durability watermark.
  Reopen is instant — no rebuild, no sidecars.
- **Search never touches the JS event loop.** Queries run on the libuv thread pool with one
  N-API crossing each; traversal is zero-copy against the mapping with SIMD (AVX2) int8
  asymmetric-cosine distance.
- **Reads and writes are genuinely concurrent.** Per-slot seqlocks, no global locks on the
  search path. Measured on one Linux box at 1M × 768-d (ef 512): **6,300+ QPS aggregate
  across 8 search threads while a writer sustains ~1,100 inserts/s**, p50 ≈ 1 ms.
- **Index build scales with cores.** `insertBatch` takes a whole chunk across one N-API
  crossing and fans the inserts out over worker threads inside the native module, off the
  event loop — the per-slot seqlocks are what make concurrent inserts safe. Scaling curve in
  [DESIGN.md §11](DESIGN.md#11-prototype-measurements).
- **Incremental by design.** Insert, update in place, delete with neighbor repair; deleted
  ids are reused via the freelist, so churn never inflates the graph. Reverse-edge overflow
  uses coverage-aware pruning (a bounded RobustPrune) — measured recall\@10 of 0.999 at 1M
  (768-d int8, ef 512) on a calibrated Gaussian-mixture corpus.
- **Filtering built in.** Allow-bitset filtering (zero callbacks), or a JS predicate
  evaluated in pipelined batches over a threadsafe function while traversal keeps expanding —
  a busy event loop costs speculative overshoot, never search-thread stalls.
- **Two integration modes.** Standalone (the library allocates ids and maintains the graph:
  `insert`/`remove`/`search`), or mirroring (`writeNodeRaw`/`clearNode`: a host application
  that already maintains an HNSW graph mirrors it in and gets the native search path —
  this is how [Harper](https://github.com/HarperFast/harper) integrates it).

Durability is deliberately relaxed: the file is msync'd on a cadence with a transaction
watermark, and the intended recovery model is "replay indexing from the watermark" against
the host's authoritative record store. Approximate indexes don't need per-commit fsyncs;
they need cheap, bounded catch-up. See [DESIGN.md](DESIGN.md) for the format, the
concurrency model, measured baselines, and the reasoning behind every trade.

## Install

```bash
npm install @harperfast/hnsw
```

Prebuilt bindings ship as platform-specific `optionalDependencies`
(`@harperfast/hnsw-<platform>-<arch>[-glibc]`) for linux-x64, linux-arm64, darwin-arm64, and
win32-x64. Platforms without a published binding (musl, darwin-x64, win32-arm64) build from
source on install when a [Rust toolchain](https://rustup.rs) is present, and throw a clear
error otherwise. Linux x86_64 is the performance target (AVX2 + kernel-lock crash recovery);
macOS and Windows are functional (no lock takeover — bounded degradation instead).

## Usage

```js
const { Plane } = require('@harperfast/hnsw');

// keyCap 40 (min 8): each slot carries up to 40 bytes of the host's key inline (longer keys overflow)
const plane = Plane.create('/data/vectors.hnsw', 768, 128, 10_000_000, 40);
// ... or with the finer storage precision (see below):
// const plane = Plane.create('/data/vectors.hnsw', 128, 128, 10_000_000, 40, undefined, 'int16');
const id = plane.insert(myFloat32Vector, Buffer.from(myRecordKey));
// bulk load: one crossing per chunk, inserted in parallel off the event loop; `ids` is in input
// order, keys are concatenated with SearchHits-style ends. Records the plane cannot hold come
// back in `rejected` (index + code) with 0xFFFFFFFF in their slot; a full plane rejects the
// promise with an error that still carries `ids`.
const { ids, rejected } = await plane.insertBatch(chunkVectors /* count × dims */, chunkKeys, chunkKeyEnds, 8 /* threads */);
// parallel typed arrays, ascending by distance; hit i's key is keys.subarray(keyEnds[i-1] ?? 0, keyEnds[i])
const { ids, distances, keys, keyEnds } = await plane.search(queryVector, 10, 512);

// filtered: allow-bitset over node ids
const allowed = new Uint8Array(Math.ceil(plane.idHighWater() / 8));
// ... set bits ...
const filtered = await plane.search(queryVector, 10, 512, allowed);

// or a JS predicate, batched off the event loop
const predicated = await plane.searchWithPredicate(queryVector, 10, 512, (ids) =>
	Uint8Array.from(ids, (id) => (isVisible(id) ? 1 : 0))
);
```

### Storage precision

Vectors are stored quantized with a per-vector symmetric scale. The default, `'int8'`, maps
each component to `max|c|/127` — about 0.8% of the vector's largest component per element,
which is usually close enough to *rank* candidates but not to *score* them, so callers
typically rerank the returned hits against exact vectors. `'int16'` maps to `max|c|/32767`
instead: ~256× finer, about 0.003% per element. Whether that is close enough to drop the
rerank is a question about your corpus and your accuracy budget, and this package cannot
answer it for you — validate it against your own data before turning a rerank off. It costs
one more byte per dimension per slot — +18% at 128 dims (704 → 832 B), +73% at 1536
(2112 → 3648 B) — so it suits small-to-mid dimensionality, while int8 stays the right choice
for wide embeddings, where doubling the bytes each traversal scans pushes search into the
memory-bound regime. The choice is fixed at create and cannot be changed without a rebuild.
Int16 planes carry a newer format version, so an older build of this package refuses to open
one rather than misreading its slot layout; `plane.precision` reports an open plane's codec.

A plane is derived state; when the host must stop maintaining one and cannot delete the file
(Windows sharing violations while another process maps it), `invalidatePlane(path)` — or
`plane.invalidateFile()` through a handle the host already holds — durably marks it
unadoptable: a one-way in-band latch (watermark reads 0, `Plane.open` refuses) plus a fsync'd
`<path>.stale` sidecar (`stalePathFor(path)`, which `open` also refuses). It throws only when
neither marker lands. Hosts delete both files and rebuild.

Full API in [index.d.ts](index.d.ts).

## Benchmarks

`cargo run --release --bin bench -- 1000000 768 100 512 /tmp/bench.hnsw 128 8` builds a 1M ×
768-d graph on a calibrated Gaussian-mixture corpus, reports p50/p95/p99, per-visit cost,
brute-force recall\@10, and a concurrent-throughput pass. A ninth argument selects the
storage precision (`int8`, `int16`, or `both` to build and measure one plane of each), and
every run prints a kernel microbenchmark first. Numbers from the design work
(Linux, single box): p50 0.75 ms, 0.33 µs/visit, recall\@10 0.999 — ~9× the wall-clock and
~13× the per-visit cost of a well-optimized pure-JS implementation of the same graph at
equal recall.

## Status

Extracted from the Harper vector-index engine; the format (v8 for int8 planes, v9 for int16)
and API are young and may
change with a version bump + reindex (an older format version fails to open; rebuild). Roadmap: prebuilds, binary-quantized slot format
(~4× smaller traversal plane), Matryoshka dimension truncation, mremap growth, index
slicing with native top-k merge.

## License

Apache-2.0
