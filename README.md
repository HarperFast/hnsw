# @harperfast/hnsw

Persistent, incrementally-maintained, concurrently-searchable HNSW vector index for Node.js —
a native (Rust) traversal engine over a memory-mapped fixed-slot graph file.

Most HNSW libraries for Node either keep the graph in JS objects (slow per-visit cost, GC
pressure) or wrap an in-memory C++ index with no durable incremental persistence. This one is
built around a different contract:

- **The file is the index.** One memory-mapped file per index: fixed-size node slots
  (int8-quantized vector + neighbor ids, page-grouped so slots never straddle page
  boundaries), an in-file upper-layer region, an id freelist, and a durability watermark.
  Reopen is instant — no rebuild, no sidecars.
- **Search never touches the JS event loop.** Queries run on the libuv thread pool with one
  N-API crossing each; traversal is zero-copy against the mapping with SIMD (AVX2) int8
  asymmetric-cosine distance.
- **Reads and writes are genuinely concurrent.** Per-slot seqlocks, no global locks on the
  search path. Measured on one Linux box at 1M × 768-d (ef 512): **6,300+ QPS aggregate
  across 8 search threads while a writer sustains ~1,100 inserts/s**, p50 ≈ 1 ms.
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

## Install & build

```bash
npm install @harperfast/hnsw
```

Prebuilds are not published yet: building requires a [Rust toolchain](https://rustup.rs)
(`npm run build`, or automatically on install when cargo is available). Linux x86_64 is the
performance target (AVX2); macOS works (scalar fallback); Windows is untested.

## Usage

```js
const { Plane } = require('@harperfast/hnsw');

const plane = Plane.create('/data/vectors.hnsw', 768, 128, 10_000_000);
const id = plane.insert(myFloat32Vector);
const hits = await plane.search(queryVector, 10, 512); // [{ id, distance }, ...]

// filtered: allow-bitset over node ids
const allowed = new Uint8Array(Math.ceil(plane.idHighWater() / 8));
// ... set bits ...
const filtered = await plane.search(queryVector, 10, 512, allowed);

// or a JS predicate, batched off the event loop
const predicated = await plane.searchWithPredicate(queryVector, 10, 512, (ids) =>
	Uint8Array.from(ids, (id) => (isVisible(id) ? 1 : 0))
);
```

Full API in [index.d.ts](index.d.ts).

## Benchmarks

`cargo run --release --bin bench -- 1000000 768 100 512 /tmp/bench.hnsw 128 8` builds a 1M ×
768-d graph on a calibrated Gaussian-mixture corpus, reports p50/p95/p99, per-visit cost,
brute-force recall\@10, and a concurrent-throughput pass. Numbers from the design work
(Linux, single box): p50 0.75 ms, 0.33 µs/visit, recall\@10 0.999 — ~9× the wall-clock and
~13× the per-visit cost of a well-optimized pure-JS implementation of the same graph at
equal recall.

## Status

Extracted from the Harper vector-index engine; the format (v2) and API are young and may
change with a major version + reindex. Roadmap: prebuilds, binary-quantized slot format
(~4× smaller traversal plane), Matryoshka dimension truncation, mremap growth, index
slicing with native top-k merge.

## License

Apache-2.0
