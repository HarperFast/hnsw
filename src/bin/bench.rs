//! Standalone cost benchmark: build an N-node graph in the plane file, run queries, report
//! per-visit cost — the number that decides whether the native plane hits its 0.25–0.4 µs
//! budget (JS baseline: 4.34 µs/visit at 5M/ef 512).
//!
//! Usage: bench [n=100000] [dims=768] [queries=200] [ef=512] [path=/tmp/bench.hnsw] [cap=32] [threads=0] [precision=int8|int16|both] [buildThreads=1]
//! Env: HNSW_BENCH_FVECS=<dir> reads SIFT-style `sift_base.fvecs` / `sift_query.fvecs` from that
//! directory (its dims must match the argument; n rows from base, queries from query) instead of the synthetic
//! corpus, so a run matches the Harper-vs-pgvector benchmark's data. HNSW_BENCH_F32=<file> reads a
//! row-major float32 pool (the harper benchmarks' `--corpus` format): rows 0..n are indexed, the next
//! `queries` rows are held out as queries. `ef` may be a comma list.
//! threads > 0 adds a concurrent-throughput pass: T searcher threads (queries each) + one
//! background writer inserting throughout, reporting aggregate QPS and per-thread p50/p99.
//! `precision=both` builds and measures an int8 and an int16 plane over the same corpus.
//! buildThreads > 1 builds through `insert_batch` in 4096-record chunks (Harper's chunk size)
//! with that many workers per chunk, instead of the serial one-insert-at-a-time path.
//! HNSW_BENCH_KERNELS=1 runs the kernel microbenchmark alone (no graph build).
//! HNSW_BENCH_NO_RECALL=1 skips the brute-force recall truths (queries only; recall prints NaN).
//! HNSW_BENCH_DUMP=<file> writes every single-thread query's visit count and (id:distance) hits,
//! one line per query, so two binaries over the same plane can be diffed for exact equivalence.

use hnsw_plane::distance::{quantize, Query};
use hnsw_plane::format::Quant;
use hnsw_plane::insert::{insert, insert_batch, insert_with_key, InsertParams};
use hnsw_plane::search::{search, SearchScratch};
use hnsw_plane::{Graph, PlaneFile};
use std::path::PathBuf;
use std::time::Instant;

// xorshift for reproducible synthetic vectors without a rand dependency
struct Rng(u64);
impl Rng {
    fn next_unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    // Box-Muller
    fn next_gauss(&mut self) -> f32 {
        let u1 = self.next_unit().max(f32::MIN_POSITIVE);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn major_faults() -> i64 {
    #[cfg(unix)]
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        ru.ru_majflt as i64
    }
    #[cfg(not(unix))]
    0
}

/// Gaussian-mixture corpus matching benchmarks/hnsw-scale.js: unit centroids, per-dim noise
/// derived from an intra-cluster cosine target of 0.75 (uniform-random 768-d is a corpus
/// "no ANN can index" per that benchmark's own calibration notes).
struct Corpus {
    centroids: Vec<f32>,
    n_clusters: usize,
    dims: usize,
    noise: f32,
}

impl Corpus {
    fn new(n: u64, dims: usize, rng: &mut Rng) -> Self {
        let intra_cos = 0.75f32;
        let noise = ((1.0 / (intra_cos * intra_cos) - 1.0) / dims as f32).sqrt();
        let n_clusters = 8.max((n as f64 / 500.0).round() as usize);
        let mut centroids = vec![0.0f32; n_clusters * dims];
        for c in 0..n_clusters {
            let mut mag = 0.0f32;
            for d in 0..dims {
                let x = rng.next_gauss();
                centroids[c * dims + d] = x;
                mag += x * x;
            }
            let mag = mag.sqrt().max(f32::MIN_POSITIVE);
            for d in 0..dims {
                centroids[c * dims + d] /= mag;
            }
        }
        Corpus { centroids, n_clusters, dims, noise }
    }

    fn row(&self, rng: &mut Rng) -> Vec<f32> {
        let c = (rng.next_unit() * self.n_clusters as f32) as usize % self.n_clusters;
        let mut v = vec![0.0f32; self.dims];
        let mut mag = 0.0f32;
        for d in 0..self.dims {
            let x = self.centroids[c * self.dims + d] + rng.next_gauss() * self.noise;
            v[d] = x;
            mag += x * x;
        }
        let mag = mag.sqrt().max(f32::MIN_POSITIVE);
        for d in 0..self.dims {
            v[d] /= mag;
        }
        v
    }
}


/// SIFT-style fvecs: per row an i32 dimension count then that many f32s.
fn read_fvecs(path: &std::path::Path, limit: usize) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).expect("read fvecs");
    let mut rows = Vec::new();
    let mut off = 0usize;
    while off + 4 <= bytes.len() && rows.len() < limit {
        let d = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let mut v = Vec::with_capacity(d);
        for i in 0..d {
            v.push(f32::from_le_bytes(bytes[off + i * 4..off + i * 4 + 4].try_into().unwrap()));
        }
        off += d * 4;
        rows.push(v);
    }
    rows
}

/// Row-major float32 pool (the harper benchmarks' `--corpus=<pool>.f32` format), memory-mapped so
/// a multi-GB pool costs page cache rather than heap. Rows `0..n` are the base; queries are the
/// rows after `n`, held out of the index.
struct F32Pool {
    map: memmap2::Mmap,
    dims: usize,
    n: usize,
}

impl F32Pool {
    fn open(path: &std::path::Path, dims: usize, n: usize, queries: usize) -> Self {
        let file = std::fs::File::open(path).expect("open f32 pool");
        let map = unsafe { memmap2::Mmap::map(&file) }.expect("map f32 pool");
        let rows = map.len() / (dims * 4);
        assert!(rows >= n + queries, "f32 pool holds {rows} rows of {dims} dims; need n + queries = {}", n + queries);
        F32Pool { map, dims, n }
    }
    fn row(&self, i: usize) -> Vec<f32> {
        let start = i * self.dims * 4;
        self.map[start..start + self.dims * 4].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }
}

enum Source {
    Synthetic(Corpus),
    Fvecs { base: Vec<Vec<f32>>, query: Vec<Vec<f32>> },
    F32(F32Pool),
}

impl Source {
    fn base_row(&self, i: usize, rng: &mut Rng) -> Vec<f32> {
        match self {
            Source::Synthetic(c) => c.row(rng),
            Source::Fvecs { base, .. } => base[i].clone(),
            Source::F32(pool) => pool.row(i),
        }
    }
    fn query_row(&self, i: usize, rng: &mut Rng) -> Vec<f32> {
        match self {
            Source::Synthetic(c) => c.row(rng),
            Source::Fvecs { query, .. } => query[i % query.len()].clone(),
            Source::F32(pool) => pool.row(pool.n + i),
        }
    }
}

/// A supplied argument that does not parse is an error, never the default.
fn arg_or<T: std::str::FromStr>(args: &[String], index: usize, name: &str, default: T) -> T {
    match args.get(index) {
        None => default,
        Some(raw) => raw.parse().unwrap_or_else(|_| panic!("{name}: cannot parse {raw:?}")),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if std::env::var("HNSW_BENCH_KERNELS").is_ok() {
        kernel_bench();
        return;
    }
    let n: u64 = arg_or(&args, 1, "n", 100_000);
    let dims: usize = arg_or(&args, 2, "dims", 768);
    let queries: usize = arg_or(&args, 3, "queries", 200);
    let efs: Vec<usize> = args
        .get(4)
        .map(|a| a.split(',').map(|e| e.parse().expect("ef")).collect())
        .unwrap_or_else(|| vec![512]);
    let path: PathBuf = args.get(5).map(Into::into).unwrap_or_else(|| "/tmp/bench.hnsw".into());
    let layer0_cap: usize = arg_or(&args, 6, "cap", 32);
    let threads: usize = arg_or(&args, 7, "threads", 0);
    let quants: Vec<Quant> = match args.get(8).map(String::as_str).unwrap_or("int8") {
        "int8" => vec![Quant::Int8],
        "int16" => vec![Quant::Int16],
        "both" => vec![Quant::Int8, Quant::Int16],
        other => panic!("precision must be int8, int16 or both (got {other})"),
    };
    let build_threads: usize = arg_or::<usize>(&args, 9, "buildThreads", 1).max(1);
    kernel_bench();
    for quant in quants {
        // per-precision path so `both` does not rebuild over the other width's file
        let plane_path = PathBuf::from(format!("{}.{}", path.display(), quant.name()));
        run(n, dims, queries, &efs, &plane_path, layer0_cap, threads, quant, build_threads);
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    n: u64,
    dims: usize,
    queries: usize,
    efs: &[usize],
    path: &std::path::Path,
    layer0_cap: usize,
    threads: usize,
    quant: Quant,
    build_threads: usize,
) {
    let path = path.to_path_buf();
    println!("\n=== precision {} ===", quant.name());

    // Reuse an existing plane file when it holds exactly n nodes at the same geometry from the
    // same corpus (a sidecar names the corpus): ef sweeps without rebuilding.
    let corpus_id = std::env::var("HNSW_BENCH_FVECS")
        .map(|d| format!("fvecs:{d}"))
        .or_else(|_| std::env::var("HNSW_BENCH_F32").map(|f| format!("f32:{f}")))
        .unwrap_or_else(|_| "synthetic".into());
    let corpus_id = format!("{corpus_id}:{}:build{}", quant.name(), build_threads);
    let sidecar = path.with_extension("hnsw.corpus");
    let reuse = PlaneFile::open(&path)
        .ok()
        .filter(|f| f.id_high_water() == n && f.layer0_cap == layer0_cap && f.dims() == dims && f.quant() == quant)
        .is_some()
        && std::fs::read_to_string(&sidecar).map(|c| c == corpus_id).unwrap_or(false);
    let file = if reuse {
        println!("reusing existing plane at {}", path.display());
        PlaneFile::open(&path).expect("open")
    } else {
        PlaneFile::create_with_options(&path, dims, layer0_cap, n + 1024, 40, hnsw_plane::format::default_key_arena_per_node(40), quant)
            .expect("create")
    };
    println!(
        "plane: {} nodes x {} dims {}, slot {} B, file {:.1} GB (sparse)",
        n,
        dims,
        quant.name(),
        file.slot_size,
        (n * file.slot_size as u64) as f64 / 1e9
    );
    let graph = Graph::new(file);
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let source = match std::env::var("HNSW_BENCH_FVECS") {
        Ok(_) if std::env::var("HNSW_BENCH_F32").is_ok() => panic!("set HNSW_BENCH_FVECS or HNSW_BENCH_F32, not both"),
        Ok(dir) => {
            let dir = PathBuf::from(dir);
            let base = read_fvecs(&dir.join("sift_base.fvecs"), n as usize);
            let query = read_fvecs(&dir.join("sift_query.fvecs"), queries);
            assert_eq!(base.len(), n as usize, "fvecs base holds fewer than n rows");
            assert_eq!(base[0].len(), dims, "fvecs dims differ from the dims argument");
            println!("fvecs corpus: {} base rows, {} query rows", base.len(), query.len());
            Source::Fvecs { base, query }
        }
        Err(_) => match std::env::var("HNSW_BENCH_F32") {
            Ok(file) => {
                let pool = F32Pool::open(std::path::Path::new(&file), dims, n as usize, queries);
                println!("f32 pool: {} rows indexed, {} held-out queries", n, queries);
                Source::F32(pool)
            }
            Err(_) => Source::Synthetic(Corpus::new(n, dims, &mut rng)),
        },
    };
    let corpus = &source;

    if reuse {
        // replay the build's RNG draws so query rows match a fresh run; the upper region
        // persists inside the plane file
        for i in 0..n {
            let _ = corpus.base_row(i as usize, &mut rng);
        }
    } else {
        let build_start = Instant::now();
        let progress = |built: u64| {
            if built.is_multiple_of(50_000) {
                println!("  built {} ({:.0} inserts/s)", built, built as f64 / build_start.elapsed().as_secs_f64());
            }
        };
        if build_threads > 1 {
            let mut scratches: Vec<SearchScratch> = (0..build_threads).map(|_| SearchScratch::new()).collect();
            let chunk = 4096u64;
            let mut start = 0u64;
            while start < n {
                let end = (start + chunk).min(n);
                let vectors: Vec<Vec<f32>> = (start..end).map(|i| corpus.base_row(i as usize, &mut rng)).collect();
                let keys: Vec<[u8; 8]> = (start..end).map(|i| i.to_le_bytes()).collect();
                let records: Vec<(&[f32], &[u8])> =
                    vectors.iter().zip(&keys).map(|(v, k)| (v.as_slice(), k.as_slice())).collect();
                let outcome = insert_batch(&graph, &params, &records, &mut scratches)
                    .unwrap_or_else(|f| panic!("batch failed at record {}: {:?}", start + f.index as u64, f.error));
                assert!(outcome.rejected.is_empty(), "rejected records: {:?}", outcome.rejected);
                for built in (start + 1..=end).filter(|b| b % 50_000 == 0) {
                    progress(built);
                }
                start = end;
            }
        } else {
            for i in 0..n {
                let v = corpus.base_row(i as usize, &mut rng);
                insert_with_key(&graph, &v, &i.to_le_bytes(), &params, &mut scratch).expect("build insert");
                progress(i + 1);
            }
        }
        let build = build_start.elapsed();
        println!(
            "build ({} thread{}): {:.1}s ({:.0} inserts/s)",
            build_threads,
            if build_threads == 1 { "" } else { "s" },
            build.as_secs_f64(),
            n as f64 / build.as_secs_f64()
        );
        graph.file.msync().expect("msync");
        // a truth file from the plane this build replaced answers for different base data
        drop_truth_caches(&path);
        std::fs::write(&sidecar, &corpus_id).expect("write corpus sidecar");
    }

    // Query with held-out vectors; measure latency and set-recall@10 vs brute-force truth
    // (same asymmetric metric, so recall isolates graph quality, not quantization).
    let rows: Vec<Vec<f32>> = (0..queries).map(|i| corpus.query_row(i, &mut rng)).collect();
    // HNSW_BENCH_NO_RECALL=1 skips the brute-force truth pass entirely; otherwise the truth is
    // cached beside the plane, keyed by node count and the query vectors themselves, so an ef
    // sweep or a run under a page-cache limit does not rescan the whole plane once per query
    let truth_path = PathBuf::from(format!("{}.truth-{n}n-{:016x}", path.display(), query_hash(&rows)));
    let qs: Vec<Query> = rows.into_iter().map(|row| Query::for_plane(&graph.file, row)).collect();
    let truths: Vec<Vec<u32>> = if std::env::var_os("HNSW_BENCH_NO_RECALL").is_some() {
        qs.iter().map(|_| Vec::new()).collect()
    } else if let Some(cached) = std::fs::read(&truth_path).ok().filter(|bytes| reuse && bytes.len() == queries * 40) {
        println!("reusing brute-force truth at {}", truth_path.display());
        cached.chunks(40).map(|q| q.chunks(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()).collect()
    } else {
        // brute-force truth is O(n x queries); spread it over the machine so an 8M plane does not
        // spend longer on truth than on the build it measures. Bounded top-10 running insert per
        // query (not collect-all-then-sort), so no query keeps an n-entry buffer resident.
        let truth_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1).min(queries.max(1));
        let truths: Vec<Vec<u32>> = std::thread::scope(|s| {
            let chunk = qs.len().div_ceil(truth_threads).max(1);
            let handles: Vec<_> = qs
                .chunks(chunk)
                .map(|qs| {
                    let graph = &graph;
                    s.spawn(move || {
                        qs.iter()
                            .map(|q| {
                                let mut top: Vec<(u32, f32)> = Vec::with_capacity(11);
                                for id in 0..n as u32 {
                                    let Some(d) = graph.distance_to(id, q) else { continue };
                                    if top.len() < 10 || d < top[9].1 {
                                        let at = top.partition_point(|&(_, td)| td <= d);
                                        top.insert(at, (id, d));
                                        top.truncate(10);
                                    }
                                }
                                top.into_iter().map(|(id, _)| id).collect::<Vec<u32>>()
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("truth thread")).collect()
        });
        let bytes: Vec<u8> = truths.iter().flat_map(|q| q.iter().flat_map(|id| id.to_le_bytes())).collect();
        if let Err(error) = std::fs::write(&truth_path, bytes) {
            eprintln!("could not write the truth cache {}: {error}", truth_path.display());
        }
        truths
    };
    let mut dump = std::env::var("HNSW_BENCH_DUMP").ok().map(|p| {
        std::io::BufWriter::new(std::fs::File::create(&p).expect("create HNSW_BENCH_DUMP"))
    });
    let mut ef = efs[0];
    for &ef_i in efs {
        ef = ef_i;
        let mut latencies = Vec::with_capacity(queries);
        let mut total_visits = 0u64;
        let mut willneed_batches = 0u64;
        let mut recall_hits = 0usize;
        let mut recall_total = 0usize;
        for q in &qs {
            let _ = search(&graph, q, 10, ef, &mut scratch);
        }
        let majflt_before = major_faults();
        for (qi, (q, truth)) in qs.iter().zip(&truths).enumerate() {
            let start = Instant::now();
            let (results, stats) = search(&graph, q, 10, ef, &mut scratch);
            latencies.push(start.elapsed());
            total_visits += stats.visits;
            willneed_batches += stats.willneed_batches;
            assert!(!results.is_empty());
            recall_total += truth.len();
            recall_hits += truth.iter().filter(|tid| results.iter().any(|(rid, _)| rid == *tid)).count();
            if let Some(out) = dump.as_mut() {
                use std::io::Write;
                write!(out, "ef {ef} q {qi} visits {}", stats.visits).unwrap();
                for (id, d) in &results {
                    write!(out, " {id}:{d:e}").unwrap();
                }
                writeln!(out).unwrap();
            }
        }
        latencies.sort();
        let p50 = latencies[queries / 2];
        let p95 = latencies[queries * 95 / 100];
        let p99 = latencies[(queries * 99 / 100).min(queries - 1)];
        let mean_visits = total_visits as f64 / queries as f64;
        let mean_us = latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>() / queries as f64 * 1e6;
        let us_per_visit = mean_us / mean_visits;
        println!(
            "search (ef {:>4}): p50 {:.3} ms  p95 {:.3} ms  p99 {:.3} ms  mean {:.1} us  visits/query {:.0}  ->  {:.3} us/visit  recall@10 {:.4}  willneed batches/query {:.1}  major faults/query {:.1}",
            ef,
            p50.as_secs_f64() * 1e3,
            p95.as_secs_f64() * 1e3,
            p99.as_secs_f64() * 1e3,
            mean_us,
            mean_visits,
            us_per_visit,
            recall_hits as f64 / recall_total as f64,
            willneed_batches as f64 / queries as f64,
            (major_faults() - majflt_before) as f64 / queries as f64
        );
    }

    drop(dump);
    // after the timed passes: the scan touches every slot, which would warm a plane that the
    // memory-limited measurement wants cold
    degree_report(&graph, n);

    if threads > 0 {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let graph = Arc::new(graph);
        let source = Arc::new(source);
        let stop = Arc::new(AtomicBool::new(false));
        let per_thread = queries.max(100);
        let anon_before = rss_anon_kb();
        let rendezvous = Arc::new(std::sync::Barrier::new(threads));
        let writer_sized = Arc::new(AtomicBool::new(false));
        let start = Instant::now();
        let mut handles = Vec::new();
        for t in 0..threads {
            let graph = graph.clone();
            let corpus = source.clone();
            let rendezvous = rendezvous.clone();
            let writer_sized = writer_sized.clone();
            handles.push(std::thread::spawn(move || {
                let mut scratch = SearchScratch::new();
                let mut rng = Rng(0x9e37_79b9 ^ (t as u64 + 1) * 0x1234_5677);
                let mut lat: Vec<std::time::Duration> = Vec::with_capacity(per_thread);
                let mut empty = 0usize;
                for i in 0..per_thread {
                    let q = Query::for_plane(&graph.file, corpus.query_row(i, &mut rng));
                    let s = Instant::now();
                    let (r, _) = search(&graph, &q, 10, ef, &mut scratch);
                    lat.push(s.elapsed());
                    empty += r.is_empty() as usize;
                }
                lat.sort();
                // a panic before the barrier would wedge the other searchers in wait()
                let anon = if rendezvous.wait().is_leader() {
                    let deadline = Instant::now() + std::time::Duration::from_secs(5);
                    while !writer_sized.load(Ordering::Relaxed) && Instant::now() < deadline {
                        std::thread::yield_now();
                    }
                    if !writer_sized.load(Ordering::Relaxed) {
                        eprintln!("warning: the writer never sized its scratch; RSS sample excludes it");
                    }
                    rss_anon_kb()
                } else {
                    0
                };
                rendezvous.wait();
                assert_eq!(empty, 0, "searcher {t}: {empty} empty result sets");
                (lat[per_thread / 2], lat[(per_thread * 99 / 100).min(per_thread - 1)], anon)
            }));
        }
        // background writer: sustained inserts while searchers run
        let writer = {
            let graph = graph.clone();
            let corpus = source.clone();
            let stop = stop.clone();
            let writer_sized = writer_sized.clone();
            std::thread::spawn(move || {
                let params = InsertParams::default();
                let mut scratch = SearchScratch::new();
                let mut rng = Rng(0xdead_beef_cafe_f00d);
                let mut count = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let v = corpus.base_row(count as usize % n as usize, &mut rng);
                    let inserted = insert(&graph, &v, &params, &mut scratch);
                    writer_sized.store(true, Ordering::Relaxed);
                    if inserted.is_err() {
                        // plane full: keep the scratch alive until stop so the RSS sample counts it
                        while !stop.load(Ordering::Relaxed) {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                        break;
                    }
                    count += 1;
                }
                count
            })
        };
        let mut p50s = Vec::new();
        let mut p99s = Vec::new();
        let mut anon_peak = 0u64;
        for h in handles {
            let (p50, p99, anon) = h.join().unwrap();
            p50s.push(p50);
            p99s.push(p99);
            anon_peak = anon_peak.max(anon);
        }
        let wall = start.elapsed();
        stop.store(true, Ordering::Relaxed);
        let inserted = writer.join().unwrap();
        let total_q = (threads * per_thread) as f64;
        p50s.sort();
        p99s.sort();
        println!(
            "concurrent: {} threads x {} queries + writer -> {:.0} QPS aggregate  p50(med) {:.2} ms  p99(worst) {:.2} ms  writer {:.0} inserts/s",
            threads,
            per_thread,
            total_q / wall.as_secs_f64(),
            p50s[threads / 2].as_secs_f64() * 1e3,
            p99s[threads - 1].as_secs_f64() * 1e3,
            inserted as f64 / wall.as_secs_f64()
        );
        let growth_mb = (anon_peak as f64 - anon_before as f64) / 1024.0;
        println!(
            "anonymous RSS: {:.1} MB before the searchers, {:.1} MB peak with {} scratches live ({:+.1} MB, {:.2} MB per scratch)",
            anon_before as f64 / 1024.0,
            anon_peak as f64 / 1024.0,
            threads + 1,
            growth_mb,
            growth_mb / (threads + 1) as f64
        );
    }
}

/// Anonymous resident memory of this process in kB (Linux; 0 elsewhere). Anonymous rather than
/// total RSS so the mmap'd plane's page-cache residency does not drown the scratch allocations
/// this reports on.
fn rss_anon_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("RssAnon:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

/// Kernel microbenchmark: the measurement that chooses the int16 accumulator, and the
/// int8-vs-int16 comparison the option's rationale rests on. Dot products only — no graph, no
/// cache misses — so it isolates the kernel from the traversal it sits inside.
///
/// The row that matters for search is `search kernel`: the asymmetric f32-query x int8-stored
/// path against the int16 query x int16-stored one. The symmetric row is the construction-time
/// neighbour-pruning kernel.
fn kernel_bench() {
    let mut rng = Rng(0x51ed_270b_a5c8_1f3d);
    let dir = std::env::temp_dir().join(format!("hnsw-kernelbench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    println!("kernel microbenchmark (ns per dot product, hot cache):");
    for &dims in &[128usize, 768, 1536, 4096] {
        let a: Vec<f32> = (0..dims).map(|_| rng.next_gauss()).collect();
        let b: Vec<f32> = (0..dims).map(|_| rng.next_gauss()).collect();
        let a8 = quantize(&a, Quant::Int8);
        let b8 = quantize(&b, Quant::Int8);
        let a16 = quantize(&a, Quant::Int16);
        let b16 = quantize(&b, Quant::Int16);
        let iters = (2_000_000 / dims).max(2_000);

        let plane8 = PlaneFile::create_with_options(&dir.join(format!("q8-{dims}.hnsw")), dims, 8, 16, 0, 0, Quant::Int8)
            .expect("bench plane");
        let plane16 = PlaneFile::create_with_options(&dir.join(format!("q16-{dims}.hnsw")), dims, 8, 16, 0, 0, Quant::Int16)
            .expect("bench plane");
        let q8 = Query::for_plane(&plane8, a.clone());
        let q16 = Query::for_plane(&plane16, a.clone());

        let time = |f: &dyn Fn()| {
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_secs_f64() * 1e9 / iters as f64
        };

        let search8 = time(&|| {
            std::hint::black_box(unsafe { hnsw_plane::distance::cosine_raw(&q8, b8.bytes.as_ptr(), b8.scale, b8.inv_mag) });
        });
        let search16 = time(&|| {
            std::hint::black_box(unsafe { hnsw_plane::distance::cosine_raw(&q16, b16.bytes.as_ptr(), b16.scale, b16.inv_mag) });
        });
        let sym8 = time(&|| {
            std::hint::black_box(unsafe {
                hnsw_plane::distance::cosine_stored_raw(
                    Quant::Int8, a8.bytes.as_ptr(), a8.scale, a8.inv_mag, b8.bytes.as_ptr(), b8.scale, b8.inv_mag, dims,
                )
            });
        });
        let sym16 = time(&|| {
            std::hint::black_box(unsafe {
                hnsw_plane::distance::cosine_stored_raw(
                    Quant::Int16, a16.bytes.as_ptr(), a16.scale, a16.inv_mag, b16.bytes.as_ptr(), b16.scale, b16.inv_mag, dims,
                )
            });
        });
        #[cfg(target_arch = "x86_64")]
        let acc_f32 = if std::arch::is_x86_feature_detected!("avx2") {
            time(&|| {
                std::hint::black_box(unsafe {
                    hnsw_plane::distance::dot_i16_i16_avx2_f32acc(a16.bytes.as_ptr(), b16.bytes.as_ptr(), dims)
                });
            })
        } else {
            f64::NAN
        };
        #[cfg(not(target_arch = "x86_64"))]
        let acc_f32 = f64::NAN;

        println!(
            "  dims {:>4}  search kernel: f32xi8 {:>7.1} ns  i16xi16 {:>7.1} ns ({:.2}x)  |  symmetric: i8 {:>7.1} ns  i16 {:>7.1} ns  |  i16 accumulator: i64 {:>7.1} ns  f32 {:>7.1} ns",
            dims, search8, search16, search16 / search8, sym8, sym16, sym16, acc_f32
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// FNV-1a over the queries' bytes: the truth cache identity.
fn query_hash(rows: &[Vec<f32>]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in rows.iter().flatten().flat_map(|v| v.to_le_bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn drop_truth_caches(plane: &std::path::Path) {
    let Some(name) = plane.file_name().and_then(|f| f.to_str()) else { return };
    let prefix = format!("{name}.truth-");
    let dir = plane.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if entry.file_name().to_str().is_some_and(|f| f.starts_with(&prefix)) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn degree_report(graph: &Graph, n: u64) {
    let mut degrees: Vec<u32> = Vec::with_capacity(n as usize);
    let mut buf = Vec::new();
    for id in 0..n as u32 {
        if graph.neighbors_into(id, &mut buf).is_some() {
            degrees.push(buf.len() as u32);
        }
    }
    if degrees.is_empty() {
        return;
    }
    degrees.sort_unstable();
    let pct = |p: usize| degrees[(degrees.len() * p / 100).min(degrees.len() - 1)];
    let mean = degrees.iter().map(|&d| d as f64).sum::<f64>() / degrees.len() as f64;
    let within = |cap: u32| degrees.iter().filter(|&&d| d <= cap).count() as f64 / degrees.len() as f64 * 100.0;
    println!(
        "layer-0 degree: mean {:.1}  p50 {}  p90 {}  p99 {}  max {}  |  nodes at degree <=32 {:.1}%  <=48 {:.1}%  <=64 {:.1}%",
        mean, pct(50), pct(90), pct(99), degrees[degrees.len() - 1], within(32), within(48), within(64)
    );
}
