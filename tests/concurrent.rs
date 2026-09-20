//! Concurrent-write torture: writers insert while readers search; then verify the graph is
//! coherent (every stored vector findable, edge lists within cap, freelist reuse works). The
//! deterministic descent tests at the bottom share `vector_for`: its clustered corpus is what
//! makes them bite.

use hnsw_plane::distance::Query;
use hnsw_plane::format::NO_ID;
use hnsw_plane::insert::{insert, insert_batch, InsertError, InsertParams};
use hnsw_plane::search::{beam_descend, search, search_layer, SearchScratch, SearchStats, DESCENT_EF};
use hnsw_plane::{Graph, PlaneFile};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};

fn vector_for(i: u32, dims: usize) -> Vec<f32> {
    // deterministic distinct unit-ish vectors on a few clusters
    let mut v = vec![0.0f32; dims];
    let cluster = (i % 7) as usize;
    for d in 0..dims {
        let x = ((i as f32 * 0.37 + d as f32 * 1.13).sin() * 0.1) + if d % 7 == cluster { 1.0 } else { 0.0 };
        v[d] = x;
    }
    // Per-node signature (unique for i < dims^3). The cluster spike plus 0.1-amplitude noise
    // alone leaves every member of a cluster inside int8 quantization noise of every other, so
    // a self-query cannot tell "found this node" from "found some other node" — and a
    // distance-only assertion over such a corpus passes even when the node is orphaned.
    v[(i as usize) % dims] += 0.5;
    v[(i as usize / dims) % dims] += 0.35;
    v[(i as usize / (dims * dims)) % dims] += 0.22;
    v
}

#[test]
fn concurrent_insert_search() {
    let dims = 64;
    let path = std::env::temp_dir().join(format!("hnsw-torture-{}.hnsw", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let file = PlaneFile::create(&path, dims, 32, 40_000).expect("create");
    let graph = Arc::new(Graph::new(file));

    let writers = 4u32;
    let per_writer = 2_000u32;
    let done = Arc::new(AtomicBool::new(false));

    // (corpus index, node id): ids come from the plane's own allocator, so writers interleave
    // them — a self-query must be checked against the id its insert actually returned
    let inserted: Vec<(u32, u32)> = std::thread::scope(|s| {
        let writers_done: Vec<_> = (0..writers)
            .map(|w| {
                let graph = graph.clone();
                s.spawn(move || {
                    let params = InsertParams::default();
                    let mut scratch = SearchScratch::new();
                    let mut mine = Vec::with_capacity(per_writer as usize);
                    for i in 0..per_writer {
                        let index = w * per_writer + i;
                        let v = vector_for(index, dims);
                        mine.push((index, insert(&graph, &v, &params, &mut scratch).expect("insert")));
                    }
                    mine
                })
            })
            .collect();
        for _ in 0..4 {
            let graph = graph.clone();
            let done = done.clone();
            s.spawn(move || {
                let mut scratch = SearchScratch::new();
                let mut q = 0u32;
                while !done.load(Ordering::Relaxed) {
                    let query = graph.query(vector_for(q % 1000, dims));
                    let (results, _) = search(&graph, &query, 10, 64, &mut scratch);
                    // once anything is inserted, results must be non-empty and finite
                    for (_, d) in &results {
                        assert!(d.is_finite());
                    }
                    q += 1;
                }
            });
        }
        // scope joins writers when their closures end; signal readers afterward via a
        // dedicated waiter thread
        let graph_ref = graph.clone();
        let done_ref = done.clone();
        s.spawn(move || {
            while graph_ref.file.id_high_water() < (writers * per_writer) as u64 {
                std::thread::yield_now();
            }
            done_ref.store(true, Ordering::Relaxed);
        });
        writers_done.into_iter().flat_map(|h| h.join().expect("writer panicked")).collect()
    });

    let total = writers * per_writer;
    assert_eq!(graph.file.id_high_water(), total as u64);

    // Every stored vector must be found as its own nearest neighbor at generous ef.
    let mut scratch = SearchScratch::new();
    let mut misses = 0;
    for &(index, id) in inserted.iter().step_by(97) {
        let query = graph.query(vector_for(index, dims));
        let (results, _) = search(&graph, &query, 10, 256, &mut scratch);
        // by ID, not by distance: this corpus is clustered near-duplicates, so a hit at
        // distance ~0 is routinely a DIFFERENT node and would mask an orphaned one
        if !results.iter().any(|&(rid, _)| rid == id) {
            misses += 1;
        }
    }
    assert_eq!(misses, 0, "self-queries missing after concurrent build");

    // Edge lists respect the cap.
    for id in (0..total).step_by(53) {
        if let Some(n) = graph.read_node(id) {
            assert!(n.neighbors.len() <= graph.file.layer0_cap);
        }
    }

    // Delete + reinsert reuses ids (freelist; the #2182 fix).
    let _ = graph.delete_node(5);
    let _ = graph.delete_node(6);
    let params = InsertParams::default();
    let a = insert(&graph, &vector_for(90_001, dims), &params, &mut scratch).unwrap();
    let b = insert(&graph, &vector_for(90_002, dims), &params, &mut scratch).unwrap();
    assert!(a == 5 || a == 6, "expected freelist reuse, got {a}");
    assert!(b == 5 || b == 6, "expected freelist reuse, got {b}");
    assert_eq!(graph.file.id_high_water(), total as u64, "high-water must not grow on reuse");

    let _ = std::fs::remove_file(&path);
}

/// Orthogonal per-writer vector: every writer's self-query has exactly one right answer, so a
/// node that lost the first-entry race is unmissable rather than covered by a near-duplicate.
fn axis_vector(writer: u32, dims: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dims];
    v[writer as usize % dims] = 1.0;
    v
}

/// Many small fresh graphs, each racing its FIRST insert: that window is where the empty-graph
/// entry-point claim races, and a single barrier in a long build samples it about once.
#[test]
fn racing_first_inserts_all_stay_reachable() {
    let dims = 32;
    let writers = 4u32;
    let rounds = 200;
    for round in 0..rounds {
        let path = std::env::temp_dir().join(format!("hnsw-first-{}-{round}.hnsw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let graph = Arc::new(Graph::new(PlaneFile::create(&path, dims, 16, 256).expect("create")));
        let barrier = Arc::new(Barrier::new(writers as usize));
        let ids: Vec<(u32, u32)> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..writers)
                .map(|w| {
                    let graph = graph.clone();
                    let barrier = barrier.clone();
                    s.spawn(move || {
                        let params = InsertParams::default();
                        let mut scratch = SearchScratch::new();
                        barrier.wait();
                        (w, insert(&graph, &axis_vector(w, dims), &params, &mut scratch).expect("insert"))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("writer panicked")).collect()
        });

        let mut scratch = SearchScratch::new();
        for (w, id) in &ids {
            let (results, _) = search(&graph, &graph.query(axis_vector(*w, dims)), 8, 64, &mut scratch);
            assert!(
                results.iter().any(|&(rid, _)| rid == *id),
                "round {round}: writer {w}'s node {id} is unreachable from the entry point (found {results:?})"
            );
        }
        drop(graph);
        let _ = std::fs::remove_file(&path);
    }
}

/// Fisher-Yates over a xorshift stream: one seed names one exact graph, with no thread
/// interleaving in it.
fn insertion_order(n: u32, seed: u64) -> Vec<u32> {
    let mut order: Vec<u32> = (0..n).collect();
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    for i in (1..order.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        order.swap(i, (s % (i as u64 + 1)) as usize);
    }
    order
}

/// `concurrent_insert_search`'s reachability assertion with the concurrency removed. Concurrency
/// only shuffles the insertion order, which this fixes outright, so what remains is the descent:
/// a width-1 hill climb halts in the first basin no neighbor improves on, and layer-0 adjacency
/// is intra-basin, leaving the query's own vector unreachable at any ef. Sampling every 97th node
/// as the test above does catches that ~1.5% of the time; these two seeds catch it every time,
/// losing 55 and 22 of their 8000 self-queries on a width-1 descent.
#[test]
fn a_descent_that_traps_at_a_local_minimum_still_reaches_the_true_neighborhood() {
    let dims = 64;
    let n = 8_000u32;
    for &seed in &[57u64, 240] {
        let path = std::env::temp_dir().join(format!("hnsw-descent-{}-{seed}.hnsw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let graph = Graph::new(PlaneFile::create(&path, dims, 32, n as u64 + 1024).expect("create"));
        let params = InsertParams::default();
        let mut scratch = SearchScratch::new();

        let inserted: Vec<(u32, u32)> = insertion_order(n, seed)
            .into_iter()
            .map(|index| {
                let v = vector_for(index, dims);
                (index, insert(&graph, &v, &params, &mut scratch).expect("insert"))
            })
            .collect();

        let mut misses = Vec::new();
        for &(index, id) in &inserted {
            let query = graph.query(vector_for(index, dims));
            let (results, _) = search(&graph, &query, 10, 256, &mut scratch);
            if !results.iter().any(|&(rid, _)| rid == id) {
                misses.push((index, id));
            }
        }
        assert!(
            misses.is_empty(),
            "seed {seed}: {} of {n} nodes are their own true nearest neighbor but unreachable \
from the entry point — the descent stranded the search. First few (corpus index, node id): {:?}",
            misses.len(),
            &misses[..misses.len().min(8)]
        );

        drop(graph);
        let _ = std::fs::remove_file(&path);
    }
}

/// `search` at an explicit descent width — what the sweep varies, since `search` itself reads the
/// `DESCENT_EF` constant. A single-threaded build always has a live entry point, so this skips
/// `search`'s dead-entry repair and is otherwise the same sequence.
fn search_at_descent_width(
    graph: &Graph,
    query: &Query,
    k: usize,
    ef: usize,
    descent_ef: usize,
    scratch: &mut SearchScratch,
) -> Vec<(u32, f32)> {
    let (entry_id, entry_level) = graph.file.entry_point();
    let Some(entry_dist) = graph.distance_to(entry_id, query) else {
        return Vec::new();
    };
    let mut stats = SearchStats::default();
    let (ep, ep_dist) =
        beam_descend(graph, query, entry_id, entry_dist, entry_level, 0, descent_ef, scratch, &mut stats);
    scratch.begin_public(graph.file.id_high_water());
    let mut out = Vec::new();
    search_layer(graph, query, ep, ep_dist, ef, 0, scratch, &mut stats, None, u64::MAX, &mut out);
    out.truncate(k);
    out
}

/// The measurement behind DESIGN.md's descent-width table, kept runnable so the numbers can be
/// re-derived when M, ml or the prune policy changes. Ignored: it reports rather than asserts,
/// and a full sweep is minutes of CPU.
///
/// `HNSW_SWEEP_READ_EF` sets the query-side descent width; the build side is whatever
/// `DESCENT_EF` is compiled as. Varying them independently is what separates a routing defect
/// from a construction one — DESIGN.md §7 records that matrix.
///
/// ```text
/// HNSW_SWEEP_SEEDS=700 HNSW_SWEEP_READ_EF=8 cargo test --release --test concurrent \
///     descent_width_sweep -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn descent_width_sweep() {
    let dims = 64;
    let n = 8_000u32;
    let seeds: u64 = std::env::var("HNSW_SWEEP_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
    // 0 would silently mean width 1, since the entry is admitted before any cap check
    let read_ef: usize = std::env::var("HNSW_SWEEP_READ_EF")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(|v: usize| v.max(1))
        .unwrap_or(DESCENT_EF);
    let mut total = 0usize;
    let mut bad = 0usize;
    for seed in 0..seeds {
        let path = std::env::temp_dir().join(format!("hnsw-sweep-{}-{seed}.hnsw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let graph = Graph::new(PlaneFile::create(&path, dims, 32, n as u64 + 1024).expect("create"));
        let params = InsertParams::default();
        let mut scratch = SearchScratch::new();
        let inserted: Vec<(u32, u32)> = insertion_order(n, seed)
            .into_iter()
            .map(|index| {
                let v = vector_for(index, dims);
                (index, insert(&graph, &v, &params, &mut scratch).expect("insert"))
            })
            .collect();
        let misses = inserted
            .iter()
            .filter(|&&(index, id)| {
                let q = graph.query(vector_for(index, dims));
                let results = search_at_descent_width(&graph, &q, 10, 256, read_ef, &mut scratch);
                !results.iter().any(|&(rid, _)| rid == id)
            })
            .count();
        if misses > 0 {
            println!("seed {seed}: {misses} misses");
            bad += 1;
        }
        total += misses;
        drop(graph);
        let _ = std::fs::remove_file(&path);
    }
    println!(
        "descent width sweep (build {DESCENT_EF} / read {read_ef}): {total} misses over {seeds} \
builds of {n}, {bad} builds affected"
    );
}

/// The same writers-while-readers shape on an int16 plane. Element width is under the seqlock,
/// not beside it, so a torn 2-byte element must be as discardable as a torn 1-byte one —
/// smaller than the torture above because the point is the codec, not the scale.
#[test]
fn concurrent_insert_search_on_an_int16_plane() {
    use hnsw_plane::format::Quant;
    let dims = 64;
    let path = std::env::temp_dir().join(format!("hnsw-torture16-{}.hnsw", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let file = PlaneFile::create_with_options(&path, dims, 32, 12_000, 0, 0, Quant::Int16).expect("create");
    let graph = Arc::new(Graph::new(file));
    let done = Arc::new(AtomicBool::new(false));
    let (writers, per_writer) = (3u32, 1_000u32);

    let inserted: Vec<(u32, u32)> = std::thread::scope(|s| {
        let writers_done: Vec<_> = (0..writers)
            .map(|w| {
                let graph = graph.clone();
                s.spawn(move || {
                    let params = InsertParams::default();
                    let mut scratch = SearchScratch::new();
                    (0..per_writer)
                        .map(|i| {
                            let index = w * per_writer + i;
                            (index, insert(&graph, &vector_for(index, dims), &params, &mut scratch).expect("insert"))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let reader = {
            let graph = graph.clone();
            let done = done.clone();
            s.spawn(move || {
                let mut scratch = SearchScratch::new();
                let mut searches = 0u32;
                while !done.load(Ordering::Relaxed) {
                    let (hits, _) = search(&graph, &graph.query(vector_for(searches % 500, dims)), 5, 64, &mut scratch);
                    assert!(hits.iter().all(|(_, d)| d.is_finite()), "a concurrent read produced a non-finite distance");
                    searches += 1;
                }
                searches
            })
        };
        let all: Vec<(u32, u32)> = writers_done.into_iter().flat_map(|h| h.join().unwrap()).collect();
        done.store(true, Ordering::Relaxed);
        reader.join().unwrap();
        all
    });

    let mut scratch = SearchScratch::new();
    let mut misses = 0;
    for &(index, id) in inserted.iter().step_by(37) {
        let (hits, _) = search(&graph, &graph.query(vector_for(index, dims)), 5, 96, &mut scratch);
        if !hits.iter().any(|&(hid, _)| hid == id) {
            misses += 1;
        }
    }
    assert_eq!(misses, 0, "every int16-stored vector must be findable as its own nearest neighbor");
    let _ = std::fs::remove_file(&path);
}

/// Pack corpus rows `indices` into the flat record form `insert_batch` takes: one vector slice and
/// one key slice per record, keys being the corpus index so a hit names its row.
fn batch_records(indices: &[u32], dims: usize) -> (Vec<Vec<f32>>, Vec<[u8; 4]>) {
    let vectors: Vec<Vec<f32>> = indices.iter().map(|&i| vector_for(i, dims)).collect();
    let keys: Vec<[u8; 4]> = indices.iter().map(|i| i.to_le_bytes()).collect();
    (vectors, keys)
}

/// A batch build over the clustered corpus, 8 threads per batch: every returned id is in input
/// order, findable by a self-query at generous ef, and carries its record's key.
#[test]
fn batch_insert_returns_ids_in_input_order_and_every_node_stays_reachable() {
    let dims = 64;
    let path = std::env::temp_dir().join(format!("hnsw-batch-{}.hnsw", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let graph = Graph::new(PlaneFile::create_with_keys(&path, dims, 32, 40_000, 8).expect("create"));
    let params = InsertParams::default();
    let mut scratches: Vec<SearchScratch> = (0..8).map(|_| SearchScratch::new()).collect();

    let total = 8_000u32;
    let chunk = 1_000usize;
    let mut inserted: Vec<(u32, u32)> = Vec::with_capacity(total as usize);
    for start in (0..total).step_by(chunk) {
        let indices: Vec<u32> = (start..(start + chunk as u32).min(total)).collect();
        let (vectors, keys) = batch_records(&indices, dims);
        let records: Vec<(&[f32], &[u8])> = vectors.iter().zip(&keys).map(|(v, k)| (v.as_slice(), k.as_slice())).collect();
        let outcome = insert_batch(&graph, &params, &records, &mut scratches).expect("batch insert");
        assert_eq!(outcome.ids.len(), indices.len());
        assert!(outcome.rejected.is_empty(), "unexpected rejections: {:?}", outcome.rejected);
        inserted.extend(indices.iter().copied().zip(outcome.ids.iter().copied()));
    }
    assert_eq!(graph.file.id_high_water(), total as u64);
    let mut ids: Vec<u32> = inserted.iter().map(|&(_, id)| id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), total as usize, "batch inserts must hand out distinct ids");

    let mut scratch = SearchScratch::new();
    let mut key = Vec::new();
    let mut misses = 0;
    for &(index, id) in inserted.iter().step_by(37) {
        graph.key_into(id, &mut key).expect("live node");
        assert_eq!(key, index.to_le_bytes(), "id {id} carries the wrong key");
        let query = graph.query(vector_for(index, dims));
        let (results, _) = search(&graph, &query, 10, 256, &mut scratch);
        if !results.iter().any(|&(rid, _)| rid == id) {
            misses += 1;
        }
    }
    assert_eq!(misses, 0, "self-queries missing after a parallel batch build");
    for id in (0..total).step_by(53) {
        if let Some(n) = graph.read_node(id) {
            assert!(n.neighbors.len() <= graph.file.layer0_cap);
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// Record-level faults — a non-finite component, a key the plane cannot store — reject exactly
/// that record (NO_ID in its slot, an entry in `rejected`) and the rest of the batch lands.
#[test]
fn batch_insert_reports_record_faults_by_index_and_inserts_the_rest() {
    let dims = 16;
    let path = std::env::temp_dir().join(format!("hnsw-batch-reject-{}.hnsw", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let graph = Graph::new(PlaneFile::create_with_keys(&path, dims, 16, 1_024, 8).expect("create"));
    let params = InsertParams::default();
    let mut scratches: Vec<SearchScratch> = (0..3).map(|_| SearchScratch::new()).collect();

    let indices: Vec<u32> = (0..64).collect();
    let (mut vectors, keys) = batch_records(&indices, dims);
    vectors[5][3] = f32::NAN;
    vectors[17][0] = f32::INFINITY;
    let long_key = vec![b'k'; 70_000];
    let short_vector = vec![1.0f32; dims - 1];
    let mut records: Vec<(&[f32], &[u8])> = vectors.iter().zip(&keys).map(|(v, k)| (v.as_slice(), k.as_slice())).collect();
    records[40].1 = &long_key;
    records[41].0 = &short_vector;

    let outcome = insert_batch(&graph, &params, &records, &mut scratches).expect("record faults do not fail the batch");
    let mut rejected: Vec<(usize, InsertError)> = outcome.rejected.clone();
    rejected.sort_by_key(|&(i, _)| i);
    assert_eq!(
        rejected,
        vec![
            (5, InsertError::NotFinite),
            (17, InsertError::NotFinite),
            (40, InsertError::KeyUnstorable),
            (41, InsertError::DimensionMismatch),
        ]
    );
    for (i, &id) in outcome.ids.iter().enumerate() {
        let rejected_here = rejected.iter().any(|&(r, _)| r == i);
        assert_eq!(id == NO_ID, rejected_here, "record {i}: id {id}");
    }
    assert_eq!(graph.file.id_high_water(), 60, "a rejected record must not consume an id");
    let mut scratch = SearchScratch::new();
    for (i, &id) in outcome.ids.iter().enumerate().filter(|(_, &id)| id != NO_ID) {
        let (results, _) = search(&graph, &graph.query(vector_for(i as u32, dims)), 5, 64, &mut scratch);
        assert!(results.iter().any(|&(rid, _)| rid == id), "record {i} (id {id}) unreachable");
    }
    let _ = std::fs::remove_file(&path);
}

/// A plane-level fault stops the batch: the failing index and error come back, records already
/// applied stay consistent, and nothing is left in flight when the call returns.
#[test]
fn batch_insert_fails_the_batch_when_the_plane_fills() {
    let dims = 16;
    let path = std::env::temp_dir().join(format!("hnsw-batch-full-{}.hnsw", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let capacity = 100u64;
    let graph = Graph::new(PlaneFile::create_with_keys(&path, dims, 16, capacity, 8).expect("create"));
    let params = InsertParams::default();
    let mut scratches: Vec<SearchScratch> = (0..4).map(|_| SearchScratch::new()).collect();

    let indices: Vec<u32> = (0..160).collect();
    let (vectors, keys) = batch_records(&indices, dims);
    let records: Vec<(&[f32], &[u8])> = vectors.iter().zip(&keys).map(|(v, k)| (v.as_slice(), k.as_slice())).collect();
    let failure = insert_batch(&graph, &params, &records, &mut scratches).expect_err("the plane cannot hold 160 nodes");
    assert_eq!(failure.error, InsertError::Full);
    assert!(failure.index < records.len());
    assert_eq!(graph.file.id_high_water(), capacity, "every slot was used before the batch failed");
    let mut scratch = SearchScratch::new();
    let mut live = 0;
    for id in 0..capacity as u32 {
        if let Some(node) = graph.read_node(id) {
            live += 1;
            assert!(node.neighbors.len() <= graph.file.layer0_cap);
            let mut key = Vec::new();
            graph.key_into(id, &mut key).expect("live node");
            let index = u32::from_le_bytes(key[..4].try_into().unwrap());
            let (results, _) = search(&graph, &graph.query(vector_for(index, dims)), 5, 64, &mut scratch);
            assert!(results.iter().any(|&(rid, _)| rid == id), "node {id} landed but is unreachable");
        }
    }
    assert_eq!(live, capacity, "every allocated slot must hold a finished node once the batch returns");
    let _ = std::fs::remove_file(&path);
}
