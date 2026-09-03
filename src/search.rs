//! Beam search over the plane, zero-copy: per-visit cost is one seqlock-guarded distance
//! against mmap bytes plus primitive heap/visited ops. Visited tracking is an epoch-stamped
//! array; neighbor ids stream through a reusable scratch buffer.

use crate::distance::Query;
use crate::format::{MAX_UPPER_LEVELS, NO_ID};
use crate::graph::Graph;
use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;

#[derive(PartialEq)]
struct Candidate {
    distance: f32,
    id: u32,
}
impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // min-heap by distance via reverse
        other.distance.partial_cmp(&self.distance).unwrap_or(CmpOrdering::Equal)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

#[derive(PartialEq)]
struct Result_ {
    distance: f32,
    id: u32,
}
impl Eq for Result_ {}
impl Ord for Result_ {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // max-heap by distance (worst result on top for eviction)
        self.distance.partial_cmp(&other.distance).unwrap_or(CmpOrdering::Equal)
    }
}
impl PartialOrd for Result_ {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

/// Reusable per-thread search scratch.
pub struct SearchScratch {
    visited: Vec<u32>,
    epoch: u32,
    neighbors: Vec<u32>,
}

impl SearchScratch {
    pub fn new() -> Self {
        SearchScratch { visited: Vec::new(), epoch: 0, neighbors: Vec::new() }
    }

    pub fn begin_public(&mut self, capacity: u64) {
        self.begin(capacity)
    }

    fn begin(&mut self, capacity: u64) {
        if self.visited.len() < capacity as usize {
            self.visited.resize(capacity as usize, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.visited.fill(0);
            self.epoch = 1;
        }
    }

    #[inline]
    fn visit(&mut self, id: u32) -> bool {
        // ids minted by concurrent inserts after begin() can exceed the sizing snapshot;
        // growth is bounded by the id itself, which write paths bound by max_nodes
        if id as usize >= self.visited.len() {
            self.visited.resize(id as usize + 1024, 0);
        }
        let slot = &mut self.visited[id as usize];
        if *slot == self.epoch {
            false
        } else {
            *slot = self.epoch;
            true
        }
    }
}

impl Default for SearchScratch {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SearchStats {
    pub visits: u64,
}

#[inline]
fn bit_allowed(filter: Option<&[u8]>, id: u32) -> bool {
    match filter {
        None => true,
        Some(bits) => {
            let byte = (id >> 3) as usize;
            byte < bits.len() && bits[byte] & (1 << (id & 7)) != 0
        }
    }
}

/// Beam search within one layer, starting from `entry`. Level 0 reads slot adjacency;
/// upper levels read the resident upper map. Returns (id, distance) ascending by distance.
/// Assumes scratch.begin() was called for this query; entry is marked visited here.
///
/// `filter`: optional allow-bitset over node ids (bit i of byte i>>3). Filtered-out nodes
/// are traversed (their edges route) but excluded from results — ACORN-style — with
/// `visit_budget` bounding total visits so a selective filter terminates.
pub fn search_layer(
    graph: &Graph,
    query: &Query,
    entry: u32,
    entry_dist: f32,
    ef: usize,
    level: u8,
    scratch: &mut SearchScratch,
    stats: &mut SearchStats,
    filter: Option<&[u8]>,
    visit_budget: u64,
) -> Vec<(u32, f32)> {
    let mut candidates = BinaryHeap::new();
    let mut results: BinaryHeap<Result_> = BinaryHeap::new();
    scratch.visit(entry);
    candidates.push(Candidate { distance: entry_dist, id: entry });
    if bit_allowed(filter, entry) {
        results.push(Result_ { distance: entry_dist, id: entry });
    }

    // take() the scratch neighbor buffer to sidestep the double-borrow of scratch
    let mut nbuf = std::mem::take(&mut scratch.neighbors);

    while let Some(c) = candidates.pop() {
        let worst = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
        if results.len() >= ef && c.distance > worst {
            break;
        }
        if stats.visits >= visit_budget {
            break;
        }
        if level == 0 {
            if graph.neighbors_into(c.id, &mut nbuf).is_none() {
                continue;
            }
        } else {
            graph.upper_neighbors_into(c.id, level, &mut nbuf);
        }
        for i in 0..nbuf.len() {
            let nid = nbuf[i];
            if (nid as u64) >= graph.file.max_nodes {
                continue; // corrupt/torn neighbor id: skip rather than size allocations by it
            }
            if !scratch.visit(nid) {
                continue;
            }
            if let Some(d) = graph.distance_to(nid, query) {
                stats.visits += 1;
                let worst = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    candidates.push(Candidate { distance: d, id: nid });
                    if bit_allowed(filter, nid) {
                        results.push(Result_ { distance: d, id: nid });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
    }
    scratch.neighbors = nbuf;

    let mut out: Vec<(u32, f32)> = results.into_iter().map(|r| (r.id, r.distance)).collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out
}

/// Beam width carried through every upper level of the descent.
///
/// A width-1 descent is plain hill climbing: it halts at the first node no neighbor improves
/// on. On a clustered corpus that local minimum can sit in the wrong basin, and layer-0
/// adjacency is intra-basin, so the layer-0 beam has no uphill edge to leave it — the query's
/// true neighborhood is then unreachable at any ef.
pub const DESCENT_EF: usize = 16;

/// Beam descent through upper layers from `from_level` down to `to_level` (exclusive).
/// Each level runs a width-`ef` beam seeded by the level above's best; that level's best in
/// turn seeds the next. Returns the improved entry for the caller's layer-`to_level` search.
///
/// The scratch epoch is rolled per level: a node reachable at several levels must be
/// expandable at each of them, so visited marks must not carry across the boundary.
pub fn beam_descend(
    graph: &Graph,
    query: &Query,
    mut current: u32,
    mut current_dist: f32,
    from_level: u32,
    to_level: u32,
    ef: usize,
    scratch: &mut SearchScratch,
    stats: &mut SearchStats,
) -> (u32, f32) {
    let mut level = from_level.min(MAX_UPPER_LEVELS as u32);
    while level > to_level {
        scratch.begin(graph.file.id_high_water());
        let found =
            search_layer(graph, query, current, current_dist, ef, level as u8, scratch, stats, None, u64::MAX);
        // search_layer admits the entry itself, so `first` is never worse than the entry
        if let Some(&(id, d)) = found.first() {
            current = id;
            current_dist = d;
        }
        level -= 1;
    }
    (current, current_dist)
}

/// Slots a read-side repair may probe when the previous-entry hint is dead too. Bounded so a
/// search never pays the write path's O(high-water) re-election scan.
const REPAIR_PROBE_LIMIT: u32 = 1024;

/// Resolve a live entry point for a read, repairing a dead one in place.
///
/// A search that finds the header naming a deleted or sanitized node returns EMPTY, and on a
/// read-mostly table nothing ever repairs it: write-path re-election only runs on delete, and
/// a slot a reader sanitized after its writer died had no delete at all.
///
/// The candidate is the O(1) previous-entry hint, then a probe capped at `REPAIR_PROBE_LIMIT` —
/// the hint is a single slot and can be dead itself. The cap is what keeps a read off the write
/// path's O(high-water) scan on the pool thread every search shares, and the repair publishes,
/// so only the first search after a wedge pays even the probe.
fn resolve_entry(graph: &Graph, query: &Query, stats: &mut SearchStats) -> Option<(u32, u32, f32)> {
    let (entry_id, entry_level) = graph.file.entry_point();
    if entry_id != NO_ID {
        if let Some(d) = graph.distance_to(entry_id, query) {
            stats.visits += 1;
            return Some((entry_id, entry_level, d));
        }
    }
    let hint = graph.file.previous_entry_point();
    let candidate = (hint != NO_ID && hint != entry_id)
        .then(|| graph.node_level(hint).map(|level| (hint, level)))
        .flatten()
        .or_else(|| graph.probe_for_entry(REPAIR_PROBE_LIMIT, entry_id));
    let (id, level) = candidate?;
    let d = graph.distance_to(id, query)?;
    stats.visits += 1;
    // Strict on the entry we observed dead, not a not-worse install: a live level-0 root claimed
    // since the read above must win, or it is orphaned with nothing pointing at it.
    graph.file.replace_entry_if(entry_id, id, level as u32);
    Some((id, level as u32, d))
}

/// Full search: beam descent through upper layers, then the layer-0 beam at `ef`.
pub fn search(
    graph: &Graph,
    query: &Query,
    k: usize,
    ef: usize,
    scratch: &mut SearchScratch,
) -> (Vec<(u32, f32)>, SearchStats) {
    let mut stats = SearchStats { visits: 0 };
    let Some((entry_id, entry_level, entry_dist)) = resolve_entry(graph, query, &mut stats) else {
        return (Vec::new(), stats);
    };
    let (ep, ep_dist) =
        beam_descend(graph, query, entry_id, entry_dist, entry_level, 0, DESCENT_EF, scratch, &mut stats);
    scratch.begin(graph.file.id_high_water());
    let mut out = search_layer(graph, query, ep, ep_dist, ef, 0, scratch, &mut stats, None, u64::MAX);
    out.truncate(k);
    (out, stats)
}

/// Full search with an optional allow-bitset filter. `filter_expansion` multiplies ef into
/// the visit budget when a filter is present (matching the JS filterExpansion semantics).
pub fn search_filtered(
    graph: &Graph,
    query: &Query,
    k: usize,
    ef: usize,
    filter: Option<&[u8]>,
    filter_expansion: usize,
    scratch: &mut SearchScratch,
) -> (Vec<(u32, f32)>, SearchStats) {
    let mut stats = SearchStats { visits: 0 };
    let Some((entry_id, entry_level, entry_dist)) = resolve_entry(graph, query, &mut stats) else {
        return (Vec::new(), stats);
    };
    let (ep, ep_dist) =
        beam_descend(graph, query, entry_id, entry_dist, entry_level, 0, DESCENT_EF, scratch, &mut stats);
    scratch.begin_public(graph.file.id_high_water());
    // offset by the descent's own visits so the budget bounds layer 0, as documented, rather
    // than layer 0 minus whatever the descent already spent
    let budget = if filter.is_some() {
        stats.visits.saturating_add((ef * filter_expansion) as u64)
    } else {
        u64::MAX
    };
    let mut out = search_layer(graph, query, ep, ep_dist, ef, 0, scratch, &mut stats, filter, budget);
    out.truncate(k);
    (out, stats)
}

/// Pipelined predicate filtering: candidate ids are batched to an external evaluator (the
/// NAPI layer wires this to a JS ThreadsafeFunction) while traversal continues expanding —
/// the search thread never blocks on the evaluator until the beam itself is done. Verdicts
/// steer result admission only; routing uses pure distance order, bounded by the visit
/// budget, so a slow or saturated JS loop degrades speculative overshoot, not correctness.
pub struct PredicatePipe {
    /// Sends one batch of ids for evaluation. Must not block. Returns whether the batch was
    /// actually handed off: a refused enqueue never produces a verdict, so counting it as
    /// outstanding would make the tail drain wait out its whole deadline for an answer that
    /// cannot arrive.
    pub dispatch: Box<dyn FnMut(Vec<u32>) -> bool + Send>,
    /// Receives (ids, verdicts) pairs; verdicts[i] != 0 admits ids[i].
    pub rx: std::sync::mpsc::Receiver<(Vec<u32>, Vec<u8>)>,
}

const PREDICATE_BATCH: usize = 64;
const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Full search with a pipelined predicate filter (upper-layer descent is unfiltered, as in
/// the JS implementation — predicates gate results, not routing). `visit_budget` is the
/// absolute layer-0 visit cap: hosts pass their own resolved budget directly, since a
/// multiplier-of-ef encoding cannot express a budget below ef.
pub fn search_predicated(
    graph: &Graph,
    query: &Query,
    k: usize,
    ef: usize,
    pipe: &mut PredicatePipe,
    visit_budget: u64,
    scratch: &mut SearchScratch,
) -> (Vec<(u32, f32)>, SearchStats) {
    let mut stats = SearchStats { visits: 0 };
    let Some((entry_id, entry_level, entry_dist)) = resolve_entry(graph, query, &mut stats) else {
        return (Vec::new(), stats);
    };
    let (ep, ep_dist) =
        beam_descend(graph, query, entry_id, entry_dist, entry_level, 0, DESCENT_EF, scratch, &mut stats);
    scratch.begin_public(graph.file.id_high_water());
    // absolute cap for the layer-0 loop below: the caller's budget counted from where the
    // descent left off, so a wider descent does not silently shrink it
    let layer0_budget = stats.visits.saturating_add(visit_budget);

    use std::collections::HashMap;
    let mut verdicts: HashMap<u32, bool> = HashMap::new();
    let mut speculative: Vec<(u32, f32)> = Vec::new(); // awaiting verdicts
    let mut batch: Vec<u32> = Vec::new();
    let mut outstanding = 0usize;

    let mut candidates = BinaryHeap::new();
    let mut results: BinaryHeap<Result_> = BinaryHeap::new();
    scratch.visit(ep);
    candidates.push(Candidate { distance: ep_dist, id: ep });
    speculative.push((ep, ep_dist));
    batch.push(ep);

    let mut nbuf = std::mem::take(&mut scratch.neighbors);

    // guarded on `outstanding` rather than draining until the channel is empty: with a blocking
    // receive the unguarded form pays another full timeout after the last verdict lands, on every
    // filtered query
    macro_rules! drain {
        ($recv:expr) => {
            while outstanding > 0 {
                let Ok((ids, flags)) = $recv else { break };
                outstanding -= 1;
                for (i, id) in ids.iter().enumerate() {
                    verdicts.insert(*id, flags.get(i).copied().unwrap_or(0) != 0);
                }
            }
        };
    }

    loop {
        // non-blocking verdict intake each iteration
        drain!(pipe.rx.try_recv());
        if !verdicts.is_empty() && !speculative.is_empty() {
            speculative.retain(|&(id, d)| match verdicts.get(&id) {
                Some(true) => {
                    results.push(Result_ { distance: d, id });
                    if results.len() > ef {
                        results.pop();
                    }
                    false
                }
                Some(false) => false,
                None => true,
            });
        }

        let Some(c) = candidates.pop() else { break };
        let worst = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
        if results.len() >= ef && c.distance > worst {
            break;
        }
        if stats.visits >= layer0_budget {
            break;
        }
        if graph.neighbors_into(c.id, &mut nbuf).is_none() {
            continue;
        }
        for i in 0..nbuf.len() {
            let nid = nbuf[i];
            if (nid as u64) >= graph.file.max_nodes {
                continue; // corrupt/torn neighbor id: skip rather than size allocations by it
            }
            if !scratch.visit(nid) {
                continue;
            }
            if let Some(d) = graph.distance_to(nid, query) {
                stats.visits += 1;
                let worst = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    candidates.push(Candidate { distance: d, id: nid });
                    speculative.push((nid, d));
                    batch.push(nid);
                    if batch.len() >= PREDICATE_BATCH && (pipe.dispatch)(std::mem::take(&mut batch)) {
                        outstanding += 1;
                    }
                }
            }
        }
    }
    scratch.neighbors = nbuf;

    // flush the tail batch and block-drain what's still in flight
    if !batch.is_empty() && (pipe.dispatch)(std::mem::take(&mut batch)) {
        outstanding += 1;
    }
    let deadline = std::time::Instant::now() + DRAIN_TIMEOUT;
    while outstanding > 0 && std::time::Instant::now() < deadline {
        drain!(pipe.rx.recv_timeout(std::time::Duration::from_millis(50)));
    }
    speculative.retain(|&(id, d)| {
        if verdicts.get(&id).copied().unwrap_or(false) {
            results.push(Result_ { distance: d, id });
            if results.len() > ef {
                results.pop();
            }
        }
        false
    });

    let mut out: Vec<(u32, f32)> = results.into_iter().map(|r| (r.id, r.distance)).collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out.truncate(k);
    (out, stats)
}

#[cfg(test)]
mod descent_tests {
    use super::*;
    use crate::format::UPPER_CAP;
    use crate::insert::{insert, InsertParams};
    use crate::PlaneFile;

    /// The descent takes no caller-supplied visit budget, so what bounds it under tied distances
    /// is `search_layer`'s strict `d < worst`: once the result set is full a tied candidate is
    /// never pushed, and the candidate heap drains after at most `ef` expansions. Relaxing that
    /// to `<=` would make a zero query — every cosine distance exactly 1.0, and any caller can
    /// send one — walk the whole upper component instead.
    #[test]
    fn a_tied_distance_descent_stops_at_its_visit_cap() {
        let dims = 16;
        let n = 6_000u32;
        let path = std::env::temp_dir().join(format!("hnsw-tied-{}.hnsw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let graph = Graph::new(PlaneFile::create(&path, dims, 16, n as u64 + 1024).expect("create"));
        let params = InsertParams::default();
        let mut scratch = SearchScratch::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dims).map(|d| ((i as f32 * 0.31 + d as f32) * 0.7).sin()).collect();
            insert(&graph, &v, &params, &mut scratch).expect("insert");
        }

        let (entry_id, entry_level) = graph.file.entry_point();
        assert!(entry_level >= 1, "precondition: the graph has an upper level to descend");
        let query = Query::new(vec![0.0f32; dims]);
        let entry_dist = graph.distance_to(entry_id, &query).expect("the entry point is live");
        assert_eq!(entry_dist, 1.0, "precondition: a zero query ties every stored vector at 1.0");

        // one level only, so the bound under test is the per-level one rather than a sum
        let ef = 2usize;
        let mut stats = SearchStats { visits: 0 };
        beam_descend(&graph, &query, entry_id, entry_dist, 1, 0, ef, &mut scratch, &mut stats);

        // ef expansions at the maximum upper degree
        let ceiling = (ef * UPPER_CAP) as u64;
        assert!(
            stats.visits <= ceiling,
            "the tied-distance descent visited {} nodes at level 1 against a ceiling of {ceiling} — \
level 1 holds roughly {} nodes, and a beam that pushed tied candidates would walk all of them",
            stats.visits,
            n / 16
        );
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod predicate_tests {
    use super::*;
    use crate::insert::{insert, InsertParams};
    use crate::PlaneFile;

    #[test]
    fn pipelined_predicate_filters_results() {
        let dims = 32;
        let path = std::env::temp_dir().join(format!("hnsw-pred-{}.hnsw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let file = PlaneFile::create(&path, dims, 16, 4_096).expect("create");
        let graph = Graph::new(file);
        let params = InsertParams::default();
        let mut scratch = SearchScratch::new();
        for i in 0..1_000u32 {
            let v: Vec<f32> = (0..dims).map(|d| ((i as f32 * 0.31 + d as f32) * 0.7).sin()).collect();
            insert(&graph, &v, &params, &mut scratch).unwrap();
        }

        // evaluator thread: admit even ids only, answering over a channel like the TSFN does
        let (req_tx, req_rx) = std::sync::mpsc::channel::<Vec<u32>>();
        let (res_tx, res_rx) = std::sync::mpsc::channel::<(Vec<u32>, Vec<u8>)>();
        let worker = std::thread::spawn(move || {
            while let Ok(ids) = req_rx.recv() {
                let verdicts: Vec<u8> = ids.iter().map(|id| (id % 2 == 0) as u8).collect();
                if res_tx.send((ids, verdicts)).is_err() {
                    break;
                }
            }
        });

        let mut pipe = PredicatePipe {
            dispatch: Box::new(move |ids| req_tx.send(ids).is_ok()),
            rx: res_rx,
        };
        let q: Vec<f32> = (0..dims).map(|d| ((41.0f32 * 0.31 + d as f32) * 0.7).sin()).collect();
        let (hits, _) =
            search_predicated(&graph, &Query::new(q), 10, 64, &mut pipe, 64 * 24, &mut scratch);
        assert!(!hits.is_empty());
        for (id, _) in &hits {
            assert_eq!(id % 2, 0, "odd id {id} leaked through the predicate");
        }
        drop(pipe);
        worker.join().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    /// The tail drain must stop receiving the moment the last verdict lands. Draining until the
    /// channel reports empty sits out another full `recv_timeout` after `outstanding` reaches
    /// zero — 50 ms added to every filtered query, against a sub-millisecond search. Measured
    /// from the evaluator's last send so the search's own cost is not in the number, and over
    /// the best of several queries so scheduler noise on one of them cannot pass for the extra
    /// receive, which every query would pay.
    #[test]
    fn a_predicated_search_returns_as_soon_as_the_last_verdict_lands() {
        let dims = 32;
        let path = std::env::temp_dir().join(format!("hnsw-preddrain-{}.hnsw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let graph = Graph::new(PlaneFile::create(&path, dims, 16, 4_096).expect("create"));
        let params = InsertParams::default();
        let mut scratch = SearchScratch::new();
        for i in 0..1_000u32 {
            let v: Vec<f32> = (0..dims).map(|d| ((i as f32 * 0.31 + d as f32) * 0.7).sin()).collect();
            insert(&graph, &v, &params, &mut scratch).unwrap();
        }

        let (req_tx, req_rx) = std::sync::mpsc::channel::<Vec<u32>>();
        let (res_tx, res_rx) = std::sync::mpsc::channel::<(Vec<u32>, Vec<u8>)>();
        // stamped before the send, so the search can never observe a verdict newer than the stamp
        let last_send = std::sync::Arc::new(std::sync::Mutex::new(None::<std::time::Instant>));
        let stamps = last_send.clone();
        let worker = std::thread::spawn(move || {
            while let Ok(ids) = req_rx.recv() {
                let verdicts = vec![1u8; ids.len()];
                *stamps.lock().unwrap() = Some(std::time::Instant::now());
                if res_tx.send((ids, verdicts)).is_err() {
                    break;
                }
            }
        });

        let mut pipe = PredicatePipe {
            dispatch: Box::new(move |ids| req_tx.send(ids).is_ok()),
            rx: res_rx,
        };
        let q: Vec<f32> = (0..dims).map(|d| ((41.0f32 * 0.31 + d as f32) * 0.7).sin()).collect();
        let mut best = std::time::Duration::MAX;
        for _ in 0..5 {
            let (hits, _) = search_predicated(
                &graph,
                &Query::new(q.clone()),
                10,
                64,
                &mut pipe,
                64 * 24,
                &mut scratch,
            );
            let tail = last_send.lock().unwrap().expect("the evaluator answered a batch").elapsed();
            assert!(!hits.is_empty(), "precondition: an admitting predicate returns results");
            best = best.min(tail);
        }
        assert!(
            best < std::time::Duration::from_millis(25),
            "the drain sat {best:?} past the last verdict on every query instead of returning on it"
        );
        drop(pipe);
        worker.join().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    /// A refused enqueue never answers. Counting it outstanding makes the tail drain wait out
    /// its whole `DRAIN_TIMEOUT` for a verdict that cannot arrive — which is exactly the state
    /// a closing environment puts every in-flight filtered query in, so teardown pays five
    /// seconds per query instead of returning on the batches that did land.
    #[test]
    fn a_refused_predicate_enqueue_does_not_hold_the_drain() {
        let dims = 32;
        let path = std::env::temp_dir().join(format!("hnsw-refused-{}.hnsw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let graph = Graph::new(PlaneFile::create(&path, dims, 16, 4_096).expect("create"));
        let params = InsertParams::default();
        let mut scratch = SearchScratch::new();
        for i in 0..1_000u32 {
            let v: Vec<f32> = (0..dims).map(|d| ((i as f32 * 0.31 + d as f32) * 0.7).sin()).collect();
            insert(&graph, &v, &params, &mut scratch).unwrap();
        }

        // the sender stays alive for the whole search, so a drain that believes a batch is
        // outstanding blocks on the deadline rather than on a disconnected channel
        let (tx, rx) = std::sync::mpsc::channel::<(Vec<u32>, Vec<u8>)>();
        let mut pipe = PredicatePipe { dispatch: Box::new(|_ids| false), rx };
        let q: Vec<f32> = (0..dims).map(|d| ((41.0f32 * 0.31 + d as f32) * 0.7).sin()).collect();
        let started = std::time::Instant::now();
        let (hits, _) =
            search_predicated(&graph, &Query::new(q), 10, 64, &mut pipe, 64 * 24, &mut scratch);
        let elapsed = started.elapsed();
        drop(tx);
        assert!(hits.is_empty(), "no verdict can arrive for a refused batch, so nothing may be admitted");
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "the search waited {elapsed:?} on batches that were never enqueued (deadline is {DRAIN_TIMEOUT:?})"
        );
        let _ = std::fs::remove_file(&path);
    }
}
