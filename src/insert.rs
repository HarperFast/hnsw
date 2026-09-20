//! HNSW insert with parity to the JS implementation's optimizeRouting selection
//! (HierarchicalNavigableSmallWorld.ts): candidate i is skipped when an already-added
//! connection reaches it indirectly at comparable cost, and inferior indirect edges are
//! replaced by the new direct route. Stored per-edge distances were dropped from the file
//! format, so neighbor↔neighbor distances are recomputed (int8×int8) on id-match hits only.

use crate::distance::{quantize, Query};
use crate::format::{key_class, NO_ID, NO_UPPER};
use crate::graph::{Graph, KeyError, WriteError};
use crate::search::{beam_descend, search_layer, SearchScratch, SearchStats, DESCENT_EF};
use std::sync::atomic::{AtomicU32, AtomicU8, AtomicUsize, Ordering};

#[derive(Clone, Copy)]
pub struct InsertParams {
    pub m: usize,               // base connection count (JS M, default 16)
    pub ef_construction: usize, // candidate list size
    pub ml: f64,                // level normalization: 1 / ln(M)
    pub optimize_routing: f32,  // JS optimizeRouting, default 0.5; 0 disables
}

impl Default for InsertParams {
    fn default() -> Self {
        InsertParams { m: 16, ef_construction: 200, ml: 1.0 / (16f64).ln(), optimize_routing: 0.5 }
    }
}

/// Deterministic pseudo-random level from the node id (reproducible benchmark builds).
fn level_for(id: u32, ml: f64) -> u8 {
    let mut x = (id as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(0x2545_f491_4f6c_dd1d);
    x ^= x >> 33;
    let unit = (x as f64) / (u64::MAX as f64);
    let level = (-unit.max(f64::MIN_POSITIVE).ln() * ml).floor();
    (level as u8).min(crate::format::MAX_UPPER_LEVELS as u8)
}

/// Remove `to` from `from`'s adjacency at `level` (edge-replacement maintenance).
fn remove_edge(graph: &Graph, from: u32, to: u32, level: u8) {
    if level == 0 {
        let _ = graph.update_neighbors(from, |list| {
            if let Some(pos) = list.iter().position(|&x| x == to) {
                list.remove(pos);
            }
        });
    } else {
        let _ = graph.update_upper_level(from, level, |list| {
            if let Some(pos) = list.iter().position(|&x| x == to) {
                list.remove(pos);
            }
        });
    }
}

/// Neighbor ids of `id` at `level` (level 0 from the slot, upper from the resident map).
fn neighbors_at(graph: &Graph, id: u32, level: u8, buf: &mut Vec<u32>) {
    if level == 0 {
        graph.neighbors_into(id, buf);
    } else {
        graph.upper_neighbors_into(id, level, buf);
    }
}

/// Prune an over-cap adjacency list by evicting the most REDUNDANT far member rather than
/// blindly the farthest: plain closest-keep can strip a node's last in-edge in dense
/// near-duplicate clusters, orphaning it from the graph (observed as unfindable self-queries
/// under concurrent builds). A far member e is redundant when some kept nearer member k has
/// d(e, k) < d(base, e) — searches reaching k still reach e. Bounded: farthest 16 candidates
/// checked against the nearest 16 keepers (~30us per overflow event); falls back to evicting
/// the plain farthest when nothing is provably redundant.
fn prune_with_coverage(graph: &Graph, base: u32, list: &mut Vec<u32>, cap: usize) {
    let mut scored: Vec<(u32, f32)> = list
        .iter()
        .filter_map(|&cand| graph.distance_between(base, cand).map(|d| (cand, d)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    while scored.len() > cap {
        let check_from = scored.len().saturating_sub(16);
        let keepers = &scored[..16.min(check_from)];
        let mut evict = scored.len() - 1; // fallback: farthest
        'hunt: for i in (check_from..scored.len()).rev() {
            let (e, d_base_e) = scored[i];
            for &(k, _) in keepers {
                if k == e {
                    continue;
                }
                if let Some(d_ek) = graph.distance_between(e, k) {
                    if d_ek < d_base_e {
                        evict = i;
                        break 'hunt;
                    }
                }
            }
        }
        scored.remove(evict);
    }
    *list = scored.into_iter().map(|(cand, _)| cand).collect();
}

/// The contended fallback's merge: add `new_id` under the slot lock, displacing the tail once the
/// list is at `cap`. Which neighbor that is is arbitrary — appends push at the tail, so a list is
/// distance-ordered only immediately after a prune — but it must not be `new_id` itself, which is
/// what a push followed by `truncate(cap)` drops. That loss is the systematic one: the edge being
/// added is the in-edge keeping a freshly inserted node reachable from `nid`, and it disappears
/// every time the list is full and the CAS path is contended. Picking a better victim needs
/// distances, which this path deliberately keeps outside the lock.
fn merge_neighbor_capped(graph: &Graph, nid: u32, new_id: u32, cap: usize) {
    let _ = graph.update_neighbors(nid, |list| {
        if list.contains(&new_id) {
            return;
        }
        if list.len() >= cap {
            list.truncate(cap.saturating_sub(1));
        }
        list.push(new_id);
    });
}

/// Add `new_id` to `nid`'s adjacency at `level`, coverage-pruning to `cap` when over. The
/// prune's distance computations (which can major-fault on a cold mapping) run OUTSIDE the
/// slot lock: the list is snapshotted, pruned, and applied with a compare-and-set; after a
/// bounded retry the fallback merges under the lock with a cheap truncation instead.
fn add_reverse_edge(graph: &Graph, nid: u32, new_id: u32, level: u8, cap: usize) {
    if level == 0 {
        for _ in 0..2 {
            let mut snapshot: Vec<u32> = Vec::new();
            if graph.neighbors_into(nid, &mut snapshot).is_none() {
                return;
            }
            if snapshot.contains(&new_id) {
                return;
            }
            let mut next = snapshot.clone();
            next.push(new_id);
            if next.len() > cap {
                prune_with_coverage(graph, nid, &mut next, cap);
            }
            if graph.set_neighbors_if(nid, &snapshot, &next).unwrap_or(false) {
                return;
            }
        }
        // contended twice: merge cheaply under the lock (bounded critical section)
        merge_neighbor_capped(graph, nid, new_id, cap);
    } else {
        let _ = graph.update_upper_level(nid, level, |list| {
            if list.contains(&new_id) {
                return;
            }
            list.push(new_id);
            if list.len() > cap {
                prune_with_coverage(graph, nid, list, cap);
            }
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertError {
    /// max_nodes reached (freeing capacity makes inserts possible again)
    Full,
    /// a slot lock could not be acquired or reclaimed within the wedge bound
    Wedged,
    /// the key overflow arena is exhausted; only a rebuild recovers the space
    KeyArenaFull,
    /// the key exceeds the format's u16 length, or the plane has no key capacity
    KeyUnstorable,
    /// the vector's length is not the plane's `dims`
    DimensionMismatch,
    /// a component is NaN or infinite: such a node would rank first (or nowhere) for every
    /// query, permanently
    NotFinite,
}

impl InsertError {
    /// A record fault is the record's own (skip it and go on). `KeyArenaFull` counts: only keys
    /// past `key_cap` need the arena, so inline-key records keep landing after it fills. `Full`
    /// and `Wedged` are the plane's, and no later insert can succeed until the host acts.
    pub fn is_record_fault(self) -> bool {
        !matches!(self, InsertError::Full | InsertError::Wedged)
    }

    /// Stable machine-readable name, for hosts that dispatch on the fault rather than its text.
    pub fn code(self) -> &'static str {
        match self {
            InsertError::Full => "full",
            InsertError::Wedged => "wedged",
            InsertError::KeyArenaFull => "key-arena-full",
            InsertError::KeyUnstorable => "key-unstorable",
            InsertError::DimensionMismatch => "dimension-mismatch",
            InsertError::NotFinite => "not-finite",
        }
    }

    fn from_code(code: u8) -> Self {
        match code {
            1 => InsertError::Full,
            2 => InsertError::Wedged,
            3 => InsertError::KeyArenaFull,
            4 => InsertError::KeyUnstorable,
            5 => InsertError::DimensionMismatch,
            _ => InsertError::NotFinite,
        }
    }

    fn to_code(self) -> u8 {
        match self {
            InsertError::Full => 1,
            InsertError::Wedged => 2,
            InsertError::KeyArenaFull => 3,
            InsertError::KeyUnstorable => 4,
            InsertError::DimensionMismatch => 5,
            InsertError::NotFinite => 6,
        }
    }
}

fn write_error(error: WriteError) -> InsertError {
    match error {
        WriteError::Wedged => InsertError::Wedged,
        WriteError::KeyArenaFull => InsertError::KeyArenaFull,
        // insert quantizes the vector itself, so the raw-write gate cannot reject it
        WriteError::BadVector(reason) => unreachable!("insert produced an unstorable vector: {reason}"),
    }
}

pub fn insert(
    graph: &Graph,
    vector: &[f32],
    params: &InsertParams,
    scratch: &mut SearchScratch,
) -> Result<u32, InsertError> {
    insert_with_key(graph, vector, &[], params, scratch)
}

/// Insert a vector carrying the host's key bytes (returned verbatim by searches), so a hit
/// resolves to the host's record without a side lookup by node id.
pub fn insert_with_key(
    graph: &Graph,
    vector: &[f32],
    key: &[u8],
    params: &InsertParams,
    scratch: &mut SearchScratch,
) -> Result<u32, InsertError> {
    if vector.len() != graph.file.dims() {
        // before allocate_id, so a rejected insert leaks no id
        return Err(InsertError::DimensionMismatch);
    }
    graph.check_key(key).map_err(|e| match e {
        KeyError::NoKeys | KeyError::TooLong => InsertError::KeyUnstorable,
    })?;
    let stored = quantize(vector, graph.file.quant());
    if !stored.finite {
        return Err(InsertError::NotFinite);
    }
    let (bytes, scale, inv_mag) = (&stored.bytes, stored.scale, stored.inv_mag);
    let id = graph.file.allocate_id();
    if id == NO_ID {
        return Err(InsertError::Full);
    }
    // An overflow key's arena range is settled before the graph is touched: selection removes
    // edges between existing nodes, and an insert abandoned after that cannot put them back,
    // so the one resource the final write could still run out of is taken first — the
    // recycled slot's own range when it still fits, else a fresh reservation.
    let (key_range, fresh_range) = if key.len() > graph.file.key_cap {
        match graph.recycled_key_range(id, key.len()) {
            Some(range) => (Some(range), false),
            None => match graph.file.allocate_key_bytes(key_class(key.len())) {
                Some(range) => (Some(range), true),
                None => {
                    graph.file.free_id(id);
                    return Err(InsertError::KeyArenaFull);
                }
            },
        }
    } else {
        (None, false)
    };
    // `slot_upper` is the entry the published slot names (freed with it); a fresh one is freed here
    let abandon = |published: bool, upper: u32, slot_upper: u32| {
        if published {
            let _ = graph.delete_node(id);
            if upper != slot_upper {
                graph.file.free_upper(upper);
            }
        } else {
            graph.file.free_upper(upper);
            graph.file.free_id(id);
            if let (Some(range), true) = (key_range, fresh_range) {
                graph.file.release_key_bytes(range, key_class(key.len()));
            }
        }
    };
    let level = level_for(id, params.ml);
    // the stored encoding IS the int16 query encoding; reusing it saves a second O(dims) pass
    let query = Query::for_plane_reusing(&graph.file, vector, &stored);
    let layer0_cap = graph.file.layer0_cap;
    let m = params.m;

    let mut stats = SearchStats::default();
    // Upper entry a first-entry claim attempt already published for `id`. Its slot names the
    // index, so the join path must rewrite it in place — freeing an index a live slot names
    // would let another node adopt it mid-traversal.
    let mut published_upper = NO_UPPER;
    let mut published = false;
    let publish_edgeless = |published: &mut bool, published_upper: &mut u32| -> Result<(), InsertError> {
        if *published {
            return Ok(());
        }
        *published_upper =
            if level > 0 { graph.write_upper(&vec![Vec::new(); level as usize]).unwrap_or(NO_UPPER) } else { NO_UPPER };
        if let Err(error) =
            graph.write_node_with_key_range(id, level, bytes, scale, inv_mag, &[], *published_upper, Some(key), key_range)
        {
            abandon(false, *published_upper, NO_UPPER);
            return Err(write_error(error));
        }
        *published = true;
        Ok(())
    };

    // Resolve an entry point to grow from. Every turn makes progress — it claims an empty
    // graph, joins a live entry, or replaces one that is provably gone — so the cap only
    // guards an insert/delete interleaving that keeps clearing the entry under us.
    let mut joined = None;
    for _ in 0..16 {
        let (entry_id, entry_level) = graph.file.entry_point();
        if entry_id == NO_ID {
            publish_edgeless(&mut published, &mut published_upper)?;
            // Claim only from EMPTY, and only the winner returns: a not-worse install would put
            // this edgeless node over a live equal-or-lower-level entry and orphan the graph
            // behind it, and it cannot report losing, which a loser must know to join instead.
            if graph.file.claim_entry_if_empty(id, level as u32) {
                return Ok(id);
            }
            continue; // a racer rooted the graph — join it rather than stand alone
        }
        if let Some(d) = graph.distance_to(entry_id, &query) {
            joined = Some((entry_id, entry_level, d));
            break;
        }
        // The stored entry point is gone (e.g. a mirroring host cleared it without
        // re-electing). Self-promoting an edgeless new node here would orphan the whole
        // existing graph behind an unreachable root — re-elect from the live graph and
        // continue; only a truly empty graph makes this node the first entry.
        graph.reelect_entry_point_replacing(&[], entry_id);
    }
    // An unresolvable entry point is an error the host retries: Ok here would report success
    // for a node no search can reach.
    let Some((entry_id, entry_level, entry_dist)) = joined else {
        // the edgeless node a failed claim left behind is a live-reading slot with no
        // in-edges: a later re-election or repair probe could root the graph at it
        abandon(published, NO_UPPER, published_upper);
        return Err(InsertError::Wedged);
    };
    let top = level.min(entry_level as u8);
    let (mut ep, mut ep_dist) = beam_descend(
        graph,
        &query,
        entry_id,
        entry_dist,
        entry_level,
        top as u32,
        DESCENT_EF,
        scratch,
        &mut stats,
    );

    // Per-level connection lists for the new node, selection-ordered.
    let mut connections: Vec<Vec<(u32, f32)>> = vec![Vec::new(); level as usize + 1];
    let mut nbuf: Vec<u32> = Vec::new();
    let mut neighbors: Vec<(u32, f32)> = Vec::new();

    for l in (0..=top).rev() {
        scratch_begin(graph, scratch);
        search_layer(
            graph,
            &query,
            ep,
            ep_dist,
            params.ef_construction,
            l,
            scratch,
            &mut stats,
            None,
            u64::MAX,
            &mut neighbors,
        );
        neighbors.truncate(m << 1);
        if let Some(&(best, best_d)) = neighbors.first() {
            ep = best;
            ep_dist = best_d;
        }

        // JS optimizeRouting selection over rank-ordered candidates.
        let take_conns = std::mem::take(&mut connections[l as usize]);
        let mut conns = take_conns;
        for (i, &(nid, ndist)) in neighbors.iter().enumerate() {
            if nid == id {
                continue;
            }
            let mut skipping = false;
            let mut replaced: Vec<(u32, u32)> = Vec::new(); // (from, to) edge removals
            if params.optimize_routing > 0.0 {
                let distance_threshold = 1.0 + params.optimize_routing * (1.0 + (0.5 * i as f32) / m as f32);
                neighbors_at(graph, nid, l, &mut nbuf);
                for (i2, &nnid) in nbuf.iter().enumerate() {
                    let neighbor_threshold = 1.0 + params.optimize_routing * (1.0 + (0.5 * i2 as f32) / m as f32);
                    if let Some(&(added_id, added_dist)) = conns.iter().find(|(aid, _)| *aid == nnid) {
                        // recompute the stored neighbor↔neighbor distance (not persisted)
                        let neighbor_distance = graph.distance_between(nid, nnid).unwrap_or(f32::INFINITY);
                        if ndist * distance_threshold > added_dist + neighbor_distance {
                            skipping = true;
                            break; // JS: `if (skipping) break` ends the neighbor scan
                        } else if neighbor_distance * neighbor_threshold > ndist + added_dist {
                            replaced.push((added_id, nid));
                            replaced.push((nid, added_id));
                        }
                        // JS breaks only the inner connections scan; keep scanning neighbors
                    }
                }
                if skipping {
                    continue;
                }
            } else if i >= if l > 0 { m } else { m << 1 } {
                continue;
            }
            conns.push((nid, ndist));
            for (from, to) in replaced {
                remove_edge(graph, from, to, l);
            }
        }
        connections[l as usize] = conns;
    }

    // Write the new node: upper entry first so a reader that sees the node sees its
    // hierarchy; layer-0 list pruned to the file cap (selection order = rank order).
    let upper_idx = if level > 0 {
        let levels: Vec<Vec<u32>> = (1..=level as usize)
            .map(|l| {
                connections
                    .get(l)
                    .map(|c| c.iter().map(|&(nid, _)| nid).collect())
                    .unwrap_or_default()
            })
            .collect();
        if published_upper != NO_UPPER {
            graph.rewrite_upper(published_upper, &levels).map_err(|_| InsertError::Wedged)?;
            published_upper
        } else {
            graph.write_upper(&levels).unwrap_or(NO_UPPER)
        }
    } else {
        NO_UPPER
    };
    let mut l0: Vec<u32> = connections[0].iter().map(|&(nid, _)| nid).collect();
    l0.truncate(layer0_cap);
    if let Err(error) = graph.write_node_with_key_range(id, level, bytes, scale, inv_mag, &l0, upper_idx, Some(key), key_range) {
        abandon(published, upper_idx, published_upper);
        return Err(write_error(error));
    }

    // Reverse edges.
    for (l, conns) in connections.iter().enumerate() {
        let cap = if l == 0 { layer0_cap } else { m << 1 };
        for &(nid, _) in conns {
            add_reverse_edge(graph, nid, id, l as u8, cap);
        }
    }

    if (level as u32) > entry_level {
        // CAS against the observed entry: a concurrent higher-level promotion wins
        graph.file.promote_entry_point(id, level as u32, entry_id);
    }
    Ok(id)
}

/// What a batch produced, in input order: `ids[i]` is the node id of record `i`, or `NO_ID`
/// where the record was rejected (listed in `rejected`, ascending by index) or never started
/// because the batch failed.
#[derive(Debug, Default)]
pub struct BatchOutcome {
    pub ids: Vec<u32>,
    pub rejected: Vec<(usize, InsertError)>,
}

/// A plane fault stopped the batch. `index` is the lowest record that hit one — not a prefix
/// boundary: workers claim records out of order, so `outcome.ids` is the only statement of
/// which records landed. Every claimed record finished before this was returned; nothing is
/// left in flight.
#[derive(Debug)]
pub struct BatchFailure {
    pub index: usize,
    pub error: InsertError,
    pub outcome: BatchOutcome,
}

const NO_FAULT: u8 = 0;
const NO_FAILURE: usize = usize::MAX;

/// Insert `records` (vector, key) with one worker per scratch in `scratches` — the calling
/// thread runs the first worker, the rest are scoped threads — pulling records from a shared
/// counter so a slow region (hub-heavy, cold pages) never idles the other workers. Record
/// faults skip that record; a plane fault stops the dispatch, lets in-flight inserts finish,
/// and fails the batch. Each worker's `SearchScratch` grows to a u32 per allocated id, so the
/// batch's working memory is `scratches.len() x 4 B x id_high_water`.
///
/// Records land in whatever order the workers reach them, so two builds of the same input
/// produce different (equally valid) graphs — DESIGN.md §10 on why that is only a shuffled
/// insertion permutation.
pub fn insert_batch(
    graph: &Graph,
    params: &InsertParams,
    records: &[(&[f32], &[u8])],
    scratches: &mut [SearchScratch],
) -> Result<BatchOutcome, BatchFailure> {
    assert!(!scratches.is_empty(), "insert_batch needs at least one scratch");
    let count = records.len();
    let ids: Vec<AtomicU32> = (0..count).map(|_| AtomicU32::new(NO_ID)).collect();
    let faults: Vec<AtomicU8> = (0..count).map(|_| AtomicU8::new(NO_FAULT)).collect();
    let next = AtomicUsize::new(0);
    let failed = AtomicUsize::new(NO_FAILURE);

    let worker = |scratch: &mut SearchScratch| {
        while failed.load(Ordering::Acquire) == NO_FAILURE {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= count {
                break;
            }
            let (vector, key) = records[i];
            match insert_with_key(graph, vector, key, params, scratch) {
                Ok(id) => ids[i].store(id, Ordering::Relaxed),
                Err(error) => {
                    faults[i].store(error.to_code(), Ordering::Relaxed);
                    if !error.is_record_fault() {
                        failed.fetch_min(i, Ordering::AcqRel);
                        break;
                    }
                }
            }
        }
    };

    let workers = scratches.len().min(count.max(1));
    std::thread::scope(|scope| {
        let (first, rest) = scratches.split_first_mut().expect("non-empty");
        for scratch in rest.iter_mut().take(workers - 1) {
            let worker = &worker;
            // a refused spawn only means fewer workers
            let _ = std::thread::Builder::new().name("hnsw-insert-batch".into()).spawn_scoped(scope, move || worker(scratch));
        }
        worker(first);
    });

    let outcome = BatchOutcome {
        ids: ids.iter().map(|id| id.load(Ordering::Relaxed)).collect(),
        rejected: faults
            .iter()
            .enumerate()
            .filter_map(|(i, fault)| match fault.load(Ordering::Relaxed) {
                NO_FAULT => None,
                code => Some((i, InsertError::from_code(code))).filter(|(_, e)| e.is_record_fault()),
            })
            .collect(),
    };
    match failed.load(Ordering::Acquire) {
        NO_FAILURE => Ok(outcome),
        index => {
            let error = InsertError::from_code(faults[index].load(Ordering::Relaxed));
            Err(BatchFailure { index, error, outcome })
        }
    }
}

#[inline]
fn scratch_begin(graph: &Graph, scratch: &mut SearchScratch) {
    // search_layer assumes a fresh epoch per sweep; SearchScratch::begin is crate-private
    // via this helper to keep the public surface small.
    scratch.begin_public(graph.file.id_high_water());
}

#[cfg(test)]
mod reverse_edge_tests {
    use super::*;
    use crate::PlaneFile;

    /// The contended fallback must still add the edge when the neighbor list is already full —
    /// the one case where a push-then-`truncate(cap)` discards `new_id` rather than a neighbor,
    /// losing the in-edge exactly in the contended-and-full case the fallback exists to serve.
    #[test]
    fn a_contended_merge_into_a_full_neighbor_list_keeps_the_edge_it_adds() {
        let dims = 8;
        let cap = 8usize;
        let path = std::env::temp_dir().join(format!("hnsw-revedge-{}.hnsw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let graph = Graph::new(PlaneFile::create(&path, dims, cap, 4_096).expect("create"));

        let vector = vec![0u8; dims];
        let full: Vec<u32> = (1..=cap as u32).collect();
        graph.write_node_raw(0, 0, &vector, 1.0, 1.0, &full, &[]).expect("seed the full list");
        for &nid in &full {
            graph.write_node_raw(nid, 0, &vector, 1.0, 1.0, &[], &[]).expect("seed a neighbor");
        }
        let newcomer = cap as u32 + 1;
        graph.write_node_raw(newcomer, 0, &vector, 1.0, 1.0, &[], &[]).expect("seed the newcomer");

        merge_neighbor_capped(&graph, 0, newcomer, cap);

        let mut neighbors: Vec<u32> = Vec::new();
        graph.neighbors_into(0, &mut neighbors).expect("node 0 is live");
        assert!(
            neighbors.contains(&newcomer),
            "the contended merge dropped the edge it was adding: {neighbors:?}"
        );
        assert_eq!(neighbors.len(), cap, "the merge must stay within the layer-0 cap");
        let _ = std::fs::remove_file(&path);
    }
}
