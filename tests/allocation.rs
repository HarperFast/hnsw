//! The module header calls the search path zero-copy with a reusable scratch. This binary is the
//! only place that can hold a counting global allocator, so it owns the one test that checks it:
//! per-query allocations must not scale with the graph's height. A descent that allocated per
//! upper level would pass every other test in the suite.

use hnsw_plane::distance::Query;
use hnsw_plane::insert::{insert, InsertParams};
use hnsw_plane::search::{search, SearchScratch};
use hnsw_plane::{Graph, PlaneFile};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// A global allocator sees every thread in the binary, the test harness's own included, so a
    /// process-wide flag would fold their allocations into the measurement. Scoping it per thread
    /// keeps the count to the one doing the searching. `const` init on a `Cell` so that reading it
    /// neither lazily initializes nor registers a destructor — either would allocate, and
    /// allocating inside the allocator recurses.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

#[inline]
fn counting_here() -> bool {
    COUNTING.try_with(|c| c.get()).unwrap_or(false)
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if counting_here() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if counting_here() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocations attributable to one `search` on a warmed scratch, averaged over `queries`.
fn allocations_per_search(graph: &Graph, dims: usize, scratch: &mut SearchScratch, queries: usize) -> f64 {
    let vector = |i: usize| -> Vec<f32> {
        (0..dims).map(|d| ((i as f32 * 0.11 + d as f32) * 0.9).sin()).collect()
    };
    for i in 0..8 {
        let _ = search(graph, &Query::new(vector(i)), 10, 64, scratch);
    }
    let queries_prepared: Vec<Query> = (0..queries).map(|i| Query::new(vector(i + 100))).collect();

    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.with(|c| c.set(true));
    for query in &queries_prepared {
        let (hits, _) = search(graph, query, 10, 64, scratch);
        std::hint::black_box(hits);
    }
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.load(Ordering::Relaxed) as f64 / queries as f64
}

fn build(path: &std::path::Path, n: u32, dims: usize) -> Graph {
    let _ = std::fs::remove_file(path);
    let graph = Graph::new(PlaneFile::create(path, dims, 16, n as u64 + 1024).expect("create"));
    // only the graph's height matters here, not its quality
    let params = InsertParams { ef_construction: 24, ..InsertParams::default() };
    let mut scratch = SearchScratch::new();
    for i in 0..n {
        let v: Vec<f32> = (0..dims).map(|d| ((i as f32 * 0.31 + d as f32) * 0.7).sin()).collect();
        insert(&graph, &v, &params, &mut scratch).expect("insert");
    }
    graph
}

/// A taller graph means more upper levels to descend. If the descent allocated per level, the
/// per-query count would rise with height; it must not.
#[test]
fn a_search_does_not_allocate_per_upper_level() {
    let dims = 16;
    let pid = std::process::id();
    let shallow_path = std::env::temp_dir().join(format!("hnsw-alloc-shallow-{pid}.hnsw"));
    let tall_path = std::env::temp_dir().join(format!("hnsw-alloc-tall-{pid}.hnsw"));

    let shallow = build(&shallow_path, 2_000, dims);
    let tall = build(&tall_path, 60_000, dims);
    let shallow_levels = shallow.file.entry_point().1;
    let tall_levels = tall.file.entry_point().1;
    assert!(
        tall_levels >= shallow_levels + 2,
        "precondition: the two graphs must differ in height ({shallow_levels} vs {tall_levels}). \
This is derived from `ml` and `level_for`; if either was tuned, re-pick the two node counts \
rather than reading this as an allocation regression"
    );

    let mut scratch = SearchScratch::new();
    let shallow_allocs = allocations_per_search(&shallow, dims, &mut scratch, 200);
    let tall_allocs = allocations_per_search(&tall, dims, &mut scratch, 200);

    assert!(
        tall_allocs <= shallow_allocs + 0.5,
        "search allocates per upper level: {shallow_allocs:.2}/query at {shallow_levels} levels \
vs {tall_allocs:.2}/query at {tall_levels}"
    );
    assert!(
        tall_allocs <= 2.0,
        "search allocates {tall_allocs:.2} times per query on a warmed scratch; the result vector \
should be the only one"
    );

    drop(shallow);
    drop(tall);
    let _ = std::fs::remove_file(&shallow_path);
    let _ = std::fs::remove_file(&tall_path);
}
