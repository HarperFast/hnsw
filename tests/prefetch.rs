//! Kernel page prefetch: the span handed to the kernel, the gate controller, and each backend
//! on the host it runs on.

use hnsw_plane::format::PlaneFile;
use hnsw_plane::insert::{insert, InsertParams};
use hnsw_plane::prefetch::{willneed, willneed_in, Mode, Outcome, PageRange};
use hnsw_plane::search::{search, willneed_hold_after, SearchScratch};
use hnsw_plane::Graph;
use std::path::PathBuf;

fn temp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("hnsw-prefetch-{}-{name}.hnsw", std::process::id()));
    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_file(hnsw_plane::stale_path_for(&p));
    p
}

fn vector_for(i: u32, dims: usize) -> Vec<f32> {
    (0..dims).map(|d| ((i as f32 * 0.31 + d as f32) * 0.7).sin()).collect()
}

fn page() -> usize {
    PageRange::covering(0, 1).len
}

#[test]
fn page_range_covers_and_aligns() {
    let page = page();
    assert!(page >= 4096 && page.is_power_of_two());
    let r = PageRange::covering(page + 100, page + 200);
    assert_eq!(r, PageRange { base: page, len: page });
    let r = PageRange::covering(page - 1, page + 1);
    assert_eq!(r, PageRange { base: 0, len: 2 * page });
    let r = PageRange::covering(3 * page, 4 * page);
    assert_eq!(r, PageRange { base: 3 * page, len: page });
}

#[test]
fn slot_read_span_covers_vector_and_adjacency_but_not_the_key() {
    // 320 B slots with a 64 KiB key capacity: the key area is 200 pages the search never reads
    let path = temp("span");
    let dims = 128;
    let file = PlaneFile::create_with_keys(&path, dims, 32, 10_000, 65_000).expect("create");
    let graph = Graph::new(file);
    let page = page();
    let read_len = graph.file.key_offset();
    for id in [0u32, 1, 9_999] {
        let start = graph.file.slot_ptr(id) as usize;
        let span = graph.slot_read_span(id);
        assert_eq!(span.base % page, 0);
        assert_eq!(span.len % page, 0);
        assert!(span.base <= start && start + read_len <= span.base + span.len, "id {id} not covered");
        assert!(span.len <= 2 * page, "id {id}: {} bytes for a {read_len}-byte read", span.len);
        assert!(span.base + span.len < start + graph.file.slot_size, "id {id}: span reaches into the key field");
    }
}

#[test]
fn slot_read_span_straddles_a_page_boundary_when_slots_are_packed() {
    // 320 B slots pack (12/page would waste 256 B > 128), so every 12.8th slot straddles
    let path = temp("straddle");
    let file = PlaneFile::create(&path, 128, 32, 1_000).expect("create");
    assert_eq!(file.slots_per_page, 0, "expected packed layout for this geometry");
    let graph = Graph::new(file);
    let page = page();
    let straddling = (0..100u32).filter(|&id| graph.slot_read_span(id).len == 2 * page).count();
    let single = (0..100u32).filter(|&id| graph.slot_read_span(id).len == page).count();
    assert!(straddling > 0 && single > 0, "straddling {straddling}, single {single}");
}

#[test]
fn hold_arms_on_a_fault_scale_expansion_and_decays_otherwise() {
    let vb = 128;
    // resident: 30 visits at 0.13 µs
    assert_eq!(willneed_hold_after(0, 4_000, 30, vb), 0);
    assert_eq!(willneed_hold_after(5, 4_000, 30, vb), 4);
    // one 80 µs fault inside an 8-visit expansion
    let armed = willneed_hold_after(0, 81_000, 8, vb);
    assert!(armed >= 8);
    // a resident batch of 128 wide vectors (0.5 µs each at 4096-d) does not trip it
    assert_eq!(willneed_hold_after(1, 64_000, 128, 4096), 0);
    // a prefetch-assisted expansion under pressure still waits one round trip: stays armed
    assert_eq!(willneed_hold_after(armed, 150_000, 30, vb), armed);
    // decays to zero and stays there
    let mut h = armed;
    for _ in 0..armed {
        h = willneed_hold_after(h, 4_000, 30, vb);
    }
    assert_eq!(h, 0);
    assert_eq!(willneed_hold_after(0, 4_000, 30, vb), 0);
}

#[test]
fn every_backend_runs_on_a_real_plane() {
    let path = temp("backends");
    let dims = 64;
    let file = PlaneFile::create(&path, dims, 16, 5_000).expect("create");
    let graph = Graph::new(file);
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();
    for i in 0..2_000u32 {
        insert(&graph, &vector_for(i, dims), &params, &mut scratch).expect("insert");
    }
    let ranges: Vec<PageRange> = (0..2_000u32).step_by(7).map(|id| graph.slot_read_span(id)).collect();
    assert_eq!(willneed_in(Mode::Off, &ranges), Outcome::Off);
    #[cfg(unix)]
    assert_eq!(willneed_in(Mode::PerRange, &ranges), Outcome::Issued);
    #[cfg(not(unix))]
    assert_eq!(willneed_in(Mode::PerRange, &ranges), Outcome::Unusable);
    // vectored: issued where the host allows unprivileged self-advice, unusable elsewhere
    let vectored = willneed_in(Mode::Vectored, &ranges);
    assert!(matches!(vectored, Outcome::Issued | Outcome::Unusable), "{vectored:?}");
    #[cfg(not(target_os = "linux"))]
    assert_eq!(vectored, Outcome::Unusable);
    // the latching entry point never fails, whatever the host supports
    let issued = willneed(&ranges);
    assert_eq!(issued, cfg!(unix));
    assert!(!willneed(&[]));
    // and results are unaffected by any of it
    let q = hnsw_plane::distance::Query::for_plane(&graph.file, vector_for(7, dims));
    let (hits, _) = search(&graph, &q, 5, 64, &mut scratch);
    let exact = graph.distance_to(7, &q).expect("node 7 present");
    assert!(hits[0].1 <= exact + 1e-6, "best hit {:?} worse than the exact match {exact}", hits[0]);
}

#[test]
fn resident_search_issues_no_kernel_prefetch() {
    let path = temp("resident");
    let dims = 32;
    let file = PlaneFile::create(&path, dims, 16, 5_000).expect("create");
    let graph = Graph::new(file);
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();
    for i in 0..3_000u32 {
        insert(&graph, &vector_for(i, dims), &params, &mut scratch).expect("insert");
    }
    let mut batches = 0;
    let mut visits = 0;
    for i in 0..200u32 {
        let q = hnsw_plane::distance::Query::for_plane(&graph.file, vector_for(i * 13, dims));
        let (_, stats) = search(&graph, &q, 5, 64, &mut scratch);
        batches += stats.willneed_batches;
        visits += stats.visits;
    }
    assert!(visits > 0);
    // a freshly written plane is resident; a pre-emption can arm the hold for 16 expansions,
    // so allow a handful of batches over 200 queries but nothing systematic
    assert!(batches < 64, "{batches} kernel prefetch batches on a resident plane");
}
