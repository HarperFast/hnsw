//! Host keys stored in slots: inline when they fit `key_cap`, otherwise in the overflow
//! arena; returned by searches and predicate batches; persisted across reopen.

use hnsw_plane::distance::Query;
use hnsw_plane::insert::{insert, insert_with_key, InsertError, InsertParams};
use hnsw_plane::search::{gather_keys, search, SearchScratch};
use hnsw_plane::{Graph, PlaneFile};
use std::path::PathBuf;

fn vector_for(i: u32, dims: usize) -> Vec<f32> {
    (0..dims).map(|d| ((i as f32 * 0.37 + d as f32 * 1.13).sin() * 0.1) + if d % 7 == (i as usize) % 7 { 1.0 } else { 0.0 }).collect()
}

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("hnsw-keys-{name}-{}.hnsw", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

fn key_for(i: u32) -> Vec<u8> {
    if i % 10 == 0 {
        format!("record-{i}-with-a-key-longer-than-the-inline-capacity").into_bytes()
    } else {
        format!("r{i}").into_bytes()
    }
}

fn key_of(graph: &Graph, id: u32) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    graph.key_into(id, &mut out).map(|_| out)
}

#[test]
fn inline_and_overflow_keys_round_trip_and_survive_reopen() {
    let dims = 16;
    let path = temp("roundtrip");
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();
    let mut ids = Vec::new();
    {
        let graph = Graph::new(PlaneFile::create_with_keys(&path, dims, 16, 1_024, 12).expect("create"));
        assert_eq!(graph.file.key_cap, 12);
        for i in 0..200u32 {
            ids.push(insert_with_key(&graph, &vector_for(i, dims), &key_for(i), &params, &mut scratch).unwrap());
        }
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(key_of(&graph, id).as_deref(), Some(key_for(i as u32).as_slice()), "key of node {id}");
        }
        // every hit carries its node's key, through the same path searches use (the corpus has
        // near-duplicates, so the top hit is not necessarily the query's own node)
        let (hits, _) = search(&graph, &graph.query(vector_for(30, dims)), 20, 64, &mut scratch);
        let hit_ids: Vec<u32> = hits.iter().map(|&(id, _)| id).collect();
        let (keys, ends) = gather_keys(&graph, &hit_ids);
        let mut start = 0usize;
        for (i, &id) in hit_ids.iter().enumerate() {
            let index = ids.iter().position(|&x| x == id).unwrap() as u32;
            assert_eq!(&keys[start..ends[i] as usize], key_for(index).as_slice(), "key of hit {id}");
            start = ends[i] as usize;
        }
        assert!(hit_ids.iter().any(|&id| ids.iter().position(|&x| x == id).unwrap() % 10 == 0), "an overflow key was among the hits");
        graph.file.msync().unwrap();
    }
    let graph = Graph::new(PlaneFile::open(&path).expect("reopen"));
    assert_eq!(graph.file.key_cap, 12);
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(key_of(&graph, id).as_deref(), Some(key_for(i as u32).as_slice()), "key of node {id} after reopen");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_reused_slot_reports_its_new_key_and_a_deleted_one_none() {
    let dims = 16;
    let path = temp("reuse");
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();
    let graph = Graph::new(PlaneFile::create_with_keys(&path, dims, 16, 256, 8).expect("create"));
    let a = insert_with_key(&graph, &vector_for(1, dims), b"first-key-overflowing", &params, &mut scratch).unwrap();
    let _b = insert_with_key(&graph, &vector_for(2, dims), b"b", &params, &mut scratch).unwrap();
    graph.delete_node(a).unwrap();
    assert_eq!(key_of(&graph, a), None, "a deleted node has no key");
    let (keys, ends) = gather_keys(&graph, &[a]);
    assert_eq!((keys.len(), ends.as_slice()), (0, &[0u32][..]), "gather_keys yields an empty key for a gone node");
    let c = insert_with_key(&graph, &vector_for(3, dims), b"c", &params, &mut scratch).unwrap();
    assert_eq!(c, a, "the freed id is reused");
    assert_eq!(key_of(&graph, c).as_deref(), Some(&b"c"[..]), "the reused slot carries the new key");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn keys_are_refused_where_they_cannot_be_stored() {
    let dims = 16;
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();

    let path = temp("nokeys");
    let graph = Graph::new(PlaneFile::create(&path, dims, 16, 256).expect("create"));
    assert_eq!(graph.file.key_cap, 0);
    assert_eq!(
        insert_with_key(&graph, &vector_for(1, dims), b"k", &params, &mut scratch),
        Err(InsertError::KeyUnstorable),
        "a plane without key capacity refuses keys"
    );
    assert!(insert(&graph, &vector_for(1, dims), &params, &mut scratch).is_ok(), "and still takes keyless inserts");
    let (keys, ends) = gather_keys(&graph, &[0]);
    assert!(keys.is_empty() && ends.is_empty(), "no key arrays without key capacity");
    let _ = std::fs::remove_file(&path);

    assert!(PlaneFile::create_with_keys(&temp("badcap"), dims, 16, 256, 4).is_err(), "keyCap below 8 is refused");

    // arena: 128 bytes per node (the default at keyCap 8); 8 nodes -> 1024 bytes; 200-byte keys
    // reserve 256 and fit 4 times
    let path = temp("arena");
    let graph = Graph::new(PlaneFile::create_with_keys(&path, dims, 16, 8, 8).expect("create"));
    let long = vec![b'x'; 200];
    for i in 0..4u32 {
        insert_with_key(&graph, &vector_for(i, dims), &long, &params, &mut scratch).expect("arena has room");
    }
    // a refused insert frees the id it took: four refusals recycle one id among them, and the
    // next success takes that id back
    let high_water = graph.file.id_high_water();
    for _ in 0..4 {
        assert_eq!(
            insert_with_key(&graph, &vector_for(5, dims), &long, &params, &mut scratch),
            Err(InsertError::KeyArenaFull)
        );
    }
    assert_eq!(graph.file.id_high_water(), high_water + 1, "refused inserts recycle one id among them");
    let short = insert_with_key(&graph, &vector_for(5, dims), b"short", &params, &mut scratch).expect("inline keys still fit");
    assert_eq!(short as u64, high_water);
    assert_eq!(graph.file.id_high_water(), high_water + 1);
    assert_eq!(
        insert_with_key(&graph, &vector_for(6, dims), &vec![b'y'; 70_000], &params, &mut scratch),
        Err(InsertError::KeyUnstorable),
        "a key past the u16 length is refused"
    );
    let _ = std::fs::remove_file(&path);

    // an explicit arena reservation: 32 nodes x 192 B holds twenty-four 256-byte ranges
    let path = temp("arena-sized");
    let graph = Graph::new(PlaneFile::create_with_key_arena(&path, dims, 16, 32, 8, 192).expect("create"));
    for i in 0..24u32 {
        insert_with_key(&graph, &vector_for(i, dims), &long, &params, &mut scratch).expect("sized arena has room");
    }
    assert_eq!(insert_with_key(&graph, &vector_for(24, dims), &long, &params, &mut scratch), Err(InsertError::KeyArenaFull));
    assert!(PlaneFile::create_with_key_arena(&temp("arena-small"), dims, 16, 32, 8, 32).is_err(), "below 64 is refused");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn rewriting_a_node_reuses_its_arena_range() {
    let dims = 16;
    let path = temp("rewrite");
    // arena: 8 nodes x 128 B = 1024 B; a 200-byte key reserves 256 and fits four times
    let graph = Graph::new(PlaneFile::create_with_keys(&path, dims, 16, 8, 8).expect("create"));
    let (bytes, scale, inv_mag) = hnsw_plane::distance::quantize_int8(&vector_for(1, dims));
    let long = vec![b'x'; 200];
    for round in 0..20u32 {
        let key: Vec<u8> = long.iter().map(|b| b + (round % 3) as u8).collect();
        graph.write_node_raw_with_key(0, 0, &bytes, scale, inv_mag, &[], &[], Some(&key)).expect("rewrite reuses the range");
        assert_eq!(key_of(&graph, 0).as_deref(), Some(key.as_slice()));
    }
    // growth allocates once; a shrink within the range's 64-byte class keeps it, so regrowing is free
    let longer = vec![b'y'; 300];
    graph.write_node_raw_with_key(0, 0, &bytes, scale, inv_mag, &[], &[], Some(&longer)).expect("growth");
    let arena_after_growth = graph.file.key_arena_high_water();
    graph.write_node_raw_with_key(0, 0, &bytes, scale, inv_mag, &[], &[], Some(&longer[..270])).expect("shrink");
    graph.write_node_raw_with_key(0, 0, &bytes, scale, inv_mag, &[], &[], Some(&longer)).expect("regrow within the kept range");
    assert_eq!(graph.file.key_arena_high_water(), arena_after_growth, "shrink then regrow allocates nothing");
    assert_eq!(key_of(&graph, 0).as_deref(), Some(longer.as_slice()));
    // a refused conditional write consumes nothing: the arena still has room after 50 refusals
    for _ in 0..50 {
        let written = graph.write_node_if_untouched(0, 0, &bytes, scale, inv_mag, &[], &[], Some(&long)).expect("no wedge");
        assert!(!written, "slot 0 is touched");
    }
    graph.write_node_raw_with_key(1, 0, &bytes, scale, inv_mag, &[], &[], Some(&long)).expect("arena still has room");
    // a rewrite without a key keeps the stored one
    graph.write_node_raw(1, 0, &bytes, scale, inv_mag, &[], &[]).expect("keyless rewrite");
    assert_eq!(key_of(&graph, 1).as_deref(), Some(long.as_slice()), "an omitted key leaves the stored key");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn reinserting_an_overflow_key_into_a_recycled_slot_reuses_its_arena_range() {
    let dims = 16;
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();
    let path = temp("churn");
    let graph = Graph::new(PlaneFile::create_with_keys(&path, dims, 16, 8, 8).expect("create"));
    let long = vec![b'x'; 200];
    let first = insert_with_key(&graph, &vector_for(1, dims), &long, &params, &mut scratch).expect("insert");
    let arena_after_first = graph.file.key_arena_high_water();
    for round in 0..12 {
        graph.delete_node(first).expect("delete");
        let again = insert_with_key(&graph, &vector_for(1, dims), &long, &params, &mut scratch).expect("reinsert");
        assert_eq!(again, first, "round {round}: the freed slot is reused");
        assert_eq!(graph.file.key_arena_high_water(), arena_after_first, "round {round}: no fresh arena range");
    }
    let _ = std::fs::remove_file(&path);
}
