//! int16 storage precision: format identity, kernel equivalence, geometry, and the accuracy
//! claim the option exists for.

use hnsw_plane::distance::{cosine_stored_raw, quantize, Query};
use hnsw_plane::format::{Quant, PlaneFile, VERSION, VERSION_INT16};
use hnsw_plane::insert::{insert, insert_with_key, InsertError, InsertParams};
use hnsw_plane::search::{search, SearchScratch};
use hnsw_plane::graph::WriteError;
use hnsw_plane::Graph;
use std::path::PathBuf;

fn temp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("hnsw-precision-{}-{name}.hnsw", std::process::id()));
    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_file(hnsw_plane::stale_path_for(&p));
    p
}

fn vector_for(i: u32, dims: usize) -> Vec<f32> {
    (0..dims).map(|d| ((i as f32 * 0.31 + d as f32) * 0.7).sin()).collect()
}

/// xorshift, so the corpora below are reproducible without a rand dependency.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn next_signed(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

fn i16_bytes(values: &[i16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Exact integer dot, computed outside the kernels under test.
fn reference_dot(a: &[i16], b: &[i16]) -> i64 {
    a.iter().zip(b).map(|(&x, &y)| x as i64 * y as i64).sum()
}

// ---------------------------------------------------------------- kernels

/// The AVX2 int16 kernel accumulates each `madd_epi16` result into i64 before adding it, so it
/// is bit-identical to the exact scalar reference at every dimension — including 1536 and 4096,
/// where an i32 accumulator would wrap. Adversarial vectors sit at the operand domain's edge.
#[test]
fn the_avx2_int16_kernel_matches_the_exact_reference() {
    let mut rng = Rng(0x2f6b_a41d_88c3_0157);
    for &dims in &[1usize, 2, 3, 15, 16, 17, 128, 129, 768, 1536, 4096] {
        let random_a: Vec<i16> = (0..dims).map(|_| (rng.next_signed() * 32767.0) as i16).collect();
        let random_b: Vec<i16> = (0..dims).map(|_| (rng.next_signed() * 32767.0) as i16).collect();
        // every component at the domain edge: one madd lane reaches 2 x 32767^2, 131,069 short
        // of i32::MAX, and the sum across dims passes i32::MAX after the second lane
        let adversarial_a: Vec<i16> = (0..dims).map(|i| if i % 2 == 0 { 32767 } else { -32767 }).collect();
        let adversarial_b: Vec<i16> = (0..dims).map(|i| if i % 3 == 0 { -32767 } else { 32767 }).collect();

        for (label, a, b) in [
            ("random", &random_a, &random_b),
            ("adversarial", &adversarial_a, &adversarial_b),
            ("adversarial-self", &adversarial_a, &adversarial_a),
        ] {
            let (ab, bb) = (i16_bytes(a), i16_bytes(b));
            let expected = reference_dot(a, b);
            // cosine_stored_raw picks AVX2 when the CPU has it; scale 1 and inv_mag 1 leave
            // 1 - dot, so the kernel's own value is what is compared
            let got = 1.0 - unsafe { cosine_stored_raw(Quant::Int16, ab.as_ptr(), 1.0, 1.0, bb.as_ptr(), 1.0, 1.0, dims) };
            assert_eq!(
                got,
                expected as f32,
                "{label} dims {dims}: kernel {got} != exact reference {expected} ({})",
                expected as f32
            );
        }
    }
}

/// The f32-accumulating variant benched beside the shipped kernel is faster but not exact; it
/// must still be within f32 rounding, which is what disqualified it from carrying the
/// acceptance criterion rather than from being measured.
#[cfg(target_arch = "x86_64")]
#[test]
fn the_f32_accumulating_variant_stays_within_f32_rounding() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    for &dims in &[128usize, 768, 1536, 4096] {
        let a: Vec<i16> = (0..dims).map(|i| if i % 2 == 0 { 32767 } else { -32767 }).collect();
        let (ab, bb) = (i16_bytes(&a), i16_bytes(&a));
        let expected = reference_dot(&a, &a) as f64;
        let got = unsafe { hnsw_plane::distance::dot_i16_i16_avx2_f32acc(ab.as_ptr(), bb.as_ptr(), dims) } as f64;
        let relative = (got - expected).abs() / expected.abs();
        assert!(relative < 1e-5, "dims {dims}: f32-accumulated {got} vs exact {expected} (relative {relative:e})");
    }
}

/// The query-side kernel agrees with the same exact reference: a searching plane and a
/// construction-time prune must not disagree about a dot product.
#[test]
fn the_int16_query_kernel_agrees_with_the_stored_kernel() {
    let dims = 768;
    let path = temp("querykernel");
    let file = PlaneFile::create_with_options(&path, dims, 16, 64, 0, 0, Quant::Int16).expect("create");
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let a: Vec<f32> = (0..dims).map(|_| rng.next_signed()).collect();
    let b: Vec<f32> = (0..dims).map(|_| rng.next_signed()).collect();
    let qa = quantize(&a, Quant::Int16);
    let qb = quantize(&b, Quant::Int16);
    let query = Query::for_plane(&file, a.clone());

    let via_query = unsafe { hnsw_plane::distance::cosine_raw(&query, qb.bytes.as_ptr(), qb.scale, qb.inv_mag) };
    let via_stored = unsafe {
        cosine_stored_raw(Quant::Int16, qa.bytes.as_ptr(), qa.scale, qa.inv_mag, qb.bytes.as_ptr(), qb.scale, qb.inv_mag, dims)
    };
    assert!((via_query - via_stored).abs() < 1e-6, "query kernel {via_query} vs stored kernel {via_stored}");
    drop(file);
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------- the -32768 domain

/// -32768 is the one i16 the madd kernel cannot take: two of them sum to exactly 2^31 inside a
/// single lane. Every writer refuses it rather than letting it reach a slot.
#[test]
fn writers_refuse_the_one_i16_the_kernel_cannot_multiply() {
    let dims = 8;
    let path = temp("domain");
    let graph = Graph::new(PlaneFile::create_with_options(&path, dims, 8, 64, 0, 0, Quant::Int16).expect("create"));

    let mut values = vec![100i16; dims];
    values[3] = i16::MIN;
    let bad = i16_bytes(&values);
    match graph.write_node_raw(0, 0, &bad, 1.0, 1.0, &[], &[]) {
        Err(WriteError::BadVector(reason)) => assert!(reason.contains("32767"), "the refusal must name the domain: {reason}"),
        other => panic!("a -32768 element must be refused, got {:?}", other.is_ok()),
    }
    // a correctly-clamped vector through the same call succeeds, so the gate is the value
    values[3] = i16::MIN + 1;
    graph.write_node_raw(0, 0, &i16_bytes(&values), 1.0, 1.0, &[], &[]).expect("-32767 is storable");

    // and the byte length is the plane's, not its dimension count
    match graph.write_node_raw(1, 0, &vec![0u8; dims], 1.0, 1.0, &[], &[]) {
        Err(WriteError::BadVector(reason)) => assert!(reason.contains("byte length"), "{reason}"),
        other => panic!("a dims-length buffer against an int16 plane must be refused, got {:?}", other.is_ok()),
    }
    let _ = std::fs::remove_file(&path);
}

/// Quantization never produces it either: the clamp is symmetric at the codec's max_abs, so
/// even a vector whose components all sit at the extreme stays inside the domain.
#[test]
fn quantization_stays_inside_the_kernel_domain() {
    for dims in [1usize, 7, 128, 1536] {
        for scale in [1.0f32, -1.0, 1e-30, 1e30] {
            let v: Vec<f32> = (0..dims).map(|i| if i % 2 == 0 { scale } else { -scale }).collect();
            let q = quantize(&v, Quant::Int16);
            assert!(
                hnsw_plane::distance::int16_bytes_in_domain(&q.bytes),
                "dims {dims} scale {scale}: quantization produced -32768"
            );
        }
    }
}

/// A file corrupted to hold -32768 must not crash a search. Its distances are perturbed —
/// that pair wraps inside the instruction — which is why the value is refused at the writer
/// rather than tolerated there.
#[test]
fn a_corrupt_minimum_valued_element_does_not_crash_a_search() {
    let dims = 64;
    let path = temp("corruptdomain");
    let graph = Graph::new(PlaneFile::create_with_options(&path, dims, 16, 256, 0, 0, Quant::Int16).expect("create"));
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();
    for i in 0..64u32 {
        insert(&graph, &vector_for(i, dims), &params, &mut scratch).expect("insert");
    }
    // reach past the writers the way a damaged file would: straight into the mapping
    unsafe {
        let p = graph.file.slot_ptr_mut(5).add(hnsw_plane::format::S_VECTOR) as *mut i16;
        p.write_unaligned(i16::MIN);
        p.add(1).write_unaligned(i16::MIN);
    }
    let (hits, _) = search(&graph, &graph.query(vector_for(5, dims)), 10, 64, &mut scratch);
    assert!(!hits.is_empty(), "a corrupt element must not empty the result set");
    assert!(hits.iter().all(|(_, d)| d.is_finite()), "a corrupt element must not produce a non-finite distance");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------- format identity

fn header_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// int8 planes keep writing a v8 header, so every released reader still opens what this build
/// creates; int16 planes are v9, which a released reader refuses at the version check before
/// it can misread the geometry.
#[test]
fn the_codec_is_carried_by_the_file_version_not_only_by_h_quant() {
    for (quant, version, quant_byte) in [(Quant::Int8, VERSION, 0u8), (Quant::Int16, VERSION_INT16, 2u8)] {
        let path = temp(&format!("version-{}", quant.name()));
        let file = PlaneFile::create_with_options(&path, 32, 16, 64, 0, 0, quant).expect("create");
        drop(file);
        let bytes = std::fs::read(&path).expect("read header");
        assert_eq!(header_u32(&bytes, 4), version, "{} must be format v{version}", quant.name());
        assert_eq!(bytes[10], quant_byte, "{} H_QUANT", quant.name());
        assert_eq!(PlaneFile::open(&path).expect("reopen").quant(), quant);
        let _ = std::fs::remove_file(&path);
    }
}

/// The geometry check a released reader relies on does NOT separate the widths: at dims 16 /
/// cap 16 both round to the same 128-byte slot, so `H_QUANT` alone would have let an old
/// binary open an int16 plane and write neighbors over its vector. This is the case the
/// version gate exists for.
#[test]
fn the_widths_collide_on_rounded_slot_size_which_is_why_the_version_gates_them() {
    let mut int8_size = 0;
    for quant in [Quant::Int8, Quant::Int16] {
        let path = temp(&format!("collide-{}", quant.name()));
        let file = PlaneFile::create_with_options(&path, 16, 16, 64, 0, 0, quant).expect("create");
        if quant == Quant::Int8 {
            int8_size = file.slot_size;
        } else {
            assert_eq!(file.slot_size, int8_size, "precondition: the collision this test guards is real");
        }
        drop(file);
        let _ = std::fs::remove_file(&path);
    }
}

/// Exactly two (version, codec) pairs are produced and exactly two open. Everything else —
/// including v9/int8, which nothing emits, and the reserved f32 byte — is a descriptive
/// refusal, never a guess.
#[test]
fn an_unproduced_version_and_codec_pair_refuses_to_open() {
    for (version, quant_byte, label) in [
        (VERSION, 2u8, "v8 claiming int16"),
        (VERSION, 1u8, "v8 claiming the reserved f32 mode"),
        (VERSION_INT16, 0u8, "v9 claiming int8"),
        (VERSION_INT16, 1u8, "v9 claiming the reserved f32 mode"),
        (VERSION_INT16, 3u8, "v9 with an unknown codec"),
        (VERSION_INT16 + 1, 0u8, "a future version"),
    ] {
        let path = temp(&format!("badpair-{version}-{quant_byte}"));
        let file = PlaneFile::create_with_options(&path, 32, 16, 64, 0, 0, Quant::Int8).expect("create");
        drop(file);
        let mut bytes = std::fs::read(&path).expect("read");
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        bytes[10] = quant_byte;
        std::fs::write(&path, &bytes).expect("write");

        let message = match PlaneFile::open(&path) {
            Ok(_) => panic!("{label} must not open"),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("unsupported plane format") && message.contains(&version.to_string()),
            "{label}: error must name the rejected format, got {message:?}"
        );
        let _ = std::fs::remove_file(&path);
    }
}

/// An int16 slot costs one more byte per dimension than an int8 one — exactly, once each
/// vector's independent 4-byte padding is accounted for. The headline "+1 byte per dimension"
/// is the 4-aligned case; at dims 13 the padded difference is 12.
#[test]
fn an_int16_slot_costs_one_more_byte_per_dimension() {
    let pad4 = |n: usize| n.div_ceil(4) * 4;
    for &dims in &[1usize, 2, 3, 13, 16, 128, 768, 1536] {
        for &cap in &[8usize, 128] {
            let small = hnsw_plane::format::key_offset(dims, cap);
            let wide = hnsw_plane::format::key_offset(dims * 2, cap);
            assert_eq!(wide - small, pad4(dims * 2) - pad4(dims), "dims {dims} cap {cap}");
            if dims % 4 == 0 {
                assert_eq!(wide - small, dims, "4-aligned dims {dims} must cost exactly dims more bytes");
            }
        }
    }
}

/// Header fields are u16; an out-of-range geometry used to narrow silently into a file no
/// later open could accept, and the truncating create destroyed whatever was already there.
#[test]
fn an_unsatisfiable_geometry_is_refused_without_touching_the_path() {
    let path = temp("badgeometry");
    std::fs::write(&path, b"not a plane, but not expendable either").expect("seed");
    for (dims, cap) in [(0usize, 16usize), (65_536, 16), (32, 0), (32, 65_536)] {
        assert!(
            PlaneFile::create_with_options(&path, dims, cap, 64, 0, 0, Quant::Int16).is_err(),
            "dims {dims} cap {cap} must be refused"
        );
        assert_eq!(
            std::fs::read(&path).expect("the file must survive a refused create"),
            b"not a plane, but not expendable either",
            "dims {dims} cap {cap}: a refused create truncated an existing file"
        );
    }
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------- behavior

/// The fixture is a plane built by the released build and committed, rather than regenerated
/// at test time: a future layout change cannot quietly regenerate agreement with itself.
#[test]
fn a_plane_from_the_released_build_opens_and_searches_identically() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v8-int8.hnsw");
    let expected = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v8-int8.expected"))
        .expect("expected results");
    // opening registers this handle in the file, so work on a copy
    let path = temp("v8fixture");
    std::fs::copy(&fixture, &path).expect("copy fixture");

    let file = PlaneFile::open(&path).expect("the released build's plane must still open");
    assert_eq!(file.quant(), Quant::Int8);
    assert_eq!(file.dims(), 32);
    let graph = Graph::new(file);
    let mut scratch = SearchScratch::new();
    for line in expected.trim().lines() {
        let (qi, want) = line.split_once(' ').expect("fixture line");
        let (hits, _) = search(&graph, &graph.query(vector_for(qi.parse().unwrap(), 32)), 10, 64, &mut scratch);
        let got: Vec<String> = hits.iter().map(|(id, d)| format!("{id}:{d:.7}")).collect();
        assert_eq!(got.join(","), want, "query {qi} no longer returns what the released build returned");
    }
    let _ = std::fs::remove_file(&path);
}

/// An int16 plane round-trips through the ordinary insert and search path, keys included.
#[test]
fn an_int16_plane_round_trips_inserts_and_searches() {
    let dims = 96;
    let path = temp("roundtrip");
    let graph = Graph::new(PlaneFile::create_with_options(&path, dims, 32, 4096, 8, 128, Quant::Int16).expect("create"));
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();
    for i in 0..1000u32 {
        insert_with_key(&graph, &vector_for(i, dims), &i.to_le_bytes(), &params, &mut scratch).expect("insert");
    }
    let mut misses = 0;
    for i in (0..1000u32).step_by(17) {
        let (hits, _) = search(&graph, &graph.query(vector_for(i, dims)), 5, 64, &mut scratch);
        if hits.first().map(|&(_, d)| d) .unwrap_or(1.0) > 1e-4 {
            misses += 1;
        }
    }
    assert_eq!(misses, 0, "every stored vector must be its own nearest neighbor at distance ~0");

    // and across a reopen, which is where a geometry the header disagreed with would show
    drop(graph);
    let reopened = PlaneFile::open(&path).expect("reopen");
    assert_eq!(reopened.quant(), Quant::Int16);
    let graph = Graph::new(reopened);
    let (hits, _) = search(&graph, &graph.query(vector_for(500, dims)), 5, 64, &mut scratch);
    assert_eq!(hits[0].0, 500, "the reopened plane must still find the same nearest neighbor");
    let _ = std::fs::remove_file(&path);
}

/// Quantization accuracy isolated from the graph: over one fixed candidate set, int16
/// reproduces the exact-f32 top-10 ordering that int8 gets wrong. The recall test below
/// conflates quantization error with graph approximation, so it cannot show this.
#[test]
fn int16_reproduces_exact_ordering_where_int8_does_not() {
    let (dims, candidates) = (128usize, 2000);
    let mut rng = Rng(0xc0ff_ee00_1234_5678);
    let corpus: Vec<Vec<f32>> = (0..candidates).map(|_| (0..dims).map(|_| rng.next_signed()).collect()).collect();
    let queries: Vec<Vec<f32>> = (0..50).map(|_| (0..dims).map(|_| rng.next_signed()).collect()).collect();

    let exact = |q: &[f32], v: &[f32]| -> f64 {
        let dot: f64 = q.iter().zip(v).map(|(&a, &b)| a as f64 * b as f64).sum();
        let qm: f64 = q.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>().sqrt();
        let vm: f64 = v.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>().sqrt();
        1.0 - dot / (qm * vm)
    };

    let mut disagreements = [0usize; 2];
    for (slot, quant) in [Quant::Int8, Quant::Int16].into_iter().enumerate() {
        let stored: Vec<_> = corpus.iter().map(|v| quantize(v, quant)).collect();
        let path = temp(&format!("ordering-{}", quant.name()));
        let file = PlaneFile::create_with_options(&path, dims, 8, 16, 0, 0, quant).expect("create");
        for q in &queries {
            let query = Query::for_plane(&file, q.clone());
            let mut approx: Vec<(usize, f32)> = stored
                .iter()
                .enumerate()
                .map(|(i, s)| (i, unsafe { hnsw_plane::distance::cosine_raw(&query, s.bytes.as_ptr(), s.scale, s.inv_mag) }))
                .collect();
            approx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut truth: Vec<(usize, f64)> = corpus.iter().enumerate().map(|(i, v)| (i, exact(q, v))).collect();
            truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for rank in 0..10 {
                if approx[rank].0 != truth[rank].0 {
                    disagreements[slot] += 1;
                }
            }
        }
        drop(file);
        let _ = std::fs::remove_file(&path);
    }
    let total = queries.len() * 10;
    assert!(
        disagreements[1] * 20 < disagreements[0].max(1),
        "int16 must reorder the exact top-10 far less than int8 does: int8 {}/{total}, int16 {}/{total}",
        disagreements[0],
        disagreements[1]
    );
    assert!(
        disagreements[1] * 1000 <= total,
        "int16 reordered {}/{total} of the exact top-10, which is too much to skip a rerank on",
        disagreements[1]
    );
}

/// End-to-end recall with no rerank: within 0.2 percentage points of an f32 brute-force truth.
#[test]
#[ignore = "builds a 20k-node graph; run with --ignored in the full gate"]
fn int16_recall_at_ten_tracks_an_f32_brute_force_truth() {
    let (dims, n) = (128usize, 20_000u32);
    let path = temp("recall");
    let graph = Graph::new(PlaneFile::create_with_options(&path, dims, 32, n as u64 + 1024, 0, 0, Quant::Int16).expect("create"));
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    // clustered, like an embedding corpus: uniform-random 128-d has no neighborhood to find
    let clusters: Vec<Vec<f32>> = (0..64).map(|_| (0..dims).map(|_| rng.next_signed()).collect()).collect();
    let mut corpus = Vec::with_capacity(n as usize);
    for i in 0..n {
        let c = &clusters[i as usize % clusters.len()];
        let v: Vec<f32> = c.iter().map(|&x| x + rng.next_signed() * 0.35).collect();
        insert(&graph, &v, &params, &mut scratch).expect("insert");
        corpus.push(v);
    }

    let exact_dot = |q: &[f32], v: &[f32]| -> f64 {
        let dot: f64 = q.iter().zip(v).map(|(&a, &b)| a as f64 * b as f64).sum();
        let qm: f64 = q.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>().sqrt();
        let vm: f64 = v.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>().sqrt();
        1.0 - dot / (qm * vm)
    };

    let (mut hit, mut total) = (0usize, 0usize);
    for qi in 0..100 {
        let c = &clusters[qi % clusters.len()];
        let q: Vec<f32> = c.iter().map(|&x| x + rng.next_signed() * 0.35).collect();
        let mut truth: Vec<(u32, f64)> = corpus.iter().enumerate().map(|(i, v)| (i as u32, exact_dot(&q, v))).collect();
        truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        truth.truncate(10);
        let (hits, _) = search(&graph, &graph.query(q), 10, 96, &mut scratch);
        total += truth.len();
        hit += truth.iter().filter(|(tid, _)| hits.iter().any(|(rid, _)| rid == tid)).count();
    }
    let recall = hit as f64 / total as f64;
    assert!(recall >= 0.998, "int16 recall@10 without rerank was {recall:.4}, more than 0.2 points below an exact truth");
    let _ = std::fs::remove_file(&path);
}

/// A wrong-dimension vector is refused at the boundary. It used to reach the slot copy; then
/// it panicked inside query construction, after `allocate_id` had already run and past the
/// cleanup path, leaking the id.
#[test]
fn a_wrong_dimension_insert_is_refused_without_consuming_an_id() {
    let path = temp("insertdims");
    let graph = Graph::new(PlaneFile::create_with_options(&path, 16, 8, 256, 0, 0, Quant::Int16).expect("create"));
    let params = InsertParams::default();
    let mut scratch = SearchScratch::new();
    insert(&graph, &vector_for(0, 16), &params, &mut scratch).expect("a correctly sized insert");
    let before = graph.file.id_high_water();
    for wrong in [15usize, 17, 32] {
        assert!(
            matches!(insert(&graph, &vector_for(1, wrong), &params, &mut scratch), Err(InsertError::DimensionMismatch)),
            "a {wrong}-dimension vector must be refused by a 16-dimension plane"
        );
    }
    assert_eq!(graph.file.id_high_water(), before, "a refused insert must not consume an id");
    let _ = std::fs::remove_file(&path);
}
