//! Distance kernels. Asymmetric: the query in the plane's own codec × the stored vector
//! (matching the per-vector symmetric scale + cached 1/|v| model). Symmetric stored×stored for
//! construction-time neighbor↔neighbor checks (stored per-edge distances were dropped from the
//! format; recompute). AVX2 with scalar fallback; Linux x86_64 is the performance target, other
//! platforms take the scalar path (fine for dev).
//!
//! The int16 kernel quantizes the query once per search and uses `_mm256_madd_epi16`, which
//! needs both operands within ±32767 — see `Quant::max_abs`.

use crate::format::{PlaneFile, Quant};

/// A vector in a plane's storage encoding: raw little-endian slot bytes plus the two scalars
/// the slot caches beside them.
pub struct Quantized {
    pub bytes: Vec<u8>,
    pub scale: f32,
    pub inv_mag: f32,
}

/// Precomputed query state, built once per search and bound to one plane's codec. Exactly one
/// of `f32_vector` and `quantized` is populated — the operand the selected kernel reads.
pub struct Query {
    dims: usize,
    /// 1/|query|.
    pub inv_mag: f32,
    /// `inv_mag` with the query-side quantization scale already folded in, so the cosine
    /// expression costs the same multiply count for both codecs.
    norm: f32,
    /// int8 operand: the caller's own f32 vector, moved rather than copied.
    f32_vector: Vec<f32>,
    /// int16 operand: the query quantized to little-endian i16.
    quantized: Vec<u8>,
    /// Private with `kernel` and the two operand buffers: `Graph::distance_to` trusts this to
    /// decide how many bytes the kernel reads out of a slot.
    quant: Quant,
    kernel: unsafe fn(&Query, *const u8) -> f32,
}

impl Query {
    /// Build a query for `file`'s codec. This is the only public constructor, because a query
    /// carrying the wrong element width would read the wrong number of bytes out of every slot.
    pub fn for_plane(file: &PlaneFile, vector: Vec<f32>) -> Self {
        match file.quant() {
            Quant::Int8 => Self::int8(vector),
            Quant::Int16 => {
                let (bytes, scale, inv_mag) = quantize_int16(&vector);
                Self::int16(vector.len(), bytes, scale, inv_mag)
            }
        }
    }

    /// `for_plane` reusing an encoding already computed for the same vector under the same
    /// codec. Crate-internal: a shorter encoding would be read past its end.
    pub(crate) fn for_plane_reusing(file: &PlaneFile, vector: &[f32], stored: &Quantized) -> Self {
        assert_eq!(stored.bytes.len(), file.vector_bytes(), "reused encoding is not this plane's");
        match file.quant() {
            Quant::Int8 => Self::int8_with(vector.to_vec(), stored.inv_mag),
            Quant::Int16 => Self::int16(vector.len(), stored.bytes.clone(), stored.scale, stored.inv_mag),
        }
    }

    fn int8(vector: Vec<f32>) -> Self {
        let inv_mag = inv_magnitude(&vector);
        Self::int8_with(vector, inv_mag)
    }

    fn int8_with(vector: Vec<f32>, inv_mag: f32) -> Self {
        let dims = vector.len();
        Query { dims, inv_mag, norm: inv_mag, f32_vector: vector, quantized: Vec::new(), quant: Quant::Int8, kernel: select_dot_f32_i8() }
    }

    /// The f32 vector is dropped: the int16 kernel never reads it, and only the two scalars
    /// derived from it survive.
    fn int16(dims: usize, quantized: Vec<u8>, scale: f32, inv_mag: f32) -> Self {
        Query {
            dims,
            inv_mag,
            norm: inv_mag * scale,
            f32_vector: Vec::new(),
            quantized,
            quant: Quant::Int16,
            kernel: select_dot_i16_i16(),
        }
    }

    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    #[inline]
    pub fn quant(&self) -> Quant {
        self.quant
    }
}

fn inv_magnitude(vector: &[f32]) -> f32 {
    let mag_sq: f32 = vector.iter().map(|v| v * v).sum();
    1.0 / mag_sq.sqrt().max(f32::MIN_POSITIVE)
}

/// Cosine distance: `query` (in the plane's codec) × the raw stored vector at `stored`.
/// Zero-copy: `stored` points into the mmap; the caller's seqlock read discards torn results.
///
/// # Safety
/// `stored` must be readable for `query.dims()` elements of the QUERY's codec. Pairing a
/// query with a plane of the other width reads the wrong length — `Graph::distance_to` is the
/// only caller and rejects a mismatched pair before it gets here.
#[inline]
pub unsafe fn cosine_raw(query: &Query, stored: *const u8, scale: f32, stored_inv_mag: f32) -> f32 {
    let dot = (query.kernel)(query, stored);
    1.0 - dot * scale * stored_inv_mag * query.norm
}

fn select_dot_f32_i8() -> unsafe fn(&Query, *const u8) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return dot_f32_i8_avx2;
        }
    }
    dot_f32_i8_scalar
}

#[inline]
unsafe fn dot_f32_i8_scalar(query: &Query, v: *const u8) -> f32 {
    let q = &query.f32_vector;
    let v = v as *const i8;
    let mut acc = [0.0f32; 8];
    let chunks = q.len() / 8;
    for c in 0..chunks {
        let base = c * 8;
        for lane in 0..8 {
            acc[lane] += q[base + lane] * unsafe { *v.add(base + lane) } as f32;
        }
    }
    let mut dot: f32 = acc.iter().sum();
    for i in chunks * 8..q.len() {
        dot += q[i] * unsafe { *v.add(i) } as f32;
    }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_i8_avx2(query: &Query, v: *const u8) -> f32 {
    use std::arch::x86_64::*;
    let q = &query.f32_vector;
    let v = v as *const i8;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let chunks = q.len() / 16;
    for c in 0..chunks {
        let base = c * 16;
        let v16 = _mm_loadu_si128(v.add(base) as *const __m128i);
        let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v16));
        let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v16, 8)));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(base)), lo, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(base + 8)), hi, acc1);
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let s = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    let mut dot = _mm_cvtss_f32(s);
    for i in chunks * 16..q.len() {
        dot += q[i] * *v.add(i) as f32;
    }
    dot
}

fn select_dot_i16_i16() -> unsafe fn(&Query, *const u8) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return dot_i16_i16_query_avx2;
        }
    }
    dot_i16_i16_query_scalar
}

#[inline]
unsafe fn dot_i16_i16_query_scalar(query: &Query, v: *const u8) -> f32 {
    dot_i16_i16_scalar(query.quantized.as_ptr(), v, query.dims()) as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i16_i16_query_avx2(query: &Query, v: *const u8) -> f32 {
    dot_i16_i16_avx2(query.quantized.as_ptr(), v, query.dims())
}

/// Exact i64 reference. Both operand buffers are `len` little-endian i16s, read unaligned:
/// the query's come from a `Vec<u8>` and the stored ones from the mmap.
///
/// # Safety
/// `a` and `b` must each be readable for `len * 2` bytes.
#[inline]
pub unsafe fn dot_i16_i16_scalar(a: *const u8, b: *const u8, len: usize) -> i64 {
    let mut dot = 0i64;
    for i in 0..len {
        let x = (a.add(i * 2) as *const i16).read_unaligned().to_le() as i64;
        let y = (b.add(i * 2) as *const i16).read_unaligned().to_le() as i64;
        dot += x * y;
    }
    dot
}

/// `_mm256_madd_epi16` sums two i16 products into each i32 lane; with both operands within
/// ±32767 one lane maxes at 2·32767² = 2,147,352,578, which fits i32 by 131,069. Accumulating
/// those lanes does NOT fit, so each madd result is widened to i64 before it is added — the
/// safe widening interval is one iteration, not "a few".
///
/// # Safety
/// `a` and `b` must each be readable for `len * 2` bytes, and every i16 they hold must be
/// within ±32767 — a `-32768` pair overflows a madd lane and yields a wrong dot product.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_i16_i16_avx2(a: *const u8, b: *const u8, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let chunks = len / 16;
    for c in 0..chunks {
        let av = _mm256_loadu_si256(a.add(c * 32) as *const __m256i);
        let bv = _mm256_loadu_si256(b.add(c * 32) as *const __m256i);
        let m = _mm256_madd_epi16(av, bv);
        acc = _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(m)));
        acc = _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(m, 1)));
    }
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256(acc, 1);
    let s = _mm_add_epi64(lo, hi);
    let mut dot = _mm_cvtsi128_si64(s) + _mm_extract_epi64(s, 1);
    dot += dot_i16_i16_scalar(a.add(chunks * 32), b.add(chunks * 32), len - chunks * 16);
    dot as f32
}

/// The f32-accumulating variant, kept for the benchmark that chose between them: one
/// `cvtepi32_ps` + `add_ps` per madd instead of two widenings and two adds, 1.5-1.7x faster,
/// but accurate only to the accumulated f32 rounding — under 1e-5 relative across 4096 dims
/// rather than exact, which is why the shipped kernel widens to i64.
///
/// # Safety
/// Same contract as [`dot_i16_i16_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_i16_i16_avx2_f32acc(a: *const u8, b: *const u8, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let chunks = len / 16;
    for c in 0..chunks {
        let av = _mm256_loadu_si256(a.add(c * 32) as *const __m256i);
        let bv = _mm256_loadu_si256(b.add(c * 32) as *const __m256i);
        acc = _mm256_add_ps(acc, _mm256_cvtepi32_ps(_mm256_madd_epi16(av, bv)));
    }
    let s = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    let tail = dot_i16_i16_scalar(a.add(chunks * 32), b.add(chunks * 32), len - chunks * 16);
    _mm_cvtss_f32(s) + tail as f32
}

#[inline]
unsafe fn dot_i8_i8_scalar(a: *const u8, b: *const u8, len: usize) -> i32 {
    let mut dot = 0i32;
    for i in 0..len {
        dot += *(a.add(i) as *const i8) as i32 * *(b.add(i) as *const i8) as i32;
    }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i8_avx2(a: *const u8, b: *const u8, len: usize) -> i32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let chunks = len / 16;
    for c in 0..chunks {
        let av = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.add(c * 16) as *const __m128i));
        let bv = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.add(c * 16) as *const __m128i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(av, bv));
    }
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256(acc, 1);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    let s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    let mut dot = _mm_cvtsi128_si32(s);
    for i in chunks * 16..len {
        dot += *(a.add(i) as *const i8) as i32 * *(b.add(i) as *const i8) as i32;
    }
    dot
}

/// Cosine distance between two stored vectors of the same codec (construction-time neighbor
/// checks). `len` is the dimension count, not a byte length.
///
/// # Safety
/// `a` and `b` must each be readable for `len * quant.elem_size()` bytes.
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn cosine_stored_raw(
    quant: Quant,
    a: *const u8,
    scale_a: f32,
    inv_mag_a: f32,
    b: *const u8,
    scale_b: f32,
    inv_mag_b: f32,
    len: usize,
) -> f32 {
    let dot = match quant {
        Quant::Int8 => {
            #[cfg(target_arch = "x86_64")]
            let dot = if std::arch::is_x86_feature_detected!("avx2") {
                dot_i8_i8_avx2(a, b, len) as f32
            } else {
                dot_i8_i8_scalar(a, b, len) as f32
            };
            #[cfg(not(target_arch = "x86_64"))]
            let dot = dot_i8_i8_scalar(a, b, len) as f32;
            dot
        }
        Quant::Int16 => {
            #[cfg(target_arch = "x86_64")]
            let dot = if std::arch::is_x86_feature_detected!("avx2") {
                dot_i16_i16_avx2(a, b, len)
            } else {
                dot_i16_i16_scalar(a, b, len) as f32
            };
            #[cfg(not(target_arch = "x86_64"))]
            let dot = dot_i16_i16_scalar(a, b, len) as f32;
            dot
        }
    };
    1.0 - dot * scale_a * scale_b * inv_mag_a * inv_mag_b
}

/// Quantize into a plane's storage encoding: symmetric per-vector scale mapping the largest
/// magnitude component to `quant.max_abs()`, plus the cached 1/|v| the slot stores beside it.
pub fn quantize(vector: &[f32], quant: Quant) -> Quantized {
    let max_abs = vector.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let limit = quant.max_abs() as f32;
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / limit };
    let inv_scale = 1.0 / scale;
    let mut bytes = Vec::with_capacity(vector.len() * quant.elem_size());
    for v in vector {
        let q = (v * inv_scale).round().clamp(-limit, limit);
        match quant {
            Quant::Int8 => bytes.push(q as i8 as u8),
            Quant::Int16 => bytes.extend_from_slice(&(q as i16).to_le_bytes()),
        }
    }
    Quantized { bytes, scale, inv_mag: inv_magnitude(vector) }
}

/// Symmetric int8 quantization matching the JS quantizeInt8: scale maps max |component| to 127.
pub fn quantize_int8(vector: &[f32]) -> (Vec<u8>, f32, f32) {
    let q = quantize(vector, Quant::Int8);
    (q.bytes, q.scale, q.inv_mag)
}

/// Symmetric int16 quantization: scale maps max |component| to 32767, stored little-endian.
pub fn quantize_int16(vector: &[f32]) -> (Vec<u8>, f32, f32) {
    let q = quantize(vector, Quant::Int16);
    (q.bytes, q.scale, q.inv_mag)
}


/// Whether every element of a stored int16 buffer is within the kernel's operand domain.
/// -32768 is the one representable value that is not: two of them overflow a madd lane.
pub fn int16_bytes_in_domain(bytes: &[u8]) -> bool {
    bytes.chunks_exact(2).all(|c| i16::from_le_bytes([c[0], c[1]]) != i16::MIN)
}
