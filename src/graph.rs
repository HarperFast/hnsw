//! Slot-level node access over the plane file, mediated by per-slot seqlocks. Hot-path
//! reads (distance, neighbor ids) are zero-copy against the mmap; full-copy read_node
//! exists for construction paths. Upper-layer adjacency lives in a fixed-entry region of
//! the same file (per-entry seqlocks), so the hierarchy persists with the graph and
//! concurrent searches share nothing mutable.

use crate::distance::{cosine_raw, cosine_stored_raw, int16_bytes_in_domain, Query};
use crate::format::{
    key_class, PlaneFile, Quant, FLAG_DELETED, FLAG_VALID, KEY_PAYLOAD, MAX_KEY_LEN, MAX_UPPER_LEVELS, NO_UPPER,
    S_DEGREE, S_FLAGS, S_INV_MAG, S_LEVEL, S_SCALE, S_UPPER_IDX, S_VECTOR, UPPER_CAP, UPPER_LEVEL_STRIDE, UL_DEGREE,
    UL_IDS, U_LEVELS, U_LISTS,
};
use crate::prefetch::PageRange;
use crate::seqlock;
use crate::seqlock::Wedged;

/// Aligned volatile load of a slot/upper-entry field another process may be mutating.
///
/// This forbids the optimizer from duplicating, splitting, or sinking the load across the
/// seqlock's validating fence, which would let a reader act on bytes the generation check
/// never covered. It does NOT make the access race-free under Rust's memory model — only
/// atomics would, and that is the format change DESIGN.md §10 records as
/// follow-up. The vector is deliberately not read this way: the distance kernel must stay
/// autovectorized, and a torn vector only perturbs a distance the generation check discards.
/// Every field read here is naturally aligned (slots are 64-aligned; the neighbor and upper
/// id arrays are 4-padded by format.rs), so these compile to single loads.
#[inline(always)]
unsafe fn vread<T: Copy>(p: *const T) -> T {
    p.read_volatile()
}

const PREFETCH_BYTES: usize = 256;

#[inline(always)]
fn prefetch_line(p: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(p as *const i8, _MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

/// Byte copy through volatile loads: key bytes are read inside a seqlock section that a
/// writer may be rewriting, like every other slot field read here.
#[inline]
unsafe fn copy_volatile(out: &mut Vec<u8>, src: *const u8, len: usize) {
    out.reserve(len);
    for i in 0..len {
        out.push(src.add(i).read_volatile());
    }
}

pub struct Graph {
    pub file: PlaneFile,
    /// Rotates `probe_for_entry`'s starting offset so this plane's consecutive repairs sample
    /// different ids. Per handle, not per process: a shared counter is advanced by every other
    /// plane's repairs too, so one plane's calls can land on a single residue indefinitely —
    /// which is the coverage the rotation exists to provide.
    probe_rotation: std::sync::atomic::AtomicU32,
    /// (high-water, write epoch) at which this handle's last `stride` consecutive probes all
    /// came back empty, so a fully dead graph stops paying the probe; any handle's node write
    /// bumps the header epoch and re-arms it.
    probe_futile_hw: std::sync::atomic::AtomicU64,
    probe_futile_epoch: std::sync::atomic::AtomicU64,
    probe_futile_runs: std::sync::atomic::AtomicU32,
    /// One repair probe at a time per handle: concurrent searches on the pool would each pay
    /// the full walk before one of them publishes.
    probe_in_flight: std::sync::atomic::AtomicBool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyError {
    NoKeys,
    TooLong,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteError {
    /// a slot lock could not be acquired or reclaimed within the wedge bound
    Wedged,
    /// the key overflow arena is exhausted (only a rebuild recovers the space)
    KeyArenaFull,
    /// the raw vector is not `file.vector_bytes` long, or holds an element outside the codec's
    /// operand domain
    BadVector(&'static str),
}

impl From<Wedged> for WriteError {
    fn from(_: Wedged) -> Self {
        WriteError::Wedged
    }
}

/// A consistent full copy of one node (construction paths only; search uses zero-copy).
pub struct NodeRead {
    pub level: u8,
    pub scale: f32,
    pub inv_mag: f32,
    /// Raw stored bytes in the plane's codec (`file.vector_bytes` long), not element values.
    pub vector: Vec<u8>,
    pub neighbors: Vec<u32>,
}

impl Graph {
    pub fn new(file: PlaneFile) -> Self {
        // probe the kernel-prefetch backend (and calibrate its clock) here, not on a search
        crate::prefetch::mode();
        Graph {
            file,
            probe_rotation: std::sync::atomic::AtomicU32::new(0),
            probe_futile_hw: std::sync::atomic::AtomicU64::new(u64::MAX),
            probe_futile_epoch: std::sync::atomic::AtomicU64::new(u64::MAX),
            probe_futile_runs: std::sync::atomic::AtomicU32::new(0),
            probe_in_flight: std::sync::atomic::AtomicBool::new(false),
        }
    }

    #[inline]
    fn node_written(&self) {
        self.file.bump_write_epoch();
    }

    #[inline]
    fn in_range(&self, id: u32) -> bool {
        (id as u64) < self.file.id_high_water()
    }

    /// Sanitizer for a slot lock taken over from a dead writer: the payload is half-written,
    /// so the slot must read as deleted until something rewrites it (heal-on-touch contract;
    /// FLAG_DELETED rather than 0 so hosts can still free/reuse the id).
    fn slot_sanitizer(&self, id: u32) -> impl Fn() + '_ {
        move || unsafe {
            let p = self.file.slot_ptr_mut(id);
            // a dead writer's slot may hold a garbage (or zero-initialized) upper index; a
            // later raw rewrite would reuse it and clobber another node's hierarchy
            (p.add(S_UPPER_IDX) as *mut u32).write_unaligned(NO_UPPER);
            // and a half-written overflow record: a zero length is never reused
            if self.file.key_cap > 0 {
                (p.add(self.file.key_offset()) as *mut u16).write_unaligned(0);
            }
            *p.add(S_FLAGS) = FLAG_DELETED;
        }
    }

    fn owner_dead(&self) -> impl Fn(u32) -> bool + '_ {
        move |tag| self.file.tag_is_dead(tag)
    }

    /// Sanitizer for an upper-entry lock taken over from a dead writer.
    fn upper_sanitizer(&self, idx: u32) -> impl Fn() + '_ {
        move || unsafe { *self.file.upper_ptr_mut(idx).add(U_LEVELS) = 0 }
    }

    /// A query in this plane's codec.
    pub fn query(&self, vector: Vec<f32>) -> Query {
        Query::for_plane(&self.file, vector)
    }

    /// Callers apply this before allocating an upper entry, so a rejected write leaks nothing.
    /// -32768 is refused because two of them overflow one `_mm256_madd_epi16` lane, which no
    /// accumulator width above the instruction can repair.
    fn check_stored_vector(&self, vector: &[u8]) -> Result<(), WriteError> {
        if vector.len() != self.file.vector_bytes() {
            return Err(WriteError::BadVector("vector byte length does not match the plane's dims x element size"));
        }
        if self.file.quant() == Quant::Int16 && !int16_bytes_in_domain(vector) {
            return Err(WriteError::BadVector("int16 vectors must stay within +/-32767; -32768 is not a storable value"));
        }
        Ok(())
    }

    /// Whether `key` can be stored on this plane at all (capacity is checked at write time).
    pub fn check_key(&self, key: &[u8]) -> Result<(), KeyError> {
        if key.is_empty() {
            return Ok(());
        }
        if self.file.key_cap == 0 {
            return Err(KeyError::NoKeys);
        }
        if key.len() > MAX_KEY_LEN {
            return Err(KeyError::TooLong);
        }
        Ok(())
    }

    /// Store `key` in the slot at `p` (caller holds the slot lock; `check_key` passed): the
    /// u16 length, then the inline bytes or the arena offset (u32 lo, u32 hi). A range is
    /// reserved at `key_class(len)`, so its capacity follows from the stored length; an
    /// overflow key reuses the slot's range while it fits that class — safe under the lock,
    /// since a reader validates the generation after copying the bytes — so only growth past
    /// the class allocates, and a key that shrinks to inline (or below its class) releases the
    /// range. A record torn by a dead writer is never reused: the lock takeover sanitizer
    /// zeroes the length. `None` leaves the stored key untouched.
    unsafe fn store_key_locked(&self, p: *mut u8, key: Option<&[u8]>, reserved: Option<u64>) -> Result<(), WriteError> {
        let Some(key) = key else { return Ok(()) };
        if self.file.key_cap == 0 {
            return Ok(());
        }
        let kp = p.add(self.file.key_offset());
        let payload = kp.add(KEY_PAYLOAD);
        if key.len() <= self.file.key_cap {
            (kp as *mut u16).write_unaligned((key.len() as u16).to_le());
            std::ptr::copy_nonoverlapping(key.as_ptr(), payload, key.len());
            return Ok(());
        }
        let offset = match reserved.or_else(|| self.reusable_key_range(p, key.len())) {
            Some(offset) => offset,
            None => self.file.allocate_key_bytes(key_class(key.len())).ok_or(WriteError::KeyArenaFull)?,
        };
        std::ptr::copy_nonoverlapping(key.as_ptr(), self.file.key_arena_ptr_mut(offset), key.len());
        (payload as *mut u32).write_unaligned((offset as u32).to_le());
        (payload.add(4) as *mut u32).write_unaligned(((offset >> 32) as u32).to_le());
        (kp as *mut u16).write_unaligned((key.len() as u16).to_le());
        Ok(())
    }

    /// The slot's own overflow range, when the key it last stored (live or deleted) reserved
    /// a class that still fits `key_len`.
    unsafe fn reusable_key_range(&self, p: *const u8, key_len: usize) -> Option<u64> {
        let kp = p.add(self.file.key_offset());
        let payload = kp.add(KEY_PAYLOAD);
        let old_len = u16::from_le((kp as *const u16).read_unaligned()) as usize;
        if old_len <= self.file.key_cap || *p.add(S_FLAGS) == 0 || key_class(old_len) < key_len {
            return None;
        }
        let lo = u32::from_le((payload as *const u32).read_unaligned()) as u64;
        let hi = u32::from_le((payload.add(4) as *const u32).read_unaligned()) as u64;
        let existing = lo | (hi << 32);
        // file-sourced: a range that does not fit the arena is not reused
        let end = existing.checked_add(key_class(old_len) as u64)?;
        (end <= self.file.key_arena_len).then_some(existing)
    }

    /// An insert that just took `id` from the allocator asks whether the recycled slot already
    /// owns an overflow range big enough for its key, before reserving a fresh one. The slot
    /// is the caller's now, so the unlocked read races nothing that writes it.
    pub(crate) fn recycled_key_range(&self, id: u32, key_len: usize) -> Option<u64> {
        if self.file.key_cap == 0 || key_len <= self.file.key_cap || !self.in_range(id) {
            return None;
        }
        unsafe { self.reusable_key_range(self.file.slot_ptr(id), key_len) }
    }

    /// Copy the host key of `id` into `out` (cleared first). None for absent/deleted nodes;
    /// Some with an empty `out` for a node stored without a key.
    pub fn key_into(&self, id: u32, out: &mut Vec<u8>) -> Option<()> {
        out.clear();
        if !self.in_range(id) {
            return None;
        }
        let key_cap = self.file.key_cap;
        if key_cap == 0 {
            return self.node_alive(id).then_some(());
        }
        let koff = self.file.key_offset();
        let arena_len = self.file.key_arena_len;
        let seq = self.file.seq_atomic(id);
        seqlock::read_consistent(seq, self.file.self_tag, || {
            out.clear();
            let p = self.file.slot_ptr(id);
            unsafe {
                let flags = vread(p.add(S_FLAGS));
                if flags & FLAG_VALID == 0 || flags & FLAG_DELETED != 0 {
                    return None;
                }
                let len = u16::from_le(vread(p.add(koff) as *const u16)) as usize;
                let payload = p.add(koff + KEY_PAYLOAD);
                if len <= key_cap {
                    copy_volatile(out, payload, len);
                } else {
                    let lo = u32::from_le(vread(payload as *const u32)) as u64;
                    let hi = u32::from_le(vread(payload.add(4) as *const u32)) as u64;
                    let offset = lo | (hi << 32);
                    // file-sourced offset: a corrupt one reads as "no key" rather than off the map
                    if offset.checked_add(len as u64).is_some_and(|end| end <= arena_len) {
                        copy_volatile(out, self.file.key_arena_ptr(offset), len);
                    }
                }
                Some(())
            }
        }, self.slot_sanitizer(id), || None, self.owner_dead())
    }

    #[inline]
    fn node_alive(&self, id: u32) -> bool {
        let seq = self.file.seq_atomic(id);
        seqlock::read_consistent(seq, self.file.self_tag, || unsafe { vread(self.file.slot_ptr(id).add(S_FLAGS)) == FLAG_VALID }, self.slot_sanitizer(id), || false, self.owner_dead())
    }

    /// Hint the cache lines `distance_to(id)` will read, so a whole adjacency list's misses
    /// overlap instead of being taken one at a time.
    #[inline]
    pub fn prefetch_slot(&self, id: u32) {
        debug_assert!((id as u64) < self.file.max_nodes);
        let p = self.file.slot_ptr(id);
        // the header and first vector lines cover a 128-d slot entirely; wider vectors get
        // their leading lines, enough to overlap the miss without flooding L1 on hub nodes
        let end = (S_VECTOR + self.file.vector_bytes()).min(PREFETCH_BYTES);
        let mut off = 0;
        while off < end {
            prefetch_line(unsafe { p.add(off) });
            off += 64;
        }
    }

    /// The pages holding what search reads from slot `id`: seqlock through adjacency, never the
    /// key field (up to 64 KiB that no distance read touches).
    #[inline]
    pub fn slot_read_span(&self, id: u32) -> PageRange {
        let start = self.file.slot_ptr(id) as usize;
        PageRange::covering(start, start + self.file.key_offset())
    }

    /// Zero-copy distance from `query` to the stored vector of `id`. None for absent/deleted.
    #[inline]
    pub fn distance_to(&self, id: u32, query: &Query) -> Option<f32> {
        if !self.in_range(id) {
            return None;
        }
        // A query built for another plane would stream the wrong number of bytes out of every
        // slot — at a narrow layer-0 cap, past the slot and off the end of the mapping.
        if query.quant() != self.file.quant() || query.dims() != self.file.dims() {
            return None;
        }
        let seq = self.file.seq_atomic(id);
        seqlock::read_consistent(seq, self.file.self_tag, || {
            let p = self.file.slot_ptr(id);
            unsafe {
                let flags = vread(p.add(S_FLAGS));
                if flags & FLAG_VALID == 0 || flags & FLAG_DELETED != 0 {
                    return None;
                }
                let scale = vread(p.add(S_SCALE) as *const f32);
                let inv_mag = vread(p.add(S_INV_MAG) as *const f32);
                Some(cosine_raw(query, p.add(S_VECTOR), scale, inv_mag))
            }
        }, self.slot_sanitizer(id), || None, self.owner_dead())
    }

    /// Symmetric stored-to-stored distance (construction-time neighbor↔neighbor checks).
    /// Plain unlocked reads: a torn read only perturbs a construction heuristic.
    pub fn distance_between(&self, a: u32, b: u32) -> Option<f32> {
        if !self.in_range(a) || !self.in_range(b) {
            return None;
        }
        let dims = self.file.dims();
        let pa = self.file.slot_ptr(a);
        let pb = self.file.slot_ptr(b);
        unsafe {
            let fa = *pa.add(S_FLAGS);
            let fb = *pb.add(S_FLAGS);
            if fa & FLAG_VALID == 0 || fa & FLAG_DELETED != 0 || fb & FLAG_VALID == 0 || fb & FLAG_DELETED != 0 {
                return None;
            }
            let scale_a = (pa.add(S_SCALE) as *const f32).read_unaligned();
            let inv_a = (pa.add(S_INV_MAG) as *const f32).read_unaligned();
            let scale_b = (pb.add(S_SCALE) as *const f32).read_unaligned();
            let inv_b = (pb.add(S_INV_MAG) as *const f32).read_unaligned();
            Some(cosine_stored_raw(
                self.file.quant(),
                pa.add(S_VECTOR),
                scale_a,
                inv_a,
                pb.add(S_VECTOR),
                scale_b,
                inv_b,
                dims,
            ))
        }
    }

    /// Copy layer-0 neighbor ids into `out` (cleared first). Returns the node's level,
    /// or None for absent/deleted.
    #[inline]
    pub fn neighbors_into(&self, id: u32, out: &mut Vec<u32>) -> Option<u8> {
        out.clear();
        if !self.in_range(id) {
            return None;
        }
        let seq = self.file.seq_atomic(id);
        let cap = self.file.layer0_cap;
        let nbase = self.file.neighbor_offset();
        seqlock::read_consistent(seq, self.file.self_tag, || {
            out.clear();
            let p = self.file.slot_ptr(id);
            unsafe {
                let flags = vread(p.add(S_FLAGS));
                if flags & FLAG_VALID == 0 || flags & FLAG_DELETED != 0 {
                    return None;
                }
                let level = vread(p.add(S_LEVEL));
                let degree = u16::from_le(vread(p.add(S_DEGREE) as *const u16)) as usize;
                let base = p.add(nbase) as *const u32;
                for i in 0..degree.min(cap) {
                    out.push(u32::from_le(vread(base.add(i))));
                }
                Some(level)
            }
        }, self.slot_sanitizer(id), || None, self.owner_dead())
    }

    /// The node's upper-region entry index, or NO_UPPER.
    #[inline]
    fn upper_idx_of(&self, id: u32) -> u32 {
        if !self.in_range(id) {
            return NO_UPPER;
        }
        let seq = self.file.seq_atomic(id);
        seqlock::read_consistent(seq, self.file.self_tag, || {
            let p = self.file.slot_ptr(id);
            unsafe {
                let flags = vread(p.add(S_FLAGS));
                if flags & FLAG_VALID == 0 || flags & FLAG_DELETED != 0 {
                    return NO_UPPER;
                }
                vread(p.add(S_UPPER_IDX) as *const u32)
            }
        }, self.slot_sanitizer(id), || NO_UPPER, self.owner_dead())
    }

    /// Copy `id`'s neighbor ids at upper `level` (1-based) into `out`. False when the node
    /// has no upper entry or no such level.
    pub fn upper_neighbors_into(&self, id: u32, level: u8, out: &mut Vec<u32>) -> bool {
        out.clear();
        debug_assert!(level >= 1);
        let idx = self.upper_idx_of(id);
        if idx == NO_UPPER || (idx as u64) >= self.file.upper_capacity || level as usize > MAX_UPPER_LEVELS {
            return false;
        }
        let seq = self.file.upper_seq_atomic(idx);
        seqlock::read_consistent(seq, self.file.self_tag, || {
            out.clear();
            let p = self.file.upper_ptr(idx);
            unsafe {
                let levels = vread(p.add(U_LEVELS));
                if level > levels {
                    return false;
                }
                let lp = p.add(U_LISTS + (level as usize - 1) * UPPER_LEVEL_STRIDE);
                let degree = u16::from_le(vread(lp.add(UL_DEGREE) as *const u16)) as usize;
                let base = lp.add(UL_IDS) as *const u32;
                for i in 0..degree.min(UPPER_CAP) {
                    out.push(u32::from_le(vread(base.add(i))));
                }
                true
            }
        }, self.upper_sanitizer(idx), || false, self.owner_dead())
    }

    /// Write a node's full upper adjacency into a fresh region entry; returns the entry
    /// index to store in the slot (NO_UPPER when the region is exhausted or levels is empty).
    pub fn write_upper(&self, levels: &[Vec<u32>]) -> Result<u32, Wedged> {
        if levels.is_empty() {
            return Ok(NO_UPPER);
        }
        let idx = self.file.allocate_upper();
        if idx == NO_UPPER {
            return Ok(NO_UPPER);
        }
        let seq = self.file.upper_seq_atomic(idx);
        let _guard = seqlock::write_lock(seq, self.file.self_tag, self.upper_sanitizer(idx), self.owner_dead())?;
        let p = self.file.upper_ptr_mut(idx);
        unsafe {
            let n = levels.len().min(MAX_UPPER_LEVELS);
            *p.add(U_LEVELS) = n as u8;
            for (l, list) in levels.iter().take(n).enumerate() {
                let lp = p.add(U_LISTS + l * UPPER_LEVEL_STRIDE);
                let deg = list.len().min(UPPER_CAP);
                (lp.add(UL_DEGREE) as *mut u16).write_unaligned((deg as u16).to_le());
                let base = lp.add(UL_IDS) as *mut u32;
                for (i, id) in list.iter().take(deg).enumerate() {
                    base.add(i).write_unaligned(id.to_le());
                }
            }
        }
        Ok(idx)
    }

    /// Rewrite an existing upper entry in place (full state). Used by the raw mirroring
    /// path so repeated updates to a high-level node reuse its entry instead of leaking one
    /// per rewrite.
    pub fn rewrite_upper(&self, idx: u32, levels: &[Vec<u32>]) -> Result<(), Wedged> {
        let seq = self.file.upper_seq_atomic(idx);
        let _guard = seqlock::write_lock(seq, self.file.self_tag, self.upper_sanitizer(idx), self.owner_dead())?;
        let p = self.file.upper_ptr_mut(idx);
        unsafe {
            let n = levels.len().min(MAX_UPPER_LEVELS);
            *p.add(U_LEVELS) = n as u8;
            for (l, list) in levels.iter().take(n).enumerate() {
                let lp = p.add(U_LISTS + l * UPPER_LEVEL_STRIDE);
                let deg = list.len().min(UPPER_CAP);
                (lp.add(UL_DEGREE) as *mut u16).write_unaligned((deg as u16).to_le());
                let base = lp.add(UL_IDS) as *mut u32;
                for (i, id) in list.iter().take(deg).enumerate() {
                    base.add(i).write_unaligned(id.to_le());
                }
            }
        }
        Ok(())
    }

    /// Whether a slot has ever been written (valid or deleted) — the builder scan's
    /// skip-if-touched check.
    pub fn node_touched(&self, id: u32) -> bool {
        if !self.in_range(id) {
            return false;
        }
        let seq = self.file.seq_atomic(id);
        seqlock::read_consistent(seq, self.file.self_tag, || unsafe { vread(self.file.slot_ptr(id).add(S_FLAGS)) != 0 }, self.slot_sanitizer(id), || true, self.owner_dead())
    }

    /// The slot's stored upper idx regardless of valid/deleted flags. Taken under the slot
    /// write lock rather than `read_consistent`, whose NO_UPPER fallback cannot be told apart
    /// from an unbound slot — reusing it as one mints a second entry for an id that already
    /// owns one.
    fn upper_idx_locked(&self, id: u32) -> Result<u32, Wedged> {
        if !self.in_range(id) {
            return Ok(NO_UPPER);
        }
        let seq = self.file.seq_atomic(id);
        let _guard = seqlock::write_lock(seq, self.file.self_tag, self.slot_sanitizer(id), self.owner_dead())?;
        let p = self.file.slot_ptr(id);
        Ok(unsafe {
            if *p.add(S_FLAGS) == 0 {
                NO_UPPER // never written
            } else {
                (p.add(S_UPPER_IDX) as *const u32).read_unaligned()
            }
        })
    }

    /// Mirror a host-maintained node into the plane: full state per call, host-allocated id
    /// (high-water is raised, the plane allocator is bypassed), upper entry reused in place
    /// when present. This is the dual-write phase-1 write path; the stored key is kept.
    pub fn write_node_raw(
        &self,
        id: u32,
        level: u8,
        vector: &[u8],
        scale: f32,
        inv_mag: f32,
        neighbors: &[u32],
        upper_levels: &[Vec<u32>],
    ) -> Result<(), WriteError> {
        self.write_node_raw_with_key(id, level, vector, scale, inv_mag, neighbors, upper_levels, None)
    }

    /// `write_node_raw` carrying the host's key bytes (`check_key` must have passed), or None
    /// to keep the stored key.
    #[allow(clippy::too_many_arguments)]
    pub fn write_node_raw_with_key(
        &self,
        id: u32,
        level: u8,
        vector: &[u8],
        scale: f32,
        inv_mag: f32,
        neighbors: &[u32],
        upper_levels: &[Vec<u32>],
        key: Option<&[u8]>,
    ) -> Result<(), WriteError> {
        self.check_stored_vector(vector)?;
        self.file.ensure_high_water(id);
        let existing = match self.upper_idx_locked(id)? {
            idx if idx != NO_UPPER && (idx as u64) >= self.file.upper_capacity => NO_UPPER, // corrupt stored index
            idx => idx,
        };
        let mut fresh = NO_UPPER;
        let upper_idx = if upper_levels.is_empty() {
            // the host reseeds its id counter to largestNodeId + 1 on restart, so an id can be
            // re-minted at level 0 over a slot that had a hierarchy; that entry must stop being
            // readable. Emptied in place rather than freed: the freelist hand-off is not atomic
            // with publishing the slot below, so a mirror that read this index first could
            // republish a slot pointing at an entry already given to another node. One idle
            // entry per id is the bounded retention DESIGN.md §10 accepts.
            if existing != NO_UPPER {
                self.rewrite_upper(existing, &[])?;
            }
            existing
        } else if existing != NO_UPPER {
            self.rewrite_upper(existing, upper_levels)?;
            existing
        } else {
            fresh = self.write_upper(upper_levels)?;
            fresh
        };
        let mut l0 = neighbors.to_vec();
        l0.truncate(self.file.layer0_cap);
        if let Err(error) = self.write_node(id, level, vector, scale, inv_mag, &l0, upper_idx, key) {
            self.file.free_upper(fresh); // unreachable from any slot until publication succeeds
            return Err(error);
        }
        Ok(())
    }

    /// Mark deleted WITHOUT returning the id to the plane freelist — dual-write mode, where
    /// the host owns id allocation and may re-mint or reuse ids on its own schedule.
    pub fn clear_node(&self, id: u32) -> Result<(), Wedged> {
        if (id as u64) >= self.file.max_nodes {
            return Ok(());
        }
        // extend the high-water rather than skipping: a delete mirrored while a backfill
        // scan runs must leave a touched (deleted) slot behind, or the scan's older
        // snapshot would resurrect the node when its cursor reaches this id
        self.file.ensure_high_water(id);
        let seq = self.file.seq_atomic(id);
        let _guard = seqlock::write_lock(seq, self.file.self_tag, self.slot_sanitizer(id), self.owner_dead())?;
        let p = self.file.slot_ptr_mut(id);
        unsafe {
            if *p.add(S_FLAGS) == 0 {
                // tombstoning a never-written slot: its zero-initialized upper_idx would
                // otherwise read as the VALID index 0, and a later raw rewrite of this id
                // would clobber upper entry 0 — another node's hierarchy
                (p.add(S_UPPER_IDX) as *mut u32).write_unaligned(NO_UPPER);
            }
            *p.add(S_FLAGS) = FLAG_DELETED;
        }
        Ok(())
    }

    /// Atomic read-modify-write of `id`'s upper adjacency at `level` (1-based). Returns
    /// false when the node has no entry or level. `f` may read other slots.
    pub fn update_upper_level<F: FnOnce(&mut Vec<u32>)>(&self, id: u32, level: u8, f: F) -> Result<bool, Wedged> {
        let idx = self.upper_idx_of(id);
        if idx == NO_UPPER || (idx as u64) >= self.file.upper_capacity || level as usize > MAX_UPPER_LEVELS {
            return Ok(false);
        }
        let seq = self.file.upper_seq_atomic(idx);
        let _guard = seqlock::write_lock(seq, self.file.self_tag, self.upper_sanitizer(idx), self.owner_dead())?;
        let p = self.file.upper_ptr_mut(idx);
        unsafe {
            let levels = *p.add(U_LEVELS);
            if level > levels {
                return Ok(false);
            }
            let lp = p.add(U_LISTS + (level as usize - 1) * UPPER_LEVEL_STRIDE);
            let degree = u16::from_le((lp.add(UL_DEGREE) as *const u16).read_unaligned()) as usize;
            let base = lp.add(UL_IDS) as *mut u32;
            let mut list: Vec<u32> = (0..degree.min(UPPER_CAP)).map(|i| u32::from_le(base.add(i).read_unaligned())).collect();
            f(&mut list);
            list.truncate(UPPER_CAP);
            (lp.add(UL_DEGREE) as *mut u16).write_unaligned((list.len() as u16).to_le());
            for (i, id) in list.iter().enumerate() {
                base.add(i).write_unaligned(id.to_le());
            }
        }
        Ok(true)
    }

    /// Seqlock-consistent full copy (construction paths).
    pub fn read_node(&self, id: u32) -> Option<NodeRead> {
        if !self.in_range(id) {
            return None;
        }
        let seq = self.file.seq_atomic(id);
        let vector_bytes = self.file.vector_bytes();
        let cap = self.file.layer0_cap;
        let nbase_off = self.file.neighbor_offset();
        seqlock::read_consistent(seq, self.file.self_tag, || {
            let p = self.file.slot_ptr(id);
            unsafe {
                let flags = vread(p.add(S_FLAGS));
                if flags & FLAG_VALID == 0 || flags & FLAG_DELETED != 0 {
                    return None;
                }
                let level = vread(p.add(S_LEVEL));
                let degree = u16::from_le(vread(p.add(S_DEGREE) as *const u16)) as usize;
                let scale = vread(p.add(S_SCALE) as *const f32);
                let inv_mag = vread(p.add(S_INV_MAG) as *const f32);
                let vector = std::slice::from_raw_parts(p.add(S_VECTOR), vector_bytes).to_vec();
                let nbase = p.add(nbase_off) as *const u32;
                let neighbors = (0..degree.min(cap)).map(|i| u32::from_le(vread(nbase.add(i)))).collect();
                Some(NodeRead { level, scale, inv_mag, vector, neighbors })
            }
        }, self.slot_sanitizer(id), || None, self.owner_dead())
    }

    /// Write a full slot under its seqlock. `neighbors` is pruned to layer0_cap by the
    /// caller; `upper_idx` is a write_upper() result (NO_UPPER for level-0 nodes); `key` is
    /// the host's key bytes (`check_key` passed; ignored on a plane without key capacity), or
    /// None to keep the key the slot already holds.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_node(&self, id: u32, level: u8, vector: &[u8], scale: f32, inv_mag: f32, neighbors: &[u32], upper_idx: u32, key: Option<&[u8]>) -> Result<(), WriteError> {
        self.write_node_with_key_range(id, level, vector, scale, inv_mag, neighbors, upper_idx, key, None)
    }

    /// `write_node` with an arena range the caller reserved for an overflow key before it
    /// touched the graph, so the write cannot fail on the arena after edges were removed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_node_with_key_range(&self, id: u32, level: u8, vector: &[u8], scale: f32, inv_mag: f32, neighbors: &[u32], upper_idx: u32, key: Option<&[u8]>, reserved_key_range: Option<u64>) -> Result<(), WriteError> {
        debug_assert!(neighbors.len() <= self.file.layer0_cap);
        // the copy below is sized by this, so an over-long slice would write through the
        // neighbor array into the following slot. The element-domain scan stays at the raw
        // entry points, off the insert hot path.
        if vector.len() != self.file.vector_bytes() {
            return Err(WriteError::BadVector("vector byte length does not match the plane's dims x element size"));
        }
        debug_assert!(self.file.quant() != Quant::Int16 || int16_bytes_in_domain(vector));
        let seq = self.file.seq_atomic(id);
        let _guard = seqlock::write_lock(seq, self.file.self_tag, self.slot_sanitizer(id), self.owner_dead())?;
        let p = self.file.slot_ptr_mut(id);
        let nbase = self.file.neighbor_offset();
        unsafe {
            // key first: an exhausted arena leaves the slot's previous state intact
            self.store_key_locked(p, key, reserved_key_range)?;
            *p.add(S_LEVEL) = level;
            (p.add(S_DEGREE) as *mut u16).write_unaligned((neighbors.len() as u16).to_le());
            (p.add(S_SCALE) as *mut f32).write_unaligned(scale);
            (p.add(S_INV_MAG) as *mut f32).write_unaligned(inv_mag);
            (p.add(S_UPPER_IDX) as *mut u32).write_unaligned(upper_idx);
            std::ptr::copy_nonoverlapping(vector.as_ptr(), p.add(S_VECTOR), vector.len());
            for (i, n) in neighbors.iter().enumerate() {
                (p.add(nbase + i * 4) as *mut u32).write_unaligned(n.to_le());
            }
            // valid last within the locked section; the seqlock release publishes it
            *p.add(S_FLAGS) = FLAG_VALID;
        }
        drop(_guard);
        // after the release: a probe that consumed a bump while the slot was still invalid
        // would otherwise latch on a graph that holds a live node
        self.node_written();
        Ok(())
    }

    /// Atomic read-modify-write of a node's layer-0 neighbor list under its seqlock.
    /// `f` may read OTHER slots (e.g. distance_between for pruning) — those are plain
    /// unlocked reads, so no lock ordering issue — but must not lock this graph's slots.
    /// Returns false for absent/deleted nodes.
    pub fn update_neighbors<F: FnOnce(&mut Vec<u32>)>(&self, id: u32, f: F) -> Result<bool, Wedged> {
        if !self.in_range(id) {
            return Ok(false);
        }
        let seq = self.file.seq_atomic(id);
        let _guard = seqlock::write_lock(seq, self.file.self_tag, self.slot_sanitizer(id), self.owner_dead())?;
        let p = self.file.slot_ptr_mut(id);
        let cap = self.file.layer0_cap;
        unsafe {
            let flags = *p.add(S_FLAGS);
            if flags & FLAG_VALID == 0 || flags & FLAG_DELETED != 0 {
                return Ok(false);
            }
            let degree = u16::from_le((p.add(S_DEGREE) as *const u16).read_unaligned()) as usize;
            let base = p.add(self.file.neighbor_offset()) as *mut u32;
            let mut list: Vec<u32> = (0..degree.min(cap)).map(|i| u32::from_le(base.add(i).read_unaligned())).collect();
            f(&mut list);
            list.truncate(cap);
            (p.add(S_DEGREE) as *mut u16).write_unaligned((list.len() as u16).to_le());
            for (i, n) in list.iter().enumerate() {
                base.add(i).write_unaligned(n.to_le());
            }
        }
        Ok(true)
    }

    /// Apply a precomputed neighbor list only if the current list still equals `expected` —
    /// the compare and the write share one lock acquisition, so heavy work (distance-based
    /// pruning, which can major-fault) happens OUTSIDE the lock and the critical section
    /// stays microseconds. Returns false when the list changed or the node is gone.
    pub fn set_neighbors_if(&self, id: u32, expected: &[u32], next: &[u32]) -> Result<bool, Wedged> {
        debug_assert!(next.len() <= self.file.layer0_cap);
        if !self.in_range(id) {
            return Ok(false);
        }
        let seq = self.file.seq_atomic(id);
        let _guard = seqlock::write_lock(seq, self.file.self_tag, self.slot_sanitizer(id), self.owner_dead())?;
        let p = self.file.slot_ptr_mut(id);
        unsafe {
            if *p.add(S_FLAGS) != FLAG_VALID {
                return Ok(false);
            }
            let degree = u16::from_le((p.add(S_DEGREE) as *const u16).read_unaligned()) as usize;
            if degree != expected.len() {
                return Ok(false);
            }
            let base = p.add(self.file.neighbor_offset()) as *mut u32;
            for (i, want) in expected.iter().enumerate() {
                if u32::from_le(base.add(i).read_unaligned()) != *want {
                    return Ok(false);
                }
            }
            (p.add(S_DEGREE) as *mut u16).write_unaligned((next.len() as u16).to_le());
            for (i, n) in next.iter().enumerate() {
                base.add(i).write_unaligned(n.to_le());
            }
        }
        Ok(true)
    }

    /// Replace only the neighbor list (single-writer construction path).
    pub fn write_neighbors(&self, id: u32, neighbors: &[u32]) -> Result<(), Wedged> {
        debug_assert!(neighbors.len() <= self.file.layer0_cap);
        let seq = self.file.seq_atomic(id);
        let _guard = seqlock::write_lock(seq, self.file.self_tag, self.slot_sanitizer(id), self.owner_dead())?;
        let p = self.file.slot_ptr_mut(id);
        let nbase = self.file.neighbor_offset();
        unsafe {
            (p.add(S_DEGREE) as *mut u16).write_unaligned((neighbors.len() as u16).to_le());
            for (i, n) in neighbors.iter().enumerate() {
                (p.add(nbase + i * 4) as *mut u32).write_unaligned(n.to_le());
            }
        }
        Ok(())
    }

    /// Mark deleted (traversals skip it), free its upper entry, and return the id to the
    /// plane freelist. Deleting the current entry point re-elects a replacement — without
    /// that, every search returns empty and every insert orphans itself against the dead
    /// entry.
    pub fn delete_node(&self, id: u32) -> Result<(), Wedged> {
        if !self.in_range(id) {
            return Ok(()); // never-allocated or out-of-range ids have nothing to delete
        }
        // capture neighbors before invalidating: they are the best re-election candidates
        let (entry_id, _) = self.file.entry_point();
        let mut candidates: Vec<u32> = Vec::new();
        if entry_id == id {
            self.neighbors_into(id, &mut candidates);
        }
        // Re-elect before the tombstone, not after: between marking the slot deleted and
        // installing a replacement, every concurrent search routes through a node that reads
        // as absent and returns nothing. The node is still live here, so a crash inside the
        // window leaves the header naming a live entry either way.
        if entry_id == id {
            self.reelect_entry_point_replacing(&candidates, id);
        }
        let upper_idx;
        {
            let seq = self.file.seq_atomic(id);
            let _guard = seqlock::write_lock(seq, self.file.self_tag, self.slot_sanitizer(id), self.owner_dead())?;
            let p = self.file.slot_ptr_mut(id);
            unsafe {
                if *p.add(S_FLAGS) != FLAG_VALID {
                    // deleting a never-written or already-deleted id must not free again:
                    // a double-push makes the freelist a self-cycle that hands the same id
                    // to every subsequent allocation
                    return Ok(());
                }
                upper_idx = (p.add(S_UPPER_IDX) as *const u32).read_unaligned();
                (p.add(S_UPPER_IDX) as *mut u32).write_unaligned(NO_UPPER);
                *p.add(S_FLAGS) = FLAG_DELETED;
            }
        }
        if upper_idx != NO_UPPER && (upper_idx as u64) < self.file.upper_capacity {
            // empty the entry under its own lock BEFORE freeing: a traversal that already
            // read this node's upper_idx must find a dead entry, not one reallocated to a
            // different node mid-read
            self.rewrite_upper(upper_idx, &[])?;
        }
        self.file.free_upper(upper_idx);
        self.file.free_id(id);
        Ok(())
    }

    /// Pick a new entry point: the highest-level live node among `preferred`, else the
    /// first live node found scanning the id range (rare path: only when the entry's whole
    /// neighborhood is gone). An empty graph clears the entry.
    /// A node's level without copying its vector or edges (cheap re-election scans).
    pub(crate) fn node_level(&self, id: u32) -> Option<u8> {
        if !self.in_range(id) {
            return None;
        }
        let seq = self.file.seq_atomic(id);
        seqlock::read_consistent(seq, self.file.self_tag, || {
            let p = self.file.slot_ptr(id);
            unsafe {
                if vread(p.add(S_FLAGS)) != FLAG_VALID {
                    return None;
                }
                Some(vread(p.add(S_LEVEL)))
            }
        }, self.slot_sanitizer(id), || None, self.owner_dead())
    }

    /// Highest-level live node among at most `limit` probes, skipping `skip`. The read-side
    /// repair's last resort, bounded because `reelect_entry_point_replacing`'s scan runs to the
    /// high-water mark and a search on the shared pool thread cannot afford it.
    ///
    /// Walks down from the newest id with a stride spanning the whole range, so it assumes
    /// nothing about where the live nodes sit: Harper allocates ids monotonically and never
    /// reuses them, so a churned table's low prefix is all tombstones, while the crate's own
    /// freelist reuses ids and keeps live nodes low.
    ///
    /// The start rotates per handle, so `stride` consecutive repairs of this plane cover every id
    /// while each stays capped at `limit`; a fixed start would probe one residue class forever
    /// and leave a graph lying between its samples invisible permanently, not for one search.
    /// That coverage rests on `stride * limit >= hw`, which is why the stride is a ceiling
    /// division: below it a walk stops short of id 0 and no offset ever reaches the tail.
    ///
    /// Best-level rather than first-live: a level-0 entry degrades every later search to a
    /// layer-0-only beam.
    pub(crate) fn probe_for_entry(&self, limit: u32, skip: u32) -> Option<(u32, u8)> {
        let hw = self.file.id_high_water().min(self.file.max_nodes) as u32;
        if hw == 0 || limit == 0 {
            return None;
        }
        let stride = hw.div_ceil(limit);
        use std::sync::atomic::Ordering::Relaxed;
        let epoch = self.file.write_epoch();
        let unchanged = self.probe_futile_hw.load(Relaxed) == hw as u64 && self.probe_futile_epoch.load(Relaxed) == epoch;
        if unchanged && self.probe_futile_runs.load(Relaxed) >= stride {
            return None; // every residue probed since the last write anywhere: nothing to find
        }
        if self.probe_in_flight.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return None; // another search on this handle is repairing; it publishes for both
        }
        let offset = self.probe_rotation.fetch_add(1, Relaxed) % stride;
        let mut best: Option<(u32, u8)> = None;
        let mut cand = hw - 1 - offset;
        for _ in 0..limit {
            if cand != skip {
                if let Some(level) = self.node_level(cand) {
                    if best.map(|(_, l)| level > l).unwrap_or(true) {
                        best = Some((cand, level));
                    }
                }
            }
            if cand < stride {
                break;
            }
            cand -= stride;
        }
        if best.is_some() {
            self.probe_futile_runs.store(0, Relaxed);
        } else if unchanged {
            self.probe_futile_runs.fetch_add(1, Relaxed);
        } else {
            self.probe_futile_hw.store(hw as u64, Relaxed);
            self.probe_futile_epoch.store(epoch, Relaxed);
            self.probe_futile_runs.store(1, Relaxed);
        }
        self.probe_in_flight.store(false, std::sync::atomic::Ordering::Release);
        best
    }

    /// Pick a new entry point: the highest-level live node among `preferred`, else the
    /// highest-level live node found scanning the id range (level reads only — no per-node
    /// vector copies; still O(high-water), which only runs when an entry point vanished
    /// with no live neighborhood). Preferring level keeps the hierarchy navigable — a
    /// level-0 entry degrades every search to a layer-0-only beam. An empty graph clears
    /// the entry.
    pub(crate) fn reelect_entry_point_replacing(&self, preferred: &[u32], replacing: u32) {
        let mut best: Option<(u32, u8)> = None;
        // the most recently replaced entry point is the best cheap candidate: usually alive,
        // usually high-level — and it makes the full fallback scan a last resort
        let prev = self.file.previous_entry_point();
        if prev != crate::format::NO_ID && prev != replacing {
            if let Some(level) = self.node_level(prev) {
                best = Some((prev, level));
            }
        }
        for &cand in preferred {
            if cand == replacing {
                continue; // the node on its way out is never its own replacement
            }
            if let Some(level) = self.node_level(cand) {
                if best.map(|(_, l)| level > l).unwrap_or(true) {
                    best = Some((cand, level));
                }
            }
        }
        if best.is_none() {
            let hw = self.file.id_high_water().min(self.file.max_nodes) as u32;
            for cand in 0..hw {
                if cand == replacing {
                    continue;
                }
                if let Some(level) = self.node_level(cand) {
                    if best.map(|(_, l)| level > l).unwrap_or(true) {
                        best = Some((cand, level));
                        if level as usize >= MAX_UPPER_LEVELS {
                            break; // cannot do better
                        }
                    }
                }
            }
        }
        match best {
            Some((cand, level)) => self.file.set_entry_point_if_not_better(cand, level as u32, replacing),
            None => self.file.clear_entry_point_if(replacing),
        }
    }

    /// write_node, but only when the slot has never been touched — the check and the write
    /// share ONE seqlock acquisition, so a concurrent live mirror's newer write can never be
    /// overwritten by a backfill scan's older snapshot (a two-step check-then-write left
    /// exactly that window). Returns true when this state was written.
    #[allow(clippy::too_many_arguments)]
    pub fn write_node_if_untouched(
        &self,
        id: u32,
        level: u8,
        vector: &[u8],
        scale: f32,
        inv_mag: f32,
        neighbors: &[u32],
        upper_levels: &[Vec<u32>],
        key: Option<&[u8]>,
    ) -> Result<bool, WriteError> {
        debug_assert!(neighbors.len() <= self.file.layer0_cap);
        self.check_stored_vector(vector)?;
        self.file.ensure_high_water(id);
        // the upper entry is allocated before taking the slot lock (allocation is cheap); it is
        // unreachable from any slot until the write below lands, so every path that does not
        // publish it — a wedged lock, a slot that turns out to be touched — has to free it
        let upper_idx = if upper_levels.is_empty() { NO_UPPER } else { self.write_upper(upper_levels)? };
        let seq = self.file.seq_atomic(id);
        let written = {
            let _guard = match seqlock::write_lock(seq, self.file.self_tag, self.slot_sanitizer(id), self.owner_dead())
            {
                Ok(guard) => guard,
                Err(wedged) => {
                    self.file.free_upper(upper_idx);
                    return Err(wedged.into());
                }
            };
            let p = self.file.slot_ptr_mut(id);
            let nbase = self.file.neighbor_offset();
            unsafe {
                if *p.add(S_FLAGS) != 0 {
                    false
                } else {
                    if let Err(error) = self.store_key_locked(p, key, None) {
                        self.file.free_upper(upper_idx);
                        return Err(error);
                    }
                    *p.add(S_LEVEL) = level;
                    (p.add(S_DEGREE) as *mut u16).write_unaligned((neighbors.len() as u16).to_le());
                    (p.add(S_SCALE) as *mut f32).write_unaligned(scale);
                    (p.add(S_INV_MAG) as *mut f32).write_unaligned(inv_mag);
                    (p.add(S_UPPER_IDX) as *mut u32).write_unaligned(upper_idx);
                    std::ptr::copy_nonoverlapping(vector.as_ptr(), p.add(S_VECTOR), vector.len());
                    for (i, n) in neighbors.iter().enumerate() {
                        (p.add(nbase + i * 4) as *mut u32).write_unaligned(n.to_le());
                    }
                    *p.add(S_FLAGS) = FLAG_VALID;
                    true
                }
            }
        };
        if written {
            self.node_written();
        } else {
            self.file.free_upper(upper_idx);
        }
        Ok(written)
    }
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    use crate::insert::{insert, InsertParams};
    use crate::search::{search, SearchScratch};
    use std::sync::atomic::Ordering::Relaxed;

    fn vector_for(i: u32, dims: usize) -> Vec<f32> {
        (0..dims).map(|d| ((i as f32 * 0.31 + d as f32) * 0.7).sin()).collect()
    }

    /// A graph whose every node is dead must stop paying the repair probe once a full rotation
    /// has come back empty, and must resume it after ANY handle writes a node — the revival
    /// here comes through a second handle on the same file, as another process's would.
    #[test]
    fn a_fully_dead_graph_stops_probing_until_a_node_is_written() {
        let dims = 32;
        let path = std::env::temp_dir().join(format!("hnsw-probefutile-{}.hnsw", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let graph = Graph::new(PlaneFile::create(&path, dims, 16, 4_096).expect("create"));
        let params = InsertParams::default();
        let mut scratch = SearchScratch::new();
        for i in 0..2_100 {
            insert(&graph, &vector_for(i, dims), &params, &mut scratch).unwrap();
        }
        let hw = graph.file.id_high_water() as u32;
        let stride = hw.div_ceil(1_024); // REPAIR_PROBE_LIMIT
        assert!(stride > 1, "precondition: a rotation the guard has to wait out");
        let (entry, _) = graph.file.entry_point();
        for id in 0..hw {
            let _ = graph.clear_node(id);
        }
        graph.file.clear_entry_point_if(entry);

        let query = graph.query(vector_for(7, dims));
        for _ in 0..stride {
            assert!(search(&graph, &query, 5, 64, &mut scratch).0.is_empty());
        }
        let rotation = graph.probe_rotation.load(Relaxed);
        assert!(search(&graph, &query, 5, 64, &mut scratch).0.is_empty());
        assert_eq!(graph.probe_rotation.load(Relaxed), rotation, "a probed-out plane must not probe again");

        // a node written through another handle, with no entry-point update (mirroring hosts do
        // not always re-elect), must be findable again within one rotation
        let revived = 3u32;
        let q = crate::distance::quantize_int8(&vector_for(revived, dims));
        let other = Graph::new(PlaneFile::open(&path).expect("a second handle"));
        other.write_node_raw(revived, 0, &q.0, q.1, q.2, &[], &[]).expect("revive");
        let found = (0..stride).any(|_| !search(&graph, &graph.query(vector_for(revived, dims)), 5, 64, &mut scratch).0.is_empty());
        assert!(found, "a write must re-arm the probe");
        let _ = std::fs::remove_file(&path);
    }
}
