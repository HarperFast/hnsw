//! Kernel-level prefetch of slot pages. `Graph::prefetch_slot`'s CPU hint is dropped on a
//! non-resident page, so once the plane exceeds page cache every unvisited neighbour of an
//! expansion is a synchronous, queue-depth-1 major fault. `MADV_WILLNEED` over the batch of
//! slots about to be read starts all their reads at once. It is issued only while the search
//! gate in `search::unvisited_prefetched` sees fault-scale latency: measured on NVMe, the
//! vectored call costs ~0.5 µs per range when the pages are already resident, against a
//! ~0.13 µs resident visit, so always-on would dominate an in-cache query.
//!
//! The mapping stays `MADV_RANDOM`: this fetches exactly the pages the next distance reads
//! need, never a readahead window around them.

use std::sync::atomic::{AtomicU8, Ordering};

/// A page-aligned span of the mapping. Same layout as `iovec` so a batch can be handed to
/// `process_madvise` without copying; addresses are held as integers so the buffer stays `Send`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageRange {
    pub base: usize,
    pub len: usize,
}

impl PageRange {
    /// The pages covering `[start, end)`, aligned to the system page size (macOS arm64 pages
    /// are 16 KiB, and `madvise` rejects a span not aligned to them).
    pub fn covering(start: usize, end: usize) -> PageRange {
        let page = page_size();
        let base = start & !(page - 1);
        let stop = (end + page - 1) & !(page - 1);
        PageRange { base, len: stop - base }
    }
}

fn page_size() -> usize {
    static PAGE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE.get_or_init(|| {
        #[cfg(unix)]
        {
            let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if n > 0 {
                return n as usize;
            }
        }
        4096
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    /// Linux `process_madvise`: one syscall per batch.
    Vectored = 1,
    /// One `madvise` per range: older Linux (unprivileged self-advice needs 6.13) and macOS.
    /// Still asynchronous readahead, so the device work stays parallel.
    PerRange = 2,
    /// No kernel prefetch (Windows, or `HNSW_KERNEL_PREFETCH=0`).
    Off = 3,
}

const UNPROBED: u8 = 0;
static MODE: AtomicU8 = AtomicU8::new(UNPROBED);

/// The process's current backend, probed on first use. `HNSW_KERNEL_PREFETCH=0` is the kill
/// switch for a kernel where the advice misbehaves, or an A/B control; there is no "on" value
/// because the gate decides per expansion.
pub fn mode() -> Mode {
    match MODE.load(Ordering::Relaxed) {
        UNPROBED => {
            let off = std::env::var_os("HNSW_KERNEL_PREFETCH").is_some_and(|v| v == "0");
            let m = if off { Mode::Off } else { sys::initial_mode() };
            MODE.store(m as u8, Ordering::Relaxed);
            m
        }
        1 => Mode::Vectored,
        2 => Mode::PerRange,
        _ => Mode::Off,
    }
}

/// Prefetch every range with the probed backend; a failure that means the backend is
/// unusable in this process latches the next one down, so no expansion pays a failing
/// syscall twice. Returns whether a kernel request was issued.
pub fn willneed(ranges: &[PageRange]) -> bool {
    if ranges.is_empty() {
        return false;
    }
    let mut m = mode();
    loop {
        match willneed_in(m, ranges) {
            Outcome::Issued => return true,
            Outcome::Off => return false,
            Outcome::Unusable => {
                m = match m {
                    Mode::Vectored => Mode::PerRange,
                    _ => Mode::Off,
                };
                MODE.store(m as u8, Ordering::Relaxed);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Issued,
    /// The backend cannot work in this process (missing syscall, no permission, no pidfd).
    Unusable,
    Off,
}

/// Prefetch with an explicit backend, without touching the latched mode. Public so a test or
/// benchmark can exercise each backend on the host.
pub fn willneed_in(mode: Mode, ranges: &[PageRange]) -> Outcome {
    match mode {
        Mode::Vectored => sys::vectored(ranges),
        Mode::PerRange => sys::per_range(ranges),
        Mode::Off => Outcome::Off,
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use super::{Mode, Outcome, PageRange};
    use std::sync::atomic::{AtomicI32, Ordering};

    static PIDFD: AtomicI32 = AtomicI32::new(-1);
    static PIDFD_PID: AtomicI32 = AtomicI32::new(0);

    pub fn initial_mode() -> Mode {
        Mode::Vectored
    }

    /// A pidfd for the calling process, opened on first use and reopened after a fork so a
    /// child never advises its parent's address space.
    fn pidfd() -> Option<i32> {
        let pid = unsafe { libc::getpid() };
        let fd = PIDFD.load(Ordering::Acquire);
        if fd >= 0 && PIDFD_PID.load(Ordering::Acquire) == pid {
            return Some(fd);
        }
        let opened = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0u32) };
        if opened < 0 {
            return None;
        }
        let opened = opened as i32;
        // a concurrent opener may have won; the pid store is ordered after the fd store so a
        // reader that sees the new pid also sees the new fd
        match PIDFD.compare_exchange(fd, opened, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                PIDFD_PID.store(pid, Ordering::Release);
                Some(opened)
            }
            Err(current) => {
                unsafe { libc::close(opened) };
                Some(current)
            }
        }
    }

    pub fn vectored(ranges: &[PageRange]) -> Outcome {
        let Some(mut fd) = pidfd() else { return Outcome::Unusable };
        for chunk in ranges.chunks(libc::UIO_MAXIOV as usize) {
            // PageRange is repr(C) { usize, usize }: the layout of iovec
            let iov = chunk.as_ptr() as *const libc::iovec;
            let mut r = unsafe { libc::syscall(libc::SYS_process_madvise, fd, iov, chunk.len(), libc::MADV_WILLNEED, 0u32) };
            if r < 0 && matches!(errno(), libc::EBADF | libc::ESRCH) {
                // stale descriptor (pidfd of a pre-fork process): reopen once
                PIDFD_PID.store(0, Ordering::Release);
                let Some(again) = pidfd() else { return Outcome::Unusable };
                fd = again;
                r = unsafe { libc::syscall(libc::SYS_process_madvise, fd, iov, chunk.len(), libc::MADV_WILLNEED, 0u32) };
            }
            if r < 0 {
                return match errno() {
                    libc::ENOSYS | libc::EPERM | libc::EINVAL | libc::EBADF | libc::ESRCH => Outcome::Unusable,
                    // transient (ENOMEM, EAGAIN): the reads that follow are correct without it
                    _ => Outcome::Issued,
                };
            }
            // a short byte count is advisory too: the kernel stopped at an unmapped hole
        }
        Outcome::Issued
    }

    pub fn per_range(ranges: &[PageRange]) -> Outcome {
        super::unix_per_range(ranges)
    }

    fn errno() -> i32 {
        std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
mod sys {
    use super::{Mode, Outcome, PageRange};

    pub fn initial_mode() -> Mode {
        Mode::PerRange
    }

    pub fn vectored(_: &[PageRange]) -> Outcome {
        Outcome::Unusable
    }

    pub fn per_range(ranges: &[PageRange]) -> Outcome {
        super::unix_per_range(ranges)
    }
}

#[cfg(not(unix))]
mod sys {
    use super::{Mode, Outcome, PageRange};

    pub fn initial_mode() -> Mode {
        Mode::Off
    }

    pub fn vectored(_: &[PageRange]) -> Outcome {
        Outcome::Unusable
    }

    pub fn per_range(_: &[PageRange]) -> Outcome {
        Outcome::Unusable
    }
}

#[cfg(unix)]
fn unix_per_range(ranges: &[PageRange]) -> Outcome {
    for r in ranges {
        // errors are ignored: the advice is a hint and the read that follows is correct either way
        unsafe { libc::madvise(r.base as *mut libc::c_void, r.len, libc::MADV_WILLNEED) };
    }
    Outcome::Issued
}
