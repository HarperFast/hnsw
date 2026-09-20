//! Kernel-level prefetch of slot pages: `MADV_WILLNEED` over the slots one expansion is about
//! to read, so non-resident pages fault in parallel instead of one at a time. Issued only
//! while the gate in `search::unvisited_prefetched` is armed; the cost model and the reason it
//! cannot be always on are in DESIGN.md §7. The mapping stays `MADV_RANDOM`: this fetches
//! exactly the pages the next distance reads need, never a readahead window around them.

use std::sync::atomic::{AtomicU8, Ordering};

/// A page-aligned span of the mapping. `repr(C)` with the layout of `iovec`, which the
/// vectored backend relies on to hand a batch to the kernel without copying; the address is an
/// integer so the scratch buffer stays `Send`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageRange {
    pub base: usize,
    pub len: usize,
}

impl PageRange {
    /// The pages covering `[start, end)`, aligned to the system page size (16 KiB on macOS
    /// arm64, where `madvise` rejects a 4 KiB alignment).
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

/// The process's backend, probed on first use. `HNSW_KERNEL_PREFETCH=0` forces `Off`; there is
/// no "on" value because the gate decides per expansion.
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

/// Prefetch every range with the probed backend; a backend unusable in this process latches
/// the next one down, so no expansion pays a failing syscall twice. Returns whether a kernel
/// request was issued.
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

    /// A pidfd for the calling process, reopened after a fork so a child never advises its
    /// parent's address space (the inherited descriptor is left open: another thread may be
    /// mid-call on it, and it is close-on-exec).
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

    /// Every argument widened to `c_long`: `syscall` is variadic, and AAPCS64 leaves the upper
    /// half of a 32-bit variadic argument unspecified.
    fn process_madvise(fd: i32, chunk: &[PageRange]) -> libc::c_long {
        let iov = chunk.as_ptr() as *const libc::iovec;
        unsafe {
            libc::syscall(
                libc::SYS_process_madvise,
                fd as libc::c_long,
                iov as libc::c_long,
                chunk.len() as libc::c_long,
                libc::MADV_WILLNEED as libc::c_long,
                0 as libc::c_long,
            )
        }
    }

    pub fn vectored(ranges: &[PageRange]) -> Outcome {
        let Some(mut fd) = pidfd() else { return Outcome::Unusable };
        for chunk in ranges.chunks(libc::UIO_MAXIOV as usize) {
            let mut r = process_madvise(fd, chunk);
            if r < 0 && matches!(errno(), libc::EBADF | libc::ESRCH) {
                PIDFD_PID.store(0, Ordering::Release);
                let Some(again) = pidfd() else { return Outcome::Unusable };
                fd = again;
                r = process_madvise(fd, chunk);
            }
            if r < 0 {
                return match errno() {
                    libc::ENOSYS | libc::EPERM | libc::EINVAL | libc::EBADF | libc::ESRCH => Outcome::Unusable,
                    // ENOMEM, EAGAIN: transient, and the reads that follow are correct without it
                    _ => Outcome::Issued,
                };
            }
            // a short byte count stopped at a range the kernel would not advise; the rest of
            // the chunk faults as before
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

/// `Unusable` only when every range is refused, so a sandbox that filters `madvise` latches
/// off instead of paying `k` failing syscalls per armed expansion.
#[cfg(unix)]
fn unix_per_range(ranges: &[PageRange]) -> Outcome {
    let mut refused = 0;
    for r in ranges {
        if unsafe { libc::madvise(r.base as *mut libc::c_void, r.len, libc::MADV_WILLNEED) } != 0 {
            refused += 1;
        }
    }
    if refused == ranges.len() {
        Outcome::Unusable
    } else {
        Outcome::Issued
    }
}
