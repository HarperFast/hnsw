//! Path-level invalidation: make a plane file that could not be deleted unadoptable, durably.
//!
//! Two markers, both attempted every call: the in-band latch (`PlaneFile::invalidate` — the
//! sticky header byte plus a zeroed watermark, msync'd) and a `<path>.stale` sidecar, fsync'd
//! along with its directory entry. `PlaneFile::open` refuses a file carrying either, so the
//! markers are enforced by the package, not by each host's attach path. In band first: the
//! sidecar is what a process that cannot map the file checks, the latch is what covers a
//! plane whose sidecar a crash lost. A temporary handle opened here is dropped before the
//! sidecar is written and before returning — its mapping is the kind of thing that keeps a
//! file undeletable in the first place, and its registry slot must not wait on a finalizer.

use crate::format::PlaneFile;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

/// Which markers landed. `Ok` for a marker means it is durable, not merely written.
#[derive(Debug)]
pub struct Invalidation {
    pub in_band: io::Result<()>,
    pub sidecar: io::Result<()>,
}

/// The sidecar convention: `<plane path>.stale`. Its presence means the plane file is stale
/// and must never be opened; hosts delete both and rebuild.
pub fn stale_path_for(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".stale");
    PathBuf::from(name)
}

/// Invalidate the plane at `path` through a temporary handle. Returns `Err` only when NEITHER
/// marker became durable. Nothing here deletes or renames, so the caller keeps whatever
/// recovery state it had; an in-band mark whose msync failed may still have landed in the
/// shared mapping, which is the safe direction (every handle reads it as incomplete).
/// Idempotent: an already invalidated plane reports both markers again.
pub fn invalidate_plane(path: &Path) -> io::Result<Invalidation> {
    invalidate_at(path, None)
}

/// Invalidate the file `plane` maps, in band through the handle itself and with the sidecar
/// next to the path it was opened at. The caller's own handle is the right one when it
/// exists: on Windows that mapping is why the unlink failed, and a second open would claim a
/// second registry slot for nothing. The path must not have been replaced underneath the
/// handle since it opened; a host that unlinked and recreated it has nothing to invalidate.
pub fn invalidate_file(plane: &PlaneFile) -> io::Result<Invalidation> {
    invalidate_at(&plane.path, Some(plane))
}

fn invalidate_at(path: &Path, attached: Option<&PlaneFile>) -> io::Result<Invalidation> {
    let in_band = match attached {
        Some(plane) => plane.invalidate(),
        None => PlaneFile::open_for_invalidation(path).and_then(|plane| plane.invalidate()),
    };
    let sidecar = write_sidecar(&stale_path_for(path));
    if let (Err(in_band), Err(sidecar)) = (&in_band, &sidecar) {
        return Err(io::Error::other(format!(
            "neither invalidation marker is durable for {}: in-band: {in_band}; sidecar: {sidecar}",
            path.display()
        )));
    }
    Ok(Invalidation { in_band, sidecar })
}

/// Create-new rather than create: the plane directory may be writable by another principal,
/// and a planted symlink at the sidecar path would otherwise be followed and its target
/// truncated. An existing marker is re-synced through a no-follow open checked on the open
/// handle, so a swap between the two calls cannot redirect the sync either.
fn write_sidecar(stale: &Path) -> io::Result<()> {
    let marker = match OpenOptions::new().write(true).create_new(true).open(stale) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => open_existing_marker(stale)?,
        Err(e) => return Err(e),
    };
    marker.sync_all()?;
    sync_dir(parent_dir(stale))
}

fn open_existing_marker(stale: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_NONBLOCK: a FIFO planted here must fail (ENXIO) rather than block the open
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(stale)?;
    if !file.metadata()?.file_type().is_file() {
        return Err(io::Error::other(format!("{} exists and is not a regular file", stale.display())));
    }
    Ok(file)
}

fn parent_dir(stale: &Path) -> &Path {
    stale.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
}

/// Make the directory entry durable. Windows has no directory fsync through `std` (a
/// directory handle needs backup semantics); there `FlushFileBuffers` on the marker itself
/// is documented to flush the metadata of its creation, so the marker's `sync_all` is the
/// durability point and this is a no-op.
#[cfg(unix)]
fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hnsw-invalidate-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("plane.hnsw")
    }

    fn complete_looking_plane(path: &Path) {
        let plane = PlaneFile::create(path, 8, 8, 256).expect("create");
        plane.flush_with_watermark(Some(4_096)).expect("barrier");
        assert_eq!(plane.watermark(), 4_096);
    }

    fn open_is_refused(path: &Path) {
        let err = PlaneFile::open(path).err().expect("an invalidated plane must not open");
        assert!(err.to_string().contains("invalidated"), "{err}");
    }

    #[test]
    fn both_markers_land_and_every_later_open_is_refused() {
        let path = tmp("both");
        complete_looking_plane(&path);
        let outcome = invalidate_plane(&path).expect("at least one marker");
        assert!(outcome.in_band.is_ok(), "{:?}", outcome.in_band);
        assert!(outcome.sidecar.is_ok(), "{:?}", outcome.sidecar);
        assert!(stale_path_for(&path).is_file());
        open_is_refused(&path);
    }

    #[test]
    fn invalidation_is_idempotent() {
        let path = tmp("twice");
        complete_looking_plane(&path);
        invalidate_plane(&path).expect("first");
        let again = invalidate_plane(&path).expect("second");
        assert!(again.in_band.is_ok() && again.sidecar.is_ok(), "{again:?}");
    }

    #[test]
    fn the_sidecar_is_named_next_to_the_plane() {
        assert_eq!(stale_path_for(Path::new("/data/t/a%2Fb.hnsw")), PathBuf::from("/data/t/a%2Fb.hnsw.stale"));
        assert_eq!(parent_dir(Path::new("plane.hnsw.stale")), Path::new("."));
        assert_eq!(parent_dir(Path::new("/data/t/plane.hnsw.stale")), Path::new("/data/t"));
    }

    /// A directory squatting the sidecar path makes its creation fail on every platform.
    #[test]
    fn in_band_alone_still_succeeds_when_the_sidecar_cannot_be_written() {
        let path = tmp("inbandonly");
        complete_looking_plane(&path);
        std::fs::create_dir(stale_path_for(&path)).unwrap();
        let outcome = invalidate_plane(&path).expect("the in-band mark is enough");
        assert!(outcome.in_band.is_ok());
        assert!(outcome.sidecar.is_err(), "a directory at the sidecar path must be reported");
        std::fs::remove_dir(stale_path_for(&path)).unwrap();
        open_is_refused(&path);
    }

    /// A file that is not a plane cannot carry the in-band mark; the sidecar must still land,
    /// and the temporary open that failed must not leave anything holding the file.
    #[test]
    fn the_sidecar_alone_still_succeeds_when_the_plane_cannot_be_opened() {
        let path = tmp("sidecaronly");
        std::fs::write(&path, b"not a plane").unwrap();
        let outcome = invalidate_plane(&path).expect("the sidecar is enough");
        assert!(outcome.in_band.is_err());
        assert!(outcome.sidecar.is_ok(), "{:?}", outcome.sidecar);
        assert!(stale_path_for(&path).is_file());
        std::fs::remove_file(&path).expect("nothing of ours may hold the file");
    }

    /// Neither marker: an error that names both causes, and nothing deleted or replaced — the
    /// caller's recovery state (retry later, stay disabled in-process) is preserved.
    #[test]
    fn a_double_failure_is_an_error_and_deletes_nothing() {
        let path = tmp("double");
        std::fs::write(&path, b"not a plane").unwrap();
        std::fs::create_dir(stale_path_for(&path)).unwrap();
        let err = invalidate_plane(&path).expect_err("no marker landed");
        let message = err.to_string();
        assert!(message.contains("in-band:") && message.contains("sidecar:"), "{message}");
        assert_eq!(std::fs::read(&path).unwrap(), b"not a plane");
        assert!(stale_path_for(&path).is_dir(), "nothing may be deleted or replaced");
        std::fs::remove_file(&path).expect("nothing of ours may hold the file");
    }

    /// The in-band mark must survive the writers that can still reach the header: a flush
    /// already in flight on this handle, and another handle's own watermark stamps. Without
    /// the sticky latch a `flushAsync(900)` racing the invalidation restored the old
    /// completion stamp and the plane was adoptable again.
    #[test]
    fn a_later_flush_or_stamp_cannot_revive_an_invalidated_plane() {
        let path = tmp("revive");
        complete_looking_plane(&path);
        let ours = PlaneFile::open(&path).expect("our handle");
        let theirs = PlaneFile::open(&path).expect("another worker's handle");
        let outcome = invalidate_file(&ours).expect("invalidate");
        assert!(outcome.in_band.is_ok() && outcome.sidecar.is_ok(), "{outcome:?}");
        ours.flush_with_watermark(Some(900)).expect("the pending flush lands after the invalidation");
        theirs.set_watermark(4_096);
        theirs.flush_with_watermark(None).expect("their cadence barrier");
        assert_eq!(ours.watermark(), 0, "an invalidated plane must read incomplete on every handle");
        assert_eq!(theirs.watermark(), 0);
        std::fs::remove_file(stale_path_for(&path)).unwrap();
        open_is_refused(&path);
    }

    /// A planted symlink at the sidecar path must not be followed, on the first invalidation
    /// (create-new) and on a repeat that finds the marker swapped for a link: the victim keeps
    /// its bytes and the sidecar is reported as not durable.
    #[cfg(unix)]
    #[test]
    fn a_symlink_at_the_sidecar_path_is_never_followed() {
        let path = tmp("symlink");
        complete_looking_plane(&path);
        let victim = path.with_file_name("victim.txt");
        std::fs::write(&victim, b"precious").unwrap();
        std::os::unix::fs::symlink(&victim, stale_path_for(&path)).unwrap();
        let outcome = invalidate_plane(&path).expect("in band still lands");
        assert!(outcome.sidecar.is_err(), "a symlink at the sidecar path must be refused");
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        let err = open_existing_marker(&stale_path_for(&path)).err().expect("the no-follow reopen must refuse a link");
        assert!(err.raw_os_error().is_some() || err.to_string().contains("regular file"), "{err}");
    }

    /// The package enforces the markers at open: a sidecar alone (the plane's own header was
    /// never reached) refuses the open, and a create over a leftover sidecar refuses too
    /// rather than minting a plane that can never be opened again.
    #[test]
    fn open_and_create_refuse_a_path_with_a_sidecar() {
        let path = tmp("sidecaropen");
        complete_looking_plane(&path);
        std::fs::write(stale_path_for(&path), b"").unwrap();
        open_is_refused(&path);
        let err = PlaneFile::create(&path, 8, 8, 256).err().expect("create must refuse");
        assert!(err.to_string().contains("stale"), "{err}");
        std::fs::remove_file(stale_path_for(&path)).unwrap();
        assert_eq!(PlaneFile::open(&path).expect("clean again").watermark(), 4_096);
    }

    /// The temporary handle must be gone before the call returns: the file is deletable
    /// (which its mapping would block on Windows) and its registry slot reads dead to a
    /// concurrent opener (Linux, where the registry exists).
    #[test]
    fn the_temporary_handle_is_released_before_returning() {
        let path = tmp("release");
        complete_looking_plane(&path);
        let observer = PlaneFile::open(&path).expect("a concurrent opener");
        invalidate_plane(&path).expect("invalidate");
        #[cfg(target_os = "linux")]
        {
            let registered: Vec<u32> = observer.registered_tags().into_iter().filter(|&t| t != observer.self_tag).collect();
            assert!(!registered.is_empty(), "the temporary open must have registered itself");
            for tag in registered {
                assert!(observer.tag_is_dead(tag), "registry tag {tag:#x} still reads alive after the call returned");
            }
        }
        drop(observer);
        std::fs::remove_file(&path).expect("no mapping of ours may hold the file");
    }

    /// Through the caller's own handle nothing is opened here, so no registry slot is claimed.
    #[test]
    fn an_attached_handle_means_no_temporary_open() {
        let path = tmp("attached");
        complete_looking_plane(&path);
        let attached = PlaneFile::open(&path).expect("open");
        let before = attached.registered_tags();
        let outcome = invalidate_file(&attached).expect("invalidate");
        assert!(outcome.in_band.is_ok() && outcome.sidecar.is_ok(), "{outcome:?}");
        assert_eq!(attached.registered_tags(), before, "no second opener may appear");
        assert_eq!(attached.watermark(), 0);
        assert!(stale_path_for(&path).is_file());
    }
}
