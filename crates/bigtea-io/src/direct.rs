//! A file opened for cache-bypassing positioned reads.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use crate::aligned::{align_down, align_up, AlignedBuf, ALIGN};

/// Whether the cache is actually being bypassed.
///
/// Recorded rather than assumed: a benchmark taken on a buffered handle is
/// measuring the page cache, and reporting it as disk throughput would be a
/// silent, order-of-magnitude lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMode {
    /// `FILE_FLAG_NO_BUFFERING` / `O_DIRECT` in force.
    Direct,
    /// The OS refused direct I/O; reads go through the page cache.
    Buffered,
}

impl std::fmt::Display for IoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoMode::Direct => f.write_str("direct (cache bypassed)"),
            IoMode::Buffered => f.write_str("buffered (page cache in use)"),
        }
    }
}

/// A file for reading weights: positioned, aligned, cache-bypassing when it can be.
#[derive(Debug)]
pub struct DirectFile {
    file: File,
    path: PathBuf,
    len: u64,
    mode: IoMode,
}

#[cfg(windows)]
const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
#[cfg(windows)]
const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;

#[cfg(target_os = "linux")]
const O_DIRECT: i32 = 0x4000;

impl DirectFile {
    /// Open `path` for reading, preferring direct I/O and falling back quietly.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        let (file, mode) = match Self::open_direct(&path) {
            Ok(f) => (f, IoMode::Direct),
            // Filesystems that refuse direct I/O are common enough (network
            // mounts especially) that this is a fallback, not a failure.
            Err(_) => (File::open(&path)?, IoMode::Buffered),
        };
        let len = file.metadata()?.len();
        Ok(DirectFile {
            file,
            path,
            len,
            mode,
        })
    }

    /// Open without attempting direct I/O. Useful for A/B measurement of what
    /// the page cache is actually doing for us.
    pub fn open_buffered(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let len = file.metadata()?.len();
        Ok(DirectFile {
            file,
            path,
            len,
            mode: IoMode::Buffered,
        })
    }

    #[cfg(windows)]
    fn open_direct(path: &Path) -> io::Result<File> {
        use std::os::windows::fs::OpenOptionsExt;
        OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN)
            .open(path)
    }

    #[cfg(target_os = "linux")]
    fn open_direct(path: &Path) -> io::Result<File> {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new()
            .read(true)
            .custom_flags(O_DIRECT)
            .open(path)
    }

    // macOS has no open-time flag; F_NOCACHE must be set afterwards via fcntl,
    // which needs raw FFI. Until that is written, be honest and report buffered.
    #[cfg(not(any(windows, target_os = "linux")))]
    fn open_direct(_path: &Path) -> io::Result<File> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "direct I/O not implemented on this platform",
        ))
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn mode(&self) -> IoMode {
        self.mode
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read exactly `len` bytes at `offset`.
    ///
    /// Handles the sector-alignment requirement internally: it reads the
    /// aligned superset of the range and copies out the exact window. The
    /// caller gets the bytes it asked for and never sees a sector.
    pub fn read_at(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let end = offset
            .checked_add(len as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "offset + len overflows"))?;
        if end > self.len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "read of {len} bytes at {offset} runs past end of file ({})",
                    self.len
                ),
            ));
        }

        let start = align_down(offset);
        // Direct reads must not run past EOF either, so the aligned tail is
        // clamped -- but the file's own length may not be a sector multiple.
        let aligned_end = align_up(end).min(align_up(self.len));
        let span = (aligned_end - start) as usize;

        let mut buf = AlignedBuf::new(span);
        let got = self.read_exact_at(&mut buf, start, span)?;

        let skip = (offset - start) as usize;
        if got < skip + len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short read: wanted {} bytes, got {got}", skip + len),
            ));
        }
        Ok(buf[skip..skip + len].to_vec())
    }

    /// Read exactly `dst.len()` bytes at `offset` **into `dst`**.
    ///
    /// [`Self::read_at`] returns a fresh `Vec`, which costs an allocation and a
    /// full copy on every call — and a streaming runner calls it for every
    /// expert slice of every token. This fills memory the caller already owns,
    /// so a slice can land directly in its final position in a stacked buffer.
    ///
    /// Returns **how many bytes had to be copied** through a scratch buffer.
    /// Zero means the drive wrote every byte straight into `dst`. A bool would
    /// be the wrong shape: a 4 MiB slice whose two edge sectors bounce is
    /// 99.9% direct, and calling that "not direct" would hide the win.
    pub fn read_at_into(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
        if dst.is_empty() {
            return Ok(0);
        }
        let end = offset
            .checked_add(dst.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "offset + len overflows"))?;
        if end > self.len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "read of {} bytes at {offset} runs past end of file ({})",
                    dst.len(),
                    self.len
                ),
            ));
        }

        // Buffered mode has no constraints at all — the kernel is copying out
        // of its own cache, so any address and length will do.
        if self.mode == IoMode::Buffered {
            let done = self.read_straight(offset, dst)?;
            if done < dst.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("short read: wanted {} bytes, got {done}", dst.len()),
                ));
            }
            return Ok(0);
        }

        // Direct I/O transfers only when the file offset and the memory address
        // agree modulo the sector size. When they do — which a caller can
        // arrange with `SkewedBuf` — the bulk of the range goes straight into
        // `dst` and only the two edge fragments, under a sector each, bounce.
        let Some((head, middle)) = self.dma_split(offset, dst) else {
            self.read_bounced(offset, dst)?;
            return Ok(dst.len());
        };

        if middle > 0 {
            let done = self.read_straight(offset + head as u64, &mut dst[head..head + middle])?;
            if done < middle {
                // A partial count left us mid-sector; the scratch finishes it.
                let rest = dst.len() - head - done;
                self.read_bounced(offset + (head + done) as u64, &mut dst[head + done..])?;
                self.read_bounced(offset, &mut dst[..head])?;
                return Ok(head + rest);
            }
        }
        if head > 0 {
            self.read_bounced(offset, &mut dst[..head])?;
        }
        let tail = head + middle;
        if tail < dst.len() {
            self.read_bounced(offset + tail as u64, &mut dst[tail..])?;
        }
        Ok(dst.len() - middle)
    }

    /// Split a read into `(head fragment, directly transferable middle)`.
    ///
    /// `None` when the destination's residue does not match the file's, so no
    /// part of the range can be transferred in place.
    fn dma_split(&self, offset: u64, dst: &[u8]) -> Option<(usize, usize)> {
        let addr = dst.as_ptr() as usize % ALIGN;
        if addr != (offset % ALIGN as u64) as usize {
            return None;
        }
        let head = (ALIGN - addr) % ALIGN;
        let middle = dst.len().checked_sub(head)? & !(ALIGN - 1);
        // Two syscalls and two small copies are not worth it for a read that is
        // barely a sector long; below that, one bounce is cheaper.
        if middle == 0 {
            return None;
        }
        Some((head, middle))
    }

    /// Read into `dst` directly, stopping if a partial count breaks alignment.
    fn read_straight(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
        let want = dst.len();
        let mut done = 0usize;
        while done < want {
            if self.mode == IoMode::Direct && done % ALIGN != 0 {
                break; // the caller finishes this tail through the scratch
            }
            let n = self.read_chunk(&mut dst[done..], offset + done as u64)?;
            if n == 0 {
                break;
            }
            done += n;
        }
        Ok(done)
    }

    /// Read into `dst` via an aligned scratch buffer, for ranges the drive
    /// cannot address directly.
    fn read_bounced(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        let len = dst.len();
        let end = offset + len as u64;
        let start = align_down(offset);
        let aligned_end = align_up(end).min(align_up(self.len));
        let span = (aligned_end - start) as usize;

        let mut buf = AlignedBuf::new(span);
        let got = self.read_exact_at(&mut buf, start, span)?;
        let skip = (offset - start) as usize;
        if got < skip + len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short read: wanted {} bytes, got {got}", skip + len),
            ));
        }
        dst.copy_from_slice(&buf[skip..skip + len]);
        Ok(())
    }

    /// Fill `buf` from `offset`, looping until `want` bytes are read or EOF.
    ///
    /// The loop must respect alignment on every iteration, not just the first.
    /// When a read lands on the file's final partial sector the OS returns a
    /// count that is *not* a sector multiple, which would leave the next read
    /// starting at an unaligned buffer offset — and direct I/O rejects that
    /// outright (`ERROR_INVALID_PARAMETER`). The short count is EOF, so there
    /// is nothing further to fetch anyway: stop.
    fn read_exact_at(&self, buf: &mut AlignedBuf, offset: u64, want: usize) -> io::Result<usize> {
        let mut done = 0usize;
        while done < want {
            if self.mode == IoMode::Direct && done % ALIGN != 0 {
                break; // final partial sector reached; no aligned read remains
            }
            let n = self.read_chunk(&mut buf[done..want], offset + done as u64)?;
            if n == 0 {
                break; // EOF, which the caller checks against what it needed
            }
            done += n;
        }
        Ok(done)
    }

    #[cfg(windows)]
    fn read_chunk(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        use std::os::windows::fs::FileExt;
        self.file.seek_read(buf, offset)
    }

    #[cfg(unix)]
    fn read_chunk(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        use std::os::unix::fs::FileExt;
        self.file.read_at(buf, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SkewedBuf;
    use std::io::Write;

    /// A temp file that deletes itself, so a failing test leaves no litter.
    struct Temp(PathBuf);

    impl Temp {
        fn with(bytes: &[u8]) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            // A timestamp is NOT unique enough: Windows' clock granularity is
            // coarse (~15 ms), so tests running in parallel collide on the same
            // name and silently overwrite each other's fixtures. A counter is.
            static SEQ: AtomicU64 = AtomicU64::new(0);

            let mut path = std::env::temp_dir();
            path.push(format!(
                "bigtea-io-{}-{}.bin",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            let mut f = File::create(&path).expect("create temp");
            f.write_all(bytes).expect("write temp");
            f.sync_all().expect("sync temp");
            Temp(path)
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Deterministic content, so a misplaced read is obvious rather than subtle.
    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn reads_exact_unaligned_ranges() {
        let data = pattern(100_000);
        let tmp = Temp::with(&data);
        let f = DirectFile::open(&tmp.0).expect("open");

        // Deliberately awkward offsets and lengths: none sector-aligned.
        for &(offset, len) in &[
            (0u64, 1usize),
            (1, 1),
            (4095, 2),
            (4096, 4096),
            (7777, 9999),
        ] {
            let got = f.read_at(offset, len).expect("read");
            assert_eq!(got.len(), len);
            assert_eq!(
                got,
                &data[offset as usize..offset as usize + len],
                "wrong bytes at offset {offset}, len {len}"
            );
        }
    }

    #[test]
    fn reads_the_final_partial_sector() {
        // A file whose length is not a sector multiple is the case most likely
        // to break aligned reads.
        let data = pattern(4096 * 3 + 17);
        let tmp = Temp::with(&data);
        let f = DirectFile::open(&tmp.0).expect("open");

        let tail = f.read_at(data.len() as u64 - 17, 17).expect("read tail");
        assert_eq!(tail, &data[data.len() - 17..]);

        let whole = f.read_at(0, data.len()).expect("read whole");
        assert_eq!(whole, data);
    }

    #[test]
    fn reading_past_the_end_is_an_error_not_garbage() {
        let data = pattern(8192);
        let tmp = Temp::with(&data);
        let f = DirectFile::open(&tmp.0).expect("open");

        assert!(f.read_at(8000, 1000).is_err());
        assert!(f.read_at(8192, 1).is_err());
        assert!(f.read_at(u64::MAX, 1).is_err(), "overflow must be caught");
    }

    #[test]
    fn zero_length_read_is_empty_not_an_error() {
        let tmp = Temp::with(&pattern(4096));
        let f = DirectFile::open(&tmp.0).expect("open");
        assert!(f.read_at(0, 0).expect("ok").is_empty());
    }

    #[test]
    fn direct_and_buffered_return_identical_bytes() {
        // The fast path must never disagree with the slow one.
        let data = pattern(50_000);
        let tmp = Temp::with(&data);
        let direct = DirectFile::open(&tmp.0).expect("open direct");
        let buffered = DirectFile::open_buffered(&tmp.0).expect("open buffered");

        assert_eq!(buffered.mode(), IoMode::Buffered);
        for &(offset, len) in &[(0u64, 4096usize), (1234, 5678), (49_000, 1000)] {
            assert_eq!(
                direct.read_at(offset, len).unwrap(),
                buffered.read_at(offset, len).unwrap(),
                "modes disagree at offset {offset}"
            );
        }
    }

    #[test]
    fn mode_is_reported_honestly() {
        let tmp = Temp::with(&pattern(4096));
        let f = DirectFile::open(&tmp.0).expect("open");
        // Whichever mode we got, it must be stated -- never assumed.
        assert!(matches!(f.mode(), IoMode::Direct | IoMode::Buffered));
        assert_eq!(f.len(), 4096);
    }

    #[test]
    fn reading_into_a_buffer_returns_the_same_bytes_as_allocating_one() {
        // The zero-copy path must never disagree with the path it replaces.
        let data = pattern(100_000);
        let tmp = Temp::with(&data);
        let f = DirectFile::open(&tmp.0).expect("open");

        for &(offset, len) in &[
            (0u64, 1usize),
            (1, 1),
            (4095, 2),
            (4096, 4096),
            (7777, 9999),
            (0, 100_000),
        ] {
            let mut into = vec![0xEE; len];
            f.read_at_into(offset, &mut into).expect("read into");
            assert_eq!(
                into,
                f.read_at(offset, len).expect("read_at"),
                "paths disagree at offset {offset}, len {len}"
            );
        }
    }

    #[test]
    fn an_aligned_destination_is_filled_without_a_bounce() {
        // The whole point of the API: when the caller's buffer is aligned and
        // the range is whole sectors, the drive writes into it directly. If
        // this regresses to `false` the copy is back and nobody would notice.
        let data = pattern(4096 * 8);
        let tmp = Temp::with(&data);
        let f = DirectFile::open(&tmp.0).expect("open");

        let mut buf = AlignedBuf::new(4096 * 2);
        let copied = f.read_at_into(4096, &mut buf[..4096 * 2]).expect("read");
        assert_eq!(&buf[..4096 * 2], &data[4096..4096 * 3]);
        if f.mode() == IoMode::Direct {
            assert_eq!(copied, 0, "an aligned whole-sector read still bounced");
        }

        // Mismatched residues: nothing can land in place, and it must say so
        // rather than quietly returning wrong bytes.
        let copied = f.read_at_into(17, &mut buf[..4096]).expect("read");
        assert_eq!(&buf[..4096], &data[17..17 + 4096]);
        if f.mode() == IoMode::Direct {
            assert_eq!(copied, 4096, "a bounced read claimed to be direct");
        }
    }

    #[test]
    fn a_skewed_destination_transfers_all_but_the_edge_sectors() {
        // The finding this API exists for: GGUF pads tensor data to 32 bytes,
        // so real tensors start mid-sector. Matching the destination's residue
        // to the file's turns a whole-range copy into two edge fragments.
        let data = pattern(4096 * 40);
        let tmp = Temp::with(&data);
        let f = DirectFile::open(&tmp.0).expect("open");

        let offset = 4096 * 3 + 2816;
        // The copied amount must be *constant* in the length of the read, not
        // proportional to it. That is the whole property: at the 4.25 MiB an
        // expert slice actually is, one sector is 0.09% and rounds to free.
        for len in [4096 * 8, 4096 * 16, 4096 * 32] {
            let mut buf = SkewedBuf::new(len, SkewedBuf::skew_for(offset));
            assert!(buf.has_skew(2816), "buffer did not get the requested skew");

            let copied = f.read_at_into(offset, &mut buf).expect("read");
            assert_eq!(&buf[..], &data[offset as usize..offset as usize + len]);
            if f.mode() == IoMode::Direct {
                assert_eq!(
                    copied, ALIGN,
                    "expected exactly the two edge fragments at len {len}"
                );
            }
        }
    }

    #[test]
    fn reading_into_past_the_end_is_an_error_not_garbage() {
        let data = pattern(8192);
        let tmp = Temp::with(&data);
        let f = DirectFile::open(&tmp.0).expect("open");

        let mut buf = vec![0u8; 1000];
        assert!(f.read_at_into(8000, &mut buf).is_err());
        assert!(f.read_at_into(u64::MAX, &mut buf).is_err());
        assert!(f.read_at_into(0, &mut []).is_ok(), "empty read is a no-op");
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(DirectFile::open("definitely-not-a-real-file-9f3a.bin").is_err());
    }
}
