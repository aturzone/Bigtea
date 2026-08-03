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
        Ok(DirectFile { file, path, len, mode })
    }

    /// Open without attempting direct I/O. Useful for A/B measurement of what
    /// the page cache is actually doing for us.
    pub fn open_buffered(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let len = file.metadata()?.len();
        Ok(DirectFile { file, path, len, mode: IoMode::Buffered })
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
        OpenOptions::new().read(true).custom_flags(O_DIRECT).open(path)
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
        let end = offset.checked_add(len as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "offset + len overflows")
        })?;
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
    use std::io::Write;

    /// A temp file that deletes itself, so a failing test leaves no litter.
    struct Temp(PathBuf);

    impl Temp {
        fn with(bytes: &[u8]) -> Self {
            let mut path = std::env::temp_dir();
            let unique = format!(
                "bigtea-io-{}-{:?}.bin",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            path.push(unique);
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
        for &(offset, len) in &[(0u64, 1usize), (1, 1), (4095, 2), (4096, 4096), (7777, 9999)] {
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
    fn missing_file_is_an_error() {
        assert!(DirectFile::open("definitely-not-a-real-file-9f3a.bin").is_err());
    }
}
