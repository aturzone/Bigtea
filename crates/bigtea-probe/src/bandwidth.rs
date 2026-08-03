//! Benchmark real sequential read throughput.
//!
//! The trap this exists to avoid: read a file that fits in RAM and the page
//! cache serves it, so you measure *memory* bandwidth — commonly 10x the disk
//! figure — and every prediction built on it is wrong by an order of
//! magnitude. The defence is to make the file larger than available RAM, so
//! the cache cannot hold it and the disk must be touched.
//!
//! This measures large sequential reads. Streaming MoE inference issues large
//! *quasi-random* reads (one expert record at a router-chosen offset), which on
//! modern NVMe converge toward sequential throughput once the transfer is
//! large, but on spinning disks or over a slow enclosure would not. Treat this
//! as an upper bound on what streaming will achieve.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

const GIB: u64 = 1 << 30;
const BLOCK: usize = 8 << 20; // 8 MiB
/// Never write a probe file bigger than this, however much RAM there is.
const MAX_PROBE: u64 = 12 * GIB;
/// Below this the timing is noise-dominated.
const MIN_PROBE: u64 = 2 * GIB;

#[derive(Debug)]
pub enum BandwidthError {
    /// Not enough free space to write a cache-defeating probe file.
    InsufficientSpace { needed: u64, free: u64 },
    Io(std::io::Error),
    /// The read finished too fast to time meaningfully.
    TooFast,
}

impl fmt::Display for BandwidthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BandwidthError::InsufficientSpace { needed, free } => write!(
                f,
                "needs {:.1} GiB free to defeat the page cache, only {:.1} GiB available",
                *needed as f64 / GIB as f64,
                *free as f64 / GIB as f64
            ),
            BandwidthError::Io(e) => write!(f, "{e}"),
            BandwidthError::TooFast => write!(f, "read completed below timer resolution"),
        }
    }
}

impl std::error::Error for BandwidthError {}

impl From<std::io::Error> for BandwidthError {
    fn from(e: std::io::Error) -> Self {
        BandwidthError::Io(e)
    }
}

pub struct Bandwidth {
    pub bytes_per_sec: f64,
    pub bytes_read: u64,
    pub seconds: f64,
    pub method: String,
}

/// Probe file size: comfortably larger than what the cache could hold.
fn probe_size(available_ram: Option<u64>) -> u64 {
    let ram = available_ram.unwrap_or(4 * GIB);
    // 1.5x available RAM leaves no room for the cache to hide the disk, while
    // staying small enough that the probe finishes in well under a minute even
    // on a slow drive.
    ram.saturating_mul(3)
        .checked_div(2)
        .unwrap_or(MIN_PROBE)
        .clamp(MIN_PROBE, MAX_PROBE)
}

/// Deletes the probe file even if the benchmark fails or panics.
struct TempFile(std::path::PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Measure sequential read throughput on the filesystem holding `dir`.
pub fn measure_read_bandwidth(
    dir: impl AsRef<Path>,
    available_ram: Option<u64>,
) -> Result<Bandwidth, BandwidthError> {
    let dir = dir.as_ref();
    let base = if dir.is_dir() {
        dir.to_path_buf()
    } else {
        dir.parent().unwrap_or(Path::new(".")).to_path_buf()
    };

    let size = probe_size(available_ram);
    if let Some((_, free)) = crate::platform::disk_space(&base) {
        // Leave headroom so the probe never fills the disk it is measuring.
        let needed = size + size / 10;
        if free < needed {
            return Err(BandwidthError::InsufficientSpace { needed, free });
        }
    }

    let path = base.join(".bigtea-iobench.tmp");
    let _cleanup = TempFile(path.clone());

    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        let block = vec![0u8; BLOCK];
        let mut written = 0u64;
        while written < size {
            let n = BLOCK.min((size - written) as usize);
            f.write_all(&block[..n])?;
            written += n as u64;
        }
        f.flush()?;
        // Force the data out of the OS write cache and onto the device, so the
        // read that follows is a genuine read.
        f.sync_all()?;
    }

    let mut f = File::open(&path)?;
    let mut buf = vec![0u8; BLOCK];
    let start = Instant::now();
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        total += n as u64;
    }
    let seconds = start.elapsed().as_secs_f64();
    if seconds <= 0.0 {
        return Err(BandwidthError::TooFast);
    }

    Ok(Bandwidth {
        bytes_per_sec: total as f64 / seconds,
        bytes_read: total,
        seconds,
        method: format!(
            "sequential read of {:.1} GiB in {} MiB blocks (sized above RAM to defeat the page cache)",
            total as f64 / GIB as f64,
            BLOCK >> 20
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_size_exceeds_available_ram() {
        // The whole point: the file must not fit in the cache.
        let ram = 8 * GIB;
        assert!(probe_size(Some(ram)) > ram);
    }

    #[test]
    fn probe_size_is_clamped_at_both_ends() {
        // A tiny-RAM box still gets a file big enough to time.
        assert_eq!(probe_size(Some(64 << 20)), MIN_PROBE);
        // A huge-RAM box does not get a 96 GiB probe file.
        assert_eq!(probe_size(Some(256 * GIB)), MAX_PROBE);
    }

    #[test]
    fn probe_size_has_a_sane_default_without_ram_info() {
        let s = probe_size(None);
        assert!((MIN_PROBE..=MAX_PROBE).contains(&s));
    }
}
