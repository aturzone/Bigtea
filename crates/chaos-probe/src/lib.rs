//! Measure what this machine can actually do.
//!
//! Three rules, because every downstream prediction is only as good as these
//! numbers:
//!
//! 1. **Measured beats reported.** Storage read bandwidth is benchmarked, never
//!    taken from a spec sheet — rated and achieved single-stream throughput
//!    routinely differ by 2x, and this one number sets the whole tok/s ceiling.
//! 2. **Unknown is `None`.** Never substitute a plausible default for a real
//!    measurement. A silently wrong number is far worse than a missing one,
//!    because it looks like knowledge.
//! 3. **Record the method.** Every measurement carries how it was obtained, so
//!    a suspicious result can be audited instead of trusted.

use std::fmt;
use std::path::{Path, PathBuf};

mod bandwidth;
mod gpu;
mod platform;
pub mod processes;

pub use bandwidth::{measure_read_bandwidth, BandwidthError};
pub use gpu::Gpu;
pub use processes::Process;

pub const GIB: u64 = 1 << 30;

/// Storage facts for one filesystem.
#[derive(Debug, Clone)]
pub struct Storage {
    pub path: PathBuf,
    pub total_bytes: u64,
    pub free_bytes: u64,
    /// Measured sequential read throughput. `None` when not measured or the
    /// benchmark could not run.
    pub read_bytes_per_sec: Option<f64>,
    /// How `read_bytes_per_sec` was obtained, or why it is missing.
    pub read_method: String,
}

/// A full picture of the host.
#[derive(Debug, Clone)]
pub struct Machine {
    pub os: String,
    pub arch: &'static str,
    pub cpu_threads: usize,
    pub ram_total_bytes: Option<u64>,
    pub ram_available_bytes: Option<u64>,
    pub ram_source: String,
    pub storage: Storage,
    pub gpus: Vec<Gpu>,
}

impl Machine {
    /// Probe the host. `measure_bandwidth` gates the only slow step.
    pub fn probe(path: impl AsRef<Path>, measure_bandwidth: bool) -> Self {
        let path = path.as_ref();
        let (ram_total, ram_available, ram_source) = platform::ram();

        let (total_bytes, free_bytes) = platform::disk_space(path).unwrap_or((0, 0));
        let mut storage = Storage {
            path: path.to_path_buf(),
            total_bytes,
            free_bytes,
            read_bytes_per_sec: None,
            read_method: "not measured".into(),
        };

        if measure_bandwidth {
            match measure_read_bandwidth(path, ram_available) {
                Ok(result) => {
                    storage.read_bytes_per_sec = Some(result.bytes_per_sec);
                    storage.read_method = result.method;
                }
                Err(e) => storage.read_method = format!("unavailable: {e}"),
            }
        }

        Machine {
            os: platform::os_description(),
            arch: std::env::consts::ARCH,
            cpu_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0),
            ram_total_bytes: ram_total,
            ram_available_bytes: ram_available,
            ram_source,
            storage,
            gpus: gpu::probe(),
        }
    }

    /// RAM a model may actually use for weights, after the operating system,
    /// KV cache, activation scratch and engine buffers take their share.
    ///
    /// Deliberately conservative: over-estimating here produces a plan that
    /// swaps, and swapping is catastrophically slower than the streaming it
    /// was meant to avoid.
    pub fn usable_ram_for_weights(&self, overhead_bytes: u64) -> u64 {
        self.ram_available_bytes
            .unwrap_or(0)
            .saturating_sub(overhead_bytes)
    }

    pub fn total_vram_bytes(&self) -> u64 {
        self.gpus.iter().filter_map(|g| g.vram_total_bytes).sum()
    }
}

/// Render a byte count the way a human reads capacity.
pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

impl fmt::Display for Machine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "os         {} ({})", self.os, self.arch)?;
        writeln!(f, "cpu        {} threads", self.cpu_threads)?;
        match (self.ram_total_bytes, self.ram_available_bytes) {
            (Some(t), Some(a)) => writeln!(
                f,
                "ram        {:.1} GiB total, {:.1} GiB available   [{}]",
                gib(t),
                gib(a),
                self.ram_source
            )?,
            (Some(t), None) => writeln!(
                f,
                "ram        {:.1} GiB total, available unknown   [{}]",
                gib(t),
                self.ram_source
            )?,
            _ => writeln!(f, "ram        unknown   [{}]", self.ram_source)?,
        }
        writeln!(
            f,
            "disk       {:.1} GiB free of {:.1} GiB   ({})",
            gib(self.storage.free_bytes),
            gib(self.storage.total_bytes),
            self.storage.path.display()
        )?;
        match self.storage.read_bytes_per_sec {
            Some(bps) => writeln!(
                f,
                "read       {:.2} GB/s   [{}]",
                bps / 1e9,
                self.storage.read_method
            )?,
            None => writeln!(f, "read       {}", self.storage.read_method)?,
        }
        if self.gpus.is_empty() {
            write!(f, "gpu        none detected")?;
        } else {
            for (i, g) in self.gpus.iter().enumerate() {
                let vram = g
                    .vram_total_bytes
                    .map(|v| format!("{:.1} GiB", gib(v)))
                    .unwrap_or_else(|| "? GiB".into());
                if i > 0 {
                    writeln!(f)?;
                }
                write!(f, "gpu        {}  {}   [{}]", g.name, vram, g.source)?;
            }
        }
        Ok(())
    }
}
