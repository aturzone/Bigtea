//! The prediction itself.

use crate::profile::ModelProfile;
use crate::GIB;

/// Fraction of theoretical bandwidth a real streaming engine achieves.
///
/// Grounded in published measurements of engines in this class, where achieved
/// throughput lands near half of the "perfect overlap" ceiling once read-ahead,
/// syscall overhead and imperfect I/O-versus-compute overlap are paid for.
/// It is a calibration constant, and it should be replaced with a measured
/// value per engine as soon as real benchmarks exist.
pub const DEFAULT_EFFICIENCY: f64 = 0.5;

/// Fallback runtime overhead when the model's attention shape is unknown.
///
/// **Prefer [`crate::overhead`], which computes this from the architecture and
/// context length.** A flat constant is wrong in both directions: it
/// double-counts the OS (measured *available* RAM already excludes it), and it
/// cannot know that a KV cache ranges over three orders of magnitude with
/// context length and attention shape. On a 16 GiB machine a 2 GiB error here
/// decides whether the dense weights are resident or re-read every token.
pub const DEFAULT_OVERHEAD_BYTES: u64 = GIB;

#[derive(Debug, Clone)]
pub struct Prediction {
    pub model: String,
    pub container_bytes: u64,

    pub dense_bytes: u64,
    pub dense_resident_bytes: u64,
    /// Dense weights that do not fit, and are therefore re-read every token.
    pub dense_shortfall_bytes: u64,
    pub expert_bytes_per_token: u64,
    /// Total read per token — the number that sets throughput.
    pub bytes_per_token: u64,

    pub usable_ram_bytes: u64,
    pub read_bytes_per_sec: Option<f64>,
    pub efficiency: f64,

    pub free_disk_bytes: Option<u64>,
    pub fits_disk: Option<bool>,

    pub tokens_per_sec_ceiling: Option<f64>,
    pub tokens_per_sec_realistic: Option<f64>,
    pub seconds_per_token: Option<f64>,

    /// Human-facing observations worth acting on.
    pub notes: Vec<String>,
}

impl Prediction {
    /// Compute a prediction for `profile` on a machine with the given limits.
    pub fn new(
        profile: &ModelProfile,
        ram_available_bytes: Option<u64>,
        read_bytes_per_sec: Option<f64>,
        free_disk_bytes: Option<u64>,
        overhead_bytes: u64,
        efficiency: f64,
    ) -> Self {
        let mut notes = Vec::new();

        let usable = ram_available_bytes
            .unwrap_or(0)
            .saturating_sub(overhead_bytes);
        let dense_resident = profile.dense_bytes.min(usable);
        let dense_shortfall = profile.dense_bytes - dense_resident;

        if profile.dense_bytes > 0 {
            if dense_shortfall == 0 {
                notes.push(
                    "dense weights fit in RAM: read once, then cached. This is the \
                     single biggest win available on a machine this size."
                        .into(),
                );
            } else {
                notes.push(format!(
                    "dense weights exceed usable RAM by {:.2} GiB, which is re-read \
                     every token and dominates the cost. A smaller quantization helps \
                     here twice: less to read, and more of it resident.",
                    dense_shortfall as f64 / GIB as f64
                ));
            }
        }

        // Leftover RAM would go to an expert cache -- but that only pays once
        // it can hold a whole token's working set. Below that, entries are
        // evicted before reuse and the hit rate collapses toward zero.
        let leftover = usable.saturating_sub(dense_resident);
        if profile.expert_bytes_per_token > 0
            && leftover > 0
            && leftover < profile.expert_bytes_per_token
        {
            notes.push(format!(
                "{:.2} GiB spare RAM is below one token's expert working set \
                 ({:.2} GiB), so expert caching is assumed to contribute nothing",
                leftover as f64 / GIB as f64,
                profile.expert_bytes_per_token as f64 / GIB as f64
            ));
        }

        let bytes_per_token = dense_shortfall + profile.expert_bytes_per_token;

        let (ceiling, realistic, spt) = match read_bytes_per_sec {
            Some(bps) if bytes_per_token > 0 && bps > 0.0 => {
                let ceiling = bps / bytes_per_token as f64;
                let realistic = ceiling * efficiency;
                let spt = if realistic > 0.0 {
                    Some(1.0 / realistic)
                } else {
                    None
                };
                (Some(ceiling), Some(realistic), spt)
            }
            _ => (None, None, None),
        };

        let container = profile.container_bytes();
        let fits_disk = free_disk_bytes.map(|free| container <= free);
        if let (Some(false), Some(free)) = (fits_disk, free_disk_bytes) {
            notes.push(format!(
                "container needs {:.1} GiB more disk than is free",
                (container - free) as f64 / GIB as f64
            ));
        }

        Prediction {
            model: profile.name.clone(),
            container_bytes: container,
            dense_bytes: profile.dense_bytes,
            dense_resident_bytes: dense_resident,
            dense_shortfall_bytes: dense_shortfall,
            expert_bytes_per_token: profile.expert_bytes_per_token,
            bytes_per_token,
            usable_ram_bytes: usable,
            read_bytes_per_sec,
            efficiency,
            free_disk_bytes,
            fits_disk,
            tokens_per_sec_ceiling: ceiling,
            tokens_per_sec_realistic: realistic,
            seconds_per_token: spt,
            notes,
        }
    }

    pub fn dense_fully_resident(&self) -> bool {
        self.dense_shortfall_bytes == 0
    }

    /// Can this run at all, ignoring how slowly?
    pub fn is_runnable(&self) -> bool {
        self.fits_disk.unwrap_or(true) && self.bytes_per_token > 0
    }
}

impl std::fmt::Display for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let g = |b: u64| b as f64 / GIB as f64;
        writeln!(f, "model            {}", self.model)?;
        writeln!(f, "container        {:.1} GiB", g(self.container_bytes))?;
        writeln!(
            f,
            "dense            {:.2} GiB   {}",
            g(self.dense_bytes),
            if self.dense_fully_resident() {
                "[fits in RAM -> cached]".to_string()
            } else {
                format!(
                    "[{:.2} GiB re-read every token]",
                    g(self.dense_shortfall_bytes)
                )
            }
        )?;
        writeln!(
            f,
            "experts/token    {:.2} GiB",
            g(self.expert_bytes_per_token)
        )?;
        writeln!(f, "read per token   {:.2} GiB", g(self.bytes_per_token))?;
        if let (Some(free), Some(fits)) = (self.free_disk_bytes, self.fits_disk) {
            writeln!(
                f,
                "disk             {:.1} GiB needed, {:.1} GiB free -> {}",
                g(self.container_bytes),
                g(free),
                if fits { "fits" } else { "DOES NOT FIT" }
            )?;
        }
        match (self.tokens_per_sec_realistic, self.seconds_per_token) {
            (Some(tps), Some(spt)) => writeln!(
                f,
                "speed            ~{:.2} tok/s ({:.1} s/token), ceiling {:.2}",
                tps,
                spt,
                self.tokens_per_sec_ceiling.unwrap_or(0.0)
            )?,
            _ => writeln!(f, "speed            unknown (no measured read bandwidth)")?,
        }
        for note in &self.notes {
            writeln!(f, "  ! {note}")?;
        }
        Ok(())
    }
}
