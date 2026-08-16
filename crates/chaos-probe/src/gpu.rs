//! GPU discovery.
//!
//! Absence of a GPU is a normal outcome, not an error — Chaos's target tier
//! is machines where the GPU is small or missing, and the CPU/disk path is the
//! one that matters. A failed query therefore yields an empty list.

use std::process::Command;

#[derive(Debug, Clone)]
pub struct Gpu {
    pub name: String,
    pub vram_total_bytes: Option<u64>,
    /// Tool the reading came from, so an odd number can be traced.
    pub source: &'static str,
}

pub fn probe() -> Vec<Gpu> {
    nvidia_smi()
}

fn nvidia_smi() -> Vec<Gpu> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let Ok(out) = out else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    let Ok(text) = String::from_utf8(out.stdout) else {
        return Vec::new();
    };

    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            let (name, mem) = line.split_once(',')?;
            // nvidia-smi reports MiB with `nounits`.
            let vram = mem
                .trim()
                .parse::<f64>()
                .ok()
                .map(|mib| (mib * 1024.0 * 1024.0) as u64);
            Some(Gpu {
                name: name.trim().to_string(),
                vram_total_bytes: vram,
                source: "nvidia-smi",
            })
        })
        .collect()
}
