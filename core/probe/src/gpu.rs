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
    let mut cmd = Command::new("nvidia-smi");
    cmd.args([
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]);
    // **A console window flashes on screen without this.** `nvidia-smi` is a
    // console program, so Windows gives it a console -- and a windowed app that
    // has none of its own gets a new one, on top of whatever the user is doing,
    // for as long as the query takes. Atur saw two of them before the window
    // appeared, because the app probed twice.
    //
    // Nothing here reads a terminal, so there is nothing to lose by suppressing
    // it. `.output()` still captures stdout exactly as before.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW. Declared here rather than depended on, because
        // chaos-probe has no Windows module and one constant is not worth one.
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output();

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
