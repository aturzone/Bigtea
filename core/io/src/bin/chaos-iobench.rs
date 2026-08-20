//! Does one file handle serialise our reads?
//!
//! Usage: `chaos-iobench <big-file> [--slice-mib N] [--slices N]`
//!
//! # The suspicion
//!
//! Expert streaming reads a layer's slices with `READERS = 4` threads, all
//! calling `read_at` on a **single shared** `DirectFile`. That was measured at
//! 1.59 GiB/s with one reader and 1.99 with four, with "no further gain at
//! eight", and the plateau was read as the drive's limit.
//!
//! There is another explanation, and it is a Windows one. A handle opened
//! without `FILE_FLAG_OVERLAPPED` is a **synchronous** handle, and the I/O
//! manager serialises operations on it — concurrent `ReadFile` calls from
//! different threads queue behind one another regardless of how many threads
//! issue them. If that is what is happening, four readers never gave the drive
//! any queue depth at all, and the 1.59 → 1.99 gain came from overlapping
//! user-space work rather than from the device.
//!
//! An NVMe drive at queue depth 1 delivers a fraction of its rated throughput.
//! This one is a Gen4-class SK Hynix; the probe measures 2.54 GB/s with a
//! single-threaded sequential read, which is roughly what a Gen4 drive does at
//! QD1 and roughly half what it does when properly fed.
//!
//! # The experiment
//!
//! The same scattered 4 MiB reads, the same thread counts, one variable: whether
//! the threads share a handle or each open their own. If per-handle scales and
//! shared does not, the plateau was ours and not the drive's — and expert
//! streaming, which is **62% of a token**, has headroom nobody has claimed.
//!
//! Offsets are pseudo-random across the file and never repeated within a run, so
//! neither the drive's read-ahead nor any cache can answer from the previous
//! pass. Direct I/O bypasses the page cache, but the drive's own is real.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use chaos_io::{DirectFile, SkewedBuf};

const THREADS: &[usize] = &[1, 2, 4, 8, 16, 32];

fn main() -> ExitCode {
    // **Before anything treats an argument as a path.** Without this,
    // `chaos-iobench --version` reported "cannot find the file specified" -- the
    // flag was being opened as a model. `--version` is how a person checks
    // whether an update landed, so it has to answer on whichever binary they
    // happen to type.
    if std::env::args()
        .skip(1)
        .any(|a| a == "--version" || a == "-V")
    {
        println!("chaos-iobench {}", env!("CARGO_PKG_VERSION"));
        std::process::exit(0);
    }
    let mut path: Option<PathBuf> = None;
    let mut slice_mib = 4usize;
    let mut slices = 256usize;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--slice-mib" => {
                slice_mib = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(slice_mib);
                i += 2;
            }
            "--slices" => {
                slices = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(slices);
                i += 2;
            }
            "-h" | "--help" => {
                println!("usage: chaos-iobench <big-file> [--slice-mib N] [--slices N]");
                println!();
                println!("Reads scattered slices with N threads, sharing one file");
                println!("handle and then with one handle each, and prints both.");
                return ExitCode::SUCCESS;
            }
            other => {
                if path.is_none() {
                    path = Some(PathBuf::from(other));
                }
                i += 1;
            }
        }
    }
    let Some(path) = path else {
        eprintln!("usage: chaos-iobench <big-file> [--slice-mib N] [--slices N]");
        return ExitCode::from(2);
    };

    match run(&path, slice_mib << 20, slices) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chaos-iobench: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: &PathBuf, slice: usize, slices: usize) -> Result<(), Box<dyn std::error::Error>> {
    let probe = DirectFile::open(path)?;
    let len = probe.len();
    let mode = probe.mode();
    if len < (slices as u64 + 1) * slice as u64 {
        return Err(format!(
            "{} is {:.1} GiB; need at least {:.1} GiB for {slices} slices of {} MiB",
            path.display(),
            len as f64 / (1u64 << 30) as f64,
            (slices as f64 * slice as f64) / (1u64 << 30) as f64,
            slice >> 20
        )
        .into());
    }
    let total = (slices * slice) as f64 / (1u64 << 30) as f64;

    println!("file       {}", path.display());
    println!(
        "           {:.1} GiB, opened {mode:?}",
        len as f64 / (1u64 << 30) as f64
    );
    println!(
        "reading    {slices} slices of {} MiB = {total:.2} GiB per pass, scattered",
        slice >> 20
    );
    println!();
    println!(
        "{:>8}  {:>14}  {:>14}  {:>10}",
        "THREADS", "SHARED GiB/s", "PER-HANDLE", "GAIN"
    );

    for &t in THREADS {
        // Fresh offsets per cell, so no cell can answer from the one before it.
        let shared = pass(path, slice, slices, t, true, t as u64 * 7919)?;
        let per = pass(path, slice, slices, t, false, t as u64 * 104_729)?;
        println!(
            "{t:>8}  {:>14.2}  {:>14.2}  {:>9.2}x",
            total / shared,
            total / per,
            shared / per
        );
    }

    println!();
    println!("If PER-HANDLE keeps climbing where SHARED flattens, the 4-reader");
    println!("plateau was a synchronous-handle artefact and not the drive. Expert");
    println!("streaming is ~62% of a token, so that headroom is the largest single");
    println!("lever left on this machine.");
    Ok(())
}

/// One pass over `slices` scattered reads, returning seconds.
fn pass(
    path: &PathBuf,
    slice: usize,
    slices: usize,
    threads: usize,
    share_handle: bool,
    seed: u64,
) -> Result<f64, Box<dyn std::error::Error>> {
    let len = DirectFile::open(path)?.len();
    // Sector-aligned pseudo-random offsets, distinct, inside the file.
    let span = (len - slice as u64) / 4096;
    let offsets: Vec<u64> = (0..slices)
        .map(|i| {
            let mut z = seed
                .wrapping_add(i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            (z % span) * 4096
        })
        .collect();

    let shared = if share_handle {
        Some(Arc::new(DirectFile::open(path)?))
    } else {
        None
    };

    // Destination buffers are allocated before timing: this measures the drive,
    // not the allocator.
    let mut bufs: Vec<SkewedBuf> = (0..slices).map(|_| SkewedBuf::new(slice, 0)).collect();

    let per_thread = slices.div_ceil(threads);
    let t0 = Instant::now();
    std::thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
        let mut handles = Vec::new();
        for (t, chunk) in bufs.chunks_mut(per_thread).enumerate() {
            let offsets = &offsets;
            let shared = shared.clone();
            let path = path.clone();
            handles.push(scope.spawn(move || -> std::io::Result<()> {
                // The variable under test: one handle for everyone, or one each.
                let own;
                let file: &DirectFile = match &shared {
                    Some(f) => f,
                    None => {
                        own = DirectFile::open(&path)?;
                        &own
                    }
                };
                for (j, dst) in chunk.iter_mut().enumerate() {
                    file.read_at_into(offsets[t * per_thread + j], dst)?;
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().expect("reader thread")?;
        }
        Ok(())
    })?;
    Ok(t0.elapsed().as_secs_f64())
}
