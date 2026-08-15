//! Phase A, step 2: what does putting a model on the card actually cost?
//!
//! ```text
//! bigtea-gpubench <model.gguf> [--device N] [--limit-gib X]
//! ```
//!
//! # Why this is a binary and not a note in the ticket
//!
//! `2.32 GiB once at load` is a **product number**, not a rounding error. A 25x
//! prefill that costs four seconds of upload is a different product from one
//! that does not, because the user experiences the sum. So the upload is
//! measured against real weights off real disk, at the sizes the model actually
//! has, rather than estimated from PCIe bandwidth — this project has a standing
//! rule that labelled arithmetic is not measurement.
//!
//! # What this does NOT do
//!
//! It does not run a forward pass, and it therefore does **not** move the GPU
//! bar. Prefill tok/s with the card working, next to a CPU number from the same
//! session, is the gate, and it needs the forward pass ported. This measures
//! exactly one of the two numbers that gate is made of, and says so.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use bigtea_ggml::{devices, Context, DeviceKind, Residency, WeightSet};
use bigtea_model::Model;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bigtea-gpubench: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or(
        "usage: bigtea-gpubench <model.gguf> [--device N] [--limit-gib X]\n\
         \n\
         Uploads every tensor of the model to a compute device and reports what \
         that cost. Does not run a forward pass, so it does not move the GPU bar.",
    )?;
    let mut want_device: Option<usize> = None;
    let mut limit_gib: Option<f64> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--device" => want_device = args.next().and_then(|v| v.parse().ok()),
            "--limit-gib" => limit_gib = args.next().and_then(|v| v.parse().ok()),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    let list = devices()?;
    for (i, d) in list.iter().enumerate() {
        println!(
            "device {i:<2} {:<10} {:<8} {:>7.2} GiB free of {:>6.2}  {}",
            d.name,
            format!("{:?}", d.kind),
            d.free_gib(),
            d.total_gib(),
            d.description
        );
    }

    // Default to the discrete GPU, never the integrated one: on this machine the
    // iGPU is enumerated first, has MORE free memory, and runs at 0.48x the CPU.
    // See research/the-igpu-is-not-a-tier-2026-08-15.md.
    let index = match want_device {
        Some(i) => i,
        None => list
            .iter()
            .position(|d| d.kind == DeviceKind::Gpu)
            .ok_or("no discrete GPU on this machine; pass --device N to force one")?,
    };
    let dev = list.get(index).ok_or(format!("no device {index}"))?;
    println!("\nusing device {index}: {}", dev.description);
    if dev.kind == DeviceKind::IGpu {
        println!(
            "  WARNING: this is an INTEGRATED GPU. Measured at 0.48x the CPU on \
             prefill here; a UMA device removes the copy, not the bottleneck."
        );
    }

    let model = Model::open_split(&path)?;
    let names: Vec<String> = model.tensor_names().map(|s| s.to_string()).collect();
    println!("container: {} tensors", names.len());

    let backend = bigtea_ggml::Backend::open(index)?;

    // One tensor struct is a few hundred bytes; this arena holds metadata only,
    // because the context is `no_alloc` and the bytes live on the device.
    let ctx = Context::new_no_alloc(names.len() * 1024 + (16 << 20))?;
    let mut ws = WeightSet::new();

    // Read and bind, timing the disk half separately from the bus half. They
    // are different costs with different fixes, and a single "load time" hides
    // which one is worth attacking.
    let mut read_seconds = 0.0f64;
    let mut bound_bytes = 0usize;
    let mut skipped = 0usize;
    let limit_bytes = limit_gib.map(|g| (g * 1024.0 * 1024.0 * 1024.0) as usize);

    for name in &names {
        let Some(loc) = model.location(name).cloned() else {
            continue;
        };
        if let Some(limit) = limit_bytes {
            let size: u64 = loc.dims.iter().product::<u64>();
            if bound_bytes + size as usize > limit {
                skipped += 1;
                continue;
            }
        }
        let started = Instant::now();
        let bytes = model.read_tensor(name)?;
        read_seconds += started.elapsed().as_secs_f64();
        let len = bytes.len();
        // `Arc::from(Box<[u8]>)` reallocates and copies; hand it the Vec.
        match ws.bind_shared_at(
            &ctx,
            name,
            loc.ty,
            &loc.dims,
            Arc::new(bytes),
            Residency::Device,
        ) {
            Ok(()) => bound_bytes += len,
            Err(e) => {
                eprintln!("  cannot bind {name}: {e}");
                skipped += 1;
            }
        }
    }

    println!(
        "read      {:>7.2} GiB from disk in {read_seconds:>6.2}s ({:.2} GiB/s)",
        bound_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        bound_bytes as f64 / (1024.0 * 1024.0 * 1024.0) / read_seconds.max(1e-9),
    );
    if skipped > 0 {
        println!("skipped   {skipped} tensor(s)");
    }

    let free_before = devices()?
        .get(index)
        .map(|d| d.free_gib())
        .unwrap_or(f64::NAN);

    let (buffer, report) = ws.place_on_device(&backend, &ctx)?;

    let free_after = devices()?
        .get(index)
        .map(|d| d.free_gib())
        .unwrap_or(f64::NAN);

    println!(
        "upload    {:>7.2} GiB in {:>6.2}s ({})",
        report.gib(),
        report.seconds,
        match report.gib_per_second() {
            Some(r) => format!("{r:.2} GiB/s"),
            None => "nothing uploaded".to_string(),
        }
    );
    println!(
        "device    {:>7.2} GiB allocated, free {free_before:.2} -> {free_after:.2} GiB",
        buffer.bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    println!(
        "load total {:>6.2}s  ({:.2}s disk + {:.2}s bus)",
        read_seconds + report.seconds,
        read_seconds,
        report.seconds
    );
    println!(
        "\nThis is the upload half of the Phase A gate. It is NOT prefill tok/s \
         with the card working, so the GPU bar does not move on it."
    );
    Ok(())
}
