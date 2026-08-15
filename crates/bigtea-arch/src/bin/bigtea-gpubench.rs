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

use bigtea_arch::{KvCache, Qwen3Config, StreamingRunner};
use bigtea_ggml::{devices, Context, DeviceKind, Residency, WeightSet};
use bigtea_model::Model;

/// Long enough that prefill is compute-bound rather than fixed-cost bound.
const PREFILL_TOKENS: usize = 512;

/// What one target's prefill series produced.
struct Series {
    /// Seconds to load and, on the device path, upload every weight.
    load_seconds: f64,
    /// tok/s for each TIMED run — the warm-up is not in here.
    rates: Vec<f64>,
    /// Seconds for the discarded warm-up prefill, kept because a cold first
    /// token is a real thing a user waits for exactly once per machine.
    first_prefill_seconds: f64,
    /// Compared between targets: a wrong device path returns plausible logits.
    checksum: f64,
}

/// One load, then `repeats + 1` prefills on the loaded weights.
///
/// **Loading inside the timed loop was the first version and it was wrong.**
/// Each load reads 2.32 GiB from disk; eight of them back to back thrashed the
/// page cache and the drive, and the CPU baseline swung 26-67 tok/s — a 2.5x
/// spread that buried the very effect being measured. Load is measured once,
/// reported once, and kept out of the prefill numbers.
///
/// The first prefill is still returned separately, because load-to-first-token
/// is a real product number and it includes exactly one cold prefill.
fn prefill_series(
    model: &Model,
    config: &Qwen3Config,
    tokens: &[u32],
    device: Option<usize>,
    repeats: usize,
) -> Result<Series, Box<dyn std::error::Error>> {
    let load_start = Instant::now();
    let mut runner = StreamingRunner::new(model, config.clone(), 0);
    if let Some(i) = device {
        runner.use_device(i)?;
    }
    let ctx = Context::new_no_alloc(64 << 20)?;
    let mut weights = WeightSet::new();
    let _held = if device.is_some() {
        let (_bytes, buffer, _report) = runner.load_resident_on_device(&ctx, &mut weights)?;
        Some(buffer)
    } else {
        runner.load_resident(&ctx, &mut weights)?;
        None
    };
    let load_seconds = load_start.elapsed().as_secs_f64();

    let mut rates = Vec::new();
    let mut first_prefill = 0.0f64;
    let mut checksum = 0.0f64;
    for r in 0..=repeats {
        let mut cache = KvCache::new(
            config.n_layer as usize,
            config.n_head_kv as usize,
            config.head_dim as usize,
        );
        if device.is_some() && r == 1 {
            // Counters cover the timed runs only, not the discarded warm-up.
            bigtea_ggml::backend::timing::reset();
        }
        let started = Instant::now();
        let logits = runner.forward_cached(&weights, &mut cache, tokens, 0)?;
        let seconds = started.elapsed().as_secs_f64();
        checksum = logits
            .iter()
            .take(64)
            .map(|v| v.abs())
            .fold(0.0f64, |a, b| a + b as f64);
        if r == 0 {
            first_prefill = seconds;
            continue; // the warm-up: cold shader cache on the device path
        }
        rates.push(tokens.len() as f64 / seconds.max(1e-9));
    }
    Ok(Series {
        load_seconds,
        rates,
        first_prefill_seconds: first_prefill,
        checksum,
    })
}

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
    // Three timed repeats after a discarded warm-up. Not a convenience: see
    // `report` below for why one run of a GPU path is not a measurement.
    let mut repeats: usize = 3;
    let mut force = false;
    let mut limit_gib: Option<f64> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--device" => want_device = args.next().and_then(|v| v.parse().ok()),
            "--limit-gib" => limit_gib = args.next().and_then(|v| v.parse().ok()),
            "--repeat" => {
                repeats = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--repeat needs a number")?
            }
            "--force" => force = true,
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
    drop(buffer);
    drop(ws);
    drop(ctx);

    // ---- the other half of the gate: prefill, both targets, one session ----
    //
    // Back to back in the same process on the same prompt. The CPU side gets
    // ALL the threads, because prefill is compute-bound and a mistuned baseline
    // inflates the ratio -- the trap that would have turned llama.cpp's 25.6x
    // into 30.1x, pointing the other way.
    // Overridable so a crash can be bisected by length: the device path's
    // shapes change with the token count, and "works at 1, dies at 512" says
    // something different from "dies at 1".
    let n_prompt: usize = std::env::var("BIGTEA_PREFILL_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(PREFILL_TOKENS);
    let prompt: Vec<u32> = (0..n_prompt).map(|i| (i % 3000 + 10) as u32).collect();
    let config = Qwen3Config::from_model(&model)?;
    println!(
        "
prefill {} tokens, same prompt, same session",
        prompt.len()
    );

    // **THE WARM-UP RUN IS DISCARDED, AND THAT IS THE POINT OF THIS HARNESS.**
    //
    // ggml's Vulkan backend compiles a large shader set on first use and the
    // driver persists the compiled pipelines to disk, so the first run of a GPU
    // path does work no later run does. Measured here: the same binary read
    // 0.42x on its first runs and 1.62-1.78x on every run after, and that wrong
    // number reached a research node before anything caught it.
    //
    // A harness that cannot report a cold-cache number as steady state is worth
    // more than any speedup measured with it.
    if repeats < 2 && !force {
        return Err(format!(
            "--repeat {repeats} cannot distinguish a cold shader cache from steady state.              The first run of a GPU path compiles pipelines inside the timed region; this              harness exists because that once produced a published 0.42x that was really              1.7x. Use --repeat 3 (the default), or --force if you genuinely want one run."
        )
        .into());
    }

    let cpu = prefill_series(&model, &config, &prompt, None, repeats)?;
    let gpu = prefill_series(&model, &config, &prompt, Some(index), repeats)?;
    let (cpu_load, cpu_rates, cpu_first, cpu_sum) = (
        cpu.load_seconds,
        cpu.rates,
        cpu.first_prefill_seconds,
        cpu.checksum,
    );
    let (gpu_load, gpu_rates, gpu_first, gpu_sum) = (
        gpu.load_seconds,
        gpu.rates,
        gpu.first_prefill_seconds,
        gpu.checksum,
    );
    let (realize_s, up_s, down_s, comp_s, realize_n, comp_n) =
        bigtea_ggml::backend::timing::snapshot();

    let stats = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a timing"));
        (
            s[s.len() / 2],
            *s.first().expect("at least one run"),
            *s.last().expect("at least one run"),
        )
    };
    let (cpu_med, cpu_lo, cpu_hi) = stats(&cpu_rates);
    let (gpu_med, gpu_lo, gpu_hi) = stats(&gpu_rates);

    for (i, (c, g)) in cpu_rates.iter().zip(&gpu_rates).enumerate() {
        println!(
            "  run {:<2}   cpu {c:>7.2}  device {g:>7.2} tok/s   {:.2}x",
            i + 1,
            g / c.max(1e-9)
        );
    }
    println!(
        "
  cpu     median {cpu_med:>8.2} tok/s   ({cpu_lo:.2}-{cpu_hi:.2}, {} timed runs)",
        cpu_rates.len()
    );
    println!(
        "  device  median {gpu_med:>8.2} tok/s   ({gpu_lo:.2}-{gpu_hi:.2}, warm-up discarded)"
    );
    println!(
        "
  prefill ratio         {:.2}x",
        gpu_med / cpu_med.max(1e-9)
    );
    // TWO first-token numbers, because they answer different questions and
    // quoting one as the other is the same error this harness exists to stop.
    //
    // COLD includes the discarded run, so on the device it includes shader
    // compilation: that is a user's very first launch on a machine whose driver
    // cache has never seen these pipelines.
    //
    // WARM uses the median prefill: every launch after, because the driver
    // persists compiled pipelines to disk.
    let warm = |load: f64, med: f64| load + prompt.len() as f64 / med.max(1e-9);
    println!(
        "  first token, cold     {:.2}s cpu vs {:.2}s device ({:+.2}s)",
        cpu_load + cpu_first,
        gpu_load + gpu_first,
        (gpu_load + gpu_first) - (cpu_load + cpu_first)
    );
    println!(
        "  first token, warm     {:.2}s cpu vs {:.2}s device ({:+.2}s)",
        warm(cpu_load, cpu_med),
        warm(gpu_load, gpu_med),
        warm(gpu_load, gpu_med) - warm(cpu_load, cpu_med)
    );
    println!(
        "
  device time (timed runs only)"
    );
    println!("    realize  {realize_s:>6.2}s  ({realize_n} allocations)");
    println!("    compute  {comp_s:>6.2}s  ({comp_n} graph submissions)");
    println!("    upload   {up_s:>6.2}s");
    println!("    download {down_s:>6.2}s");
    println!("  logit checksum        cpu {cpu_sum:.4} vs device {gpu_sum:.4}");
    Ok(())
}
