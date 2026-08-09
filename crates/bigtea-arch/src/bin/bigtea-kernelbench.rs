//! Time the expert FFN with the disk taken out of the picture.
//!
//! Usage: `bigtea-kernelbench <model.gguf> [--layer N] [--reps N]`
//!
//! # Why this exists
//!
//! A single-token step on V4-Flash costs ~2.3 s of disk and ~1.0 s of compute.
//! Everything this project has done has attacked the 2.3, and the 1.0 has been
//! treated as a floor. It is not obviously one:
//!
//! ```text
//! per token:  43 blocks x 6 experts x 3 matrices x 4096 x 2048 x 2 = 13.0 GFLOP
//! at 1.0 s                                                        = 13 GFLOP/s
//! ```
//!
//! 13 GFLOP/s on a CPU that should do a hundred. That gap is either a real
//! property of the workload — a one-token pass is a matrix-*vector* product, so
//! it is bound by how fast weights arrive from DRAM rather than by arithmetic —
//! or it is something fixable. **Nothing in the project's numbers distinguishes
//! those two**, because compute has only ever been measured with disk reads
//! interleaved.
//!
//! So this reads one block's experts once and then times the arithmetic alone,
//! repeatedly, with the bytes already in memory.
//!
//! # The question that matters more than the headline
//!
//! **Is compute flat or linear in the batch?** If a one-token pass is
//! bandwidth-bound, then 8 tokens cost nearly what 1 costs — the weights are
//! read once either way — and every batching idea gets dramatically better.
//! `v4flash-has-no-slack-2026-08-10.md` priced speculative decoding assuming
//! verify compute scales *linearly* with the batch. If that assumption is wrong,
//! that conclusion is too, and this is the measurement that decides it.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use bigtea_ggml::{Context, WeightSet};
use bigtea_gguf::GgmlType;
use bigtea_model::Model;

/// One expert-stack tensor, held in memory for the whole benchmark.
struct Slice {
    name: String,
    ty: GgmlType,
    /// `[ne0, ne1, N_USED]` — `ne0` is the reduction dimension, and it differs
    /// between `down` and the other two, so it is carried rather than assumed.
    dims: Vec<u64>,
    bytes: Arc<Vec<u8>>,
}

/// Batch sizes to sweep. 1 is a cached generation step; 8-32 is the range a
/// speculative verify pass would live in.
const BATCHES: &[i64] = &[1, 2, 4, 8, 16, 32];

fn main() -> ExitCode {
    let mut path: Option<PathBuf> = None;
    let mut layer = 20u32;
    let mut reps = 5usize;
    let mut threads_sweep = false;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--layer" => {
                layer = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(layer);
                i += 2;
            }
            "--reps" => {
                reps = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(reps);
                i += 2;
            }
            "--threads" => {
                threads_sweep = true;
                i += 1;
            }
            "-h" | "--help" => {
                println!(
                    "usage: bigtea-kernelbench <model.gguf> [--layer N] [--reps N] [--threads]"
                );
                println!();
                println!("  --threads   also sweep thread counts at batch 1");
                println!();
                println!("Times the expert FFN with the weights already in memory,");
                println!("so the arithmetic is measured without the disk in the way.");
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
        eprintln!("usage: bigtea-kernelbench <model.gguf> [--layer N] [--reps N] [--threads]");
        return ExitCode::from(2);
    };

    match run(&path, layer, reps, threads_sweep) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bigtea-kernelbench: {e}");
            ExitCode::FAILURE
        }
    }
}

const N_USED: i64 = 6;

fn run(
    path: &PathBuf,
    layer: u32,
    reps: usize,
    threads_sweep: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::open_split(path)?;
    let suffixes = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

    // Read the six experts this benchmark will use, once. From here on the
    // disk is out of the picture, which is the entire point.
    let mut slices = Vec::new();
    let mut resident_bytes = 0u64;
    for s in suffixes {
        let name = format!("blk.{layer}.{s}.weight");
        let loc = model
            .location(&name)
            .ok_or_else(|| format!("{name} is not in this container"))?
            .clone();
        let n_expert = *loc.dims.last().expect("stacked") as usize;
        let slice = loc.size / n_expert as u64;
        let mut bytes = Vec::with_capacity(N_USED as usize * slice as usize);
        for e in 0..N_USED as u64 {
            bytes.extend_from_slice(&model.read_tensor_range(&name, e * slice, slice)?);
        }
        resident_bytes += bytes.len() as u64;
        let mut dims = loc.dims.clone();
        *dims.last_mut().expect("stacked") = N_USED as u64;
        slices.push(Slice {
            name,
            ty: loc.ty,
            dims,
            bytes: Arc::new(bytes),
        });
    }

    let (ne0, ne1) = (slices[0].dims[0] as i64, slices[0].dims[1] as i64);
    let max_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    println!("layer      {layer}, {N_USED} experts, {ne0} x {ne1} per matrix");
    println!(
        "resident   {:.1} MiB of expert weights, read once, then never again",
        resident_bytes as f64 / (1 << 20) as f64
    );
    println!("threads    {max_threads}");
    println!("reps       {reps} per cell, best taken (a slow rep is interference)");
    println!();

    // FLOP for one pass at nt tokens: three matrices, six experts, 2 per MAC.
    let flop_per_token = 3.0 * N_USED as f64 * ne0 as f64 * ne1 as f64 * 2.0;

    println!(
        "{:>6}  {:>9}  {:>10}  {:>9}  {:>10}  {:>12}",
        "TOKENS", "MS/PASS", "MS/TOKEN", "GFLOP/s", "GiB/s", "vs 1 TOKEN"
    );
    let mut base_ms = 0.0f64;
    for &nt in BATCHES {
        let ms = time_pass(&slices, ne0, ne1, nt, reps, max_threads)?;
        if nt == 1 {
            base_ms = ms;
        }
        let gflops = flop_per_token * nt as f64 / (ms / 1000.0) / 1e9;
        // Weights are read once per pass whatever the batch, so this is the rate
        // the kernel pulls them out of DRAM.
        let gibs = resident_bytes as f64 / (1 << 30) as f64 / (ms / 1000.0);
        println!(
            "{nt:>6}  {ms:>8.2}  {:>9.2}  {gflops:>9.1}  {gibs:>10.2}  {:>11.2}x",
            ms / nt as f64,
            ms / base_ms,
        );
    }

    println!();
    println!("MS/TOKEN is the number that decides generation speed; GiB/s is the");
    println!("rate the kernel pulls weights out of DRAM. If GiB/s is far below what");
    println!("the machine can do, the kernel is the problem. If it is close, the");
    println!("pass is bandwidth-bound and only a batch or a different device helps.");

    if threads_sweep {
        println!();
        println!("{:>8}  {:>9}  {:>10}", "THREADS", "MS/PASS", "SPEEDUP");
        let mut one = 0.0f64;
        let mut t = 1;
        while t <= max_threads {
            let ms = time_pass(&slices, ne0, ne1, 1, reps, t)?;
            if t == 1 {
                one = ms;
            }
            println!("{t:>8}  {ms:>8.2}  {:>9.2}x", one / ms);
            t *= 2;
        }
        if !max_threads.is_power_of_two() {
            let ms = time_pass(&slices, ne0, ne1, 1, reps, max_threads)?;
            println!("{max_threads:>8}  {ms:>8.2}  {:>9.2}x", one / ms);
        }
    }

    // What the machine can actually do, so the numbers above have a ceiling to
    // be read against rather than a vibe.
    println!();
    reference(max_threads, reps)?;
    Ok(())
}

/// One expert-FFN pass, timed. Best of `reps` — a slow repetition is another
/// process on the machine, not the kernel.
fn time_pass(
    slices: &[Slice],
    ne0: i64,
    ne1: i64,
    nt: i64,
    reps: usize,
    threads: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Arena sized for the intermediates at this batch, with slack. ggml aborts
    // rather than erroring when it runs out, so this is deliberately generous.
    let arena = (ne1 * nt * N_USED * 4 * 8) as usize + (256 << 20);
    let ctx = Context::new(arena)?;
    let wctx = Context::new_no_alloc(8 << 20)?;
    let mut weights = WeightSet::new();
    for s in slices {
        // `bind_shared` so the bytes are handed over by pointer and not copied
        // once per cell — a copy would be measuring `memcpy`, not the kernel.
        weights.bind_shared(&wctx, &s.name, s.ty, &s.dims, s.bytes.clone())?;
    }
    let stack = |suffix: &str| {
        let s = slices
            .iter()
            .find(|s| s.name.contains(suffix))
            .expect("bound");
        ctx.reshape_3d(
            weights.get(&s.name).expect("bound"),
            s.dims[0] as i64,
            s.dims[1] as i64,
            N_USED,
        )
    };

    let x = ctx.new_f32_3d(ne0, 1, nt)?;
    // Values, not zeros: a quantised kernel's cost can depend on its data, and
    // an all-zero input is not a case the model ever sees.
    let vals: Vec<f32> = (0..ne0 * nt)
        .map(|i| ((i % 97) as f32 - 48.0) / 64.0)
        .collect();
    x.set_f32(&vals)?;

    let ids = ctx.new_i32_2d(N_USED, nt)?;
    let sel: Vec<i32> = (0..N_USED * nt).map(|i| (i % N_USED) as i32).collect();
    ids.set_i32(&sel)?;

    let gate = ctx.mul_mat_id(&stack("gate_exps")?, &x, &ids)?;
    let up = ctx.mul_mat_id(&stack("up_exps")?, &x, &ids)?;
    let act = ctx.swiglu_split(&gate, &up)?;
    let down = ctx.mul_mat_id(&stack("down_exps")?, &act, &ids)?;

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        // `compute` re-evaluates the whole ancestor graph, which here is exactly
        // what is wanted: every repetition redoes all three matmuls.
        ctx.compute(&down, threads)?;
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(best)
}

/// What this machine does on dense `f32` arithmetic and on a large copy.
///
/// Without these the expert numbers float free. "23 ms per block" means nothing
/// until it is next to what the same hardware manages when nothing is in its
/// way.
fn reference(threads: usize, reps: usize) -> Result<(), Box<dyn std::error::Error>> {
    let n = 1024i64;
    let ctx = Context::new((n * n * 4 * 4) as usize + (64 << 20))?;
    let a = ctx.new_f32_2d(n, n)?;
    let b = ctx.new_f32_2d(n, n)?;
    let vals: Vec<f32> = (0..n * n).map(|i| (i % 13) as f32 * 0.1).collect();
    a.set_f32(&vals)?;
    b.set_f32(&vals)?;
    let c = ctx.mul_mat(&a, &b)?;
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        ctx.compute(&c, threads)?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    let gflops = 2.0 * (n as f64).powi(3) / best / 1e9;
    println!("reference  dense f32 {n}x{n} matmul: {gflops:.0} GFLOP/s on {threads} threads");

    // Memory bandwidth, measured the crude honest way.
    let words = 64 << 20; // 256 MiB of f32
    let src = vec![1.0f32; words];
    let mut dst = vec![0.0f32; words];
    let mut best_bw = f64::INFINITY;
    for _ in 0..reps.max(3) {
        let t = Instant::now();
        dst.copy_from_slice(&src);
        best_bw = best_bw.min(t.elapsed().as_secs_f64());
    }
    let gibs = 2.0 * (words * 4) as f64 / (1 << 30) as f64 / best_bw;
    println!("reference  single-threaded memcpy: {gibs:.1} GiB/s (read+write)");
    Ok(())
}
