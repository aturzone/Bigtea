//! Measure how compressible a layer's expert bank is, without running the model.
//!
//! Usage: `chaos-spectrum <model.gguf> [--layer N] [--tensor up|gate|down]
//!                         [--experts N] [--rank K] [--random]`
//!
//! # The one number this exists to produce
//!
//! Streaming an expert costs its full bytes, and on DeepSeek-V4-Flash that is
//! 3.21 GiB per token — the entire reason generation is slow. Every byte-cutting
//! idea this project has tried has died against measurement: contextual
//! sparsity (V4-Flash's experts are 9.1% negligible, not the 80-95% the dense-FFN
//! literature reports) and pinned hot sets (37.5% cross-subject against 25.0%
//! for caching at random).
//!
//! This asks a different question. Not "which parts of an expert can be
//! skipped" but **"do all 256 experts in a layer share a subspace?"** If the
//! rows of every expert lie near a common `r`-dimensional subspace, the bank
//! factors as `W_i ≈ C_i Bᵀ` — one shared `B` resident for the layer, and only
//! the small `C_i` streamed. That is `ne0 / r` on bytes, and `B ᵀx` is computed
//! once per layer and shared by all six selected experts, so the arithmetic gets
//! *cheaper* too.
//!
//! Whether that is real is entirely decided by how fast the bank's singular
//! spectrum decays, which needs no forward pass, no tokenizer, and no GPU —
//! only the weights. Hence this tool.
//!
//! # Read the control, not the headline
//!
//! `--random` runs the identical pipeline over matched-shape noise. A rank-512
//! subspace of a 4096-dimensional space holds ~12.5% of *any* matrix's energy by
//! construction, and this project has already published one figure that turned
//! out to be its own null. **A result here means nothing except as a difference
//! from the control**, so the control is printed alongside every run.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use chaos_arch::spectrum::{energy_curve, gaussian, Gram};
use chaos_ggml::Context;
use chaos_model::Model;

/// Ranks worth reporting: powers of two, because the win is `ne0 / r` and only
/// a factor-of-two change in that is worth acting on.
const RANKS: &[usize] = &[16, 32, 64, 128, 256, 512, 1024];

/// Fixed so a run is reproducible and two layers differ for a reason other than
/// the random sketch.
const SEED: u64 = 0xB16_7EA;

fn main() -> ExitCode {
    // **Before anything treats an argument as a path.** Without this,
    // `chaos-spectrum --version` reported "cannot find the file specified" -- the
    // flag was being opened as a model. `--version` is how a person checks
    // whether an update landed, so it has to answer on whichever binary they
    // happen to type.
    if std::env::args()
        .skip(1)
        .any(|a| a == "--version" || a == "-V")
    {
        println!("chaos-spectrum {}", env!("CARGO_PKG_VERSION"));
        std::process::exit(0);
    }
    let mut path: Option<PathBuf> = None;
    let mut layer = 20u32;
    let mut which = "up".to_string();
    let mut experts = 32usize;
    let mut rank = 1024usize;
    let mut random = false;
    let mut power_iters = 3usize;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let take = |i: &mut usize| -> Option<String> {
            let v = args.get(*i + 1).cloned();
            *i += 2;
            v
        };
        match args[i].as_str() {
            "--layer" => layer = take(&mut i).and_then(|v| v.parse().ok()).unwrap_or(layer),
            "--tensor" => which = take(&mut i).unwrap_or(which),
            "--experts" => experts = take(&mut i).and_then(|v| v.parse().ok()).unwrap_or(experts),
            "--rank" => rank = take(&mut i).and_then(|v| v.parse().ok()).unwrap_or(rank),
            "--power-iters" => {
                power_iters = take(&mut i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(power_iters)
            }
            "--random" => {
                random = true;
                i += 1;
            }
            "-h" | "--help" => {
                usage();
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
        usage();
        return ExitCode::from(2);
    };

    match run(&path, layer, &which, experts, rank, power_iters, random) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("chaos-spectrum: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    println!("usage: chaos-spectrum <model.gguf> [options]");
    println!();
    println!("  --layer N       which block (default 20)");
    println!("  --tensor NAME   up | gate | down (default up)");
    println!("  --experts N     how many experts to sample (default 32)");
    println!("  --rank K        largest rank to report (default 1024)");
    println!("  --power-iters N subspace iterations (default 3)");
    println!("  --random        also run matched-shape noise as a control");
    println!();
    println!("Reports how much of the expert bank's energy a shared rank-r");
    println!("subspace holds. Bytes streamed per expert would fall by ne0/r.");
}

fn run(
    path: &PathBuf,
    layer: u32,
    which: &str,
    experts: usize,
    rank: usize,
    power_iters: usize,
    random: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let suffix = match which {
        "up" => "ffn_up_exps",
        "gate" => "ffn_gate_exps",
        "down" => "ffn_down_exps",
        other => return Err(format!("--tensor must be up, gate or down, not {other:?}").into()),
    };
    let name = format!("blk.{layer}.{suffix}.weight");

    let model = Model::open_split(path)?;
    let loc = model
        .location(&name)
        .ok_or_else(|| format!("{name} is not in this container"))?
        .clone();

    // Stacked expert tensors carry the expert index in the *last* dimension;
    // ne0 is the reduction dimension and therefore the space the shared basis
    // would live in.
    let n_expert = *loc.dims.last().expect("stacked expert tensor") as usize;
    let ne0 = loc.dims[0] as usize;
    let ne1 = loc.dims[1] as usize;
    let slice_bytes = loc.size / n_expert as u64;
    let sampled = experts.min(n_expert);

    println!("tensor     {name}");
    println!(
        "shape      ne0 {ne0} x ne1 {ne1} x {n_expert} experts, {:?}",
        loc.ty
    );
    println!(
        "sampling   {sampled} experts, {:.1} MiB on disk each",
        slice_bytes as f64 / (1 << 20) as f64
    );
    println!("basis      lives in R^{ne0}; a rank-r basis costs {ne0}xr, shared by the layer");
    println!();

    let t0 = Instant::now();
    let g = accumulate(&model, &name, &loc, ne0, ne1, sampled, slice_bytes, false)?;
    let acc = t0.elapsed();

    let t1 = Instant::now();
    let curve = energy_curve(&g, rank, power_iters, SEED);
    let solve = t1.elapsed();

    let control = if random {
        let gr = accumulate(&model, &name, &loc, ne0, ne1, sampled, slice_bytes, true)?;
        Some(energy_curve(&gr, rank, power_iters, SEED))
    } else {
        None
    };

    println!(
        "{:>6}  {:>10}  {:>10}  {:>9}  {:>10}",
        "RANK", "ENERGY", "CONTROL", "BYTES/x", "GiB/token"
    );
    // What the whole layer's routed reads cost today, so the last column is in
    // the unit that actually decides tok/s.
    let gib_today = 3.21f64;
    for &r in RANKS {
        if r > curve.eigenvalues.len() || r > ne0 {
            continue;
        }
        let e = curve.captured(r).expect("inside k");
        let c = control
            .as_ref()
            .and_then(|c| c.captured(r))
            .map(|v| format!("{:.1}%", v * 100.0))
            .unwrap_or_else(|| "-".into());
        let factor = ne0 as f64 / r as f64;
        println!(
            "{r:>6}  {:>9.1}%  {c:>10}  {:>8.1}x  {:>10.2}",
            e * 100.0,
            factor,
            gib_today / factor
        );
    }
    println!();
    for want in [0.90, 0.95, 0.99] {
        match curve.rank_for(want) {
            Some(r) => println!(
                "{:.0}% of the bank's energy needs rank {r} -> {:.1}x fewer bytes per expert",
                want * 100.0,
                ne0 as f64 / r as f64
            ),
            None => println!(
                "{:.0}% of the bank's energy needs rank > {} -- not compressible at this budget",
                want * 100.0,
                curve.eigenvalues.len().min(ne0)
            ),
        }
    }
    println!();
    println!(
        "accumulate {:.1}s, eigensolve {:.1}s",
        acc.as_secs_f64(),
        solve.as_secs_f64()
    );
    println!();
    println!("Energy is a screen, not a quality result. A rank that holds 95% of");
    println!("the Frobenius norm still has to be checked against the oracle before");
    println!("anything is claimed -- a wrong forward pass gives fluent nonsense.");
    Ok(())
}

/// `G = Σ_i W_iᵀ W_i` over the sampled experts, one expert at a time.
///
/// Streamed rather than stacked: the full bank dequantised to `f32` is
/// `256 x 4096 x 2048 x 4 B` = 8.6 GB and would not fit. One expert is 33 MB and
/// the Gram it contributes is `ne0 x ne0`, so memory is flat in the sample size.
///
/// `ggml` does the multiply. It is already linked, threaded and vectorised, and
/// this is 34 GFLOP per expert — enough that a hand-rolled loop would be the
/// slowest part of the tool by an order of magnitude.
#[allow(clippy::too_many_arguments)]
fn accumulate(
    model: &Model,
    name: &str,
    loc: &chaos_model::Location,
    ne0: usize,
    ne1: usize,
    sampled: usize,
    slice_bytes: u64,
    random: bool,
) -> Result<Gram, Box<dyn std::error::Error>> {
    let mut g = Gram::zeros(ne0);
    let elements = ne0 * ne1;
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    for e in 0..sampled {
        let w: Vec<f32> = if random {
            // Matched shape and matched scale. Scale matters: the control is a
            // claim about *structure*, and a variance mismatch would make the
            // two runs incomparable for a reason that has nothing to do with it.
            gaussian(elements, 0x5EED + e as u64)
        } else {
            let raw = model.read_tensor_range(name, e as u64 * slice_bytes, slice_bytes)?;
            chaos_ggml::dequantize(loc.ty, &raw, elements)?
        };

        // A (ne0 x ne1) -> transpose to (ne1 x ne0) so mul_mat contracts over
        // ne1 and yields the ne0 x ne0 Gram. `cont` is required: mul_mat will
        // not take the strided view a bare transpose produces.
        let bytes = (elements * 2 + ne0 * ne0) * 4 + (64 << 20);
        let ctx = Context::new(bytes)?;
        let a = ctx.new_f32_2d(ne0 as i64, ne1 as i64)?;
        a.set_f32(&w)?;
        let at = ctx.cont(&ctx.transpose(&a)?)?;
        let block = ctx.mul_mat(&at, &at)?;
        // Not `0`: ggml floors the thread count at 1 rather than defaulting to
        // all cores, and this multiply is the whole cost of the tool.
        ctx.compute(&block, threads)?;
        g.accumulate(&block.to_vec_f32());

        if !random && (e + 1) % 8 == 0 {
            eprintln!("  {} / {sampled} experts", e + 1);
        }
    }
    Ok(g)
}
