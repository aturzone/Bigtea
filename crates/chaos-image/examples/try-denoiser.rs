//! One forward pass of the denoiser, at the smallest grid that means anything.
//!
//! **An example, not a test.** Every shape in this graph is new — a four-axis
//! permute, a strided pair view, a `[2, 2, 128, tokens]` rotary table — and ggml
//! *aborts* on a shape it dislikes rather than returning an error, taking the
//! whole test binary with it. A crash here costs one `cargo run` and names the
//! call that did it.
//!
//! ```text
//! cargo run --release -p chaos-image --example try-denoiser -- [grid] [layers]
//! ```
//!
//! It runs the **unconditional** twin, which needs no text encoder: that model
//! takes no context at all, so a forward pass is reachable before any of the
//! conditioning is written. What it proves is that the graph builds, runs and
//! produces finite numbers of the right shape — *not* that the arithmetic is
//! right. Nothing here is evidence of a correct image.

use chaos_image::dit::{Config, Denoiser, Inputs};
use chaos_model::Model;

fn main() {
    let mut args = std::env::args().skip(1);
    let grid: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let layers: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);

    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    let path = format!("{home}/.chaos/models/ideogram4_uncond-Q4_0.gguf");

    let model = match Model::open_split(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot open {path}: {e}");
            eprintln!("  fetch it with: chaos-pull ideogram-4-uncond");
            std::process::exit(1);
        }
    };

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut d = Denoiser::open(model, threads);
    let full = d.config;
    println!(
        "container    {} layers, {} wide, {} heads of {}",
        full.num_layers,
        full.emb_dim,
        full.num_heads,
        full.head_dim()
    );

    let missing = d.missing();
    if !missing.is_empty() {
        eprintln!(
            "missing {} tensors, first: {:?}",
            missing.len(),
            &missing[..1]
        );
        std::process::exit(1);
    }
    println!("tensors      all {} present", full.required_tensors().len());

    // Fewer layers than the container has, so a shape bug shows up in seconds
    // rather than minutes. The arithmetic per layer is identical.
    d.config = Config {
        num_layers: layers.min(full.num_layers),
        ..full
    };
    println!(
        "running      {} of {} layers, {grid}x{grid} grid ({} image tokens)",
        d.config.num_layers,
        full.num_layers,
        grid * grid
    );

    // A deterministic latent, so two runs are comparable.
    let n = (grid * grid * full.in_channels) as usize;
    let latent: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7391).sin() * 0.9).collect();

    let started = std::time::Instant::now();
    let inp = Inputs {
        latent: &latent,
        grid_w: grid,
        grid_h: grid,
        timestep: 500.0,
        context: &[],
        context_len: 0,
    };
    let out = match d.forward_with(&inp, &mut |i, n| {
        eprint!("\r  layer {i}/{n}      ");
    }) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("\nforward: {e}");
            std::process::exit(1);
        }
    };
    eprintln!();

    println!("took         {:.1}s", started.elapsed().as_secs_f32());
    println!(
        "output       {} values (input was {})",
        out.len(),
        latent.len()
    );
    if out.len() != latent.len() {
        eprintln!("SHAPE MISMATCH -- the velocity must match the latent");
        std::process::exit(1);
    }

    let finite = out.iter().filter(|v| v.is_finite()).count();
    let (lo, hi) = out
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
    println!("finite       {finite}/{}", out.len());
    println!("range        {lo:.4} .. {hi:.4}");
    println!("mean         {mean:.4}, rms {rms:.4}");

    if finite != out.len() {
        eprintln!("NOT FINITE -- something overflowed");
        std::process::exit(1);
    }
    if rms == 0.0 {
        eprintln!("ALL ZERO -- the graph ran and computed nothing");
        std::process::exit(1);
    }
    println!("\nthe graph builds, runs, and returns finite numbers of the right shape.");
    println!("that is NOT evidence the arithmetic is right -- only a picture is.");
}
