//! `chaos-draw` — an image from a prompt, on a machine that cannot hold the models.
//!
//! ```text
//! chaos-draw "a red apple on a white table" --grid 64 --steps 20 --out apple.png
//! ```
//!
//! Four files, 16.7 GB: **Qwen3-VL-8B** turns the prompt into conditioning, two
//! copies of **Ideogram 4** — the conditional model and a separately trained
//! unconditional twin, 5.26 GiB each — denoise a latent, and the **FLUX.2**
//! autoencoder turns it into pixels. `chaos-pull` fetches them.
//!
//! # What is verified, and what is not
//!
//! **This was an example until v0.0.12**, held back with the note that it would
//! become a binary "when there is evidence, not before" — because a diffusion
//! pipeline that is subtly wrong produces a *plausible* picture, so running is
//! not evidence and neither is looking. What is now measured:
//!
//! - the autoencoder round-trips real photographs at **36.09–40.89 dB**, with
//!   the check ablated three ways first (each deliberate bug still produced a
//!   recognisable picture, and each scored 14–32 dB);
//! - the text encoder answers `" Paris"` after "The capital of France is" at
//!   **logit 22.58**, so its attention, rotary positions, QK norm, grouped-query
//!   broadcasting and causal mask are all right;
//! - the denoiser scores **0.85 velocity cosine** against a real latent at
//!   512×512, decomposed into its two terms rather than taken as one number.
//!
//! **What is still imperfect is object form.** At 1024×1024 the output is
//! photorealistic — skin texture, individual hairs, catchlights — and it follows
//! the prompt's colour and scene. A named object may still come out the wrong
//! shape. Structured, JSON-shaped prompts condition roughly three times as
//! strongly as a bare phrase, which is what these models were trained on.
//!
//! # What it costs
//!
//! Both denoisers are dense: every one of their 5.26 GiB is read on every step,
//! and twice per step when guidance is on. `--grid` is the lever — the token
//! count is its square, and attention is quadratic in that again.
//!
//! | grid | image | tokens | note |
//! |---|---|---|---|
//! | 16 | 256×256 | 256 | quick, and flat |
//! | 32 | 512×512 | 1024 | faceted |
//! | 64 | 1024×1024 | 4096 | photorealistic, and hours on a laptop |
//!
//! Memory is the other ceiling, and an exhausted ggml arena aborts the process
//! with no message — so the requirement is printed before anything starts, and
//! a request too large to decode is refused up front with the largest grid that
//! would work.

use chaos_image::pipeline::{generate, Paths, Request, Stage};
use chaos_image::png;

fn main() {
    let mut req = Request {
        threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        ..Default::default()
    };
    let mut out = String::from("chaos-image.png");
    let mut dir = {
        let home = std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .unwrap_or_default();
        std::path::PathBuf::from(home).join(".chaos").join("models")
    };

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let take = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--grid" => {
                req.grid = take(i).parse().unwrap_or(req.grid);
                i += 2;
            }
            "--steps" => {
                req.steps = take(i).parse().unwrap_or(req.steps);
                i += 2;
            }
            "--cfg" => {
                req.cfg = take(i).parse().unwrap_or(req.cfg);
                i += 2;
            }
            "--seed" => {
                req.seed = take(i).parse().unwrap_or(req.seed);
                i += 2;
            }
            "-t" | "--threads" => {
                req.threads = take(i).parse().unwrap_or(req.threads);
                i += 2;
            }
            "--out" | "-o" => {
                out = take(i);
                i += 2;
            }
            "--models" => {
                dir = std::path::PathBuf::from(take(i));
                i += 2;
            }
            "-h" | "--help" => {
                usage();
                return;
            }
            "--version" => {
                println!("chaos-draw {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            other if other.starts_with("--") => {
                eprintln!("chaos-draw: unknown option {other:?}\n");
                usage();
                std::process::exit(2);
            }
            other => {
                if req.prompt.is_empty() {
                    req.prompt = other.to_string();
                }
                i += 1;
            }
        }
    }
    if req.prompt.is_empty() {
        usage();
        std::process::exit(2);
    }

    let paths = Paths::under(&dir);
    println!("prompt       {:?}", req.prompt);
    println!(
        "image        {0}x{0} from a {1}x{1} grid, {2} tokens",
        req.image_size(),
        req.grid,
        req.tokens()
    );
    println!(
        "sampler      {} steps, cfg {}, seed {}, {} threads",
        req.steps, req.cfg, req.seed, req.threads
    );
    let passes = req.steps * if req.cfg == 1.0 { 1 } else { 2 };
    println!(
        "work         {passes} denoiser passes, {:.1} GiB of reads",
        passes as f64 * 5.26
    );
    // The arena is the real ceiling on image size, and an exhausted one kills
    // the process with no message -- so it is printed before anything starts.
    let (arena, _) = chaos_image::pipeline::arena_estimate(&req);
    println!(
        "memory       {:.1} GiB per denoiser layer, {:.1} GiB to decode",
        arena as f64 / (1u64 << 30) as f64,
        chaos_image::vae::decode_planned_bytes(req.image_size() as usize, req.image_size() as usize)
            as f64
            / (1u64 << 30) as f64
    );

    let started = std::time::Instant::now();
    let mut step_started = std::time::Instant::now();
    let image = generate(&paths, &req, &mut |s| match s {
        Stage::Text { tokens } => {
            println!("\n[1/3] encoding the prompt -- {tokens} tokens");
            step_started = std::time::Instant::now();
        }
        Stage::Step { index, total } => {
            if index == 0 {
                println!(
                    "      done in {:.1}s\n\n[2/3] denoising",
                    step_started.elapsed().as_secs_f32()
                );
            } else {
                let per = started.elapsed().as_secs_f32() / index as f32;
                eprint!(
                    "\r      step {}/{}  {:.0}s/step  about {:.0}s left      ",
                    index + 1,
                    total,
                    per,
                    per * (total - index) as f32
                );
            }
        }
        Stage::Decode => {
            eprintln!();
            println!("\n[3/3] decoding to pixels");
        }
    });

    let image = match image {
        Ok(i) => i,
        Err(e) => {
            eprintln!("\n{e}");
            std::process::exit(1);
        }
    };

    match png::encode_rgb(image.width as u32, image.height as u32, &image.rgb) {
        Some(bytes) => match std::fs::write(&out, &bytes) {
            Ok(()) => {
                println!(
                    "\nwrote {out} -- {}x{}, {} KiB, in {:.0}s",
                    image.width,
                    image.height,
                    bytes.len() >> 10,
                    started.elapsed().as_secs_f32()
                );
            }
            Err(e) => {
                eprintln!("cannot write {out}: {e}");
                std::process::exit(1);
            }
        },
        None => {
            eprintln!("cannot encode the image");
            std::process::exit(1);
        }
    }

    // A flat image is the loudest failure this can have, and it is worth saying
    // rather than leaving to the eye.
    let mean = image.rgb.iter().map(|v| *v as f64).sum::<f64>() / image.rgb.len() as f64;
    let var = image
        .rgb
        .iter()
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / image.rgb.len() as f64;
    println!(
        "pixels       mean {mean:.1}, standard deviation {:.1}",
        var.sqrt()
    );
    if var.sqrt() < 2.0 {
        println!("             NEARLY FLAT -- that is not an image");
    }
}

/// What this does and what it costs, before it is asked to spend two hours.
///
/// **The caveat is in the help text, not only in the README.** These models
/// follow colour and scene and can still get an object's form wrong, and
/// somebody who reads that after waiting has been misled by omission.
fn usage() {
    println!(
        "chaos-draw {} -- an image from a prompt",
        env!("CARGO_PKG_VERSION")
    );
    println!();
    println!("usage: chaos-draw \"a prompt\" [options]");
    println!();
    println!("  --grid N       latent grid; the image is 16x this. 16, 32, 64 (default 32)");
    println!("  --steps N      denoising steps (default 20)");
    println!("  --cfg F        guidance; 1 turns it off and halves the work (default 4)");
    println!("  --seed N       reproducible noise");
    println!("  -t, --threads N");
    println!("  -o, --out FILE where to write the PNG (default chaos-image.png)");
    println!("  --models DIR   where the four model files are");
    println!("  --version");
    println!();
    println!("  grid 16 -> 256x256, 256 tokens      quick, and flat");
    println!("  grid 32 -> 512x512, 1024 tokens     faceted");
    println!("  grid 64 -> 1024x1024, 4096 tokens   photorealistic, and slow");
    println!();
    println!("Needs four files (16.7 GB): Qwen3-VL-8B for the prompt, two Ideogram 4");
    println!("denoisers, and the FLUX.2 autoencoder. `chaos-pull` fetches them.");
    println!();
    println!("The autoencoder is verified to 36-41 dB and the text encoder is diffed");
    println!("against a reference. What is still imperfect is OBJECT FORM: colour and");
    println!("scene follow the prompt, a named object may come out the wrong shape.");
    println!("Structured, JSON-shaped prompts condition about 3x more strongly than a");
    println!("bare phrase -- that is what these models were trained on.");
}
