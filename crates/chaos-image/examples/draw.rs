//! Draw an image from a prompt.
//!
//! ```text
//! cargo run --release -p chaos-image --example draw -- "a red apple on a white table" \
//!     --grid 32 --steps 20 --cfg 4 --out apple.png
//! ```
//!
//! **An example rather than a binary, on purpose.** Everything it calls is
//! tested, and the autoencoder at the end is verified to 36 dB — but nobody has
//! yet shown that the *whole chain* produces the right picture, and a diffusion
//! pipeline that is subtly wrong produces a plausible one. It becomes
//! `chaos-draw` when there is evidence, not before.
//!
//! # What it costs
//!
//! Both denoisers are dense: every one of their 5.26 GiB is read on every step,
//! twice per step when guidance is on. `--grid` is the lever — the token count
//! is its square, and attention is quadratic in that again.
//!
//! | grid | image | tokens |
//! |---|---|---|
//! | 16 | 256x256 | 256 |
//! | 32 | 512x512 | 1024 |
//! | 64 | 1024x1024 | 4096 |

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
            other if other.starts_with("--") => {
                eprintln!("draw: unknown option {other:?}");
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
        eprintln!(
            "usage: draw \"a prompt\" [--grid N] [--steps N] [--cfg F] [--seed N] [--out FILE]"
        );
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
        chaos_image::vae::decode_arena_bytes(req.image_size() as usize, req.image_size() as usize)
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
