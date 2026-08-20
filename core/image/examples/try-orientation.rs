//! Which way up is the autoencoder's latent?
//!
//! # Why a round trip cannot answer this
//!
//! The decoder scores 36 dB against a photograph, and that would be just as true
//! if the encoder flipped the image and the decoder flipped it back. **Two
//! opposite errors cancel exactly**, and every test so far ran both halves.
//!
//! A latent from the *denoiser* never goes through the encoder, so a flip that
//! cancels in a round trip does not cancel there — and the first 1024x1024
//! generation came out upside down.
//!
//! So this asks each half on its own, with a picture that is not symmetric:
//! white in the **top-left quadrant** and black elsewhere.
//!
//! - Encode it, and see which quadrant of the latent stands out.
//! - Decode that latent, and see which quadrant of the image is bright.
//!
//! Agreement between the two proves only that they are consistent. What matters
//! is the **first** answer: the latent's bright quadrant must be top-left, in
//! the same row-major, top-row-first order everything else here uses.
//!
//! ```text
//! cargo run --release -p chaos-image --example try-orientation
//! ```

use chaos_image::{safetensors::SafeTensors, vae};

fn main() {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    let path = std::path::Path::new(&home)
        .join(".chaos")
        .join("models")
        .join("flux2-vae.safetensors");
    let file = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("{}: {e}", path.display());
        std::process::exit(1);
    });
    let st = SafeTensors::parse(&file).expect("parse the autoencoder");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());

    // -- a picture with no symmetry to hide behind ---------------------------
    const N: usize = 256;
    let mut rgb = vec![0u8; N * N * 3];
    for y in 0..N / 2 {
        for x in 0..N / 2 {
            for c in 0..3 {
                rgb[(y * N + x) * 3 + c] = 255;
            }
        }
    }
    println!("input        {N}x{N}, white in the TOP-LEFT quadrant only");

    // -- encode --------------------------------------------------------------
    let (lw, lh) = (N / vae::SCALE, N / vae::SCALE);
    let latent = {
        let ctx = chaos_ggml::Context::new((512 << 20) + N * N * 48 * 1024).expect("arena");
        let v = vae::Vae::new(&st, &file, &ctx);
        let img = ctx.new_f32_4d(N as i64, N as i64, 3, 1).expect("image");
        img.set_f32(&vae::from_rgb8(&rgb, N, N)).expect("set");
        let moments = v.encode(&img).expect("encode");
        let mean = v.latent_mean(&moments).expect("mean");
        ctx.compute(&mean, threads).expect("compute");
        mean.to_vec_f32()
    };

    // Which quadrant of the latent is unlike the others? Measured as the mean
    // absolute value over every channel, which needs no idea of what a latent
    // channel *means*.
    let quad = |v: &[f32], w: usize, h: usize, ch: usize, right: bool, bottom: bool| -> f64 {
        let (x0, x1) = if right { (w / 2, w) } else { (0, w / 2) };
        let (y0, y1) = if bottom { (h / 2, h) } else { (0, h / 2) };
        let mut sum = 0.0;
        let mut n = 0;
        for c in 0..ch {
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += v[x + w * y + w * h * c].abs() as f64;
                    n += 1;
                }
            }
        }
        sum / n as f64
    };
    let l = |r, b| quad(&latent, lw, lh, vae::LATENT_CHANNELS as usize, r, b);
    println!("\nlatent quadrant magnitudes ({lw}x{lh}):");
    println!(
        "   top-left {:.3}    top-right {:.3}",
        l(false, false),
        l(true, false)
    );
    println!(
        "bottom-left {:.3} bottom-right {:.3}",
        l(false, true),
        l(true, true)
    );

    let brightest = [
        ("top-left", l(false, false)),
        ("top-right", l(true, false)),
        ("bottom-left", l(false, true)),
        ("bottom-right", l(true, true)),
    ]
    .into_iter()
    .max_by(|a, b| a.1.total_cmp(&b.1))
    .unwrap()
    .0;
    println!("\nthe latent's distinct quadrant is: {brightest}");
    if brightest == "top-left" {
        println!("  which matches the input -- the encoder does not flip.");
    } else if brightest == "bottom-left" {
        println!("  the input was TOP-left. THE ENCODER FLIPS VERTICALLY.");
    } else if brightest == "top-right" {
        println!("  the input was top-LEFT. THE ENCODER FLIPS HORIZONTALLY.");
    } else {
        println!("  the input was top-left. THE ENCODER ROTATES BY 180 DEGREES.");
    }

    // -- decode the same latent ----------------------------------------------
    let (pixels, _) = vae::decode_planned(&st, &file, &latent, lw as i64, lh as i64, threads)
        .unwrap_or_else(|e| {
            eprintln!("decode: {e}");
            std::process::exit(1);
        });
    let out = vae::to_rgb8(&pixels, N, N);
    let bright = |right: bool, bottom: bool| -> f64 {
        let (x0, x1) = if right { (N / 2, N) } else { (0, N / 2) };
        let (y0, y1) = if bottom { (N / 2, N) } else { (0, N / 2) };
        let mut sum = 0.0;
        let mut n = 0;
        for y in y0..y1 {
            for x in x0..x1 {
                sum += out[(y * N + x) * 3] as f64;
                n += 1;
            }
        }
        sum / n as f64
    };
    println!("\ndecoded image quadrant brightness:");
    println!(
        "   top-left {:.0}    top-right {:.0}",
        bright(false, false),
        bright(true, false)
    );
    println!(
        "bottom-left {:.0} bottom-right {:.0}",
        bright(false, true),
        bright(true, true)
    );
    let same =
        bright(false, false) > bright(false, true) && bright(false, false) > bright(true, false);
    println!(
        "\nround trip {} -- which is the part that was always true,",
        if same {
            "puts it back top-left"
        } else {
            "MOVED IT"
        }
    );
    println!("and says nothing on its own about which half is right.");
}
