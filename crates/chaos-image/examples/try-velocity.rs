//! Does the denoiser point at the image, and how hard?
//!
//! # The only check on a denoiser that is not "look at the picture"
//!
//! A rectified-flow model is trained on a straight line between a real latent
//! and pure noise. At noise level `sigma` the input is
//!
//! ```text
//! x = latent * (1 - sigma) + noise * sigma
//! ```
//!
//! and the velocity it should predict is exactly `noise - latent`. **Both terms
//! are known here**, because the autoencoder's encoder — already verified to 36
//! dB by round trip — turns a real photograph into a real latent, and the noise
//! is ours. So the model's answer can be scored against the truth by cosine
//! similarity, with no image involved and nothing to fool the eye.
//!
//! | cosine | meaning |
//! |---|---|
//! | near +1 | the denoiser is right |
//! | near 0 | it is answering noise; something is wrong |
//! | near -1 | correct magnitude, **inverted sign** |
//!
//! The sign case is worth naming: the timestep runs backwards *and* the output
//! is negated, and getting exactly one of the two wrong lands here.
//!
//! ```text
//! powershell -File scripts/image-to-ppm.ps1 -In photo.jpg -Out p.ppm -Size 256
//! cargo run --release -p chaos-image --example try-velocity -- p.ppm
//! ```
//!
//! It uses the **unconditional** twin, so no text encoder is involved: that
//! model predicts what an average image would do, which is most of the velocity
//! at any noticeable noise level.

use chaos_image::pipeline::Noise;
use chaos_image::{dit, flow, safetensors::SafeTensors, vae};
use chaos_model::Model;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(ppm) = args.next() else {
        eprintln!("usage: try-velocity <photo.ppm> [sigma ...]");
        std::process::exit(2);
    };
    let sigmas: Vec<f32> = {
        let rest: Vec<f32> = args.filter_map(|s| s.parse().ok()).collect();
        if rest.is_empty() {
            vec![0.9, 0.7, 0.5, 0.3]
        } else {
            rest
        }
    };

    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    let dir = std::path::Path::new(&home).join(".chaos").join("models");

    // -- a real latent, from the verified encoder ----------------------------
    let (w, h, rgb) = read_ppm(&ppm).unwrap_or_else(|e| {
        eprintln!("{ppm}: {e}");
        std::process::exit(1);
    });
    println!("photo        {ppm} -- {w}x{h}");

    let ae_path = dir.join("flux2-vae.safetensors");
    let file = std::fs::read(&ae_path).unwrap_or_else(|e| {
        eprintln!("{}: {e}", ae_path.display());
        std::process::exit(1);
    });
    let st = SafeTensors::parse(&file).expect("parse the autoencoder");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());

    let (lw, lh) = (w / vae::SCALE, h / vae::SCALE);
    let latent = {
        let arena = (512 << 20) + w * h * 48 * 1024;
        let ctx = chaos_ggml::Context::new(arena).expect("encoder arena");
        let v = vae::Vae::new(&st, &file, &ctx);
        let img = ctx
            .new_f32_4d(w as i64, h as i64, 3, 1)
            .expect("image tensor");
        img.set_f32(&vae::from_rgb8(&rgb, w, h)).expect("set image");
        let moments = v.encode(&img).expect("encode");
        let mean = v.latent_mean(&moments).expect("mean");
        ctx.compute(&mean, threads).expect("compute");
        mean.to_vec_f32()
    };
    let stats = |name: &str, v: &[f32]| {
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        let (lo, hi) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
        println!("{name:<12} mean {mean:>8.4}  sd {sd:>8.4}  range {lo:>8.3} .. {hi:<8.3}");
    };
    stats("latent", &latent);

    // The denoiser works on a 2x2-packed grid, not the autoencoder's latent.
    let c = dit::Config::default();
    let packed = vae::pack_latent(&latent, lw, lh, c.ae_channels as usize, c.patch as usize);
    let (gw, gh) = (lw / 2, lh / 2);
    println!("grid         {gw}x{gh} ({} image tokens)", gw * gh);

    // **The file's own latent normalisation.** The diffusion model is trained on
    // a normalised latent; `bn.running_mean` and `bn.running_var` are how this
    // autoencoder says which one. Set CHAOS_RAW_LATENT=1 to skip it and watch
    // the model stop being able to see the image.
    let mut packed = packed;
    if std::env::var("CHAOS_RAW_LATENT").is_err() {
        match vae::latent_stats(&st, &file) {
            Some((m, v)) => {
                vae::normalize_latent(&mut packed, &m, &v);
                stats("normalised", &packed);
            }
            None => println!("(no bn.* tensors in this file)"),
        }
    } else {
        println!("(raw latent, normalisation skipped)");
    }

    // **Test the one convention that was derived and never measured.** The 2x2
    // patch is folded into the channel index as `px + 2*py + 4*c`; swapping the
    // two patch axes, or moving the latent channel to the fast end, both produce
    // a latent of exactly the right shape. CHAOS_PACK=swap|cfast tries them.
    if let Ok(mode) = std::env::var("CHAOS_PACK") {
        let (ae, pp) = (c.ae_channels as usize, c.patch as usize);
        let plane = packed.len() / (ae * pp * pp);
        let mut out = vec![0.0f32; packed.len()];
        for ch in 0..ae * pp * pp {
            let (px, py, cc) = (ch % pp, (ch / pp) % pp, ch / (pp * pp));
            let src = match mode.as_str() {
                "swap" => py + pp * px + pp * pp * cc,
                "cfast" => cc + ae * px + ae * pp * py,
                _ => ch,
            };
            out[ch * plane..(ch + 1) * plane]
                .copy_from_slice(&packed[src * plane..(src + 1) * plane]);
        }
        packed = out;
        println!("(channel order: {mode})");
    }

    // -- the denoiser --------------------------------------------------------
    let path = dir.join("ideogram4_uncond-Q4_0.gguf");
    let model = Model::open_split(&path).unwrap_or_else(|e| {
        eprintln!("{}: {e}", path.display());
        std::process::exit(1);
    });
    let d = dit::Denoiser::open(model, threads);

    // One cosine says "wrong" without saying which half is wrong. The velocity
    // is `noise - latent`, so scoring against each term separately separates
    // "cannot see the noise" from "cannot see the image" — and only the second
    // has anything to do with spatial structure.
    println!(
        "\n{:>7} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "sigma", "timestep", "cos(v)", "cos(-L)", "cos(N)", "x0 err"
    );
    for sigma in &sigmas {
        let noise = Noise::seeded(1234).normals(packed.len());
        let x: Vec<f32> = packed
            .iter()
            .zip(&noise)
            .map(|(l, n)| l * (1.0 - sigma) + n * sigma)
            .collect();
        let truth: Vec<f32> = noise.iter().zip(&packed).map(|(n, l)| n - l).collect();

        // CHAOS_FLIP_T=1 feeds the opposite convention, to settle by measurement
        // whether the timestep really does count down from noise.
        let t = if std::env::var("CHAOS_FLIP_T").is_ok() {
            sigma * 1000.0
        } else {
            flow::timestep_for(*sigma)
        };
        let pred = d
            .forward(&dit::Inputs {
                latent: &x,
                grid_w: gw as i64,
                grid_h: gh as i64,
                timestep: t,
                context: &[],
                context_len: 0,
            })
            .unwrap_or_else(|e| {
                eprintln!("forward: {e}");
                std::process::exit(1);
            });

        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
            dot / (na * nb)
        };
        let neg_latent: Vec<f32> = packed.iter().map(|l| -l).collect();
        // The denoiser's own estimate of the clean latent: `x - sigma * v`,
        // which is the reference's `c_skip * x + c_out * model_out`.
        let x0: Vec<f32> = x.iter().zip(&pred).map(|(xi, v)| xi - sigma * v).collect();
        let err: f64 = {
            let num: f64 = x0
                .iter()
                .zip(&packed)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            let den: f64 = packed.iter().map(|b| (*b as f64).powi(2)).sum();
            (num / den).sqrt()
        };
        println!(
            "{sigma:>7.2} {t:>9.1} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
            cos(&pred, &truth),
            cos(&pred, &neg_latent),
            cos(&pred, &noise),
            err
        );
    }

    println!("\ncos(v) near +1 is a working denoiser, near -1 an inverted sign.");
    println!("cos(-L) is whether it can see the image, cos(N) whether it can see");
    println!("the noise. x0 err under 1.0 means its guess at the clean latent");
    println!("beats guessing zero.");
}

fn read_ppm(path: &str) -> Result<(usize, usize, Vec<u8>), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if !bytes.starts_with(b"P6") {
        return Err("not a binary PPM".into());
    }
    let mut i = 2;
    let mut f = [0usize; 3];
    for field in &mut f {
        while bytes.get(i).is_some_and(|c| c.is_ascii_whitespace()) {
            i += 1;
        }
        let s = i;
        while bytes.get(i).is_some_and(|c| c.is_ascii_digit()) {
            i += 1;
        }
        *field = std::str::from_utf8(&bytes[s..i])
            .map_err(|e| e.to_string())?
            .parse()
            .map_err(|_| "bad PPM header".to_string())?;
    }
    i += 1;
    let want = f[0] * f[1] * 3;
    let data = bytes.get(i..i + want).ok_or("truncated PPM")?;
    Ok((f[0], f[1], data.to_vec()))
}
