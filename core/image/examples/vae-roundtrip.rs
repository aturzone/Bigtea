//! Encode a real photograph, decode it, and score the result.
//!
//! **This is the only evidence that the autoencoder port is right.** A decoder
//! alone can only be judged by looking at what it produces, and a subtly wrong
//! one produces a plausible picture — which is this project's oldest hazard. The
//! encoder and decoder are separately trained weights over a shared latent
//! space, so running both and measuring the difference catches a transposed
//! kernel, a missing group-norm scale, an attention contracted over the wrong
//! axis, or a symmetric downsample padding. None of those can cancel out.
//!
//! ```text
//! powershell -File scripts/image-to-ppm.ps1 -In photo.jpg -Out photo.ppm -Size 256
//! cargo run --release -p chaos-image --example vae-roundtrip -- photo.ppm
//! ```
//!
//! It writes `<input>-in.png` and `<input>-out.png` next to the input so the two
//! can be compared by eye as well as by number, and prints the PSNR — which is
//! the part that is actually evidence.
//!
//! An example rather than a test because ggml **aborts** on an exhausted arena,
//! and an abort takes the whole test binary with it rather than one case.

use chaos_image::{png, safetensors::SafeTensors, vae};

/// Bytes of arena per output pixel, per half.
///
/// Measured by raising it until the 256x256 round trip stopped aborting, then
/// left with room to spare. The deepest levels dominate: the decoder holds three
/// resnets of eight full-resolution 128-channel intermediates, which is most of
/// it, and the encoder's first block is the same shape.
const ARENA_PER_PIXEL: usize = 48 * 1024;

/// Room for the weights on top, plus ggml's own per-tensor overhead.
const ARENA_FIXED: usize = 512 << 20;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(input) = args.next() else {
        eprintln!("usage: vae-roundtrip <photo.ppm> [autoencoder.safetensors]");
        eprintln!("  make the .ppm with scripts/image-to-ppm.ps1");
        std::process::exit(2);
    };
    let model = args.next().unwrap_or_else(default_model);

    let (w, h, rgb) = match read_ppm(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{input}: {e}");
            std::process::exit(1);
        }
    };
    println!("input        {input} -- {w}x{h}");

    let file = match std::fs::read(&model) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {model}: {e}");
            std::process::exit(1);
        }
    };
    let st = match SafeTensors::parse(&file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{model}: {e}");
            std::process::exit(1);
        }
    };
    if !st.is_complete(&file) {
        eprintln!("{model} holds only its header -- the download did not finish");
        std::process::exit(1);
    }
    println!("autoencoder  {model} -- {} tensors", st.entries().len());

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let arena = ARENA_FIXED + w * h * ARENA_PER_PIXEL;
    println!(
        "arena        {} MiB per half, {threads} threads",
        arena >> 20
    );

    // -- encode ---------------------------------------------------------------
    // Two contexts rather than one. The latent between them is a few hundred
    // kilobytes, so carrying it across as a Vec halves the peak footprint and
    // lets each half be sized on its own.
    let started = std::time::Instant::now();
    let (lw, lh) = (w / vae::SCALE, h / vae::SCALE);
    let latent = {
        let ctx = chaos_ggml::Context::new(arena).expect("encoder arena");
        let v = vae::Vae::new(&st, &file, &ctx);

        let img = ctx
            .new_f32_4d(w as i64, h as i64, 3, 1)
            .expect("image tensor");
        img.set_f32(&vae::from_rgb8(&rgb, w, h)).expect("set image");

        let moments = match v.encode(&img) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("encode: {e}");
                std::process::exit(1);
            }
        };
        let mean = v.latent_mean(&moments).expect("latent mean");
        ctx.compute(&mean, threads).expect("compute encode");
        let ne = mean.ne();
        println!(
            "latent       {}x{}x{} in {:.1}s",
            ne[0],
            ne[1],
            ne[2],
            started.elapsed().as_secs_f32()
        );
        assert_eq!((ne[0], ne[1]), (lw as i64, lh as i64), "8x downsample");
        mean.to_vec_f32()
    };

    let finite = latent.iter().filter(|v| v.is_finite()).count();
    if finite != latent.len() {
        eprintln!("the latent has {} non-finite values", latent.len() - finite);
        std::process::exit(1);
    }
    let (lo, hi) = latent
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let mean = latent.iter().sum::<f32>() / latent.len() as f32;
    println!("             range {lo:.3} .. {hi:.3}, mean {mean:.4}");

    // -- decode ---------------------------------------------------------------
    let decoded = {
        let ctx = chaos_ggml::Context::new(arena).expect("decoder arena");
        let v = vae::Vae::new(&st, &file, &ctx);

        let z = ctx
            .new_f32_4d(lw as i64, lh as i64, vae::LATENT_CHANNELS, 1)
            .expect("latent tensor");
        z.set_f32(&latent).expect("set latent");

        let out = match v.decode(&z) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("decode: {e}");
                std::process::exit(1);
            }
        };
        ctx.compute(&out, threads).expect("compute decode");
        let ne = out.ne();
        assert_eq!(
            (ne[0], ne[1], ne[2]),
            (w as i64, h as i64, 3),
            "8x upsample"
        );
        out.to_vec_f32()
    };
    println!(
        "decoded      in {:.1}s total",
        started.elapsed().as_secs_f32()
    );

    let out_rgb = vae::to_rgb8(&decoded, w, h);

    // -- score ----------------------------------------------------------------
    let score = vae::psnr(&rgb, &out_rgb);
    let stem = input.trim_end_matches(".ppm").to_string();
    write_png(&format!("{stem}-in.png"), w, h, &rgb);
    write_png(&format!("{stem}-out.png"), w, h, &out_rgb);

    println!("\nPSNR         {score:.2} dB");
    // Measured on this file: 36.09, 36.29, 36.49 and 40.89 dB on four different
    // 256x256 photographs. The interesting number is not the exact one -- it is
    // that a broken port cannot reach it. Ablated against the same input, a
    // group norm without its scale gives 16.77 and a symmetric downsample
    // padding 14.60, because the decoder is then reading a latent the encoder
    // did not write.
    println!(
        "             {}",
        match score {
            s if s >= 30.0 => "a faithful reconstruction -- both halves agree",
            s if s >= 20.0 => "recognisable but degraded -- something is off",
            _ => "NOT a reconstruction -- the port is wrong somewhere",
        }
    );
}

fn default_model() -> String {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    format!("{home}/.chaos/models/flux2-vae.safetensors")
}

fn write_png(path: &str, w: usize, h: usize, rgb: &[u8]) {
    match png::encode_rgb(w as u32, h as u32, rgb) {
        Some(bytes) => match std::fs::write(path, &bytes) {
            Ok(()) => println!("wrote        {path} ({} KiB)", bytes.len() >> 10),
            Err(e) => eprintln!("cannot write {path}: {e}"),
        },
        None => eprintln!("cannot encode {path}"),
    }
}

/// Parse a binary PPM: `P6`, width, height, maxval, then RGB bytes.
///
/// Whitespace-separated and `#` comments are allowed anywhere in the header,
/// which is the whole format.
fn read_ppm(path: &str) -> Result<(usize, usize, Vec<u8>), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if !bytes.starts_with(b"P6") {
        return Err("not a binary PPM (no P6 magic)".into());
    }
    let mut i = 2;
    let mut fields = [0usize; 3];
    for field in &mut fields {
        // Skip whitespace and comment lines.
        loop {
            match bytes.get(i) {
                Some(c) if c.is_ascii_whitespace() => i += 1,
                Some(b'#') => {
                    while bytes.get(i).is_some_and(|c| *c != b'\n') {
                        i += 1;
                    }
                }
                _ => break,
            }
        }
        let start = i;
        while bytes.get(i).is_some_and(|c| c.is_ascii_digit()) {
            i += 1;
        }
        if start == i {
            return Err("truncated PPM header".into());
        }
        *field = std::str::from_utf8(&bytes[start..i])
            .map_err(|e| e.to_string())?
            .parse()
            .map_err(|_| "bad number in PPM header".to_string())?;
    }
    let [w, h, maxval] = fields;
    if maxval != 255 {
        return Err(format!(
            "maxval {maxval}, but only 8-bit PPMs are read here"
        ));
    }
    i += 1; // exactly one whitespace byte after maxval, then the data
    let want = w * h * 3;
    let data = bytes.get(i..i + want).ok_or_else(|| {
        format!(
            "header says {w}x{h} = {want} bytes, file has {}",
            bytes.len().saturating_sub(i)
        )
    })?;
    if w % vae::SCALE != 0 || h % vae::SCALE != 0 {
        return Err(format!(
            "{w}x{h} is not a multiple of {}; re-run image-to-ppm.ps1 with a -Size that is",
            vae::SCALE
        ));
    }
    Ok((w, h, data.to_vec()))
}
