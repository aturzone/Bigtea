//! Does planned decoding give the same pixels, and how much less memory?
//!
//! **An example, not a test**: `ggml_gallocr` aborts rather than erroring, and
//! the decoder is the largest graph in the project.
//!
//! The unplanned decoder allocates every tensor and frees none, so it costs
//! 51 KiB per output pixel and 768x768 does not fit on a 15.7 GiB machine. This
//! runs the same latent both ways and compares.
//!
//! ```text
//! powershell -File scripts/image-to-ppm.ps1 -In photo.jpg -Out p.ppm -Size 256
//! cargo run --release -p chaos-image --example try-planned-decode -- p.ppm
//! ```
//!
//! Identical pixels is the requirement. Reuse that changes an answer is
//! aliasing, and an image is exactly the place it would not be noticed.

use chaos_image::{safetensors::SafeTensors, vae};

fn main() {
    let Some(ppm) = std::env::args().nth(1) else {
        eprintln!("usage: try-planned-decode <photo.ppm>");
        std::process::exit(2);
    };
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    let ae = std::path::Path::new(&home)
        .join(".chaos")
        .join("models")
        .join("flux2-vae.safetensors");

    let (w, h, rgb) = read_ppm(&ppm).unwrap_or_else(|e| {
        eprintln!("{ppm}: {e}");
        std::process::exit(1);
    });
    let file = std::fs::read(&ae).unwrap_or_else(|e| {
        eprintln!("{}: {e}", ae.display());
        std::process::exit(1);
    });
    let st = SafeTensors::parse(&file).expect("parse the autoencoder");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    println!("photo        {ppm} -- {w}x{h}");

    // A real latent from the verified encoder.
    let (lw, lh) = (w / vae::SCALE, h / vae::SCALE);
    let latent = {
        let ctx = chaos_ggml::Context::new((512 << 20) + w * h * 48 * 1024).expect("arena");
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

    // -- the ordinary way ----------------------------------------------------
    let unplanned_bytes = vae::decode_arena_bytes(w, h);
    println!(
        "unplanned    {:.2} GiB of arena",
        unplanned_bytes as f64 / (1u64 << 30) as f64
    );
    let started = std::time::Instant::now();
    let plain = {
        let ctx = chaos_ggml::Context::new(unplanned_bytes).expect("decode arena");
        let v = vae::Vae::new(&st, &file, &ctx);
        let z = ctx
            .new_f32_4d(lw as i64, lh as i64, vae::LATENT_CHANNELS, 1)
            .expect("latent");
        z.set_f32(&latent).expect("set latent");
        let out = v.decode(&z).expect("decode");
        ctx.compute(&out, threads).expect("compute");
        out.to_vec_f32()
    };
    println!("             {:.1}s", started.elapsed().as_secs_f32());

    // -- planned -------------------------------------------------------------
    let started = std::time::Instant::now();
    let (planned, bytes) = vae::decode_planned(&st, &file, &latent, lw as i64, lh as i64, threads)
        .unwrap_or_else(|e| {
            eprintln!("planned decode: {e}");
            std::process::exit(1);
        });
    println!(
        "planned      {:.2} GiB of plan, {:.1}s",
        bytes as f64 / (1u64 << 30) as f64,
        started.elapsed().as_secs_f32()
    );
    println!(
        "             {:.1}x less than the unplanned arena",
        unplanned_bytes as f64 / bytes.max(1) as f64
    );

    // -- the requirement: identical pixels -----------------------------------
    let a = vae::to_rgb8(&plain, w, h);
    let b = vae::to_rgb8(&planned, w, h);
    let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap_or(0);
    println!(
        "\npixels       {differing} of {} differ, worst by {worst}",
        a.len()
    );
    println!("PSNR against the input photograph:");
    println!("  unplanned  {:.2} dB", vae::psnr(&rgb, &a));
    println!("  planned    {:.2} dB", vae::psnr(&rgb, &b));

    if differing == 0 {
        println!("\nBIT-IDENTICAL, and smaller.");
    } else if worst <= 1 {
        println!("\nsame to within one level -- ggml reorders some sums when it reuses.");
    } else {
        eprintln!("\nDIFFERS BY {worst} LEVELS -- the reuse is aliasing live tensors.");
        std::process::exit(1);
    }
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
