//! The autoencoder decodes a latent into the picture the encoder saw.
//!
//! # Why these are `#[ignore]` and why they still fail rather than skip
//!
//! They need `flux2-vae.safetensors`, which is 336 MB and not in the repository,
//! so they cannot run on a machine that has not fetched it — hence `#[ignore]`,
//! which keeps them out of the default count instead of quietly passing.
//!
//! **When they do run and the file is missing, they panic.** A test that skips
//! itself reports success for work it did not do; this project has been burned
//! by exactly that, with two GPU tests that skipped rather than failed and were
//! reported as a green run for months.
//!
//! ```text
//! cargo test --release -p chaos-image -- --ignored
//! ```
//!
//! # Why a round trip is the check, and not "does it look like a picture"
//!
//! A decoder can only be judged by what it produces, and **a subtly wrong one
//! produces a plausible picture** — the oldest hazard in this project. The
//! encoder and decoder are separately trained weights over one shared latent
//! space, so running both and measuring the difference catches errors that
//! looking cannot. Measured on this file, against the same 128x128 input:
//!
//! | version                                  | PSNR     |
//! |------------------------------------------|----------|
//! | correct                                   | 37.59 dB |
//! | group norm without its scale and shift    | 25.89 dB |
//! | downsample padded symmetrically           | 17.32 dB |
//! | mid-block attention skipped               | 33.89 dB |
//! | convolution kernels not dimension-reversed| ggml aborts |
//!
//! [`MIN_PSNR`] sits below the correct number and above every one of those, so
//! the assertion is a real discriminator rather than a shape check. The
//! attention row is the narrow one — 3.7 dB — and is the reason the threshold is
//! 35 and not 25.

use chaos_image::{safetensors::SafeTensors, vae};

/// Where `chaos-pull` puts it.
fn model_path() -> std::path::PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .expect("neither USERPROFILE nor HOME is set");
    std::path::Path::new(&home)
        .join(".chaos")
        .join("models")
        .join("flux2-vae.safetensors")
}

fn read_model() -> Vec<u8> {
    let path = model_path();
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}\n\
             This test is #[ignore]d because it needs the autoencoder. If you asked \
             for it by name, fetch it first: chaos-pull flux2-vae",
            path.display()
        )
    })
}

/// The edge of the test image. 128 gives a 16x16 latent and a ~1.3 GiB arena per
/// half; 1024 would be the real thing and would not fit beside a test suite.
const SIDE: usize = 128;

/// The score a correct autoencoder clears and every ablation above does not.
const MIN_PSNR: f32 = 35.0;

/// Bytes of arena per pixel, per half — see the example, which is where this was
/// measured. ggml **aborts** on an exhausted arena and takes the whole test
/// binary with it, so this is deliberately generous.
const ARENA_PER_PIXEL: usize = 48 * 1024;
const ARENA_FIXED: usize = 512 << 20;

/// A picture with the structure that makes a round trip informative.
///
/// Not a photograph, because the only photographs to hand are other people's.
/// It carries what the ablations need in order to show up: smooth colour so a
/// missing group-norm scale shifts it, hard edges so a half-pixel padding error
/// blurs them, and a soft lobe so the whole thing is not piecewise constant.
/// Measured at 37.59 dB against 36.09 for a real photograph on this file, so it
/// is not an easier input than the real thing.
fn test_image(n: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity(n * n * 3);
    for y in 0..n {
        for x in 0..n {
            let (u, v) = (x as f32 / n as f32, y as f32 / n as f32);
            let r = ((u - 0.35).powi(2) + (v - 0.4).powi(2)).sqrt();
            let lobe = (-(r * 3.2).powi(2)).exp();
            let mut rgb = [
                0.55 + 0.35 * lobe - 0.20 * v,
                0.30 + 0.45 * u * (1.0 - v) + 0.25 * lobe,
                0.70 - 0.40 * u + 0.30 * (6.0 * (u + v)).sin(),
            ];
            if (0.60..0.88).contains(&u) && (0.15..0.42).contains(&v) {
                rgb = [0.92, 0.88, 0.30];
            }
            if (0.12..0.34).contains(&u) && (0.66..0.90).contains(&v) {
                rgb = [0.10, 0.22, 0.55];
            }
            for c in rgb {
                px.push((c.clamp(0.0, 1.0) * 255.0) as u8);
            }
        }
    }
    px
}

/// Encode a picture, decode the latent, and require the two to agree.
#[test]
#[ignore = "needs flux2-vae.safetensors (336 MB); run with --ignored"]
fn the_autoencoder_round_trips_a_picture() {
    let file = read_model();
    let st = SafeTensors::parse(&file).expect("parse the autoencoder");
    assert!(
        st.is_complete(&file),
        "the file holds only its header -- the download did not finish"
    );

    let rgb = test_image(SIDE);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let arena = ARENA_FIXED + SIDE * SIDE * ARENA_PER_PIXEL;
    let side = SIDE as i64;
    let lside = (SIDE / vae::SCALE) as i64;

    // Two contexts, not one: the latent between them is small, so carrying it
    // across as a Vec halves the peak footprint.
    let latent = {
        let ctx = chaos_ggml::Context::new(arena).expect("encoder arena");
        let v = vae::Vae::new(&st, &file, &ctx);
        let img = ctx.new_f32_4d(side, side, 3, 1).expect("image tensor");
        img.set_f32(&vae::from_rgb8(&rgb, SIDE, SIDE))
            .expect("set image");

        let moments = v.encode(&img).expect("encode");
        assert_eq!(
            moments.ne(),
            [lside, lside, 2 * vae::LATENT_CHANNELS, 1],
            "three stride-2 convolutions is 8x, and the encoder emits mean and log-variance"
        );
        let mean = v.latent_mean(&moments).expect("latent mean");
        assert_eq!(mean.ne(), [lside, lside, vae::LATENT_CHANNELS, 1]);
        ctx.compute(&mean, threads).expect("compute encode");
        mean.to_vec_f32()
    };

    // A NaN here would make the reconstruction fail for a reason that has
    // nothing to do with the decoder, so it is worth separating.
    assert!(
        latent.iter().all(|v| v.is_finite()),
        "the latent has non-finite values"
    );
    // A latent that is all one value would round-trip to a flat image and could
    // still clear a loose threshold, so check the encoder said something.
    let (lo, hi) = latent
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    assert!(hi - lo > 1.0, "the latent is nearly constant: {lo} .. {hi}");

    let decoded = {
        let ctx = chaos_ggml::Context::new(arena).expect("decoder arena");
        let v = vae::Vae::new(&st, &file, &ctx);
        let z = ctx
            .new_f32_4d(lside, lside, vae::LATENT_CHANNELS, 1)
            .expect("latent tensor");
        z.set_f32(&latent).expect("set latent");

        let out = v.decode(&z).expect("decode");
        assert_eq!(out.ne(), [side, side, 3, 1], "three upsamplers is 8x back");
        ctx.compute(&out, threads).expect("compute decode");
        out.to_vec_f32()
    };
    assert!(
        decoded.iter().all(|v| v.is_finite()),
        "the decoded image has non-finite values"
    );

    let score = vae::psnr(&rgb, &vae::to_rgb8(&decoded, SIDE, SIDE));
    assert!(
        score >= MIN_PSNR,
        "round-trip PSNR {score:.2} dB, below {MIN_PSNR}. \
         The module docs list what each kind of error scores; \
         a symmetric downsample padding gives about 17 and a group norm \
         missing its scale about 26."
    );
}

/// Every tensor the two halves name is present, F32, and the shape the graph
/// assumes.
///
/// Cheap next to the round trip, and it fails with the missing name rather than
/// with a bad picture — which is the difference between a minute and an hour
/// when a download was truncated.
#[test]
#[ignore = "needs flux2-vae.safetensors (336 MB); run with --ignored"]
fn the_file_holds_every_tensor_the_two_halves_name() {
    let file = read_model();
    let st = SafeTensors::parse(&file).expect("parse the autoencoder");

    let mut named = 0;
    for name in vae::decoder_tensors()
        .into_iter()
        .chain(vae::encoder_tensors())
    {
        let e = st
            .get(&name)
            .unwrap_or_else(|| panic!("the autoencoder has no tensor {name:?}"));
        assert_eq!(
            e.dtype,
            chaos_image::safetensors::Dtype::F32,
            "{name} is {:?}; conv_2d_direct is used precisely because the file is F32",
            e.dtype
        );
        assert!(
            st.bytes_of(&file, e).is_some(),
            "{name} has no bytes -- a partial download"
        );
        named += 1;
    }

    // 138 decoder + 106 encoder + the two 1x1 quant convolutions.
    assert_eq!(named, 248, "tensors named by the two halves");
    assert_eq!(st.entries().len(), 251, "tensors in the file");

    // **The three left over are not slack, and they matter later.** FLUX.2 keeps
    // a BatchNorm's running statistics beside the autoencoder, and normalising
    // the latent with them is what earlier VAEs did with a scalar
    // `scaling_factor`. They are 128-wide, not 32: that is the *patchified*
    // channel count, 32 latent channels times a 2x2 patch, which is exactly what
    // the denoiser consumes.
    //
    // A round trip never touches them -- encode and decode are inverses whatever
    // the normalisation is, which is what makes the round trip a fair test of
    // this port. The denoiser is where leaving them out would produce a
    // confident, plausible, wrong image, so they are asserted here to keep them
    // from being forgotten.
    for (name, elements) in [("bn.running_mean", 128), ("bn.running_var", 128)] {
        let e = st
            .get(name)
            .unwrap_or_else(|| panic!("no {name}: the latent normalisation is missing"));
        assert_eq!(e.shape, vec![elements], "{name} is per patch channel");
    }
    let tracked = st
        .get("bn.num_batches_tracked")
        .expect("no bn.num_batches_tracked");
    assert!(
        tracked.shape.is_empty() && tracked.elements() == 1,
        "a safetensors scalar has an empty shape and one element, not zero"
    );
}
