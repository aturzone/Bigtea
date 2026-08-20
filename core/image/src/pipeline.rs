//! Prompt in, pixels out: the four models in the order they run.
//!
//! ```text
//! text  ->  Qwen3-VL-8B      13 layers of hidden states
//!           Ideogram 4       34 layers, once per step, conditioned
//!           Ideogram 4 uncond  34 layers, once per step, no text at all
//!           FLUX.2 decoder   latent -> RGB
//! ```
//!
//! # Why the unconditional pass is a second 5.26 GiB model
//!
//! Classifier-free guidance needs an answer to "what would you draw with no
//! prompt?" Most models get it by running the same weights on an empty string.
//! Ideogram 4 ships a **separately trained set of weights** for it, run with no
//! text tokens in the sequence at all — not an empty prompt, no prompt. That is
//! why [`Pipeline`] holds two denoisers and why a step costs two forward passes
//! over two different containers.
//!
//! # What one image costs
//!
//! Every byte of both denoisers is read on every step: a diffusion transformer
//! is dense, so there is no routed-expert set to skip. That is 10.5 GiB of
//! reads per step before any arithmetic, and the arithmetic is
//! `2 x tokens x 9e9` multiply-adds twice over. The token count is the lever
//! that matters — it is quadratic in the image's width — which is why
//! [`Request::grid`] exists and why a small image is not merely faster but
//! *differently* feasible.
//!
//! # Memory
//!
//! Each stage opens its own containers and closes them: the text encoder is
//! finished with before the first denoiser opens. Nothing here holds two models
//! resident at once, because 4.68 + 5.26 + 5.26 GiB does not fit on the machine
//! this was written on.

use crate::{dit, flow, text, vae};

/// What to draw, and how hard to work at it.
pub struct Request {
    /// The prompt. Ideogram 4 was trained on **structured JSON** descriptions;
    /// plain prose works and is what most callers will pass.
    pub prompt: String,
    /// The denoiser's grid in each direction. The image is `16 * grid` pixels:
    /// 8x for the autoencoder times the 2x2 patch. 64 is 1024x1024.
    pub grid: i64,
    pub steps: usize,
    /// Classifier-free guidance. 1.0 disables it and halves the work.
    pub cfg: f32,
    pub seed: u64,
    pub threads: usize,
}

impl Default for Request {
    fn default() -> Self {
        Request {
            prompt: String::new(),
            grid: 32,
            steps: 20,
            cfg: 4.0,
            seed: 42,
            threads: 4,
        }
    }
}

impl Request {
    pub fn image_size(&self) -> i64 {
        self.grid * vae::SCALE as i64 * dit::Config::default().patch
    }
    pub fn tokens(&self) -> usize {
        (self.grid * self.grid) as usize
    }
}

/// The largest decode plan this will attempt.
///
/// Not a hardware probe: it is a line past which the request is refused with an
/// explanation instead of aborting the process inside ggml.
///
/// **This used to be the binding constraint and is not any more.** Decoding was
/// one unplanned graph costing 51 KiB per output pixel — 29.5 GiB at 768x768,
/// which aborted after an hour of denoising. With the graph planned it is about
/// 3.3 KiB per pixel, so 1024x1024 asks for roughly 3.5 GiB instead of 52.
pub const DECODE_ARENA_LIMIT: usize = 10 << 30;

/// The largest `--grid` whose decode fits under [`DECODE_ARENA_LIMIT`].
pub fn largest_grid() -> i64 {
    let mut g = 1;
    while vae::decode_planned_bytes(((g + 1) * 16) as usize, ((g + 1) * 16) as usize)
        <= DECODE_ARENA_LIMIT
    {
        g += 1;
    }
    g
}

/// Bytes one denoiser layer needs for this request, and the image tokens it is
/// quadratic in.
///
/// **Both halves of the pipeline plan their graphs now**, so these are the sizes
/// of the *live set* rather than of every tensor ever written. The denoiser fell
/// from 14.6 GiB to a measured 2.0 GiB at 1024x1024, and the decoder from
/// 52 GiB to about 3.5 — which is the difference between 512x512 being the
/// ceiling and 1024x1024 being ordinary.
pub fn arena_estimate(req: &Request) -> (usize, usize) {
    let c = dit::Config::default();
    let tokens = req.tokens();
    let scores = c.num_heads as usize * tokens * tokens * 4;
    let planes = tokens * (3 * c.emb_dim + c.intermediate) as usize * 4;
    ((256 << 20) + scores * 3 / 2 + planes, tokens)
}

/// Where the four files live.
pub struct Paths {
    pub text_encoder: std::path::PathBuf,
    pub denoiser: std::path::PathBuf,
    pub uncond: std::path::PathBuf,
    pub autoencoder: std::path::PathBuf,
}

impl Paths {
    /// The names `chaos-pull` writes, under the models directory.
    pub fn under(dir: &std::path::Path) -> Self {
        Paths {
            text_encoder: dir.join("Qwen3-VL-8B-Instruct-Q4_K_M.gguf"),
            denoiser: dir.join("ideogram4-Q4_0.gguf"),
            uncond: dir.join("ideogram4_uncond-Q4_0.gguf"),
            autoencoder: dir.join("flux2-vae.safetensors"),
        }
    }

    /// Which of the four are absent, with the command that fetches each.
    pub fn missing(&self) -> Vec<(&'static str, String)> {
        [
            ("chaos-pull qwen3-vl-8b", &self.text_encoder),
            ("chaos-pull ideogram-4", &self.denoiser),
            ("chaos-pull ideogram-4-uncond", &self.uncond),
            ("chaos-pull flux2-vae", &self.autoencoder),
        ]
        .into_iter()
        .filter(|(_, p)| !p.exists())
        .map(|(cmd, p)| (cmd, p.display().to_string()))
        .collect()
    }
}

/// A deterministic normal-noise source.
///
/// Its own generator rather than a crate: the workspace has no dependencies,
/// and a seeded stream is wanted only so that two runs of the same request give
/// the same picture. It does **not** reproduce any other implementation's noise,
/// so the same seed here and in `stable-diffusion.cpp` are different images.
pub struct Noise(u64);

impl Noise {
    pub fn seeded(seed: u64) -> Self {
        // SplitMix64's constant, and a non-zero state so a seed of 0 works.
        Noise(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f32 {
        // In (0, 1]: exactly zero would make the logarithm below infinite.
        ((self.next_u64() >> 11) as f32 + 1.0) / (1u64 << 53) as f32
    }

    /// `n` standard normal values, by Box-Muller.
    pub fn normals(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let (u1, u2) = (self.uniform(), self.uniform());
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            out.push(r * theta.cos());
            if out.len() < n {
                out.push(r * theta.sin());
            }
        }
        out
    }
}

/// What happened, for a caller that wants to report it.
#[derive(Debug, Clone, Copy)]
pub enum Stage {
    Text { tokens: usize },
    Step { index: usize, total: usize },
    Decode,
}

#[derive(Debug)]
pub enum Error {
    Text(text::Error),
    Dit(dit::Error),
    Vae(vae::Error),
    Model(String),
    Missing(String),
    /// The image asked for cannot be decoded in the memory available.
    TooLarge(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Text(e) => write!(f, "text encoder: {e}"),
            Error::Dit(e) => write!(f, "denoiser: {e}"),
            Error::Vae(e) => write!(f, "autoencoder: {e}"),
            Error::Model(m) => write!(f, "{m}"),
            Error::Missing(m) => write!(f, "{m}"),
            Error::TooLarge(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<text::Error> for Error {
    fn from(e: text::Error) -> Self {
        Error::Text(e)
    }
}
impl From<dit::Error> for Error {
    fn from(e: dit::Error) -> Self {
        Error::Dit(e)
    }
}
impl From<vae::Error> for Error {
    fn from(e: vae::Error) -> Self {
        Error::Vae(e)
    }
}

/// The finished image, and the latent it came from.
pub struct Image {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// Run the whole thing.
///
/// `on` is called as each stage begins, because a run takes minutes and silence
/// is indistinguishable from a hang.
pub fn generate(paths: &Paths, req: &Request, on: &mut dyn FnMut(Stage)) -> Result<Image, Error> {
    // **Refused first, before anything is loaded.** The decoder runs last and is
    // the memory ceiling (see `vae::decode_arena_bytes`). A 768x768 request once
    // spent sixteen denoising steps and then died inside ggml, taking the
    // process with it -- an hour to learn the answer would not fit.
    let side = req.image_size() as usize;
    let need = vae::decode_planned_bytes(side, side);
    if need > DECODE_ARENA_LIMIT {
        let g = largest_grid();
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        let mut m = format!(
            "decoding {side}x{side} needs about {:.1} GiB in one graph, over the {:.1} GiB cap.\n",
            gib(need),
            gib(DECODE_ARENA_LIMIT)
        );
        m.push_str("The denoisers are not the limit -- they stream a layer at a time.\n");
        m.push_str("The autoencoder is: it decodes in one pass, with no memory reuse.\n");
        m.push_str(&format!(
            "Ask for --grid {g} or less ({}x{}).",
            g * 16,
            g * 16
        ));
        return Err(Error::TooLarge(m));
    }

    let missing = paths.missing();
    if !missing.is_empty() {
        let mut m = String::from("these files are not here yet:\n");
        for (cmd, p) in &missing {
            m.push_str(&format!("  {p}\n    fetch it with: {cmd}\n"));
        }
        return Err(Error::Missing(m));
    }

    // -- 1. the prompt ------------------------------------------------------
    // In its own scope so 4.68 GiB of text encoder is closed before 5.26 GiB of
    // denoiser opens.
    let context = {
        let model = chaos_model::Model::open_split(&paths.text_encoder)
            .map_err(|e| Error::Model(format!("{}: {e}", paths.text_encoder.display())))?;
        let tok = chaos_tokenizer::Tokenizer::from_metadata(model.metadata())
            .map_err(|e| Error::Model(format!("tokenizer: {e:?}")))?;
        let ids = tok.encode(&text::wrap_prompt(&req.prompt));
        let enc = text::TextEncoder::open(model, req.threads)?;
        on(Stage::Text { tokens: ids.len() });
        enc.encode(&ids, &mut |_, _| {})?
    };

    // -- 2. the noise the image is carved out of ----------------------------
    //
    // **The denoiser works in a normalised latent space**, and the autoencoder
    // file says which one: `bn.running_mean` and `bn.running_var`, 128-wide to
    // match the packed channel count. Sampling starts from unit noise, which is
    // already in that space, so nothing is normalised here -- but the result has
    // to be brought back out of it before the autoencoder sees it.
    //
    // Skipping this is not subtle once it is measured: against a real latent the
    // denoiser's velocity scored cos 0.17 at sigma 0.3 without it and 0.49 with
    // it, because unit-variance *noise* still looks like noise whatever the
    // scale, while the image content does not.
    let stats = {
        let f = std::fs::read(&paths.autoencoder)
            .map_err(|e| Error::Model(format!("{}: {e}", paths.autoencoder.display())))?;
        crate::safetensors::SafeTensors::parse(&f)
            .ok()
            .and_then(|st| vae::latent_stats(&st, &f))
    };
    let c = dit::Config::default();
    let n = req.tokens() * c.in_channels as usize;
    let mut x = Noise::seeded(req.seed).normals(n);

    let sigmas = flow::Schedule::default()
        .resolution_aware(req.tokens())
        .sigmas(req.steps);
    // The first sigma is not quite 1, and the latent starts scaled by it:
    // `x = image * (1 - sigma) + noise * sigma` with no image yet.
    for v in x.iter_mut() {
        *v *= sigmas[0];
    }

    // -- 3. denoise ----------------------------------------------------------
    {
        let cond = dit::Denoiser::open(
            chaos_model::Model::open_split(&paths.denoiser)
                .map_err(|e| Error::Model(format!("{}: {e}", paths.denoiser.display())))?,
            req.threads,
        );
        let uncond = if req.cfg == 1.0 {
            None
        } else {
            Some(dit::Denoiser::open(
                chaos_model::Model::open_split(&paths.uncond)
                    .map_err(|e| Error::Model(format!("{}: {e}", paths.uncond.display())))?,
                req.threads,
            ))
        };

        for i in 0..req.steps {
            on(Stage::Step {
                index: i,
                total: req.steps,
            });
            let t = flow::timestep_for(sigmas[i]);
            let v_cond = cond.forward(&dit::Inputs {
                latent: &x,
                grid_w: req.grid,
                grid_h: req.grid,
                timestep: t,
                context: &context.hidden,
                context_len: context.tokens,
            })?;
            let v = match &uncond {
                None => v_cond,
                Some(u) => {
                    // **No text at all**, not an empty prompt: this model was
                    // trained for a sequence of image tokens only.
                    let v_uncond = u.forward(&dit::Inputs {
                        latent: &x,
                        grid_w: req.grid,
                        grid_h: req.grid,
                        timestep: t,
                        context: &[],
                        context_len: 0,
                    })?;
                    flow::guide(&v_cond, &v_uncond, req.cfg)
                }
            };
            flow::euler_step(&mut x, &v, sigmas[i], sigmas[i + 1]);
        }
    }

    // -- 4. pixels -----------------------------------------------------------
    on(Stage::Decode);
    // Out of the diffusion model's normalised space and back into the
    // autoencoder's. See `stats` below for why this is not optional.
    if let Some((m, v)) = stats.as_ref() {
        vae::denormalize_latent(&mut x, m, v);
    }
    let latent = vae::unpack_latent(
        &x,
        req.grid as usize,
        req.grid as usize,
        c.ae_channels as usize,
        c.patch as usize,
    );
    let lw = req.grid * c.patch;
    let pixels = decode_latent(&paths.autoencoder, &latent, lw, lw, req.threads)?;
    let side = (lw as usize) * vae::SCALE;
    Ok(Image {
        rgb: vae::to_rgb8(&pixels, side, side),
        width: side,
        height: side,
    })
}

/// The autoencoder's decoder over one latent.
///
/// Split out because it is the one stage that is already verified on its own:
/// see the round-trip tests, which score 36 dB on a photograph.
pub fn decode_latent(
    path: &std::path::Path,
    latent: &[f32],
    w: i64,
    h: i64,
    threads: usize,
) -> Result<Vec<f32>, Error> {
    let file = std::fs::read(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    let st = crate::safetensors::SafeTensors::parse(&file)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    let (pixels, _bytes) = vae::decode_planned(&st, &file, latent, w, h, threads)?;
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three scale factors have to compose: 8x autoencoder, 2x patch.
    #[test]
    fn the_grid_and_the_image_size_agree() {
        let r = Request {
            grid: 64,
            ..Default::default()
        };
        assert_eq!(r.image_size(), 1024, "64 grid is the reference resolution");
        assert_eq!(r.tokens(), 4096);
        let small = Request {
            grid: 32,
            ..Default::default()
        };
        assert_eq!(small.image_size(), 512);
        assert_eq!(small.tokens(), 1024, "the schedule's reference token count");
    }

    /// The noise is normal, and the same seed gives the same noise.
    #[test]
    fn the_noise_is_normal_and_reproducible() {
        let v = Noise::seeded(7).normals(20_000);
        assert_eq!(v.len(), 20_000);
        assert!(v.iter().all(|x| x.is_finite()), "Box-Muller logged a zero");
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
        // An odd count must not run off the end of a Box-Muller pair.
        assert_eq!(Noise::seeded(1).normals(3).len(), 3);
        // Same seed, same picture.
        assert_eq!(Noise::seeded(9).normals(64), Noise::seeded(9).normals(64));
        assert_ne!(Noise::seeded(9).normals(64), Noise::seeded(10).normals(64));
    }

    /// The decode ceiling is real, refuses what will not fit, and points at a
    /// size that does.
    ///
    /// This guards an hour of wasted compute: a 768x768 request once ran all
    /// sixteen denoising steps and then died inside ggml, because the decoder
    /// was one graph with no memory reuse. **Planning the graph moved the
    /// ceiling rather than removing it** — the numbers below are the planned
    /// ones, and the cap still has to be enforced somewhere.
    #[test]
    fn the_decode_ceiling_refuses_what_it_cannot_finish() {
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        // Planning is worth about 15x: measured 0.20 GiB at 256x256 where the
        // unplanned graph wanted 3.69.
        assert!(gib(vae::decode_planned_bytes(256, 256)) < 0.5);
        assert!(gib(vae::decode_arena_bytes(256, 256)) > 3.0);
        assert!(vae::decode_planned_bytes(256, 256) * 10 < vae::decode_arena_bytes(256, 256));
        // It grows with pixels, so doubling the side is four times the memory.
        let a = vae::decode_planned_bytes(256, 256) - (64 << 20);
        let b = vae::decode_planned_bytes(512, 512) - (64 << 20);
        assert_eq!(b, a * 4);

        // The largest grid accepted decodes under the cap, and one step larger
        // does not -- so the number the user is told is the real boundary.
        let g = largest_grid();
        let side = (g * 16) as usize;
        assert!(vae::decode_planned_bytes(side, side) <= DECODE_ARENA_LIMIT);
        let over = ((g + 1) * 16) as usize;
        assert!(vae::decode_planned_bytes(over, over) > DECODE_ARENA_LIMIT);
        // 1024x1024 is what planning bought, and it must stay bought.
        assert!(g >= 64, "grid {g} would refuse 1024x1024, which now fits");
    }

    /// Every missing file is reported with the command that fetches it.
    #[test]
    fn missing_files_name_the_command_that_gets_them() {
        let p = Paths::under(std::path::Path::new("/nowhere-at-all"));
        let m = p.missing();
        assert_eq!(m.len(), 4, "none of them exist under that path");
        assert!(m.iter().any(|(c, _)| c.contains("qwen3-vl-8b")));
        assert!(m.iter().any(|(c, _)| c.contains("ideogram-4-uncond")));
        assert!(m.iter().any(|(c, _)| c.contains("flux2-vae")));
        // The uncond twin and the conditional model are different files, and
        // asking for the wrong one is a plausible slip.
        assert_ne!(p.denoiser, p.uncond);
    }
}
