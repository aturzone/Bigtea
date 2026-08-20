//! The sampler: a noise schedule, an Euler step, and classifier-free guidance.
//!
//! # What a diffusion sampler actually is
//!
//! The denoiser is a function of one noisy latent and one timestep. The sampler
//! decides *which* timesteps to visit and what to do with each answer. Ideogram
//! 4 is a **rectified flow** model, which makes both parts unusually simple:
//! the path from noise to image is a straight line, so the model predicts a
//! velocity along it and each step is `x += v * dt`. There is no variance
//! schedule and no predictor-corrector.
//!
//! # The three numbers that are not obvious
//!
//! **The schedule is logit-normal, not linear.** Ideogram's own scheduler draws
//! `sigma = sigmoid(mean + std * ndtri(1 - i/n))`, which spends most of its
//! steps where the image is decided and few at the ends. A linear schedule runs,
//! produces a picture, and wastes half its steps.
//!
//! **The mean moves with the image size.** `mean += 0.5 * ln(tokens / 1024)`,
//! where 1024 is the token count of the 512x512 image the default was tuned at.
//! Leaving it fixed makes large images soft and small ones over-sharpened —
//! which looks like a model quality problem rather than an arithmetic one.
//!
//! **The model's timestep input is `1000 * (1 - sigma)`, not `1000 * sigma`.**
//! It runs backwards, and the denoiser's output is negated on the way out. The
//! two conventions cancel; implementing one without the other produces an image
//! that is exactly as wrong as pure noise, so at least that failure is loud.
//!
//! Everything here is `f32`-in, `f32`-out and has no dependency on ggml or on a
//! model, which is why it is unit-tested against hand-computed values rather
//! than against a picture.

/// Latent tokens in the 512x512 image the schedule's default mean was tuned on:
/// `(512 * 512) / (16 * 16)`. A 16x16 pixel patch is 8x for the autoencoder
/// times a 2x2 patch in the denoiser.
const KNOWN_SEQ_LEN: f32 = 1024.0;

/// How the noise level is spread over the steps.
///
/// The fields are Ideogram's own defaults, read from the reference
/// implementation rather than tuned here.
#[derive(Debug, Clone, Copy)]
pub struct Schedule {
    /// Centre of the logit-normal draw, before the resolution correction.
    pub mean: f32,
    /// Spread of the draw. Larger keeps more steps near the middle.
    pub std: f32,
    /// Clamps on the log signal-to-noise ratio, which bound sigma away from
    /// exactly 0 and exactly 1 — both of which are degenerate.
    pub logsnr_min: f32,
    pub logsnr_max: f32,
}

impl Default for Schedule {
    fn default() -> Self {
        Schedule {
            mean: 0.0,
            std: 1.75,
            logsnr_min: -15.0,
            logsnr_max: 18.0,
        }
    }
}

impl Schedule {
    /// Shift the centre for an image of `tokens` latent tokens.
    ///
    /// **Not cosmetic.** At 1024x1024 this is `+0.69`, which moves every sigma
    /// in the schedule.
    pub fn resolution_aware(mut self, tokens: usize) -> Self {
        if tokens > 0 {
            self.mean += 0.5 * (tokens as f32 / KNOWN_SEQ_LEN).ln();
        }
        self
    }

    /// The `n + 1` noise levels to visit, descending, ending at exactly zero.
    ///
    /// `sigma[0]` is the starting noise level and `sigma[n]` is 0, so a step
    /// `i` moves from `sigma[i]` to `sigma[i + 1]`.
    pub fn sigmas(&self, n: usize) -> Vec<f32> {
        let one_minus_t_min = sigmoid(0.5 * self.logsnr_max);
        let one_minus_t_max = sigmoid(0.5 * self.logsnr_min);
        let mut out = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let t = i as f32 / n as f32;
            // ndtri(1 - t) == -ndtri(t), and the reference uses the identity.
            let z = -ndtri(t as f64) as f32;
            let sigma = sigmoid(self.mean + self.std * z);
            out.push(sigma.clamp(one_minus_t_max, one_minus_t_min));
        }
        // The last one is exactly zero however the clamp landed: the final step
        // must arrive at a noise-free latent, not merely a quiet one.
        out[n] = 0.0;
        out
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The timestep the denoiser is given for a noise level.
///
/// **Backwards on purpose**: this model counts from clean to noisy, so a sigma
/// of 1 (pure noise) is timestep 0. See the module docs.
pub fn timestep_for(sigma: f32) -> f32 {
    1000.0 - sigma * 1000.0
}

/// One Euler step along the straight path from noise to image.
///
/// `velocity` is the denoiser's output for `x` at `sigma`, already through
/// [`guide`] if guidance is in use. The step is `x + v * (sigma_next - sigma)`,
/// and `sigma_next < sigma`, so the move is *against* the velocity.
pub fn euler_step(x: &mut [f32], velocity: &[f32], sigma: f32, sigma_next: f32) {
    let dt = sigma_next - sigma;
    for (xi, v) in x.iter_mut().zip(velocity) {
        *xi += v * dt;
    }
}

/// Classifier-free guidance: push away from what the model would draw with no
/// prompt at all.
///
/// `scale` of 1 is the conditional answer unchanged; larger follows the prompt
/// harder at the cost of variety, and far too large saturates into garish
/// colour. Ideogram 4 carries a **separate set of weights** for the
/// unconditional pass rather than an empty prompt, which is why `uncond` comes
/// from a different model rather than a second run of this one.
pub fn guide(cond: &[f32], uncond: &[f32], scale: f32) -> Vec<f32> {
    cond.iter()
        .zip(uncond)
        .map(|(c, u)| u + scale * (c - u))
        .collect()
}

/// Inverse of the standard normal CDF — Acklam's rational approximation.
///
/// The schedule needs it to turn evenly spaced step numbers into normally
/// distributed noise levels. Written out rather than approximated with
/// something simpler because the schedule's whole point is *where* it puts its
/// steps, and a sloppy quantile function moves them.
///
/// Accurate to about 1.15e-9 in the central region, which is far beyond what a
/// 20-step schedule can notice.
// The coefficients are Acklam's, copied digit for digit. Clippy would round
// two of them to the nearest f64 -- which is the same value -- but a constant
// that no longer matches the paper it came from cannot be checked against it.
#[allow(clippy::excessive_precision)]
pub fn ndtri(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 5] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
        1.0,
    ];
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 6] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
        1.0,
    ];

    // The tails and the centre need different expansions; one rational fit over
    // the whole range is not accurate at either end.
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        poly(&C, q) / poly(&D, q)
    } else if p > P_HIGH {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(poly(&C, q) / poly(&D, q))
    } else {
        let q = p - 0.5;
        let r = q * q;
        q * poly(&A, r) / poly(&B, r)
    }
}

/// Horner's method over a coefficient list, highest power first.
fn poly(c: &[f64], x: f64) -> f64 {
    let mut acc = c[0];
    for k in &c[1..] {
        acc = acc * x + k;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ndtri` is the inverse normal CDF, checked at points with known answers.
    #[test]
    fn the_quantile_function_inverts_the_normal_cdf() {
        assert!(ndtri(0.5).abs() < 1e-12, "the median of a normal is 0");
        // The textbook quantiles.
        assert!(
            (ndtri(0.975) - 1.959963985).abs() < 1e-6,
            "{}",
            ndtri(0.975)
        );
        assert!((ndtri(0.025) + 1.959963985).abs() < 1e-6);
        assert!((ndtri(0.8413447461) - 1.0).abs() < 1e-5, "one sigma");
        // Both tails, past the switch to the tail expansion at 0.02425.
        assert!(
            (ndtri(0.001) + 3.090232306).abs() < 1e-5,
            "{}",
            ndtri(0.001)
        );
        assert!((ndtri(0.999) - 3.090232306).abs() < 1e-5);
        // Symmetry is what the schedule relies on to use -ndtri(t) for 1 - t.
        for p in [0.01, 0.1, 0.3, 0.45] {
            assert!(
                (ndtri(p) + ndtri(1.0 - p)).abs() < 1e-9,
                "asymmetric at {p}"
            );
        }
        assert!(ndtri(0.0).is_infinite() && ndtri(0.0) < 0.0);
        assert!(ndtri(1.0).is_infinite() && ndtri(1.0) > 0.0);
    }

    /// The schedule descends from nearly pure noise to exactly zero.
    #[test]
    fn the_schedule_descends_to_zero() {
        let s = Schedule::default();
        let sigmas = s.sigmas(20);
        assert_eq!(sigmas.len(), 21, "n steps is n + 1 boundaries");
        assert_eq!(sigmas[20], 0.0, "the last step must land on a clean latent");
        for w in sigmas.windows(2) {
            assert!(w[0] > w[1], "not descending: {w:?}");
        }
        // sigma[0] is clamped by logsnr_max, not left at 1.0: an exactly-1
        // sigma makes the first Euler step divide the whole image by zero.
        assert!(sigmas[0] < 1.0 && sigmas[0] > 0.99, "{}", sigmas[0]);
        // The middle of a 20-step run sits near the centre of the draw, which
        // is what "most of the steps where the image is decided" means.
        assert!(sigmas[10] > 0.3 && sigmas[10] < 0.7, "{}", sigmas[10]);
    }

    /// The resolution correction moves the schedule, and by the documented
    /// amount.
    #[test]
    fn the_mean_shifts_with_the_token_count() {
        // 1024 tokens is the reference resolution, so nothing moves.
        let base = Schedule::default().resolution_aware(1024);
        assert!((base.mean - 0.0).abs() < 1e-6);
        // A 1024x1024 image is a 128x128 latent in 2x2 patches: 64 * 64 = 4096
        // tokens, four times the reference, so +0.5 * ln(4).
        let big = Schedule::default().resolution_aware(4096);
        assert!((big.mean - 0.5 * 4.0f32.ln()).abs() < 1e-6, "{}", big.mean);
        assert!(
            (big.mean - std::f32::consts::LN_2).abs() < 1e-5,
            "{}",
            big.mean
        );
        // And a bigger mean means more noise left at every step.
        let (a, b) = (base.sigmas(20), big.sigmas(20));
        for i in 1..20 {
            assert!(b[i] > a[i], "step {i}: {} should exceed {}", b[i], a[i]);
        }
        // Zero tokens is "do not correct", not a log of zero.
        assert!(Schedule::default().resolution_aware(0).mean.is_finite());
    }

    /// The timestep runs backwards, which is the convention the negated model
    /// output pairs with.
    #[test]
    fn the_timestep_counts_down_from_noise() {
        assert!(
            (timestep_for(1.0) - 0.0).abs() < 1e-3,
            "pure noise is t = 0"
        );
        assert!(
            (timestep_for(0.0) - 1000.0).abs() < 1e-3,
            "clean is t = 1000"
        );
        assert!((timestep_for(0.25) - 750.0).abs() < 1e-3);
    }

    /// An Euler step moves against the velocity, because sigma descends.
    #[test]
    fn an_euler_step_moves_along_the_path() {
        let mut x = vec![1.0f32, 2.0, -1.0];
        let v = vec![0.5f32, -1.0, 2.0];
        euler_step(&mut x, &v, 1.0, 0.8);
        // dt is -0.2, so x -= 0.2 * v.
        assert!((x[0] - 0.9).abs() < 1e-6, "{}", x[0]);
        assert!((x[1] - 2.2).abs() < 1e-6, "{}", x[1]);
        assert!((x[2] + 1.4).abs() < 1e-6, "{}", x[2]);
        // A step that goes nowhere changes nothing.
        let before = x.clone();
        euler_step(&mut x, &v, 0.8, 0.8);
        assert_eq!(x, before);
    }

    /// Guidance at 1 is the conditional answer, and it extrapolates above that.
    #[test]
    fn guidance_extrapolates_away_from_the_unconditional() {
        let cond = vec![2.0f32, 0.0];
        let uncond = vec![1.0f32, 1.0];
        assert_eq!(guide(&cond, &uncond, 1.0), cond, "scale 1 is a no-op");
        assert_eq!(
            guide(&cond, &uncond, 0.0),
            uncond,
            "scale 0 ignores the prompt"
        );
        // Scale 3 goes twice as far past cond as cond is past uncond.
        assert_eq!(guide(&cond, &uncond, 3.0), vec![4.0, -2.0]);
    }
}
