//! Turning logits into a token.
//!
//! # Why this is not optional
//!
//! Until now Bigtea took the argmax. That is a legitimate decoding strategy and
//! it is also the one that makes a model look worst: every answer is identical,
//! lists repeat, and long generations fall into loops. Comparing quality against
//! llama.cpp without samplers compares a greedy decode to a sampled one and
//! learns nothing about either model.
//!
//! # The order matters and is not arbitrary
//!
//! ```text
//! repeat penalty -> top_k -> top_p -> min_p -> temperature -> draw
//! ```
//!
//! Penalties act on **logits**; the truncations act on **probabilities**; and
//! temperature is applied last so that `top_p` cuts the distribution the model
//! actually produced rather than a flattened one. Applying temperature first —
//! an easy and invisible mistake — makes `top_p = 0.9` mean something different
//! at every temperature.
//!
//! Every filter has a documented disabled value, and the defaults here disable
//! all of them, so `Sampler::default()` is exactly the old greedy behaviour and
//! nothing changes for a caller that does not opt in.

/// Which filters run, and how hard.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// `0.0` means greedy — take the argmax and skip every filter.
    pub temperature: f32,
    /// Keep only the `k` highest-probability tokens. `0` disables.
    pub top_k: usize,
    /// Keep the smallest set whose probabilities sum to `p`. `1.0` disables.
    pub top_p: f32,
    /// Drop tokens below `min_p * p_max`. `0.0` disables.
    ///
    /// Unlike `top_p` this scales with the model's own confidence: when the
    /// model is sure, it keeps almost nothing; when it is unsure, it keeps a
    /// lot. That makes it far less sensitive to the value chosen.
    pub min_p: f32,
    /// Divide the logit of a recently used token by this. `1.0` disables.
    pub repeat_penalty: f32,
    /// Subtract `frequency_penalty * count` from a token's logit. `0.0`
    /// disables. OpenAI's `frequency_penalty`, and llama.cpp's
    /// `--frequency-penalty`: proportional to how often the token was used.
    pub frequency_penalty: f32,
    /// Subtract `presence_penalty` from any token used at all in the window.
    /// `0.0` disables. OpenAI's `presence_penalty`: flat, so it discourages
    /// returning to a subject rather than repeating a word within one.
    pub presence_penalty: f32,
    /// How far back all three penalties look.
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        // Greedy: identical to what the runner did before samplers existed.
        SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

impl SamplerConfig {
    /// llama.cpp's defaults, for a like-for-like comparison.
    ///
    /// Quoting a quality difference against a competitor while running
    /// different sampling settings measures the settings, not the engines.
    pub fn llamacpp_defaults() -> Self {
        SamplerConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            // llama.cpp's own defaults for these two are 0.0 — off.
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }

    /// Greedy decoding takes the argmax and skips every filter.
    ///
    /// The penalties are the exception worth stating: they change *which* token
    /// is the argmax, so a caller asking for `temperature 0` plus a penalty
    /// wants the penalised argmax, not the raw one. An OpenAI client sending
    /// `temperature: 0, frequency_penalty: 1.0` is a normal request and would
    /// otherwise be silently ignored.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
            && self.repeat_penalty == 1.0
            && self.frequency_penalty == 0.0
            && self.presence_penalty == 0.0
    }

    /// Whether any penalty is active, so a greedy-with-penalties path can
    /// still be taken without running the full sampling pipeline.
    pub fn has_penalties(&self) -> bool {
        self.repeat_penalty != 1.0 || self.frequency_penalty != 0.0 || self.presence_penalty != 0.0
    }
}

/// Deterministic given its seed, so a run can be reproduced exactly.
pub struct Sampler {
    config: SamplerConfig,
    state: u64,
    /// Scratch, reused across tokens to keep sampling allocation-free on the
    /// hot path — a 150k-entry vocabulary allocated per token is real work.
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        // A zero seed is the common "I did not choose one" value, and it must
        // not degenerate: SplitMix64 from 0 is fine, but being explicit here
        // means a caller passing 0 still gets a usable stream.
        let state = config.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        Sampler {
            config,
            state,
            candidates: Vec::new(),
        }
    }

    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Uniform in `[0, 1)`, SplitMix64.
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 24 bits is the whole mantissa of an f32; taking more would not add
        // resolution and would bias the low bit.
        (z >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Pick the next token. `history` is the tokens so far, for the penalty.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        if self.config.is_greedy() {
            return argmax(logits);
        }

        self.candidates.clear();
        self.candidates
            .extend(logits.iter().enumerate().map(|(i, &l)| (i as u32, l)));

        apply_penalties(
            &mut self.candidates,
            history,
            self.config.repeat_penalty,
            self.config.frequency_penalty,
            self.config.presence_penalty,
            self.config.repeat_last_n,
        );

        // Temperature 0 with a penalty active is a normal request — an OpenAI
        // client sending `temperature: 0, frequency_penalty: 1.0` means "argmax
        // of the penalised logits". It must NOT fall through to the pipeline
        // below: temperature 0 there becomes `powf(1e6)`, which drives every
        // probability to zero or one and picks nonsense.
        if self.config.temperature <= 0.0 {
            return self
                .candidates
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|c| c.0)
                .unwrap_or(0);
        }

        // Sort once, descending. Every truncation below is then a prefix, which
        // is why they compose in any order without re-sorting.
        self.candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        if self.config.top_k > 0 {
            self.candidates.truncate(self.config.top_k);
        }

        softmax(&mut self.candidates);
        apply_top_p(&mut self.candidates, self.config.top_p);
        apply_min_p(&mut self.candidates, self.config.min_p);

        // Temperature last, on the surviving set, so top_p meant what it says.
        if (self.config.temperature - 1.0).abs() > f32::EPSILON {
            let inv = 1.0 / self.config.temperature.max(1e-6);
            for c in self.candidates.iter_mut() {
                c.1 = c.1.powf(inv);
            }
            let total: f32 = self.candidates.iter().map(|c| c.1).sum();
            if total > 0.0 {
                for c in self.candidates.iter_mut() {
                    c.1 /= total;
                }
            }
        }

        let r = self.next_f32();
        let mut acc = 0.0;
        for &(id, p) in &self.candidates {
            acc += p;
            if r < acc {
                return id;
            }
        }
        // Floating point can leave `acc` a hair under `r`; the last candidate is
        // the correct answer rather than a failure.
        self.candidates.last().map(|c| c.0).unwrap_or(0)
    }
}

/// `-log softmax(logits)[target]`, in nats, without ever forming the softmax.
///
/// This is the per-token quantity perplexity averages, so its accuracy is the
/// accuracy of the whole measurement.
///
/// Two things it must get right, both of which fail silently:
///
/// - **Subtract the max first.** Logits reach ±30 and `exp(30)` overflows an
///   `f32` to `inf`, which turns the sum into `inf`, the log into `inf`, and
///   the reported perplexity into `NaN` — after the run, with nothing to
///   indicate which token did it.
/// - **Accumulate in `f64`.** The sum runs over a vocabulary of up to 256,000
///   terms and is then averaged across thousands of tokens; `f32` loses real
///   precision at that length, and the error is one-directional rather than
///   noise.
pub fn neg_log_prob(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    if !max.is_finite() {
        return f64::INFINITY;
    }
    let sum: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
    let target_logit = logits.get(target).copied().unwrap_or(f32::NEG_INFINITY) as f64;
    // log P = (x_t - max) - log(sum exp(x - max))
    -((target_logit - max) - sum.ln())
}

pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
fn apply_repeat_penalty(
    candidates: &mut [(u32, f32)],
    history: &[u32],
    penalty: f32,
    last_n: usize,
) {
    apply_penalties(candidates, history, penalty, 0.0, 0.0, last_n)
}

/// Repetition, frequency and presence penalties, in llama.cpp's order.
///
/// A negative logit must be *multiplied* by the repeat penalty and a positive
/// one divided; using one operation for both makes the penalty **reward**
/// tokens whose logit is negative, which is the opposite of the intent and is a
/// real bug in more than one implementation.
///
/// Three different ideas that share one window:
///
/// - **repeat** is *multiplicative* — divide a positive logit, multiply a
///   negative one — and does not care how often the token appeared.
/// - **frequency** is *subtractive* and scales with the count, so a token used
///   five times is pushed down five times as far as one used once.
/// - **presence** is *subtractive* and flat: it costs the same to reuse a token
///   once as ten times, which discourages returning to a topic rather than
///   discouraging repetition within it.
///
/// The last two are what the OpenAI API means by `frequency_penalty` and
/// `presence_penalty`, and a client that sends them expects these semantics
/// exactly. llama.cpp applies the multiplicative one first; so do we.
fn apply_penalties(
    candidates: &mut [(u32, f32)],
    history: &[u32],
    repeat: f32,
    frequency: f32,
    presence: f32,
    last_n: usize,
) {
    if last_n == 0 || history.is_empty() {
        return;
    }
    if repeat == 1.0 && frequency == 0.0 && presence == 0.0 {
        return;
    }
    let window = &history[history.len().saturating_sub(last_n)..];
    // Count the window once. The previous shape swept the whole candidate list
    // calling `contains` on each — 151,936 x 64 comparisons per token, to act
    // on at most 64 distinct tokens.
    let mut counts: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::with_capacity(window.len());
    for &t in window {
        *counts.entry(t).or_insert(0) += 1;
    }
    // Looked up by token id rather than by index: this runs before the sort
    // today, but a caller that sorted first would otherwise penalise whichever
    // tokens happened to land in those slots — silently, and only visibly as
    // slightly odd text.
    for c in candidates.iter_mut() {
        let Some(&count) = counts.get(&c.0) else {
            continue;
        };
        if repeat != 1.0 {
            c.1 = if c.1 > 0.0 {
                c.1 / repeat
            } else {
                c.1 * repeat
            };
        }
        c.1 -= count as f32 * frequency + presence;
    }
}

/// In-place softmax over a descending-sorted candidate list.
fn softmax(candidates: &mut [(u32, f32)]) {
    let Some(&(_, max)) = candidates.first() else {
        return;
    };
    let mut total = 0.0;
    for c in candidates.iter_mut() {
        // Subtract the max before exponentiating: `exp(30)` overflows f32, and
        // logits reach that range.
        c.1 = (c.1 - max).exp();
        total += c.1;
    }
    if total > 0.0 {
        for c in candidates.iter_mut() {
            c.1 /= total;
        }
    }
}

fn apply_top_p(candidates: &mut Vec<(u32, f32)>, top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    let mut acc = 0.0;
    let mut keep = candidates.len();
    for (i, c) in candidates.iter().enumerate() {
        acc += c.1;
        if acc >= top_p {
            // Inclusive: the token that crosses the threshold is kept, or
            // top_p just below the top token's probability would keep nothing.
            keep = i + 1;
            break;
        }
    }
    candidates.truncate(keep.max(1));
    renormalise(candidates);
}

fn apply_min_p(candidates: &mut Vec<(u32, f32)>, min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    let Some(&(_, max)) = candidates.first() else {
        return;
    };
    let floor = min_p * max;
    let keep = candidates.iter().take_while(|c| c.1 >= floor).count();
    candidates.truncate(keep.max(1));
    renormalise(candidates);
}

fn renormalise(candidates: &mut [(u32, f32)]) {
    let total: f32 = candidates.iter().map(|c| c.1).sum();
    if total > 0.0 {
        for c in candidates.iter_mut() {
            c.1 /= total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Logits where token 3 is clearly best and 1 is second.
    fn logits() -> Vec<f32> {
        vec![0.0, 2.0, -1.0, 5.0, 1.0]
    }

    #[test]
    fn the_default_is_exactly_the_old_greedy_behaviour() {
        // Adding samplers must not change what an existing caller gets.
        let mut s = Sampler::new(SamplerConfig::default());
        assert!(s.config().is_greedy());
        for _ in 0..10 {
            assert_eq!(s.sample(&logits(), &[]), 3);
        }
    }

    #[test]
    fn the_same_seed_reproduces_the_same_sequence() {
        // Without this a bug report cannot be reproduced, and neither can a
        // benchmark.
        let cfg = SamplerConfig {
            temperature: 1.0,
            seed: 42,
            ..SamplerConfig::default()
        };
        let run = || {
            let mut s = Sampler::new(cfg.clone());
            (0..32)
                .map(|_| s.sample(&logits(), &[]))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());

        let other = SamplerConfig { seed: 43, ..cfg };
        let mut s = Sampler::new(other);
        let different: Vec<u32> = (0..32).map(|_| s.sample(&logits(), &[])).collect();
        assert_ne!(run(), different, "a different seed must differ");
    }

    #[test]
    fn top_k_one_is_greedy_however_hot_the_temperature() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 5.0,
            top_k: 1,
            ..SamplerConfig::default()
        });
        for _ in 0..32 {
            assert_eq!(s.sample(&logits(), &[]), 3);
        }
    }

    #[test]
    fn temperature_widens_the_distribution() {
        // Cold sampling should almost always take the best token; hot sampling
        // should not. If both look the same, temperature is not being applied.
        let count = |t: f32| {
            let mut s = Sampler::new(SamplerConfig {
                temperature: t,
                seed: 7,
                ..SamplerConfig::default()
            });
            (0..400).filter(|_| s.sample(&logits(), &[]) == 3).count()
        };
        let cold = count(0.1);
        let hot = count(3.0);
        assert!(cold > 380, "cold sampling should be near-greedy: {cold}");
        assert!(hot < cold, "hot {hot} must be more varied than cold {cold}");
    }

    #[test]
    fn top_p_keeps_a_prefix_and_min_p_scales_with_confidence() {
        let mut c = vec![(0u32, 0.6f32), (1, 0.3), (2, 0.07), (3, 0.03)];
        apply_top_p(&mut c, 0.9);
        assert_eq!(c.len(), 2, "0.6 + 0.3 crosses 0.9, so two are kept");
        assert!((c.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5);

        // min_p is relative to the top token, so the same cutoff behaves
        // differently on a confident and an unsure distribution.
        let mut confident = vec![(0u32, 0.9f32), (1, 0.06), (2, 0.04)];
        apply_min_p(&mut confident, 0.1);
        assert_eq!(confident.len(), 1, "0.09 floor drops both tails");

        let mut unsure = vec![(0u32, 0.3f32), (1, 0.28), (2, 0.24), (3, 0.18)];
        apply_min_p(&mut unsure, 0.1);
        assert_eq!(unsure.len(), 4, "a flat distribution keeps everything");
    }

    #[test]
    fn a_filter_never_empties_the_candidate_list() {
        // An aggressive setting must degrade to greedy, not to a panic or to
        // token 0 — which would look like the model emitting padding.
        let mut c = vec![(9u32, 1.0f32)];
        apply_top_p(&mut c, 0.0);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].0, 9);

        let mut c = vec![(9u32, 0.5f32), (1, 0.5)];
        apply_min_p(&mut c, 1.5);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn the_repeat_penalty_pushes_a_token_down_whatever_its_sign() {
        // The sign trap: a negative logit must be multiplied, not divided, or
        // the "penalty" makes a repeated token MORE likely.
        let mut c = vec![(0u32, 4.0f32), (1, -4.0)];
        apply_repeat_penalty(&mut c, &[0, 1], 2.0, 64);
        assert_eq!(c[0].1, 2.0, "positive logit is divided");
        assert_eq!(
            c[1].1, -8.0,
            "negative logit must be multiplied, not raised"
        );
    }

    #[test]
    fn neg_log_prob_matches_a_hand_computed_softmax() {
        // Two equal logits: each has probability 1/2, so -log p = ln 2.
        let nll = neg_log_prob(&[1.0, 1.0], 0);
        assert!((nll - std::f64::consts::LN_2).abs() < 1e-9, "got {nll}");
        // Uniform over four: -log(1/4) = ln 4.
        let nll = neg_log_prob(&[3.0, 3.0, 3.0, 3.0], 2);
        assert!((nll - 4f64.ln()).abs() < 1e-9, "got {nll}");
    }

    #[test]
    fn a_large_logit_does_not_overflow_to_nan() {
        // exp(400) is inf in f64, let alone f32. Subtracting the max is the
        // only reason this is finite, and getting it wrong poisons the whole
        // average with no indication of which token did it.
        let nll = neg_log_prob(&[400.0, 399.0, -400.0], 0);
        assert!(nll.is_finite(), "got {nll}");
        assert!(
            nll > 0.0 && nll < 1.0,
            "the max logit should be likely: {nll}"
        );
    }

    #[test]
    fn shifting_every_logit_leaves_the_answer_alone() {
        // Softmax is invariant under adding a constant, so this is a free
        // check that the max subtraction has not changed the maths.
        let base = [2.5f32, -1.0, 0.75, 4.0];
        let shifted: Vec<f32> = base.iter().map(|l| l + 17.0).collect();
        for t in 0..base.len() {
            let a = neg_log_prob(&base, t);
            let b = neg_log_prob(&shifted, t);
            assert!((a - b).abs() < 1e-6, "target {t}: {a} vs {b}");
        }
    }

    #[test]
    fn probabilities_over_every_target_sum_to_one() {
        let logits = [1.5f32, -2.0, 0.25, 3.0, -0.5];
        let total: f64 = (0..logits.len())
            .map(|t| (-neg_log_prob(&logits, t)).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-9, "got {total}");
    }

    #[test]
    fn frequency_scales_with_the_count_and_presence_does_not() {
        // Token 0 appears three times, token 1 once. Frequency must punish 0
        // three times as hard; presence must punish them identically. Getting
        // these two the same way round is the whole difference between the
        // OpenAI fields, and neither errors when swapped.
        let mut c = vec![(0u32, 10.0f32), (1, 10.0)];
        apply_penalties(&mut c, &[0, 0, 0, 1], 1.0, 1.0, 0.0, 64);
        assert_eq!(c[0].1, 7.0, "used 3 times, so -3");
        assert_eq!(c[1].1, 9.0, "used once, so -1");

        let mut c = vec![(0u32, 10.0f32), (1, 10.0)];
        apply_penalties(&mut c, &[0, 0, 0, 1], 1.0, 0.0, 2.0, 64);
        assert_eq!(c[0].1, 8.0, "presence is flat");
        assert_eq!(c[1].1, 8.0, "...for both");
    }

    #[test]
    fn an_unused_token_is_untouched_by_any_penalty() {
        let mut c = vec![(7u32, 3.0f32)];
        apply_penalties(&mut c, &[0, 1, 2], 2.0, 5.0, 5.0, 64);
        assert_eq!(c[0].1, 3.0);
    }

    #[test]
    fn the_three_penalties_compose_in_llamacpp_s_order() {
        // Multiplicative first, then the two subtractive ones: 8/2 - 1*1 - 3.
        let mut c = vec![(0u32, 8.0f32)];
        apply_penalties(&mut c, &[0], 2.0, 1.0, 3.0, 64);
        assert_eq!(c[0].1, 0.0);
    }

    #[test]
    fn temperature_zero_with_a_penalty_is_the_penalised_argmax() {
        // An OpenAI client sending `temperature: 0, frequency_penalty: N` is
        // normal. Falling through to the sampling pipeline would raise every
        // probability to the power of 1e6; returning the raw argmax would
        // ignore the penalty. Neither is right.
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            frequency_penalty: 5.0,
            ..SamplerConfig::default()
        });
        // Token 0 wins on raw logits, but it has been used and 1 has not.
        let chosen = s.sample(&[3.0, 1.0, 0.5], &[0, 0]);
        assert_eq!(chosen, 1, "the penalty must decide it");

        // ...and with no penalty at all, greedy is still greedy.
        let mut s = Sampler::new(SamplerConfig::default());
        assert_eq!(s.sample(&[3.0, 1.0, 0.5], &[0, 0]), 0);
    }

    #[test]
    fn the_penalty_only_looks_at_its_window() {
        let mut c = vec![(5u32, 4.0f32)];
        // Token 5 is in history but outside the last 2 entries.
        apply_repeat_penalty(&mut c, &[5, 1, 2], 2.0, 2);
        assert_eq!(c[0].1, 4.0, "outside the window, nothing is penalised");

        let mut c = vec![(5u32, 4.0f32)];
        apply_repeat_penalty(&mut c, &[5, 1, 2], 2.0, 3);
        assert_eq!(c[0].1, 2.0, "inside the window, it is");
    }

    #[test]
    fn softmax_survives_logits_that_would_overflow() {
        // exp(500) is inf in f32. Without the max-subtraction this is NaN and
        // the sampler silently returns token 0 forever.
        let mut c = vec![(0u32, 500.0f32), (1, 499.0), (2, -500.0)];
        c.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        softmax(&mut c);
        let total: f32 = c.iter().map(|x| x.1).sum();
        assert!((total - 1.0).abs() < 1e-5, "probabilities must sum to 1");
        assert!(c.iter().all(|x| x.1.is_finite()));
    }

    #[test]
    fn empty_logits_do_not_panic() {
        let mut s = Sampler::new(SamplerConfig::llamacpp_defaults());
        assert_eq!(s.sample(&[], &[]), 0);
    }
}
