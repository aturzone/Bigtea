//! Qwen3 — dense and mixture-of-experts.
//!
//! A standard pre-norm transformer, with two details that are easy to get
//! wrong and produce fluent nonsense rather than an error:
//!
//! * **Per-head Q/K normalisation.** Qwen3 RMS-normalises each attention head
//!   *before* RoPE, using a weight of `head_dim`, not `n_embd`. Skipping it,
//!   or applying it across the whole vector, changes every attention score.
//! * **NeoX-style RoPE.** Rotation pairs dimension `i` with `i + head_dim/2`,
//!   not with `i + 1`. The other convention is also "valid RoPE" and produces
//!   plausible output, so this cannot be caught by eyeballing results.
//!
//! The MoE variant differs only in the feed-forward block: a router picks
//! `n_expert_used` of `n_expert`, and `mul_mat_id` applies just those.

use chaos_ggml::{Context, Tensor, WeightSet};
use chaos_model::Model;

use crate::{ArchError, Result};

/// NeoX rotary convention — see the module note on why this matters.
const ROPE_TYPE_NEOX: i32 = 2;

/// The other RoPE convention, and the one Llama uses.
///
/// NORM rotates **adjacent** pairs `(x0,x1), (x2,x3), …`; NEOX rotates halves,
/// `(x0, x[d/2]), (x1, x[d/2+1]), …`. Both are "rotary position embedding" and
/// both run without error on either layout — **the wrong one produces fluent
/// nonsense**, which is this project's most expensive failure mode.
///
/// llama.cpp splits them by architecture: `llama`, `baichuan` and `deci` are
/// NORM; `qwen2`, `qwen3`, `phi3`, `gemma` and most others are NEOX.
const ROPE_TYPE_NORM: i32 = 0;

/// Which convention an architecture uses, by name.
///
/// Defaulting to NEOX rather than NORM is deliberate: NEOX is the majority, and
/// an architecture this list has never seen is more likely to be one of them.
/// It is still a guess, which is why [`Qwen3Config::rope_type_is_known`] exists
/// and the runner says so out loud.
/// `GGML_ROPE_TYPE_IMROPE`, the interleaved multimodal rotation Qwen3.5 uses.
///
/// 40, not 8: `MROPE` is 8 and `IMROPE` sets an extra bit on top of it. The two
/// differ in which dimension pairs the four sections address, and both run
/// without error on the other's layout.
pub const ROPE_TYPE_IMROPE: i32 = 40;

fn rope_type_for(arch: &str) -> (i32, bool) {
    match arch {
        // `olmo` and `internlm2` were read straight out of llama.cpp's
        // `llama_model_rope_type`, which lists both in the NORM branch beside
        // `LLM_ARCH_LLAMA`. `olmo` was previously in the NEOX arm below **with
        // `known = true`** — a guess wearing the label of a checked fact, and
        // wrong. Nothing had ever been run against the reference for it.
        "llama" | "llama4" | "baichuan" | "deci" | "mistral" | "olmo" | "internlm2" | "minicpm" => {
            (ROPE_TYPE_NORM, true)
        }
        "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" | "phi3" | "gemma" | "gemma2" | "gemma3"
        | "stablelm" | "starcoder2" | "falcon" | "phi2" | "gptneox" => (ROPE_TYPE_NEOX, true),
        // **`qwen35` is Qwen3.5, Qwen3.6 and Qwen3.8** — all three report the
        // same architecture string, read from their own containers. llama.cpp
        // returns `LLAMA_ROPE_TYPE_IMROPE` for it: interleaved multimodal RoPE,
        // four sections wide, applied through `ggml_rope_multi` rather than
        // `ggml_rope_ext`. Its 16 attention layers use this; its 48 recurrent
        // layers have no rotation at all.
        //
        // Read from the container's header without downloading it: a 1 MB
        // range request over `Qwen3.8-27B-...IQ4_XS.gguf` gives
        // `general.architecture = qwen35` in under a second, which is how a new
        // model should be triaged before 16 GB moves.
        "qwen35" => (ROPE_TYPE_IMROPE, true),
        //
        // **`mpt` and `bloom` are deliberately absent.** llama.cpp returns
        // `LLAMA_ROPE_TYPE_NONE` for both: they position with ALiBi instead of
        // rotation, and this engine refuses ALiBi rather than silently running
        // a model without positional information — see `uses_alibi` in
        // `verify`. Falling through to the `_` arm would mark them "unknown"
        // and the runner would warn, which is right, but writing them here as
        // NEOX would be a guess wearing the label of a checked fact. That is
        // the exact mistake `olmo` cost.
        _ => (ROPE_TYPE_NEOX, false),
    }
}

/// The gate non-linearity of a gated feed-forward.
///
/// Both forms are `down(act(gate(x)) * up(x))` and differ only in `act`, so a
/// mismatch changes no shape, allocates no tensor, and raises nothing. It just
/// moves every FFN output a little — which reads as a different, equally
/// fluent model. **Gemma-2 was listed as verified while running SiLU**; the
/// reference said `a) Paris b) Lyon c) Marseille` where we said `**Paris**.`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnAct {
    /// Llama, Mistral, Qwen, Phi, DeepSeek — the majority.
    Silu,
    /// The Gemma family, and the one llama.cpp calls `LLM_FFN_GELU`.
    Gelu,
    /// **Not gated at all**: `down(gelu(up(x)))`, no `ffn_gate` tensor.
    /// StarCoder2 and the GPT-2 lineage. Detected from the container rather
    /// than the name, because a missing `ffn_gate` is a fact about the weights.
    UngatedGelu,
}

/// Which non-linearity an architecture's FFN uses, by name.
///
/// By name rather than by tensor, because nothing in the container records it:
/// a GELU model and a SiLU model hold byte-identical tensor sets. An unknown
/// architecture gets SiLU, and [`Qwen3Config::verify`] still refuses it.
fn ffn_act_for(arch: &str) -> FfnAct {
    match arch {
        "gemma" | "gemma2" | "gemma3" | "gemma3n" | "gemma-embedding" => FfnAct::Gelu,
        _ => FfnAct::Silu,
    }
}

/// Architectures whose output has been diffed against llama.cpp, token for
/// token, on this build.
///
/// **Membership means someone ran the reference.** `gemma2` sat here for weeks
/// while running SiLU where llama.cpp runs GELU, because "it answered in
/// English" was mistaken for a check. The diff is two commands; run them.
///
/// # Why a list, and why refusing is the right default
///
/// Gemma-2 loads through the generic dense path without a single error and
/// answers "The capital of France is" with **"himſelf"**. It has post-norms
/// after attention and the FFN, logit soft-capping, attention soft-capping,
/// embedding scaling by `sqrt(n_embd)` and sliding-window attention on
/// alternate layers — none of which this path implements, and none of which
/// announce themselves as a missing tensor.
///
/// That is the failure this project is most expensive at: **fluent nonsense
/// rather than an error.** A runner whose selling point is telling you the
/// truth about your machine cannot answer a question wrongly and confidently.
///
/// Phi-3 is the other outcome and the safe one: it uses a fused `attn_qkv`
/// rather than separate projections, so it fails immediately with the name of
/// the tensor it wanted.
///
/// So the default is to refuse an architecture nobody has checked. `--force`
/// runs it anyway, which is the right escape hatch for someone testing a new
/// architecture — but it has to be asked for.
/// A name here is **not** a claim that every model calling itself that will
/// run. `baichuan` covers a 7B this path is verified against and a 13B it
/// refuses (see [`Qwen3Config::uses_alibi`]); membership means the eight-prompt
/// diff was run against *a* container, and the refusals guard the rest.
pub const VERIFIED_ARCHITECTURES: &[&str] = &[
    "baichuan",
    // **Verified 2026-08-16 on gemma-1.1-2b-it-Q4_K_M**: 8 of 8 prompts exact
    // against llama.cpp, twice, zero unstable. Gemma v1 is NOT gemma2 -- it has
    // no post-norms and no logit soft-capping -- so this entry is its own
    // measurement and not an inference from the sibling that was already here.
    // The activation is GELU, detected from the container rather than assumed;
    // gemma2 sat on this list for weeks running SiLU by mistake.
    "gemma",
    "deepseek4",
    "gemma2",
    "gemma3",
    "internlm2",
    "llama",
    "olmo",
    "phi3",
    "qwen2",
    "qwen3",
    // **Qwen3.5, Qwen3.6 and Qwen3.8 all report `qwen35`.** Verified against
    // llama.cpp on Qwen3.5-0.8B-Q8_0 three ways, with the debug dump OFF:
    // all 24 layers by value and by sum, and byte-identical greedy output at
    // 1-, 5- and 22-token prompts -- which is what covers both regimes of the
    // fused delta rule, since it takes a different path at one token.
    //
    // `qwen35moe` is deliberately NOT here: no MoE container of this family has
    // been run, and its routed path is untested. Sharing an implementation in
    // llama.cpp is not the same as having been checked.
    "qwen35",
    // `qwen3moe` WAS HERE AND HAS BEEN REMOVED — but what still keeps it out is
    // the *harness*, not the engine.
    //
    // It had never been through the eight-prompt diff; run at last it came back
    // 1 FAIL + 6 unstable. One real bug was found and fixed: an MoE container
    // has no `ffn_gate` (its gate is `ffn_gate_exps`), so it was classified
    // ungated and ran GELU where the reference runs SiLU. That killed a loop
    // and four of the near-ties.
    //
    // The remaining FAIL is a **near-tie, demonstrated**: llama.cpp produces
    // our exact answer under `-b 1` and `-ub 1`, which change batching, and so
    // summation order, and nothing else.
    //
    //   default / -fa off / --no-repack : …Madrid. The capital of Germany is Berlin.
    //   -b 1 / -ub 1                    : …Madrid. The capital of Portugal is Lisbon.
    //   chaos                          : …Madrid. The capital of Portugal is Lisbon.
    //
    // `parity-check.sh` re-checks a mismatch against `-fa off` and
    // `--no-repack` only, and neither moves this prompt, so it reports FAIL.
    // **The set of no-op configurations you test decides what you call a bug**,
    // and `-b 1` is the axis that moves this model. It also reproduces both of
    // Phi-3's otherwise-unexplained near-ties.
    //
    // Left out regardless, because the rule is that the diff passes — not that
    // someone argues it should have.
    //
    // # 2026-08-14: the harness gained `-b 1`, and the re-run did NOT clear it
    //
    // The prediction above held exactly: with `-b 1` in the re-check set the
    // FAIL became `unstable`, because the reference now demonstrably disagrees
    // with itself on that prompt. **The count did not move.** Eight prompts,
    // 0 FAIL, and *six* unstable — the France prompt under `-b 1` alone, the
    // other five under `-fa off --no-repack -b 1`. `parity-check.sh` exits
    // non-zero and calls six a cluster rather than chance, which is the right
    // reading: one near-tie in eight is a coin landing on its edge, six is
    // something systematic that has not been found yet.
    //
    // So the entry stays out, and for a *better* reason than before. What
    // widening the probe bought was not a pass — it was the removal of the
    // explanation that made the failure look accounted for. The remaining
    // divergence is now unexplained rather than excused, and every prompt
    // tokenizes identically in both engines, so it is not the input.
    //
    // **Anything measured on this model is measured on a container that does
    // not pass the diff** — the RAM-frontier and expert-cache work is all
    // Qwen3-30B-A3B, and those nodes now say so.
    "stablelm",
    "starcoder2",
];

/// Whether this build has been run against `arch` and had its output checked.
pub fn architecture_is_verified(arch: &str) -> bool {
    VERIFIED_ARCHITECTURES.contains(&arch)
}

/// The gated delta net's shape, for the `qwen35` family.
///
/// **Three quarters of a Qwen3.5/3.6/3.8 layer stack is not attention.** Those
/// layers are a *gated delta net*: a linear-attention recurrence carrying a
/// state matrix from token to token, with a depthwise causal convolution over
/// its own q/k/v before the recurrence runs. Everything here is read from the
/// container's `qwen35.ssm.*` keys, and the derived widths below are what make
/// the fused tensors decomposable.
///
/// For Qwen3.5-0.8B: `group_count 16`, `state_size 128`, `time_step_rank 16`,
/// `inner_size 2048`, so `key_dim = 2048`, `value_dim = 2048`, `head_v_dim =
/// 128`, and `conv_dim = 6144` — which is exactly the width of `attn_qkv` and
/// the second dimension of `ssm_conv1d` in that file. The 27B is the same
/// arithmetic at `time_step_rank 48`, giving 10240.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsmConfig {
    /// `ssm.conv_kernel` — the convolution window, 4. The rolling state keeps
    /// `conv_kernel - 1` past values per channel.
    pub conv_kernel: u32,
    /// `ssm.inner_size` — the value path's total width, and `ssm_out`'s input.
    pub inner_size: u32,
    /// `ssm.state_size` — the head dimension shared by keys and values, 128.
    /// **Also the side of the carried state matrix**, which is
    /// `[state_size, state_size]` per value head.
    pub state_size: u32,
    /// `ssm.group_count` — how many *key* heads. Fewer than the value heads,
    /// so q and k are broadcast up before the recurrence.
    pub group_count: u32,
    /// `ssm.time_step_rank` — how many *value* heads, and the width of
    /// `ssm_a`, `ssm_dt.bias`, `ssm_alpha` and `ssm_beta`.
    pub time_step_rank: u32,
    /// `full_attention_interval` — every `n`th layer is real attention.
    pub full_attention_interval: u32,
}

impl SsmConfig {
    /// `group_count * state_size`: the width of q, and of k, in `attn_qkv`.
    pub fn key_dim(&self) -> u32 {
        self.group_count * self.state_size
    }

    /// `inner_size`: the width of v, and of the output gate `attn_gate`.
    pub fn value_dim(&self) -> u32 {
        self.inner_size
    }

    /// `inner_size / time_step_rank`. Equal to `state_size` in every published
    /// Qwen3.5 size, and computed rather than assumed because the fused op
    /// requires `S_k == S_v` and would **abort** if they ever diverged.
    pub fn head_v_dim(&self) -> u32 {
        self.inner_size / self.time_step_rank.max(1)
    }

    /// `2 * key_dim + value_dim`: the width of `attn_qkv` and the channel count
    /// of `ssm_conv1d`.
    pub fn conv_dim(&self) -> u32 {
        2 * self.key_dim() + self.value_dim()
    }

    /// Floats of rolling convolution window per sequence, per layer.
    pub fn conv_state_len(&self) -> usize {
        ((self.conv_kernel - 1) * self.conv_dim()) as usize
    }

    /// Floats of carried recurrent state per sequence, per layer:
    /// `head_v_dim * head_v_dim * time_step_rank`.
    pub fn recurrent_state_len(&self) -> usize {
        (self.head_v_dim() * self.head_v_dim() * self.time_step_rank) as usize
    }
}

/// Shape and hyper-parameters, read from the container rather than assumed.
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub n_ff: u32,
    pub vocab_size: u32,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    /// Linear RoPE scaling. `1.0` is unscaled; `0.5` doubles the effective
    /// context by halving every frequency. llama.cpp's `--rope-freq-scale`,
    /// and the reciprocal of its `--rope-scale`.
    pub rope_freq_scale: f32,
    /// YaRN mix, `0.0` = pure linear scaling (i.e. YaRN off). ggml's
    /// `rope_ext` consumes all four of these; nothing else here reads them.
    pub rope_ext_factor: f32,
    /// YaRN's magnitude correction, applied to the whole attention score.
    pub rope_attn_factor: f32,
    pub rope_beta_fast: f32,
    pub rope_beta_slow: f32,
    /// The context length the model was trained at, which YaRN interpolates
    /// *from*. `0` means the container did not say.
    pub rope_orig_ctx: u32,
    /// Expert count; zero for the dense variant.
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_ff_expert: u32,
    /// Present only for the `qwen35` family; `None` means every layer is
    /// attention, which is every other architecture here.
    pub ssm: Option<SsmConfig>,
    /// mRoPE section widths, four of them, from `rope.dimension_sections`.
    ///
    /// **Not decoration.** The sections decide which dimension pairs rotate
    /// together, so a text-only prompt still goes through `rope_multi` rather
    /// than `rope_ext` — and substituting the plain rotation is silent.
    pub rope_sections: Option<[i32; 4]>,
    /// What the norm in front of the feed-forward is *called* in the container.
    ///
    /// `ffn_norm` everywhere except `qwen35`, which names the same slot
    /// `post_attention_norm`. It sits between the attention residual and the
    /// FFN, which is the ordinary pre-norm position — the name is the only
    /// thing unusual about it, and looking for the usual name refuses the
    /// container outright.
    pub ffn_norm_name: &'static str,
    /// `attn_q` holds the query **and** a gate, each `head_dim * n_head` wide.
    ///
    /// Qwen3.5 projects both in one matmul and multiplies the attention output
    /// by `sigmoid(gate)`. Reading the tensor as a plain query takes the first
    /// half of an interleaved pair and produces fluent nonsense.
    pub fused_q_gate: bool,
    /// Whether each block carries `attn_q_norm` / `attn_k_norm`.
    ///
    /// Qwen3 normalises every attention head separately, with a weight of
    /// `head_dim` rather than `n_embd`. **Llama, Mistral, Qwen2, Gemma and Phi
    /// do not have these tensors at all** — and requiring them is what refused
    /// the entire Llama family before a byte was read, since the container check
    /// runs against `required_tensors` up front.
    ///
    /// Detected from the container rather than from the architecture name: the
    /// tensor either exists or it does not, and that is a fact about this file
    /// rather than about what the file claims to be. A finetune that drops or
    /// adds QK-norm is then handled without a new arch name.
    pub qk_norm: bool,
    /// `ROPE_TYPE_NORM` or `ROPE_TYPE_NEOX` — see [`rope_type_for`].
    pub rope_type: i32,
    /// False when the architecture was not in [`rope_type_for`]'s list and the
    /// type is a default rather than a fact. Nothing can detect this from the
    /// weights, so the only honest thing is to say so.
    pub rope_type_is_known: bool,
    /// Q, K and V share one `attn_qkv` tensor rather than three.
    ///
    /// Phi-3 ships them fused. The split is along the output dimension and the
    /// rows are whole quantisation blocks, so three views cost nothing — but
    /// asking for `attn_q.weight` on such a container fails outright, which is
    /// what refused Phi-3 before this existed.
    /// Q, K and V projections carry a bias vector.
    ///
    /// **Qwen2 has these and Qwen3 does not.** Ignoring them is not a missing
    /// tensor and not an error — every attention score is simply shifted, and
    /// Qwen2-0.5B answers "The capital of France is" with
    /// `睢已经是成人istentation帮助企业 Hague(ord壑屁`. Detected from the
    /// container rather than the architecture name, so a finetune that adds or
    /// drops them is handled without a new arch.
    pub attn_bias: bool,
    /// Whether every norm in the block is a **LayerNorm** rather than RMSNorm.
    ///
    /// Asked of the container, like `qk_norm` and `fused_qkv`: **a norm with a
    /// bias is a LayerNorm.** RMSNorm divides by the root-mean-square and has
    /// weight only; LayerNorm centres first and has weight *and* bias.
    /// Substituting one is not an error and not a crash — StableLM and
    /// StarCoder2 read as fluent CJK noise before this existed.
    pub layer_norm: bool,
    /// Whether the norms have learned parameters at all.
    ///
    /// **OLMo has none**: llama.cpp builds every one of its norms as
    /// `build_norm(x, NULL, NULL, LLM_NORM)` — centre, divide by the standard
    /// deviation, and stop. The container holds no `attn_norm.weight`, no
    /// `ffn_norm.weight` and no `output_norm.weight`, so the previous code
    /// refused it up front with `container has no tensor "output_norm.weight"`.
    /// That refusal was the *good* outcome; the danger is the opposite reading,
    /// where an affine architecture loses a norm weight and quietly runs
    /// non-parametric. Hence a flag rather than "apply the weight if you find
    /// one": when this is true a missing weight is still `MissingTensor`.
    pub norm_affine: bool,
    /// Whether the norms carry a **shift** as well as a scale.
    ///
    /// Separate from [`layer_norm`](Self::layer_norm), and the separation is
    /// the OLMo lesson: LayerNorm-ness and having-a-bias used to be the same
    /// boolean, because every LayerNorm seen so far had one. OLMo is a
    /// LayerNorm with neither parameter, so folding the two together made the
    /// loader demand an `output_norm.bias` that cannot exist.
    ///
    /// It still gates `required_tensors`, which is what keeps the original
    /// guarantee: if `blk.0` has a bias then every layer must list one, and a
    /// bias that is not listed is never loaded and silently skipped.
    pub norm_bias: bool,
    /// Symmetric clamp on Q, K and V after projection; `0.0` means none.
    ///
    /// llama.cpp applies it inside `build_qkv`, after the bias and before the
    /// reshape. Declared by MPT, DBRX and OLMo — the OLMo-1B container says
    /// `0.0`, so this is implemented against the reference's *code* rather than
    /// against a run, and the 7B is the container that would exercise it.
    pub clamp_kqv: f32,
    /// This model biases attention by distance instead of rotating — and
    /// **nothing in the container says so.**
    ///
    /// llama.cpp reads it from the *layer count*: `baichuan.cpp` sets
    /// `f_max_alibi_bias = 8.0` when `n_layer == 40`, which is the 13B. So one
    /// architecture name covers a model this path runs correctly (the 7B,
    /// verified) and one it cannot. [`Qwen3::verify`] refuses the second rather
    /// than rotating keys that should not be rotated.
    pub uses_alibi: bool,
    /// The container ships `rope_freqs.weight`: **per-frequency RoPE divisors,
    /// carried as a tensor rather than as metadata.**
    ///
    /// This is how llama.cpp represents `rope_scaling.rope_type = "llama3"` —
    /// the low/high frequency factors are folded into `n_rot/2` numbers at
    /// conversion time, and `ggml_rope_ext` takes them as `freq_factors`. The
    /// metadata says `rope scaling = linear, freq_scale_train = 1`, which reads
    /// exactly like a model that was never extended.
    pub rope_freqs: bool,
    /// Biases on `attn_output`, `ffn_up` and `ffn_down`.
    ///
    /// Separate from [`attn_bias`](Self::attn_bias), which is Q/K/V: Qwen2 has
    /// the latter and not these, StarCoder2 has both.
    pub ffn_bias: bool,
    pub attn_out_bias: bool,
    /// Rotated dimensions per head. **Not always `head_dim`** — StableLM
    /// declares 16 of its 64, and the rest pass through unrotated. Rotating all
    /// of them is fluent nonsense, and this was ignored entirely before.
    pub n_rot: u32,
    pub fused_qkv: bool,
    /// The FFN gate and up projections share one `ffn_up` tensor.
    ///
    /// Also Phi-3. `ffn_up` is `2 * n_ff` rows: gate first, then up.
    pub fused_gate_up: bool,
    /// **Attention and the FFN read the SAME normalisation, and both add into
    /// one residual.** GPT-NeoX's shape, used by `falcon`, `gptneox` and
    /// `phi2`:
    ///
    /// ```text
    ///   serial    x -> attn_norm -> attn -> +x -> ffn_norm -> ffn -> +x
    ///   parallel  h = attn_norm(x);  x = x + attn(h) + ffn(h)
    /// ```
    ///
    /// Detected from the container rather than the architecture name: a
    /// parallel model has **no `ffn_norm`**, because there is only one norm per
    /// block. phi-2 failed to load with `container has no tensor
    /// "blk.0.ffn_norm.weight"`, which is the honest symptom -- unlike the
    /// activation, this one cannot be got wrong silently, because the missing
    /// tensor is a hard error rather than a different number.
    pub parallel_residual: bool,
    /// Gemma normalises again *after* attention and *after* the FFN, on top of
    /// the pre-norms every other architecture here uses.
    ///
    /// Detected from the container. Omitting these does not fail — the residual
    /// stream simply drifts, and the model answers fluently and wrongly.
    pub post_norms: bool,
    /// Multiply the embeddings by `sqrt(n_embd)` on the way in. Gemma only.
    pub scale_embeddings: bool,
    /// The dimension the attention scale is `1 / sqrt(·)` of.
    ///
    /// Normally `head_dim`. **Gemma's 27B variants use `n_embd / n_head`
    /// instead** — 4608/32 = 144 against a `head_dim` of 128, a 6% difference
    /// that changes every attention score and breaks no shape. llama.cpp keys
    /// it off the model size, so this does too; a check that passed on the 1B
    /// would still have been wrong at 27B.
    pub attn_scale_dim: u32,
    /// Scale Q before the attention kernel instead of handing it the scale.
    ///
    /// Algebraically the same thing, and **not the same in floating point**:
    /// ggml folds the scale into the soft cap (`scale /= cap`), so the two
    /// orders round differently and the `tanh` sees a different number. One
    /// ULP was enough to flip Gemma-2's first token from `:` to ` Paris` and
    /// with it the whole completion. Set for whatever llama.cpp pre-scales, so
    /// the two agree bit for bit rather than merely in exact arithmetic.
    pub prescale_q: bool,
    /// Which non-linearity the gated feed-forward applies to its gate.
    ///
    /// **Llama, Mistral, Qwen and Phi use SiLU; the whole Gemma family uses
    /// GELU.** Running Gemma with SiLU is not a missing tensor and not an
    /// error — every FFN output is merely a slightly different curve, and the
    /// model still answers fluently. Gemma-2 shipped in
    /// `VERIFIED_ARCHITECTURES` for weeks in exactly that state.
    pub ffn_act: FfnAct,
    /// `tanh` soft cap on attention logits; `0.0` means none. Gemma-2 uses 50.
    pub attn_logit_softcap: f32,
    /// `tanh` soft cap on the final logits; `0.0` means none. Gemma-2 uses 30.
    pub final_logit_softcap: f32,
    /// How often a **full-attention** layer appears among windowed ones.
    ///
    /// llama.cpp calls this the SWA pattern. `n` means layer `il` is windowed
    /// when `il % n < n - 1`, so the last of every `n` is global. Gemma-2 is
    /// **2** (alternating) and Gemma-3 is **6** (five local, one global).
    ///
    /// Getting it wrong windows the wrong layers, and **below the window
    /// length every pattern is identical** — which is why a short prompt
    /// cannot reveal the mistake and Gemma-3 answered plausibly and wrongly.
    /// `0` or `1` means no windowing at all.
    pub swa_pattern: u32,
    /// RoPE base for the **windowed** layers, which is not the global one.
    ///
    /// Gemma-3 trains local layers at 10,000 and global layers at 1,000,000.
    /// One base for the whole model is therefore wrong for five layers in six,
    /// and wrong RoPE is fluent nonsense rather than an error.
    pub rope_freq_base_swa: f32,
    /// Sliding-window attention width, `0` for full attention.
    ///
    /// Gemma-2 alternates full and windowed layers. Below the window every
    /// layer is effectively full, which is why a short prompt cannot reveal a
    /// missing implementation — see the length guard in `Qwen3Config::verify`.
    pub sliding_window: u32,
}

impl Qwen3Config {
    pub fn from_model(model: &Model) -> Result<Self> {
        let arch = model.architecture().to_string();
        let need = |suffix: &str| -> Result<u64> {
            model
                .arch_u64(suffix)
                .ok_or_else(|| ArchError::MissingMetadata(format!("{arch}.{suffix}")))
        };

        let n_embd = need("embedding_length")? as u32;
        // The gated delta net's shape. Its presence is what makes a layer stack
        // hybrid, and it is read from the container rather than from the arch
        // name so a variant that drops it is handled without a new name.
        let ssm = model
            .arch_u64("ssm.conv_kernel")
            .map(|conv_kernel| SsmConfig {
                conv_kernel: conv_kernel as u32,
                inner_size: model.arch_u64("ssm.inner_size").unwrap_or(0) as u32,
                state_size: model.arch_u64("ssm.state_size").unwrap_or(0) as u32,
                group_count: model.arch_u64("ssm.group_count").unwrap_or(1) as u32,
                time_step_rank: model.arch_u64("ssm.time_step_rank").unwrap_or(1) as u32,
                // Absent means "all attention"; llama.cpp defaults it to 4 and so
                // does this, because a container that carries the ssm tensors and
                // omits the interval is asking for the family default.
                full_attention_interval: model.arch_u64("full_attention_interval").unwrap_or(4)
                    as u32,
            });
        let rope_sections = model
            .arch_i64_array("rope.dimension_sections")
            .and_then(|v| {
                // Four, or it is not the mRoPE this understands.
                (v.len() >= 4).then(|| [v[0] as i32, v[1] as i32, v[2] as i32, v[3] as i32])
            });
        let n_head = need("attention.head_count")? as u32;
        // Qwen3 declares head dimension explicitly; older models imply it.
        let head_dim = model
            .arch_u64("attention.key_length")
            .unwrap_or((n_embd / n_head.max(1)) as u64) as u32;

        // A missing `ffn_gate` means one of **three** very different things.
        //
        // Phi-3 fuses gate and up into one tensor twice `n_ff` wide; StarCoder2
        // has no gate at all and its `ffn_up` is `n_ff` wide — the shape
        // separates those two. And an **MoE container has no `ffn_gate` either**,
        // because its gate is `ffn_gate_exps`, one tensor holding every expert.
        //
        // Testing the dense name alone made every all-MoE model "ungated" and
        // ran **GELU over its experts where Qwen3-MoE runs SiLU** — the Gemma
        // activation bug again, introduced by the fix for StarCoder2 and caught
        // only when `qwen3moe` was finally put through the eight-prompt sweep.
        // It answered `Paris. True or False? The capital of France is Paris.
        // True or False?` against the reference's `Paris. The capital of Italy
        // is Rome.` — a loop, which is what a subtly wrong FFN looks like.
        //
        // DeepSeek-V4-Flash escaped it because its first layers are dense, so
        // `blk.0.ffn_gate.weight` exists there. Being saved by which layer is
        // numbered zero is not a check.
        let moe_gate = model.location("blk.0.ffn_gate_exps.weight").is_some();
        let no_gate = !moe_gate && model.location("blk.0.ffn_gate.weight").is_none();
        let up_ne1 = model
            .location("blk.0.ffn_up.weight")
            .and_then(|l| l.dims.get(1).copied())
            .unwrap_or(0);
        let declared_ff = model.arch_u64("feed_forward_length").unwrap_or(0);
        let fused_gate_up = no_gate && declared_ff > 0 && up_ne1 == 2 * declared_ff;
        let ungated = no_gate && !fused_gate_up;

        Ok(Qwen3Config {
            n_layer: need("block_count")? as u32,
            n_embd,
            n_head,
            n_head_kv: model
                .arch_u64("attention.head_count_kv")
                .unwrap_or(n_head as u64) as u32,
            head_dim,
            n_ff: model.arch_u64("feed_forward_length").unwrap_or(0) as u32,
            vocab_size: model
                .arch_u64("vocab_size")
                .or_else(|| {
                    // Not every container declares vocab_size; the embedding
                    // table's own shape is authoritative when it does not.
                    model.location("token_embd.weight").map(|l| l.dims[1])
                })
                .unwrap_or(0) as u32,
            // Two spellings for one number: RMSNorm models declare
            // `..._rms_epsilon`, LayerNorm models declare `layer_norm_epsilon`.
            rms_eps: model
                .arch_f32("attention.layer_norm_rms_epsilon")
                .or_else(|| model.arch_f32("attention.layer_norm_epsilon"))
                .unwrap_or(1e-6),
            // 10000 is llama.cpp's default and what every container that omits
            // the key was trained with. The previous default of 1e6 was Qwen3's
            // *declared* value generalised into a fallback, which silently gave
            // Phi-3 the wrong rotation.
            rope_freq_base: model.arch_f32("rope.freq_base").unwrap_or(10_000.0),
            // `rope.scaling.factor` is the multiplier on the *context*, so the
            // frequency scale is its reciprocal. Storing the factor here
            // instead would invert every long-context model silently.
            rope_freq_scale: model
                .arch_f32("rope.scaling.factor")
                .filter(|f| *f > 0.0)
                .map(|f| 1.0 / f)
                .unwrap_or(1.0),
            // YaRN off unless the container asks for it by name. Applying its
            // correction to a model trained without it is not an error, just
            // subtly wrong attention at every position.
            rope_ext_factor: match model.arch_str("rope.scaling.type") {
                Some("yarn") => 1.0,
                _ => 0.0,
            },
            rope_attn_factor: model.arch_f32("rope.scaling.attn_factor").unwrap_or(1.0),
            rope_beta_fast: model.arch_f32("rope.scaling.beta_fast").unwrap_or(32.0),
            rope_beta_slow: model.arch_f32("rope.scaling.beta_slow").unwrap_or(1.0),
            rope_orig_ctx: model
                .arch_u64("rope.scaling.original_context_length")
                .unwrap_or(0) as u32,
            n_expert: model.arch_u64("expert_count").unwrap_or(0) as u32,
            n_expert_used: model.arch_u64("expert_used_count").unwrap_or(0) as u32,
            n_ff_expert: model.arch_u64("expert_feed_forward_length").unwrap_or(0) as u32,
            // Asked of the container, not of the architecture name.
            // Asked of the *first attention layer*, not of block 0: on a hybrid
            // stack block 0 is a delta net and carries no `attn_q_norm` at all,
            // so testing block 0 reported "no QK norm" for a model that has it
            // on every attention layer it does have.
            qk_norm: (0..need("block_count")? as u32)
                .find(|il| ssm.is_none_or(|s| (il + 1) % s.full_attention_interval.max(1) == 0))
                .is_some_and(|il| {
                    model
                        .location(&format!("blk.{il}.attn_q_norm.weight"))
                        .is_some()
                }),
            ssm,
            rope_sections,
            // `qwen35` names the FFN's norm after the thing before it rather
            // than after the thing it feeds.
            ffn_norm_name: if ssm.is_some() {
                "post_attention_norm"
            } else {
                "ffn_norm"
            },
            // Detected from the shape, not the arch name: a query projection
            // twice as wide as `head_dim * n_head` is carrying a gate beside it.
            fused_q_gate: (0..need("block_count")? as u32)
                .filter_map(|il| model.location(&format!("blk.{il}.attn_q.weight")))
                .next()
                .is_some_and(|l| {
                    l.dims.last().copied() == Some(u64::from(head_dim) * u64::from(n_head) * 2)
                }),
            rope_type: rope_type_for(&arch).0,
            rope_type_is_known: rope_type_for(&arch).1,
            // Asked of the container, like `qk_norm`: a fusion is a fact about
            // this file, not about what it calls itself.
            attn_bias: model.location("blk.0.attn_q.bias").is_some(),
            // A norm carrying a bias is a LayerNorm. Nothing else distinguishes
            // them, and the metadata key differs too:
            // `attention.layer_norm_epsilon` against `..._rms_epsilon`.
            //
            // A norm with **neither** weight nor bias is a LayerNorm too, and
            // only a LayerNorm: llama.cpp's non-parametric form is `LLM_NORM`,
            // and a parameterless RMSNorm exists nowhere in it. OLMo is that
            // case, and reading it as RMSNorm would skip the centring.
            layer_norm: model.location("blk.0.attn_norm.bias").is_some()
                || model.location("blk.0.attn_norm.weight").is_none(),
            norm_affine: model.location("blk.0.attn_norm.weight").is_some(),
            norm_bias: model.location("blk.0.attn_norm.bias").is_some(),
            // **One norm per block means the block is parallel.** Detected from
            // the container, not the architecture name, so a new parallel model
            // works without being listed anywhere. The guard on `attn_norm`
            // matters: OLMo has neither norm tensor, and calling that "parallel"
            // would restructure a block that is perfectly serial.
            // **"One norm" is what makes a block parallel, and the test for it
            // is the absence of a second norm -- by whatever name.** `qwen35`
            // has two: `attn_norm` in front of attention and
            // `post_attention_norm` in front of the feed-forward, which is an
            // ordinary serial block that simply does not use the usual name.
            // Testing for `ffn_norm` alone read it as parallel, and the FFN then
            // consumed `attn_norm(x)` -- the value attention had already seen.
            // The symptom was `attn_post_norm-0` coming out **identical to
            // llama.cpp's `attn_norm-0`**, which is what named the bug.
            parallel_residual: model.location("blk.0.attn_norm.weight").is_some()
                && model.location("blk.0.ffn_norm.weight").is_none()
                && model.location("blk.0.post_attention_norm.weight").is_none(),
            clamp_kqv: model.arch_f32("attention.clamp_kqv").unwrap_or(0.0),
            uses_alibi: arch == "baichuan" && need("block_count")? == 40,
            rope_freqs: model.location("rope_freqs.weight").is_some(),
            ffn_bias: model.location("blk.0.ffn_down.bias").is_some(),
            attn_out_bias: model.location("blk.0.attn_output.bias").is_some(),
            // Declared by the containers that rotate only part of each head;
            // `head_dim` for everything else, which is what was assumed before.
            n_rot: model
                .arch_u64("rope.dimension_count")
                .map(|v| v as u32)
                .unwrap_or(head_dim),
            // **Not on a hybrid model, whatever the tensor is called.**
            // `qwen35` carries `blk.0.attn_qkv.weight` and it is not attention's
            // fused QKV -- it is the delta net's input projection, `2 * key_dim
            // + value_dim` wide, convolved before it is split. Reading that as
            // Phi-3's fused QKV asked the *attention* layers for a tensor they
            // do not have, and had they had one it would have been the wrong
            // shape and silent.
            fused_qkv: ssm.is_none() && model.location("blk.0.attn_qkv.weight").is_some(),
            // A missing `ffn_gate` means one of two very different things, and
            // the **shape** is what separates them: Phi-3 fuses gate and up
            // into one tensor twice `n_ff` wide, while StarCoder2 simply has no
            // gate and its `ffn_up` is `n_ff` wide. Treating the second as the
            // first splits a tensor in half and computes SwiGLU over nonsense.
            fused_gate_up,
            // **Both, not either.** Gemma's post-norms come in a pair --
            // `post_attention_norm` *and* `post_ffw_norm` -- and it is the pair
            // that means "normalise each sub-block's output before it rejoins
            // the residual". `qwen35` carries the first name for a different
            // job: it is the norm in *front of* its feed-forward, in the
            // ordinary pre-norm position. Testing one name alone read that as
            // Gemma and then demanded a `post_ffw_norm` the container does not
            // have, refusing a loadable model.
            post_norms: model.location("blk.0.post_attention_norm.weight").is_some()
                && model.location("blk.0.post_ffw_norm.weight").is_some(),
            // Gemma scales by sqrt(n_embd) on the way in. Keyed on the
            // architecture because nothing in the weights reveals it.
            scale_embeddings: arch.starts_with("gemma"),
            // llama.cpp picks this by model size, not by a metadata key: the
            // 27B Gemmas are the exception and every other size is head_dim.
            // Sizes are identified by layer count there and here.
            attn_scale_dim: match (arch.as_str(), need("block_count")? as u32) {
                ("gemma2", 46) | ("gemma3", 62) => n_embd / n_head.max(1),
                _ => head_dim,
            },
            prescale_q: arch.starts_with("gemma"),
            // Ungated is a fact about the weights, so it overrides the
            // by-name choice — but **only when the gate is genuinely absent
            // rather than fused**. Phi-3 also has no `ffn_gate` and is very
            // much gated, so testing for the tensor alone made it ungated and
            // broke a verified architecture.
            ffn_act: if ungated {
                FfnAct::UngatedGelu
            } else {
                ffn_act_for(&arch)
            },
            attn_logit_softcap: model.arch_f32("attn_logit_softcapping").unwrap_or(0.0),
            final_logit_softcap: model.arch_f32("final_logit_softcapping").unwrap_or(0.0),
            sliding_window: model.arch_u64("attention.sliding_window").unwrap_or(0) as u32,
            // Declared by the container when it knows; otherwise by
            // architecture, because a wrong default here is silent.
            swa_pattern: model
                .arch_u64("attention.sliding_window_pattern")
                .map(|v| v as u32)
                .unwrap_or(match arch.as_str() {
                    "gemma2" => 2,
                    "gemma3" => 6,
                    _ => 0,
                }),
            rope_freq_base_swa: model.arch_f32("rope.freq_base_swa").unwrap_or(
                match arch.as_str() {
                    // Gemma-3's local layers, which the container often omits.
                    "gemma3" => 10_000.0,
                    _ => model.arch_f32("rope.freq_base").unwrap_or(10_000.0),
                },
            ),
        })
    }

    /// Whether layer `il` is a gated delta net rather than attention.
    ///
    /// llama.cpp's rule, verbatim: `(il + 1) % full_attention_interval != 0`.
    /// With an interval of 4 over 24 blocks that makes **18 recurrent layers
    /// and 6 attention layers** — 3, 7, 11, 15, 19, 23 — which is exactly the
    /// set of blocks in Qwen3.5-0.8B that carry `attn_q.weight`. Checking the
    /// rule against the container's own tensor names is how it was confirmed
    /// rather than assumed.
    pub fn is_recurrent(&self, il: u32) -> bool {
        match self.ssm {
            Some(s) => (il + 1) % s.full_attention_interval.max(1) != 0,
            None => false,
        }
    }

    /// The container's name for the norm in front of the feed-forward.
    pub fn ffn_norm(&self, il: u32) -> String {
        format!("blk.{il}.{}", self.ffn_norm_name)
    }

    pub fn is_moe(&self) -> bool {
        self.n_expert > 0 && self.n_expert_used > 0
    }

    /// The longest sequence this build can run *correctly* for this model.
    ///
    /// Gemma-2 alternates a sliding-window layer with a full-attention one.
    /// The window **is** implemented now (`stream.rs` builds a second mask and
    /// hands it to the even layers), so there is no length past which the local
    /// layers quietly see too far, and nothing left to refuse.
    ///
    /// Kept as a function rather than deleted: it is the right place for the
    /// next architecture whose limit is this implementation rather than the
    /// model, and refusing beats confident nonsense at 5000 tokens that nobody
    /// would trace back to attention.
    pub fn correct_context_limit(&self) -> usize {
        usize::MAX
    }

    /// Whether layer `il` uses the sliding window rather than full attention.
    ///
    /// One rule for every architecture with mixed attention, rather than a
    /// hardcoded parity per model. Gemma-2 (`n = 2`) gives even layers, which
    /// is what the previous `il % 2 == 0` did; Gemma-3 (`n = 6`) gives five in
    /// six, which that rule got wrong for a third of the model.
    pub fn is_swa_layer(&self, il: u32) -> bool {
        self.sliding_window > 0
            && self.swa_pattern > 1
            && il % self.swa_pattern < self.swa_pattern - 1
    }

    /// The RoPE base layer `il` was trained with.
    pub fn rope_base_for(&self, il: u32) -> f32 {
        if self.is_swa_layer(il) {
            self.rope_freq_base_swa
        } else {
            self.rope_freq_base
        }
    }

    /// Scale applied to attention scores before softmax.
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.attn_scale_dim.max(1) as f32).sqrt()
    }

    /// Apply the architecture's gate non-linearity.
    ///
    /// One place, so a new FFN site cannot quietly pick SiLU by habit. Every
    /// gated feed-forward in this crate routes through here.
    pub fn activate<'a>(&self, ctx: &'a Context, gate: &Tensor<'a>) -> Result<Tensor<'a>> {
        Ok(match self.ffn_act {
            FfnAct::Silu => ctx.silu(gate)?,
            FfnAct::Gelu | FfnAct::UngatedGelu => ctx.gelu(gate)?,
        })
    }
}

/// Weight names an architecture needs, so a missing one is reported by name
/// rather than surfacing as a null dereference deep in the graph.
pub struct Qwen3Model {
    pub config: Qwen3Config,
}

impl Qwen3Model {
    pub fn new(config: Qwen3Config) -> Self {
        Qwen3Model { config }
    }

    /// Q, K and V for one block, whether the container fuses them or not.
    ///
    /// A fused `attn_qkv` is `n_q + n_k + n_v` rows in that order. The split is
    /// along the output dimension, so each part is a contiguous run of whole
    /// rows — and since `ne0` is a multiple of the quantisation block, the
    /// views land on block boundaries and no dequantisation is needed.
    pub(crate) fn qkv_weights<'a>(
        &self,
        ctx: &'a Context,
        weights: &WeightSet<'a>,
        il: u32,
    ) -> Result<(Tensor<'a>, Tensor<'a>, Tensor<'a>)> {
        let c = &self.config;
        let get = |name: String| -> Result<&Tensor<'a>> {
            weights.get(&name).ok_or(ArchError::MissingTensor(name))
        };
        if !c.fused_qkv {
            return Ok((
                *get(format!("blk.{il}.attn_q.weight"))?,
                *get(format!("blk.{il}.attn_k.weight"))?,
                *get(format!("blk.{il}.attn_v.weight"))?,
            ));
        }
        let w = get(format!("blk.{il}.attn_qkv.weight"))?;
        let (dims, strides) = w.dims_and_strides();
        let ne0 = dims[0];
        let row = strides[1];
        let n_q = (c.n_head * c.head_dim) as i64;
        let n_kv = (c.n_head_kv * c.head_dim) as i64;
        Ok((
            ctx.view_2d(w, ne0, n_q, row, 0)?,
            ctx.view_2d(w, ne0, n_kv, row, n_q as usize * row)?,
            ctx.view_2d(w, ne0, n_kv, row, (n_q + n_kv) as usize * row)?,
        ))
    }

    /// The FFN gate and up projections, fused or separate.
    ///
    /// When fused, `ffn_up` is `2 * n_ff` rows with gate first.
    pub(crate) fn gate_up_weights<'a>(
        &self,
        ctx: &'a Context,
        weights: &WeightSet<'a>,
        il: u32,
    ) -> Result<(Tensor<'a>, Tensor<'a>)> {
        let c = &self.config;
        let get = |name: String| -> Result<&Tensor<'a>> {
            weights.get(&name).ok_or(ArchError::MissingTensor(name))
        };
        if !c.fused_gate_up {
            return Ok((
                *get(format!("blk.{il}.ffn_gate.weight"))?,
                *get(format!("blk.{il}.ffn_up.weight"))?,
            ));
        }
        let w = get(format!("blk.{il}.ffn_up.weight"))?;
        let (dims, strides) = w.dims_and_strides();
        let (ne0, half) = (dims[0], dims[1] / 2);
        let row = strides[1];
        Ok((
            ctx.view_2d(w, ne0, half, row, 0)?,
            ctx.view_2d(w, ne0, half, row, half as usize * row)?,
        ))
    }

    /// Every tensor this architecture reads, in load order.
    ///
    /// Used both to bind weights and to check a container up front — finding
    /// out at layer 37 that a tensor is missing is a poor way to learn it.
    pub fn required_tensors(&self) -> Vec<String> {
        let c = &self.config;
        let mut names = vec!["token_embd.weight".to_string()];
        // OLMo's norms have no parameters at all, so demanding these refuses a
        // container that is perfectly loadable — which is exactly how it failed
        // before: `container has no tensor "output_norm.weight"`.
        if c.norm_affine {
            names.push("output_norm.weight".to_string());
        }
        // The final norm has a bias too on a LayerNorm architecture, and it is
        // the easiest of the lot to miss: it is applied once, so a wrong final
        // norm shifts every logit by the same vector and the text stays fluent.
        if c.norm_bias {
            names.push("output_norm.bias".to_string());
        }
        // Listed so that it is **loaded**. A tensor absent from here is never
        // bound, `weights.get` returns `None`, and the graph carries on without
        // it — which for RoPE divisors is a slightly wrong rotation, not an
        // error. Asked of the container because only the extended Llama-3.x
        // models carry one.
        if c.rope_freqs {
            names.push("rope_freqs.weight".to_string());
        }
        for il in 0..c.n_layer {
            // A recurrent layer's output projection is `ssm_out`, listed below.
            if !c.is_recurrent(il) {
                names.push(format!("blk.{il}.attn_output.weight"));
            }
            if c.norm_affine {
                names.push(format!("blk.{il}.attn_norm.weight"));
                // A parallel block has ONE norm, so asking for `ffn_norm` is
                // asking for a tensor that does not exist.
                if !c.parallel_residual {
                    names.push(format!("{}.weight", c.ffn_norm(il)));
                }
            }
            // **A bias that is not listed here is never loaded**, and the graph
            // then silently skips it — `weights.get` returns `None` and the
            // shift is simply not applied. That is not a missing-tensor error;
            // it is a slightly wrong answer, which is how StableLM read after
            // LayerNorm landed but before this did.
            if c.norm_bias {
                names.push(format!("blk.{il}.attn_norm.bias"));
                if !c.parallel_residual {
                    names.push(format!("blk.{il}.ffn_norm.bias"));
                }
            }
            if c.attn_out_bias {
                names.push(format!("blk.{il}.attn_output.bias"));
            }
            if c.ffn_bias {
                for suffix in ["ffn_up.bias", "ffn_down.bias"] {
                    names.push(format!("blk.{il}.{suffix}"));
                }
            }
            if c.post_norms {
                for suffix in ["post_attention_norm.weight", "post_ffw_norm.weight"] {
                    names.push(format!("blk.{il}.{suffix}"));
                }
            }
            // **A recurrent layer has no attention tensors at all**, and an
            // attention layer on this architecture has no `ssm_*`. Listing
            // either set for the wrong layer refuses the container up front --
            // which is the good failure, but it refuses a loadable model.
            if c.is_recurrent(il) {
                for suffix in [
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_alpha.weight",
                    "ssm_beta.weight",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ] {
                    names.push(format!("blk.{il}.{suffix}"));
                }
            } else if c.fused_qkv {
                names.push(format!("blk.{il}.attn_qkv.weight"));
            } else {
                for suffix in ["attn_q.weight", "attn_k.weight", "attn_v.weight"] {
                    names.push(format!("blk.{il}.{suffix}"));
                }
                if c.attn_bias {
                    for suffix in ["attn_q.bias", "attn_k.bias", "attn_v.bias"] {
                        names.push(format!("blk.{il}.{suffix}"));
                    }
                }
            }
            // Only Qwen3 carries these. Listing them unconditionally is what
            // refused every Llama-family container up front.
            if c.qk_norm && !c.is_recurrent(il) {
                for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                    names.push(format!("blk.{il}.{suffix}"));
                }
            }
            if c.is_moe() {
                for suffix in [
                    "ffn_gate_inp.weight",
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                ] {
                    names.push(format!("blk.{il}.{suffix}"));
                }
            } else if c.fused_gate_up || c.ffn_act == FfnAct::UngatedGelu {
                // Both cases have no `ffn_gate`: Phi-3 fuses it into `ffn_up`,
                // StarCoder2 does not have one at all. Demanding it here is an
                // up-front refusal of a container that is perfectly loadable.
                for suffix in ["ffn_up.weight", "ffn_down.weight"] {
                    names.push(format!("blk.{il}.{suffix}"));
                }
            } else {
                for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                    names.push(format!("blk.{il}.{suffix}"));
                }
            }
        }
        names
    }

    /// Check the container has everything, before any weights are read.
    pub fn verify(&self, model: &Model) -> Result<()> {
        // **ALiBi is not implemented, and the container never says so.**
        // Baichuan-7B and Baichuan-13B hold the same tensor set under the same
        // architecture name; llama.cpp picks positional encoding by layer
        // count, giving the 40-layer 13B a linear attention bias and no RoPE at
        // all. The 7B is verified here. The 13B would load, rotate keys it
        // should not rotate, skip a bias it should apply, and answer fluently.
        // **Refused rather than approximated.** A parallel block feeds attention
        // and the FFN from ONE normalisation and adds both into one residual:
        //
        //   serial    x -> attn_norm -> attn -> +x -> ffn_norm -> ffn -> +x
        //   parallel  h = attn_norm(x);  x = x + attn(h) + ffn(h)
        //
        // The container is detected correctly (no `ffn_norm` means one norm per
        // block) but `forward_cached_inner` still runs the serial shape. Letting
        // it through would normalise the FFN input twice and chain the residuals
        // — arithmetic that produces fluent, wrong text rather than an error,
        // which is the failure mode this project keeps paying for. falcon,
        // gptneox and phi2 all land here.
        if self.config.uses_alibi {
            return Err(ArchError::Unimplemented(
                "this model uses ALiBi rather than RoPE (baichuan at 40 layers \
                 is the 13B), and no ALiBi path exists — it would run and be \
                 wrong rather than fail",
            ));
        }
        for name in self.required_tensors() {
            if model.location(&name).is_none() {
                return Err(ArchError::MissingTensor(name));
            }
        }
        // The output projection may be tied to the embedding table, so it is
        // optional by design rather than missing.
        Ok(())
    }

    /// Name of the output projection, which is tied to the embeddings when the
    /// container ships no separate `output.weight`.
    pub fn output_weight_name(&self, model: &Model) -> &'static str {
        if model.location("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        }
    }

    /// Attention through ggml's fused kernel.
    ///
    /// Same result as [`Self::attention`], without building the scores matrix.
    /// `mask_f16` holds the causal mask already in F16 — ggml asserts that
    /// type, and since the only values are 0 and -inf the bit patterns
    /// (`0x0000`, `0xFC00`) are written directly rather than converted.
    ///
    /// This superseded an `attention_cached` that took the same arguments and
    /// built the scores by hand. That one kept compiling, kept being documented
    /// against, and had **no callers at all** from the commit this landed in
    /// until it was deleted — a third attention implementation nothing
    /// exercised, which is the shape of hazard that let a second *forward*
    /// implementation miss four fixes.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash<'a>(
        &self,
        ctx: &'a Context,
        q: &Tensor<'a>,
        k_all: &Tensor<'a>,
        v_all: &Tensor<'a>,
        n_new: i64,
        n_total: i64,
        mask_f16: &[u8],
    ) -> Result<(Tensor<'a>, Tensor<'a>)> {
        let c = &self.config;

        // ggml wants [head_dim, n_batch, n_head] for q and [head_dim, n_kv,
        // n_head_kv] for k and v. Ours are head-major, so permute — and v is
        // NOT transposed here, which is the one place this differs from the
        // mul_mat path and would silently produce nonsense if copied across.
        let q = ctx.cont(&ctx.permute(q, [0, 2, 1, 3])?)?;
        let k = ctx.cont(&ctx.permute(k_all, [0, 2, 1, 3])?)?;
        let v = ctx.cont(&ctx.permute(v_all, [0, 2, 1, 3])?)?;

        // **The mask tensor is returned, not written here.** On a device it has
        // no memory until the whole context is realized, and realizing cannot
        // happen until the graph is finished -- which is after this function
        // returns. Writing it here segfaulted at layer 0 the moment the device
        // path reached attention. The caller writes it alongside q, k and v.
        let mask = ctx.new_typed_2d(chaos_gguf::GgmlType(1), n_total, n_new)?;
        let _ = mask_f16;

        // [head_dim, n_head, n_new], already permuted for the reshape.
        // See `prescale_q`: same algebra, different rounding, and the rounding
        // is what decides Gemma-2's first token.
        let (q, scale) = if c.prescale_q {
            (ctx.scale(&q, c.attn_scale())?, 1.0)
        } else {
            (q, c.attn_scale())
        };
        let out = ctx.flash_attn_ext(&q, &k, &v, &mask, scale, c.attn_logit_softcap)?;
        Ok((
            ctx.reshape_2d(&ctx.cont(&out)?, (c.head_dim * c.n_head) as i64, n_new)?,
            mask,
        ))
    }

    /// Normalise then scale by a learned weight — the pattern every norm
    /// in this architecture uses. No bias; see [`norm_named`](Self::norm_named).
    pub fn norm_scaled<'a>(
        &self,
        ctx: &'a Context,
        x: &Tensor<'a>,
        weight: &Tensor<'a>,
    ) -> Result<Tensor<'a>> {
        self.rms_norm_mul(ctx, x, weight)
    }

    /// A norm named by its tensor prefix, picking up `{prefix}.bias` when the
    /// container carries one.
    ///
    /// Taking the prefix rather than a resolved tensor is what lets one call
    /// site serve both kinds: a LayerNorm model's bias is found, an RMSNorm
    /// model's absent bias is simply not applied, and neither needs the caller
    /// to know which it is.
    pub fn norm_named<'a>(
        &self,
        ctx: &'a Context,
        weights: &WeightSet<'a>,
        x: &Tensor<'a>,
        prefix: &str,
    ) -> Result<Tensor<'a>> {
        let weight = weights.get(&format!("{prefix}.weight"));
        // Absent is only allowed where the config already established that this
        // architecture's norms have no parameters. Otherwise it is the missing
        // tensor it looks like — silently dropping a scale is exactly the class
        // of bug that made StableLM read "almost right".
        if weight.is_none() && self.config.norm_affine {
            return Err(ArchError::MissingTensor(format!("{prefix}.weight")));
        }
        let bias = weights.get(&format!("{prefix}.bias"));
        self.norm_mul(ctx, x, weight, bias)
    }

    /// Add `{name}.bias` to `x` when the container has one.
    ///
    /// Used for the projection biases StarCoder2 carries on `attn_output`,
    /// `ffn_up` and `ffn_down`. Absent is the common case and not an error.
    pub fn add_bias<'a>(
        &self,
        ctx: &'a Context,
        weights: &WeightSet<'a>,
        x: Tensor<'a>,
        name: &str,
    ) -> Result<Tensor<'a>> {
        match weights.get(&format!("{name}.bias")) {
            Some(b) => Ok(ctx.add(&x, b)?),
            None => Ok(x),
        }
    }

    fn rms_norm_mul<'a>(
        &self,
        ctx: &'a Context,
        x: &Tensor<'a>,
        weight: &Tensor<'a>,
    ) -> Result<Tensor<'a>> {
        self.norm_mul(ctx, x, Some(weight), None)
    }

    /// Normalise, scale, and shift when the architecture has a shift.
    ///
    /// The kind comes from the config rather than the name, and the bias is
    /// applied only when the container carries one — so an RMSNorm model that
    /// gained a bias, or a LayerNorm model that lost one, is a shape error at
    /// load rather than a silently different function.
    fn norm_mul<'a>(
        &self,
        ctx: &'a Context,
        x: &Tensor<'a>,
        weight: Option<&Tensor<'a>>,
        bias: Option<&Tensor<'a>>,
    ) -> Result<Tensor<'a>> {
        let normed = if self.config.layer_norm {
            ctx.norm(x, self.config.rms_eps)?
        } else {
            ctx.rms_norm(x, self.config.rms_eps)?
        };
        // Both are optional and independently so, which is the whole point:
        // OLMo has neither, StableLM has both, Qwen has weight only. Three
        // shapes, one expression.
        let scaled = match weight {
            Some(w) => ctx.mul(&normed, w)?,
            None => normed,
        };
        match bias {
            Some(b) => Ok(ctx.add(&scaled, b)?),
            None => Ok(scaled),
        }
    }

    /// Gated feed-forward: `down(act(gate(x)) * up(x))`, `act` per architecture.
    pub(crate) fn dense_ffn<'a>(
        &self,
        ctx: &'a Context,
        weights: &WeightSet<'a>,
        x: &Tensor<'a>,
        il: u32,
    ) -> Result<Tensor<'a>> {
        let get = |name: String| -> Result<&Tensor<'a>> {
            weights.get(&name).ok_or(ArchError::MissingTensor(name))
        };
        // Ungated: `down(gelu(up(x)))`. There is no `ffn_gate` tensor at all,
        // so the gated path below would ask for one that does not exist -- and
        // the biases StarCoder2 carries on `ffn_up` and `ffn_down` are applied
        // where llama.cpp applies them, before and after the activation.
        if self.config.ffn_act == FfnAct::UngatedGelu {
            let up = ctx.mul_mat(get(format!("blk.{il}.ffn_up.weight"))?, x)?;
            let up = self.add_bias(ctx, weights, up, &format!("blk.{il}.ffn_up"))?;
            let activated = self.config.activate(ctx, &up)?;
            let down = ctx.mul_mat(get(format!("blk.{il}.ffn_down.weight"))?, &activated)?;
            return self.add_bias(ctx, weights, down, &format!("blk.{il}.ffn_down"));
        }
        let (gate_w, up_w) = self.gate_up_weights(ctx, weights, il)?;
        let gate = ctx.mul_mat(&gate_w, x)?;
        let up = ctx.mul_mat(&up_w, x)?;
        let activated = ctx.mul(&self.config.activate(ctx, &gate)?, &up)?;
        let down = ctx.mul_mat(get(format!("blk.{il}.ffn_down.weight"))?, &activated)?;
        self.add_bias(ctx, weights, down, &format!("blk.{il}.ffn_down"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_config() -> Qwen3Config {
        Qwen3Config {
            n_layer: 2,
            n_embd: 64,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 16,
            n_ff: 128,
            vocab_size: 100,
            rms_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            rope_freq_scale: 1.0,
            rope_ext_factor: 0.0,
            rope_attn_factor: 1.0,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            rope_orig_ctx: 0,
            n_expert: 0,
            n_expert_used: 0,
            n_ff_expert: 0,
            qk_norm: true,
            rope_type: ROPE_TYPE_NEOX,
            rope_type_is_known: true,
            attn_bias: false,
            layer_norm: false,
            norm_affine: true,
            norm_bias: false,
            clamp_kqv: 0.0,
            uses_alibi: false,
            rope_freqs: false,
            ffn_bias: false,
            attn_out_bias: false,
            n_rot: 16,
            fused_qkv: false,
            fused_gate_up: false,
            parallel_residual: false,
            post_norms: false,
            ssm: None,
            rope_sections: None,
            ffn_norm_name: "ffn_norm",
            fused_q_gate: false,
            scale_embeddings: false,
            attn_scale_dim: 16,
            prescale_q: false,
            ffn_act: FfnAct::Silu,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            sliding_window: 0,
            swa_pattern: 0,
            rope_freq_base_swa: 10_000.0,
        }
    }

    /// A non-parametric-norm container asks for no norm weights at all.
    ///
    /// OLMo's norms are `build_norm(x, NULL, NULL, LLM_NORM)` there, so
    /// `output_norm.weight` does not exist and demanding it refused a container
    /// that runs — the first thing OLMo did here was fail on that exact name.
    #[test]
    fn a_non_parametric_norm_is_not_a_missing_tensor() {
        let c = Qwen3Config {
            norm_affine: false,
            norm_bias: false,
            layer_norm: true,
            qk_norm: false,
            ..dense_config()
        };
        let names = Qwen3Model::new(c).required_tensors();
        for n in &names {
            assert!(
                !n.ends_with("_norm.weight") && !n.ends_with("_norm.bias"),
                "asked for {n} from a model whose norms have no parameters"
            );
        }
        // Everything else is still demanded, so this is not a blanket weakening.
        assert!(names.contains(&"blk.0.attn_output.weight".to_string()));
        assert!(names.contains(&"token_embd.weight".to_string()));

        // And an affine architecture still gets the full list.
        let affine = Qwen3Model::new(dense_config_no_qk_norm()).required_tensors();
        assert!(affine.contains(&"output_norm.weight".to_string()));
        assert!(affine.contains(&"blk.1.ffn_norm.weight".to_string()));
    }

    /// `rope_freqs.weight` has to be **listed** to be loaded.
    ///
    /// Llama-3.1/3.2/3.3 carry their RoPE scaling as this tensor and nothing in
    /// the metadata says so — it reports `rope scaling = linear, freq_scale = 1`
    /// either way. Unlisted, `weights.get` returns `None`, the rotation is
    /// quietly the un-extended one, and Llama-3.2-1B scored 3 of 8 on parity
    /// with four prompts blaming the reference for the disagreement.
    #[test]
    fn rope_freqs_is_required_when_the_container_has_one() {
        let with = Qwen3Model::new(Qwen3Config {
            rope_freqs: true,
            ..dense_config()
        })
        .required_tensors();
        assert!(with.contains(&"rope_freqs.weight".to_string()));

        let without = Qwen3Model::new(dense_config()).required_tensors();
        assert!(
            !without.contains(&"rope_freqs.weight".to_string()),
            "demanding it would refuse every model that is not an extended Llama-3"
        );
    }

    /// The same model without per-head QK norm — a Llama-family container.
    fn dense_config_no_qk_norm() -> Qwen3Config {
        Qwen3Config {
            qk_norm: false,
            ..dense_config()
        }
    }

    #[test]
    fn dense_and_moe_are_distinguished_by_expert_counts() {
        let dense = dense_config();
        assert!(!dense.is_moe());

        let moe = Qwen3Config {
            n_expert: 128,
            n_expert_used: 8,
            n_ff_expert: 768,
            ..dense_config()
        };
        assert!(moe.is_moe());
    }

    #[test]
    fn attention_scale_is_one_over_sqrt_head_dim() {
        // Not n_embd -- a common and silently wrong substitution.
        let c = dense_config();
        assert!(
            (c.attn_scale() - 0.25).abs() < 1e-6,
            "got {}",
            c.attn_scale()
        );
    }

    #[test]
    fn the_gemma_family_uses_gelu_and_everyone_else_silu() {
        // Nothing in a container records this: a GELU model and a SiLU model
        // hold byte-identical tensor sets, so the wrong choice is not a
        // missing tensor, not a shape error, and not a crash -- it is a model
        // that still answers in fluent English and disagrees with llama.cpp
        // from the first token. `gemma2` shipped as "verified" in exactly that
        // state; the reference said "a) Paris b) Lyon" where we said "Paris".
        for arch in ["gemma", "gemma2", "gemma3", "gemma3n"] {
            assert_eq!(ffn_act_for(arch), FfnAct::Gelu, "{arch}");
        }
        for arch in ["llama", "qwen2", "qwen3", "qwen3moe", "phi3", "deepseek4"] {
            assert_eq!(ffn_act_for(arch), FfnAct::Silu, "{arch}");
        }
    }

    #[test]
    fn only_the_27b_gemmas_scale_by_n_embd_over_n_head() {
        // llama.cpp picks the attention scale by model size, and the two
        // formulas COINCIDE at every size but 27B: gemma-3-1b has head_dim 256
        // and n_embd/n_head = 1152/4 = 288, but its scale is 1/sqrt(256).
        // A check that passed on the 1B would still have been wrong at 27B,
        // which is why the rule is encoded rather than the observation.
        let dim = |arch: &str, n_layer: u32, n_embd: u32, n_head: u32, head_dim: u32| match (
            arch, n_layer,
        ) {
            ("gemma2", 46) | ("gemma3", 62) => n_embd / n_head.max(1),
            _ => head_dim,
        };
        // gemma-2-2b and gemma-3-1b: head_dim wins even though it differs
        // from n_embd/n_head.
        assert_eq!(dim("gemma2", 26, 2304, 8, 256), 256);
        assert_eq!(dim("gemma3", 26, 1152, 4, 256), 256);
        // gemma-2-27b: 4608/32 = 144 against a head_dim of 128.
        assert_eq!(dim("gemma2", 46, 4608, 32, 128), 144);
        assert_eq!(dim("gemma3", 62, 5376, 32, 128), 168);
        // Everything else, at every size.
        assert_eq!(dim("llama", 46, 4096, 32, 128), 128);
    }

    #[test]
    fn llama_gets_norm_rope_and_qwen_gets_neox() {
        // The two conventions run without error on each other's layout and
        // produce plausible text either way, so nothing downstream can catch a
        // mix-up. It has to be right here.
        assert_eq!(rope_type_for("llama"), (ROPE_TYPE_NORM, true));
        assert_eq!(rope_type_for("mistral"), (ROPE_TYPE_NORM, true));
        assert_eq!(rope_type_for("qwen3"), (ROPE_TYPE_NEOX, true));
        assert_eq!(rope_type_for("qwen3moe"), (ROPE_TYPE_NEOX, true));
        assert_eq!(rope_type_for("gemma2"), (ROPE_TYPE_NEOX, true));

        // `olmo` sat in the NeoX arm claiming `known = true` while llama.cpp
        // lists `LLM_ARCH_OLMO` in the NORM branch beside `LLM_ARCH_LLAMA` —
        // a guess wearing the label of a checked fact, since nothing had ever
        // been run against the reference for it. These four are now diffed at
        // eight prompts each.
        assert_eq!(rope_type_for("olmo"), (ROPE_TYPE_NORM, true));
        // Read out of llama.cpp's `llama_model_rope_type`: MINICPM sits in the
        // NORM arm beside LLAMA, while FALCON, PHI2 and GPTNEOX are in the
        // NEOX arm below it.
        assert_eq!(rope_type_for("minicpm"), (ROPE_TYPE_NORM, true));
        assert_eq!(rope_type_for("falcon"), (ROPE_TYPE_NEOX, true));
        assert_eq!(rope_type_for("phi2"), (ROPE_TYPE_NEOX, true));
        assert_eq!(rope_type_for("gptneox"), (ROPE_TYPE_NEOX, true));
        // ALiBi models: llama.cpp says ROPE_TYPE_NONE, so they must stay
        // UNKNOWN here rather than be given a rotation they do not use.
        assert!(!rope_type_for("mpt").1);
        // **`qwen35` is implemented now** -- Qwen3.5, 3.6 and 3.8 all report it,
        // and its 16 attention layers rotate through `ggml_rope_multi` with the
        // container's four `rope.dimension_sections`. Diffed against llama.cpp
        // layer by layer on Qwen3.5-0.8B, so this is a checked fact.
        assert_eq!(rope_type_for("qwen35"), (ROPE_TYPE_IMROPE, true));
        // `qwen35moe` still is not: no MoE container of this family has been
        // run here, and its routed path is untested. Identical in llama.cpp's
        // source is not the same as checked.
        assert!(!rope_type_for("qwen35moe").1);
        assert!(!rope_type_for("bloom").1);
        assert_eq!(rope_type_for("internlm2"), (ROPE_TYPE_NORM, true));
        assert_eq!(rope_type_for("baichuan"), (ROPE_TYPE_NORM, true));
        assert_eq!(rope_type_for("starcoder2"), (ROPE_TYPE_NEOX, true));

        // An architecture nobody has checked gets a default AND is flagged as a
        // guess, so the runner can say so rather than quietly being wrong.
        let (ty, known) = rope_type_for("some-model-invented-next-year");
        assert_eq!(ty, ROPE_TYPE_NEOX);
        assert!(!known, "an unknown architecture must not claim to be known");
    }

    #[test]
    fn a_llama_family_container_is_not_asked_for_qk_norm() {
        // Requiring `attn_q_norm` unconditionally is what refused Llama,
        // Mistral, Qwen2, Gemma and Phi *before a byte was read*, because the
        // container check runs against `required_tensors` up front. The tensors
        // do not exist in those files, so asking for them is not a strictness
        // choice — it is a false negative on every one of them.
        let m = Qwen3Model::new(dense_config_no_qk_norm());
        let names = m.required_tensors();
        assert!(
            !names.iter().any(|n| n.contains("attn_q_norm")),
            "a model without QK norm must not be asked for it: {names:?}"
        );
        assert!(!names.iter().any(|n| n.contains("attn_k_norm")));
        // Everything else is still required — this must not become a blanket
        // relaxation that lets a genuinely incomplete container through.
        for il in 0..2 {
            for suffix in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
            ] {
                assert!(
                    names.contains(&format!("blk.{il}.{suffix}")),
                    "blk.{il}.{suffix} is still required"
                );
            }
        }

        // And Qwen3 itself must still demand them, or the detection is useless.
        let qwen = Qwen3Model::new(dense_config());
        assert!(qwen
            .required_tensors()
            .contains(&"blk.0.attn_q_norm.weight".to_string()));
    }

    /// **An MoE model is not an ungated one.**
    ///
    /// `ffn_gate` is absent from an MoE container because the gate is
    /// `ffn_gate_exps`, and reading that absence as "no gate" gave every
    /// all-MoE model `UngatedGelu`. Qwen3-30B-A3B then ran **GELU over its
    /// experts where the reference runs SiLU** and looped —
    /// `Paris. True or False? The capital of France is Paris. True or False?`
    /// against `Paris. The capital of Italy is Rome.`
    ///
    /// The activation is not in `required_tensors`, so no test that checks
    /// tensor names can catch this. It has to be asserted directly.
    #[test]
    fn an_moe_container_is_gated_even_though_ffn_gate_is_absent() {
        for (arch, expected) in [
            ("qwen3moe", FfnAct::Silu),
            ("qwen2moe", FfnAct::Silu),
            ("deepseek4", FfnAct::Silu),
        ] {
            assert_eq!(
                ffn_act_for(arch),
                expected,
                "{arch} must not be reclassified by the ungated-FFN detection"
            );
        }
        // The MoE config the streaming path actually runs.
        let moe = Qwen3Model::new(Qwen3Config {
            n_expert: 128,
            n_expert_used: 8,
            ..dense_config()
        });
        assert!(moe.config.is_moe());
        assert_eq!(
            moe.config.ffn_act,
            FfnAct::Silu,
            "an MoE config must never carry UngatedGelu"
        );
    }

    #[test]
    fn required_tensors_cover_every_layer_and_match_the_variant() {
        let dense = Qwen3Model::new(dense_config());
        let names = dense.required_tensors();
        assert!(names.contains(&"token_embd.weight".to_string()));
        assert!(names.contains(&"blk.1.ffn_gate.weight".to_string()));
        assert!(
            !names.iter().any(|n| n.contains("_exps")),
            "dense model must not require expert tensors"
        );

        let moe = Qwen3Model::new(Qwen3Config {
            n_expert: 128,
            n_expert_used: 8,
            ..dense_config()
        });
        let names = moe.required_tensors();
        assert!(names.contains(&"blk.0.ffn_gate_exps.weight".to_string()));
        assert!(
            !names.iter().any(|n| n.ends_with("ffn_gate.weight")),
            "MoE model must not require dense FFN tensors"
        );
    }

    #[test]
    fn every_layer_is_accounted_for() {
        let m = Qwen3Model::new(dense_config());
        let names = m.required_tensors();
        for il in 0..2 {
            assert!(names.contains(&format!("blk.{il}.attn_q_norm.weight")));
            assert!(names.contains(&format!("blk.{il}.attn_k_norm.weight")));
        }
    }

    #[test]
    fn only_checked_architectures_are_called_verified() {
        // The list is a claim about what has been RUN and its output read, not
        // about what loads. Before its post-norms and soft-capping were
        // implemented, Gemma-2 loaded through this path with no error at all
        // and answered "The capital of France is" with "himselff" — which is
        // exactly why loading is not evidence of anything.
        for arch in [
            "deepseek4",
            "gemma",
            "gemma2",
            "gemma3",
            "llama",
            "phi3",
            "qwen3",
        ] {
            assert!(architecture_is_verified(arch), "{arch} should be verified");
        }
        // `gemma` (v1) JOINED the list on 2026-08-16, after gemma-1.1-2b-it
        // came back 8 of 8 against llama.cpp twice. It was absent before that
        // for the right reason -- close to `gemma2` but not identical, and
        // never run. `gemma3n` is still absent: a different model despite the
        // name.
        //
        // **`qwen3moe` is absent because it was REMOVED**, not because nobody
        // tried. It sat here through a diff it had never been given, and the
        // first eight-prompt run returned 1 FAIL + 6 unstable. Re-run once the
        // harness could probe `-b 1`, it came back 0 FAIL + **6 unstable** —
        // the near-tie explanation survived and the divergence did not shrink.
        // This assertion is what would have caught someone quietly putting it
        // back on the strength of the FAIL turning into a softer word.
        for arch in ["gemma3n", "qwen3moe", "falcon", "mamba", "something-new"] {
            assert!(
                !architecture_is_verified(arch),
                "{arch} has not been checked and must not claim to be"
            );
        }
    }
}

#[cfg(test)]
mod catalogue_agreement {
    /// **The catalogue's idea of what runs must be this engine's idea.**
    ///
    /// `chaos_model::catalogue::RUNNABLE_ARCHS` exists because the app and
    /// `chaos-pull` decide whether to offer a download *before* the engine is
    /// involved, and neither depends on this crate. That is a second copy of a
    /// fact, so this test — which lives on the side that owns the fact — is what
    /// keeps it from drifting. A model offered as runnable that this engine
    /// refuses is a download button that wastes 22 GB.
    #[test]
    fn the_catalogue_agrees_with_the_verified_list() {
        use chaos_model::catalogue::{why_not_runnable, CATALOGUE, RUNNABLE_ARCHS};
        let verified: std::collections::HashSet<&str> =
            super::VERIFIED_ARCHITECTURES.iter().copied().collect();
        let runnable: std::collections::HashSet<&str> = RUNNABLE_ARCHS.iter().copied().collect();
        assert_eq!(
            runnable, verified,
            "RUNNABLE_ARCHS and VERIFIED_ARCHITECTURES disagree; \
             the difference is either a model offered that cannot run, or one \
             withheld that can"
        );
        // And every catalogue entry's verdict must follow from that list.
        for e in CATALOGUE {
            assert_eq!(
                why_not_runnable(e.arch).is_none(),
                verified.contains(&e.arch),
                "{} ({}) is offered inconsistently with the verified list",
                e.name,
                e.arch
            );
        }
    }
}
