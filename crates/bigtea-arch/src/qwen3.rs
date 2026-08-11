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

use bigtea_ggml::{Context, RopeParams, Tensor, WeightSet};
use bigtea_model::Model;

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
fn rope_type_for(arch: &str) -> (i32, bool) {
    match arch {
        "llama" | "llama4" | "baichuan" | "deci" | "mistral" => (ROPE_TYPE_NORM, true),
        "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" | "phi3" | "gemma" | "gemma2" | "gemma3"
        | "stablelm" | "olmo" | "starcoder2" => (ROPE_TYPE_NEOX, true),
        _ => (ROPE_TYPE_NEOX, false),
    }
}

/// Architectures this build has actually been run against and checked.
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
pub const VERIFIED_ARCHITECTURES: &[&str] = &[
    "deepseek4",
    "gemma2",
    "llama",
    "phi3",
    "qwen2",
    "qwen3",
    "qwen3moe",
];

/// Whether this build has been run against `arch` and had its output checked.
pub fn architecture_is_verified(arch: &str) -> bool {
    VERIFIED_ARCHITECTURES.contains(&arch)
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
    pub fused_qkv: bool,
    /// The FFN gate and up projections share one `ffn_up` tensor.
    ///
    /// Also Phi-3. `ffn_up` is `2 * n_ff` rows: gate first, then up.
    pub fused_gate_up: bool,
    /// Gemma normalises again *after* attention and *after* the FFN, on top of
    /// the pre-norms every other architecture here uses.
    ///
    /// Detected from the container. Omitting these does not fail — the residual
    /// stream simply drifts, and the model answers fluently and wrongly.
    pub post_norms: bool,
    /// Multiply the embeddings by `sqrt(n_embd)` on the way in. Gemma only.
    pub scale_embeddings: bool,
    /// `tanh` soft cap on attention logits; `0.0` means none. Gemma-2 uses 50.
    pub attn_logit_softcap: f32,
    /// `tanh` soft cap on the final logits; `0.0` means none. Gemma-2 uses 30.
    pub final_logit_softcap: f32,
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
        let n_head = need("attention.head_count")? as u32;
        // Qwen3 declares head dimension explicitly; older models imply it.
        let head_dim = model
            .arch_u64("attention.key_length")
            .unwrap_or((n_embd / n_head.max(1)) as u64) as u32;

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
            rms_eps: model
                .arch_f32("attention.layer_norm_rms_epsilon")
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
            qk_norm: model.location("blk.0.attn_q_norm.weight").is_some(),
            rope_type: rope_type_for(&arch).0,
            rope_type_is_known: rope_type_for(&arch).1,
            // Asked of the container, like `qk_norm`: a fusion is a fact about
            // this file, not about what it calls itself.
            attn_bias: model.location("blk.0.attn_q.bias").is_some(),
            fused_qkv: model.location("blk.0.attn_qkv.weight").is_some(),
            fused_gate_up: model.location("blk.0.ffn_gate.weight").is_none()
                && model.location("blk.0.ffn_up.weight").is_some(),
            post_norms: model.location("blk.0.post_attention_norm.weight").is_some(),
            // Gemma scales by sqrt(n_embd) on the way in. Keyed on the
            // architecture because nothing in the weights reveals it.
            scale_embeddings: arch.starts_with("gemma"),
            attn_logit_softcap: model.arch_f32("attn_logit_softcapping").unwrap_or(0.0),
            final_logit_softcap: model.arch_f32("final_logit_softcapping").unwrap_or(0.0),
            sliding_window: model.arch_u64("attention.sliding_window").unwrap_or(0) as u32,
        })
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

    /// Scale applied to attention scores before softmax.
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
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
        let mut names = vec![
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
        ];
        for il in 0..c.n_layer {
            for suffix in ["attn_norm.weight", "attn_output.weight", "ffn_norm.weight"] {
                names.push(format!("blk.{il}.{suffix}"));
            }
            if c.post_norms {
                for suffix in ["post_attention_norm.weight", "post_ffw_norm.weight"] {
                    names.push(format!("blk.{il}.{suffix}"));
                }
            }
            if c.fused_qkv {
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
            if c.qk_norm {
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
            } else if c.fused_gate_up {
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

    /// Build the forward graph for one batch of tokens and return the logits
    /// tensor. Nothing is computed until [`Context::compute`] runs.
    ///
    /// `positions` must hold each token's absolute position; RoPE depends on
    /// it, and off-by-one here degrades output subtly rather than obviously.
    pub fn build_graph<'a>(
        &self,
        ctx: &'a Context,
        weights: &WeightSet<'a>,
        tokens: &Tensor<'a>,
        positions: &Tensor<'a>,
        n_tokens: i64,
    ) -> Result<Tensor<'a>> {
        let c = &self.config;
        let get = |name: &str| -> Result<&Tensor<'a>> {
            weights
                .get(name)
                .ok_or_else(|| ArchError::MissingTensor(name.to_string()))
        };
        let rope = RopeParams {
            freq_base: c.rope_freq_base,
            ..RopeParams::default()
        };

        // Token ids -> embedding vectors.
        let mut cur = ctx.get_rows(get("token_embd.weight")?, tokens)?;

        for il in 0..c.n_layer {
            let residual = cur;

            // --- attention ---------------------------------------------------
            let normed =
                self.rms_norm_mul(ctx, &cur, get(&format!("blk.{il}.attn_norm.weight"))?)?;

            let (qw, kw, vw) = self.qkv_weights(ctx, weights, il)?;
            let q = ctx.mul_mat(&qw, &normed)?;
            let k = ctx.mul_mat(&kw, &normed)?;
            let v = ctx.mul_mat(&vw, &normed)?;

            // Split into heads before normalising: Qwen3 normalises each head
            // separately, with a weight of head_dim rather than n_embd.
            let q = ctx.reshape_3d(&q, c.head_dim as i64, c.n_head as i64, n_tokens)?;
            let k = ctx.reshape_3d(&k, c.head_dim as i64, c.n_head_kv as i64, n_tokens)?;

            // Absent on Llama, Mistral, Qwen2, Gemma and Phi: those normalise
            // once before the projections and not again per head.
            let (q, k) = if c.qk_norm {
                (
                    self.rms_norm_mul(ctx, &q, get(&format!("blk.{il}.attn_q_norm.weight"))?)?,
                    self.rms_norm_mul(ctx, &k, get(&format!("blk.{il}.attn_k_norm.weight"))?)?,
                )
            } else {
                (q, k)
            };

            let q = ctx.rope_ext(&q, positions, None, c.head_dim as i32, c.rope_type, 0, rope)?;
            let k = ctx.rope_ext(&k, positions, None, c.head_dim as i32, c.rope_type, 0, rope)?;

            let attn = self.attention(ctx, &q, &k, &v, n_tokens)?;
            let attn = ctx.mul_mat(get(&format!("blk.{il}.attn_output.weight"))?, &attn)?;

            let ffn_input = ctx.add(&attn, &residual)?;

            // --- feed forward ------------------------------------------------
            let normed =
                self.rms_norm_mul(ctx, &ffn_input, get(&format!("blk.{il}.ffn_norm.weight"))?)?;

            let ffn_out = if c.is_moe() {
                self.moe_ffn(ctx, weights, &normed, il, n_tokens)?
            } else {
                self.dense_ffn(ctx, weights, &normed, il)?
            };

            cur = ctx.add(&ffn_out, &ffn_input)?;
        }

        // Only the final position predicts the next token, so the output
        // projection is taken on that row alone.
        //
        // Projecting all of them is what a naive graph does, and it is enormous:
        // the vocabulary is 151936 wide, so a 651-token prompt costs
        // `651 x 2560 x 151936` = **253 GFLOP** and 395 MB of logits, of which
        // one row is used. It also made the arena quadratic-looking when the
        // real driver was this term, and a 651-token prompt aborted with
        // `GGML_ASSERT` on a 2 GiB arena.
        let cur = ctx.view_2d(
            &cur,
            c.n_embd as i64,
            1,
            c.n_embd as usize * std::mem::size_of::<f32>(),
            (n_tokens - 1) as usize * c.n_embd as usize * std::mem::size_of::<f32>(),
        )?;
        let cur = self.rms_norm_mul(ctx, &cur, get("output_norm.weight")?)?;
        // Output projection; tied to the embedding table when absent.
        let out_name = if weights.get("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        Ok(ctx.mul_mat(get(out_name)?, &cur)?)
    }

    /// One layer's attention, from the pre-norm through the output projection.
    ///
    /// Shared by the single-graph path and the streaming path so the
    /// architecture is defined once; two copies would drift.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_block<'a>(
        &self,
        ctx: &'a Context,
        weights: &WeightSet<'a>,
        x: &Tensor<'a>,
        positions: &Tensor<'a>,
        n_tokens: i64,
        il: u32,
        rope: RopeParams,
        rope_type: i32,
    ) -> Result<Tensor<'a>> {
        let c = &self.config;
        let get = |name: String| -> Result<&Tensor<'a>> {
            weights.get(&name).ok_or(ArchError::MissingTensor(name))
        };

        let normed = self.rms_norm_mul(ctx, x, get(format!("blk.{il}.attn_norm.weight"))?)?;

        let (qw, kw, vw) = self.qkv_weights(ctx, weights, il)?;
        let q = ctx.mul_mat(&qw, &normed)?;
        let k = ctx.mul_mat(&kw, &normed)?;
        let v = ctx.mul_mat(&vw, &normed)?;

        let q = ctx.reshape_3d(&q, c.head_dim as i64, c.n_head as i64, n_tokens)?;
        let k = ctx.reshape_3d(&k, c.head_dim as i64, c.n_head_kv as i64, n_tokens)?;

        let (q, k) = if c.qk_norm {
            (
                self.rms_norm_mul(ctx, &q, get(format!("blk.{il}.attn_q_norm.weight"))?)?,
                self.rms_norm_mul(ctx, &k, get(format!("blk.{il}.attn_k_norm.weight"))?)?,
            )
        } else {
            (q, k)
        };

        let q = ctx.rope_ext(&q, positions, None, c.head_dim as i32, rope_type, 0, rope)?;
        let k = ctx.rope_ext(&k, positions, None, c.head_dim as i32, rope_type, 0, rope)?;

        let attn = self.attention(ctx, &q, &k, &v, n_tokens)?;
        Ok(ctx.mul_mat(get(format!("blk.{il}.attn_output.weight"))?, &attn)?)
    }

    /// Attention through ggml's fused kernel.
    ///
    /// Same result as [`Self::attention_cached`], without building the scores
    /// matrix. `mask_f16` holds the causal mask already in F16 — ggml asserts
    /// that type, and since the only values are 0 and -inf the bit patterns
    /// (`0x0000`, `0xFC00`) are written directly rather than converted.
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
    ) -> Result<Tensor<'a>> {
        let c = &self.config;

        // ggml wants [head_dim, n_batch, n_head] for q and [head_dim, n_kv,
        // n_head_kv] for k and v. Ours are head-major, so permute — and v is
        // NOT transposed here, which is the one place this differs from the
        // mul_mat path and would silently produce nonsense if copied across.
        let q = ctx.cont(&ctx.permute(q, [0, 2, 1, 3])?)?;
        let k = ctx.cont(&ctx.permute(k_all, [0, 2, 1, 3])?)?;
        let v = ctx.cont(&ctx.permute(v_all, [0, 2, 1, 3])?)?;

        let mask = ctx.new_typed_2d(bigtea_gguf::GgmlType(1), n_total, n_new)?;
        mask.set_bytes(mask_f16)?;

        // [head_dim, n_head, n_new], already permuted for the reshape.
        let out = ctx.flash_attn_ext(&q, &k, &v, &mask, c.attn_scale(), c.attn_logit_softcap)?;
        Ok(ctx.reshape_2d(&ctx.cont(&out)?, (c.head_dim * c.n_head) as i64, n_new)?)
    }

    /// Attention where K and V come from a cache covering the whole history.
    ///
    /// The mask must be offset by the query's absolute position: query `i` may
    /// attend to keys up to `pos_start + i`, not up to `i`. Getting that wrong
    /// lets a token see its own future during incremental decoding — the same
    /// failure as omitting the mask entirely, but only visible after the first
    /// generated token. The caller builds it; see `forward_cached`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_cached<'a>(
        &self,
        ctx: &'a Context,
        q: &Tensor<'a>,
        k_all: &Tensor<'a>,
        v_all: &Tensor<'a>,
        n_new: i64,
        n_total: i64,
        mask_data: &[f32],
    ) -> Result<Tensor<'a>> {
        let c = &self.config;

        let q = ctx.cont(&ctx.permute(q, [0, 2, 1, 3])?)?;
        let k = ctx.cont(&ctx.permute(k_all, [0, 2, 1, 3])?)?;

        // [n_total, n_new, n_head]
        let scores = ctx.mul_mat(&k, &q)?;

        // The mask depends only on positions, so it is identical for every
        // layer and is built once per call by the caller. Rebuilding it here
        // cost an n_total * n_new scalar loop and a copy of the same size,
        // 48 times per block.
        let mask = ctx.new_f32_2d(n_total, n_new)?;
        mask.set_f32(mask_data)?;
        let probs = ctx.soft_max_ext(&scores, Some(&mask), c.attn_scale(), 0.0)?;

        let v = ctx.cont(&ctx.transpose(&ctx.permute(v_all, [0, 2, 1, 3])?)?)?;
        let out = ctx.mul_mat(&v, &probs)?;
        let out = ctx.cont(&ctx.permute(&out, [0, 2, 1, 3])?)?;
        Ok(ctx.reshape_2d(&out, (c.head_dim * c.n_head) as i64, n_new)?)
    }

    /// RMS-normalise then scale by a learned weight — the pattern every norm
    /// in this architecture uses.
    pub fn norm_scaled<'a>(
        &self,
        ctx: &'a Context,
        x: &Tensor<'a>,
        weight: &Tensor<'a>,
    ) -> Result<Tensor<'a>> {
        self.rms_norm_mul(ctx, x, weight)
    }

    fn rms_norm_mul<'a>(
        &self,
        ctx: &'a Context,
        x: &Tensor<'a>,
        weight: &Tensor<'a>,
    ) -> Result<Tensor<'a>> {
        let normed = ctx.rms_norm(x, self.config.rms_eps)?;
        Ok(ctx.mul(&normed, weight)?)
    }

    /// Scaled dot-product attention with a causal mask.
    fn attention<'a>(
        &self,
        ctx: &'a Context,
        q: &Tensor<'a>,
        k: &Tensor<'a>,
        v: &Tensor<'a>,
        n_tokens: i64,
    ) -> Result<Tensor<'a>> {
        let c = &self.config;

        // Shapes are the whole difficulty here, so each step names what it
        // produces. ggml's ne[0] is the fastest dimension, and mul_mat
        // contracts over ne[0] of both operands.

        // [head_dim, n_head, n_tok] -> [head_dim, n_tok, n_head]
        let q = ctx.cont(&ctx.permute(q, [0, 2, 1, 3])?)?;
        // [head_dim, n_kv, n_tok] -> [head_dim, n_tok, n_kv]
        let k = ctx.cont(&ctx.permute(k, [0, 2, 1, 3])?)?;

        // Contracts over head_dim -> [n_tok, n_tok, n_head]. Grouped-query
        // attention works because ggml broadcasts when n_head is a multiple
        // of n_kv.
        let scores = ctx.mul_mat(&k, &q)?;

        // Causal mask. Without it every position attends to future tokens --
        // the model sees the answer before predicting it, and collapses into
        // repeating one token. Added to the scores before softmax, so masked
        // positions need -inf rather than 0.
        let mask = ctx.new_f32_2d(n_tokens, n_tokens)?;
        let mut m = vec![0f32; (n_tokens * n_tokens) as usize];
        for query in 0..n_tokens {
            for key in 0..n_tokens {
                if key > query {
                    m[(query * n_tokens + key) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        mask.set_f32(&m)?;

        let probs = ctx.soft_max_ext(&scores, Some(&mask), c.attn_scale(), 0.0)?;

        // V must contract over n_tok, so it needs n_tok in ne[0]:
        //   [head_dim, n_kv, n_tok] --permute--> [head_dim, n_tok, n_kv]
        //                           --transpose-> [n_tok, head_dim, n_kv]
        // Transposing without the permute first leaves [n_kv, head_dim, n_tok],
        // whose ne[0] is n_kv -- which is exactly the mismatch that aborts
        // ggml with `ggml_can_mul_mat` failing.
        let v = ctx.reshape_3d(v, c.head_dim as i64, c.n_head_kv as i64, n_tokens)?;
        let v = ctx.cont(&ctx.transpose(&ctx.permute(&v, [0, 2, 1, 3])?)?)?;

        // [head_dim, n_tok, n_head]
        let out = ctx.mul_mat(&v, &probs)?;

        // -> [head_dim, n_head, n_tok] -> flat [n_head*head_dim, n_tok].
        // Note n_head*head_dim need not equal n_embd: Qwen3-4B has 32*128 =
        // 4096 against n_embd 2560, and the output projection maps between
        // them.
        let out = ctx.cont(&ctx.permute(&out, [0, 2, 1, 3])?)?;
        Ok(ctx.reshape_2d(&out, (c.head_dim * c.n_head) as i64, n_tokens)?)
    }

    /// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
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
        let (gate_w, up_w) = self.gate_up_weights(ctx, weights, il)?;
        let gate = ctx.mul_mat(&gate_w, x)?;
        let up = ctx.mul_mat(&up_w, x)?;
        let activated = ctx.mul(&ctx.silu(&gate)?, &up)?;
        Ok(ctx.mul_mat(get(format!("blk.{il}.ffn_down.weight"))?, &activated)?)
    }

    /// Mixture-of-experts feed-forward.
    ///
    /// The router scores every expert, the top `n_expert_used` are selected,
    /// and `mul_mat_id` applies only those — which is the whole reason a model
    /// with 128 experts costs about as much as one with a handful.
    fn moe_ffn<'a>(
        &self,
        ctx: &'a Context,
        weights: &WeightSet<'a>,
        x: &Tensor<'a>,
        il: u32,
        n_tokens: i64,
    ) -> Result<Tensor<'a>> {
        let c = &self.config;
        let get = |name: String| -> Result<&Tensor<'a>> {
            weights.get(&name).ok_or(ArchError::MissingTensor(name))
        };

        // Router: one score per expert per token, softmaxed into weights.
        let logits = ctx.mul_mat(get(format!("blk.{il}.ffn_gate_inp.weight"))?, x)?;
        let probs = ctx.soft_max_ext(&logits, None, 1.0, 0.0)?;

        // top_k returns indices. NOTE: they are NOT ordered by score, so the
        // per-expert weight must be looked up by index rather than by
        // position -- see the ggml top_k test.
        let selected = ctx.top_k(&probs, c.n_expert_used as i32)?;

        let x3 = ctx.reshape_3d(x, c.n_embd as i64, 1, n_tokens)?;
        let gate = ctx.mul_mat_id(
            get(format!("blk.{il}.ffn_gate_exps.weight"))?,
            &x3,
            &selected,
        )?;
        let up = ctx.mul_mat_id(get(format!("blk.{il}.ffn_up_exps.weight"))?, &x3, &selected)?;
        let activated = ctx.mul(&ctx.silu(&gate)?, &up)?;
        let down = ctx.mul_mat_id(
            get(format!("blk.{il}.ffn_down_exps.weight"))?,
            &activated,
            &selected,
        )?;

        // Weight each expert's output by its router probability, then sum.
        let weights_sel = ctx.get_rows(&probs, &selected)?;
        let weighted = ctx.mul(&down, &weights_sel)?;
        Ok(ctx.sum_rows(&weighted)?)
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
            fused_qkv: false,
            fused_gate_up: false,
            post_norms: false,
            scale_embeddings: false,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            sliding_window: 0,
        }
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
    fn llama_gets_norm_rope_and_qwen_gets_neox() {
        // The two conventions run without error on each other's layout and
        // produce plausible text either way, so nothing downstream can catch a
        // mix-up. It has to be right here.
        assert_eq!(rope_type_for("llama"), (ROPE_TYPE_NORM, true));
        assert_eq!(rope_type_for("mistral"), (ROPE_TYPE_NORM, true));
        assert_eq!(rope_type_for("qwen3"), (ROPE_TYPE_NEOX, true));
        assert_eq!(rope_type_for("qwen3moe"), (ROPE_TYPE_NEOX, true));
        assert_eq!(rope_type_for("gemma2"), (ROPE_TYPE_NEOX, true));

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
        for arch in ["deepseek4", "gemma2", "llama", "phi3", "qwen3", "qwen3moe"] {
            assert!(architecture_is_verified(arch), "{arch} should be verified");
        }
        // `gemma` (v1) is deliberately absent: it is close to `gemma2` but not
        // identical, and nobody has run it.
        for arch in ["gemma", "gemma3", "falcon", "mamba", "something-new"] {
            assert!(
                !architecture_is_verified(arch),
                "{arch} has not been checked and must not claim to be"
            );
        }
    }
}
