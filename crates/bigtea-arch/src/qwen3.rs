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
    /// Expert count; zero for the dense variant.
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_ff_expert: u32,
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
            rms_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            n_expert: model.arch_u64("expert_count").unwrap_or(0) as u32,
            n_expert_used: model.arch_u64("expert_used_count").unwrap_or(0) as u32,
            n_ff_expert: model
                .arch_u64("expert_feed_forward_length")
                .unwrap_or(0) as u32,
        })
    }

    pub fn is_moe(&self) -> bool {
        self.n_expert > 0 && self.n_expert_used > 0
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
            for suffix in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "ffn_norm.weight",
            ] {
                names.push(format!("blk.{il}.{suffix}"));
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
            let normed = self.rms_norm_mul(ctx, &cur, get(&format!("blk.{il}.attn_norm.weight"))?)?;

            let q = ctx.mul_mat(get(&format!("blk.{il}.attn_q.weight"))?, &normed)?;
            let k = ctx.mul_mat(get(&format!("blk.{il}.attn_k.weight"))?, &normed)?;
            let v = ctx.mul_mat(get(&format!("blk.{il}.attn_v.weight"))?, &normed)?;

            // Split into heads before normalising: Qwen3 normalises each head
            // separately, with a weight of head_dim rather than n_embd.
            let q = ctx.reshape_3d(&q, c.head_dim as i64, c.n_head as i64, n_tokens)?;
            let k = ctx.reshape_3d(&k, c.head_dim as i64, c.n_head_kv as i64, n_tokens)?;

            let q = self.rms_norm_mul(ctx, &q, get(&format!("blk.{il}.attn_q_norm.weight"))?)?;
            let k = self.rms_norm_mul(ctx, &k, get(&format!("blk.{il}.attn_k_norm.weight"))?)?;

            let q = ctx.rope_ext(
                &q,
                positions,
                None,
                c.head_dim as i32,
                ROPE_TYPE_NEOX,
                0,
                rope,
            )?;
            let k = ctx.rope_ext(
                &k,
                positions,
                None,
                c.head_dim as i32,
                ROPE_TYPE_NEOX,
                0,
                rope,
            )?;

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

        let q = ctx.mul_mat(get(format!("blk.{il}.attn_q.weight"))?, &normed)?;
        let k = ctx.mul_mat(get(format!("blk.{il}.attn_k.weight"))?, &normed)?;
        let v = ctx.mul_mat(get(format!("blk.{il}.attn_v.weight"))?, &normed)?;

        let q = ctx.reshape_3d(&q, c.head_dim as i64, c.n_head as i64, n_tokens)?;
        let k = ctx.reshape_3d(&k, c.head_dim as i64, c.n_head_kv as i64, n_tokens)?;

        let q = self.rms_norm_mul(ctx, &q, get(format!("blk.{il}.attn_q_norm.weight"))?)?;
        let k = self.rms_norm_mul(ctx, &k, get(format!("blk.{il}.attn_k_norm.weight"))?)?;

        let q = ctx.rope_ext(&q, positions, None, c.head_dim as i32, rope_type, 0, rope)?;
        let k = ctx.rope_ext(&k, positions, None, c.head_dim as i32, rope_type, 0, rope)?;

        let attn = self.attention(ctx, &q, &k, &v, n_tokens)?;
        Ok(ctx.mul_mat(get(format!("blk.{il}.attn_output.weight"))?, &attn)?)
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
    fn dense_ffn<'a>(
        &self,
        ctx: &'a Context,
        weights: &WeightSet<'a>,
        x: &Tensor<'a>,
        il: u32,
    ) -> Result<Tensor<'a>> {
        let get = |name: String| -> Result<&Tensor<'a>> {
            weights
                .get(&name)
                .ok_or(ArchError::MissingTensor(name))
        };
        let gate = ctx.mul_mat(get(format!("blk.{il}.ffn_gate.weight"))?, x)?;
        let up = ctx.mul_mat(get(format!("blk.{il}.ffn_up.weight"))?, x)?;
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
            weights
                .get(&name)
                .ok_or(ArchError::MissingTensor(name))
        };

        // Router: one score per expert per token, softmaxed into weights.
        let logits = ctx.mul_mat(get(format!("blk.{il}.ffn_gate_inp.weight"))?, x)?;
        let probs = ctx.soft_max_ext(&logits, None, 1.0, 0.0)?;

        // top_k returns indices. NOTE: they are NOT ordered by score, so the
        // per-expert weight must be looked up by index rather than by
        // position -- see the ggml top_k test.
        let selected = ctx.top_k(&probs, c.n_expert_used as i32)?;

        let x3 = ctx.reshape_3d(x, c.n_embd as i64, 1, n_tokens)?;
        let gate = ctx.mul_mat_id(get(format!("blk.{il}.ffn_gate_exps.weight"))?, &x3, &selected)?;
        let up = ctx.mul_mat_id(get(format!("blk.{il}.ffn_up_exps.weight"))?, &x3, &selected)?;
        let activated = ctx.mul(&ctx.silu(&gate)?, &up)?;
        let down =
            ctx.mul_mat_id(get(format!("blk.{il}.ffn_down_exps.weight"))?, &activated, &selected)?;

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
            n_expert: 0,
            n_expert_used: 0,
            n_ff_expert: 0,
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
        assert!((c.attn_scale() - 0.25).abs() < 1e-6, "got {}", c.attn_scale());
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
}
