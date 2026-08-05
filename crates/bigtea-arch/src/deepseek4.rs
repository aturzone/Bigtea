//! DeepSeek-V4-Flash — the architecture this runner exists for.
//!
//! Qwen3-30B nearly fits this machine, which is why llama.cpp's page cache
//! beats a hand-managed budget there. V4-Flash inverts that: **7.38 GiB of
//! always-read weights, which fit, and 137.06 GiB of routed experts, which do
//! not**. Pinning the first and streaming the second is the whole design, and
//! this is the model that tests it.
//!
//! Everything here is read from the container, not from a model card. See
//! `docs/graph/research/v4flash-port-recon.md` for the measurements and
//! `head-to-head-llamacpp-2026-08-05.md` for the bar (llama.cpp: 0.45 tok/s).
//!
//! # What makes this harder than Qwen3
//!
//! * **Attention is MLA-style and low-rank.** Q is projected down to
//!   `q_lora_rank` (1024) and back up (`attn_q_a` then `attn_q_b`), and K/V are
//!   a single compressed 512-wide tensor (`attn_kv`) rather than per-head K and
//!   V. `head_count_kv` is 1. The output projection is also low-rank
//!   (`attn_output_a`, `attn_output_b`). [`crate::KvCache`]'s
//!   `n_kv_heads * head_dim` layout cannot express this.
//! * **Attention sinks**: `attn_sinks` is one learned value per head, an extra
//!   always-attended logit. ggml's fused attention takes these via
//!   `ggml_flash_attn_ext_add_sinks`.
//! * **Hyper-connections** replace the plain residual: `hc_*_base`, `hc_*_fn`
//!   and `hc_*_scale` per block, with 4 streams and 20 Sinkhorn iterations.
//! * **A shared expert** (`ffn_*_shexp`) runs for every token in addition to
//!   the 6 routed ones, so it is always-read and belongs in the resident set.
//! * **Experts are MXFP4** (ggml type 39), not Q4_K.
//! * **`ffn_gate_tid2eid`** is an I32 table of shape `[6, vocab]` — six expert
//!   ids per *token id*. Nothing in Qwen3 corresponds to it; it is presumably
//!   the hash routing implied by `hash_layer_count`. Its exact role is an open
//!   question and is deliberately not guessed at here.
//!
//! This module currently covers the container: shape, the resident/streamed
//! split, the per-layer attention plan, and verification that every tensor
//! named is actually present. The forward pass is not implemented — a wrong
//! one produces fluent nonsense rather than an error, so it is staged
//! separately and checked against llama.cpp's logits.
//!
//! # Attention, exactly
//!
//! Transcribed from `llama.cpp/src/models/deepseek4.cpp` (build 2026-08-03),
//! `build_attention` at line 801, so the next person does not have to derive
//! it from tensor shapes — which does not work, because
//! [`Deepseek4Config::output_group_count`] makes the dimensions appear not to
//! connect.
//!
//! ```text
//! qr      = wq_a · x                       4096 -> 1024
//! qr      = rms_norm(qr) * attn_q_a_norm
//! q       = wq_b · qr                      1024 -> 32768
//! q       = reshape_3d(q, 512, 64, nt)     [head_dim, n_head, tokens]
//! q       = rms_norm(q)                    ** no weight — unweighted **
//! q_nope  = q[.., 0..448]                  view, no position
//! q_pe    = q[.., 448..512]                view, rotated
//! q_pe    = rope_ext(q_pe, pos, n_rot=64, ..)
//! q       = concat(q_nope, q_pe, dim 0)
//!
//! kv      = wkv · x                        4096 -> 512
//! kv      = rms_norm(kv) * attn_kv_a_norm
//! kv      = reshape_3d(kv, 512, 1, nt)     ** one head, shared **
//! kv_nope = kv[.., 0..448]
//! kv_pe   = rope_ext(kv[.., 448..512], pos, 64, ..)
//! kv      = concat(kv_nope, kv_pe, dim 0)
//!
//! out     = attn_mha(q, kv, kv, mask, sinks, scale)   ** kv is K *and* V **
//!
//! // the output is de-roped before projection
//! out     = reshape_3d(out, 512, 64, nt)
//! out_pe  = rope_back(out[.., 448..512], pos, ..)      ** inverse rotation **
//! out     = concat(out[.., 0..448], out_pe, dim 0)
//! out     = permute(reshape(out, 4096, 8, nt))         ** 8 groups **
//! oa      = mul_mat(reshape_3d(wo_a, 4096, o_lora_rank, 8), out)
//! oa      = cont(permute(oa))                          -> [8192, nt]
//! out     = wo_b · oa                                  -> [4096, nt]
//! ```
//!
//! Five things there are easy to get wrong and none of them raise an error:
//!
//! 1. **`kv` is passed as both K and V** (`deepseek4.cpp:792`). There is no
//!    separate V projection and no `wkv_b` — the one compressed 512-wide
//!    tensor serves as both, which is why `head_count_kv` is 1.
//! 2. **The per-head `rms_norm` on `q` carries no weight** (line 839), unlike
//!    every other norm in the model. Applying `attn_q_a_norm` again here would
//!    look reasonable and be wrong.
//! 3. **Only `q_pe`/`kv_pe` are rotated**, the trailing 64 of 512 dims. The
//!    leading 448 are concatenated back unchanged.
//! 4. **RoPE parameters are per layer.** `dsv4_compress_ratios[il] != 0`
//!    selects `compress_rope_freq_base` (160000) with YaRN scaling; the other
//!    layers use the plain base with no scaling at all (lines 822-829).
//! 5. **The attention output is de-roped** with `rope_back` before projection,
//!    on the same trailing 64 dims. Skipping it leaves the rotation baked into
//!    the residual stream. This is visible only in the captured trace, not in
//!    the tensor shapes.
//!
//! # The reference these were checked against
//!
//! `tests/fixtures/v4flash-layer0-oracle*.txt` hold every tensor of the
//! prologue and layer 0 with its shape and the sum of its elements, captured
//! from `llama-eval-callback` on the real container. A sum is an unforgiving
//! checksum — a transposed matrix, an unrotated RoPE or a missing norm all
//! change it — so the forward pass is built against those numbers rather than
//! against whether the output reads like English.
//!
//! There are two captures, at one token and at five, and the second is not
//! redundant: **at a single token, position 0, RoPE is the identity**, so the
//! one-token trace shows `q_pe` with the same sum as its input and cannot
//! distinguish a correct rotation from no rotation at all. Point 3 above is
//! checked only by the five-token fixture.
//!
//! RoPE parameters are per layer and are *not* the container's top-level ones.
//! Layer 0 has `dsv4_compress_ratios[0] == 0` and so takes the uncompressed
//! path: plain `freq_base`, `freq_scale` 1, and scaling off entirely —
//! `ext_factor` 0, `attn_factor` 1, both betas 0, `n_ctx_orig` 0
//! (`deepseek4.cpp:822-829`). The YaRN settings in the metadata belong to the
//! other 41 layers. The mode is `NORM`, not `NEOX`.
//!
//! # ggml already implements the hard parts
//!
//! This build's public `ggml.h` exposes the operations that looked like the
//! expensive half of this port:
//!
//! * `ggml_dsv4_hc_pre`, `ggml_dsv4_hc_post`, `ggml_dsv4_hc_comb` —
//!   hyper-connections, with `hc_comb`'s `n_iter` being the Sinkhorn iteration
//!   count (20 for this model).
//! * `ggml_lightning_indexer` — the sparse-attention indexer.
//! * `ggml_flash_attn_ext_add_sinks` — the per-head attention sinks.
//!
//! So hyper-connections and the indexer are FFI bindings here, not
//! implementations. That removes the two pieces the recon flagged as hardest.

use bigtea_model::Model;

use crate::{ArchError, Result};

/// Which attention a given layer uses.
///
/// V4-Flash does not use one attention throughout. llama.cpp's `deepseek4.cpp`
/// dispatches to three different builders per layer, and the container says
/// which by what tensors the layer carries:
///
/// | kind | needs | layers |
/// |---|---|---|
/// | [`Raw`](AttentionKind::Raw) | neither compressor nor indexer | 2 |
/// | [`HeavilyCompressed`](AttentionKind::HeavilyCompressed) | `attn_compressor_*` | 20 |
/// | [`CompressedSparse`](AttentionKind::CompressedSparse) | compressor **and** `indexer.*` | 21 |
///
/// Implementing one of these and applying it everywhere would produce fluent
/// output that is quietly wrong on half the model — the failure mode this
/// project keeps meeting. So the kind is derived from the container, never
/// assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    /// Plain attention over the full context.
    Raw,
    /// Heavily Compressed Attention: keys and values summarised into blocks.
    HeavilyCompressed,
    /// Compressed Sparse Attention with the lightning indexer choosing which
    /// keys each query keeps (`indexer.top_k` of them).
    CompressedSparse,
}

/// Shape and hyper-parameters, read from the container.
#[derive(Debug, Clone)]
pub struct Deepseek4Config {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    /// Width of the compressed K/V projection (`attention.key_length`).
    pub kv_lora_rank: u32,
    /// Width Q is projected down to before being projected back up.
    pub q_lora_rank: u32,
    /// Width the attention output is projected through.
    pub output_lora_rank: u32,
    pub vocab_size: u32,
    pub n_expert: u32,
    pub n_expert_used: u32,
    /// Always-active experts, in addition to the routed ones.
    pub n_expert_shared: u32,
    pub n_ff_expert: u32,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    /// Router output scaling; 1.5 here, and not 1.0 as in Qwen3.
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
    /// Parallel residual streams in the hyper-connection block.
    pub hc_count: u32,
    pub hc_sinkhorn_iterations: u32,
    pub sliding_window: u32,
    /// Sparse-attention indexer: how many keys each query keeps.
    pub indexer_top_k: u32,
    pub indexer_head_count: u32,
    pub indexer_key_length: u32,
    /// Dimensions of a head that carry rotary position (`rope.dimension_count`).
    ///
    /// Only 64 of each head's 512 dims are rotated; the other 448 carry no
    /// position at all. This is DeepSeek's decoupled RoPE, and rotating the
    /// whole head — the obvious reading — changes every score.
    pub n_rot: u32,
    /// Groups the attention output projection is split across.
    ///
    /// `attn_output_a` ships as 2-D `[4096, 8192]` but is used as
    /// `[n_head * head_dim / groups, output_lora_rank, groups]`: a batched
    /// matmul, not a single one. llama.cpp reshapes it at
    /// `deepseek4.cpp:1081`.
    pub output_group_count: u32,
    /// Parallel residual streams the hyper-connection block carries.
    pub hc_mult: u32,

    /// SwiGLU clamp limit **per layer**, for the routed experts.
    ///
    /// 43 values. Applied asymmetrically: the gate is clamped to
    /// `(-inf, limit]` and the up projection to `[-limit, limit]`
    /// (`llama-graph.cpp:2050-2057`, an `LLM_ARCH_DEEPSEEK4` branch). A limit
    /// of 0 or less means no clamp at all.
    pub swiglu_clamp_exp: Vec<f32>,
    /// The same, for the shared expert. Falls back to
    /// [`Self::swiglu_clamp_exp`] when the container omits it.
    pub swiglu_clamp_shexp: Vec<f32>,
    /// Per-layer compression ratio; **0 means the layer is uncompressed**.
    ///
    /// The container ships 44 values for 43 blocks and that is not an
    /// off-by-one — it is indexed as `dsv4_compress_ratios[il]`, so only the
    /// first 43 are ever consulted. Non-zero selects the compressed RoPE base
    /// with YaRN scaling; see [`Self::rope_for_layer`].
    pub compress_ratios: Vec<i64>,
    /// RoPE base used by compressed layers, in place of [`Self::rope_freq_base`].
    pub compress_rope_freq_base: f32,
    /// YaRN, for compressed layers only.
    pub rope_freq_scale: f32,
    pub rope_ext_factor: f32,
    pub rope_beta_fast: f32,
    pub rope_beta_slow: f32,
    pub rope_n_ctx_orig: u32,
}

/// Everything `ggml_rope_ext` needs for one layer.
///
/// `mode` is not included: deepseek4 is `LLAMA_ROPE_TYPE_NORM` throughout, so
/// rotated pairs are adjacent rather than offset by `n_rot/2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerRope {
    pub params: bigtea_ggml::RopeParams,
    pub n_ctx_orig: i32,
}

impl Deepseek4Config {
    pub fn from_model(model: &Model) -> Result<Self> {
        let arch = model.architecture().to_string();
        if arch != "deepseek4" {
            return Err(ArchError::Unsupported(arch));
        }
        let need = |suffix: &str| -> Result<u64> {
            model
                .arch_u64(suffix)
                .ok_or_else(|| ArchError::MissingMetadata(format!("{arch}.{suffix}")))
        };
        let opt = |suffix: &str, default: u64| model.arch_u64(suffix).unwrap_or(default);

        Ok(Deepseek4Config {
            n_layer: need("block_count")? as u32,
            n_embd: need("embedding_length")? as u32,
            n_head: need("attention.head_count")? as u32,
            n_head_kv: opt("attention.head_count_kv", 1) as u32,
            kv_lora_rank: need("attention.key_length")? as u32,
            q_lora_rank: need("attention.q_lora_rank")? as u32,
            output_lora_rank: opt("attention.output_lora_rank", 0) as u32,
            vocab_size: model
                .arch_u64("vocab_size")
                .or_else(|| model.location("token_embd.weight").map(|l| l.dims[1]))
                .unwrap_or(0) as u32,
            n_expert: need("expert_count")? as u32,
            n_expert_used: need("expert_used_count")? as u32,
            n_expert_shared: opt("expert_shared_count", 0) as u32,
            n_ff_expert: need("expert_feed_forward_length")? as u32,
            rms_eps: 1e-6,
            rope_freq_base: opt("rope.freq_base", 10_000) as f32,
            expert_weights_scale: 1.5,
            expert_weights_norm: true,
            hc_count: opt("hyper_connection.count", 1) as u32,
            hc_sinkhorn_iterations: opt("hyper_connection.sinkhorn_iterations", 0) as u32,
            sliding_window: opt("attention.sliding_window", 0) as u32,
            indexer_top_k: opt("attention.indexer.top_k", 0) as u32,
            indexer_head_count: opt("attention.indexer.head_count", 0) as u32,
            indexer_key_length: opt("attention.indexer.key_length", 0) as u32,
            n_rot: opt("rope.dimension_count", 0) as u32,
            output_group_count: opt("attention.output_group_count", 1) as u32,
            hc_mult: opt("hyper_connection.count", 1) as u32,

            swiglu_clamp_exp: model.arch_f32_array("swiglu_clamp_exp").unwrap_or_default(),
            // llama.cpp falls back to the routed limits when the shared-expert
            // key is absent (`deepseek4.cpp:28`), rather than to no clamp.
            swiglu_clamp_shexp: model
                .arch_f32_array("swiglu_clamp_shexp")
                .or_else(|| model.arch_f32_array("swiglu_clamp_exp"))
                .unwrap_or_default(),
            compress_ratios: model
                .arch_i64_array("attention.compress_ratios")
                .unwrap_or_default(),
            compress_rope_freq_base: model
                .arch_f32("attention.compress_rope_freq_base")
                .unwrap_or(10_000.0),
            // The container stores the YaRN *factor* (16); ggml wants its
            // reciprocal as freq_scale.
            rope_freq_scale: model
                .arch_f32("rope.scaling.factor")
                .filter(|f| *f > 0.0)
                .map(|f| 1.0 / f)
                .unwrap_or(1.0),
            // llama.cpp defaults ext_factor to 1.0 when the scaling type is
            // yarn and 0.0 otherwise.
            rope_ext_factor: match model
                .metadata()
                .get(&format!("{arch}.rope.scaling.type"))
                .and_then(bigtea_gguf::Value::as_str)
            {
                Some("yarn") => 1.0,
                _ => 0.0,
            },
            rope_beta_fast: model.arch_f32("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
            rope_beta_slow: model.arch_f32("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
            rope_n_ctx_orig: opt("rope.scaling.original_context_length", 0) as u32,
        })
    }

    /// Whether `layer` uses the compressed RoPE base and YaRN scaling.
    ///
    /// `dsv4_compress_ratios[il] != 0` (`deepseek4.cpp:822`). A layer past the
    /// end of the array is treated as uncompressed rather than panicking.
    pub fn uses_compress_rope(&self, layer: u32) -> bool {
        self.compress_ratios
            .get(layer as usize)
            .is_some_and(|r| *r != 0)
    }

    /// RoPE parameters for `layer`.
    ///
    /// **These are per layer and are not the container's top-level keys.**
    /// Transcribed from `deepseek4.cpp:822-829`: an uncompressed layer uses the
    /// plain `freq_base` with scaling switched off *entirely* — freq_scale 1,
    /// ext_factor 0, attn_factor 1, both betas 0, `n_ctx_orig` 0. Applying the
    /// container's YaRN settings there changes every score and reports nothing.
    ///
    /// # Verification status
    ///
    /// The **uncompressed** branch is checked against llama.cpp's trace on
    /// layer 0 (`tests/deepseek4_forward.rs`). The **compressed** branch is
    /// transcribed from the source but *not* verified against any capture —
    /// the oracle stops at the end of layer 0, which is uncompressed. Treat it
    /// as unproven until a compressed layer has rows.
    pub fn rope_for_layer(&self, layer: u32) -> LayerRope {
        if !self.uses_compress_rope(layer) {
            return LayerRope {
                params: bigtea_ggml::RopeParams {
                    freq_base: self.rope_freq_base,
                    freq_scale: 1.0,
                    ext_factor: 0.0,
                    attn_factor: 1.0,
                    beta_fast: 0.0,
                    beta_slow: 0.0,
                },
                n_ctx_orig: 0,
            };
        }
        LayerRope {
            params: bigtea_ggml::RopeParams {
                freq_base: self.compress_rope_freq_base,
                freq_scale: self.rope_freq_scale,
                ext_factor: self.rope_ext_factor,
                attn_factor: Self::rope_attn_factor(self.rope_freq_scale, self.rope_ext_factor),
                beta_fast: self.rope_beta_fast,
                beta_slow: self.rope_beta_slow,
            },
            n_ctx_orig: self.rope_n_ctx_orig as i32,
        }
    }

    /// `dsv4_rope_attn_factor` (`deepseek4.cpp:10`).
    ///
    /// Not the usual YaRN `mscale`: DeepSeek-V4 uses `1/(1 + 0.1*ln(1/s))`, and
    /// returns exactly 1 when there is no extension.
    fn rope_attn_factor(freq_scale: f32, ext_factor: f32) -> f32 {
        if ext_factor == 0.0 {
            return 1.0;
        }
        1.0 / (1.0 + 0.1 * (1.0 / freq_scale).ln())
    }

    /// The SwiGLU clamp limit for `layer`, or `None` when the layer has none.
    ///
    /// llama.cpp treats any limit at or below 1e-6 as "no clamp"
    /// (`llama-graph.cpp:2049`).
    pub fn swiglu_limit(&self, layer: u32, shared: bool) -> Option<f32> {
        let table = if shared {
            &self.swiglu_clamp_shexp
        } else {
            &self.swiglu_clamp_exp
        };
        table.get(layer as usize).copied().filter(|l| *l > 1e-6)
    }

    /// Head dimensions carrying no positional encoding.
    ///
    /// `head_dim - n_rot` = 512 - 64 = 448 here.
    pub fn n_rot_none(&self) -> u32 {
        self.kv_lora_rank.saturating_sub(self.n_rot)
    }

    /// Width of the residual stream the hyper-connection block operates on:
    /// `n_embd * hc_mult`, 16384 here, which is why `hc_attn_fn` is
    /// `[16384, 24]`.
    pub fn hc_dim(&self) -> u32 {
        self.n_embd * self.hc_mult
    }

    /// Bytes of routed-expert weight a single token pulls off disk.
    ///
    /// The number the whole design lives or dies by: 6 of 256 experts, three
    /// tensors each. Measured at 3.21 GiB per token for this container.
    pub fn expert_bytes_per_token(&self, model: &Model) -> u64 {
        let mut per_expert = 0u64;
        for suffix in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            if let Some(loc) = model.location(&format!("blk.0.{suffix}.weight")) {
                per_expert += loc.size / self.n_expert.max(1) as u64;
            }
        }
        per_expert * self.n_expert_used as u64 * self.n_layer as u64
    }
}

/// Tensor names, and which of them stay resident.
pub struct Deepseek4Model {
    pub config: Deepseek4Config,
}

impl Deepseek4Model {
    pub fn new(config: Deepseek4Config) -> Self {
        Deepseek4Model { config }
    }

    /// Per-block tensors that are read for every token.
    ///
    /// Note the shared expert is here: it runs unconditionally, so it is
    /// resident weight, not streamed. Getting that wrong would either re-read
    /// it 43 times per token or leave it out of the residency budget.
    const RESIDENT_BLOCK_SUFFIXES: &'static [&'static str] = &[
        "attn_norm.weight",
        "attn_q_a.weight",
        "attn_q_a_norm.weight",
        "attn_q_b.weight",
        "attn_kv.weight",
        "attn_kv_a_norm.weight",
        "attn_output_a.weight",
        "attn_output_b.weight",
        "attn_sinks.weight",
        "ffn_norm.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
        "hc_attn_base.weight",
        "hc_attn_fn.weight",
        "hc_attn_scale.weight",
        "hc_ffn_base.weight",
        "hc_ffn_fn.weight",
        "hc_ffn_scale.weight",
    ];

    /// Per-block tensors that only some layers carry.
    ///
    /// **The layers of this model are not uniform**, which is the single most
    /// load-bearing fact about porting it. Counted against the container:
    ///
    /// | suffix | blocks |
    /// |---|---|
    /// | the 20 above, plus the 3 routed | 43 |
    /// | `attn_compressor_*` | 41 |
    /// | `exp_probs_b.bias` | 40 |
    /// | `indexer.*`, `indexer_compressor_*` | 21 |
    /// | `ffn_gate_tid2eid.weight` | 3 |
    ///
    /// 43x23 + 41x4 + 40 + 21x6 + 3 + 6 global = 1328, which is exactly the
    /// container's tensor count, so nothing is unaccounted for.
    ///
    /// The three `ffn_gate_tid2eid` layers resolve an open question from the
    /// recon: `hash_layer_count 3` in the metadata means *these* three.
    ///
    /// A manifest that assumed every block looked like block 0 would name
    /// tensors that do not exist on 22 of the 43 layers, and would silently
    /// under-count the resident set by half a gigabyte.
    const OPTIONAL_BLOCK_SUFFIXES: &'static [&'static str] = &[
        "attn_compressor_ape.weight",
        "attn_compressor_gate.weight",
        "attn_compressor_kv.weight",
        "attn_compressor_norm.weight",
        "exp_probs_b.bias",
        "indexer.attn_q_b.weight",
        "indexer.proj.weight",
        "indexer_compressor_ape.weight",
        "indexer_compressor_gate.weight",
        "indexer_compressor_kv.weight",
        "indexer_compressor_norm.weight",
        "ffn_gate_tid2eid.weight",
    ];

    /// The stacked expert tensors, read one slice at a time.
    const ROUTED_BLOCK_SUFFIXES: &'static [&'static str] =
        &["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];

    /// Tensors outside any block.
    const GLOBAL: &'static [&'static str] = &[
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "output_hc_base.weight",
        "output_hc_fn.weight",
        "output_hc_scale.weight",
    ];

    /// Everything held in RAM for the whole run.
    ///
    /// Takes the model because the optional per-layer tensors have to be
    /// discovered rather than assumed — see [`Self::OPTIONAL_BLOCK_SUFFIXES`].
    pub fn resident_tensor_names(&self, model: &Model) -> Vec<String> {
        let mut names: Vec<String> = Self::GLOBAL.iter().map(|s| s.to_string()).collect();
        for il in 0..self.config.n_layer {
            for suffix in Self::RESIDENT_BLOCK_SUFFIXES {
                names.push(format!("blk.{il}.{suffix}"));
            }
            for suffix in Self::OPTIONAL_BLOCK_SUFFIXES {
                let name = format!("blk.{il}.{suffix}");
                if model.location(&name).is_some() {
                    names.push(name);
                }
            }
        }
        names
    }

    /// Which attention `layer` uses, decided by what the container ships.
    ///
    /// See [`AttentionKind`]. The indexer implies a compressor, so the check
    /// is ordered most-specific first.
    pub fn attention_kind(&self, model: &Model, layer: u32) -> AttentionKind {
        let has = |suffix: &str| {
            model
                .location(&format!("blk.{layer}.{suffix}"))
                .is_some()
        };
        if has("indexer.proj.weight") {
            AttentionKind::CompressedSparse
        } else if has("attn_compressor_kv.weight") {
            AttentionKind::HeavilyCompressed
        } else {
            AttentionKind::Raw
        }
    }

    /// The attention kind of every layer, in order.
    pub fn attention_plan(&self, model: &Model) -> Vec<AttentionKind> {
        (0..self.config.n_layer)
            .map(|il| self.attention_kind(model, il))
            .collect()
    }

    /// How many blocks carry each optional tensor.
    ///
    /// Exposed so the non-uniformity is inspectable rather than folded away —
    /// a change in these counts between model revisions would otherwise show up
    /// as a mysterious accuracy loss.
    pub fn optional_layer_counts(&self, model: &Model) -> Vec<(&'static str, usize)> {
        Self::OPTIONAL_BLOCK_SUFFIXES
            .iter()
            .map(|suffix| {
                let n = (0..self.config.n_layer)
                    .filter(|il| model.location(&format!("blk.{il}.{suffix}")).is_some())
                    .count();
                (*suffix, n)
            })
            .collect()
    }

    /// Stacked expert tensors, streamed per token.
    pub fn routed_tensor_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for il in 0..self.config.n_layer {
            for suffix in Self::ROUTED_BLOCK_SUFFIXES {
                names.push(format!("blk.{il}.{suffix}"));
            }
        }
        names
    }

    /// Fail on a missing tensor now, not at layer 37 of a two-minute load.
    ///
    /// `ffn_gate_tid2eid` is checked but not listed as resident: its role is
    /// unresolved, so claiming to know when it is read would be a guess.
    pub fn verify(&self, model: &Model) -> Result<()> {
        for name in self
            .resident_tensor_names(model)
            .iter()
            .chain(self.routed_tensor_names().iter())
        {
            if model.location(name).is_none() {
                return Err(ArchError::MissingTensor(name.clone()));
            }
        }
        Ok(())
    }

    /// Tensors in the container that this manifest does not name.
    ///
    /// The reverse of [`Self::verify`], and the more useful direction while
    /// porting: a tensor nobody reads is a piece of the architecture that has
    /// been silently skipped, which shows up as degraded output rather than an
    /// error.
    pub fn unclaimed_tensors(&self, model: &Model) -> Vec<String> {
        let mut claimed: std::collections::HashSet<String> =
            self.resident_tensor_names(model).into_iter().collect();
        claimed.extend(self.routed_tensor_names());
        let mut left: Vec<String> = model
            .tensor_names()
            .filter(|n| !claimed.contains(*n))
            .map(|n| n.to_string())
            .collect();
        left.sort_unstable();
        left
    }

    /// Bytes held in RAM if every resident tensor is loaded.
    pub fn resident_bytes(&self, model: &Model) -> u64 {
        self.resident_tensor_names(model)
            .iter()
            .filter_map(|n| model.location(n))
            .map(|l| l.size)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split that decides whether this model runs at all: the shared
    /// expert is always-active and must be resident, the 256 routed ones must
    /// not be. Confusing them either way is the difference between 7 GiB and
    /// 144 GiB of RAM.
    #[test]
    fn shared_expert_is_routed_separately_from_the_256() {
        // The split that decides whether this model runs at all: the shared
        // expert is always-active and must be resident, the 256 routed ones
        // must not be. Confusing them is the difference between 7 GiB and
        // 144 GiB of RAM.
        let model = Deepseek4Model::new(config_for_test());
        let routed = model.routed_tensor_names();
        assert!(routed.iter().any(|n| n == "blk.0.ffn_gate_exps.weight"));
        assert!(
            !routed.iter().any(|n| n.contains("shexp")),
            "the shared expert must never be streamed"
        );
        assert!(Deepseek4Model::RESIDENT_BLOCK_SUFFIXES
            .iter()
            .any(|s| *s == "ffn_gate_shexp.weight"));
        assert!(!Deepseek4Model::RESIDENT_BLOCK_SUFFIXES
            .iter()
            .any(|s| s.contains("_exps.")));
    }

    #[test]
    fn every_block_contributes_its_routed_tensors() {
        let model = Deepseek4Model::new(config_for_test());
        assert_eq!(model.routed_tensor_names().len(), model.config.n_layer as usize * 3);
    }

    fn config_for_test() -> Deepseek4Config {
        Deepseek4Config {
            n_layer: 43,
            n_embd: 4096,
            n_head: 64,
            n_head_kv: 1,
            kv_lora_rank: 512,
            q_lora_rank: 1024,
            output_lora_rank: 1024,
            vocab_size: 129_280,
            n_expert: 256,
            n_expert_used: 6,
            n_expert_shared: 1,
            n_ff_expert: 2048,
            rms_eps: 1e-6,
            rope_freq_base: 10_000.0,
            expert_weights_scale: 1.5,
            expert_weights_norm: true,
            hc_count: 4,
            hc_sinkhorn_iterations: 20,
            sliding_window: 128,
            indexer_top_k: 512,
            indexer_head_count: 64,
            indexer_key_length: 128,
            n_rot: 64,
            output_group_count: 8,
            hc_mult: 4,
            // Layers 0-1 uncompressed, the rest compressed — the container's
            // own shape, so the per-layer branches are exercised.
            swiglu_clamp_exp: vec![10.0; 43],
            swiglu_clamp_shexp: vec![10.0; 43],
            compress_ratios: {
                let mut r = vec![4i64; 44];
                r[0] = 0;
                r[1] = 0;
                r
            },
            compress_rope_freq_base: 160_000.0,
            rope_freq_scale: 1.0 / 16.0,
            rope_ext_factor: 1.0,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            rope_n_ctx_orig: 65536,
        }
    }

    /// Layer 0 is uncompressed, so it must get plain RoPE with **scaling off**
    /// — not the container's YaRN keys, which belong to the other 41 layers.
    ///
    /// This is the config side of the checkpoint the forward test verifies
    /// against llama.cpp; keeping it here means a change to `rope_for_layer`
    /// fails without needing the 144 GB container.
    #[test]
    fn layer_0_gets_plain_rope_and_a_compressed_layer_gets_yarn() {
        let c = config_for_test();
        assert!(!c.uses_compress_rope(0));
        let plain = c.rope_for_layer(0);
        assert_eq!(plain.n_ctx_orig, 0);
        assert_eq!(plain.params.freq_base, 10_000.0);
        assert_eq!(plain.params.freq_scale, 1.0);
        assert_eq!(plain.params.ext_factor, 0.0);
        assert_eq!(plain.params.attn_factor, 1.0);
        assert_eq!(plain.params.beta_fast, 0.0);
        assert_eq!(plain.params.beta_slow, 0.0);

        // Layer 2 is compressed: different base, YaRN on, and an attn_factor
        // that is 1/(1 + 0.1*ln(16)) rather than the usual YaRN mscale.
        assert!(c.uses_compress_rope(2));
        let yarn = c.rope_for_layer(2);
        assert_eq!(yarn.params.freq_base, 160_000.0);
        assert_eq!(yarn.n_ctx_orig, 65536);
        assert_eq!(yarn.params.beta_fast, 32.0);
        let want = 1.0 / (1.0 + 0.1 * 16f32.ln());
        assert!((yarn.params.attn_factor - want).abs() < 1e-6);
        assert!(yarn.params.attn_factor < 1.0, "extension must attenuate");
    }

    /// The clamp table is per layer, and "no clamp" is a real state.
    #[test]
    fn swiglu_limits_are_per_layer_and_optional() {
        let mut c = config_for_test();
        assert_eq!(c.swiglu_limit(0, false), Some(10.0));
        assert_eq!(c.swiglu_limit(0, true), Some(10.0));
        c.swiglu_clamp_exp[7] = 0.0;
        assert_eq!(c.swiglu_limit(7, false), None, "0 means no clamp, not clamp to 0");
        // Past the end of the table is not a panic: a shorter array than
        // block_count is a container problem, not a reason to abort mid-run.
        assert_eq!(c.swiglu_limit(99, false), None);
    }

    /// 44 ratios for 43 blocks is not an off-by-one — only the first 43 are
    /// ever consulted, so the trailing value must not shift anything.
    #[test]
    fn the_extra_compress_ratio_is_never_consulted() {
        let c = config_for_test();
        assert_eq!(c.compress_ratios.len(), 44);
        assert_eq!(c.n_layer, 43);
        let consulted: Vec<bool> =
            (0..c.n_layer).map(|il| c.uses_compress_rope(il)).collect();
        assert_eq!(consulted.len(), 43);
        assert_eq!(consulted[0], false);
        assert_eq!(consulted[42], true);
    }
}
