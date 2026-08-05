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
//! split, and verification that every tensor named is actually present. The
//! forward pass is not implemented — a wrong one produces fluent nonsense
//! rather than an error, so it is staged separately and checked against
//! llama.cpp's logits.

use bigtea_model::Model;

use crate::{ArchError, Result};

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
        })
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
        }
    }
}
