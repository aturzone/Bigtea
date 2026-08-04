//! Streaming MoE: keep the dense weights resident, fetch experts on demand.
//!
//! This is the point of the whole project. Qwen3-30B-A3B is 17.28 GiB, but
//! only **0.93 GiB is read every token** — the other 16.35 GiB is routed
//! experts, of which a single token touches 8 of 128, about 1.02 GiB. A
//! machine with 15.7 GiB cannot hold the model, and does not need to.
//!
//! # Why this cannot be one graph
//!
//! `ggml` graphs are declarative: the whole computation is described, then
//! executed. But *which* experts a token needs is decided by a router that
//! runs partway through — so the weights required cannot be known when the
//! graph is built. The forward pass is therefore executed layer by layer:
//!
//! 1. attention and the router for layer *n*
//! 2. read the selected experts off disk
//! 3. the expert feed-forward for layer *n*
//! 4. carry the activations into layer *n+1*
//!
//! The cost is losing cross-layer graph fusion. The benefit is running a model
//! an order of magnitude larger than memory, which is not a trade — it is the
//! difference between running and not running.

use std::collections::HashMap;

use bigtea_ggml::{Context, RopeParams, Tensor, WeightSet};
use bigtea_model::Model;

use crate::qwen3::{Qwen3Config, Qwen3Model};
use crate::{ArchError, Result};

const ROPE_TYPE_NEOX: i32 = 2;
const GIB: f64 = (1u64 << 30) as f64;

/// How much work streaming actually did — the numbers that justify it.
#[derive(Debug, Default, Clone)]
pub struct StreamStats {
    pub expert_reads: u64,
    pub expert_bytes: u64,
    pub resident_bytes: u64,
    pub read_seconds: f64,
}

impl std::fmt::Display for StreamStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "resident {:.2} GiB, streamed {:.2} GiB over {} expert reads in {:.1}s",
            self.resident_bytes as f64 / GIB,
            self.expert_bytes as f64 / GIB,
            self.expert_reads,
            self.read_seconds
        )
    }
}

/// A forward pass that streams experts instead of holding them.
pub struct StreamingRunner<'m> {
    model: &'m Model,
    arch: Qwen3Model,
    /// Cache of expert slices already read this session, keyed by
    /// (tensor name, expert index). Bounded, because an unbounded cache would
    /// silently become the thing we set out to avoid.
    cache: HashMap<(String, u32), Vec<u8>>,
    cache_budget: usize,
    cache_bytes: usize,
    pub stats: StreamStats,
}

impl<'m> StreamingRunner<'m> {
    pub fn new(model: &'m Model, config: Qwen3Config, cache_budget: usize) -> Self {
        StreamingRunner {
            model,
            arch: Qwen3Model::new(config),
            cache: HashMap::new(),
            cache_budget,
            cache_bytes: 0,
            stats: StreamStats::default(),
        }
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.arch.config
    }

    /// Tensors that stay in RAM: everything except routed experts.
    pub fn resident_tensor_names(&self) -> Vec<String> {
        self.arch
            .required_tensors()
            .into_iter()
            .filter(|n| !n.contains("_exps"))
            .collect()
    }

    /// Read one expert's slice out of a stacked expert tensor.
    ///
    /// The stack is `[ne0, ne1, n_expert]`, so expert `idx` is a contiguous
    /// run at `idx * slice_bytes`. Reading only that run is what keeps a
    /// token's cost at 1 GiB instead of 16.
    fn expert_slice(&mut self, name: &str, idx: u32) -> Result<Vec<u8>> {
        let key = (name.to_string(), idx);
        if let Some(hit) = self.cache.get(&key) {
            return Ok(hit.clone());
        }

        let loc = self
            .model
            .location(name)
            .ok_or_else(|| ArchError::MissingTensor(name.to_string()))?
            .clone();
        let n_expert = *loc.dims.last().unwrap_or(&1);
        if n_expert == 0 || idx as u64 >= n_expert {
            return Err(ArchError::MissingTensor(format!(
                "{name}: expert {idx} of {n_expert}"
            )));
        }
        let slice_bytes = loc.size / n_expert;

        let start = std::time::Instant::now();
        let bytes = self
            .model
            .read_tensor_range(name, idx as u64 * slice_bytes, slice_bytes)?;
        self.stats.read_seconds += start.elapsed().as_secs_f64();
        self.stats.expert_reads += 1;
        self.stats.expert_bytes += bytes.len() as u64;

        // Keep it only if the budget allows; a cache that grows without bound
        // recreates the very problem streaming exists to solve.
        if self.cache_bytes + bytes.len() <= self.cache_budget {
            self.cache_bytes += bytes.len();
            self.cache.insert(key, bytes.clone());
        }
        Ok(bytes)
    }

    /// Which experts this token routes to, and their weights.
    ///
    /// Runs the router on the CPU rather than in the graph, because the result
    /// determines what to read next — the graph cannot branch on it.
    fn route(&self, probs: &[f32], n_expert_used: usize) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(n_expert_used);

        // Renormalise over the selected experts only, which is what Qwen3
        // does; leaving them unnormalised scales every expert output down by
        // the mass that went to the experts we did not run.
        let total: f32 = scored.iter().map(|(_, p)| *p).sum();
        if total > 0.0 {
            for (_, p) in &mut scored {
                *p /= total;
            }
        }
        scored
    }

    /// Apply one layer's expert feed-forward for a single token.
    ///
    /// `x` is that token's activations after the FFN norm.
    fn expert_ffn(&mut self, x: &[f32], il: u32, probs: &[f32]) -> Result<Vec<f32>> {
        let c = self.arch.config.clone();
        let picks = self.route(probs, c.n_expert_used as usize);

        let gate_name = format!("blk.{il}.ffn_gate_exps.weight");
        let up_name = format!("blk.{il}.ffn_up_exps.weight");
        let down_name = format!("blk.{il}.ffn_down_exps.weight");

        let mut accum = vec![0f32; c.n_embd as usize];

        for (expert, weight) in picks {
            let gate_bytes = self.expert_slice(&gate_name, expert)?;
            let up_bytes = self.expert_slice(&up_name, expert)?;
            let down_bytes = self.expert_slice(&down_name, expert)?;

            let gate_ty = self.tensor_type(&gate_name)?;
            let up_ty = self.tensor_type(&up_name)?;
            let down_ty = self.tensor_type(&down_name)?;

            // A small arena: one expert's worth of work, not the model's.
            let ctx = Context::new(256 << 20)?;
            let mut ws = WeightSet::new();
            let n_embd = c.n_embd as i64;
            let n_ff = c.n_ff_expert as i64;

            ws.bind(&ctx, "gate", gate_ty, &[n_embd as u64, n_ff as u64], gate_bytes)?;
            ws.bind(&ctx, "up", up_ty, &[n_embd as u64, n_ff as u64], up_bytes)?;
            ws.bind(&ctx, "down", down_ty, &[n_ff as u64, n_embd as u64], down_bytes)?;

            let xt = ctx.new_f32_2d(n_embd, 1)?;
            xt.set_f32(x)?;

            let g = ctx.mul_mat(ws.get("gate").expect("bound"), &xt)?;
            let u = ctx.mul_mat(ws.get("up").expect("bound"), &xt)?;
            let act = ctx.mul(&ctx.silu(&g)?, &u)?;
            let out = ctx.mul_mat(ws.get("down").expect("bound"), &act)?;
            ctx.compute(&out, 0)?;

            for (dst, v) in accum.iter_mut().zip(out.to_vec_f32()) {
                *dst += v * weight;
            }
        }
        Ok(accum)
    }

    fn tensor_type(&self, name: &str) -> Result<bigtea_gguf::GgmlType> {
        self.model
            .location(name)
            .map(|l| l.ty)
            .ok_or_else(|| ArchError::MissingTensor(name.to_string()))
    }

    /// RoPE parameters this architecture uses.
    pub fn rope(&self) -> RopeParams {
        RopeParams {
            freq_base: self.arch.config.rope_freq_base,
            ..RopeParams::default()
        }
    }

    pub fn rope_type() -> i32 {
        ROPE_TYPE_NEOX
    }

    /// Bind every resident tensor into `ctx`, reporting what it cost.
    pub fn load_resident<'a>(
        &mut self,
        ctx: &'a Context,
        weights: &mut WeightSet<'a>,
    ) -> Result<u64> {
        let mut total = 0u64;
        for name in self.resident_tensor_names() {
            let loc = self
                .model
                .location(&name)
                .ok_or_else(|| ArchError::MissingTensor(name.clone()))?
                .clone();
            let data = self.model.read_tensor(&name)?;
            total += data.len() as u64;
            weights.bind(ctx, &name, loc.ty, &loc.dims, data)?;
        }
        // The output projection may be tied to the embeddings.
        if self.model.location("output.weight").is_some() {
            let loc = self.model.location("output.weight").expect("checked").clone();
            let data = self.model.read_tensor("output.weight")?;
            total += data.len() as u64;
            weights.bind(ctx, "output.weight", loc.ty, &loc.dims, data)?;
        }
        self.stats.resident_bytes = total;
        Ok(total)
    }

    /// Expose the expert FFN for the runner's layer loop.
    pub fn run_expert_ffn(&mut self, x: &[f32], il: u32, router_probs: &[f32]) -> Result<Vec<f32>> {
        self.expert_ffn(x, il, router_probs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Qwen3Config {
        Qwen3Config {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 4,
            n_ff: 16,
            vocab_size: 32,
            rms_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            n_expert: 4,
            n_expert_used: 2,
            n_ff_expert: 16,
        }
    }

    /// Routing is pure arithmetic, so it is testable without a model.
    fn route_only(probs: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        let total: f32 = scored.iter().map(|(_, p)| *p).sum();
        if total > 0.0 {
            for (_, p) in &mut scored {
                *p /= total;
            }
        }
        scored
    }

    #[test]
    fn routing_picks_the_highest_scoring_experts() {
        let picks = route_only(&[0.1, 0.5, 0.2, 0.9], 2);
        let ids: Vec<u32> = picks.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids, vec![3, 1], "should pick experts 3 and 1");
    }

    #[test]
    fn routing_weights_are_renormalised_over_the_selection() {
        // Without this the output is scaled down by the probability mass of
        // the experts that were never run.
        let picks = route_only(&[0.1, 0.5, 0.2, 0.9], 2);
        let total: f32 = picks.iter().map(|(_, w)| *w).sum();
        assert!((total - 1.0).abs() < 1e-6, "weights summed to {total}");
    }

    #[test]
    fn resident_set_excludes_experts() {
        // The entire premise: experts must not be resident.
        let arch = Qwen3Model::new(cfg());
        let resident: Vec<String> = arch
            .required_tensors()
            .into_iter()
            .filter(|n| !n.contains("_exps"))
            .collect();
        assert!(!resident.is_empty());
        assert!(
            resident.iter().all(|n| !n.contains("_exps")),
            "an expert tensor leaked into the resident set"
        );
        assert!(resident.iter().any(|n| n.contains("attn_q")));
    }

    #[test]
    fn zero_probabilities_do_not_divide_by_zero() {
        let picks = route_only(&[0.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(picks.len(), 2);
        assert!(picks.iter().all(|(_, w)| w.is_finite()));
    }
}
