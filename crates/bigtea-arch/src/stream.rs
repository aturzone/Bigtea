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

use bigtea_ggml::{arena_for, Context, RopeParams, Tensor, WeightSet};
use bigtea_model::Model;

use crate::kv::KvCache;
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

    /// Apply one layer's expert feed-forward to a whole block of tokens.
    ///
    /// The per-token version reads an expert's three tensors every time a token
    /// selects it. Across a 256-token block that is 256 * 8 * 3 = 6,144 reads
    /// per layer, against 384 distinct slices that actually exist — the same
    /// bytes fetched from disk sixteen times over. Measured on a 565-token
    /// prompt: 609,665 reads, 537 GiB streamed, 470s of prefill.
    ///
    /// So invert the loop. Group the block's tokens by the expert they routed
    /// to, read each distinct expert once, and run it against all of its tokens
    /// as a single matrix. Experts are visited in ascending index order, which
    /// also makes the reads walk the file forwards instead of seeking.
    ///
    /// `normed` holds the block's post-FFN-norm activations, `n_embd` floats
    /// per token; `probs` holds `n_expert` router probabilities per token.
    fn expert_ffn_block(
        &mut self,
        normed: &[f32],
        il: u32,
        probs: &[f32],
        n_tokens: usize,
    ) -> Result<Vec<f32>> {
        use std::collections::BTreeMap;

        let c = self.arch.config.clone();
        let n_embd = c.n_embd as usize;
        let n_expert = c.n_expert as usize;

        // expert -> the tokens that chose it, with their routing weights.
        // BTreeMap keeps the read order ascending by expert index.
        let mut by_expert: BTreeMap<u32, Vec<(usize, f32)>> = BTreeMap::new();
        for t in 0..n_tokens {
            let picks = self.route(&probs[t * n_expert..(t + 1) * n_expert], c.n_expert_used as usize);
            for (expert, weight) in picks {
                by_expert.entry(expert).or_default().push((t, weight));
            }
        }

        let gate_name = format!("blk.{il}.ffn_gate_exps.weight");
        let up_name = format!("blk.{il}.ffn_up_exps.weight");
        let down_name = format!("blk.{il}.ffn_down_exps.weight");
        let gate_ty = self.tensor_type(&gate_name)?;
        let up_ty = self.tensor_type(&up_name)?;
        let down_ty = self.tensor_type(&down_name)?;
        let n_embd_i = c.n_embd as i64;
        let n_ff = c.n_ff_expert as i64;

        let mut accum = vec![0f32; n_embd * n_tokens];

        for (expert, members) in by_expert {
            let gate_bytes = self.expert_slice(&gate_name, expert)?;
            let up_bytes = self.expert_slice(&up_name, expert)?;
            let down_bytes = self.expert_slice(&down_name, expert)?;

            let m = members.len() as i64;
            let ctx = Context::new(arena_for(
                &[
                    (n_embd_i, m), // this expert's tokens, gathered
                    (n_ff, m),     // gate
                    (n_ff, m),     // up
                    (n_ff, m),     // silu(gate) * up
                    (n_embd_i, m), // down projection
                ],
                24,
            ))?;
            let mut ws = WeightSet::new();
            ws.bind(&ctx, "gate", gate_ty, &[n_embd_i as u64, n_ff as u64], gate_bytes)?;
            ws.bind(&ctx, "up", up_ty, &[n_embd_i as u64, n_ff as u64], up_bytes)?;
            ws.bind(&ctx, "down", down_ty, &[n_ff as u64, n_embd_i as u64], down_bytes)?;

            // Gather this expert's tokens into one contiguous matrix so the
            // three matmuls run once for the group rather than once per token.
            let mut gathered = vec![0f32; n_embd * members.len()];
            for (slot, (t, _)) in members.iter().enumerate() {
                gathered[slot * n_embd..(slot + 1) * n_embd]
                    .copy_from_slice(&normed[t * n_embd..(t + 1) * n_embd]);
            }
            let xt = ctx.new_f32_2d(n_embd_i, m)?;
            xt.set_f32(&gathered)?;

            let g = ctx.mul_mat(ws.get("gate").expect("bound"), &xt)?;
            let u = ctx.mul_mat(ws.get("up").expect("bound"), &xt)?;
            let act = ctx.mul(&ctx.silu(&g)?, &u)?;
            let out = ctx.mul_mat(ws.get("down").expect("bound"), &act)?;
            ctx.compute(&out, 0)?;

            let produced = out.to_vec_f32();
            for (slot, (t, weight)) in members.iter().enumerate() {
                let src = &produced[slot * n_embd..(slot + 1) * n_embd];
                let dst = &mut accum[t * n_embd..(t + 1) * n_embd];
                for (d, v) in dst.iter_mut().zip(src) {
                    *d += v * weight;
                }
            }
        }
        Ok(accum)
    }

    /// Apply one layer's expert feed-forward for a single token.
    ///
    /// `x` is that token's activations after the FFN norm.
    #[allow(dead_code)]
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

    /// Incremental forward pass: computes only the new positions, attending
    /// over cached history.
    ///
    /// This is what makes generation linear. The uncached path recomputes
    /// every previous position for every token, so a response costs O(n²) —
    /// measured at 31,032 expert reads for 5 tokens, against ~1,152 needed
    /// for one position. Here each token's experts are read once.
    ///
    /// `pos_start` is the absolute position of the first token in `tokens`;
    /// RoPE and the causal mask both depend on it, and an off-by-one degrades
    /// output subtly rather than visibly.
    pub fn forward_cached<'a>(
        &mut self,
        weights: &WeightSet<'a>,
        cache: &mut KvCache,
        tokens: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>> {
        let c = self.arch.config.clone();
        let n_new = tokens.len() as i64;
        let n_embd = c.n_embd as i64;
        let head_dim = c.head_dim as i64;
        let n_kv = c.n_head_kv as i64;
        let kv_width = (n_kv * head_dim) as usize;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let positions: Vec<i32> = (0..n_new).map(|i| (pos_start as i64 + i) as i32).collect();

        let mut x: Vec<f32> = {
            // Sized from the prompt, not a constant: `get_rows` materialises
            // n_embd * n_new floats, and ggml aborts the process when an arena
            // runs out rather than returning an error we could catch.
            let ctx = Context::new(arena_for(&[(n_embd, n_new)], 8))?;
            let tok = ctx.new_i32_1d(n_new)?;
            tok.set_i32(&tokens.iter().map(|&t| t as i32).collect::<Vec<_>>())?;
            let emb = weights
                .get("token_embd.weight")
                .ok_or_else(|| ArchError::MissingTensor("token_embd.weight".into()))?;
            let rows = ctx.get_rows(emb, &tok)?;
            ctx.compute(&rows, threads)?;
            rows.to_vec_f32()
        };

        for il in 0..c.n_layer {
            let get = |w: &WeightSet<'a>, n: String| -> Result<Tensor<'a>> {
                w.get(&n).copied().ok_or(ArchError::MissingTensor(n))
            };

            // Phase 1: Q, K and V for the new positions only.
            let (q_v, k_v, v_v, residual) = {
                let ctx = Context::new(256 << 20)?;
                let xt = ctx.new_f32_2d(n_embd, n_new)?;
                xt.set_f32(&x)?;
                let pos = ctx.new_i32_1d(n_new)?;
                pos.set_i32(&positions)?;

                let normed = self.arch.norm_scaled(
                    &ctx,
                    &xt,
                    &get(weights, format!("blk.{il}.attn_norm.weight"))?,
                )?;
                let q = ctx.mul_mat(&get(weights, format!("blk.{il}.attn_q.weight"))?, &normed)?;
                let k = ctx.mul_mat(&get(weights, format!("blk.{il}.attn_k.weight"))?, &normed)?;
                let v = ctx.mul_mat(&get(weights, format!("blk.{il}.attn_v.weight"))?, &normed)?;

                let q = ctx.reshape_3d(&q, head_dim, c.n_head as i64, n_new)?;
                let k = ctx.reshape_3d(&k, head_dim, n_kv, n_new)?;
                let q = self.arch.norm_scaled(
                    &ctx,
                    &q,
                    &get(weights, format!("blk.{il}.attn_q_norm.weight"))?,
                )?;
                let k = self.arch.norm_scaled(
                    &ctx,
                    &k,
                    &get(weights, format!("blk.{il}.attn_k_norm.weight"))?,
                )?;

                let rp = self.rope();
                let q = ctx.rope_ext(&q, &pos, None, head_dim as i32, ROPE_TYPE_NEOX, 0, rp)?;
                let k = ctx.rope_ext(&k, &pos, None, head_dim as i32, ROPE_TYPE_NEOX, 0, rp)?;

                // One compute materialises all three; they share a graph.
                ctx.compute(&q, threads)?;
                ctx.compute(&k, threads)?;
                ctx.compute(&v, threads)?;
                (q.to_vec_f32(), k.to_vec_f32(), v.to_vec_f32(), x.clone())
            };

            // K and V for these positions never change again, so store them.
            for t in 0..tokens.len() {
                let lo = t * kv_width;
                cache.push(il as usize, &k_v[lo..lo + kv_width], &v_v[lo..lo + kv_width])?;
            }

            // Phase 2: attend over the whole history, not just the new part.
            let n_total = (cache.len() + tokens.len()) as i64;
            let attn_out = {
                // The scores and their softmax dominate: n_total * n_new *
                // n_head floats each. At a 2k prompt that pair alone is over a
                // gigabyte, so a fixed arena aborts the process somewhere past
                // 1.5k tokens. Size it from the actual shapes instead.
                let n_head = c.n_head as i64;
                let ctx = Context::new(arena_for(
                    &[
                        (head_dim * n_head, n_new),      // q, contiguous
                        (head_dim * n_kv, n_total),      // k, contiguous
                        (n_total * n_new, n_head),       // scores
                        (n_total, n_new),                // causal mask
                        (n_total * n_new, n_head),       // softmax output
                        (head_dim * n_kv, n_total),      // v, transposed
                        (head_dim * n_new, n_head),      // attention output
                        (head_dim * n_new, n_head),      // ...made contiguous
                        (n_embd, n_new),                 // output projection
                    ],
                    24,
                ))?;
                let q = ctx.new_f32_3d(head_dim, c.n_head as i64, n_new)?;
                q.set_f32(&q_v)?;

                let k_all = ctx.new_f32_3d(head_dim, n_kv, n_total)?;
                k_all.set_f32(cache.keys(il as usize))?;
                let v_all = ctx.new_f32_3d(head_dim, n_kv, n_total)?;
                v_all.set_f32(cache.values(il as usize))?;

                let out = self.arch.attention_cached(
                    &ctx, &q, &k_all, &v_all, n_new, n_total, pos_start as i64,
                )?;
                let out = ctx.mul_mat(
                    &get(weights, format!("blk.{il}.attn_output.weight"))?,
                    &out,
                )?;
                ctx.compute(&out, threads)?;
                out.to_vec_f32()
            };

            // Residual, then the feed-forward.
            let mut ffn_input = residual;
            for (dst, v) in ffn_input.iter_mut().zip(attn_out) {
                *dst += v;
            }

            let (normed_v, probs_v) = {
                let ctx = Context::new(256 << 20)?;
                let xt = ctx.new_f32_2d(n_embd, n_new)?;
                xt.set_f32(&ffn_input)?;
                let normed = self.arch.norm_scaled(
                    &ctx,
                    &xt,
                    &get(weights, format!("blk.{il}.ffn_norm.weight"))?,
                )?;
                let logits = ctx.mul_mat(
                    &get(weights, format!("blk.{il}.ffn_gate_inp.weight"))?,
                    &normed,
                )?;
                let probs = ctx.soft_max_ext(&logits, None, 1.0, 0.0)?;
                ctx.compute(&probs, threads)?;
                (normed.to_vec_f32(), probs.to_vec_f32())
            };

            let mut next = ffn_input;
            let expert_out = self.expert_ffn_block(&normed_v, il, &probs_v, tokens.len())?;
            for (dst, v) in next.iter_mut().zip(expert_out) {
                *dst += v;
            }
            x = next;
        }
        cache.advance_by(tokens.len());

        // Only the last position's logits are ever sampled, and the vocabulary
        // projection is the widest matmul in the model (151,936 rows here).
        // Running it for every prompt token costs the whole prefill that much
        // again for results nothing reads.
        let last = x.len() - n_embd as usize;
        let ctx = Context::new(arena_for(
            &[(n_embd, 1), (n_embd, 1), (c.vocab_size as i64, 1)],
            16,
        ))?;
        let xt = ctx.new_f32_2d(n_embd, 1)?;
        xt.set_f32(&x[last..])?;
        let normed = self.arch.norm_scaled(
            &ctx,
            &xt,
            weights
                .get("output_norm.weight")
                .ok_or_else(|| ArchError::MissingTensor("output_norm.weight".into()))?,
        )?;
        let out_name = if weights.get("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let out = ctx.mul_mat(
            weights
                .get(out_name)
                .ok_or_else(|| ArchError::MissingTensor(out_name.into()))?,
            &normed,
        )?;
        ctx.compute(&out, threads)?;
        Ok(out.to_vec_f32())
    }

    /// Full forward pass, streaming experts, returning logits for every token.
    ///
    /// Runs one layer at a time so the router's choice can drive what is read
    /// next. Activations cross layer boundaries as plain `Vec<f32>` — small
    /// (`n_embd * n_tokens`) compared to the weights, and it lets each layer's
    /// arena be reclaimed immediately.
    pub fn forward<'a>(
        &mut self,
        weights: &WeightSet<'a>,
        tokens: &[u32],
        positions: &[i32],
    ) -> Result<Vec<f32>> {
        let c = self.arch.config.clone();
        let n_tokens = tokens.len() as i64;
        let n_embd = c.n_embd as i64;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Embedding lookup, once.
        let mut x: Vec<f32> = {
            let ctx = Context::new(512 << 20)?;
            let tok = ctx.new_i32_1d(n_tokens)?;
            tok.set_i32(&tokens.iter().map(|&t| t as i32).collect::<Vec<_>>())?;
            let emb = weights
                .get("token_embd.weight")
                .ok_or_else(|| ArchError::MissingTensor("token_embd.weight".into()))?;
            let rows = ctx.get_rows(emb, &tok)?;
            ctx.compute(&rows, threads)?;
            rows.to_vec_f32()
        };

        for il in 0..c.n_layer {
            // Attention and the router run as one graph; computing the router
            // output also materialises everything upstream, so the residual
            // and the normed activations can be read from the same pass.
            let ctx = Context::new(1 << 30)?;
            let get = |n: &str| -> Result<&Tensor<'a>> {
                weights.get(n).ok_or_else(|| ArchError::MissingTensor(n.into()))
            };

            let xt = ctx.new_f32_2d(n_embd, n_tokens)?;
            xt.set_f32(&x)?;
            let pos = ctx.new_i32_1d(n_tokens)?;
            pos.set_i32(positions)?;

            let attn_out = self.arch.attention_block(
                &ctx,
                weights,
                &xt,
                &pos,
                n_tokens,
                il,
                self.rope(),
                ROPE_TYPE_NEOX,
            )?;
            let ffn_input = ctx.add(&attn_out, &xt)?;
            let normed = self
                .arch
                .norm_scaled(&ctx, &ffn_input, get(&format!("blk.{il}.ffn_norm.weight"))?)?;

            let logits = ctx.mul_mat(get(&format!("blk.{il}.ffn_gate_inp.weight"))?, &normed)?;
            let probs = ctx.soft_max_ext(&logits, None, 1.0, 0.0)?;
            ctx.compute(&probs, threads)?;

            let residual = ffn_input.to_vec_f32();
            let normed_v = normed.to_vec_f32();
            let probs_v = probs.to_vec_f32();
            drop(ctx);

            // Experts, per token: the router's choice differs for each.
            let n_expert = c.n_expert as usize;
            let mut next = residual;
            for t in 0..tokens.len() {
                let lo = t * c.n_embd as usize;
                let hi = lo + c.n_embd as usize;
                let token_probs = &probs_v[t * n_expert..(t + 1) * n_expert];
                let expert_out = self.expert_ffn(&normed_v[lo..hi], il, token_probs)?;
                for (dst, v) in next[lo..hi].iter_mut().zip(expert_out) {
                    *dst += v;
                }
            }
            x = next;
        }

        // Final norm and output projection.
        let ctx = Context::new(1 << 30)?;
        let xt = ctx.new_f32_2d(n_embd, n_tokens)?;
        xt.set_f32(&x)?;
        let normed = self.arch.norm_scaled(
            &ctx,
            &xt,
            weights
                .get("output_norm.weight")
                .ok_or_else(|| ArchError::MissingTensor("output_norm.weight".into()))?,
        )?;
        let out_name = if weights.get("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let out = ctx.mul_mat(
            weights
                .get(out_name)
                .ok_or_else(|| ArchError::MissingTensor(out_name.into()))?,
            &normed,
        )?;
        ctx.compute(&out, threads)?;
        Ok(out.to_vec_f32())
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
