//! The DeepSeek-V4-Flash forward pass, as library code rather than as a test.
//!
//! Every step here was built checkpoint-by-checkpoint against `llama.cpp`'s own
//! element-sums — see `tests/deepseek4_forward.rs`, which now checks *this*
//! code rather than a copy of it. That distinction matters: the port spent its
//! first weeks with the implementation living inside the test, which meant the
//! library shipped nothing and the verification proved only that the test
//! agreed with itself.
//!
//! # What is verified, and at which prompt length
//!
//! **Prompt length decides which code paths run**, because the compressed
//! attention builders are guarded on their compressed caches being non-empty.
//! The same layer runs different attention at different lengths:
//!
//! | tokens | layers 0-1 | even layers | odd layers |
//! |---|---|---|---|
//! | ≤3 | Raw | Raw (fallback) | Raw (fallback) |
//! | 5 | Raw | Compressed Sparse | Raw (fallback) |
//! | ≥128 | Raw | Compressed Sparse | Heavily Compressed |
//!
//! All three are checked against llama.cpp: the whole 43-block stack at 2
//! tokens, and layers 0-3 at 165 where both compressed kinds fire.
//!
//! # The one deliberate omission
//!
//! # STATUS: PARTIAL PORT, NOT YET WIRED IN
//!
//! Ported and compiling: the embedding, the hyper-connection gates, the block
//! entry, Q/KV with per-layer RoPE, both compressors, and attention with the
//! sliding window and the optional compressed key half.
//!
//! **Still living only in `tests/deepseek4_forward.rs`**: the layer tail (post
//! hyper-connection, FFN gates, `ffn_norm`), both MoE routing schemes, the
//! routed experts, the shared expert, the output head, and the block loop.
//! Until those move, `bigtea-run` cannot run this model and the test still
//! verifies its own copy rather than this one. That is the remaining work for
//! a 0.0.0, and it is translation rather than discovery — every piece is
//! already checked against llama.cpp on the test side.
//!
//! # The one deliberate omission
//!
//! **The lightning indexer is not run**, and below ~2048 tokens that is exact
//! rather than approximate: `n_top_k = min(n_lid, indexer_top_k)` selects
//! *every* compressed slot, so the indexer's mask is precisely the visibility
//! mask and cannot change any output. Above that length this becomes an
//! approximation and [`Deepseek4Forward::indexer_is_exact`] returns false.

use bigtea_ggml::{Context, RopeParams, Tensor, WeightSet};
use bigtea_model::Model;

use crate::{AttentionKind, Deepseek4Config, Deepseek4Model, Result};

/// `LLAMA_ROPE_TYPE_NORM`: rotated pairs are adjacent, not offset by `n_rot/2`.
const ROPE_MODE_NORM: i32 = 0;

/// F16 `-inf`, written as bits. Mask values are only ever 0 or -inf, so writing
/// the pattern beats converting.
const F16_NEG_INF: [u8; 2] = [0x00, 0xFC];

/// The padded key window each half of the cache occupies.
const N_KV: i64 = 256;

/// One block's forward pass, and the state it threads.
pub struct Deepseek4Forward<'m> {
    model: &'m Model,
    config: Deepseek4Config,
    arch: Deepseek4Model,
}

impl<'m> Deepseek4Forward<'m> {
    pub fn new(model: &'m Model, config: Deepseek4Config) -> Self {
        let arch = Deepseek4Model::new(config.clone());
        Deepseek4Forward { model, config, arch }
    }

    pub fn config(&self) -> &Deepseek4Config {
        &self.config
    }

    /// Whether skipping the lightning indexer is exact at this prompt length.
    ///
    /// It is, until the compressed cache holds more entries than the indexer
    /// would keep — `n_top_k = min(n_lid, indexer_top_k)`. Below that the
    /// indexer selects everything and changes nothing.
    pub fn indexer_is_exact(&self, n_tokens: usize) -> bool {
        let blocks = n_tokens as i64 / Deepseek4Config::CSA_RATIO;
        blocks.min(N_KV) <= self.config.indexer_top_k as i64
    }

    /// Tensor names one block needs, plus the globals for block 0.
    pub fn block_tensor_names(&self, il: u32) -> Vec<String> {
        let mut names = Vec::new();
        for suffix in [
            "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
            "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
            "attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b",
            "attn_kv", "attn_kv_a_norm", "attn_sinks",
            "attn_output_a", "attn_output_b",
            "ffn_norm", "ffn_gate_inp",
            "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
        ] {
            names.push(format!("blk.{il}.{suffix}.weight"));
        }
        // Only some blocks carry these; the two routing schemes are mutually
        // exclusive and a compressor is absent on the two Raw layers.
        for suffix in [
            "ffn_gate_tid2eid.weight",
            "exp_probs_b.bias",
            "attn_compressor_kv.weight",
            "attn_compressor_gate.weight",
            "attn_compressor_ape.weight",
            "attn_compressor_norm.weight",
        ] {
            let n = format!("blk.{il}.{suffix}");
            if self.model.location(&n).is_some() {
                names.push(n);
            }
        }
        names
    }

    /// RoPE for `il`, from the shipped per-layer selection.
    fn rope(&self, il: u32) -> (RopeParams, i32) {
        let r = self.config.rope_for_layer(il);
        (r.params, r.n_ctx_orig)
    }
}

/// The four residual streams between blocks, as plain floats.
///
/// Handing the boundary across as a `Vec` is what lets each block own its arena
/// and drop it: freeing weights *inside* one `ggml` context is unsound, because
/// every `compute` rebuilds the graph through its sources and a dropped buffer
/// becomes a dangling pointer that reads freed memory successfully.
pub type Streams = Vec<f32>;

/// Build `hc_init`: the embedding repeated across the hyper-connection streams.
pub fn embed<'c>(
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    tokens: &[i32],
) -> Result<Tensor<'c>> {
    let nt = tokens.len() as i64;
    let hc = config.hc_mult as i64;
    let tok = ctx.new_i32_1d(nt)?;
    tok.set_i32(tokens)?;
    let embd = ctx.get_rows(weights.get("token_embd.weight").expect("bound"), &tok)?;
    ctx.compute(&embd, 12)?;
    let embd_r = ctx.reshape_3d(&embd, config.n_embd as i64, 1, nt)?;
    let shape = ctx.new_f32_3d(config.n_embd as i64, hc, nt)?;
    let hc_init = ctx.repeat(&embd_r, &shape)?;
    ctx.compute(&hc_init, 12)?;
    Ok(hc_init)
}

/// The three gates one `build_hc_pre` call produces, all from one mixes matmul.
struct HcGates<'c> {
    pre: Tensor<'c>,
    post: Tensor<'c>,
    comb: Tensor<'c>,
}

/// Slice the 24 mixes into the three gates.
///
/// Layout is `[0..hc]` pre, `[hc..2hc]` post, then the combination matrix, with
/// `hc_scale` indexed `[pre, post, comb]`. **Every one of those views is the
/// right size whichever slice you take**, so wrong offsets have no shape
/// consequence at all. `pre` ends with `scale_bias(x, 1, eps)` and `post` with
/// `scale(x, 2.0)` — different tails, same shape.
fn hc_gates<'c>(
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    prefix: &str,
    mixes: &Tensor<'c>,
    nt: i64,
) -> Result<HcGates<'c>> {
    let hc = config.hc_mult as i64;
    let f32_size = std::mem::size_of::<f32>();
    // The stride is the *source's* row — `(2 + hc) * hc` floats — not the 4 the
    // view is wide. At one token the stride is never traversed, so any value
    // passes; only a multi-token prompt pins it.
    let stride = ((2 + hc) * hc) as usize * f32_size;
    let scale_w = weights.get(&format!("{prefix}_scale.weight")).expect("bound");
    let base_w = weights.get(&format!("{prefix}_base.weight")).expect("bound");

    let gate = |mix_off: i64, scale_idx: usize, base_off: i64| -> Result<Tensor<'c>> {
        let view = ctx.view_2d(mixes, hc, nt, stride, mix_off as usize * f32_size)?;
        let s = ctx.view_1d(scale_w, 1, scale_idx * f32_size)?;
        let b = ctx.view_1d(base_w, hc, base_off as usize * f32_size)?;
        let scaled = ctx.mul(&view, &s)?;
        let biased = ctx.add(&scaled, &b)?;
        Ok(ctx.sigmoid(&biased)?)
    };

    let pre_gated = gate(0, 0, 0)?;
    let eps = ctx.new_f32_1d(hc)?;
    eps.set_f32(&vec![1e-6f32; hc as usize])?;
    let pre = ctx.add(&pre_gated, &eps)?;
    ctx.compute(&pre, 12)?;

    let post_gated = gate(hc, 1, hc)?;
    let post = ctx.scale(&post_gated, 2.0)?;
    ctx.compute(&post, 12)?;

    let comb = ctx.dsv4_hc_comb(
        mixes,
        scale_w,
        base_w,
        1e-6,
        config.hc_sinkhorn_iterations as i32,
    )?;
    ctx.compute(&comb, 12)?;
    Ok(HcGates { pre, post, comb })
}

/// A block's entry: hyper-connection gates and `attn_norm`, from whatever
/// residual streams it was handed.
///
/// Block 0 reaches here from the embedding, every other block from the previous
/// block's output. **That is the only structural difference between the first
/// block and the rest.**
struct Entry<'c> {
    streams: Tensor<'c>,
    attn_norm: Tensor<'c>,
    gates: HcGates<'c>,
}

fn entry<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    streams: Tensor<'c>,
    nt: i64,
) -> Result<Entry<'c>> {
    let config = &fw.config;
    let flat = ctx.reshape_2d(&streams, config.hc_dim() as i64, nt)?;
    let normed = ctx.rms_norm(&flat, config.rms_eps)?;
    let mixes = ctx.mul_mat(
        weights.get(&format!("blk.{il}.hc_attn_fn.weight")).expect("bound"),
        &normed,
    )?;
    ctx.compute(&mixes, 12)?;
    let gates = hc_gates(ctx, weights, config, &format!("blk.{il}.hc_attn"), &mixes, nt)?;

    let collapsed = ctx.dsv4_hc_pre(&streams, &gates.pre)?;
    let normed = ctx.rms_norm(&collapsed, config.rms_eps)?;
    let attn_norm = ctx.mul(
        &normed,
        weights.get(&format!("blk.{il}.attn_norm.weight")).expect("bound"),
    )?;
    ctx.compute(&attn_norm, 12)?;
    Ok(Entry { streams, attn_norm, gates })
}

/// Q and KV, both low-rank, both with only their trailing `n_rot` dims rotated.
///
/// `kv` becomes K **and** V — there is no separate V projection, which is why
/// `head_count_kv` is 1. The per-head norm on `q` carries **no weight**, unlike
/// every other norm in this model.
fn q_and_kv<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    attn_norm: &Tensor<'c>,
    nt: i64,
) -> Result<(Tensor<'c>, Tensor<'c>)> {
    let config = &fw.config;
    let head = config.kv_lora_rank as i64;
    let n_head = config.n_head as i64;
    let n_rot = config.n_rot as i64;
    let n_nope = config.n_rot_none() as i64;
    let f32_size = std::mem::size_of::<f32>();
    let hs = head as usize * f32_size;
    let (rope, rope_orig) = fw.rope(il);

    let pos = ctx.new_i32_1d(nt)?;
    pos.set_i32(&(0..nt as i32).collect::<Vec<i32>>())?;

    let qr = ctx.mul_mat(
        weights.get(&format!("blk.{il}.attn_q_a.weight")).expect("bound"),
        attn_norm,
    )?;
    let qr = ctx.rms_norm(&qr, config.rms_eps)?;
    let qr = ctx.mul(
        &qr,
        weights.get(&format!("blk.{il}.attn_q_a_norm.weight")).expect("bound"),
    )?;
    let q = ctx.mul_mat(
        weights.get(&format!("blk.{il}.attn_q_b.weight")).expect("bound"),
        &qr,
    )?;
    let q = ctx.reshape_3d(&q, head, n_head, nt)?;
    let q = ctx.rms_norm(&q, config.rms_eps)?; // unweighted, deliberately
    ctx.compute(&q, 12)?;

    let q_nope = ctx.view_3d(&q, n_nope, n_head, nt, hs, hs * n_head as usize, 0)?;
    let q_pe_in = ctx.view_3d(
        &q, n_rot, n_head, nt, hs, hs * n_head as usize, n_nope as usize * f32_size,
    )?;
    let q_pe = ctx.rope_ext(&q_pe_in, &pos, None, n_rot as i32, ROPE_MODE_NORM, rope_orig, rope)?;
    let q_full = ctx.concat(&q_nope, &q_pe, 0)?;
    ctx.compute(&q_full, 12)?;

    let kv = ctx.mul_mat(
        weights.get(&format!("blk.{il}.attn_kv.weight")).expect("bound"),
        attn_norm,
    )?;
    let kv = ctx.rms_norm(&kv, config.rms_eps)?;
    let kv = ctx.mul(
        &kv,
        weights.get(&format!("blk.{il}.attn_kv_a_norm.weight")).expect("bound"),
    )?;
    let kv = ctx.reshape_3d(&kv, head, 1, nt)?;
    ctx.compute(&kv, 12)?;
    let kv_nope = ctx.view_3d(&kv, n_nope, 1, nt, hs, hs, 0)?;
    let kv_pe_in = ctx.view_3d(&kv, n_rot, 1, nt, hs, hs, n_nope as usize * f32_size)?;
    let kv_pe = ctx.rope_ext(&kv_pe_in, &pos, None, n_rot as i32, ROPE_MODE_NORM, rope_orig, rope)?;
    let kv_full = ctx.concat(&kv_nope, &kv_pe, 0)?;
    ctx.compute(&kv_full, 12)?;
    Ok((q_full, kv_full))
}

/// The overlap compressor (CSA) or the plain one (HCA), for a prefill.
///
/// Both summarise completed blocks of raw KV into one entry each. They differ
/// in more than a ratio: the overlap form keeps a state `2*n_embd_head` wide and
/// averages over **two** windows (`ratio` previous plus `ratio` current), while
/// the plain form is head-wide and uses the current window only.
///
/// The persistent ring llama.cpp maintains is not needed on a prefill:
/// `state_source_idx` resolves to an appended zero row for `pos < 0` and to the
/// current batch otherwise, so the ring is never read.
fn compressor<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    attn_norm: &Tensor<'c>,
    nt: i64,
    overlap: bool,
) -> Result<Tensor<'c>> {
    let config = &fw.config;
    let head = config.kv_lora_rank as i64;
    let ratio = config.compress_block(il).expect("compressed layer");
    let n_blocks = nt / ratio;
    let wide = if overlap { 2 * head } else { head };
    let n_read = ratio * n_blocks;
    let state_rows = if overlap { 8 } else { ratio };

    let kv = ctx.mul_mat(
        weights.get(&format!("blk.{il}.attn_compressor_kv.weight")).expect("bound"),
        attn_norm,
    )?;
    let score = ctx.mul_mat(
        weights.get(&format!("blk.{il}.attn_compressor_gate.weight")).expect("bound"),
        attn_norm,
    )?;
    // The gate's position embedding is indexed by the token's offset *within its
    // block*, not by its absolute position.
    let pos_t = ctx.new_i32_1d(nt)?;
    pos_t.set_i32(&(0..nt).map(|p| (p % ratio) as i32).collect::<Vec<i32>>())?;
    let ape = ctx.get_rows(
        weights.get(&format!("blk.{il}.attn_compressor_ape.weight")).expect("bound"),
        &pos_t,
    )?;
    let score = ctx.add(&score, &ape)?;
    ctx.compute(&kv, 12)?;
    ctx.compute(&score, 12)?;

    let pad = if overlap { 1 } else { 0 };
    let total = state_rows + nt + pad;
    let kv_vals = kv.to_vec_f32();
    let score_vals = score.to_vec_f32();
    let mut kv_buf = vec![0.0f32; (state_rows * wide) as usize];
    kv_buf.extend_from_slice(&kv_vals);
    kv_buf.extend(std::iter::repeat(0.0f32).take((pad * wide) as usize));
    let kv_state = ctx.new_f32_2d(wide, total)?;
    kv_state.set_f32(&kv_buf)?;
    let mut sc_buf = vec![0.0f32; (state_rows * wide) as usize];
    sc_buf.extend_from_slice(&score_vals);
    // -inf so the softmax ignores the padding rather than averaging it in.
    sc_buf.extend(std::iter::repeat(f32::NEG_INFINITY).take((pad * wide) as usize));
    let score_state = ctx.new_f32_2d(wide, total)?;
    score_state.set_f32(&sc_buf)?;

    let zero_row = (state_rows + nt) as i32;
    let mut idxs: Vec<i32> = Vec::new();
    if overlap {
        for b in 0..n_blocks {
            for j in 0..ratio {
                let p = b * ratio - ratio + j;
                idxs.push(if p < 0 { zero_row } else { (state_rows + p) as i32 });
            }
        }
    }
    for b in 0..n_blocks {
        for j in 0..ratio {
            idxs.push((state_rows + b * ratio + j) as i32);
        }
    }
    let idx_t = ctx.new_i32_1d(idxs.len() as i64)?;
    idx_t.set_i32(&idxs)?;

    let f32_size = std::mem::size_of::<f32>();
    let row = wide as usize * f32_size;

    let mut halves = Vec::with_capacity(2);
    for src in [&kv_state, &score_state] {
        let rows = ctx.get_rows(src, &idx_t)?;
        let joined = if overlap {
            // The first `head` of one set of rows, and the *second* `head` of
            // the next: reading one entry per row summarises the wrong span.
            let prev = ctx.cont(&ctx.view_2d(&rows, head, n_read, row, 0)?)?;
            let cur = ctx.cont(&ctx.view_2d(
                &rows,
                head,
                n_read,
                row,
                n_read as usize * row + head as usize * f32_size,
            )?)?;
            let prev = ctx.reshape_3d(&prev, head, ratio, n_blocks)?;
            let cur = ctx.reshape_3d(&cur, head, ratio, n_blocks)?;
            ctx.concat(&prev, &cur, 1)?
        } else {
            ctx.reshape_3d(&rows, head, ratio, n_blocks)?
        };
        halves.push(ctx.cont(&ctx.permute(&joined, [1, 0, 2, 3])?)?);
    }
    let scores = halves.pop().expect("scores");
    let values = halves.pop().expect("values");

    let w = ctx.soft_max(&scores)?;
    let weighted = ctx.mul(&values, &w)?;
    let comp = ctx.sum_rows(&weighted)?;
    let comp = ctx.cont(&ctx.permute(&comp, [1, 0, 2, 3])?)?;
    let comp = ctx.rms_norm(&comp, config.rms_eps)?;
    let comp = ctx.mul(
        &comp,
        weights.get(&format!("blk.{il}.attn_compressor_norm.weight")).expect("bound"),
    )?;
    ctx.compute(&comp, 12)?;

    // Rotated at the *block start* position, with the compressed base.
    let n_rot = config.n_rot as i64;
    let n_nope = config.n_rot_none() as i64;
    let hs = head as usize * f32_size;
    let nope = ctx.view_3d(&comp, n_nope, 1, n_blocks, hs, hs, 0)?;
    let pe_in = ctx.view_3d(&comp, n_rot, 1, n_blocks, hs, hs, n_nope as usize * f32_size)?;
    let comp_pos = ctx.new_i32_1d(n_blocks)?;
    comp_pos.set_i32(&(0..n_blocks).map(|b| (b * ratio) as i32).collect::<Vec<i32>>())?;
    let (rope, rope_orig) = fw.rope(il);
    let pe = ctx.rope_ext(&pe_in, &comp_pos, None, n_rot as i32, ROPE_MODE_NORM, rope_orig, rope)?;
    let out = ctx.concat(&nope, &pe, 0)?;
    ctx.compute(&out, 12)?;
    Ok(out)
}

/// Attention over the raw window, and optionally the compressed summaries.
///
/// The raw half is causal **and sliding**: every layer's raw window is an SWA
/// window of `attention.sliding_window` (128). A plain causal mask passes on any
/// prompt shorter than the window and is wrong beyond it — which is exactly how
/// it went unnoticed until a 165-token capture. The compressed half is
/// visibility-limited instead: a token sees block `b` once that block is
/// complete and behind it.
fn attention<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    q_full: &Tensor<'c>,
    kv_full: &Tensor<'c>,
    comp: Option<&Tensor<'c>>,
    nt: i64,
) -> Result<Tensor<'c>> {
    let config = &fw.config;
    let head = config.kv_lora_rank as i64;
    let n_head = config.n_head as i64;
    let groups = config.output_group_count as i64;
    let n_rot = config.n_rot as i64;
    let n_nope = config.n_rot_none() as i64;
    let f32_size = std::mem::size_of::<f32>();

    let kv_vals = kv_full.to_vec_f32();
    let mut cache = vec![0u16; (head * N_KV) as usize];
    bigtea_ggml::f32_to_f16(&kv_vals, &mut cache[..kv_vals.len()]);
    let n_kv = match comp {
        None => N_KV,
        Some(c) => {
            let cv = c.to_vec_f32();
            let mut ch = vec![0u16; (head * N_KV) as usize];
            bigtea_ggml::f32_to_f16(&cv, &mut ch[..cv.len()]);
            cache.extend_from_slice(&ch);
            2 * N_KV
        }
    };
    let k = ctx.new_f16_3d(head, n_kv, 1)?;
    let bytes: Vec<u8> = cache.iter().flat_map(|h| h.to_le_bytes()).collect();
    k.set_bytes(&bytes)?;

    let ratio = config.compress_block(il).unwrap_or(1);
    let window = config.sliding_window as i64;
    let mut mask = vec![0u8; (n_kv * nt) as usize * 2];
    for query in 0..nt {
        let row = (query * n_kv) as usize * 2;
        for key in 0..N_KV {
            if key > query || (window > 0 && query - key >= window) {
                let at = row + key as usize * 2;
                mask[at..at + 2].copy_from_slice(&F16_NEG_INF);
            }
        }
        if comp.is_some() {
            for blk in ((query + 1) / ratio)..N_KV {
                let at = row + (N_KV + blk) as usize * 2;
                mask[at..at + 2].copy_from_slice(&F16_NEG_INF);
            }
        }
    }
    let mask_t = ctx.new_typed_2d(bigtea_gguf::GgmlType(1), n_kv, nt)?;
    mask_t.set_bytes(&mask)?;

    let q_perm = ctx.permute(q_full, [0, 2, 1, 3])?;
    let sinks = weights.get(&format!("blk.{il}.attn_sinks.weight")).expect("bound");
    let scale = 1.0f32 / (head as f32).sqrt();
    let out = ctx.flash_attn_ext_with_sinks(&q_perm, &k, &k, &mask_t, sinks, scale)?;
    ctx.compute(&out, 12)?;

    // The output is **de-roped** before projection. Skipping this leaves the
    // rotation baked into the residual stream, and no shape reveals it.
    let out = ctx.reshape_3d(&out, head, n_head, nt)?;
    let hs = head as usize * f32_size;
    let o_nope = ctx.view_3d(&out, n_nope, n_head, nt, hs, hs * n_head as usize, 0)?;
    let o_pe_in = ctx.view_3d(
        &out,
        n_rot,
        n_head,
        nt,
        hs,
        hs * n_head as usize,
        n_nope as usize * f32_size,
    )?;
    let pos = ctx.new_i32_1d(nt)?;
    pos.set_i32(&(0..nt as i32).collect::<Vec<i32>>())?;
    let (rope, rope_orig) = fw.rope(il);
    let o_pe =
        ctx.rope_ext_back(&o_pe_in, &pos, None, n_rot as i32, ROPE_MODE_NORM, rope_orig, rope)?;
    let out = ctx.concat(&o_nope, &o_pe, 0)?;

    // A batched matmul across `output_group_count` groups, not one matmul —
    // which is why the dimensions appear not to connect.
    let group_dim = n_head * head / groups;
    let out = ctx.reshape_3d(&out, group_dim, groups, nt)?;
    let out = ctx.cont(&ctx.permute(&out, [0, 2, 1, 3])?)?;
    let wo_a = weights.get(&format!("blk.{il}.attn_output_a.weight")).expect("bound");
    let wo_a = ctx.reshape_3d(wo_a, group_dim, config.output_lora_rank as i64, groups)?;
    let oa = ctx.mul_mat(&wo_a, &out)?;
    let oa = ctx.cont(&ctx.permute(&oa, [0, 2, 1, 3])?)?;
    let oa = ctx.reshape_2d(&oa, config.output_lora_rank as i64 * groups, nt)?;
    let out = ctx.mul_mat(
        weights.get(&format!("blk.{il}.attn_output_b.weight")).expect("bound"),
        &oa,
    )?;
    ctx.compute(&out, 12)?;
    Ok(out)
}
