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
use bigtea_io::SkewedBuf;
use bigtea_model::{Model, ResidentSet};

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
    /// Always-read weights held in RAM. `None` re-reads them per block, which
    /// is correct but costs 23% of a prefill and would cost it again on every
    /// generated token.
    resident: Option<&'m ResidentSet>,
}

impl<'m> Deepseek4Forward<'m> {
    pub fn new(model: &'m Model, config: Deepseek4Config) -> Self {
        let arch = Deepseek4Model::new(config.clone());
        Deepseek4Forward { model, config, arch, resident: None }
    }

    /// Serve always-read weights from `resident` instead of from disk.
    pub fn with_resident(mut self, resident: &'m ResidentSet) -> Self {
        self.resident = Some(resident);
        self
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

/// The block tail: write attention back across the streams, then the FFN's own
/// gate block and `ffn_norm`.
///
/// A plain transformer does `x = x + f(x)`. This does
/// `x[dst] = f(x)*post[dst] + sum_src x[src]*comb[dst, src]`, with `comb` a
/// Sinkhorn-normalised `hc x hc`. None of that changes a shape.
///
/// The FFN's gates come from a **second, independent** mixes matmul against
/// `hc_ffn_fn` over the post-attention streams — reusing the attention block's
/// would be free of any error.
fn layer_tail<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    e: &Entry<'c>,
    attn_out: &Tensor<'c>,
    nt: i64,
) -> Result<(Tensor<'c>, Tensor<'c>, HcGates<'c>)> {
    let config = &fw.config;
    let streams = ctx.dsv4_hc_post(attn_out, &e.streams, &e.gates.post, &e.gates.comb)?;
    ctx.compute(&streams, 12)?;

    let flat = ctx.reshape_2d(&streams, config.hc_dim() as i64, nt)?;
    let normed = ctx.rms_norm(&flat, config.rms_eps)?;
    let mixes = ctx.mul_mat(
        weights.get(&format!("blk.{il}.hc_ffn_fn.weight")).expect("bound"),
        &normed,
    )?;
    ctx.compute(&mixes, 12)?;
    let gates = hc_gates(ctx, weights, config, &format!("blk.{il}.hc_ffn"), &mixes, nt)?;

    let collapsed = ctx.dsv4_hc_pre(&streams, &gates.pre)?;
    let normed = ctx.rms_norm(&collapsed, config.rms_eps)?;
    let ffn_norm = ctx.mul(
        &normed,
        weights.get(&format!("blk.{il}.ffn_norm.weight")).expect("bound"),
    )?;
    ctx.compute(&ffn_norm, 12)?;
    Ok((streams, ffn_norm, gates))
}

/// The router: probabilities, the six experts, and their normalised weights.
///
/// **Two entirely different selection schemes**, chosen by `hash_layer_count`.
/// The first three blocks look their experts up in `ffn_gate_tid2eid` by *token
/// id* — no top-k at all, and `exp_probs_b` unused. Every other block adds the
/// selection bias and takes `argsort_top_k`, where **the bias steers selection
/// only**: the weights are gathered from the *unbiased* probabilities.
fn moe_routing<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    ffn_norm: &Tensor<'c>,
    tokens: &[i32],
) -> Result<(Tensor<'c>, Vec<i32>)> {
    let config = &fw.config;
    let nt = tokens.len() as i64;
    let n_expert = config.n_expert as i64;
    let n_used = config.n_expert_used as i64;

    let logits = ctx.mul_mat(
        weights.get(&format!("blk.{il}.ffn_gate_inp.weight")).expect("bound"),
        ffn_norm,
    )?;
    // sqrt(softplus(x)) — `expert_gating_func 4`, neither softmax nor sigmoid.
    let probs = ctx.sqrt(&ctx.softplus(&logits)?)?;
    ctx.compute(&probs, 12)?;
    let probs3 = ctx.reshape_3d(&probs, 1, n_expert, nt)?;

    let topk = if il < config.hash_layer_count {
        let tok = ctx.new_i32_1d(nt)?;
        tok.set_i32(tokens)?;
        ctx.get_rows(
            weights.get(&format!("blk.{il}.ffn_gate_tid2eid.weight")).expect("bound"),
            &tok,
        )?
    } else {
        let biased = ctx.add(
            &probs,
            weights.get(&format!("blk.{il}.exp_probs_b.bias")).expect("bound"),
        )?;
        ctx.compute(&biased, 12)?;
        ctx.argsort_top_k(&biased, n_used as i32)?
    };
    ctx.compute(&topk, 12)?;
    let ids = topk.to_vec_i32();

    // Renormalised over the selected six only, then scaled. The divisor is
    // clamped at the smallest F16 normal, not at an epsilon.
    let w = ctx.get_rows(&probs3, &topk)?;
    let w2 = ctx.reshape_2d(&w, n_used, nt)?;
    let sum = ctx.clamp(&ctx.sum_rows(&w2)?, 6.103515625e-5, f32::INFINITY)?;
    let w_norm = ctx.div(&w2, &sum)?;
    let w3 = ctx.reshape_3d(&w_norm, 1, n_used, nt)?;
    let w_scaled = ctx.scale(&w3, config.expert_weights_scale)?;
    ctx.compute(&w_scaled, 12)?;
    Ok((w_scaled, ids))
}

/// Read only the expert slices these tokens route to, and bind them compactly.
///
/// A stacked expert tensor is `[ne0, ne1, n_expert]` with equal slices, so slice
/// `i` starts at `i * size / n_expert`. Binding all 256 for one block is 3.19
/// GiB and does not fit this machine; the tokens' own selection is a fraction of
/// that. **This is what the runner has to do anyway**, not a test convenience.
///
/// # Why the destination is deliberately misaligned
///
/// Each slice is read straight into its final position in one stacked buffer,
/// so no byte is copied between the drive and `ggml`. That only works if the
/// memory address and the file offset agree modulo the sector size — and GGUF
/// pads tensor data to `general.alignment`, which is **32**, so V4-Flash's
/// experts sit at file offsets ≡ 2816 (mod 4096). A conventionally aligned
/// buffer can never match, and every byte bounces through a scratch.
///
/// The slices of one tensor are all the same size, and that size is a sector
/// multiple, so **one skew serves the whole stack**. Measured on
/// `blk.5.ffn_up_exps.weight`: 0.78 → 1.57 GiB/s, with 0.09% of bytes copied
/// (the two edge sectors of each 4.25 MiB slice) instead of 300%.
fn bind_expert_slices<'c>(
    model: &Model,
    ctx: &'c Context,
    weights: &mut WeightSet<'c>,
    name: &str,
    unique: &[i32],
) -> Result<(u64, Vec<u64>)> {
    let loc = model.location(name).expect("stacked tensor").clone();
    let n_expert = *loc.dims.last().expect("stacked");
    let slice = loc.size / n_expert;
    let total = unique.len() * slice as usize;

    let mut buf = SkewedBuf::new(total, SkewedBuf::skew_for(loc.file_offset));
    let mut disk = 0f64;
    let mut copied = 0usize;
    for (i, e) in unique.iter().enumerate() {
        let at = i * slice as usize;
        let t = std::time::Instant::now();
        copied += model.read_range_into(name, *e as u64 * slice, &mut buf[at..at + slice as usize])?;
        disk += t.elapsed().as_secs_f64();
    }
    if std::env::var("BIGTEA_IO_TIMING").is_ok() {
        eprintln!(
            "    io {name}: disk {disk:.3}s  {:.2} GiB/s  {:.2}% copied",
            total as f64 / (1u64 << 30) as f64 / disk,
            copied as f64 / total.max(1) as f64 * 100.0
        );
    }
    let mut dims = loc.dims.clone();
    *dims.last_mut().expect("stacked") = unique.len() as u64;
    weights.bind(ctx, name, loc.ty, &dims, buf)?;
    Ok((total as u64, dims))
}

/// The routed experts and the shared one, summed into the block's FFN output.
///
/// The shared expert runs for **every** token and is therefore resident weight;
/// confusing it with the 256 routed ones is the difference between a 7 GiB
/// resident set and a 144 GiB one. Both clamp their SwiGLU asymmetrically:
/// `(-inf, limit]` on the gate, `[-limit, limit]` on the up projection.
fn ffn<'c>(
    fw: &Deepseek4Forward<'_>,
    model: &Model,
    ctx: &'c Context,
    wctx: &'c Context,
    weights: &mut WeightSet<'c>,
    il: u32,
    ffn_norm: &Tensor<'c>,
    w_scaled: &Tensor<'c>,
    ids: &[i32],
    nt: i64,
) -> Result<Tensor<'c>> {
    let config = &fw.config;
    let n_embd = config.n_embd as i64;
    let n_used = config.n_expert_used as i64;
    let f32_size = std::mem::size_of::<f32>();
    let limit = config.swiglu_limit(il, false).unwrap_or(f32::INFINITY);
    let limit_sh = config.swiglu_limit(il, true).unwrap_or(f32::INFINITY);

    // ---- the shared expert ----
    let sh_gate = ctx.mul_mat(
        weights.get(&format!("blk.{il}.ffn_gate_shexp.weight")).expect("bound"),
        ffn_norm,
    )?;
    let sh_gate = ctx.clamp(&sh_gate, f32::NEG_INFINITY, limit_sh)?;
    let sh_up = ctx.mul_mat(
        weights.get(&format!("blk.{il}.ffn_up_shexp.weight")).expect("bound"),
        ffn_norm,
    )?;
    let sh_up = ctx.clamp(&sh_up, -limit_sh, limit_sh)?;
    let sh = ctx.mul_mat(
        weights.get(&format!("blk.{il}.ffn_down_shexp.weight")).expect("bound"),
        &ctx.swiglu_split(&sh_gate, &sh_up)?,
    )?;
    ctx.compute(&sh, 12)?;

    // ---- the routed experts, read as slices ----
    let mut unique = ids.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let compact: Vec<i32> = ids
        .iter()
        .map(|e| unique.iter().position(|u| u == e).expect("in set") as i32)
        .collect();
    let mut dims_of = std::collections::HashMap::new();
    let t_exp = std::time::Instant::now();
    let mut exp_bytes = 0u64;
    for suffix in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
        let name = format!("blk.{il}.{suffix}.weight");
        let (read, dims) = bind_expert_slices(model, wctx, weights, &name, &unique)?;
        exp_bytes += read;
        dims_of.insert(suffix, dims);
    }
    if std::env::var("BIGTEA_BLOCK_TIMING").is_ok() {
        eprintln!(
            "  block {il:>2}  experts {:.2}s ({:.0} MiB, {} of {} slices)",
            t_exp.elapsed().as_secs_f64(),
            exp_bytes as f64 / (1 << 20) as f64,
            unique.len(),
            config.n_expert,
        );
    }
    let n_uniq = unique.len() as i64;
    let ids_t = ctx.new_i32_2d(n_used, nt)?;
    ids_t.set_i32(&compact)?;

    let stack = |suffix: &str| -> Result<Tensor<'c>> {
        let d = &dims_of[suffix];
        Ok(ctx.reshape_3d(
            weights.get(&format!("blk.{il}.{suffix}.weight")).expect("bound"),
            d[0] as i64,
            d[1] as i64,
            n_uniq,
        )?)
    };

    let cur3 = ctx.reshape_3d(ffn_norm, n_embd, 1, nt)?;
    let gate = ctx.mul_mat_id(&stack("ffn_gate_exps")?, &cur3, &ids_t)?;
    let gate = ctx.clamp(&gate, f32::NEG_INFINITY, limit)?;
    let up = ctx.mul_mat_id(&stack("ffn_up_exps")?, &cur3, &ids_t)?;
    let up = ctx.clamp(&up, -limit, limit)?;
    let act = ctx.swiglu_split(&gate, &up)?;
    if std::env::var("BIGTEA_SPARSITY").is_ok() {
        // How much of the intermediate actually matters? The router picks 6 of
        // 256 experts; this asks how much of a CHOSEN expert is dead weight for
        // this token. Rows whose activation is negligible never reach the
        // output, so their  rows and  columns need not be read.
        ctx.compute(&act, 12)?;
        let v = act.to_vec_f32();
        let peak = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        let mut buckets = [0usize; 4]; // >1%, >0.1%, >0.01% of peak, and rest
        for x in &v {
            let r = x.abs() / peak.max(f32::MIN_POSITIVE);
            if r > 1e-2 { buckets[0] += 1 } else if r > 1e-3 { buckets[1] += 1 }
            else if r > 1e-4 { buckets[2] += 1 } else { buckets[3] += 1 }
        }
        let n = v.len() as f64;
        eprintln!(
            "  sparsity blk {il:>2}: >1% {:.1}%  >0.1% {:.1}%  >0.01% {:.1}%  negligible {:.1}%",
            100.0 * buckets[0] as f64 / n,
            100.0 * buckets[1] as f64 / n,
            100.0 * buckets[2] as f64 / n,
            100.0 * buckets[3] as f64 / n,
        );
    }
    let down = ctx.mul_mat_id(&stack("ffn_down_exps")?, &act, &ids_t)?;
    let weighted = ctx.mul(&down, w_scaled)?;
    ctx.compute(&weighted, 12)?;

    // Sum across the six experts as six strided views and five adds, which is
    // the shape llama.cpp uses.
    let row = n_embd as usize * f32_size;
    let mut moe_out: Option<Tensor<'c>> = None;
    for j in 0..n_used as usize {
        let v = ctx.view_2d(&weighted, n_embd, nt, row * n_used as usize, j * row)?;
        moe_out = Some(match moe_out {
            None => v,
            Some(acc) => ctx.add(&acc, &v)?,
        });
    }
    let out = ctx.add(&moe_out.expect("experts"), &sh)?;
    ctx.compute(&out, 12)?;
    Ok(out)
}

/// Bind one always-read tensor, from RAM if it is resident and from disk if it
/// is not. Returns its size, so a caller can report what it moved.
///
/// # Why residency is the difference between a demo and a runner
///
/// V4-Flash's always-read weights are 7.38 GiB and every one of them is touched
/// on **every token**. Read per block, they cost 7.1s of a 5-token prefill — 23%
/// — and a generation loop would pay that again for each token produced, forever.
/// Held in RAM they cost one read for the whole session.
///
/// Binding from the resident set is a refcount bump, not a copy: the same bytes
/// are pointed at by a fresh `ggml` tensor on every block of every token, and
/// copying 7.38 GiB per token to achieve that would defeat the purpose.
///
/// Falling back to disk is not a failure path but the design working: the
/// budget is a hard ceiling, and a machine too small for the whole set streams
/// the remainder rather than swapping. Swapping is slower than the streaming it
/// replaces.
fn bind_dense<'c>(
    fw: &Deepseek4Forward<'_>,
    wctx: &'c Context,
    weights: &mut WeightSet<'c>,
    name: &str,
) -> Result<u64> {
    let loc = fw.model.location(name).expect("present").clone();
    match fw.resident.and_then(|r| r.get_shared(name)) {
        Some(shared) => {
            weights.bind_shared(wctx, name, loc.ty, &loc.dims, shared)?;
            Ok(0)
        }
        None => {
            let data = fw.model.read_tensor_shared(name)?;
            let n = data.len() as u64;
            weights.bind_shared(wctx, name, loc.ty, &loc.dims, data)?;
            Ok(n)
        }
    }
}

/// One whole block, in its own arena, streams in and streams out as floats.
///
/// Owning the arena per block is what makes depth free: chaining blocks inside
/// one `ggml` context costs hundreds of megabytes each. Freeing weights *inside*
/// a context instead would be unsound — every `compute` rebuilds the graph
/// through its sources, so a dropped buffer reads freed memory successfully.
pub fn block(
    fw: &Deepseek4Forward<'_>,
    il: u32,
    tokens: &[i32],
    streams_in: Option<&[f32]>,
    arena: usize,
) -> Result<Streams> {
    let config = fw.config.clone();
    let nt = tokens.len() as i64;
    let t_block = std::time::Instant::now();
    let ctx = Context::new(arena)?;
    let wctx = Context::new_no_alloc(32 << 20)?;
    let mut weights = WeightSet::new();

    let mut names = fw.block_tensor_names(il);
    if il == 0 {
        names.push("token_embd.weight".to_string());
    }
    let t_bind = std::time::Instant::now();
    let mut dense_bytes = 0u64;
    for name in &names {
        dense_bytes += bind_dense(fw, &wctx, &mut weights, name)?;
    }
    let dense_secs = t_bind.elapsed().as_secs_f64();

    let streams = match streams_in {
        None => embed(&ctx, &weights, &config, tokens)?,
        Some(v) => {
            let t = ctx.new_f32_3d(config.n_embd as i64, config.hc_mult as i64, nt)?;
            t.set_f32(v)?;
            t
        }
    };

    let e = entry(fw, &ctx, &weights, il, streams, nt)?;
    let (q, kv) = q_and_kv(fw, &ctx, &weights, il, &e.attn_norm, nt)?;

    // Which attention runs is decided by the block's compression ratio *and*
    // whether a block has completed yet: below the first boundary a compressed
    // layer falls back to Raw, exactly as llama.cpp's guards do.
    let kind = config.attention_kind_from_ratio(il).expect("known ratio");
    let fired = config.compress_block(il).is_some_and(|r| nt / r > 0);
    let comp = match (kind, fired) {
        (AttentionKind::Raw, _) | (_, false) => None,
        (AttentionKind::CompressedSparse, true) => {
            Some(compressor(fw, &ctx, &weights, il, &e.attn_norm, nt, true)?)
        }
        (AttentionKind::HeavilyCompressed, true) => {
            Some(compressor(fw, &ctx, &weights, il, &e.attn_norm, nt, false)?)
        }
    };
    let attn_out = attention(fw, &ctx, &weights, il, &q, &kv, comp.as_ref(), nt)?;

    let (streams, ffn_norm, ffn_gates) = layer_tail(fw, &ctx, &weights, il, &e, &attn_out, nt)?;
    let (w_scaled, ids) = moe_routing(fw, &ctx, &weights, il, &ffn_norm, tokens)?;
    let ffn_out = ffn(
        fw, fw.model, &ctx, &wctx, &mut weights, il, &ffn_norm, &w_scaled, &ids, nt,
    )?;

    let out = ctx.dsv4_hc_post(&ffn_out, &streams, &ffn_gates.post, &ffn_gates.comb)?;
    ctx.compute(&out, 12)?;

    if std::env::var("BIGTEA_BLOCK_TIMING").is_ok() {
        eprintln!(
            "  block {il:>2}  dense {:.2}s ({:.0} MiB)  rest {:.2}s",
            dense_secs,
            dense_bytes as f64 / (1 << 20) as f64,
            t_block.elapsed().as_secs_f64() - dense_secs,
        );
    }
    Ok(out.to_vec_f32())
}

/// The output head: the **last** token's streams, collapsed and projected.
///
/// Its gate block is the `pre` half only — nothing writes back into the streams
/// after this, so there is no `post` and no combination matrix.
pub fn head(fw: &Deepseek4Forward<'_>, streams: &[f32], arena: usize) -> Result<Vec<f32>> {
    let config = &fw.config;
    let ctx = Context::new(arena)?;
    let wctx = Context::new_no_alloc(8 << 20)?;
    let mut weights = WeightSet::new();
    for name in [
        "output_hc_fn.weight",
        "output_hc_scale.weight",
        "output_hc_base.weight",
        "output_norm.weight",
        "output.weight",
    ] {
        bind_dense(fw, &wctx, &mut weights, name)?;
    }

    let hc = config.hc_mult as i64;
    let n_embd = config.n_embd as i64;
    let hc_dim = config.hc_dim() as usize;
    let last = &streams[streams.len() - hc_dim..];

    let x = ctx.new_f32_3d(n_embd, hc, 1)?;
    x.set_f32(last)?;
    let flat = ctx.reshape_2d(&x, hc_dim as i64, 1)?;
    let normed = ctx.rms_norm(&flat, config.rms_eps)?;
    let mixes = ctx.mul_mat(weights.get("output_hc_fn.weight").expect("bound"), &normed)?;
    ctx.compute(&mixes, 12)?;

    let scale = ctx.view_1d(weights.get("output_hc_scale.weight").expect("bound"), 1, 0)?;
    let base = ctx.view_1d(weights.get("output_hc_base.weight").expect("bound"), hc, 0)?;
    let gated = ctx.sigmoid(&ctx.add(&ctx.mul(&mixes, &scale)?, &base)?)?;
    let eps = ctx.new_f32_1d(hc)?;
    eps.set_f32(&vec![1e-6f32; hc as usize])?;
    let pre = ctx.add(&gated, &eps)?;
    ctx.compute(&pre, 12)?;

    let collapsed = ctx.dsv4_hc_pre(&x, &pre)?;
    let normed = ctx.rms_norm(&collapsed, config.rms_eps)?;
    let result = ctx.mul(&normed, weights.get("output_norm.weight").expect("bound"))?;
    let logits = ctx.mul_mat(weights.get("output.weight").expect("bound"), &result)?;
    ctx.compute(&logits, 12)?;
    Ok(logits.to_vec_f32())
}

/// Prefill: every block in order, then the head. Returns one logit per token id.
pub fn prefill(fw: &Deepseek4Forward<'_>, tokens: &[i32], arena: usize) -> Result<Vec<f32>> {
    let mut streams = block(fw, 0, tokens, None, arena)?;
    for il in 1..fw.config.n_layer {
        streams = block(fw, il, tokens, Some(&streams), arena)?;
    }
    head(fw, &streams, arena)
}
