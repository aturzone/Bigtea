//! The gated delta net — three quarters of a Qwen3.5 / 3.6 / 3.8 layer stack.
//!
//! # What is different about this architecture
//!
//! Every other architecture in this crate is attention all the way down. This
//! one is **hybrid**: `full_attention_interval` is 4, and llama.cpp's rule is
//! `is_recurrent(il) = (il + 1) % interval != 0`, so on the 24-block 0.8B
//! **18 layers are a gated delta net and 6 are attention** — and the container
//! agrees, carrying `attn_q.weight` on exactly blocks 3, 7, 11, 15, 19 and 23.
//!
//! A delta-net layer is a *linear attention* recurrence. Instead of attending
//! over a stored history, it carries a state **matrix** per value head,
//! `[state_size, state_size]`, and rewrites it at every token. That is why a KV
//! cache cannot stand in for it: a KV cache appends and never revises, and this
//! state is revised in place. It also means the state is small and fixed —
//! ~1 MB per layer here regardless of context length, where a KV cache grows.
//!
//! # Why this is a port and not a research project
//!
//! **`ggml_gated_delta_net` is the entire chunked delta rule in one op.** It
//! takes q, k, v, the gates and the incoming state, and returns the attention
//! scores followed by the outgoing state. Everything in this file is therefore
//! projections, one convolution, and a gated norm — the hard arithmetic is
//! ggml's. The `1/sqrt(S_k)` scaling on q lives *inside* the op, which is why
//! nothing here scales it.
//!
//! # The state lives on the host
//!
//! llama.cpp writes the new state back into a cache tensor inside the graph
//! (`delta-net-base.cpp:build_conv_state`, most of its length). Chaos already
//! keeps its KV cache host-side and binds it per layer, and `stream.rs` hands
//! plain `Vec<f32>` between phases, so this follows the same pattern: the state
//! goes in as an input, comes out as an output, and `RecurrentState` holds it
//! between tokens. No in-graph cache views, no `ggml_cpy`.
//!
//! # The trap
//!
//! **Prompt length changes which regime runs.** With `n_tokens > 1` the fused op
//! takes its chunked path and with `n_tokens == 1` its autoregressive one, so a
//! prefill that matches llama.cpp says nothing about generation. Both are
//! checked, at 1, 5 and 20 tokens.

use chaos_ggml::{Context, Tensor, WeightSet};

use crate::qwen3::{Qwen3Config, SsmConfig};
use crate::{ArchError, Result};

/// f32, which every activation in this file is.
const F32: usize = 4;

/// The carried state of every recurrent layer, between tokens.
///
/// Two pieces per layer, and **the convolution window is the one that is easy
/// to forget**: the depthwise convolution reaches `conv_kernel - 1` tokens back,
/// so dropping it makes the first token of every step see zeros where it should
/// see history. Attention layers hold empty vectors, so indexing is by absolute
/// layer number and there is no second numbering to get wrong.
pub struct RecurrentState {
    conv: Vec<Vec<f32>>,
    ssm: Vec<Vec<f32>>,
}

impl RecurrentState {
    /// Zeroed state for a fresh sequence.
    pub fn new(c: &Qwen3Config) -> Self {
        let mut conv = Vec::with_capacity(c.n_layer as usize);
        let mut ssm = Vec::with_capacity(c.n_layer as usize);
        for il in 0..c.n_layer {
            match c.ssm {
                Some(s) if c.is_recurrent(il) => {
                    conv.push(vec![0.0; s.conv_state_len()]);
                    ssm.push(vec![0.0; s.recurrent_state_len()]);
                }
                _ => {
                    conv.push(Vec::new());
                    ssm.push(Vec::new());
                }
            }
        }
        Self { conv, ssm }
    }

    /// Total bytes held, for the load report.
    #[allow(dead_code)]
    pub fn bytes(&self) -> usize {
        let f = |v: &Vec<Vec<f32>>| v.iter().map(|x| x.len() * F32).sum::<usize>();
        f(&self.conv) + f(&self.ssm)
    }

    pub fn conv(&self, il: u32) -> &[f32] {
        &self.conv[il as usize]
    }

    pub fn ssm(&self, il: u32) -> &[f32] {
        &self.ssm[il as usize]
    }

    /// Replace one layer's state after a step.
    ///
    /// **Both halves or neither.** A step that updated the recurrent state and
    /// not the convolution window would be correct for exactly one token and
    /// then quietly wrong, which is the failure this whole file is careful
    /// about.
    pub fn store(&mut self, il: u32, conv: Vec<f32>, ssm: Vec<f32>) {
        self.conv[il as usize] = conv;
        self.ssm[il as usize] = ssm;
    }

    /// Back to a fresh sequence, keeping the allocations.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        for v in self.conv.iter_mut().chain(self.ssm.iter_mut()) {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }
}

/// The tensors a delta-net layer needs bound before it runs.
pub struct Inputs<'a> {
    /// `[n_embd, n_tokens]` — the layer input, already through `attn_norm`.
    pub x: Tensor<'a>,
    /// `[conv_kernel - 1, conv_dim, 1]` — the rolling convolution window.
    pub conv: Tensor<'a>,
    /// `[state_size * state_size * n_v_heads]` — the carried state, flat.
    pub state: Tensor<'a>,
}

/// What one delta-net layer produces.
pub struct Outputs<'a> {
    /// `[n_embd, n_tokens]` — what rejoins the residual stream.
    pub out: Tensor<'a>,
    /// `[conv_kernel - 1, conv_dim, 1]` — the window to carry forward.
    pub conv: Tensor<'a>,
    /// `[state_size, state_size, n_v_heads, 1]` — the state to carry forward.
    pub state: Tensor<'a>,
    /// Intermediates, for diffing one layer against `llama-eval-callback`.
    ///
    /// **The whole layer agreeing is not the same as the layer being right**,
    /// and when it disagrees these are what say where. Named after llama.cpp's
    /// own callback labels so the two dumps read side by side.
    pub qkv_mixed: Tensor<'a>,
    pub conv_raw: Tensor<'a>,
    pub scores: Tensor<'a>,
}

/// How much arena one delta-net layer needs, in units of `(ne0, ne1)` pairs for
/// [`crate::stream::arena_for`].
///
/// **ggml aborts rather than failing when an arena runs short**, and it takes
/// the whole process with it, so every tensor this function can allocate is
/// listed. The convolution input is the widest: `conv_dim` by
/// `conv_kernel - 1 + n_tokens`.
pub fn arena_shapes(s: &SsmConfig, n_embd: i64, n_tokens: i64) -> Vec<(i64, i64)> {
    let conv_dim = i64::from(s.conv_dim());
    let value_dim = i64::from(s.value_dim());
    let heads = i64::from(s.time_step_rank);
    let side = i64::from(s.state_size);
    let window = i64::from(s.conv_kernel) - 1;
    vec![
        (n_embd, n_tokens),                                 // x
        (conv_dim, n_tokens),                               // qkv_mixed
        (n_tokens, conv_dim),                               // transposed
        (window + n_tokens, conv_dim),                      // conv input, the widest
        (window + n_tokens, conv_dim),                      // and its copy for the op
        (conv_dim, n_tokens),                               // conv output
        (conv_dim, n_tokens),                               // silu of it
        (value_dim, n_tokens),                              // z
        (value_dim, n_tokens),                              // silu(z)
        (heads, n_tokens),                                  // alpha
        (heads, n_tokens),                                  // + dt
        (heads, n_tokens),                                  // softplus
        (heads, n_tokens),                                  // * a
        (heads, n_tokens),                                  // beta
        (heads, n_tokens),                                  // sigmoid(beta)
        (side * side * heads, 1),                           // state in
        (side * heads * n_tokens + side * side * heads, 1), // the fused result
        (value_dim, n_tokens),                              // normalised scores
        (value_dim, n_tokens),                              // gated
        (n_embd, n_tokens),                                 // output projection
        (window * conv_dim, 1),                             // the conv window, copied
        (side * side * heads, 1),                           // the state, copied
    ]
}

/// Build one gated delta-net layer.
///
/// Follows `llama_model_qwen35::graph::build_layer_attn_linear` step for step.
/// The order matters in one place that is not obvious: **`ssm_dt.bias` is added
/// to `alpha` before the softplus, not after**, and the result is multiplied by
/// `ssm_a` rather than exponentiated — `ssm_a` already holds `-A_log.exp()`.
pub fn layer<'a>(
    ctx: &'a Context,
    weights: &WeightSet<'a>,
    c: &Qwen3Config,
    il: u32,
    inp: &Inputs<'a>,
) -> Result<Outputs<'a>> {
    let s = c.ssm.ok_or_else(|| {
        ArchError::MissingMetadata("qwen35.ssm.conv_kernel (not a hybrid model)".into())
    })?;
    let get = |name: String| -> Result<&Tensor<'a>> {
        weights.get(&name).ok_or(ArchError::MissingTensor(name))
    };

    let n_tokens = inp.x.dims_and_strides().0[1];
    let side = i64::from(s.state_size);
    let h_k = i64::from(s.group_count);
    let h_v = i64::from(s.time_step_rank);
    let head_v = i64::from(s.head_v_dim());
    let conv_dim = i64::from(s.conv_dim());
    let key_dim = i64::from(s.key_dim());
    let value_dim = i64::from(s.value_dim());
    let window = i64::from(s.conv_kernel) - 1;

    // The fused op asserts these and **aborts the process** if they fail, so
    // they are checked here where the failure is a returned error.
    if head_v != side {
        return Err(ArchError::MissingMetadata(format!(
            "qwen35: head_v_dim {head_v} != state_size {side}; the fused delta \
             rule requires S_k == S_v"
        )));
    }
    if h_k == 0 || h_v % h_k != 0 {
        return Err(ArchError::MissingMetadata(format!(
            "qwen35: {h_v} value heads is not a multiple of {h_k} key heads"
        )));
    }

    // -- projections ---------------------------------------------------------
    // `attn_qkv` here is **not** attention's QKV. It is the delta net's input
    // projection, `2 * key_dim + value_dim` wide, and `ssm_conv1d` convolves
    // all of it before it is split.
    let qkv = ctx.mul_mat(get(format!("blk.{il}.attn_qkv.weight"))?, &inp.x)?;
    let z = ctx.mul_mat(get(format!("blk.{il}.attn_gate.weight"))?, &inp.x)?;

    // beta: one scalar per value head per token, through a sigmoid.
    let beta = ctx.mul_mat(get(format!("blk.{il}.ssm_beta.weight"))?, &inp.x)?;
    let beta = ctx.reshape_4d(&beta, [1, h_v, n_tokens, 1])?;
    let beta = ctx.sigmoid(&beta)?;

    // g: the forget gate. `softplus(alpha + dt) * a`, where `a` is already
    // negative -- it holds `-A_log.exp()`. Applying `exp` here as well, or
    // adding the bias after the softplus, both give a gate that is plausible
    // and wrong.
    let alpha = ctx.mul_mat(get(format!("blk.{il}.ssm_alpha.weight"))?, &inp.x)?;
    let alpha = ctx.add(&alpha, get(format!("blk.{il}.ssm_dt.bias"))?)?;
    let g = ctx.softplus(&alpha)?;
    let g = ctx.mul(&g, get(format!("blk.{il}.ssm_a"))?)?;
    let g = ctx.reshape_4d(&g, [1, h_v, n_tokens, 1])?;

    // -- the rolling convolution --------------------------------------------
    // The stored window goes *in front of* this step's tokens along the time
    // axis, so token 0 of a single-token step still sees three tokens of
    // history. Transposed first because `ssm_conv` wants time on the fast axis.
    let qkv_t = ctx.transpose(&qkv)?;
    let conv_in = ctx.concat(&inp.conv, &qkv_t, 0)?;

    // The tail of that same buffer is the window for next time -- taken before
    // the convolution, because the convolution consumes it.
    let (cne, cnb) = conv_in.dims_and_strides();
    let conv_next = ctx.view_3d(
        &conv_in,
        window,
        conv_dim,
        1,
        cnb[1],
        cnb[2],
        F32 * (cne[0] - window) as usize,
    )?;

    // **The convolution gets its own copy of the window.** `conv_next` above is
    // a view into `conv_in`, and ggml's allocator is free to place
    // `ssm_conv`'s output over an input buffer it can prove is dead -- so the
    // window being carried forward was read *after* the op had written over it.
    // The symptom was as bad as symptoms get: with a debug pass over the same
    // graph the tokens matched llama.cpp exactly, and without it they did not,
    // because the extra pass changed when the clobber happened. A `cont` here
    // makes the op read a buffer nothing else needs.
    let conv_for_op = ctx.cont_4d(&conv_in, [cne[0], conv_dim, 1, 1])?;
    let convolved_raw = ctx.ssm_conv(&conv_for_op, get(format!("blk.{il}.ssm_conv1d.weight"))?)?;
    let convolved = ctx.silu(&convolved_raw)?;

    // -- split q, k, v out of the convolved block ---------------------------
    // `convolved` is `[conv_dim, n_tokens, 1]` and contiguous, so the three
    // views share one token stride and differ only by their offset. Reading
    // them row-major instead would transpose every head.
    let token_stride = F32 * conv_dim as usize;
    let seq_stride = token_stride * n_tokens as usize;
    let split = |ne1: i64, offset_elems: i64| -> Result<Tensor<'a>> {
        Ok(ctx.view_4d(
            &convolved,
            [side, ne1, n_tokens, 1],
            [F32 * side as usize, token_stride, seq_stride],
            F32 * offset_elems as usize,
        )?)
    };
    let q = split(h_k, 0)?;
    let k = split(h_k, key_dim)?;
    let v = split(h_v, 2 * key_dim)?;

    // L2, not RMS: divide each head by its own norm. `rms_norm` would divide by
    // the root *mean* square instead and scale every row by sqrt(head_dim).
    // **Made contiguous first.** All three are strided views into the convolved
    // block -- their token stride is `conv_dim`, not `side * heads` -- and the
    // fused op reads them as packed. llama.cpp has `ggml_cont_4d` calls here
    // commented out because its default path permutes and copies anyway; taking
    // the fused path without them gave scores four orders of magnitude too
    // small, which is a misread rather than an error.
    let q = ctx.l2_norm(&ctx.cont_4d(&q, [side, h_k, n_tokens, 1])?, c.rms_eps)?;
    let k = ctx.l2_norm(&ctx.cont_4d(&k, [side, h_k, n_tokens, 1])?, c.rms_eps)?;
    let v = ctx.cont_4d(&v, [side, h_v, n_tokens, 1])?;

    // -- the delta rule ------------------------------------------------------
    // No scaling of q here: the `1/sqrt(S_k)` lives inside the op. And no
    // explicit broadcast of q and k from `h_k` to `h_v` heads -- the fused path
    // does that itself, which is why llama.cpp's `repeat_4d` sits behind a
    // `!fused` guard.
    let state = ctx.reshape_4d(&inp.state, [side, side, h_v, 1])?;
    let fused = ctx.gated_delta_net(&q, &k, &v, &g, &beta, &state, 1)?;

    // The result packs the scores, then one state snapshot.
    let scores = ctx.view_4d(
        &fused,
        [side, h_v, n_tokens, 1],
        [
            F32 * side as usize,
            F32 * (side * h_v) as usize,
            F32 * (side * h_v * n_tokens) as usize,
        ],
        0,
    )?;
    let state_next = ctx.view_4d(
        &fused,
        [side, side, h_v, 1],
        [
            F32 * side as usize,
            F32 * (side * side) as usize,
            F32 * (side * side * h_v) as usize,
        ],
        F32 * (side * h_v * n_tokens) as usize,
    )?;

    // -- gated normalisation and the output projection ----------------------
    // `norm(scores) * silu(z)`, per value head, with `ssm_norm` one head wide.
    let normed = ctx.rms_norm(&scores, c.rms_eps)?;
    let normed = ctx.mul(&normed, get(format!("blk.{il}.ssm_norm.weight"))?)?;
    let gate = ctx.silu(&ctx.reshape_4d(&z, [head_v, h_v, n_tokens, 1])?)?;
    let gated = ctx.mul(&normed, &gate)?;

    let flat = ctx.reshape_2d(&ctx.cont(&gated)?, value_dim, n_tokens)?;
    let out = ctx.mul_mat(get(format!("blk.{il}.ssm_out.weight"))?, &flat)?;

    // **Both carried states are copied out, not handed back as views.**
    // `conv_next` is a window into the concat buffer and `state_next` a window
    // into the fused op's output; reading either as a view made the result
    // depend on *what else* was computed in the same graph -- with a debug pass
    // over `scores` the tokens matched llama.cpp exactly, and without it they
    // did not. A `cont` gives each its own buffer and makes the read
    // order-independent, which is the difference between a port that works and
    // one that works while being measured.
    let conv_next = ctx.cont_4d(&conv_next, [window, conv_dim, 1, 1])?;
    let state_next = ctx.cont_4d(&state_next, [side, side, h_v, 1])?;

    Ok(Outputs {
        out,
        conv: conv_next,
        state: state_next,
        qkv_mixed: qkv,
        conv_raw: convolved_raw,
        scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3::SsmConfig;

    fn ssm() -> SsmConfig {
        // Qwen3.5-0.8B, read from the container.
        SsmConfig {
            conv_kernel: 4,
            inner_size: 2048,
            state_size: 128,
            group_count: 16,
            time_step_rank: 16,
            full_attention_interval: 4,
        }
    }

    /// The derived widths must reproduce the container's own tensor shapes, or
    /// every view into the convolved block lands in the wrong place.
    #[test]
    fn the_derived_widths_match_the_container() {
        let s = ssm();
        // `attn_qkv.weight` is [1024, 6144] and `ssm_conv1d.weight` is [4, 6144].
        assert_eq!(s.conv_dim(), 6144);
        assert_eq!(s.key_dim(), 2048);
        // `attn_gate.weight` is [1024, 2048] and `ssm_out.weight` is [2048, 1024].
        assert_eq!(s.value_dim(), 2048);
        // `ssm_norm.weight` is [128].
        assert_eq!(s.head_v_dim(), 128);
    }

    /// The 27B is the same arithmetic at 48 value heads, and the fused op needs
    /// `S_k == S_v`, so `head_v_dim` must still land on `state_size`.
    #[test]
    fn the_27b_widths_also_work_out() {
        let s = SsmConfig {
            inner_size: 6144,
            time_step_rank: 48,
            ..ssm()
        };
        assert_eq!(s.conv_dim(), 10240);
        assert_eq!(s.head_v_dim(), 128);
        assert_eq!(s.head_v_dim(), s.state_size);
    }

    /// **The state is fixed-size, unlike a KV cache.** This is the whole reason
    /// a 256K context is affordable here, and the number is worth pinning: the
    /// 0.8B carries 1 MiB per recurrent layer whatever the context length.
    #[test]
    fn the_carried_state_does_not_grow_with_context() {
        let s = ssm();
        assert_eq!(s.recurrent_state_len(), 128 * 128 * 16);
        assert_eq!(s.recurrent_state_len() * F32, 1 << 20);
        assert_eq!(s.conv_state_len(), 3 * 6144);
    }

    /// The convolution window must be `conv_kernel - 1`, not `conv_kernel`.
    /// One too many and the concat is a token longer than `ssm_conv` expects.
    #[test]
    fn the_window_is_one_short_of_the_kernel() {
        let s = ssm();
        assert_eq!(s.conv_state_len() / s.conv_dim() as usize, 3);
    }
}
