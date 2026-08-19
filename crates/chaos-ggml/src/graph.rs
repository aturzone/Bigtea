//! Building and running `ggml` computation graphs.
//!
//! `ggml` is declarative: you describe a graph of tensor operations into a
//! context, then execute it. Nothing computes until [`Context::compute`].
//!
//! # Lifetimes, and why they matter here
//!
//! Tensors do not own their memory — the context does. Every tensor is a
//! pointer into the context's arena, so a tensor outliving its context is a
//! dangling pointer. [`Tensor`] therefore borrows the [`Context`], which makes
//! that mistake a compile error rather than a crash on a machine where
//! debugging one would be miserable.
//!
//! # Sizing the arena
//!
//! `ggml` allocates from a fixed arena chosen up front. Too small and graph
//! construction fails; too large and the memory is wasted on a machine that
//! has none to spare. [`Context::new`] takes the size explicitly rather than
//! guessing, and reports overflow as an error instead of aborting.

#![cfg(have_ggml)]

use std::marker::PhantomData;
use std::os::raw::{c_int, c_void};
use std::ptr::NonNull;

use crate::GgmlError;

#[repr(C)]
struct InitParams {
    mem_size: usize,
    mem_buffer: *mut c_void,
    no_alloc: bool,
}

#[allow(non_camel_case_types)]
type ggml_context = c_void;
#[allow(non_camel_case_types)]
type ggml_tensor = c_void;
#[allow(non_camel_case_types)]
type ggml_cgraph = c_void;

extern "C" {
    fn ggml_init(params: InitParams) -> *mut ggml_context;
    fn ggml_free(ctx: *mut ggml_context);

    // The scheduler reads these flags; see `Tensor::set_input`.
    fn ggml_set_input(tensor: *mut ggml_tensor);
    fn ggml_set_output(tensor: *mut ggml_tensor);

    fn ggml_new_tensor_1d(ctx: *mut ggml_context, ty: c_int, ne0: i64) -> *mut ggml_tensor;
    fn ggml_new_tensor_2d(
        ctx: *mut ggml_context,
        ty: c_int,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;

    fn ggml_nelements(t: *const ggml_tensor) -> i64;
    fn ggml_nbytes(t: *const ggml_tensor) -> usize;
    fn ggml_get_data_f32(t: *const ggml_tensor) -> *mut f32;
    fn ggml_get_data(t: *const ggml_tensor) -> *mut c_void;

    fn ggml_mul_mat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_add(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_mul(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_rms_norm(ctx: *mut ggml_context, a: *mut ggml_tensor, eps: f32) -> *mut ggml_tensor;
    fn ggml_norm(ctx: *mut ggml_context, a: *mut ggml_tensor, eps: f32) -> *mut ggml_tensor;
    fn ggml_soft_max(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_fp32_to_fp16_row(src: *const f32, dst: *mut u16, n: i64);
    fn ggml_fp16_to_fp32_row(src: *const u16, dst: *mut f32, n: i64);
    fn ggml_repeat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_view_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        offset: usize,
    ) -> *mut ggml_tensor;
    fn ggml_dsv4_hc_pre(
        ctx: *mut ggml_context,
        x: *mut ggml_tensor,
        weights: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_dsv4_hc_post(
        ctx: *mut ggml_context,
        x: *mut ggml_tensor,
        residual: *mut ggml_tensor,
        post: *mut ggml_tensor,
        comb: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_dsv4_hc_comb(
        ctx: *mut ggml_context,
        mixes: *mut ggml_tensor,
        scale: *mut ggml_tensor,
        base: *mut ggml_tensor,
        eps: f32,
        n_iter: c_int,
    ) -> *mut ggml_tensor;
    fn ggml_lightning_indexer(
        ctx: *mut ggml_context,
        q: *mut ggml_tensor,
        k: *mut ggml_tensor,
        weights: *mut ggml_tensor,
        mask: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_flash_attn_ext(
        ctx: *mut ggml_context,
        q: *mut ggml_tensor,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        mask: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> *mut ggml_tensor;
    fn ggml_silu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_gelu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    fn ggml_new_tensor_3d(
        ctx: *mut ggml_context,
        ty: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;

    fn ggml_new_tensor_4d(
        ctx: *mut ggml_context,
        ty: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;

    fn ggml_get_rows(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_concat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        dim: c_int,
    ) -> *mut ggml_tensor;
    fn ggml_permute(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        axis0: c_int,
        axis1: c_int,
        axis2: c_int,
        axis3: c_int,
    ) -> *mut ggml_tensor;
    fn ggml_transpose(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_cont(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_reshape_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;
    fn ggml_reshape_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;
    fn ggml_view_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
    #[allow(clippy::too_many_arguments)]
    fn ggml_view_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
    fn ggml_view_4d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> *mut ggml_tensor;
    fn ggml_reshape_4d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;
    fn ggml_cont_4d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;
    fn ggml_repeat_4d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> *mut ggml_tensor;
    /// Root-mean-square normalisation without the learned weight: divides each
    /// row by its own L2 norm. Not `rms_norm` -- that divides by the root
    /// *mean* square, this by the norm itself, and the delta net wants the
    /// latter on Q and K.
    fn ggml_l2_norm(ctx: *mut ggml_context, a: *mut ggml_tensor, eps: f32) -> *mut ggml_tensor;
    /// Depthwise causal 1-D convolution over a rolling window.
    ///
    /// `a` is `[window + n_tokens - 1, channels, n_seqs]` -- the stored tail of
    /// the previous call concatenated with this call's input -- and `b` is the
    /// kernel `[window, channels]`. Output is `[channels, n_tokens, n_seqs]`.
    fn ggml_ssm_conv(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    /// 2-D convolution, for a VAE decoder rather than a language model.
    ///
    /// **`a` is the kernel and `b` is the data**, which is the opposite order to
    /// every reading of "convolve the image with the kernel" -- and the arguments
    /// are the same type, so swapping them compiles and produces a differently
    /// shaped answer instead of an error.
    ///
    /// The kernel must be **F16**: this goes through `im2col` plus a matmul, and
    /// that path is written for a half-precision kernel. An F32 kernel aborts.
    fn ggml_conv_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        s0: i32,
        s1: i32,
        p0: i32,
        p1: i32,
        d0: i32,
        d1: i32,
    ) -> *mut ggml_tensor;
    /// Group normalisation: the VAE's norm, and not one any layer here uses.
    ///
    /// Normalises over `ne[0] * ne[1]` and groups of channels along `ne[2]`,
    /// which is why it belongs to a convolutional decoder and `rms_norm` does
    /// not.
    fn ggml_group_norm(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_groups: i32,
        eps: f32,
    ) -> *mut ggml_tensor;
    /// The whole gated delta rule, fused.
    ///
    /// **This is why running Qwen3.5/3.6/3.8 is a port rather than a project:**
    /// the chunked delta scan is one op, so the layer around it is projections,
    /// a convolution and a gated norm. Shapes, from `ggml.h`:
    ///
    /// ```text
    /// q, k  : [S_k, H_k, n_tokens, n_seqs]
    /// v     : [S_v, H_v, n_tokens, n_seqs]
    /// g     : [1,   H_v, n_tokens, n_seqs]   scalar gate
    /// beta  : [1,   H_v, n_tokens, n_seqs]
    /// state : [S_v, S_v, H_v, n_seqs]        the carried state going in
    /// ```
    ///
    /// The result packs the attention scores `[S_v, H_v, n_tokens, n_seqs]`
    /// **followed by `k` state snapshots**, most recent first. With `k == 1`
    /// that is one final state, which is all a linear decode needs.
    fn ggml_gated_delta_net(
        ctx: *mut ggml_context,
        q: *mut ggml_tensor,
        k: *mut ggml_tensor,
        v: *mut ggml_tensor,
        g: *mut ggml_tensor,
        beta: *mut ggml_tensor,
        state: *mut ggml_tensor,
        n_state_snapshots: i64,
    ) -> *mut ggml_tensor;
    /// Multi-section rotary embedding -- llama.cpp's mRoPE / IMROPE.
    ///
    /// `sections` is four counts splitting `n_dims` between temporal and
    /// spatial axes; a text-only prompt still goes through it, because the
    /// section split changes which dimension pairs rotate together.
    fn ggml_rope_multi(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: i32,
        sections: *const i32,
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;
    /// Attaches per-head sink logits to an existing `flash_attn_ext` node.
    /// Mutates `a` in place and returns nothing — it is not a graph builder.
    fn ggml_flash_attn_ext_add_sinks(a: *mut ggml_tensor, sinks: *mut ggml_tensor);
    fn ggml_scale(ctx: *mut ggml_context, a: *mut ggml_tensor, s: f32) -> *mut ggml_tensor;
    fn ggml_sigmoid(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_relu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_div(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_sum_rows(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_softplus(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_sqrt(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_clamp(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        min: f32,
        max: f32,
    ) -> *mut ggml_tensor;
    fn ggml_swiglu_split(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    fn ggml_top_k(ctx: *mut ggml_context, a: *mut ggml_tensor, k: c_int) -> *mut ggml_tensor;
    fn ggml_argsort_top_k(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        k: c_int,
    ) -> *mut ggml_tensor;
    #[allow(clippy::too_many_arguments)]
    fn ggml_rope_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: c_int,
        mode: c_int,
        n_ctx_orig: c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    #[allow(clippy::too_many_arguments)]
    fn ggml_rope_ext_back(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        c: *mut ggml_tensor,
        n_dims: c_int,
        mode: c_int,
        n_ctx_orig: c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> *mut ggml_tensor;

    /// Indexed matmul: picks a matrix per row from a stacked 3-D tensor.
    /// This is what makes MoE tractable — only the selected experts are
    /// multiplied, rather than all of them followed by a mask.
    fn ggml_mul_mat_id(
        ctx: *mut ggml_context,
        as_: *mut ggml_tensor,
        b: *mut ggml_tensor,
        ids: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    fn ggml_tanh(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_soft_max_ext(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        mask: *mut ggml_tensor,
        scale: f32,
        max_bias: f32,
    ) -> *mut ggml_tensor;

    fn ggml_new_graph(ctx: *mut ggml_context) -> *mut ggml_cgraph;
    fn ggml_build_forward_expand(graph: *mut ggml_cgraph, t: *mut ggml_tensor);
    fn ggml_graph_compute_with_ctx(
        ctx: *mut ggml_context,
        graph: *mut ggml_cgraph,
        n_threads: c_int,
    ) -> c_int;
}

/// Per-tensor bookkeeping ggml adds on top of the data itself.
///
/// `ggml_tensor` plus its object header, rounded up generously. Exact only
/// matters in that under-estimating is fatal (see [`arena_for`]).
const TENSOR_OVERHEAD: usize = 512;

/// Bytes an arena needs to hold f32 tensors of the given shapes.
///
/// **Size the arena with this, or something at least as generous.** Running
/// out is not recoverable: ggml calls `GGML_ASSERT` and aborts the process
/// rather than returning an error, so there is nothing to catch. Verified by
/// deliberately under-sizing an arena — the process dies with
/// "not enough space in the context's memory pool".
///
/// `slack_tensors` covers intermediates the caller did not enumerate; graph
/// building creates a tensor per operation, not just per named value.
pub fn arena_for(shapes: &[(i64, i64)], slack_tensors: usize) -> usize {
    let data: usize = shapes
        .iter()
        .map(|(a, b)| (a.max(&1) * b.max(&1)) as usize * std::mem::size_of::<f32>())
        .sum();
    let count = shapes.len() + slack_tensors;
    // Double the data budget so intermediates have room, and add graph
    // structure overhead. Over-allocating costs a little memory; under-
    // allocating costs the process.
    data * 2 + count * TENSOR_OVERHEAD + GRAPH_RESERVE
}

/// Convert f32 to f16, using ggml's own routine so the rounding matches what
/// its kernels expect.
///
/// Used for the KV cache: keeping it in f16 halves its memory (192 KiB per
/// position for this model becomes 96 KiB) and halves the bytes attention has
/// to read. It is what llama.cpp stores by default.
pub fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
    let n = src.len().min(dst.len());
    // SAFETY: both slices are valid for `n` elements of their own type.
    unsafe { ggml_fp32_to_fp16_row(src.as_ptr(), dst.as_mut_ptr(), n as i64) };
}

/// Convert f16 back to f32, again through ggml's own routine.
///
/// The KV cache is written in f16, so anything compared against a reference
/// taken *after* the cache write has to be rounded the same way — llama.cpp's
/// own trace shows the compressed KV summing to 63.125298 before the cache and
/// 63.123978 after it, and that 1.3e-3 is the rounding, not an error.
pub fn f16_to_f32(src: &[u16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    // SAFETY: both slices are valid for `n` elements of their own type.
    unsafe { ggml_fp16_to_fp32_row(src.as_ptr(), dst.as_mut_ptr(), n as i64) };
}

/// Arena space `compute` needs beyond the tensors themselves.
///
/// `ggml_graph_compute_with_ctx` allocates two things out of the same arena as
/// the tensors: the graph object, and the work buffer that quantized matmuls
/// use to hold their converted operands. The graph is the larger of the two and
/// its size is fixed — `ggml_new_graph` builds a default 2048-node graph
/// whatever the actual node count, which measured 3,060,816 bytes here.
///
/// A 1 MiB reserve was not enough, and the failure is an abort, not an error:
/// `ggml_new_object: not enough space in the context's memory pool (needed
/// 3060816, available 2087424)` followed by `GGML_ASSERT(obj_new) failed`.
const GRAPH_RESERVE: usize = 16 << 20;

/// RoPE scaling parameters, grouped because they always travel together.
///
/// [`RopeParams::default`] is plain RoPE with no context extension — the
/// values a model uses unless it declares otherwise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeParams {
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Default for RopeParams {
    fn default() -> Self {
        RopeParams {
            freq_base: 10000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

/// An arena that owns every tensor built into it.
///
/// # Sizing is not optional
///
/// ggml allocates from a fixed arena and **aborts the process** if it runs
/// out. Use [`arena_for`] rather than guessing.
pub struct Context {
    raw: NonNull<ggml_context>,
}

impl Context {
    /// Create a context with an arena of `mem_size` bytes.
    pub fn new(mem_size: usize) -> Result<Self, GgmlError> {
        Self::with_alloc(mem_size, false)
    }

    /// A context that allocates tensor *metadata* but not tensor *data*.
    ///
    /// This is how weights are bound without copying: the tensor exists, its
    /// `data` pointer starts null, and the caller aims it at memory they
    /// already hold. Without this the model would be stored twice, which for
    /// a 7.38 GiB dense set on a 15.7 GiB machine simply does not fit.
    ///
    /// The arena only needs room for tensor structs — a few hundred bytes
    /// each — not for the weights themselves.
    pub fn new_no_alloc(mem_size: usize) -> Result<Self, GgmlError> {
        Self::with_alloc(mem_size, true)
    }

    fn with_alloc(mem_size: usize, no_alloc: bool) -> Result<Self, GgmlError> {
        let params = InitParams {
            mem_size,
            mem_buffer: std::ptr::null_mut(),
            no_alloc,
        };
        // SAFETY: `params` is fully initialised; a null mem_buffer asks ggml to
        // allocate the arena itself, which is the documented contract.
        let raw = unsafe { ggml_init(params) };
        NonNull::new(raw)
            .map(|raw| Context { raw })
            .ok_or(GgmlError::ContextAlloc { bytes: mem_size })
    }

    /// A context that builds its graph inside memory the caller already owns.
    ///
    /// The streaming forward pass creates on the order of 200 contexts per
    /// token — one per layer for Q/K/V, attention, the router and the experts.
    /// Letting ggml allocate each arena means hundreds of multi-megabyte
    /// `malloc`s and first-touch page faults per token, which measured as the
    /// single largest unattributed cost in generation. One buffer, reused,
    /// removes all of it.
    ///
    /// # Safety
    ///
    /// `buf` must outlive the returned context and must not be aliased by
    /// another live context — tensors point directly into it, so reusing the
    /// buffer while an earlier context's tensors are still read would hand back
    /// overwritten weights rather than fail.
    pub unsafe fn in_buffer(buf: &mut [u8], no_alloc: bool) -> Result<Self, GgmlError> {
        let params = InitParams {
            mem_size: buf.len(),
            mem_buffer: buf.as_mut_ptr() as *mut c_void,
            no_alloc,
        };
        // SAFETY: the buffer is valid for `buf.len()` bytes and the caller
        // guarantees it outlives the context.
        let raw = unsafe { ggml_init(params) };
        NonNull::new(raw)
            .map(|raw| Context { raw })
            .ok_or(GgmlError::ContextAlloc { bytes: buf.len() })
    }

    /// The raw `ggml_context*`, for the backend path in `backend.rs`.
    ///
    /// `pub(crate)` and no further: handing this out publicly would let a
    /// caller allocate tensor data behind the arena's back, which is the one
    /// thing the whole memory design depends on not happening.
    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    /// Build a forward graph over `outputs` and hand back the raw `ggml_cgraph*`.
    ///
    /// Split out of `compute_many` because a *device* backend runs the same
    /// graph through `ggml_backend_graph_compute` instead of
    /// `ggml_graph_compute_with_ctx`. The graph itself is identical — only who
    /// executes it differs, which is the point of ggml's backend split.
    ///
    /// The returned pointer lives in this context's arena and must not outlive
    /// it; it is not freed separately.
    pub(crate) fn build_forward(&self, outputs: &[&Tensor<'_>]) -> Result<*mut c_void, GgmlError> {
        // SAFETY: valid context; the graph lives in the same arena.
        let graph = unsafe { ggml_new_graph(self.raw.as_ptr()) };
        if graph.is_null() {
            return Err(GgmlError::ArenaExhausted);
        }
        for out in outputs {
            // SAFETY: `graph` is non-null and every output was built here.
            unsafe { ggml_build_forward_expand(graph, out.raw.as_ptr()) };
        }
        Ok(graph.cast())
    }

    fn tensor<'a>(&'a self, raw: *mut ggml_tensor) -> Result<Tensor<'a>, GgmlError> {
        NonNull::new(raw)
            .map(|raw| Tensor {
                raw,
                _ctx: PhantomData,
            })
            // A null here means the arena ran out mid-graph, which is a sizing
            // mistake rather than a bug in the graph itself.
            .ok_or(GgmlError::ArenaExhausted)
    }

    pub fn new_f32_1d(&self, n: i64) -> Result<Tensor<'_>, GgmlError> {
        // SAFETY: valid context; type 0 is GGML_TYPE_F32.
        self.tensor(unsafe { ggml_new_tensor_1d(self.raw.as_ptr(), 0, n) })
    }

    pub fn new_f32_2d(&self, ne0: i64, ne1: i64) -> Result<Tensor<'_>, GgmlError> {
        // SAFETY: valid context; type 0 is GGML_TYPE_F32.
        self.tensor(unsafe { ggml_new_tensor_2d(self.raw.as_ptr(), 0, ne0, ne1) })
    }

    /// Matrix multiply. Follows ggml's convention: the result has `a`'s rows
    /// and `b`'s columns, and `a` is the one that may be quantized.
    pub fn mul_mat<'a>(&'a self, a: &Tensor<'a>, b: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: both tensors were built in this context and remain live.
        self.tensor(unsafe { ggml_mul_mat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) })
    }

    pub fn add<'a>(&'a self, a: &Tensor<'a>, b: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_add(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) })
    }

    pub fn mul<'a>(&'a self, a: &Tensor<'a>, b: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_mul(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) })
    }

    pub fn rms_norm<'a>(&'a self, a: &Tensor<'a>, eps: f32) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_rms_norm(self.raw.as_ptr(), a.raw.as_ptr(), eps) })
    }

    /// LayerNorm: subtract the mean, divide by the standard deviation.
    ///
    /// **Not interchangeable with [`rms_norm`](Self::rms_norm).** RMSNorm
    /// divides by the root-mean-square and never centres; LayerNorm centres
    /// first. Substituting one for the other is not an error and not a crash —
    /// it is fluent noise, which is exactly how StableLM and StarCoder2 read
    /// before this existed.
    ///
    /// The scale and shift are separate: `ggml` returns the normalised tensor
    /// and the caller multiplies by `weight` and adds `bias`. A LayerNorm
    /// *always* has a bias, which is the tell in a container.
    pub fn norm<'a>(&'a self, a: &Tensor<'a>, eps: f32) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_norm(self.raw.as_ptr(), a.raw.as_ptr(), eps) })
    }

    pub fn soft_max<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_soft_max(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    pub fn silu<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_silu(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    /// GELU with the tanh approximation — the Gemma family's gate activation.
    ///
    /// `ggml_gelu`, not `ggml_gelu_erf`: llama.cpp's `LLM_FFN_GELU` reaches
    /// `ggml_geglu_split`, whose kernel is the same tanh curve. The exact-erf
    /// variant is a *different function* and picking it silently shifts every
    /// FFN output.
    pub fn gelu<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_gelu(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    /// An I32 tensor — required for token ids and positions, which ggml
    /// rejects as f32.
    pub fn new_i32_1d(&self, n: i64) -> Result<Tensor<'_>, GgmlError> {
        // SAFETY: valid context; type 26 is GGML_TYPE_I32.
        self.tensor(unsafe { ggml_new_tensor_1d(self.raw.as_ptr(), 26, n) })
    }

    /// A 2-D I32 tensor — `mul_mat_id` requires expert indices in this shape.
    pub fn new_i32_2d(&self, ne0: i64, ne1: i64) -> Result<Tensor<'_>, GgmlError> {
        // SAFETY: valid context; type 26 is GGML_TYPE_I32.
        self.tensor(unsafe { ggml_new_tensor_2d(self.raw.as_ptr(), 26, ne0, ne1) })
    }

    pub fn new_f32_3d(&self, ne0: i64, ne1: i64, ne2: i64) -> Result<Tensor<'_>, GgmlError> {
        // SAFETY: valid context; type 0 is GGML_TYPE_F32.
        self.tensor(unsafe { ggml_new_tensor_3d(self.raw.as_ptr(), 0, ne0, ne1, ne2) })
    }

    /// A tensor of the given ggml type — used to hold quantized weights in
    /// their stored format, with no dequantization step.
    /// An F16 3-D tensor — the KV cache's layout, and what fused attention
    /// consumes without conversion.
    pub fn new_f16_3d(&self, ne0: i64, ne1: i64, ne2: i64) -> Result<Tensor<'_>, GgmlError> {
        // SAFETY: valid context; type 1 is GGML_TYPE_F16.
        self.tensor(unsafe { ggml_new_tensor_3d(self.raw.as_ptr(), 1, ne0, ne1, ne2) })
    }

    /// An F16 4-D tensor — a 2-D convolution kernel's shape,
    /// `[kw, kh, in_channels, out_channels]`.
    ///
    /// Exists so that building a kernel `conv_2d` will accept does not require
    /// remembering that F32 aborts.
    pub fn new_f16_4d(
        &self,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> Result<Tensor<'_>, GgmlError> {
        // SAFETY: valid context; type 1 is GGML_TYPE_F16.
        self.tensor(unsafe { ggml_new_tensor_4d(self.raw.as_ptr(), 1, ne0, ne1, ne2, ne3) })
    }

    pub fn new_typed_2d(
        &self,
        ty: chaos_gguf::GgmlType,
        ne0: i64,
        ne1: i64,
    ) -> Result<Tensor<'_>, GgmlError> {
        // SAFETY: valid context; the type id is passed through to ggml, which
        // validates it and returns null for anything it does not know.
        self.tensor(unsafe { ggml_new_tensor_2d(self.raw.as_ptr(), ty.0 as c_int, ne0, ne1) })
    }

    /// Embedding lookup: gather rows of `a` at the indices in `b`.
    pub fn get_rows<'a>(&'a self, a: &Tensor<'a>, b: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: both tensors belong to this context.
        self.tensor(unsafe { ggml_get_rows(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) })
    }

    pub fn view_4d<'a>(
        &'a self,
        a: &Tensor<'a>,
        ne: [i64; 4],
        nb: [usize; 3],
        offset_bytes: usize,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above. ggml does not bounds-check views.
        self.tensor(unsafe {
            ggml_view_4d(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                ne[0],
                ne[1],
                ne[2],
                ne[3],
                nb[0],
                nb[1],
                nb[2],
                offset_bytes,
            )
        })
    }

    pub fn reshape_4d<'a>(&'a self, a: &Tensor<'a>, ne: [i64; 4]) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe {
            ggml_reshape_4d(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                ne[0],
                ne[1],
                ne[2],
                ne[3],
            )
        })
    }

    pub fn cont_4d<'a>(&'a self, a: &Tensor<'a>, ne: [i64; 4]) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe {
            ggml_cont_4d(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                ne[0],
                ne[1],
                ne[2],
                ne[3],
            )
        })
    }

    /// Broadcast to a larger shape. Used to widen the delta net's 16 key heads
    /// to its 48 value heads.
    pub fn repeat_4d<'a>(&'a self, a: &Tensor<'a>, ne: [i64; 4]) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe {
            ggml_repeat_4d(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                ne[0],
                ne[1],
                ne[2],
                ne[3],
            )
        })
    }

    /// Divide each row by its own L2 norm.
    ///
    /// **Not `rms_norm`.** That divides by the root *mean* square and takes a
    /// learned weight; this divides by the norm and takes none. Substituting
    /// one for the other scales every row by `sqrt(n)` and the model answers
    /// fluently and wrongly.
    pub fn l2_norm<'a>(&'a self, a: &Tensor<'a>, eps: f32) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_l2_norm(self.raw.as_ptr(), a.raw.as_ptr(), eps) })
    }

    /// Depthwise causal convolution over `[window + n_tokens - 1, channels]`.
    pub fn ssm_conv<'a>(
        &'a self,
        a: &Tensor<'a>,
        kernel: &Tensor<'a>,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe {
            ggml_ssm_conv(self.raw.as_ptr(), a.raw.as_ptr(), kernel.raw.as_ptr())
        })
    }

    /// 2-D convolution. `kernel` must be F16; see the declaration for why.
    ///
    /// Argument order is `(kernel, data)`, matching ggml. The wrapper keeps that
    /// order rather than "fixing" it, because a wrapper that disagrees with the
    /// library it wraps is worse than one that reads oddly.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_2d<'a>(
        &'a self,
        kernel: &Tensor<'a>,
        data: &Tensor<'a>,
        stride: (i32, i32),
        pad: (i32, i32),
        dilation: (i32, i32),
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above. **The F16 kernel requirement is not checked here**
        // because `Tensor` exposes no type accessor, and inventing one for a
        // single call site is worse than saying it plainly: an F32 kernel aborts
        // the process. `new_f16_4d` exists so the correct thing is the easy one.
        self.tensor(unsafe {
            ggml_conv_2d(
                self.raw.as_ptr(),
                kernel.raw.as_ptr(),
                data.raw.as_ptr(),
                stride.0,
                stride.1,
                pad.0,
                pad.1,
                dilation.0,
                dilation.1,
            )
        })
    }

    /// Group normalisation over `n_groups` groups of channels.
    pub fn group_norm<'a>(
        &'a self,
        a: &Tensor<'a>,
        n_groups: i32,
        eps: f32,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_group_norm(self.raw.as_ptr(), a.raw.as_ptr(), n_groups, eps) })
    }

    /// The fused gated delta rule. See the declaration for the shapes; the
    /// result packs the scores then `snapshots` copies of the carried state.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net<'a>(
        &'a self,
        q: &Tensor<'a>,
        k: &Tensor<'a>,
        v: &Tensor<'a>,
        g: &Tensor<'a>,
        beta: &Tensor<'a>,
        state: &Tensor<'a>,
        snapshots: i64,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above. ggml asserts the shape relationships itself and
        // **aborts** rather than returning an error, so the caller checks them
        // on the Rust side first.
        self.tensor(unsafe {
            ggml_gated_delta_net(
                self.raw.as_ptr(),
                q.raw.as_ptr(),
                k.raw.as_ptr(),
                v.raw.as_ptr(),
                g.raw.as_ptr(),
                beta.raw.as_ptr(),
                state.raw.as_ptr(),
                snapshots,
            )
        })
    }

    /// Multi-section rotary embedding (mRoPE).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_multi<'a>(
        &'a self,
        a: &Tensor<'a>,
        pos: &Tensor<'a>,
        n_dims: i32,
        sections: [i32; 4],
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above. `sections` is read as four `int`s and not retained.
        self.tensor(unsafe {
            ggml_rope_multi(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                pos.raw.as_ptr(),
                std::ptr::null_mut(),
                n_dims,
                sections.as_ptr(),
                mode,
                n_ctx_orig,
                freq_base,
                freq_scale,
                0.0,
                1.0,
                32.0,
                1.0,
            )
        })
    }

    pub fn concat<'a>(
        &'a self,
        a: &Tensor<'a>,
        b: &Tensor<'a>,
        dim: i32,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_concat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr(), dim) })
    }

    pub fn permute<'a>(&'a self, a: &Tensor<'a>, axes: [i32; 4]) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe {
            ggml_permute(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                axes[0],
                axes[1],
                axes[2],
                axes[3],
            )
        })
    }

    pub fn transpose<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_transpose(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    /// Materialise a view into contiguous memory.
    ///
    /// Views and permutes only change how a tensor is *interpreted*; several
    /// ops require contiguous input, and this is what makes them legal.
    pub fn cont<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_cont(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    pub fn reshape_2d<'a>(
        &'a self,
        a: &Tensor<'a>,
        ne0: i64,
        ne1: i64,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above; ggml validates that the element count matches.
        self.tensor(unsafe { ggml_reshape_2d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, ne1) })
    }

    pub fn reshape_3d<'a>(
        &'a self,
        a: &Tensor<'a>,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_reshape_3d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, ne1, ne2) })
    }

    pub fn view_2d<'a>(
        &'a self,
        a: &Tensor<'a>,
        ne0: i64,
        ne1: i64,
        row_stride_bytes: usize,
        offset_bytes: usize,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above. The caller is responsible for the offset and
        // stride being inside `a` -- ggml does not bounds-check views.
        self.tensor(unsafe {
            ggml_view_2d(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                ne0,
                ne1,
                row_stride_bytes,
                offset_bytes,
            )
        })
    }

    /// A strided 3-D window into `a`.
    ///
    /// This is how DeepSeek's decoupled RoPE is expressed: `q` is
    /// `[head_dim, n_head, n_tokens]` and only the trailing `n_rot` of each
    /// head's `head_dim` carries position, so `q_pe` is a view with the full
    /// head stride but a shorter `ne0` and a non-zero offset. Both strides stay
    /// those of the *source*, which is what makes the view non-contiguous and
    /// what a `reshape` cannot express.
    #[allow(clippy::too_many_arguments)]
    pub fn view_3d<'a>(
        &'a self,
        a: &Tensor<'a>,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset_bytes: usize,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above. ggml does not bounds-check views, so the offset and
        // both strides are the caller's responsibility.
        self.tensor(unsafe {
            ggml_view_3d(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                ne0,
                ne1,
                ne2,
                nb1,
                nb2,
                offset_bytes,
            )
        })
    }

    /// Fused attention: scores, mask, softmax and the value product in one op,
    /// without ever materialising the scores.
    ///
    /// The explicit version writes an `n_kv * n_batch * n_head` float matrix,
    /// reads it back for the softmax and again for the value product. At 4395
    /// tokens with a 512-token block that is 288 MiB written and read twice,
    /// per layer — which measured at roughly 4 GFLOPS, an order of magnitude
    /// below what the arithmetic alone should cost. This keeps the running
    /// softmax in registers instead.
    ///
    /// Shapes ggml requires, which are not the ones the explicit path uses:
    /// - `q`: `[head_dim, n_batch, n_head]`
    /// - `k`: `[head_dim, n_kv, n_head_kv]`
    /// - `v`: `[head_dim, n_kv, n_head_kv]` — **not** transposed, unlike `mul_mat`
    /// - `mask`: `[n_kv, n_batch]`, **F16 and contiguous** (ggml asserts both)
    /// - result: `[head_dim, n_head, n_batch]`, already permuted for reshaping
    pub fn flash_attn_ext<'a>(
        &'a self,
        q: &Tensor<'a>,
        k: &Tensor<'a>,
        v: &Tensor<'a>,
        mask: &Tensor<'a>,
        scale: f32,
        // Gemma-2 caps attention logits at 50 before the softmax; 0.0 means no
        // cap, which is every other architecture here. It has to reach the
        // fused kernel rather than be applied afterwards -- the logits do not
        // exist outside it.
        logit_softcap: f32,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: all four tensors live in this context; ggml validates the
        // shape relationships and returns null on a mismatch.
        self.tensor(unsafe {
            ggml_flash_attn_ext(
                self.raw.as_ptr(),
                q.raw.as_ptr(),
                k.raw.as_ptr(),
                v.raw.as_ptr(),
                mask.raw.as_ptr(),
                scale,
                0.0, // max_bias: ALiBi, which this architecture does not use
                logit_softcap,
            )
        })
    }

    /// Fused attention with per-head sinks.
    ///
    /// A sink is one extra always-attended logit per head, learned, with no
    /// key and no value: it lets a head attend to "nothing" and so changes the
    /// softmax denominator for every score. V4-Flash ships one per head in
    /// `attn_sinks.weight`, and dropping them is not a small error — it
    /// rescales the whole attention output while producing a tensor of exactly
    /// the right shape.
    ///
    /// ggml attaches sinks by mutating the node rather than by taking them as
    /// an argument, so the two calls are wrapped together here: an
    /// `add_sinks` that a caller forgets is silent.
    pub fn flash_attn_ext_with_sinks<'a>(
        &'a self,
        q: &Tensor<'a>,
        k: &Tensor<'a>,
        v: &Tensor<'a>,
        mask: &Tensor<'a>,
        sinks: &Tensor<'a>,
        scale: f32,
    ) -> Result<Tensor<'a>, GgmlError> {
        // Sinks and Gemma soft-capping have never co-occurred; 0.0 is correct
        // for every architecture that uses this path.
        let out = self.flash_attn_ext(q, k, v, mask, scale, 0.0)?;
        // SAFETY: both nodes live in this context, and `out` is the tensor
        // ggml just returned from `flash_attn_ext`, which is what this expects.
        unsafe { ggml_flash_attn_ext_add_sinks(out.raw.as_ptr(), sinks.raw.as_ptr()) };
        Ok(out)
    }

    /// Broadcast `a` up to the shape of `b`.
    pub fn repeat<'a>(&'a self, a: &Tensor<'a>, b: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: both tensors live in this context; ggml checks that the
        // shapes are compatible for repetition.
        self.tensor(unsafe { ggml_repeat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) })
    }

    /// A contiguous 1-D window into `a`, `offset` bytes from its start.
    pub fn view_1d<'a>(
        &'a self,
        a: &Tensor<'a>,
        ne0: i64,
        offset: usize,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: valid context and tensor; ggml validates the window against
        // the source's extent.
        self.tensor(unsafe { ggml_view_1d(self.raw.as_ptr(), a.raw.as_ptr(), ne0, offset) })
    }

    /// Collapse the hyper-connection streams into one vector.
    ///
    /// `x` is `[n_embd, hc, n_tokens]`, `weights` is `[hc, n_tokens]`, and the
    /// result is `[n_embd, n_tokens]` where
    /// `result[i, t] = sum_h x[i, h, t] * weights[h, t]`.
    ///
    /// DeepSeek-V4 replaces the plain residual (`x = f(x) + x`) with several
    /// parallel streams; this is the read side of that. See
    /// <https://arxiv.org/pdf/2512.24880>.
    pub fn dsv4_hc_pre<'a>(
        &'a self,
        x: &Tensor<'a>,
        weights: &Tensor<'a>,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: both tensors live in this context.
        self.tensor(unsafe {
            ggml_dsv4_hc_pre(self.raw.as_ptr(), x.raw.as_ptr(), weights.raw.as_ptr())
        })
    }

    /// Write a block's output back across the hyper-connection streams.
    ///
    /// `result[i, dst, t] = x[i, t]*post[dst, t] + sum_src residual[i, src, t]*comb[dst, src, t]`
    pub fn dsv4_hc_post<'a>(
        &'a self,
        x: &Tensor<'a>,
        residual: &Tensor<'a>,
        post: &Tensor<'a>,
        comb: &Tensor<'a>,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: all four tensors live in this context.
        self.tensor(unsafe {
            ggml_dsv4_hc_post(
                self.raw.as_ptr(),
                x.raw.as_ptr(),
                residual.raw.as_ptr(),
                post.raw.as_ptr(),
                comb.raw.as_ptr(),
            )
        })
    }

    /// Build the stream-mixing matrix, Sinkhorn-normalised.
    ///
    /// `n_iter` is the Sinkhorn iteration count — 20 for V4-Flash, read from
    /// `hyper_connection.sinkhorn_iterations`. ggml implements the
    /// normalisation itself, so this is a binding rather than an algorithm to
    /// reproduce.
    pub fn dsv4_hc_comb<'a>(
        &'a self,
        mixes: &Tensor<'a>,
        scale: &Tensor<'a>,
        base: &Tensor<'a>,
        eps: f32,
        n_iter: i32,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: all three tensors live in this context.
        self.tensor(unsafe {
            ggml_dsv4_hc_comb(
                self.raw.as_ptr(),
                mixes.raw.as_ptr(),
                scale.raw.as_ptr(),
                base.raw.as_ptr(),
                eps,
                n_iter as c_int,
            )
        })
    }

    /// Sparse-attention indexer: score every key so the top-k can be kept.
    ///
    /// `mask` must be F16, as with [`Self::flash_attn_ext`], and `weights` is
    /// expected already scaled.
    pub fn lightning_indexer<'a>(
        &'a self,
        q: &Tensor<'a>,
        k: &Tensor<'a>,
        weights: &Tensor<'a>,
        mask: &Tensor<'a>,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: all four tensors live in this context.
        self.tensor(unsafe {
            ggml_lightning_indexer(
                self.raw.as_ptr(),
                q.raw.as_ptr(),
                k.raw.as_ptr(),
                weights.raw.as_ptr(),
                mask.raw.as_ptr(),
            )
        })
    }

    /// Hyperbolic tangent, elementwise.
    ///
    /// Needed for Gemma's soft-capping: `tanh(x / cap) * cap` bounds a tensor
    /// smoothly instead of clipping it. Gemma-2 caps attention logits at 50 and
    /// final logits at 30, and without it the model produces fluent nonsense
    /// rather than any error.
    pub fn tanh<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: valid context and tensor from it.
        self.tensor(unsafe { ggml_tanh(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    /// Gemma's soft cap: `tanh(x / cap) * cap`, a smooth bound on magnitude.
    ///
    /// A `cap` of zero means "no capping" and returns the tensor unchanged,
    /// which is what a container that declares no soft-capping wants.
    pub fn softcap<'a>(&'a self, a: &Tensor<'a>, cap: f32) -> Result<Tensor<'a>, GgmlError> {
        if cap <= 0.0 {
            return Ok(*a);
        }
        let scaled = self.scale(a, 1.0 / cap)?;
        let t = self.tanh(&scaled)?;
        self.scale(&t, cap)
    }

    pub fn scale<'a>(&'a self, a: &Tensor<'a>, s: f32) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_scale(self.raw.as_ptr(), a.raw.as_ptr(), s) })
    }

    pub fn sigmoid<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_sigmoid(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    pub fn relu<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_relu(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    pub fn div<'a>(&'a self, a: &Tensor<'a>, b: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_div(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) })
    }

    pub fn sum_rows<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_sum_rows(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    /// `log(1 + exp(x))`.
    ///
    /// V4-Flash's router gate is `sqrt(softplus(logits))` —
    /// `expert_gating_func 4`, `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`.
    /// Neither softmax nor sigmoid, and substituting either produces a
    /// perfectly ordinary-looking distribution over experts.
    pub fn softplus<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: valid context and tensor.
        self.tensor(unsafe { ggml_softplus(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    pub fn sqrt<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: valid context and tensor.
        self.tensor(unsafe { ggml_sqrt(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    /// Clamp elementwise into `[min, max]`.
    ///
    /// Bounds are often *not* symmetric here: V4-Flash clamps the SwiGLU gate
    /// to `(-inf, limit]` and the up projection to `[-limit, limit]`
    /// (`llama-graph.cpp:2050-2057`, a `LLM_ARCH_DEEPSEEK4` branch). Pass
    /// `f32::NEG_INFINITY` / `f32::INFINITY` for a one-sided bound.
    pub fn clamp<'a>(
        &'a self,
        a: &Tensor<'a>,
        min: f32,
        max: f32,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: valid context and tensor.
        self.tensor(unsafe { ggml_clamp(self.raw.as_ptr(), a.raw.as_ptr(), min, max) })
    }

    /// `silu(gate) * up`, with gate and up as separate tensors.
    ///
    /// The fused form. Doing it as `mul(silu(gate), up)` is arithmetically the
    /// same but is a different graph, and llama.cpp uses this one for
    /// DeepSeek4 — worth matching when comparing against its numbers.
    pub fn swiglu_split<'a>(
        &'a self,
        gate: &Tensor<'a>,
        up: &Tensor<'a>,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: both tensors live in this context; ggml checks their shapes
        // agree.
        self.tensor(unsafe {
            ggml_swiglu_split(self.raw.as_ptr(), gate.raw.as_ptr(), up.raw.as_ptr())
        })
    }

    /// Indices of the `k` largest values per row — MoE expert selection.
    pub fn top_k<'a>(&'a self, a: &Tensor<'a>, k: i32) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_top_k(self.raw.as_ptr(), a.raw.as_ptr(), k) })
    }

    /// The `k` largest per row, as indices, via a full descending argsort.
    ///
    /// Not the same op as [`Self::top_k`], and the difference matters here:
    /// `top_k` returns its indices in *no particular order* — already a
    /// hard-won note in this project — whereas this is `argsort` followed by a
    /// view, so index 0 is the largest. llama.cpp selects MoE experts with this
    /// one (`llama-graph.cpp:1932`), and the captured trace shows both nodes:
    /// `ffn_moe_argsort` for the sort and `ffn_moe_topk` for the view.
    pub fn argsort_top_k<'a>(&'a self, a: &Tensor<'a>, k: i32) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: valid context and tensor.
        self.tensor(unsafe { ggml_argsort_top_k(self.raw.as_ptr(), a.raw.as_ptr(), k) })
    }

    /// Rotary position embedding.
    ///
    /// `positions` must be an I32 tensor of token positions. `freq_factors`
    /// is optional and carries per-frequency scaling for extended context.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_ext<'a>(
        &'a self,
        a: &Tensor<'a>,
        positions: &Tensor<'a>,
        freq_factors: Option<&Tensor<'a>>,
        n_dims: i32,
        mode: i32,
        n_ctx_orig: i32,
        rope: RopeParams,
    ) -> Result<Tensor<'a>, GgmlError> {
        let c = freq_factors
            .map(|t| t.raw.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        // SAFETY: all tensors belong to this context; a null `c` is the
        // documented way to omit frequency factors.
        self.tensor(unsafe {
            ggml_rope_ext(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                positions.raw.as_ptr(),
                c,
                n_dims,
                mode,
                n_ctx_orig,
                rope.freq_base,
                rope.freq_scale,
                rope.ext_factor,
                rope.attn_factor,
                rope.beta_fast,
                rope.beta_slow,
            )
        })
    }

    /// The inverse rotation.
    ///
    /// Named "back" because ggml uses it for the backward pass, but here it is
    /// forward arithmetic: V4-Flash **de-ropes its attention output** before the
    /// output projection, undoing the rotation on the same trailing `n_rot`
    /// dims (`deepseek4.cpp:1074`). Skip it and the rotation stays baked into
    /// the residual stream — visible only in a captured trace, never in the
    /// tensor shapes. Parameters must match the forward `rope_ext` exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_ext_back<'a>(
        &'a self,
        a: &Tensor<'a>,
        positions: &Tensor<'a>,
        freq_factors: Option<&Tensor<'a>>,
        n_dims: i32,
        mode: i32,
        n_ctx_orig: i32,
        rope: RopeParams,
    ) -> Result<Tensor<'a>, GgmlError> {
        let c = freq_factors
            .map(|t| t.raw.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        // SAFETY: all tensors belong to this context; a null `c` omits the
        // frequency factors, as in `rope_ext`.
        self.tensor(unsafe {
            ggml_rope_ext_back(
                self.raw.as_ptr(),
                a.raw.as_ptr(),
                positions.raw.as_ptr(),
                c,
                n_dims,
                mode,
                n_ctx_orig,
                rope.freq_base,
                rope.freq_scale,
                rope.ext_factor,
                rope.attn_factor,
                rope.beta_fast,
                rope.beta_slow,
            )
        })
    }

    /// Indexed matmul for mixture-of-experts.
    ///
    /// `experts` is a stack of matrices; `ids` selects which one each row
    /// uses. This is the operation that makes MoE cheap: only the chosen
    /// experts are multiplied, instead of computing all of them and masking.
    pub fn mul_mat_id<'a>(
        &'a self,
        experts: &Tensor<'a>,
        b: &Tensor<'a>,
        ids: &Tensor<'a>,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: all three tensors belong to this context.
        self.tensor(unsafe {
            ggml_mul_mat_id(
                self.raw.as_ptr(),
                experts.raw.as_ptr(),
                b.raw.as_ptr(),
                ids.raw.as_ptr(),
            )
        })
    }

    /// Softmax with an optional additive mask and a scale applied first.
    ///
    /// Attention needs all three in one op: scale by 1/sqrt(head_dim), add the
    /// causal mask, then normalise. Doing them separately is both slower and
    /// numerically worse.
    pub fn soft_max_ext<'a>(
        &'a self,
        a: &Tensor<'a>,
        mask: Option<&Tensor<'a>>,
        scale: f32,
        max_bias: f32,
    ) -> Result<Tensor<'a>, GgmlError> {
        let mask_ptr = mask.map(|m| m.raw.as_ptr()).unwrap_or(std::ptr::null_mut());
        // SAFETY: tensors belong to this context; a null mask is the
        // documented way to omit it.
        self.tensor(unsafe {
            ggml_soft_max_ext(self.raw.as_ptr(), a.raw.as_ptr(), mask_ptr, scale, max_bias)
        })
    }

    /// Build a graph ending at `output` and run it on `threads` threads.
    ///
    /// Nothing has been computed before this call — the tensors describe a
    /// plan, not values.
    /// Evaluate several outputs in **one** graph.
    ///
    /// `compute` re-evaluates the whole ancestor graph of its output, so
    /// computing `q`, then `k`, then `v` — which share a normalisation — does
    /// that normalisation three times, and pays three graph builds and three
    /// threadpool cycles for it. At one token those fixed costs are the
    /// dominant term: the matmuls are matrix-*vector* products and tiny.
    ///
    /// `ggml_build_forward_expand` accepts several roots on one graph, so this
    /// is the same work with the sharing preserved.
    pub fn compute_many(&self, outputs: &[&Tensor<'_>], threads: usize) -> Result<(), GgmlError> {
        if outputs.is_empty() {
            return Ok(());
        }
        // SAFETY: valid context; the graph lives in the same arena.
        let graph = unsafe { ggml_new_graph(self.raw.as_ptr()) };
        if graph.is_null() {
            return Err(GgmlError::ArenaExhausted);
        }
        for out in outputs {
            // SAFETY: `graph` is non-null and every output was built here.
            unsafe { ggml_build_forward_expand(graph, out.raw.as_ptr()) };
        }
        // SAFETY: graph and context match.
        let status = unsafe {
            ggml_graph_compute_with_ctx(self.raw.as_ptr(), graph, threads.max(1) as c_int)
        };
        if status != 0 {
            return Err(GgmlError::ComputeFailed(status));
        }
        Ok(())
    }

    pub fn compute(&self, output: &Tensor<'_>, threads: usize) -> Result<(), GgmlError> {
        // SAFETY: valid context; the returned graph lives in the same arena.
        let graph = unsafe { ggml_new_graph(self.raw.as_ptr()) };
        if graph.is_null() {
            return Err(GgmlError::ArenaExhausted);
        }
        // SAFETY: `graph` is non-null and `output` was built in this context.
        unsafe { ggml_build_forward_expand(graph, output.raw.as_ptr()) };
        // SAFETY: graph and context match; ggml allocates its own scratch for
        // the requested thread count.
        let status = unsafe {
            ggml_graph_compute_with_ctx(self.raw.as_ptr(), graph, threads.max(1) as c_int)
        };
        if status != 0 {
            return Err(GgmlError::ComputeFailed(status));
        }
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: `raw` came from ggml_init and is freed exactly once, here.
        // Tensors borrow the context, so none can outlive this.
        unsafe { ggml_free(self.raw.as_ptr()) };
    }
}

/// A tensor inside a [`Context`]'s arena.
///
/// Borrows the context: a tensor cannot outlive the memory backing it.
#[derive(Clone, Copy)]
pub struct Tensor<'a> {
    raw: NonNull<ggml_tensor>,
    _ctx: PhantomData<&'a Context>,
}

impl Tensor<'_> {
    /// The raw `ggml_tensor*`, for the backend path in `backend.rs`.
    /// Mark this tensor as a **graph input**, i.e. something the caller fills.
    ///
    /// # Why this is not cosmetic
    ///
    /// `ggml_backend_sched` reads the flag. A leaf with no buffer, no data and
    /// no op is otherwise something it cannot place, and an unplaced node
    /// reaches `ggml_gallocr_allocate_node` as backend `-1`:
    ///
    /// ```text
    /// ggml-alloc.c:623: GGML_ASSERT(buffer_id >= 0) failed
    /// ```
    ///
    /// which **aborts the process**. With the flag it takes the documented
    /// path -- `cur_backend_id = sched->n_backends - 1`, the last backend,
    /// which is why the CPU must be passed last.
    ///
    /// llama.cpp marks every graph input this way. That is not a style choice,
    /// and the day spent finding it is the argument for saying so here.
    pub fn set_input(&self) {
        // SAFETY: `self` is a live tensor in a live context; the call only sets
        // a flag bit.
        unsafe { ggml_set_input(self.raw.as_ptr()) };
    }

    /// Mark this tensor as a graph **output**, read back after the compute.
    pub fn set_output(&self) {
        // SAFETY: as above.
        unsafe { ggml_set_output(self.raw.as_ptr()) };
    }

    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    pub fn len(&self) -> i64 {
        // SAFETY: valid tensor pointer for the context's lifetime.
        unsafe { ggml_nelements(self.raw.as_ptr()) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> usize {
        // SAFETY: as above.
        unsafe { ggml_nbytes(self.raw.as_ptr()) }
    }

    /// Fill this tensor with `values`.
    ///
    /// Rejects a length mismatch rather than writing past the allocation.
    pub fn set_f32(&self, values: &[f32]) -> Result<(), GgmlError> {
        let n = self.len() as usize;
        if values.len() != n {
            return Err(GgmlError::WrongSize {
                expected: n,
                actual: values.len(),
            });
        }
        // SAFETY: the tensor holds `n` f32 slots (checked above) and `values`
        // has exactly `n`; the regions are distinct allocations.
        unsafe {
            let dst = ggml_get_data_f32(self.raw.as_ptr());
            std::ptr::copy_nonoverlapping(values.as_ptr(), dst, n);
        }
        Ok(())
    }

    /// Write raw bytes — used to place already-quantized weights directly,
    /// with no dequantization step.
    pub fn set_bytes(&self, data: &[u8]) -> Result<(), GgmlError> {
        let n = self.bytes();
        if data.len() != n {
            return Err(GgmlError::WrongSize {
                expected: n,
                actual: data.len(),
            });
        }
        // SAFETY: the tensor's allocation is `n` bytes (from ggml_nbytes) and
        // `data` has exactly `n`; distinct allocations.
        unsafe {
            let dst = ggml_get_data(self.raw.as_ptr()) as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, n);
        }
        Ok(())
    }

    /// Fill an I32 tensor — token ids, positions, expert indices.
    pub fn set_i32(&self, values: &[i32]) -> Result<(), GgmlError> {
        let n = self.len() as usize;
        if values.len() != n {
            return Err(GgmlError::WrongSize {
                expected: n,
                actual: values.len(),
            });
        }
        // SAFETY: the tensor holds `n` i32 slots and `values` has exactly `n`;
        // distinct allocations.
        unsafe {
            let dst = ggml_get_data(self.raw.as_ptr()) as *mut i32;
            std::ptr::copy_nonoverlapping(values.as_ptr(), dst, n);
        }
        Ok(())
    }

    /// Read an I32 tensor back, in logical index order — index tensors, not
    /// values.
    ///
    /// **Strides are honoured**, for the same reason as [`Self::to_vec_f32`]
    /// and with a sharper failure mode. `argsort_top_k` returns a *view* of the
    /// full sort — 6 indices out of every row of 256 — so a flat read returns
    /// the first `6 * n_tokens` entries of token 0's ranking and calls them
    /// every token's experts. Token 0 then looks perfect and every later token
    /// silently routes to the wrong experts. Nothing about the result's shape,
    /// type or magnitude gives it away.
    pub fn to_vec_i32(&self) -> Vec<i32> {
        let n = self.len() as usize;
        // SAFETY: valid tensor of i32 for the context's lifetime.
        let src = unsafe { ggml_get_data(self.raw.as_ptr()) } as *const u8;
        if self.is_contiguous() {
            // SAFETY: contiguous, so `n` i32s follow the data pointer.
            return unsafe { std::slice::from_raw_parts(src as *const i32, n).to_vec() };
        }

        let (ne, nb) = self.dims_and_strides();
        let mut out = Vec::with_capacity(n);
        for i3 in 0..ne[3] {
            for i2 in 0..ne[2] {
                for i1 in 0..ne[1] {
                    for i0 in 0..ne[0] {
                        let off = i3 as usize * nb[3]
                            + i2 as usize * nb[2]
                            + i1 as usize * nb[1]
                            + i0 as usize * nb[0];
                        // SAFETY: offset built from this tensor's own extents
                        // and strides, so it lands inside its buffer.
                        out.push(unsafe { *(src.add(off) as *const i32) });
                    }
                }
            }
        }
        out
    }

    /// This tensor's data pointer, read through the mirrored struct layout.
    ///
    /// # Safety
    /// The tensor must be live. Verified against ggml's own accessor by
    /// `weights::tests::our_struct_layout_matches_ggmls`.
    /// Used only by the layout-verification test; the production path writes
    /// this pointer rather than reading it. Kept because a silent ggml struct
    /// change would corrupt every weight binding, and that test is the alarm.
    #[allow(dead_code)]
    pub(crate) unsafe fn data_ptr(&self) -> *mut std::os::raw::c_void {
        (*(self.raw.as_ptr() as *const crate::weights::RawTensor)).data
    }

    /// Aim this tensor at memory the caller owns, without copying.
    ///
    /// # Safety
    /// `ptr` must address at least [`Self::bytes`] readable bytes and must stay
    /// valid and unmoved for as long as this tensor is used. The tensor does
    /// not take ownership and will not keep the memory alive — a dangling
    /// pointer here reads freed memory *successfully*, yielding plausible
    /// numbers instead of a crash.
    /// The raw `ggml_tensor`, for the repack path which must call into
    /// `ggml-backend` with it.
    pub(crate) fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.raw.as_ptr().cast()
    }

    pub(crate) unsafe fn set_data_ptr(&self, ptr: *mut std::os::raw::c_void) {
        (*(self.raw.as_ptr() as *mut crate::weights::RawTensor)).data = ptr;
    }

    /// This tensor's extents and byte strides, in ggml's `ne`/`nb` order.
    pub fn dims_and_strides(&self) -> ([i64; 4], [usize; 4]) {
        // SAFETY: valid tensor pointer for the context's lifetime; `RawTensor`
        // mirrors ggml's layout, which the FFI already relies on elsewhere.
        let raw = unsafe { &*(self.raw.as_ptr() as *const crate::weights::RawTensor) };
        (raw.ne, raw.nb)
    }

    /// True when the elements are laid out back to back, so a flat read is
    /// equivalent to a strided one.
    pub fn is_contiguous(&self) -> bool {
        let (ne, nb) = self.dims_and_strides();
        let mut expect = nb[0];
        for d in 1..4 {
            if ne[d] > 1 && nb[d] != expect {
                return false;
            }
            expect *= ne[d].max(1) as usize;
        }
        true
    }

    /// Read the tensor's values back as `f32`, in logical index order.
    ///
    /// **Strides are honoured, and that is not a detail.** A view produced by
    /// [`Context::view_2d`] or [`Context::view_3d`] keeps the *source's*
    /// strides, so its elements are scattered through the parent's buffer — the
    /// decoupled-RoPE split (64 rotated dims out of every 512) is exactly this
    /// shape. Reading such a view as a flat run of `nelements` floats returns
    /// entirely different numbers, with no error and nothing obviously wrong
    /// about them: this function used to do that, and it made a correct graph
    /// look like a wrong one.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        let n = self.len() as usize;
        // SAFETY: valid tensor of f32 for the context's lifetime.
        let src = unsafe { ggml_get_data_f32(self.raw.as_ptr()) } as *const u8;
        if self.is_contiguous() {
            // SAFETY: contiguous, so `n` floats follow the data pointer.
            return unsafe { std::slice::from_raw_parts(src as *const f32, n).to_vec() };
        }

        let (ne, nb) = self.dims_and_strides();
        let mut out = Vec::with_capacity(n);
        for i3 in 0..ne[3] {
            for i2 in 0..ne[2] {
                for i1 in 0..ne[1] {
                    for i0 in 0..ne[0] {
                        let off = i3 as usize * nb[3]
                            + i2 as usize * nb[2]
                            + i1 as usize * nb[1]
                            + i0 as usize * nb[0];
                        // SAFETY: the offset is built from the tensor's own
                        // extents and strides, so it lands inside its buffer.
                        out.push(unsafe { *(src.add(off) as *const f32) });
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARENA: usize = 16 << 20;

    /// A strided view must read the elements it describes, not the bytes that
    /// happen to follow its data pointer.
    ///
    /// This is the shape of DeepSeek's decoupled RoPE — the trailing dims of
    /// each head, so a short window with the *parent's* stride — and reading it
    /// flat is silent: the right number of plausible floats, all of them the
    /// wrong ones. It made a correct V4-Flash graph look broken for as long as
    /// it took to notice the two candidate sums differed by more than any
    /// arithmetic error could.
    #[test]
    fn a_strided_view_reads_its_own_elements_not_the_flat_run() {
        let ctx = Context::new(ARENA).expect("context");

        // 3 rows of 4: [0 1 2 3][4 5 6 7][8 9 10 11]
        let t = ctx.new_f32_2d(4, 3).expect("t");
        t.set_f32(&(0..12).map(|v| v as f32).collect::<Vec<_>>())
            .expect("set");

        // The last 2 of each row: [2 3][6 7][10 11].
        let f32_size = std::mem::size_of::<f32>();
        let tail = ctx
            .view_2d(&t, 2, 3, 4 * f32_size, 2 * f32_size)
            .expect("view");

        assert!(!tail.is_contiguous(), "a windowed view is not contiguous");
        assert_eq!(tail.to_vec_f32(), vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0]);

        // What the flat read gave instead, spelled out: six consecutive floats
        // from the offset. Same length, same types, no error.
        let flat: Vec<f32> = (2..8).map(|v| v as f32).collect();
        assert_ne!(tail.to_vec_f32(), flat);
    }

    /// The 3-D case, which is the one the Q projection actually uses: 2 of
    /// every 4 dims, across 2 heads and 2 tokens.
    #[test]
    fn a_strided_3d_view_walks_both_strides() {
        let ctx = Context::new(ARENA).expect("context");

        // [4 dims, 2 heads, 2 tokens] filled 0..15.
        let t = ctx.new_f32_3d(4, 2, 2).expect("t");
        t.set_f32(&(0..16).map(|v| v as f32).collect::<Vec<_>>())
            .expect("set");

        let f32_size = std::mem::size_of::<f32>();
        let pe = ctx
            .view_3d(&t, 2, 2, 2, 4 * f32_size, 8 * f32_size, 2 * f32_size)
            .expect("view_3d");

        // The trailing pair of each head: rows start at 0,4,8,12.
        assert_eq!(
            pe.to_vec_f32(),
            vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0]
        );
    }

    /// The i32 twin of the strided-view test, and the one with teeth.
    ///
    /// `argsort_top_k` hands back a view of the full sort — k indices out of
    /// every row — so reading it flat returns token 0's ranking spread across
    /// every token. In a MoE that means token 0 routes correctly and every
    /// other token routes to whatever happened to follow in memory, with the
    /// right count of plausible expert ids and no error anywhere.
    #[test]
    fn a_strided_i32_view_reads_its_own_indices() {
        let ctx = Context::new(ARENA).expect("context");

        // 3 rows of 4: a full "ranking" per row.
        let t = ctx.new_i32_2d(4, 3).expect("t");
        t.set_i32(&(0..12).collect::<Vec<i32>>()).expect("set");

        // The top 2 of each row, as argsort_top_k would return them.
        let i32_size = std::mem::size_of::<i32>();
        let top2 = ctx.view_2d(&t, 2, 3, 4 * i32_size, 0).expect("view");

        assert!(!top2.is_contiguous());
        assert_eq!(top2.to_vec_i32(), vec![0, 1, 4, 5, 8, 9]);
        // What a flat read gave instead: row 0 spilling into every other row.
        assert_ne!(top2.to_vec_i32(), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn computes_a_matmul_with_a_known_answer() {
        // The point is not that ggml can multiply -- it is that our graph
        // building, memory layout and execution plumbing are correct. A
        // hand-checkable result is the only way to know.
        let ctx = Context::new(ARENA).expect("context");

        // ggml stores column-major: ne0 is the fastest-moving dimension.
        // a is 2x2 = [[1,2],[3,4]] laid out row by row.
        let a = ctx.new_f32_2d(2, 2).expect("a");
        a.set_f32(&[1.0, 2.0, 3.0, 4.0]).expect("set a");
        let b = ctx.new_f32_2d(2, 1).expect("b");
        b.set_f32(&[1.0, 1.0]).expect("set b");

        let c = ctx.mul_mat(&a, &b).expect("mul_mat");
        ctx.compute(&c, 1).expect("compute");

        // Each output row is the dot product of a row of `a` with `b`.
        assert_eq!(c.to_vec_f32(), vec![3.0, 7.0]);
    }

    #[test]
    fn elementwise_ops_compose() {
        let ctx = Context::new(ARENA).expect("context");
        let x = ctx.new_f32_1d(4).expect("x");
        x.set_f32(&[1.0, 2.0, 3.0, 4.0]).expect("set");
        let y = ctx.new_f32_1d(4).expect("y");
        y.set_f32(&[10.0, 20.0, 30.0, 40.0]).expect("set");

        let sum = ctx.add(&x, &y).expect("add");
        let scaled = ctx.mul(&sum, &y).expect("mul");
        ctx.compute(&scaled, 2).expect("compute");

        assert_eq!(scaled.to_vec_f32(), vec![110.0, 440.0, 990.0, 1760.0]);
    }

    #[test]
    fn softmax_produces_a_distribution() {
        let ctx = Context::new(ARENA).expect("context");
        let x = ctx.new_f32_1d(4).expect("x");
        x.set_f32(&[1.0, 2.0, 3.0, 4.0]).expect("set");
        let p = ctx.soft_max(&x).expect("softmax");
        ctx.compute(&p, 1).expect("compute");

        let out = p.to_vec_f32();
        let total: f32 = out.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax summed to {total}");
        // Monotonic input must give monotonic probabilities.
        for pair in out.windows(2) {
            assert!(pair[1] > pair[0]);
        }
    }

    #[test]
    fn rms_norm_normalises() {
        let ctx = Context::new(ARENA).expect("context");
        let x = ctx.new_f32_1d(4).expect("x");
        x.set_f32(&[3.0, 3.0, 3.0, 3.0]).expect("set");
        let n = ctx.rms_norm(&x, 1e-6).expect("rms_norm");
        ctx.compute(&n, 1).expect("compute");
        // A constant vector normalises to all ones.
        for v in n.to_vec_f32() {
            assert!((v - 1.0).abs() < 1e-4, "got {v}");
        }
    }

    #[test]
    fn a_length_mismatch_is_refused_not_written_past() {
        let ctx = Context::new(ARENA).expect("context");
        let x = ctx.new_f32_1d(4).expect("x");
        assert!(matches!(
            x.set_f32(&[1.0, 2.0]),
            Err(GgmlError::WrongSize { .. })
        ));
    }

    #[test]
    fn get_rows_gathers_embeddings() {
        // The first op of any forward pass: turn token ids into vectors.
        let ctx = Context::new(ARENA).expect("context");
        // 4 rows of width 2: [[0,1],[2,3],[4,5],[6,7]]
        let table = ctx.new_f32_2d(2, 4).expect("table");
        table
            .set_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .expect("set");

        // Row indices must be I32, not f32.
        let ids = ctx.new_i32_1d(2).expect("ids");
        ids.set_i32(&[2, 0]).expect("set ids");

        let rows = ctx.get_rows(&table, &ids).expect("get_rows");
        ctx.compute(&rows, 1).expect("compute");
        assert_eq!(rows.to_vec_f32(), vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn top_k_selects_the_right_experts_but_not_in_score_order() {
        // MoE routing: pick the k highest-scoring experts.
        //
        // IMPORTANT, and the reason this test spells it out: ggml's top_k does
        // NOT return indices sorted by descending score. Measured here it
        // returns [3, 1] for scores where index 1 is the highest -- the *set*
        // is right, the order is not what the name suggests. Routing code must
        // therefore look each expert's weight up by index rather than assuming
        // position 0 is the best match. Getting this wrong would silently
        // weight the wrong experts and produce plausible-looking garbage.
        let ctx = Context::new(ARENA).expect("context");
        let scores = ctx.new_f32_1d(6).expect("scores");
        scores
            .set_f32(&[0.1, 0.9, 0.3, 0.7, 0.2, 0.5])
            .expect("set");
        let top = ctx.top_k(&scores, 2).expect("top_k");
        ctx.compute(&top, 1).expect("compute");

        let mut idx = top.to_vec_i32();
        assert_eq!(idx.len(), 2);
        idx.sort_unstable();
        // 0.9 is index 1 and 0.7 is index 3 -- those two, in some order.
        assert_eq!(idx, vec![1, 3], "top_k selected the wrong experts");
    }

    #[test]
    fn concat_joins_along_a_dimension() {
        let ctx = Context::new(ARENA).expect("context");
        let a = ctx.new_f32_1d(2).expect("a");
        a.set_f32(&[1.0, 2.0]).expect("set");
        let b = ctx.new_f32_1d(3).expect("b");
        b.set_f32(&[3.0, 4.0, 5.0]).expect("set");
        let c = ctx.concat(&a, &b, 0).expect("concat");
        ctx.compute(&c, 1).expect("compute");
        assert_eq!(c.to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn transpose_then_cont_materialises_the_new_layout() {
        // Views only reinterpret; several ops need real contiguous memory.
        let ctx = Context::new(ARENA).expect("context");
        // 3 wide, 2 tall: rows [1,2,3] and [4,5,6].
        let m = ctx.new_f32_2d(3, 2).expect("m");
        m.set_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("set");

        let t = ctx.transpose(&m).expect("transpose");
        let c = ctx.cont(&t).expect("cont");
        ctx.compute(&c, 1).expect("compute");
        // Transposed: 2 wide, 3 tall -> [1,4],[2,5],[3,6].
        assert_eq!(c.to_vec_f32(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn sum_rows_reduces_each_row() {
        let ctx = Context::new(ARENA).expect("context");
        let m = ctx.new_f32_2d(3, 2).expect("m");
        m.set_f32(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).expect("set");
        let s = ctx.sum_rows(&m).expect("sum_rows");
        ctx.compute(&s, 1).expect("compute");
        assert_eq!(s.to_vec_f32(), vec![6.0, 60.0]);
    }

    /// **The delta net's ops, exercised once each before anything is built on
    /// them.** A wrong FFI declaration here does not fail to compile -- it
    /// mis-reads arguments and produces confident numbers, which is the failure
    /// mode this whole project is organised against. So each new binding gets
    /// one case with an answer worked out by hand.
    #[test]
    fn l2_norm_divides_by_the_norm_not_the_root_mean_square() {
        let ctx = Context::new(ARENA).expect("context");
        // A row of four 2s: L2 norm is 4, so every element becomes 0.5.
        // `rms_norm` would divide by 2 and leave 1.0 -- which is the mistake
        // this test exists to catch.
        let x = ctx.new_f32_2d(4, 1).expect("x");
        x.set_f32(&[2.0, 2.0, 2.0, 2.0]).expect("set");
        let n = ctx.l2_norm(&x, 1e-6).expect("l2_norm");
        ctx.compute(&n, 1).expect("compute");
        let out = n.to_vec_f32();
        for (i, v) in out.iter().enumerate() {
            assert!((v - 0.5).abs() < 1e-5, "element {i} is {v}, wanted 0.5");
        }
    }

    /// A depthwise causal convolution, one channel, kernel `[1, 1, 1, 1]`.
    ///
    /// Input is the rolling window: three stored values then two new tokens, so
    /// the two outputs are the sums of the trailing four values at each step.
    #[test]
    fn ssm_conv_sums_its_window_per_channel() {
        let ctx = Context::new(ARENA).expect("context");
        let window = 4;
        let n_tokens = 2;
        // [window - 1 + n_tokens, channels, seqs] = [5, 1, 1]
        let a = ctx.new_f32_3d(window - 1 + n_tokens, 1, 1).expect("a");
        a.set_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("set a");
        let k = ctx.new_f32_2d(window, 1).expect("k");
        k.set_f32(&[1.0, 1.0, 1.0, 1.0]).expect("set k");
        let c = ctx.ssm_conv(&a, &k).expect("ssm_conv");
        assert_eq!(c.dims_and_strides().0[0], 1, "channels on the fast axis");
        assert_eq!(c.dims_and_strides().0[1], n_tokens);
        ctx.compute(&c, 1).expect("compute");
        let out = c.to_vec_f32();
        assert!((out[0] - 10.0).abs() < 1e-5, "1+2+3+4 = {}", out[0]);
        assert!((out[1] - 14.0).abs() < 1e-5, "2+3+4+5 = {}", out[1]);
    }

    /// The fused delta rule, at the shapes Qwen3.5 uses but one head wide.
    ///
    /// Checks the contract rather than the arithmetic: the result must pack the
    /// scores and then the state snapshot, so its element count is
    /// `S*H*T*N + S*S*H*N`. Getting `snapshots` wrong, or the argument order,
    /// makes ggml **abort the whole test binary** -- so this test passing at all
    /// is most of what it is for.
    #[test]
    fn gated_delta_net_packs_scores_then_the_state() {
        let ctx = Context::new(ARENA).expect("context");
        let (s, h, tokens, seqs) = (4i64, 1i64, 3i64, 1i64);
        let mk4 = |ne: [i64; 4], fill: f32| {
            let n = (ne[0] * ne[1] * ne[2] * ne[3]) as usize;
            let t = ctx.new_f32_1d(n as i64).expect("new");
            t.set_f32(&vec![fill; n]).expect("set");
            ctx.reshape_4d(&t, ne).expect("reshape")
        };
        let q = mk4([s, h, tokens, seqs], 0.1);
        let k = mk4([s, h, tokens, seqs], 0.2);
        let v = mk4([s, h, tokens, seqs], 0.3);
        let g = mk4([1, h, tokens, seqs], -0.5);
        let beta = mk4([1, h, tokens, seqs], 0.4);
        let state = mk4([s, s, h, seqs], 0.0);

        let out = ctx
            .gated_delta_net(&q, &k, &v, &g, &beta, &state, 1)
            .expect("gated_delta_net");
        ctx.compute(&out, 1).expect("compute");
        let got = out.to_vec_f32();
        let want = (s * h * tokens * seqs + s * s * h * seqs) as usize;
        assert_eq!(got.len(), want, "scores then one state snapshot");
        assert!(
            got.iter().all(|v| v.is_finite()),
            "the delta rule produced a non-finite value"
        );
        // With a zero initial state and a positive beta the first token must
        // move the state off zero, or the op is not accumulating anything.
        let scores = (s * h * tokens * seqs) as usize;
        assert!(
            got[scores..].iter().any(|v| v.abs() > 1e-9),
            "the carried state came back all zero"
        );
    }

    /// **Does the fused op broadcast fewer key heads over more value heads?**
    ///
    /// Qwen3.5-0.8B has `group_count 16` key heads and `time_step_rank 16`
    /// value heads -- a 1:1 ratio, where a broadcast is a no-op and its absence
    /// cannot be seen. Qwen3.6-27B has 16 key heads and **48** value heads, and
    /// on that container the port generated fluent nonsense while the 0.8B was
    /// byte-identical to llama.cpp. This test asks the op directly, on a 2:6
    /// ratio, instead of trusting the comment that said it broadcasts.
    ///
    /// If the two calls agree, the op broadcasts and the caller must not.
    /// If they differ, the caller must repeat q and k up to the value head
    /// count before calling -- which is what llama.cpp's `repeat_4d` does.
    #[test]
    fn gated_delta_net_and_the_key_head_broadcast() {
        let ctx = Context::new(ARENA).expect("context");
        let (s, h_k, h_v, tokens, seqs) = (4i64, 2i64, 6i64, 3i64, 1i64);
        let ramp = |ne: [i64; 4], step: f32| {
            let n = (ne[0] * ne[1] * ne[2] * ne[3]) as usize;
            let t = ctx.new_f32_1d(n as i64).expect("new");
            // A ramp rather than a constant: with every head identical, a
            // missing broadcast and a correct one give the same answer, and the
            // test would pass while proving nothing.
            let data: Vec<f32> = (0..n).map(|i| 0.05 + step * i as f32).collect();
            t.set_f32(&data).expect("set");
            ctx.reshape_4d(&t, ne).expect("reshape")
        };
        let v = ramp([s, h_v, tokens, seqs], 0.011);
        let g = ramp([1, h_v, tokens, seqs], -0.03);
        let beta = ramp([1, h_v, tokens, seqs], 0.017);
        let state = ramp([s, s, h_v, seqs], 0.0);

        // Narrow q and k: h_k heads, as the container stores them.
        let q_narrow = ramp([s, h_k, tokens, seqs], 0.013);
        let k_narrow = ramp([s, h_k, tokens, seqs], 0.019);
        let narrow = ctx
            .gated_delta_net(&q_narrow, &k_narrow, &v, &g, &beta, &state, 1)
            .expect("narrow");
        ctx.compute(&narrow, 1).expect("compute narrow");
        let narrow = narrow.to_vec_f32();

        // The same q and k, repeated up to h_v heads by hand.
        let q_wide = ctx
            .repeat_4d(&q_narrow, [s, h_v, tokens, seqs])
            .expect("repeat q");
        let k_wide = ctx
            .repeat_4d(&k_narrow, [s, h_v, tokens, seqs])
            .expect("repeat k");
        let widened = ctx
            .gated_delta_net(&q_wide, &k_wide, &v, &g, &beta, &state, 1)
            .expect("widened");
        ctx.compute(&widened, 1).expect("compute widened");
        let widened = widened.to_vec_f32();

        assert_eq!(narrow.len(), widened.len(), "same output shape either way");
        assert!(
            narrow.iter().chain(widened.iter()).all(|v| v.is_finite()),
            "a non-finite value from the delta rule"
        );
        let same = narrow
            .iter()
            .zip(widened.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);
        assert!(
            same,
            "**the fused op does NOT broadcast key heads.** With {h_k} key \
             heads and {h_v} value heads, calling it with narrow q and k gives \
             a different answer than repeating them first, so `qwen35.rs` must \
             repeat q and k up to the value head count. This is why \
             Qwen3.6-27B (16 key, 48 value) generated nonsense while \
             Qwen3.5-0.8B (16 and 16) was exact.\nnarrow  {:?}\nwidened {:?}",
            &narrow[..8.min(narrow.len())],
            &widened[..8.min(widened.len())]
        );
    }

    /// The two ops a VAE decoder needs, against arithmetic done by hand.
    ///
    /// **Tried in `examples/try-vae-ops.rs` first.** ggml aborts rather than
    /// returning errors, and an abort takes the whole test binary rather than one
    /// test — so a new op is exercised where a crash costs one `cargo run` and
    /// names the call that did it. Both of these survived that before arriving
    /// here.
    #[test]
    fn conv_2d_and_group_norm_do_the_arithmetic() {
        let ctx = Context::new(ARENA).expect("context");

        // A 1x1 kernel of 2 over [1,2,3,4] doubles every element.
        let data = ctx.new_f32_1d(4).expect("data");
        data.set_f32(&[1.0, 2.0, 3.0, 4.0]).expect("set");
        let data = ctx.reshape_4d(&data, [2, 2, 1, 1]).expect("reshape");
        let kernel = ctx.new_f16_4d(1, 1, 1, 1).expect("kernel");
        // **F16, and 0x4000 is 2.0.** An F32 kernel aborts: ggml's conv_2d goes
        // through im2col and a matmul written for half precision.
        kernel
            .set_bytes(&0x4000u16.to_le_bytes())
            .expect("set kernel");
        let out = ctx
            .conv_2d(&kernel, &data, (1, 1), (0, 0), (1, 1))
            .expect("conv_2d");
        ctx.compute(&out, 1).expect("compute");
        let got = out.to_vec_f32();
        assert_eq!(got.len(), 4, "1x1 stride 1 no padding keeps the shape");
        for (g, w) in got.iter().zip([2.0f32, 4.0, 6.0, 8.0]) {
            assert!((g - w).abs() < 1e-3, "conv_2d gave {got:?}");
        }

        // Group norm over one group of [1,2,3,4]: mean 2.5, population variance
        // 1.25, so (x - 2.5) / sqrt(1.25). **Population, not sample** -- dividing
        // by n-1 would give 1.1547 instead of 1.3416, and that is the kind of
        // difference a "looks about right" check misses.
        let x = ctx.new_f32_1d(4).expect("x");
        x.set_f32(&[1.0, 2.0, 3.0, 4.0]).expect("set");
        let x = ctx.reshape_4d(&x, [4, 1, 1, 1]).expect("reshape");
        let out = ctx.group_norm(&x, 1, 1e-5).expect("group_norm");
        ctx.compute(&out, 1).expect("compute");
        let got = out.to_vec_f32();
        for (g, w) in got.iter().zip([-1.3416f32, -0.4472, 0.4472, 1.3416]) {
            assert!((g - w).abs() < 2e-3, "group_norm gave {got:?}");
        }
        // Mean zero is the property, and it is worth stating separately: a
        // normalisation that scaled correctly but shifted wrongly would pass the
        // comparison above only by luck.
        let mean: f32 = got.iter().sum::<f32>() / got.len() as f32;
        assert!(mean.abs() < 1e-5, "group norm must centre: mean {mean}");
    }

    #[test]
    fn scale_and_sigmoid_behave() {
        let ctx = Context::new(ARENA).expect("context");
        let x = ctx.new_f32_1d(3).expect("x");
        x.set_f32(&[-1.0, 0.0, 1.0]).expect("set");
        let s = ctx.sigmoid(&x).expect("sigmoid");
        ctx.compute(&s, 1).expect("compute");
        let out = s.to_vec_f32();
        assert!((out[1] - 0.5).abs() < 1e-6, "sigmoid(0) = {}", out[1]);
        assert!(out[0] < out[1] && out[1] < out[2], "must be monotonic");
    }

    #[test]
    fn soft_max_ext_applies_the_scale_before_normalising() {
        // Attention needs scale-then-softmax as one op. A larger scale
        // sharpens the distribution; verifying that catches the case where
        // the scale is silently ignored.
        let ctx = Context::new(ARENA).expect("context");
        let x = ctx.new_f32_1d(3).expect("x");
        x.set_f32(&[1.0, 2.0, 3.0]).expect("set");

        let soft = ctx.soft_max_ext(&x, None, 1.0, 0.0).expect("softmax");
        ctx.compute(&soft, 1).expect("compute");
        let flat = soft.to_vec_f32();

        let ctx2 = Context::new(ARENA).expect("context2");
        let y = ctx2.new_f32_1d(3).expect("y");
        y.set_f32(&[1.0, 2.0, 3.0]).expect("set");
        let sharp = ctx2.soft_max_ext(&y, None, 4.0, 0.0).expect("softmax");
        ctx2.compute(&sharp, 1).expect("compute");
        let scaled = sharp.to_vec_f32();

        assert!((flat.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!((scaled.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(
            scaled[2] > flat[2],
            "a larger scale must concentrate mass on the max: {scaled:?} vs {flat:?}"
        );
    }

    #[test]
    fn mul_mat_id_selects_per_row_experts() {
        // The op MoE depends on: two stacked 2x2 "experts", each row of the
        // input routed to a different one.
        let ctx = Context::new(ARENA).expect("context");

        // experts[0] = identity, experts[1] = 2 * identity
        let experts = ctx.new_f32_3d(2, 2, 2).expect("experts");
        experts
            .set_f32(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0])
            .expect("set experts");

        // One input vector, routed to expert 1 (the doubling one).
        let b = ctx.new_f32_3d(2, 1, 1).expect("b");
        b.set_f32(&[3.0, 4.0]).expect("set b");

        let ids = ctx.new_i32_2d(1, 1).expect("ids");
        ids.set_i32(&[1]).expect("set ids");

        let out = ctx.mul_mat_id(&experts, &b, &ids).expect("mul_mat_id");
        ctx.compute(&out, 1).expect("compute");
        assert_eq!(out.to_vec_f32(), vec![6.0, 8.0], "wrong expert applied");
    }

    #[test]
    fn arena_sizing_helper_covers_what_a_graph_needs() {
        // NOT a test that exhaustion is survivable -- it is not. Verified by
        // running it: ggml prints "not enough space in the context's memory
        // pool" and then GGML_ASSERT aborts the process. There is no NULL to
        // check and no unwinding to catch, so `ArenaExhausted` can never
        // actually be observed for tensor allocation.
        //
        // The only defence is to size the arena correctly up front, which is
        // what `arena_for` exists to do.
        let need = arena_for(&[(2, 2), (2, 1), (2, 1)], 8);
        let ctx = Context::new(need).expect("context");
        let a = ctx.new_f32_2d(2, 2).expect("a fits");
        a.set_f32(&[1.0, 2.0, 3.0, 4.0]).expect("set");
        let b = ctx.new_f32_2d(2, 1).expect("b fits");
        b.set_f32(&[1.0, 1.0]).expect("set");
        let c = ctx.mul_mat(&a, &b).expect("result fits");
        ctx.compute(&c, 1).expect("compute");
        assert_eq!(c.to_vec_f32(), vec![3.0, 7.0]);
    }
}
