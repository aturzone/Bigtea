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

    fn ggml_mul_mat(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor)
        -> *mut ggml_tensor;
    fn ggml_add(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor)
        -> *mut ggml_tensor;
    fn ggml_mul(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor)
        -> *mut ggml_tensor;
    fn ggml_rms_norm(ctx: *mut ggml_context, a: *mut ggml_tensor, eps: f32)
        -> *mut ggml_tensor;
    fn ggml_soft_max(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_silu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    fn ggml_new_tensor_3d(
        ctx: *mut ggml_context,
        ty: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;

    fn ggml_get_rows(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor)
        -> *mut ggml_tensor;
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
    fn ggml_scale(ctx: *mut ggml_context, a: *mut ggml_tensor, s: f32) -> *mut ggml_tensor;
    fn ggml_sigmoid(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_relu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_div(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor)
        -> *mut ggml_tensor;
    fn ggml_sum_rows(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    fn ggml_top_k(ctx: *mut ggml_context, a: *mut ggml_tensor, k: c_int) -> *mut ggml_tensor;
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

    /// Indexed matmul: picks a matrix per row from a stacked 3-D tensor.
    /// This is what makes MoE tractable — only the selected experts are
    /// multiplied, rather than all of them followed by a mask.
    fn ggml_mul_mat_id(
        ctx: *mut ggml_context,
        as_: *mut ggml_tensor,
        b: *mut ggml_tensor,
        ids: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

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
#[derive(Debug, Clone, Copy)]
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

    fn tensor<'a>(&'a self, raw: *mut ggml_tensor) -> Result<Tensor<'a>, GgmlError> {
        NonNull::new(raw)
            .map(|raw| Tensor { raw, _ctx: PhantomData })
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

    pub fn soft_max<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_soft_max(self.raw.as_ptr(), a.raw.as_ptr()) })
    }

    pub fn silu<'a>(&'a self, a: &Tensor<'a>) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_silu(self.raw.as_ptr(), a.raw.as_ptr()) })
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
    pub fn new_typed_2d(
        &self,
        ty: bigtea_gguf::GgmlType,
        ne0: i64,
        ne1: i64,
    ) -> Result<Tensor<'_>, GgmlError> {
        // SAFETY: valid context; the type id is passed through to ggml, which
        // validates it and returns null for anything it does not know.
        self.tensor(unsafe { ggml_new_tensor_2d(self.raw.as_ptr(), ty.0 as c_int, ne0, ne1) })
    }

    /// Embedding lookup: gather rows of `a` at the indices in `b`.
    pub fn get_rows<'a>(
        &'a self,
        a: &Tensor<'a>,
        b: &Tensor<'a>,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: both tensors belong to this context.
        self.tensor(unsafe { ggml_get_rows(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr()) })
    }

    pub fn concat<'a>(
        &'a self,
        a: &Tensor<'a>,
        b: &Tensor<'a>,
        dim: i32,
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe {
            ggml_concat(self.raw.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr(), dim)
        })
    }

    pub fn permute<'a>(
        &'a self,
        a: &Tensor<'a>,
        axes: [i32; 4],
    ) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe {
            ggml_permute(self.raw.as_ptr(), a.raw.as_ptr(), axes[0], axes[1], axes[2], axes[3])
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

    /// Indices of the `k` largest values per row — MoE expert selection.
    pub fn top_k<'a>(&'a self, a: &Tensor<'a>, k: i32) -> Result<Tensor<'a>, GgmlError> {
        // SAFETY: as above.
        self.tensor(unsafe { ggml_top_k(self.raw.as_ptr(), a.raw.as_ptr(), k) })
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
        let status =
            unsafe { ggml_graph_compute_with_ctx(self.raw.as_ptr(), graph, threads.max(1) as c_int) };
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

    /// Read an I32 tensor back — `top_k` returns indices, not values.
    pub fn to_vec_i32(&self) -> Vec<i32> {
        let n = self.len() as usize;
        // SAFETY: valid tensor holding `n` contiguous i32 values.
        unsafe {
            let src = ggml_get_data(self.raw.as_ptr()) as *const i32;
            std::slice::from_raw_parts(src, n).to_vec()
        }
    }

    /// This tensor's data pointer, read through the mirrored struct layout.
    ///
    /// # Safety
    /// The tensor must be live. Verified against ggml's own accessor by
    /// `weights::tests::our_struct_layout_matches_ggmls`.
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
    pub(crate) unsafe fn set_data_ptr(&self, ptr: *mut std::os::raw::c_void) {
        (*(self.raw.as_ptr() as *mut crate::weights::RawTensor)).data = ptr;
    }

    /// Read the tensor's values back as `f32`.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        let n = self.len() as usize;
        // SAFETY: valid tensor holding `n` f32 values, contiguous.
        unsafe {
            let src = ggml_get_data_f32(self.raw.as_ptr());
            std::slice::from_raw_parts(src, n).to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARENA: usize = 16 << 20;

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
        scores.set_f32(&[0.1, 0.9, 0.3, 0.7, 0.2, 0.5]).expect("set");
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
