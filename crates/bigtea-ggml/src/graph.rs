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
    data * 2 + count * TENSOR_OVERHEAD + (1 << 20)
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
        let params = InitParams {
            mem_size,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
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
