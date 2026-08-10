//! Rearranging quantised weights into the layout the CPU kernels want.
//!
//! # The measurement this exists for
//!
//! On Qwen3-4B prefill, same file and same prompt, `llama-completion`:
//!
//! ```text
//! repacking on (its default)   88.26 tok/s
//! --no-repack                  63.68 tok/s
//! Bigtea                       60.29 tok/s
//! ```
//!
//! **Repacking is worth 1.39x, and without it the two engines are 6% apart.**
//! Both link the same `ggml`, so that is the expected result — and it means the
//! whole remaining prefill gap is this one thing.
//!
//! # What it actually does
//!
//! A `Q4_K` tensor is stored as independent blocks, each with its own scales.
//! The kernels want several blocks *interleaved* so one SIMD register can hold
//! matching lanes from eight rows at once. `ggml` ships that rearrangement and
//! the kernels that consume it; what it does not ship is a way to reach them
//! from the raw graph API.
//!
//! llama.cpp gets there through `ggml-backend`'s **extra buffer types**: a
//! tensor allocated in the repack buffer type is rearranged by its `set_tensor`,
//! which also hangs the matching `tensor_traits` off `tensor->extra`. Then
//! `ggml_compute_forward` calls `ggml_cpu_extra_compute_forward` first, sees the
//! traits, and dispatches to the repacked kernel. **That last part happens on
//! the plain graph path too**, which is why this works at all without adopting
//! `ggml-backend` wholesale.
//!
//! # Why this is not on by default everywhere
//!
//! Bigtea binds weights **zero-copy** — `ggml` is handed a pointer into the
//! mapped container — and that is what makes a 144 GB model run on a 15.7 GiB
//! machine. Repacking needs its own buffer, so it *doubles* the memory for
//! whatever it touches.
//!
//! For a **resident dense model** that costs nothing extra: the weights were
//! already copied into RAM, so the repacked copy replaces them rather than
//! adding to them. For the streaming path it would be fatal. So this is offered
//! per tensor and the caller decides.

use std::ffi::c_void;
// Every use of `NonNull` is behind `cfg(have_ggml)` — the `raw` field and the
// allocation that fills it — so an unconditional import is unused in the build
// that has no ggml, and `-D warnings` turns that into a build failure. That is
// the one CI job whose whole purpose is proving the other seven crates are
// usable on a machine that has never compiled a line of C.
#[cfg(have_ggml)]
use std::ptr::NonNull;

use bigtea_gguf::GgmlType;

use crate::GgmlError;

#[cfg(have_ggml)]
mod ffi {
    use std::ffi::{c_char, c_void};

    // Opaque to us: `ggml` never lets a caller see inside these, so an empty
    // `repr(C)` struct would be an FFI-safety lint for a type we only ever hold
    // a pointer to. `c_void` says exactly that.
    pub type BufferType = c_void;
    pub type Buffer = c_void;

    unsafe extern "C" {
        /// C++ linkage: `ggml`'s repack buffer type is defined in `repack.cpp`,
        /// so the symbol is mangled and has to be named explicitly.
        #[link_name = "_Z35ggml_backend_cpu_repack_buffer_typev"]
        pub fn ggml_backend_cpu_repack_buffer_type() -> *mut BufferType;

        pub fn ggml_backend_buft_name(buft: *mut BufferType) -> *const c_char;
        pub fn ggml_backend_buft_alloc_buffer(buft: *mut BufferType, size: usize) -> *mut Buffer;
        pub fn ggml_backend_buft_get_alloc_size(
            buft: *mut BufferType,
            tensor: *mut c_void,
        ) -> usize;
        pub fn ggml_backend_tensor_alloc(
            buffer: *mut Buffer,
            tensor: *mut c_void,
            addr: *mut c_void,
        ) -> i32;
        pub fn ggml_backend_buffer_get_base(buffer: *mut Buffer) -> *mut c_void;
        pub fn ggml_backend_tensor_set(
            tensor: *mut c_void,
            data: *const c_void,
            offset: usize,
            size: usize,
        );
        pub fn ggml_backend_buffer_free(buffer: *mut Buffer);
    }
}

/// A `ggml` buffer holding repacked weights.
///
/// Owns the allocation. Dropping it frees the weights, so it must outlive every
/// graph that reads them — which is why callers hold it beside the `WeightSet`.
pub struct RepackBuffer {
    #[cfg(have_ggml)]
    raw: NonNull<ffi::Buffer>,
    bytes: usize,
}

// SAFETY: the buffer is only read during graph evaluation, which `ggml`
// already parallelises internally, and it is never mutated after `set`.
unsafe impl Send for RepackBuffer {}
unsafe impl Sync for RepackBuffer {}

impl RepackBuffer {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// A tensor **already** rearranged, ready to be bound into any number of
/// contexts without paying the rearrangement again.
///
/// # Why this exists separately from [`repack`]
///
/// [`repack`] rearranges and binds in one step, which suits a `WeightSet` that
/// is built once and lives for the session — the dense path's
/// `load_resident`. **V4-Flash does not work that way.** Its arena is per
/// block, so it builds a fresh context and a fresh `WeightSet` for every one of
/// its 43 blocks, on every pass. Repacking there would rearrange the whole
/// 7.38 GiB always-read set 43 times *per token*, which is not a smaller win —
/// it is a large loss.
///
/// So the rearrangement is hoisted to load time and the result kept. Binding it
/// afterwards is `ggml_backend_tensor_alloc`, which points a fresh tensor at
/// the existing bytes and re-runs the buffer's `init_tensor` — the step that
/// hangs the repacked `tensor_traits` off `extra`. **`ggml_backend_tensor_set`
/// is not called again**, so no bytes move.
///
/// Held behind an `Arc` by whoever loaded the weights, because the tensors
/// pointing into it are created and dropped once per block.
pub struct Repacked {
    buffer: RepackBuffer,
    ty: GgmlType,
    ne0: i64,
    ne1: i64,
}

impl Repacked {
    /// Rearrange `data` once. `Ok(None)` means `ggml` has no repacked kernel
    /// for this tensor and the caller should bind it normally — a routine
    /// outcome, not a failure.
    pub fn new(ty: GgmlType, ne0: i64, ne1: i64, data: &[u8]) -> Result<Option<Self>, GgmlError> {
        #[cfg(not(have_ggml))]
        {
            let _ = (ty, ne0, ne1, data);
            Err(GgmlError::Unavailable)
        }
        #[cfg(have_ggml)]
        {
            // A scratch context for the one tensor the rearrangement needs as
            // its destination shape. It is dropped on the way out; the buffer
            // outlives it, and freeing a context does not touch a buffer that
            // ggml allocated separately.
            let scratch = crate::graph::Context::new_no_alloc(1 << 16)?;
            let tensor = scratch.new_typed_2d(ty, ne0, ne1)?;
            let expected = tensor.bytes();
            if data.len() != expected {
                return Err(GgmlError::WrongSize {
                    expected,
                    actual: data.len(),
                });
            }
            // SAFETY: `tensor` is live in `scratch`, created no_alloc so its
            // data pointer is null and nothing is orphaned; `data` is exactly
            // the tensor's size, checked above.
            let buffer = unsafe { repack(tensor.as_ptr(), data) }?;
            Ok(buffer.map(|buffer| Repacked {
                buffer,
                ty,
                ne0,
                ne1,
            }))
        }
    }

    pub fn ty(&self) -> GgmlType {
        self.ty
    }

    pub fn shape(&self) -> (i64, i64) {
        (self.ne0, self.ne1)
    }

    pub fn bytes(&self) -> usize {
        self.buffer.bytes()
    }

    /// Point a fresh tensor at the already-rearranged bytes.
    ///
    /// # Safety
    ///
    /// `tensor` must be a live `ggml_tensor` of exactly this `Repacked`'s type
    /// and shape, from a `no_alloc` context, whose data pointer is not yet set.
    /// It must not outlive `self` — dropping the buffer frees the weights the
    /// graph reads.
    pub unsafe fn attach(&self, tensor: *mut c_void) -> Result<(), GgmlError> {
        #[cfg(not(have_ggml))]
        {
            let _ = tensor;
            Err(GgmlError::Unavailable)
        }
        #[cfg(have_ggml)]
        // SAFETY: the caller guarantees `tensor`; `raw` is a live buffer this
        // value owns, and its base is where the rearranged bytes were written.
        unsafe {
            let buffer = self.buffer.raw.as_ptr();
            let base = ffi::ggml_backend_buffer_get_base(buffer);
            if base.is_null() {
                return Err(GgmlError::Unavailable);
            }
            // Sets `tensor->data` and runs `init_tensor`, which re-attaches the
            // repacked `tensor_traits`. Without that step the graph would read
            // rearranged bytes with the ordinary kernel — not an error, and
            // entirely wrong.
            if ffi::ggml_backend_tensor_alloc(buffer, tensor, base) != 0 {
                return Err(GgmlError::Unavailable);
            }
            Ok(())
        }
    }
}

impl Drop for RepackBuffer {
    fn drop(&mut self) {
        #[cfg(have_ggml)]
        // SAFETY: `raw` came from `ggml_backend_buft_alloc_buffer` and is freed
        // exactly once, here.
        unsafe {
            ffi::ggml_backend_buffer_free(self.raw.as_ptr())
        };
    }
}

/// Whether repacking is worth attempting for a tensor of this type and shape.
///
/// **A pre-filter, not the decision.** Whether a repacked kernel exists depends
/// on the *CPU* as well as the tensor — on x86 `Q8_0` has none at all and
/// `Q2_K` needs AVX-512 — and mirroring ggml's feature matrix here would be a
/// copy that drifts. [`repack`] asks ggml and reads its answer; this only
/// avoids allocating a buffer to be told no in the obvious cases.
pub fn is_repackable(ty: GgmlType, ne0: i64, ne1: i64) -> bool {
    // Q4_0, Q4_K, Q2_K, IQ4_NL and Q8_0 are the types ggml ships repacked
    // kernels for on *some* target. The interleave is 4 or 8 rows wide, so the
    // output dimension must divide by 8, and the row length by the block size.
    const Q4_0: u32 = 2;
    const Q8_0: u32 = 8;
    const Q2_K: u32 = 10;
    const Q4_K: u32 = 12;
    const IQ4_NL: u32 = 20;
    matches!(ty.0, Q4_0 | Q8_0 | Q2_K | Q4_K | IQ4_NL) && ne1 % 8 == 0 && ne0 % 32 == 0
}

/// The name `ggml` gives its repack buffer type, for reporting.
pub fn buffer_type_name() -> Option<String> {
    #[cfg(not(have_ggml))]
    return None;
    #[cfg(have_ggml)]
    {
        // SAFETY: the buffer type is a static in ggml-cpu; the name is a
        // 'static C string.
        unsafe {
            let buft = ffi::ggml_backend_cpu_repack_buffer_type();
            if buft.is_null() {
                return None;
            }
            let name = ffi::ggml_backend_buft_name(buft);
            if name.is_null() {
                return None;
            }
            Some(
                std::ffi::CStr::from_ptr(name)
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }
}

/// Repack `data` into a `ggml` buffer and point `tensor` at it.
///
/// Returns the buffer, which **must outlive every graph that reads the
/// tensor** — dropping it frees the weights.
///
/// `Ok(None)` means `ggml` has no repacked kernel for this tensor and the
/// caller should bind it normally. That is a routine outcome, not a failure:
/// the type/shape check is a heuristic and `ggml` has the final say.
///
/// # Safety
///
/// `tensor` must be a live `ggml_tensor` from a `no_alloc` context whose data
/// pointer is not yet set, and `data` must hold exactly the tensor's bytes in
/// its stored quantisation.
pub unsafe fn repack(tensor: *mut c_void, data: &[u8]) -> Result<Option<RepackBuffer>, GgmlError> {
    #[cfg(not(have_ggml))]
    {
        let _ = (tensor, data);
        Err(GgmlError::Unavailable)
    }
    #[cfg(have_ggml)]
    // SAFETY: the caller guarantees `tensor` and `data`; every ggml call below
    // is checked for null and the buffer is owned by the returned value.
    unsafe {
        let buft = ffi::ggml_backend_cpu_repack_buffer_type();
        if buft.is_null() {
            return Ok(None);
        }
        // ggml decides the real size — a repacked tensor is not necessarily the
        // same number of bytes as the original.
        let size = ffi::ggml_backend_buft_get_alloc_size(buft, tensor);
        if size == 0 {
            return Ok(None);
        }
        let buffer = ffi::ggml_backend_buft_alloc_buffer(buft, size);
        let Some(raw) = NonNull::new(buffer) else {
            return Ok(None);
        };
        let owned = RepackBuffer { raw, bytes: size };

        let base = ffi::ggml_backend_buffer_get_base(buffer);
        if base.is_null() {
            return Ok(None);
        }
        // Points the tensor at the buffer and runs the buffer's `init_tensor`,
        // which is what attaches the repacked `tensor_traits` to `extra`.
        // Without that the graph would read the rearranged bytes with the
        // ordinary kernel, which is not an error and is entirely wrong.
        if ffi::ggml_backend_tensor_alloc(buffer, tensor, base) != 0 {
            return Ok(None);
        }
        // **ggml does not decline; it crashes.** `init_tensor`, which the call
        // above just ran, sets `extra` to whatever
        // `ggml_repack_get_optimal_repack_type` returns — and that is `nullptr`
        // whenever this CPU has no repacked kernel for the type and shape. It
        // then returns `GGML_STATUS_SUCCESS` anyway. `set_tensor` immediately
        // does `((tensor_traits_base *) tensor->extra)->repack(...)`, so a null
        // there is a null dereference: `STATUS_ACCESS_VIOLATION`, no assert, no
        // error code, the whole process gone.
        //
        // [`is_repackable`] cannot prevent this on its own, because the answer
        // depends on the **CPU**, not just the tensor: on x86 `Q8_0` has no
        // repacked kernel at all (the ones that exist are NEON and RISC-V) and
        // `Q2_K` needs AVX-512. V4-Flash's Q4_K_XL mixes `Q8_0` tensors in, and
        // offering the first one killed the test binary.
        //
        // So the shape check stays a cheap pre-filter and **this** is the
        // authority: ask ggml, then look at what it actually decided.
        if (*(tensor as *mut crate::weights::RawTensor))
            .extra
            .is_null()
        {
            // `owned` is dropped here, freeing the buffer. The tensor's data
            // pointer is left dangling, which is why every caller must treat
            // `None` as "bind it again, normally" rather than reusing it.
            return Ok(None);
        }
        // The rearrangement itself happens here, inside the buffer's
        // `set_tensor`.
        ffi::ggml_backend_tensor_set(tensor, data.as_ptr() as *const c_void, 0, data.len());
        Ok(Some(owned))
    }
}

#[cfg(all(test, have_ggml))]
mod ggml_tests {
    use super::*;
    use crate::graph::Context;
    use crate::weights::WeightSet;
    use std::sync::Arc;

    const Q4_K: GgmlType = GgmlType(12);
    const Q8_0: GgmlType = GgmlType(8);
    /// 256 quantised values per `Q4_K` superblock, in 144 bytes.
    const Q4_K_BLOCK: usize = 144;

    /// Deterministic `Q4_K` bytes with finite scales.
    ///
    /// The two `f16` fields (`d`, `dmin`) lead each superblock, and arbitrary
    /// bytes there can spell an all-ones exponent — infinity or NaN — which
    /// would make the comparison below vacuous. Everything after them is a
    /// 6-bit scale or a nibble and cannot be invalid, so only those four bytes
    /// need care.
    fn q4_k_bytes(rows: i64, cols: i64) -> Vec<u8> {
        let blocks = (rows / 256 * cols) as usize;
        let mut v = vec![0u8; blocks * Q4_K_BLOCK];
        let mut state = 0x2026_u32;
        for (i, b) in v.iter_mut().enumerate() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (state >> 16) as u8;
            // Clear the top exponent bits of `d` and `dmin` so both stay finite
            // and small.
            if i % Q4_K_BLOCK < 4 && i % 2 == 1 {
                *b &= 0x2F;
            }
        }
        v
    }

    /// **ggml's repacked kernel must agree with its ordinary one.**
    ///
    /// This is the oracle for the whole mechanism. A repacked tensor's rows are
    /// interleaved, so if the rearranged bytes were bound but dispatched to the
    /// ordinary kernel — or the traits were attached and the bytes were not —
    /// the result is numbers, not an error. Comparing the two paths on the same
    /// weights is the only thing that catches it.
    #[test]
    fn a_repacked_matmul_agrees_with_an_ordinary_one() {
        const NE0: i64 = 512;
        const NE1: i64 = 32;
        let bytes = q4_k_bytes(NE0, NE1);

        let x: Vec<f32> = (0..NE0).map(|i| (i as f32 * 0.017).sin()).collect();

        let plain = {
            let wctx = Context::new_no_alloc(1 << 20).expect("weight ctx");
            let mut w = WeightSet::new();
            w.bind(&wctx, "w", Q4_K, &[NE0 as u64, NE1 as u64], bytes.clone())
                .expect("bind");
            let ctx = Context::new(16 << 20).expect("ctx");
            let xt = ctx.new_f32_2d(NE0, 1).expect("x");
            xt.set_f32(&x).expect("set x");
            let y = ctx
                .mul_mat(w.get("w").expect("bound"), &xt)
                .expect("mul_mat");
            ctx.compute(&y, 1).expect("compute");
            y.to_vec_f32()
        };

        let Some(repacked) = Repacked::new(Q4_K, NE0, NE1, &bytes).expect("repack") else {
            // Q4_K needs AVX2 on x86 and dotprod on ARM. Without them there is
            // no second path to compare against, and that is not a failure.
            eprintln!("skipping: ggml has no repacked Q4_K kernel on this CPU");
            return;
        };
        let repacked = Arc::new(repacked);

        // Bound into **two** separate contexts from the one rearrangement,
        // because that is exactly what the V4-Flash path does: it rebuilds its
        // `WeightSet` for every block of every pass, so the bytes are attached
        // to a fresh tensor 43 times a token. A mechanism that only worked once
        // would pass a single-bind test and fail in the runner.
        for pass in 0..2 {
            let wctx = Context::new_no_alloc(1 << 20).expect("weight ctx");
            let mut w = WeightSet::new();
            w.bind_repacked_shared(&wctx, "w", repacked.clone())
                .expect("bind repacked");
            let ctx = Context::new(16 << 20).expect("ctx");
            let xt = ctx.new_f32_2d(NE0, 1).expect("x");
            xt.set_f32(&x).expect("set x");
            let y = ctx
                .mul_mat(w.get("w").expect("bound"), &xt)
                .expect("mul_mat");
            ctx.compute(&y, 1).expect("compute");
            let got = y.to_vec_f32();

            assert_eq!(got.len(), plain.len());
            for (i, (a, b)) in plain.iter().zip(got.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-3 * a.abs().max(1.0),
                    "pass {pass}, row {i}: ordinary {a} vs repacked {b}"
                );
            }
        }
    }

    /// **The crash this guard exists for.**
    ///
    /// `ggml`'s repack `init_tensor` sets `tensor->extra` to `nullptr` when the
    /// CPU has no kernel for the type — and returns `GGML_STATUS_SUCCESS`.
    /// `set_tensor` then dereferences it. On x86 `Q8_0` is exactly that case
    /// (its repacked kernels are NEON and RISC-V only), and offering one used
    /// to end the process with `STATUS_ACCESS_VIOLATION`: no assert, no error,
    /// no output.
    ///
    /// The assertion is that an **answer comes back at all**. Which answer is
    /// right depends on the machine — `Some` on an ARM box with `matmul_int8`,
    /// `None` on this x86 one — and pinning either would fail on the other half
    /// of CI.
    #[test]
    fn a_type_with_no_kernel_on_this_cpu_is_declined_rather_than_fatal() {
        const NE0: i64 = 512;
        const NE1: i64 = 32;
        // Q8_0 is 34 bytes per 32 values.
        let bytes = vec![0x11u8; (NE0 / 32 * NE1) as usize * 34];
        let answer = Repacked::new(Q8_0, NE0, NE1, &bytes);
        assert!(
            answer.is_ok(),
            "a missing kernel is not an error, it is a decline: {answer:?}",
            answer = answer.err()
        );
    }

    /// A type ggml has no repacked kernel for **anywhere**. `is_repackable`
    /// filters these out first, but it is a pre-filter and not a gate — nothing
    /// stops a caller reaching this directly, and it must not be fatal.
    #[test]
    fn an_unquantised_type_is_declined_rather_than_fatal() {
        const NE0: i64 = 512;
        const NE1: i64 = 32;
        let bytes = vec![0u8; (NE0 * NE1) as usize * 4];
        assert!(
            Repacked::new(GgmlType(0), NE0, NE1, &bytes).is_ok(),
            "F32 has nothing to repack and must decline, not crash"
        );
    }

    /// Bytes that are not the tensor's size are a caller error, and reporting
    /// it beats letting `set_tensor`'s `GGML_ASSERT(size == ggml_nbytes)` abort
    /// the process.
    #[test]
    fn the_wrong_number_of_bytes_is_an_error_not_an_abort() {
        assert!(matches!(
            Repacked::new(Q4_K, 512, 32, &[0u8; 10]),
            Err(GgmlError::WrongSize { .. })
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_types_with_repacked_kernels_are_offered() {
        // Q4_K, the quantisation nearly every model here ships.
        assert!(is_repackable(GgmlType(12), 2560, 9216));
        assert!(is_repackable(GgmlType(2), 4096, 4096)); // Q4_0
                                                         // F32 and F16 have nothing to repack.
        assert!(!is_repackable(GgmlType(0), 2560, 9216));
        assert!(!is_repackable(GgmlType(1), 2560, 9216));
    }

    #[test]
    fn shapes_that_do_not_divide_the_interleave_are_refused() {
        // The kernels interleave 8 rows, so an output dimension that is not a
        // multiple of 8 has no repacked form. Offering it anyway would allocate
        // a buffer to be told no.
        assert!(!is_repackable(GgmlType(12), 2560, 9215));
        // And the row length must hold whole quantisation blocks.
        assert!(!is_repackable(GgmlType(12), 2559, 9216));
    }
}
