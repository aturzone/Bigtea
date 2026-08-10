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
/// `ggml` only has repacked kernels for some quantisations, and only when the
/// row length divides the interleave. Asking for one it cannot do is not an
/// error — [`repack`] simply reports that nothing happened — but checking first
/// avoids allocating a buffer to discover it.
pub fn is_repackable(ty: GgmlType, ne0: i64, ne1: i64) -> bool {
    // Q4_0, Q4_K, Q2_K, IQ4_NL and Q8_0 are the types ggml ships repacked
    // kernels for. The interleave is 4 or 8 rows wide, so the output dimension
    // must divide by 8, and the row length by the block size.
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
        // The rearrangement itself happens here, inside the buffer's
        // `set_tensor`.
        ffi::ggml_backend_tensor_set(tensor, data.as_ptr() as *const c_void, 0, data.len());
        Ok(Some(owned))
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
