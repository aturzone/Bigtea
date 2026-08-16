//! The arithmetic we borrow.
//!
//! Bigtea's contribution is the memory side — deciding what lives in RAM,
//! streaming the rest, and scheduling reads so a model far larger than the
//! machine still runs. The arithmetic underneath (quantized matmul kernels,
//! hand-written SIMD per instruction set) is years of specialist work that is
//! already done well in `ggml`. Rewriting it would be a multi-year detour that
//! makes the product no better.
//!
//! So this crate is deliberately thin: enough FFI to turn the quantized bytes
//! our loader produces into numbers, and no more. It is not a `ggml` wrapper
//! and does not aspire to be one.
//!
//! # Building
//!
//! Set `GGML_LIB_DIR` to a directory holding `ggml-base.a`, `ggml-cpu.a` and
//! `ggml.a`. Without it the crate still compiles — every entry point returns
//! [`GgmlError::Unavailable`] — so the workspace builds on a machine that has
//! not built `ggml`.

use std::fmt;

use bigtea_gguf::GgmlType;

#[cfg(have_ggml)]
pub mod backend;
pub mod device;
mod graph;
pub mod repack;
#[cfg(have_ggml)]
pub mod sched;
mod weights;

#[cfg(have_ggml)]
pub use backend::{
    download, download_f32, upload, upload_f32, Backend, Compute, DeviceBuffer, GraphAllocator,
};
// `device` is unconditional, unlike everything around it: it answers
// `Unavailable` rather than vanishing when ggml is absent, so a caller can ask
// "is there a GPU here?" in a build that cannot use one and get an answer
// instead of a missing symbol.
pub use device::{best_offload_device, devices, vulkan_available, DeviceInfo, DeviceKind};
#[cfg(have_ggml)]
pub use graph::{arena_for, f16_to_f32, f32_to_f16, Context, RopeParams, Tensor};
pub use repack::{is_repackable, Repacked};
#[cfg(have_ggml)]
pub use sched::{HostBuffer, Scheduler};
#[cfg(have_ggml)]
pub use weights::{Residency, UploadReport, WeightSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgmlError {
    /// The crate was built without linking ggml.
    Unavailable,
    /// ggml does not know this type, or cannot convert it to floats.
    UnsupportedType(u32),
    /// Element count is not a whole number of blocks.
    PartialBlock { elements: usize, block_size: i64 },
    /// The input buffer is the wrong size for the requested element count.
    WrongSize { expected: usize, actual: usize },
    /// ggml refused to create a context of this size.
    ContextAlloc { bytes: usize },
    /// The context's arena ran out while building the graph.
    ArenaExhausted,
    /// Graph execution returned a non-zero status.
    ComputeFailed(i32),
    /// No device at that index in the backend registry.
    NoSuchDevice(usize),
    /// The device exists but refused to produce a backend or a buffer type.
    DeviceInitFailed(usize),
    /// The device could not allocate a buffer for the context's tensors.
    ///
    /// Distinct from `ArenaExhausted`, which is host memory: this one means the
    /// *card* is full, and the answer is a smaller model or fewer resident
    /// layers rather than a bigger arena.
    DeviceOutOfMemory,
    /// Host memory offered to ggml as a buffer is not `TENSOR_ALIGNMENT`-aligned.
    ///
    /// **This one exists because ggml aborts instead of refusing.**
    /// `ggml_backend_cpu_buffer_from_ptr` asserts the pointer is 32-aligned and
    /// a `Vec<u8>` is aligned to 1, so the natural call takes the whole process
    /// down with `GGML_ASSERT ... "buffer pointer must be aligned"` — reported
    /// as "process didn't exit successfully", not as a failure anyone can
    /// catch. Checked on our side so it becomes a value.
    Misaligned { address: usize, required: usize },
}

impl fmt::Display for GgmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgmlError::Unavailable => f.write_str(
                "built without ggml: set GGML_LIB_DIR to a directory containing \
                 ggml-base.a, ggml-cpu.a and ggml.a, then rebuild",
            ),
            GgmlError::UnsupportedType(t) => {
                write!(f, "ggml cannot convert type {t} to floats")
            }
            GgmlError::PartialBlock {
                elements,
                block_size,
            } => write!(
                f,
                "{elements} elements is not a whole number of {block_size}-element blocks"
            ),
            GgmlError::WrongSize { expected, actual } => {
                write!(f, "buffer is {actual} bytes, expected {expected}")
            }
            GgmlError::ContextAlloc { bytes } => {
                write!(f, "ggml refused a context arena of {bytes} bytes")
            }
            GgmlError::ArenaExhausted => f.write_str(
                "the ggml arena ran out while building the graph; give the context more memory",
            ),
            GgmlError::ComputeFailed(s) => {
                write!(f, "ggml graph computation failed with status {s}")
            }
            GgmlError::NoSuchDevice(i) => {
                write!(f, "no compute device at index {i}")
            }
            GgmlError::DeviceInitFailed(i) => {
                write!(f, "device {i} refused to initialise a backend")
            }
            GgmlError::DeviceOutOfMemory => f.write_str(
                "the device could not allocate the requested tensors; it is out of memory, \
                 which needs a smaller model rather than a bigger arena",
            ),
            GgmlError::Misaligned { address, required } => write!(
                f,
                "host memory at {address:#x} is {} bytes off a {required}-byte boundary; \
                 ggml requires buffer pointers and tensor offsets to be {required}-aligned",
                address % required
            ),
        }
    }
}

impl std::error::Error for GgmlError {}

pub type Result<T> = std::result::Result<T, GgmlError>;

/// True when this build can actually call `ggml`.
pub const fn available() -> bool {
    cfg!(have_ggml)
}

/// `GGML_TYPE_F32`. Special-cased because ggml offers no conversion kernel
/// for it — the conversion is the identity.
// Referenced only by the ggml-backed paths, so a build without ggml sees it
// as dead. It is the type tag, not a convenience constant -- keep it.
#[cfg_attr(not(have_ggml), allow(dead_code))]
const GGML_TYPE_F32: u32 = 0;

#[cfg(have_ggml)]
mod ffi {
    use std::os::raw::{c_char, c_int, c_void};

    pub type ToFloat = unsafe extern "C" fn(*const c_void, *mut f32, i64);
    pub type FromFloatRef = unsafe extern "C" fn(*const f32, *mut c_void, i64);

    #[repr(C)]
    pub struct TypeTraits {
        pub type_name: *const c_char,
        pub blck_size: i64,
        pub blck_size_interleave: i64,
        pub type_size: usize,
        pub is_quantized: bool,
        pub to_float: Option<ToFloat>,
        pub from_float_ref: Option<FromFloatRef>,
    }

    // Declared as a set even though only `ggml_get_type_traits` is called
    // today: getting an FFI signature wrong is silent corruption, so these are
    // transcribed once, together, from one header revision rather than added
    // piecemeal later under time pressure.
    #[allow(dead_code)]
    extern "C" {
        pub fn ggml_get_type_traits(ty: c_int) -> *const TypeTraits;
        pub fn ggml_type_size(ty: c_int) -> usize;
        pub fn ggml_blck_size(ty: c_int) -> i64;
        pub fn ggml_type_name(ty: c_int) -> *const c_char;
    }
}

/// What `ggml` reports about a tensor type.
///
/// Worth cross-checking against our own table in `bigtea-gguf`: if the two
/// disagree about a block size, one of them is wrong and every byte count
/// derived from it is too.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    pub name: String,
    pub block_elems: i64,
    pub block_bytes: usize,
    pub is_quantized: bool,
    pub can_dequantize: bool,
}

/// Ask `ggml` about a type.
pub fn type_info(ty: GgmlType) -> Result<TypeInfo> {
    #[cfg(not(have_ggml))]
    {
        let _ = ty;
        Err(GgmlError::Unavailable)
    }
    #[cfg(have_ggml)]
    {
        // SAFETY: ggml_get_type_traits is a pure lookup over a static table.
        // It returns a pointer into that table, valid for the process lifetime.
        let traits = unsafe { ffi::ggml_get_type_traits(ty.0 as i32) };
        if traits.is_null() {
            return Err(GgmlError::UnsupportedType(ty.0));
        }
        // SAFETY: non-null pointer into ggml's static type table.
        let t = unsafe { &*traits };
        // SAFETY: type_name is a static NUL-terminated C string.
        let name = unsafe { std::ffi::CStr::from_ptr(t.type_name) }
            .to_string_lossy()
            .into_owned();
        Ok(TypeInfo {
            name,
            block_elems: t.blck_size,
            block_bytes: t.type_size,
            is_quantized: t.is_quantized,
            can_dequantize: t.to_float.is_some(),
        })
    }
}

/// Convert quantized bytes to `f32`, using `ggml`'s kernel for the type.
///
/// This is the join between our loader and the math: `data` is exactly what
/// [`bigtea_model::Model::read_tensor`] returns, still in its stored format.
pub fn dequantize(ty: GgmlType, data: &[u8], elements: usize) -> Result<Vec<f32>> {
    #[cfg(not(have_ggml))]
    {
        let _ = (ty, data, elements);
        Err(GgmlError::Unavailable)
    }
    #[cfg(have_ggml)]
    {
        // SAFETY: pure lookup into ggml's static type table.
        let traits = unsafe { ffi::ggml_get_type_traits(ty.0 as i32) };
        if traits.is_null() {
            return Err(GgmlError::UnsupportedType(ty.0));
        }
        // SAFETY: non-null pointer into a static table.
        let t = unsafe { &*traits };

        if t.blck_size <= 0 || elements as i64 % t.blck_size != 0 {
            return Err(GgmlError::PartialBlock {
                elements,
                block_size: t.blck_size,
            });
        }
        // The kernel reads exactly this many bytes; a short buffer would read
        // out of bounds, so it is checked rather than trusted. Checked before
        // the F32 path too, so both routes reject a malformed buffer alike.
        let expected = elements / t.blck_size as usize * t.type_size;
        if data.len() != expected {
            return Err(GgmlError::WrongSize {
                expected,
                actual: data.len(),
            });
        }

        // ggml supplies no `to_float` for F32 because the conversion is a
        // no-op -- there is genuinely nothing to call. Reinterpret instead of
        // reporting the type as unsupported.
        if ty.0 == GGML_TYPE_F32 {
            return Ok(data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect());
        }

        let Some(to_float) = t.to_float else {
            return Err(GgmlError::UnsupportedType(ty.0));
        };

        let mut out = vec![0f32; elements];
        // SAFETY: `data` holds exactly `expected` bytes as checked above, which
        // is what the kernel reads for `elements` values; `out` has capacity for
        // `elements` floats, which is what it writes. Neither aliases the other.
        unsafe {
            to_float(
                data.as_ptr() as *const std::os::raw::c_void,
                out.as_mut_ptr(),
                elements as i64,
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_whether_ggml_is_linked() {
        // Not an assertion about which: the crate must build both ways.
        let linked = available();
        if !linked {
            assert_eq!(type_info(GgmlType(0)), Err(GgmlError::Unavailable));
        }
    }

    #[cfg(have_ggml)]
    #[test]
    fn ggml_agrees_with_our_own_block_size_table() {
        // If these disagree, one table is wrong and every byte count derived
        // from it is wrong too -- including the numbers the whole plan rests on.
        for id in [0u32, 1, 8, 12, 14, 19, 23, 29, 30] {
            let ours = GgmlType(id);
            let (Some(our_elems), Some(our_bytes)) = (ours.block_elems(), ours.block_bytes())
            else {
                continue;
            };
            let theirs = type_info(ours).expect("ggml knows this type");
            assert_eq!(
                our_elems as i64, theirs.block_elems,
                "block elems disagree for type {id} ({})",
                theirs.name
            );
            assert_eq!(
                our_bytes as usize, theirs.block_bytes,
                "block bytes disagree for type {id} ({})",
                theirs.name
            );
        }
    }

    #[cfg(have_ggml)]
    #[test]
    fn dequantizes_f32_as_an_identity() {
        // F32 -> f32 must be exact; anything else means the plumbing is wrong.
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.5 - 8.0).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = dequantize(GgmlType(0), &bytes, values.len()).expect("dequantize");
        assert_eq!(out, values);
    }

    #[cfg(have_ggml)]
    #[test]
    fn rejects_a_buffer_of_the_wrong_size() {
        // Trusting the caller here would read out of bounds.
        let err = dequantize(GgmlType(0), &[0u8; 8], 64);
        assert!(matches!(err, Err(GgmlError::WrongSize { .. })));
    }

    #[cfg(have_ggml)]
    #[test]
    fn rejects_a_partial_block() {
        // Q4_K packs 256 elements per block; 100 is not a whole number of them.
        let err = dequantize(GgmlType(12), &[0u8; 144], 100);
        assert!(matches!(err, Err(GgmlError::PartialBlock { .. })));
    }
}
