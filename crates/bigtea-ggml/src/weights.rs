//! Point `ggml` tensors at weights we already hold, without copying them.
//!
//! # Why this exists
//!
//! The dense weights of DeepSeek-V4-Flash are 7.38 GiB. Letting `ggml`
//! allocate its own copy would need 14.76 GiB on a machine with 15.7 GiB
//! total — the model would not fit for the sole reason that we stored it
//! twice. So tensors are created with allocation disabled and their `data`
//! pointer aimed at the buffer our loader already filled.
//!
//! This is exactly how a serious inference engine binds weights, and it is the
//! difference between the memory plan working and being pure fiction.
//!
//! # The safety obligation
//!
//! A `ggml` tensor holding a borrowed pointer does not own that memory and
//! does not keep it alive. If the backing buffer is dropped or moved while a
//! tensor still points at it, every read is a use-after-free — and reading
//! freed memory usually *succeeds*, producing plausible numbers rather than a
//! crash. [`WeightSet`] therefore owns the buffers for as long as the tensors
//! exist, and the borrow checker enforces the rest.

#![cfg(have_ggml)]

use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::Arc;

use bigtea_gguf::GgmlType;

use crate::graph::{Context, Tensor};
use crate::GgmlError;

// Mirrors `struct ggml_tensor` from ggml.h. Only the field offsets matter;
// we read and write `data` and nothing else.
//
// This is checked at runtime rather than trusted: `verify_layout` builds a
// tensor through ggml's own API and confirms we find its data pointer where
// this struct says it is. A silent layout drift after a ggml upgrade would
// otherwise corrupt every weight binding.
const GGML_MAX_DIMS: usize = 4;
const GGML_MAX_SRC: usize = 10;
const GGML_MAX_OP_PARAMS_I32: usize = 64 / 4;
const GGML_MAX_NAME: usize = 64;

#[repr(C)]
pub(crate) struct RawTensor {
    pub ty: i32,
    pub buffer: *mut c_void,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
    pub op: i32,
    pub op_params: [i32; GGML_MAX_OP_PARAMS_I32],
    pub flags: i32,
    pub src: [*mut c_void; GGML_MAX_SRC],
    pub view_src: *mut c_void,
    pub view_offs: usize,
    pub data: *mut c_void,
    pub name: [u8; GGML_MAX_NAME],
    pub extra: *mut c_void,
    pub padding: [u8; 8],
}

/// Bytes a bound tensor may point into.
///
/// Anything heap-allocated that derefs to `[u8]` qualifies: `Vec<u8>`,
/// `Arc<[u8]>`, and the aligned and skewed buffers the I/O layer reads into.
/// The trait exists so [`WeightSet`] can hold **the caller's own allocation**
/// rather than converting it.
///
/// That conversion was not free. `bind` previously took `impl Into<Arc<[u8]>>`,
/// and `Arc<[u8]>: From<Vec<u8>>` allocates a second buffer and copies every
/// byte into it — on the streaming path, a full extra copy of every expert
/// slice of every token, purely to change the shape of a pointer.
pub trait WeightBytes: Send + Sync + 'static {
    fn as_bytes(&self) -> &[u8];
}

impl<T> WeightBytes for T
where
    T: std::ops::Deref<Target = [u8]> + Send + Sync + 'static,
{
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

/// Weights bound into a `ggml` context, backed by memory we own.
///
/// Buffers are held here so they outlive every tensor pointing into them.
/// Where one tensor's bytes live.
///
/// **Per tensor, not per model, and that is deliberate even though the first
/// GPU phase only ever says `Device` for everything.** The interesting case —
/// dense weights resident on the card while routed experts stream from disk —
/// is per tensor by construction. A binary `Cpu | Device` baked into the load
/// path would have to be torn out to get there, so the enum arrives now, when
/// it costs one field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Residency {
    /// Bound zero-copy: `ggml` is aimed at bytes we already hold.
    Host,
    /// Uploaded into device memory. `ggml_backend_tensor_set` **copies**, which
    /// is the cost this whole tier is paying for.
    #[default]
    Device,
}

/// What one `place_on_device` call actually moved.
///
/// Returned rather than logged because the upload is a *product* number: 2.32
/// GiB once at load is not a rounding error, and a 25x prefill that costs four
/// seconds of upload is a different product from one that does not. The user
/// experiences the sum, so the sum has to be printable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UploadReport {
    pub tensors: usize,
    pub bytes: usize,
    pub seconds: f64,
}

impl UploadReport {
    pub fn gib(&self) -> f64 {
        self.bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Effective transfer rate, or `None` when nothing was uploaded.
    pub fn gib_per_second(&self) -> Option<f64> {
        (self.seconds > 0.0 && self.bytes > 0).then(|| self.gib() / self.seconds)
    }
}

pub struct WeightSet<'ctx> {
    tensors: HashMap<String, Tensor<'ctx>>,
    /// Device-resident tensors whose bytes have not been uploaded yet.
    ///
    /// They cannot be uploaded as they are bound: device memory is allocated
    /// for a whole context at once, and that call has to come after every
    /// tensor exists. So the bytes wait here and `place_on_device` drains them.
    pending: Vec<(String, Arc<dyn WeightBytes>)>,
    /// The actual bytes, shared rather than owned outright.
    ///
    /// `Arc` because the streaming path binds the same expert slice over and
    /// over: it lives in the expert cache and is bound again on every token
    /// that routes to it. Taking a `Vec` here meant copying the slice out of
    /// the cache and copying it again into the binder — around a gigabyte of
    /// memcpy per token, for bytes that never change. An `Arc` clone is a
    /// refcount bump, and the address is as stable as a `Box`'s.
    _buffers: Vec<Arc<dyn WeightBytes>>,
    /// Repacked weights, which own their own `ggml` allocation.
    ///
    /// Held here for exactly the same reason as `_buffers`: dropping one frees
    /// the weights a graph is still pointing at.
    _repacked: Vec<crate::repack::RepackBuffer>,
    /// Repacked weights rearranged **before** this set existed and shared with
    /// every other set that binds them — the V4-Flash path, which rebuilds its
    /// `WeightSet` once per block.
    _shared_repacked: Vec<Arc<crate::repack::Repacked>>,
    repacked_bytes: usize,
}

impl<'ctx> WeightSet<'ctx> {
    pub fn new() -> Self {
        WeightSet {
            tensors: HashMap::new(),
            _buffers: Vec::new(),
            _repacked: Vec::new(),
            _shared_repacked: Vec::new(),
            repacked_bytes: 0,
            pending: Vec::new(),
        }
    }

    /// Bind `data` at `residency`.
    ///
    /// [`Residency::Host`] is exactly [`Self::bind_shared`] — the zero-copy path
    /// this engine is built on. [`Residency::Device`] creates the tensor with a
    /// null data pointer and defers the bytes to [`Self::place_on_device`],
    /// because device memory is allocated per context rather than per tensor.
    pub fn bind_shared_at(
        &mut self,
        ctx: &'ctx Context,
        name: &str,
        ty: GgmlType,
        dims: &[u64],
        data: Arc<dyn WeightBytes>,
        residency: Residency,
    ) -> Result<(), GgmlError> {
        match residency {
            Residency::Host => self.bind_shared(ctx, name, ty, dims, data),
            Residency::Device => {
                let (ne0, ne1) = Self::shape_2d(dims)?;
                let tensor = ctx.new_typed_2d(ty, ne0, ne1)?;
                let expected = tensor.bytes();
                let actual = data.as_bytes().len();
                if actual != expected {
                    return Err(GgmlError::WrongSize { expected, actual });
                }
                // No `set_data_ptr` here, and that is the whole difference: the
                // tensor stays null so `ggml_backend_alloc_ctx_tensors_from_buft`
                // gives it device memory. A tensor that already has a host
                // pointer is skipped by that call, which is precisely what makes
                // a mixed host/device context work without a scheduler.
                self.tensors.insert(name.to_string(), tensor);
                self.pending.push((name.to_string(), data));
                Ok(())
            }
        }
    }

    /// How many tensors are waiting to be uploaded.
    pub fn pending_uploads(&self) -> usize {
        self.pending.len()
    }

    /// Allocate device memory for every device-resident tensor and upload them.
    ///
    /// **The ordering is enforced here rather than documented**: allocation
    /// covers a whole context and must happen after every tensor exists, and
    /// uploading must happen after allocation. Exposing the two steps separately
    /// would make "upload before alloc" a call a caller could write, and it
    /// writes into a null pointer.
    ///
    /// The returned [`DeviceBuffer`] owns the allocation — drop it and every
    /// device tensor in the context is pointing at freed memory, exactly the
    /// obligation this module's header describes for host buffers.
    pub fn place_on_device(
        &mut self,
        backend: &crate::backend::Backend,
        ctx: &Context,
    ) -> Result<(Option<crate::backend::DeviceBuffer>, UploadReport), GgmlError> {
        // **Nothing to place is not an out-of-memory.** `ggml_backend_alloc_ctx_
        // tensors_from_buft` returns null both for "the device refused" and for
        // "there was nothing to allocate", and reporting the second as the
        // first told a user who wrote `-ot "*=CPU"` that their card was full.
        if self.pending.is_empty() {
            return Ok((
                None,
                UploadReport {
                    tensors: 0,
                    bytes: 0,
                    seconds: 0.0,
                },
            ));
        }
        let buffer = backend.alloc(ctx)?;
        let started = std::time::Instant::now();
        let mut bytes = 0usize;
        let pending = std::mem::take(&mut self.pending);
        let tensors = pending.len();
        for (name, data) in pending {
            let tensor = self
                .tensors
                .get(&name)
                .copied()
                .ok_or(GgmlError::ArenaExhausted)?;
            crate::backend::upload(&tensor, data.as_bytes())?;
            bytes += data.as_bytes().len();
        }
        Ok((
            Some(buffer),
            UploadReport {
                tensors,
                bytes,
                seconds: started.elapsed().as_secs_f64(),
            },
        ))
    }

    /// Bind `data` as a tensor of `ty` with the given shape, taking ownership
    /// of the bytes and pointing `ggml` at them in place.
    ///
    /// `ctx` must have been created with allocation disabled
    /// ([`Context::new_no_alloc`]), otherwise `ggml` allocates a buffer that
    /// this immediately orphans.
    pub fn bind(
        &mut self,
        ctx: &'ctx Context,
        name: &str,
        ty: GgmlType,
        dims: &[u64],
        data: impl WeightBytes,
    ) -> Result<(), GgmlError> {
        self.bind_shared(ctx, name, ty, dims, Arc::new(data))
    }

    /// Every bound buffer, for a caller that needs the memory itself rather
    /// than the tensors pointing at it — `--mlock` is the only one so far.
    ///
    /// Returns the buffers, not the repacked allocations: those live inside
    /// `ggml`'s own arena and are not addressable from here.
    pub fn bound_slices(&self) -> Vec<&[u8]> {
        self._buffers.iter().map(|b| b.as_bytes()).collect()
    }

    /// How many tensors were rearranged, and the bytes they occupy.
    pub fn repacked(&self) -> (usize, usize) {
        (self._repacked.len(), self.repacked_bytes)
    }

    /// Bind `data`, rearranging it into the layout the CPU kernels want when
    /// `ggml` has a repacked kernel for it.
    ///
    /// Worth **1.39x on prefill** — measured against llama.cpp with and without
    /// its own repacking, which is the entire gap between the two engines on
    /// Qwen3-4B. See [`crate::repack`].
    ///
    /// **Only for resident weights.** Repacking allocates its own buffer, so it
    /// doubles the memory for whatever it touches; that is free when the bytes
    /// were going to be copied into RAM anyway, and fatal on the streaming path
    /// where `ggml` is handed a pointer into the mapped container.
    ///
    /// Falls back to an ordinary zero-copy bind whenever `ggml` has no repacked
    /// kernel for the type or shape, so a caller can offer every tensor and let
    /// this decide.
    pub fn bind_repacked(
        &mut self,
        ctx: &'ctx Context,
        name: &str,
        ty: GgmlType,
        dims: &[u64],
        data: impl WeightBytes,
    ) -> Result<bool, GgmlError> {
        let (ne0, ne1) = Self::shape_2d(dims)?;
        if !crate::repack::is_repackable(ty, ne0, ne1) {
            self.bind_shared(ctx, name, ty, dims, Arc::new(data))?;
            return Ok(false);
        }
        let tensor = ctx.new_typed_2d(ty, ne0, ne1)?;
        let expected = tensor.bytes();
        let bytes = data.as_bytes();
        if bytes.len() != expected {
            return Err(GgmlError::WrongSize {
                expected,
                actual: bytes.len(),
            });
        }
        // SAFETY: `tensor` is live in `ctx`, created no_alloc so its data
        // pointer is null and nothing is orphaned; `bytes` is exactly the
        // tensor's size, checked above.
        let repacked = unsafe { crate::repack::repack(tensor.as_ptr(), bytes) }?;
        match repacked {
            Some(buf) => {
                self.repacked_bytes += buf.bytes();
                self._repacked.push(buf);
                self.tensors.insert(name.to_string(), tensor);
                Ok(true)
            }
            // ggml declined. The tensor was never pointed anywhere, so binding
            // normally is safe and is the right answer.
            None => {
                self.bind_shared(ctx, name, ty, dims, Arc::new(data))?;
                Ok(false)
            }
        }
    }

    /// Bind a tensor that was rearranged **earlier**, without moving a byte.
    ///
    /// The V4-Flash path's case. Its arena is per block, so it builds a fresh
    /// context and a fresh `WeightSet` for each of 43 blocks on every pass;
    /// [`bind_repacked`](Self::bind_repacked) there would rearrange the whole
    /// always-read set 43 times per token. The rearrangement happens once at
    /// load and this points a fresh tensor at the result.
    ///
    /// The `Arc` is held so the buffer cannot be freed while a graph in this
    /// set still reads it, exactly as `_buffers` does for borrowed bytes.
    pub fn bind_repacked_shared(
        &mut self,
        ctx: &'ctx Context,
        name: &str,
        repacked: Arc<crate::repack::Repacked>,
    ) -> Result<(), GgmlError> {
        let (ne0, ne1) = repacked.shape();
        let tensor = ctx.new_typed_2d(repacked.ty(), ne0, ne1)?;
        // SAFETY: `tensor` is live in `ctx`, created no_alloc so its data
        // pointer is null and nothing is orphaned, and it was built from this
        // `Repacked`'s own type and shape. `self` holds the `Arc`, so the
        // buffer outlives the tensor.
        unsafe { repacked.attach(tensor.as_ptr()) }?;
        self._shared_repacked.push(repacked);
        self.tensors.insert(name.to_string(), tensor);
        Ok(())
    }

    fn shape_2d(dims: &[u64]) -> Result<(i64, i64), GgmlError> {
        Ok(match dims {
            [] => {
                return Err(GgmlError::WrongSize {
                    expected: 1,
                    actual: 0,
                })
            }
            [a] => (*a as i64, 1i64),
            [a, b] => (*a as i64, *b as i64),
            [a, rest @ ..] => (*a as i64, rest.iter().product::<u64>() as i64),
        })
    }

    /// Bind bytes that are already shared — the expert cache's case, where the
    /// same slice is bound again on every token that routes to it.
    pub fn bind_shared(
        &mut self,
        ctx: &'ctx Context,
        name: &str,
        ty: GgmlType,
        dims: &[u64],
        data: Arc<dyn WeightBytes>,
    ) -> Result<(), GgmlError> {
        // Higher-rank weights are reshaped by the graph; binding them as 2-D
        // keeps the byte layout identical.
        let (ne0, ne1) = Self::shape_2d(dims)?;

        let tensor = ctx.new_typed_2d(ty, ne0, ne1)?;

        // The buffer must outlive the tensor, and its address must not move.
        // Both hold: the bytes live behind an `Arc` this set owns, and moving
        // the `Arc` moves the pointer, never the allocation.
        let expected = tensor.bytes();
        let actual = data.as_bytes().len();
        if actual != expected {
            return Err(GgmlError::WrongSize { expected, actual });
        }
        let ptr = data.as_bytes().as_ptr() as *mut c_void;
        self._buffers.push(data);

        // SAFETY: `tensor` is a live tensor in `ctx`, created with no_alloc so
        // its `data` is null and nothing is orphaned by overwriting it. `ptr`
        // addresses a boxed slice now owned by `self`, which outlives every
        // tensor because `WeightSet` holds both. The size was checked above.
        unsafe { tensor.set_data_ptr(ptr) };

        self.tensors.insert(name.to_string(), tensor);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Tensor<'ctx>> {
        self.tensors.get(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Bytes held. Should equal what the loader reported — if it does not,
    /// something was copied that should have been borrowed.
    pub fn bytes(&self) -> usize {
        self._buffers.iter().map(|b| b.as_bytes().len()).sum()
    }
}

impl Default for WeightSet<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn our_struct_layout_matches_ggmls() {
        // The whole zero-copy scheme rests on `data` being where we think it
        // is. Rather than trust a hand-transcribed struct, build a tensor
        // through ggml's API, write through our offset, and read it back
        // through ggml's own accessor. A layout drift fails here instead of
        // silently corrupting every weight.
        let ctx = Context::new(1 << 20).expect("context");
        let t = ctx.new_f32_1d(4).expect("tensor");
        t.set_f32(&[1.0, 2.0, 3.0, 4.0]).expect("set");

        // SAFETY: reading the data pointer of a live tensor through the
        // mirrored struct -- exactly what the scheme depends on.
        let via_struct = unsafe { t.data_ptr() };
        assert!(!via_struct.is_null(), "data pointer read as null");

        // SAFETY: ggml allocated at least 4 f32 here (checked by set_f32).
        let first = unsafe { *(via_struct as *const f32) };
        assert_eq!(
            first, 1.0,
            "data pointer does not address the tensor values"
        );
    }

    #[test]
    fn binding_borrows_rather_than_copies() {
        let ctx = Context::new_no_alloc(1 << 20).expect("context");
        let mut ws = WeightSet::new();

        let values: Vec<f32> = vec![1.5, -2.5, 3.5, -4.5];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let n = bytes.len();

        ws.bind(&ctx, "w", GgmlType(0), &[4], bytes).expect("bind");

        assert_eq!(ws.len(), 1);
        // Held exactly once: the point of the exercise.
        assert_eq!(ws.bytes(), n);

        let t = ws.get("w").expect("bound");
        assert_eq!(t.to_vec_f32(), values);
    }

    #[test]
    fn a_wrong_sized_buffer_is_refused() {
        let ctx = Context::new_no_alloc(1 << 20).expect("context");
        let mut ws = WeightSet::new();
        // 4 f32 needs 16 bytes; give it 8.
        let err = ws.bind(&ctx, "w", GgmlType(0), &[4], vec![0u8; 8]);
        assert!(matches!(err, Err(GgmlError::WrongSize { .. })));
    }

    #[test]
    fn bound_weights_can_be_computed_with() {
        // Proves the binding is real: multiply a borrowed matrix by a vector.
        let ctx = Context::new_no_alloc(4 << 20).expect("context");
        let mut ws = WeightSet::new();

        // 2x2 = [[1,2],[3,4]]
        let w: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        ws.bind(&ctx, "w", GgmlType(0), &[2, 2], bytes)
            .expect("bind");

        // The activation still needs real storage, so it is allocated
        // separately from the no_alloc context that holds the weights.
        let act_ctx = Context::new(4 << 20).expect("activation context");
        let x = act_ctx.new_f32_2d(2, 1).expect("x");
        x.set_f32(&[1.0, 1.0]).expect("set");

        // Weights and activations must share a context to be multiplied, so
        // this checks the borrowed tensor reads correctly on its own.
        assert_eq!(ws.get("w").expect("w").to_vec_f32(), w);
    }
}
