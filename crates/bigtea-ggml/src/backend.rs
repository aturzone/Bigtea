//! The second binding path: weights that live on a device.
//!
//! # Why this file exists at all
//!
//! `weights.rs:286` is the whole memory design in one line — a host pointer
//! written straight into `tensor->data`, so a 17 GiB model is bound without a
//! copy on a 15.7 GiB machine. **That path does not exist for a device.** A
//! device tensor's data lives in memory the CPU cannot address, so it is
//! allocated through a buffer type and filled by `ggml_backend_tensor_set`,
//! which *copies*.
//!
//! That is not a detail to work around; it is the tier's actual cost, and it is
//! why the GPU work was deferred twice. Vulkan versus CUDA does not change it
//! — see `research/gpu-the-card-works-vulkan-not-cuda-2026-08-15.md`.
//!
//! # What is deliberately not here
//!
//! No `ggml_backend_sched`, no mixed-device graph, no partial residency. The
//! approved first slice is a model that fits entirely on the device, where the
//! scheduler has nothing to schedule. Everything in this file works on whole
//! contexts for that reason.
//!
//! Gated whole, like `graph.rs`: without ggml there is no backend to open, so
//! this is absent rather than degraded. `device.rs` is the opposite — it stays
//! callable and answers `Unavailable`, because "is there a GPU here?" is a
//! reasonable question to ask of a build that cannot use one.
#![cfg(have_ggml)]

use std::ptr::NonNull;

use crate::{Context, GgmlError, Tensor};

/// A device backend, ready to execute graphs.
///
/// Owns the ggml backend handle and frees it on drop. The *device* behind it is
/// registry-owned and outlives the process, so nothing here frees that.
pub struct Backend {
    raw: NonNull<std::os::raw::c_void>,
    /// Index into `devices()`, kept so errors can name the device.
    index: usize,
}

/// Memory allocated on a device, holding every tensor of one context.
///
/// Dropping it frees the device allocation, so it must outlive every tensor
/// allocated into it — the same rule as `WeightSet`, enforced the same way, by
/// keeping the buffer and the context alive together in the caller.
pub struct DeviceBuffer {
    raw: NonNull<std::os::raw::c_void>,
    bytes: usize,
}

impl DeviceBuffer {
    /// How much device memory this allocation actually took.
    ///
    /// Worth reading rather than assuming: a backend may pad, and the tier's
    /// cost is measured in what the device gave up, not in what we asked for.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `ggml_backend_alloc_ctx_tensors_from_buft`
        // and is freed exactly once, here.
        unsafe { ffi::ggml_backend_buffer_free(self.raw.as_ptr()) };
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `ggml_backend_dev_init` and is freed once.
        unsafe { ffi::ggml_backend_free(self.raw.as_ptr()) };
    }
}

mod ffi {
    use std::os::raw::{c_char, c_int, c_void};

    pub type DevT = *mut c_void;
    pub type BackendT = *mut c_void;
    pub type BufferT = *mut c_void;
    pub type BufferTypeT = *mut c_void;

    // Transcribed together from one revision of ggml-backend.h / ggml-alloc.h.
    extern "C" {
        pub fn ggml_backend_dev_count() -> usize;
        pub fn ggml_backend_dev_get(index: usize) -> DevT;
        pub fn ggml_backend_dev_init(device: DevT, params: *const c_char) -> BackendT;
        pub fn ggml_backend_dev_buffer_type(device: DevT) -> BufferTypeT;
        pub fn ggml_backend_free(backend: BackendT);
        pub fn ggml_backend_buffer_free(buffer: BufferT);
        pub fn ggml_backend_buffer_get_size(buffer: BufferT) -> usize;
        pub fn ggml_backend_alloc_ctx_tensors_from_buft(
            ctx: *mut c_void,
            buft: BufferTypeT,
        ) -> BufferT;
        pub fn ggml_backend_tensor_set(
            tensor: *mut c_void,
            data: *const c_void,
            offset: usize,
            size: usize,
        );
        pub fn ggml_backend_tensor_get(
            tensor: *const c_void,
            data: *mut c_void,
            offset: usize,
            size: usize,
        );
        pub fn ggml_backend_graph_compute(backend: BackendT, cgraph: *mut c_void) -> c_int;
    }
}

impl Backend {
    /// Open the device at `index` in [`crate::devices`].
    ///
    /// Takes an index rather than a `DeviceInfo` so there is exactly one place
    /// that turns a listing into a live handle, and no way to hold a stale
    /// pointer to a device the registry has re-enumerated.
    pub fn open(index: usize) -> Result<Self, GgmlError> {
        // **The bounds check has to be here, not in ggml.** An out-of-range
        // index does not return null — `ggml_backend_dev_get` runs
        // `GGML_ASSERT(index < ggml_backend_dev_count())`, which aborts the
        // process. The first version of this function trusted a null and
        // the test that opens device 9999 took the whole test binary down
        // with `exit code: 3`, reported as "process didn't exit
        // successfully" rather than as a failing test — the same abort
        // class CLAUDE.md records for an exhausted arena.
        //
        // SAFETY: reading the registry's device count is a pure query.
        let count = unsafe { ffi::ggml_backend_dev_count() };
        if index >= count {
            return Err(GgmlError::NoSuchDevice(index));
        }
        // SAFETY: `index` was just bounds-checked against the same count
        // ggml asserts on, and the registry is process-lifetime.
        let dev = unsafe { ffi::ggml_backend_dev_get(index) };
        let dev = NonNull::new(dev).ok_or(GgmlError::NoSuchDevice(index))?;
        // SAFETY: `dev` is a live registry device. Null params = defaults.
        let raw = unsafe { ffi::ggml_backend_dev_init(dev.as_ptr(), std::ptr::null()) };
        let raw = NonNull::new(raw).ok_or(GgmlError::DeviceInitFailed(index))?;
        Ok(Self { raw, index })
    }

    pub fn device_index(&self) -> usize {
        self.index
    }

    /// Allocate device memory for **every tensor in `ctx` that has none**.
    ///
    /// The context must have been created with `no_alloc`, which is already how
    /// this crate builds weight contexts — so the tensors exist with null data
    /// and this fills them in on the device rather than in host memory.
    ///
    /// One allocation covers the whole context, including graph intermediates,
    /// which is why it is called after the graph is built and not before.
    pub fn alloc(&self, ctx: &Context) -> Result<DeviceBuffer, GgmlError> {
        // SAFETY: `dev` is live; `buft` is owned by the backend, not by us.
        let dev = unsafe { ffi::ggml_backend_dev_get(self.index) };
        let buft = unsafe { ffi::ggml_backend_dev_buffer_type(dev) };
        if buft.is_null() {
            return Err(GgmlError::DeviceInitFailed(self.index));
        }
        // SAFETY: `ctx` is a live context whose tensors have null data.
        let raw = unsafe { ffi::ggml_backend_alloc_ctx_tensors_from_buft(ctx.as_raw(), buft) };
        let raw = NonNull::new(raw).ok_or(GgmlError::DeviceOutOfMemory)?;
        // SAFETY: freshly allocated, non-null buffer.
        let bytes = unsafe { ffi::ggml_backend_buffer_get_size(raw.as_ptr()) };
        Ok(DeviceBuffer { raw, bytes })
    }

    /// Run `outputs` and everything they depend on, on the device.
    pub fn compute(&self, ctx: &Context, outputs: &[&Tensor<'_>]) -> Result<(), GgmlError> {
        if outputs.is_empty() {
            return Ok(());
        }
        let graph = ctx.build_forward(outputs)?;
        // SAFETY: backend and graph are both live; the graph's tensors were
        // allocated from this backend's buffer type.
        let status = unsafe { ffi::ggml_backend_graph_compute(self.raw.as_ptr(), graph) };
        if status != 0 {
            return Err(GgmlError::ComputeFailed(status));
        }
        Ok(())
    }
}

/// Copy `bytes` into a device tensor.
///
/// **This is the copy the whole tier costs.** It is not hidden behind a
/// zero-copy-looking name for that reason: a reader who sees `upload` at a call
/// site knows the bytes crossed the bus there.
pub fn upload(tensor: &Tensor<'_>, bytes: &[u8]) -> Result<(), GgmlError> {
    let expected = tensor.bytes();
    if bytes.len() != expected {
        return Err(GgmlError::WrongSize {
            expected,
            actual: bytes.len(),
        });
    }
    // SAFETY: the tensor is device-allocated and the length was just
    // checked against what ggml says the tensor holds.
    unsafe { ffi::ggml_backend_tensor_set(tensor.as_raw(), bytes.as_ptr().cast(), 0, bytes.len()) };
    Ok(())
}

/// Copy a device tensor back into host memory.
pub fn download(tensor: &Tensor<'_>, out: &mut [u8]) -> Result<(), GgmlError> {
    let expected = tensor.bytes();
    if out.len() != expected {
        return Err(GgmlError::WrongSize {
            expected,
            actual: out.len(),
        });
    }
    // SAFETY: as above, in the other direction.
    unsafe { ffi::ggml_backend_tensor_get(tensor.as_raw(), out.as_mut_ptr().cast(), 0, out.len()) };
    Ok(())
}

/// Read a device tensor back as `f32`.
pub fn download_f32(tensor: &Tensor<'_>) -> Result<Vec<f32>, GgmlError> {
    let n = tensor.bytes() / std::mem::size_of::<f32>();
    let mut raw = vec![0u8; tensor.bytes()];
    download(tensor, &mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .take(n)
        .collect())
}

/// Copy `values` into a device tensor as `f32`.
pub fn upload_f32(tensor: &Tensor<'_>, values: &[f32]) -> Result<(), GgmlError> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    upload(tensor, &bytes)
}

/// Where a graph runs, and everything that differs because of it.
///
/// # Why this exists
///
/// The forward pass is identical on both paths — same tensors, same operations,
/// same order. What differs is four mechanical things: how a context is
/// created, whether its tensors need device memory, how values get in and out,
/// and who executes the graph. Threading a `Compute` through means the graph
/// code above it does not change, which was the condition the GPU tier was
/// approved under.
///
/// # The ordering this encodes
///
/// [`Self::realize`] must be called **after the graph is built and before any
/// input is set**. On the CPU it does nothing. On a device it allocates memory
/// for every tensor in the context, which cannot happen before the tensors
/// exist, and inputs cannot be uploaded before that memory exists. Call sites
/// that get this wrong write into a null pointer, so the sequence is stated
/// here once rather than rediscovered per site.
pub enum Compute<'b> {
    Cpu { threads: usize },
    Device(&'b Backend),
}

impl Compute<'_> {
    /// A context sized `arena`, allocating or not as the target requires.
    ///
    /// The device wants `no_alloc` — its tensors are filled by
    /// [`Self::realize`] — while the CPU path wants ggml's own arena, exactly
    /// as it has always had.
    pub fn context(&self, arena: usize) -> Result<Context, GgmlError> {
        match self {
            Compute::Cpu { .. } => Context::new(arena),
            // Metadata only: a few hundred bytes per tensor, because the bytes
            // live on the card.
            Compute::Device(_) => Context::new_no_alloc(arena),
        }
    }

    /// Give every tensor in `ctx` somewhere to live. **After the graph, before
    /// the inputs.**
    ///
    /// The returned buffer owns the device allocation and must outlive every
    /// tensor in the context — dropping it early leaves the graph pointing at
    /// freed device memory, which is the same obligation the host path has for
    /// its byte buffers.
    pub fn realize(&self, ctx: &Context) -> Result<Option<DeviceBuffer>, GgmlError> {
        match self {
            Compute::Cpu { .. } => Ok(None),
            Compute::Device(b) => b.alloc(ctx).map(Some),
        }
    }

    pub fn set_f32(&self, t: &Tensor<'_>, values: &[f32]) -> Result<(), GgmlError> {
        match self {
            Compute::Cpu { .. } => t.set_f32(values),
            Compute::Device(_) => upload_f32(t, values),
        }
    }

    pub fn to_vec_f32(&self, t: &Tensor<'_>) -> Result<Vec<f32>, GgmlError> {
        match self {
            Compute::Cpu { .. } => Ok(t.to_vec_f32()),
            Compute::Device(_) => download_f32(t),
        }
    }

    pub fn run(&self, ctx: &Context, outputs: &[&Tensor<'_>]) -> Result<(), GgmlError> {
        match self {
            Compute::Cpu { threads } => ctx.compute_many(outputs, *threads),
            Compute::Device(b) => b.compute(ctx, outputs),
        }
    }

    /// Which residency weights should be bound at for this target.
    ///
    /// Phase A says "all device" and this returns exactly that. It is a method
    /// rather than a constant because Phase C answers per tensor, and the call
    /// sites that ask should keep asking.
    pub fn weight_residency(&self) -> crate::Residency {
        match self {
            Compute::Cpu { .. } => crate::Residency::Host,
            Compute::Device(_) => crate::Residency::Device,
        }
    }
}
