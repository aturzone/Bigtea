//! `ggml_backend_sched` — the thing that makes a mixed host/device graph legal.
//!
//! # What was blocked on this, exactly
//!
//! `backend.rs` says it in its own header: *no `ggml_backend_sched`, no
//! mixed-device graph, no partial residency*. Phase A's slice was a model that
//! fits **entirely** on the device, because a graph with one host operand and
//! one device operand does not fail — it **segfaults at compute**, recorded in
//! `research/mixed-residency-segfaults-2026-08-15.md`, three of them in one
//! session.
//!
//! Five declined flags name that single missing piece: `--split-mode`,
//! `--tensor-split`, `--op-offload`, `-ngl` and `--n-gpu-layers`. All five are
//! one question — *which parts run where* — and none of them can be answered
//! while a mixed graph is undefined behaviour.
//!
//! # Why a scheduler is not just a bigger allocator
//!
//! [`crate::GraphAllocator`] plans storage for a graph whose tensors are all on
//! **one** backend. It has no opinion about where an operation runs, because
//! there is only one place it can run.
//!
//! A scheduler answers a different question. Given a graph whose tensors live in
//! several buffers, it partitions the nodes into *splits* — maximal runs that
//! execute on one backend — and inserts the copies between them. The count of
//! those splits is the honest measure of what it did: **one split means it found
//! nothing to schedule** and the whole apparatus was an expensive allocator.
//! [`Scheduler::splits`] is exposed for that reason, and the tests assert on it.
//!
//! # The rule this file encodes
//!
//! **A tensor with a data pointer and no buffer cannot cross a split.** Our
//! zero-copy host bind (`weights.rs`) writes straight into `tensor->data` and
//! leaves `tensor->buffer` null, which is exactly right for the CPU path and
//! exactly what the scheduler cannot copy from: it reaches the copy through
//! `buffer->iface`, and a null buffer is the segfault. [`HostBuffer`] is the
//! fix — `ggml_backend_cpu_buffer_from_ptr` wraps memory we already own, still
//! without a copy, and gives it the buffer identity the scheduler needs.
//!
//! So the zero-copy design survives; it just has to say *whose* memory it is.
#![cfg(have_ggml)]

use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::backend::{ffi as bffi, Backend};
use crate::{Context, GgmlError, Tensor};

mod ffi {
    use std::os::raw::{c_int, c_void};

    use crate::backend::ffi::{BackendT, BufferT, BufferTypeT};

    pub type SchedT = *mut c_void;

    extern "C" {
        pub fn ggml_backend_sched_new(
            backends: *const BackendT,
            bufts: *const BufferTypeT,
            n_backends: c_int,
            graph_size: usize,
            parallel: bool,
            op_offload: bool,
        ) -> SchedT;
        pub fn ggml_backend_sched_free(sched: SchedT);
        pub fn ggml_backend_sched_reserve(sched: SchedT, measure_graph: *mut c_void) -> bool;
        pub fn ggml_backend_sched_alloc_graph(sched: SchedT, graph: *mut c_void) -> bool;
        pub fn ggml_backend_sched_graph_compute(sched: SchedT, graph: *mut c_void) -> c_int;
        pub fn ggml_backend_sched_reset(sched: SchedT);
        pub fn ggml_backend_sched_get_n_splits(sched: SchedT) -> c_int;
        pub fn ggml_backend_sched_get_n_copies(sched: SchedT) -> c_int;
        pub fn ggml_backend_sched_get_buffer_size(sched: SchedT, backend: BackendT) -> usize;
        pub fn ggml_backend_sched_set_tensor_backend(
            sched: SchedT,
            node: *mut c_void,
            backend: BackendT,
        );
        pub fn ggml_backend_sched_get_tensor_backend(sched: SchedT, node: *mut c_void) -> BackendT;

        // From ggml-backend.h's "utils" section. Wraps host memory we already
        // own in a buffer, without copying it.
        pub fn ggml_backend_cpu_buffer_from_ptr(ptr: *mut c_void, size: usize) -> BufferT;
        pub fn ggml_backend_tensor_alloc(
            buffer: BufferT,
            tensor: *mut c_void,
            addr: *mut c_void,
        ) -> c_int;
    }
}

/// The alignment ggml requires of any pointer it is handed as a buffer.
///
/// `TENSOR_ALIGNMENT` in `ggml-backend.cpp`. Not a suggestion and not checked
/// politely: `ggml_backend_cpu_buffer_from_ptr` **asserts** on it, so an
/// ordinary `Vec<u8>` (alignment 1, in practice 16 from the allocator) aborts
/// the process rather than returning an error.
pub const TENSOR_ALIGNMENT: usize = 32;

/// An owned host allocation that ggml will accept as a buffer.
///
/// # Why this is not a `Vec<u8>`
///
/// Because a `Vec<u8>` aborts. Rust guarantees byte vectors alignment 1 and the
/// system allocator gives 16 in practice; ggml wants 32 and takes the process
/// down when it does not get it. So memory intended for a scheduled graph has
/// to be allocated for that purpose, and this is the type that says so at the
/// call site instead of leaving it to luck.
///
/// **This matters beyond the tests.** Streamed expert bytes land in whatever the
/// reader allocated, and `io`'s `SkewedBuf` deliberately offsets its destination
/// to `file_offset % 4096` to make direct transfers possible — which is not
/// 32-aligned in general. Host weights that are to cross a split must be read
/// into one of these.
pub struct AlignedBytes {
    ptr: NonNull<u8>,
    len: usize,
    layout: std::alloc::Layout,
}

impl AlignedBytes {
    /// `len` zeroed bytes, aligned for ggml.
    pub fn zeroed(len: usize) -> Result<Self, GgmlError> {
        // A zero-length allocation is undefined; ggml has nothing to point at
        // either, so it is refused rather than fudged to one byte.
        if len == 0 {
            return Err(GgmlError::WrongSize {
                expected: 1,
                actual: 0,
            });
        }
        let layout = std::alloc::Layout::from_size_align(len, TENSOR_ALIGNMENT)
            .map_err(|_| GgmlError::ContextAlloc { bytes: len })?;
        // SAFETY: `layout` has a non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or(GgmlError::ContextAlloc { bytes: len })?;
        Ok(Self { ptr, len, layout })
    }

    /// A copy of `bytes`, aligned for ggml.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, GgmlError> {
        let mut out = Self::zeroed(bytes.len())?;
        out.copy_from_slice(bytes);
        Ok(out)
    }
}

impl std::ops::Deref for AlignedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is a live allocation of `len` initialised bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl std::ops::DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` makes the borrow exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // SAFETY: same pointer and layout the allocation came from, freed once.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Host memory given a buffer identity, so the scheduler can copy out of it.
///
/// **It does not own or copy the bytes.** It wraps a slice the caller already
/// holds, which is the same contract `WeightSet` has with its buffers: drop the
/// underlying memory while a graph still points at it and the graph is reading
/// freed pages. The lifetime parameter is what says so.
pub struct HostBuffer<'a> {
    raw: NonNull<std::os::raw::c_void>,
    base: *mut u8,
    len: usize,
    _owner: PhantomData<&'a mut [u8]>,
}

impl<'a> HostBuffer<'a> {
    /// Wrap `bytes` as a CPU buffer.
    ///
    /// Takes `&mut` rather than `&`: ggml's CPU backend will write into a buffer
    /// it is told about — a graph output landing here is a store, not a load —
    /// and the aliasing has to be true at the type level rather than in a
    /// comment.
    pub fn wrap(bytes: &'a mut [u8]) -> Result<Self, GgmlError> {
        let base = bytes.as_mut_ptr();
        let len = bytes.len();
        // **Checked here because ggml does not check, it aborts.** The assert is
        // `(uintptr_t)ptr % TENSOR_ALIGNMENT == 0`, and the first version of the
        // test that exercised this handed it a `Vec<u8>` and took the whole test
        // binary down with exit code 3 — the same shape as an exhausted arena,
        // and the reason `AlignedBytes` exists.
        if (base as usize) % TENSOR_ALIGNMENT != 0 {
            return Err(GgmlError::Misaligned {
                address: base as usize,
                required: TENSOR_ALIGNMENT,
            });
        }
        // SAFETY: `bytes` is live for `'a` and exclusively borrowed; ggml stores
        // the pointer and does not free it (a from_ptr buffer's free is a no-op).
        let raw = unsafe { ffi::ggml_backend_cpu_buffer_from_ptr(base.cast(), len) };
        NonNull::new(raw)
            .map(|raw| Self {
                raw,
                base,
                len,
                _owner: PhantomData,
            })
            .ok_or(GgmlError::DeviceOutOfMemory)
    }

    /// Point `tensor` at `offset` bytes into this buffer, and record the buffer
    /// on the tensor so a split can copy from it.
    ///
    /// This is the scheduler-safe form of the zero-copy bind. The plain one sets
    /// only `data`; this one sets `data` *and* `buffer`, which is the entire
    /// difference between a graph that computes and one that faults.
    pub fn attach(&self, tensor: &Tensor<'_>, offset: usize) -> Result<(), GgmlError> {
        let need = tensor.bytes();
        if offset > self.len || need > self.len - offset {
            return Err(GgmlError::WrongSize {
                expected: need,
                actual: self.len.saturating_sub(offset),
            });
        }
        // The base is aligned, so an aligned offset keeps the tensor aligned.
        // Unaligned tensor data does not abort here — it produces wrong reads on
        // backends that assume it, which is worse.
        if offset % TENSOR_ALIGNMENT != 0 {
            return Err(GgmlError::Misaligned {
                address: offset,
                required: TENSOR_ALIGNMENT,
            });
        }
        // SAFETY: bounds were just checked against the wrapped length.
        let addr = unsafe { self.base.add(offset) };
        // SAFETY: `raw` is a live CPU buffer and `tensor` is a live tensor from
        // a no_alloc context, i.e. one that still has a null data pointer.
        let status = unsafe {
            ffi::ggml_backend_tensor_alloc(self.raw.as_ptr(), tensor.as_raw(), addr.cast())
        };
        if status != 0 {
            return Err(GgmlError::ComputeFailed(status));
        }
        Ok(())
    }
}

impl Drop for HostBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: freed exactly once. A `from_ptr` buffer's free releases the
        // wrapper, never the caller's bytes.
        unsafe { bffi::ggml_backend_buffer_free(self.raw.as_ptr()) };
    }
}

/// Runs one graph across several backends, inserting the copies between them.
///
/// # Ordering, which is the same rule the rest of this crate has
///
/// [`Self::realize`] then inputs then [`Self::run`] — after the graph is built,
/// before anything is written into it. Identical to `Compute::realize_graph`,
/// and for the identical reason: allocation is what gives a tensor an address.
///
/// # The backend order is the priority order
///
/// ggml treats index 0 as most preferred and walks down. Passing the device
/// first and the CPU last is what makes "run it on the card unless you cannot"
/// the default, and passing them the other way round silently produces a
/// CPU-only run that still reports success.
pub struct Scheduler<'b> {
    raw: NonNull<std::os::raw::c_void>,
    /// Borrowed, not owned: the scheduler stores raw backend pointers and using
    /// one after its `Backend` dropped is a use-after-free. The lifetime is the
    /// enforcement.
    _backends: PhantomData<&'b [&'b Backend]>,
}

impl<'b> Scheduler<'b> {
    /// A scheduler over `backends`, **most preferred first**.
    ///
    /// `graph_size` must be at least as large as the largest graph it will be
    /// given; ggml sizes its internal hash set from it once and does not grow.
    ///
    /// `op_offload` lets it move a large matmul onto a device even when the
    /// operands are host-resident — llama.cpp's `--op-offload`, defaulted on
    /// there and left to the caller here.
    pub fn new(
        backends: &'b [&'b Backend],
        graph_size: usize,
        op_offload: bool,
    ) -> Result<Self, GgmlError> {
        if backends.is_empty() {
            return Err(GgmlError::NoSuchDevice(0));
        }
        let handles: Vec<bffi::BackendT> = backends.iter().map(|b| b.as_raw()).collect();
        let bufts: Vec<bffi::BufferTypeT> = backends
            .iter()
            .map(|b| b.buffer_type())
            .collect::<Result<_, _>>()?;
        // SAFETY: both arrays are `backends.len()` long and live for the call;
        // ggml copies them into the scheduler.
        //
        // `parallel: false` — that flag turns on multiple copies of the graph
        // for pipelined multi-GPU execution, and there is one card here. Turning
        // it on would double the buffers to hide a latency that does not exist.
        let raw = unsafe {
            ffi::ggml_backend_sched_new(
                handles.as_ptr(),
                bufts.as_ptr(),
                handles.len() as std::os::raw::c_int,
                graph_size,
                false,
                op_offload,
            )
        };
        NonNull::new(raw)
            .map(|raw| Self {
                raw,
                _backends: PhantomData,
            })
            .ok_or(GgmlError::DeviceOutOfMemory)
    }

    /// Measure a graph of this shape and size the buffers for it, without
    /// running it.
    ///
    /// Worth doing once with the **largest** graph the caller will submit: the
    /// buffers then never grow mid-run, which is the difference between a
    /// predictable footprint and an out-of-memory at layer 30.
    pub fn reserve(&self, ctx: &Context, outputs: &[&Tensor<'_>]) -> Result<(), GgmlError> {
        let graph = ctx.build_forward(outputs)?;
        // SAFETY: `graph` lives in `ctx`'s arena and is valid for this call.
        if unsafe { ffi::ggml_backend_sched_reserve(self.raw.as_ptr(), graph) } {
            Ok(())
        } else {
            Err(GgmlError::DeviceOutOfMemory)
        }
    }

    /// Split the graph, assign every node a backend, and give it storage.
    pub fn realize(&self, ctx: &Context, outputs: &[&Tensor<'_>]) -> Result<(), GgmlError> {
        self.realize_with(ctx, outputs, &[])
    }

    /// The same, with `pins` forcing named nodes onto named backends.
    ///
    /// # Why the pins are a parameter and not a method
    ///
    /// Because a `pin()` method is impossible to call correctly. The sequence
    /// ggml requires is **reset, then assign, then allocate** —
    /// `ggml_backend_sched_reset` clears the tensor-to-backend map along with
    /// everything else, so an override set before it is silently erased, and one
    /// set after allocation arrives when the splits are already cut.
    ///
    /// That is not hypothetical. The first version of this file exposed `pin`
    /// separately and its test pinned a node to the GPU, got the CPU, and
    /// reported `left: Some(1), right: Some(0)` — no error anywhere, because an
    /// override ggml does not see is indistinguishable from one it declined.
    /// Taking the pins here makes the wrong order unwriteable, the same way
    /// `place_on_device` enforces alloc-before-upload.
    ///
    /// **This is what `-ngl` and `--tensor-split` become** when the forward pass
    /// eventually builds one graph per pass instead of one per block: a list of
    /// nodes and where they go, not a second code path.
    pub fn realize_with(
        &self,
        ctx: &Context,
        outputs: &[&Tensor<'_>],
        pins: &[(&Tensor<'_>, &Backend)],
    ) -> Result<(), GgmlError> {
        let graph = ctx.build_forward(outputs)?;
        // SAFETY: live scheduler; reset is unconditional and always valid.
        // Without it a second graph inherits the previous one's assignments and
        // ggml reports success while computing the wrong partition.
        unsafe { ffi::ggml_backend_sched_reset(self.raw.as_ptr()) };
        for (node, backend) in pins {
            // SAFETY: live scheduler, live tensor, live backend.
            unsafe {
                ffi::ggml_backend_sched_set_tensor_backend(
                    self.raw.as_ptr(),
                    node.as_raw(),
                    backend.as_raw(),
                )
            };
        }
        // SAFETY: `graph` lives in `ctx`'s arena.
        if unsafe { ffi::ggml_backend_sched_alloc_graph(self.raw.as_ptr(), graph) } {
            Ok(())
        } else {
            Err(GgmlError::DeviceOutOfMemory)
        }
    }

    /// Execute the splits, in order, copying between them.
    ///
    /// Allocates first if [`Self::realize`] was not called — which is correct
    /// only when every input already holds its values, i.e. when the inputs are
    /// pre-bound weights rather than tensors this graph fills in.
    pub fn run(&self, ctx: &Context, outputs: &[&Tensor<'_>]) -> Result<(), GgmlError> {
        if outputs.is_empty() {
            return Ok(());
        }
        let graph = ctx.build_forward(outputs)?;
        let started = std::time::Instant::now();
        // SAFETY: live scheduler and a graph from a live context.
        let status = unsafe { ffi::ggml_backend_sched_graph_compute(self.raw.as_ptr(), graph) };
        crate::backend::timing::add(&crate::backend::timing::COMPUTE_NS, started);
        crate::backend::timing::COMPUTE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if status != 0 {
            return Err(GgmlError::ComputeFailed(status));
        }
        Ok(())
    }

    /// How many maximal single-backend runs the last graph was cut into.
    ///
    /// **The measurement that says whether anything was scheduled.** One split
    /// means every node landed on the same backend and the scheduler did the
    /// work of a graph allocator at the cost of a scheduler.
    ///
    /// **A single-node graph can never report more than one**, however its
    /// operands are placed. Splits partition *nodes*; a leaf that lives on
    /// another backend is copied into the split as an input and does not open a
    /// new one. The first version of this crate's test asserted `>= 2` on
    /// `mul_mat(host, device)` and was wrong about what it was measuring —
    /// which only surfaced when the test stopped skipping.
    pub fn splits(&self) -> usize {
        // SAFETY: valid on a live scheduler; zero before the first graph.
        unsafe { ffi::ggml_backend_sched_get_n_splits(self.raw.as_ptr()).max(0) as usize }
    }

    /// How many tensor copies the splits cost.
    ///
    /// The price of the partition, in the same units the transfer counters use.
    pub fn copies(&self) -> usize {
        // SAFETY: as above.
        unsafe { ffi::ggml_backend_sched_get_n_copies(self.raw.as_ptr()).max(0) as usize }
    }

    /// Bytes the scheduler allocated on `backend`.
    pub fn buffer_bytes(&self, backend: &Backend) -> usize {
        // SAFETY: `backend` is one of the handles this scheduler was built with;
        // ggml returns 0 for one it does not know rather than faulting.
        unsafe { ffi::ggml_backend_sched_get_buffer_size(self.raw.as_ptr(), backend.as_raw()) }
    }

    /// Which backend a node was actually assigned, or `None` before the split.
    ///
    /// Reads the decision back rather than trusting [`Self::pin`] to have
    /// stuck — an override for a node the scheduler cannot honour is silently
    /// ignored, and this is the only way to see that.
    pub fn assignment_of(&self, node: &Tensor<'_>, candidates: &[&Backend]) -> Option<usize> {
        // SAFETY: live scheduler and tensor; null when unassigned.
        let got =
            unsafe { ffi::ggml_backend_sched_get_tensor_backend(self.raw.as_ptr(), node.as_raw()) };
        if got.is_null() {
            return None;
        }
        candidates.iter().position(|b| b.as_raw() == got)
    }
}

impl Drop for Scheduler<'_> {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `ggml_backend_sched_new` and is freed once.
        // The backends it points at are still alive: the lifetime says so.
        unsafe { ffi::ggml_backend_sched_free(self.raw.as_ptr()) };
    }
}
