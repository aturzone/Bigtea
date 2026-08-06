//! Reading weights from disk, on purpose rather than by accident.
//!
//! # Why not just `mmap` the file and let the OS handle it
//!
//! Because the OS is optimising for the wrong thing, and it lies to you about
//! it. Three concrete failures, all of which this crate exists to avoid:
//!
//! 1. **The page cache double-buffers.** A cached read costs RAM for the OS
//!    copy *and* our copy. On a machine where RAM is the binding constraint,
//!    paying twice for the same bytes is the one thing we cannot afford.
//! 2. **Measured hit rates become fiction.** With a container smaller than RAM
//!    the kernel caches everything, so any cache-hit-rate we report is really
//!    the kernel's — a number that evaporates the moment the model is larger
//!    than RAM, which is the only case we target.
//! 3. **A paged "hit" can be slower than the read it replaced.** Once the OS is
//!    under memory pressure it starts evicting and faulting, and a fault is a
//!    disk read we did not schedule, at a moment we did not choose.
//!
//! So: bypass the cache, own the memory, and schedule the reads ourselves.
//! That is [`DirectFile`].
//!
//! # The alignment tax
//!
//! Cache-bypassing I/O transfers by DMA straight into our pages, so the buffer
//! address, the file offset and the read length must all be sector multiples.
//! Tensors are not so obliging. [`DirectFile::read_at`] therefore reads the
//! aligned superset of the requested range and returns the exact slice from
//! within it — the caller never has to think about sectors.
//!
//! # Graceful degradation
//!
//! Direct I/O is not available everywhere: some filesystems refuse it, and
//! network mounts commonly do. [`DirectFile::open`] falls back to a normal
//! buffered handle and records that it did, rather than failing. Correctness
//! never depends on the fast path being available.

mod aligned;
mod direct;

pub use aligned::{align_down, align_up, AlignedBuf, SkewedBuf, ALIGN};
pub use direct::{DirectFile, IoMode};
