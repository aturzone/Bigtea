//! Page-aligned heap buffers.
//!
//! Cache-bypassing I/O (`FILE_FLAG_NO_BUFFERING` on Windows, `O_DIRECT` on
//! Linux) does not merely prefer aligned buffers — it **fails** without them.
//! The DMA engine transfers straight into the pages we hand it, so the address,
//! the file offset and the length must all be sector multiples. `Vec<u8>` gives
//! no alignment guarantee beyond one byte, so it cannot be used here.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::{Deref, DerefMut};

/// Alignment used for all direct I/O.
///
/// 4096 covers every common sector size: 512-byte and 512e drives (4096 is a
/// multiple of 512) and native 4K drives alike. Over-aligning is free; under-
/// aligning fails the read outright.
pub const ALIGN: usize = 4096;

/// Round `n` up to the next multiple of [`ALIGN`].
#[inline]
pub const fn align_up(n: u64) -> u64 {
    n.div_ceil(ALIGN as u64) * ALIGN as u64
}

/// Round `n` down to a multiple of [`ALIGN`].
#[inline]
pub const fn align_down(n: u64) -> u64 {
    n & !(ALIGN as u64 - 1)
}

/// A heap buffer whose address and length are both [`ALIGN`]-multiples.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: `AlignedBuf` owns its allocation exclusively and holds no thread-
// affine state, so moving it between threads is sound.
unsafe impl Send for AlignedBuf {}
// SAFETY: `&AlignedBuf` only permits reads of plain bytes.
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Allocate a zeroed buffer of at least `size` bytes, rounded up to a
    /// whole number of sectors.
    ///
    /// # Panics
    /// If the rounded size overflows, or the allocator fails.
    pub fn new(size: usize) -> Self {
        let len = align_up(size as u64) as usize;
        if len == 0 {
            return AlignedBuf {
                ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                len: 0,
            };
        }
        let layout = Layout::from_size_align(len, ALIGN).expect("valid layout");
        // SAFETY: `layout` has non-zero size and a valid power-of-two align.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        AlignedBuf { ptr, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True when the allocation really is sector-aligned. Direct I/O depends
    /// on this, so it is worth being able to assert it.
    pub fn is_aligned(&self) -> bool {
        self.ptr as usize % ALIGN == 0
    }
}

/// A heap buffer whose start address is congruent to `skew` modulo [`ALIGN`].
///
/// # Why anyone would want a *deliberately* misaligned buffer
///
/// Direct I/O can only transfer when the file offset and the memory address
/// agree modulo the sector size. GGUF pads tensor data to `general.alignment`,
/// which defaults to **32** — so a real container puts the experts of
/// `blk.5.ffn_up_exps.weight` at file offsets ≡ 2816 (mod 4096). Reading those
/// into an aligned buffer can never be direct: every byte has to bounce through
/// a scratch buffer and be copied into place.
///
/// Skewing the destination by the same 2816 makes the residues match, and the
/// drive writes straight into the caller's memory. The skew is not a hack
/// around alignment — it is alignment, measured from the file rather than from
/// zero.
///
/// Because every expert slice in a stacked tensor is the same size, they all
/// share one residue, so a single skew serves the whole stack.
pub struct SkewedBuf {
    inner: AlignedBuf,
    skew: usize,
    len: usize,
}

impl SkewedBuf {
    /// A buffer of `len` bytes whose first byte sits at an address ≡ `skew`.
    ///
    /// # Panics
    /// If `skew >= ALIGN`, which would mean the caller computed it wrong
    /// rather than merely chose it oddly.
    pub fn new(len: usize, skew: usize) -> Self {
        assert!(skew < ALIGN, "skew {skew} must be less than one sector");
        // One extra sector so a direct read of the aligned middle, which may
        // start before the view does, always has room.
        let inner = AlignedBuf::new(skew + len + ALIGN);
        SkewedBuf { inner, skew, len }
    }

    /// The skew the drive must see for reads into this buffer to be direct:
    /// `file_offset % ALIGN`.
    pub fn skew_for(file_offset: u64) -> usize {
        (file_offset % ALIGN as u64) as usize
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True when the view really does start at the requested residue.
    pub fn has_skew(&self, skew: usize) -> bool {
        self[..].as_ptr() as usize % ALIGN == skew
    }
}

impl Deref for SkewedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.inner[self.skew..self.skew + self.len]
    }
}

impl DerefMut for SkewedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        let (skew, len) = (self.skew, self.len);
        &mut self.inner[skew..skew + len]
    }
}

impl std::fmt::Debug for SkewedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkewedBuf")
            .field("len", &self.len)
            .field("skew", &self.skew)
            .finish()
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        let layout = Layout::from_size_align(self.len, ALIGN).expect("valid layout");
        // SAFETY: `ptr` came from `alloc_zeroed` with this exact layout and
        // has not been freed.
        unsafe { dealloc(self.ptr, layout) };
    }
}

impl Deref for AlignedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` is valid for `len` initialised bytes for our lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: as above, and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuf")
            .field("len", &self.len)
            .field("aligned", &self.is_aligned())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_are_sector_aligned() {
        for size in [1usize, 100, 4095, 4096, 4097, 1 << 20] {
            let buf = AlignedBuf::new(size);
            assert!(buf.is_aligned(), "size {size} produced an unaligned buffer");
            assert!(buf.len() >= size);
            assert_eq!(buf.len() % ALIGN, 0, "length must be a sector multiple");
        }
    }

    #[test]
    fn buffers_start_zeroed_and_are_writable() {
        let mut buf = AlignedBuf::new(8192);
        assert!(buf.iter().all(|&b| b == 0));
        buf[0] = 0xAB;
        buf[8191] = 0xCD;
        assert_eq!(buf[0], 0xAB);
        assert_eq!(buf[8191], 0xCD);
    }

    #[test]
    fn zero_sized_buffer_is_safe() {
        let buf = AlignedBuf::new(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(buf.iter().next().is_none());
    }

    #[test]
    fn alignment_helpers_round_correctly() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 4096);
        assert_eq!(align_up(4096), 4096);
        assert_eq!(align_up(4097), 8192);
        assert_eq!(align_down(0), 0);
        assert_eq!(align_down(4095), 0);
        assert_eq!(align_down(4096), 4096);
        assert_eq!(align_down(8191), 4096);
    }
}
