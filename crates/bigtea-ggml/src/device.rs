//! What compute devices does this build of `ggml` actually see?
//!
//! This is the first half of the GPU tier and deliberately the *only* half in
//! this module: it enumerates and it reports. It allocates nothing on a device
//! and binds nothing, because binding is a second memory design and belongs in
//! its own file with its own tests.
//!
//! # Why enumeration is worth its own commit
//!
//! `research/gpu-the-card-works-vulkan-not-cuda-2026-08-15.md` measured this
//! machine's card through *llama.cpp's* binary: 25.6x prefill on Qwen3-4B. That
//! is the precondition, not the result. Until our own process prints the device
//! name, "the card works" is a statement about someone else's executable.
//!
//! # The enum is transcribed, not remembered
//!
//! `ggml_backend_dev_type` has **five** variants and an integrated GPU is its
//! own kind, sitting at index 2:
//!
//! ```text
//! 0 CPU   1 GPU   2 IGPU   3 ACCEL   4 META
//! ```
//!
//! An older ggml had three (`CPU`, `GPU`, `ACCEL`), and writing that from
//! memory puts `ACCEL` where `IGPU` now is. On *this* machine that is not a
//! cosmetic error: there is a real integrated device, Vulkan enumerates it
//! first, and `research/the-igpu-is-not-a-tier-2026-08-15.md` measured it at
//! **0.48x the CPU on prefill**. A wrong enum would select it and report a GPU
//! tier that runs at half the speed of the path it replaced.

// `GgmlError` is named fully qualified at its one use below: importing it here
// makes the import itself dead in a build that HAS ggml, since the only
// mention is in the `not(have_ggml)` arm.
use crate::Result;

/// What kind of device `ggml` says this is.
///
/// `Other` carries the raw value rather than collapsing to a catch-all: a new
/// ggml adding a sixth kind should be visible as an unknown number, not
/// silently classified as something it is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    /// Dedicated memory. The only kind worth offloading to on this machine.
    Gpu,
    /// Integrated, using host memory. **Measured slower than the CPU here.**
    IGpu,
    /// Meant to be used *with* the CPU backend — BLAS, AMX.
    Accel,
    /// A wrapper over several devices for tensor parallelism.
    Meta,
    Other(i32),
}

impl DeviceKind {
    fn from_raw(v: i32) -> Self {
        match v {
            0 => Self::Cpu,
            1 => Self::Gpu,
            2 => Self::IGpu,
            3 => Self::Accel,
            4 => Self::Meta,
            other => Self::Other(other),
        }
    }
}

/// One device, as `ggml` describes it.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Backend-scoped name, e.g. `Vulkan0`.
    pub name: String,
    /// Human description, e.g. `NVIDIA GeForce RTX 3050 6GB Laptop GPU`.
    pub description: String,
    pub kind: DeviceKind,
    pub free_bytes: usize,
    pub total_bytes: usize,
}

impl DeviceInfo {
    pub fn free_gib(&self) -> f64 {
        self.free_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn total_gib(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Was this build linked against a Vulkan-enabled `ggml`?
///
/// Decided by `build.rs` from the presence of `ggml-vulkan.a` beside the other
/// archives, not by a cargo feature — see the note there for why.
pub fn vulkan_available() -> bool {
    cfg!(have_vulkan)
}

#[cfg(have_ggml)]
mod ffi {
    use std::os::raw::{c_char, c_int, c_void};

    /// `ggml_backend_dev_t` — an opaque pointer to a registry-owned device.
    /// Never freed by us: the registry outlives the process.
    pub type DevT = *mut c_void;

    // Transcribed together from one revision of `ggml/include/ggml-backend.h`,
    // for the reason the type-traits block above states: a wrong FFI signature
    // is silent corruption, not a compile error.
    extern "C" {
        pub fn ggml_backend_dev_count() -> usize;
        pub fn ggml_backend_dev_get(index: usize) -> DevT;
        pub fn ggml_backend_dev_name(device: DevT) -> *const c_char;
        pub fn ggml_backend_dev_description(device: DevT) -> *const c_char;
        pub fn ggml_backend_dev_memory(device: DevT, free: *mut usize, total: *mut usize);
        pub fn ggml_backend_dev_type(device: DevT) -> c_int;
    }
}

/// Every device this `ggml` has registered, in its own order.
///
/// Always non-empty when ggml is present: the CPU backend is always registered,
/// which is what makes this callable in CI on a machine with no accelerator.
pub fn devices() -> Result<Vec<DeviceInfo>> {
    #[cfg(not(have_ggml))]
    {
        Err(crate::GgmlError::Unavailable)
    }
    #[cfg(have_ggml)]
    {
        // SAFETY: the device registry is initialised on first use inside ggml
        // and lives for the process. Every call below takes a pointer that came
        // from `ggml_backend_dev_get` in this same loop, and none of them
        // transfer ownership.
        let count = unsafe { ffi::ggml_backend_dev_count() };
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let dev = unsafe { ffi::ggml_backend_dev_get(i) };
            if dev.is_null() {
                continue;
            }
            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: both pointers address live locals for the call's duration.
            unsafe { ffi::ggml_backend_dev_memory(dev, &mut free, &mut total) };
            out.push(DeviceInfo {
                name: unsafe { cstr(ffi::ggml_backend_dev_name(dev)) },
                description: unsafe { cstr(ffi::ggml_backend_dev_description(dev)) },
                kind: DeviceKind::from_raw(unsafe { ffi::ggml_backend_dev_type(dev) }),
                free_bytes: free,
                total_bytes: total,
            });
        }
        Ok(out)
    }
}

/// The device to offload to, or `None` if there is nothing worth offloading to.
///
/// **Integrated GPUs are excluded on purpose, not by oversight.** This machine
/// has one, Vulkan enumerates it *before* the discrete card, and it has more
/// free memory (7387 MiB against 5233) — so "pick the GPU with the most memory"
/// selects it. It is also 0.48x the CPU on prefill and 0.51x on generation,
/// because it has no matrix cores and shares the DRAM the CPU path already
/// saturates. A UMA device removes the copy, not the bottleneck.
///
/// Among true GPUs, most free memory wins: this engine's constraint is capacity.
pub fn best_offload_device() -> Result<Option<DeviceInfo>> {
    let mut gpus: Vec<_> = devices()?
        .into_iter()
        .filter(|d| d.kind == DeviceKind::Gpu)
        .collect();
    gpus.sort_by_key(|d| std::cmp::Reverse(d.free_bytes));
    Ok(gpus.into_iter().next())
}

#[cfg(have_ggml)]
/// # Safety
/// `p` must be NUL-terminated and valid for reads, or null.
unsafe fn cstr(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_device_reports_a_name_and_a_kind() {
        let Ok(list) = devices() else {
            return; // built without ggml
        };
        // The CPU backend is always registered, so an empty list means the
        // registry did not initialise — which is a real failure, not an
        // accelerator-less machine.
        assert!(!list.is_empty(), "ggml registered no devices at all");
        for d in &list {
            assert!(!d.name.is_empty(), "device with no name: {d:?}");
            assert!(
                !matches!(d.kind, DeviceKind::Other(_)),
                "unknown device kind {:?} — ggml's enum grew, check device.rs \
                 against ggml-backend.h before trusting any selection",
                d.kind
            );
        }
    }

    #[test]
    fn offload_target_is_never_an_integrated_gpu() {
        let Ok(picked) = best_offload_device() else {
            return;
        };
        if let Some(d) = picked {
            assert_eq!(
                d.kind,
                DeviceKind::Gpu,
                "selected {d:?} — an integrated GPU measured 0.48x the CPU on \
                 this machine and must never be chosen as an offload target"
            );
        }
    }

    #[test]
    fn a_cpu_device_is_always_present() {
        let Ok(list) = devices() else {
            return;
        };
        assert!(
            list.iter().any(|d| d.kind == DeviceKind::Cpu),
            "no CPU device registered; devices: {list:?}"
        );
    }
}
