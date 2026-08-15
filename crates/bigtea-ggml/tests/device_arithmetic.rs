//! Does the device compute the same numbers the CPU does?
//!
//! This is the milestone the GPU tier actually turns on. Enumerating a card
//! proves the registry works; allocating on it proves the buffer type works;
//! only *matching the CPU element for element* proves the binding is right.
//!
//! The project's standing lesson applies with full force here: a wrong forward
//! pass produces fluent nonsense, never a crash. A device path that uploads to
//! the wrong offset, or reads back a stale buffer, returns numbers — plausible
//! ones — and the first symptom is a model that answers slightly wrongly. So
//! the acceptance test is an exact comparison against the path we already trust.
//!
//! Skips itself when there is no GPU, which is every CI runner.

use bigtea_ggml::{backend, devices, Backend, Context, DeviceKind};

/// Serialise everything that opens a device.
///
/// **The Vulkan backend's device is process-global state, and dropping one
/// invalidates the other's.** Run in parallel, these tests took the whole
/// binary down with
///
/// ```text
/// [Vulkan Loader] ERROR: vkCreateFence: Invalid device
/// exit code: 0xc0000409, STATUS_STACK_BUFFER_OVERRUN
/// ```
///
/// reported as "process didn't exit successfully" rather than as a failing
/// test, with every later result lost — the same shape as the V4-Flash suite's
/// parallel aborts, and solved the same way. The guard is held for the whole
/// test, including the `Backend`'s drop, because the free is half the race.
fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A poisoned lock means an earlier device test panicked; that is already
    // reported, and the rest should still run.
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The device to test on, or `None` if this machine has no discrete GPU.
///
/// Integrated GPUs are excluded for the reason recorded in
/// `research/the-igpu-is-not-a-tier-2026-08-15.md`: this machine has one, it is
/// enumerated *first*, and it is slower than the CPU path. Testing on it would
/// pass while proving nothing about the tier we are building.
fn discrete_gpu() -> Option<usize> {
    devices()
        .ok()?
        .into_iter()
        .position(|d| d.kind == DeviceKind::Gpu)
}

/// A row-major reference matmul, written out rather than borrowed from ggml.
///
/// Comparing ggml-on-device against ggml-on-CPU would catch a binding mistake
/// but not a shared misunderstanding of the layout, and this crate has been
/// bitten by exactly that: `ne[0]` is the fastest dimension, and reading shapes
/// row-major "yields confident nonsense".
fn reference_mul_mat(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += a[i * k + t] * b[j * k + t];
            }
            out[j * m + i] = acc;
        }
    }
    out
}

#[test]
fn device_matmul_matches_the_cpu_elementwise() {
    let _guard = one_at_a_time();
    let Some(index) = discrete_gpu() else {
        eprintln!("skipping: no discrete GPU");
        return;
    };
    let Ok(backend_handle) = Backend::open(index) else {
        eprintln!("skipping: device {index} would not initialise");
        return;
    };

    // ggml's convention: `mul_mat(a[k, m], b[k, n]) -> [m, n]`, and ne[0] is the
    // fastest dimension. Deliberately non-square so a transposed read cannot
    // accidentally agree.
    let (k, m, n) = (4usize, 3usize, 2usize);
    let a: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| 2.0 - (i as f32) * 0.25).collect();

    // `no_alloc`: tensors exist with null data, and the device fills them in.
    // This is the same context mode the zero-copy host path uses, which is why
    // both bindings can share every graph-building routine above them.
    let ctx = Context::new_no_alloc(16 * 1024 * 1024).expect("context");
    let ta = ctx.new_f32_2d(k as i64, m as i64).expect("a");
    let tb = ctx.new_f32_2d(k as i64, n as i64).expect("b");
    let out = ctx.mul_mat(&ta, &tb).expect("mul_mat");

    // One allocation for the whole context, taken *after* the graph tensors
    // exist so intermediates are covered too.
    let buffer = backend_handle.alloc(&ctx).expect("device allocation");
    assert!(buffer.bytes() > 0, "device reported a zero-byte allocation");

    backend::upload_f32(&ta, &a).expect("upload a");
    backend::upload_f32(&tb, &b).expect("upload b");
    backend_handle
        .compute(&ctx, &[&out])
        .expect("device compute");
    let got = backend::download_f32(&out).expect("download");

    let want = reference_mul_mat(&a, &b, k, m, n);
    assert_eq!(got.len(), want.len(), "shape mismatch: {got:?} vs {want:?}");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "element {i}: device {g}, reference {w}\n  device {got:?}\n  want   {want:?}"
        );
    }
}

#[test]
fn a_round_trip_through_the_device_preserves_the_bytes() {
    let _guard = one_at_a_time();
    let Some(index) = discrete_gpu() else {
        eprintln!("skipping: no discrete GPU");
        return;
    };
    let Ok(backend_handle) = Backend::open(index) else {
        eprintln!("skipping: device {index} would not initialise");
        return;
    };

    // Separated from the matmul on purpose. If both fail, this one says whether
    // the transfer or the arithmetic is at fault — and a stale-readback bug
    // looks exactly like a wrong kernel from the other test alone.
    let ctx = Context::new_no_alloc(4 * 1024 * 1024).expect("context");
    let t = ctx.new_f32_2d(8, 4).expect("tensor");
    let _buffer = backend_handle.alloc(&ctx).expect("device allocation");

    let values: Vec<f32> = (0..32).map(|i| i as f32 * -1.5 + 0.25).collect();
    backend::upload_f32(&t, &values).expect("upload");
    let got = backend::download_f32(&t).expect("download");

    assert_eq!(got, values, "device round trip altered the data");
}

#[test]
fn opening_a_device_that_does_not_exist_is_an_error() {
    let _guard = one_at_a_time();
    // The registry is small; this index cannot be real. Worth asserting because
    // the failure mode of a missing bounds check here is a null dereference
    // inside ggml rather than a Rust error.
    let far_past_the_end = 9_999;
    assert!(
        Backend::open(far_past_the_end).is_err(),
        "opening device {far_past_the_end} should fail, not succeed"
    );
}
