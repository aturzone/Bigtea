//! Does a graph with operands in two different buffers compute the right answer?
//!
//! This is the test the GPU tier stalled on. `mixed-residency-segfaults` records
//! three access violations in one session from exactly this shape — one host
//! tensor, one device tensor, one graph — and the response at the time was to
//! forbid it and ship a device-only slice. Five declined flags (`--split-mode`,
//! `--tensor-split`, `--op-offload`, `-ngl`, `--n-gpu-layers`) are all downstream
//! of that refusal.
//!
//! Two things have to be true and only one of them is the answer:
//!
//! 1. the numbers match a reference computed by hand, and
//! 2. **the scheduler actually split the graph.** A partition of one is a
//!    scheduler that found everything on one backend, which would pass check (1)
//!    while proving nothing. `splits()` is asserted for that reason.
//!
//! The CPU-only cases run everywhere, including CI, and they are not filler:
//! `HostBuffer` — the buffer identity that makes a zero-copy host tensor legal
//! to copy *from* — is where the segfault actually lived, and it is testable
//! without a card.
#![cfg(have_ggml)]

use chaos_ggml::sched::{AlignedBytes, TENSOR_ALIGNMENT};
use chaos_ggml::{backend, devices, Backend, Context, DeviceKind, HostBuffer, Scheduler};

/// Serialise everything that opens a device — see `device_arithmetic.rs`.
///
/// The Vulkan device is process-global and dropping one invalidates another's.
/// Two test binaries cannot share a lock, so these files must not both hold a
/// device at once; cargo runs them sequentially unless told otherwise, and
/// `heavy()`-style cross-binary locking is reserved for the suites that need it.
fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn discrete_gpu() -> Option<usize> {
    devices()
        .ok()?
        .into_iter()
        .position(|d| d.kind == DeviceKind::Gpu)
}

/// `mul_mat(a[k, m], b[k, n]) -> [m, n]`, written out rather than borrowed.
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

/// f32s in an allocation ggml will accept as a buffer.
///
/// **Not a `Vec<u8>`, and that is the point.** ggml asserts 32-byte alignment on
/// any pointer handed to `cpu_buffer_from_ptr`; a byte vector is aligned to 1
/// and the first version of these tests aborted the binary on it.
fn aligned(values: &[f32]) -> AlignedBytes {
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    AlignedBytes::from_slice(&raw).expect("aligned allocation")
}

/// **The one that was undefined behaviour before this file existed.**
///
/// `a` lives in host memory the test owns; `b` lives on the card. One graph
/// consumes both, and a second node is forced back onto the host. Without a
/// scheduler this shape is the access violation; with one it is two splits and
/// the right numbers.
///
/// # Why the graph has two nodes and not one
///
/// Because splits partition **nodes**. The first version of this test built
/// `mul_mat(host, device)` — a single node — and asserted `splits() >= 2`,
/// which cannot happen however the operands are placed: a leaf on another
/// backend is copied in as an input, not run as its own split. It passed
/// anyway, because the ggml build it ran against had no Vulkan archive and the
/// test was skipping. **Two failures, and only one of them was the assertion.**
#[test]
fn a_graph_spanning_host_and_device_computes_and_splits() {
    let _guard = one_at_a_time();
    let Some(index) = discrete_gpu() else {
        eprintln!("skipping: no discrete GPU");
        return;
    };
    let Ok(gpu) = Backend::open(index) else {
        eprintln!("skipping: device {index} would not initialise");
        return;
    };
    let cpu = Backend::cpu().expect("cpu backend");
    cpu.set_threads(1);

    let (k, m, n) = (4usize, 3usize, 2usize);
    let a: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| 2.0 - (i as f32) * 0.25).collect();
    let bias: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.125).collect();

    let ctx = Context::new_no_alloc(16 * 1024 * 1024).expect("context");
    let ta = ctx.new_f32_2d(k as i64, m as i64).expect("a");
    let tb = ctx.new_f32_2d(k as i64, n as i64).expect("b");
    let tbias = ctx.new_f32_2d(m as i64, n as i64).expect("bias");
    let product = ctx.mul_mat(&ta, &tb).expect("mul_mat");
    let out = ctx.add(&product, &tbias).expect("add");

    // `a` and the bias get host buffers. This is the whole fix: the bytes do
    // not move, but the tensors now name the buffer they live in, so a split
    // can copy them.
    let mut a_bytes = aligned(&a);
    let host_a = HostBuffer::wrap(&mut a_bytes).expect("host buffer a");
    host_a.attach(&ta, 0).expect("attach a");
    let mut bias_bytes = aligned(&bias);
    let host_bias = HostBuffer::wrap(&mut bias_bytes).expect("host buffer bias");
    host_bias.attach(&tbias, 0).expect("attach bias");

    // The matmul on the card, the add back on the host: a partition ggml would
    // not choose on its own, which is exactly what makes it a test of pinning
    // AND of the copy between splits.
    let backends = [&gpu, &cpu];
    let sched = Scheduler::new(&backends, 2048, false).expect("scheduler");
    sched
        .realize_with(&ctx, &[&out], &[(&product, &gpu), (&out, &cpu)])
        .expect("realize");

    // `b` is device-side and is filled after allocation, per the ordering rule.
    backend::upload_f32(&tb, &b).expect("upload b");

    sched.run(&ctx, &[&out]).expect("run");

    let splits = sched.splits();
    assert!(
        splits >= 2,
        "the graph was not split ({splits}); a partition of one means every node          landed on the same backend and this test proves nothing"
    );
    assert!(
        sched.copies() > 0,
        "two splits with no copies between them is not a partition"
    );

    let product_want = reference_mul_mat(&a, &b, k, m, n);
    let want: Vec<f32> = product_want.iter().zip(&bias).map(|(p, c)| p + c).collect();
    let got = backend::download_f32(&out).expect("download");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "element {i}: device+host graph gave {g}, reference {w}"
        );
    }
}

/// The host side alone, so CI without a card still covers the thing that faulted.
///
/// One backend, so there is nothing to split — the point here is that a tensor
/// bound through `HostBuffer` is a legal graph operand at all, and that the
/// bytes were adopted rather than copied.
#[test]
fn a_host_buffer_binds_without_copying_and_computes() {
    let cpu = Backend::cpu().expect("cpu backend");
    cpu.set_threads(1);

    let (k, m, n) = (4usize, 3usize, 2usize);
    let a: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| 2.0 - (i as f32) * 0.25).collect();

    let ctx = Context::new_no_alloc(16 * 1024 * 1024).expect("context");
    let ta = ctx.new_f32_2d(k as i64, m as i64).expect("a");
    let tb = ctx.new_f32_2d(k as i64, n as i64).expect("b");
    let out = ctx.mul_mat(&ta, &tb).expect("mul_mat");

    // Both operands in ONE host buffer at different offsets — the layout a
    // memory-mapped container actually has, rather than one buffer per tensor.
    // `a` is 48 bytes; `b` must start on a 32-byte boundary, so the offset is
    // rounded up and the gap is padding. Packing them tight is what a container
    // does NOT do either -- GGUF pads tensor data to `general.alignment`.
    let a_raw: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_raw: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_offset = a_raw.len().div_ceil(TENSOR_ALIGNMENT) * TENSOR_ALIGNMENT;
    let mut bytes = AlignedBytes::zeroed(b_offset + b_raw.len()).expect("aligned");
    bytes[..a_raw.len()].copy_from_slice(&a_raw);
    bytes[b_offset..b_offset + b_raw.len()].copy_from_slice(&b_raw);
    let host = HostBuffer::wrap(&mut bytes).expect("host buffer");
    host.attach(&ta, 0).expect("attach a");
    host.attach(&tb, b_offset).expect("attach b");

    let backends = [&cpu];
    let sched = Scheduler::new(&backends, 2048, false).expect("scheduler");
    sched.realize(&ctx, &[&out]).expect("realize");
    sched.run(&ctx, &[&out]).expect("run");

    let got = backend::download_f32(&out).expect("download");
    let want = reference_mul_mat(&a, &b, k, m, n);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "element {i}: host-buffer graph gave {g}, reference {w}"
        );
    }
}

/// An attach that would run past the end of the buffer is refused, not written.
///
/// The failure it prevents is silent: ggml would take the address, the graph
/// would build, and the read would come from whatever followed the allocation.
#[test]
fn attaching_past_the_end_is_refused() {
    let _cpu = Backend::cpu().expect("cpu backend");
    let ctx = Context::new_no_alloc(1024 * 1024).expect("context");
    let t = ctx.new_f32_2d(4, 3).expect("t"); // 48 bytes

    let mut bytes = AlignedBytes::zeroed(32).expect("aligned");
    let host = HostBuffer::wrap(&mut bytes).expect("host buffer");

    assert!(host.attach(&t, 0).is_err(), "48 bytes accepted into 32");
    assert!(host.attach(&t, 64).is_err(), "offset past the end accepted");
    assert!(host.attach(&t, 8).is_err(), "unaligned offset accepted");
}

/// Misaligned host memory is an error, not an abort.
///
/// **The regression this guards is a process death, not a wrong answer.** ggml
/// asserts on the pointer, and an assert is not catchable: the first run of the
/// buffer test died with `GGML_ASSERT ... "buffer pointer must be aligned"` and
/// took three passing tests' results with it. So the check has to be on our
/// side, and it has to stay.
#[test]
fn misaligned_host_memory_is_refused_rather_than_fatal() {
    // Deliberately step one byte into an aligned allocation.
    let mut backing = AlignedBytes::zeroed(128).expect("aligned");
    let skewed = &mut backing[1..];
    let err = HostBuffer::wrap(skewed).err();
    match err {
        Some(chaos_ggml::GgmlError::Misaligned { required, .. }) => {
            assert_eq!(required, TENSOR_ALIGNMENT);
        }
        Some(other) => panic!("wrong error: {other}"),
        None => panic!("a misaligned pointer was accepted; ggml would have aborted"),
    }
}

/// A scheduler over zero backends is an error, not an abort.
#[test]
fn a_scheduler_needs_at_least_one_backend() {
    let backends: [&Backend; 0] = [];
    assert!(Scheduler::new(&backends, 2048, false).is_err());
}

/// Pinning moves a node, and the assignment can be read back.
///
/// **This is what `-ngl` becomes** once the forward pass builds one graph per
/// pass — a list of nodes and where they go, not a second code path. So the
/// thing that has to be true is that an override sticks and that a caller can
/// tell when it did not: an override ggml declines to honour is silently
/// ignored, which is why the assignment is read rather than assumed.
///
/// The first version called a standalone `pin()` before `realize()`, and
/// `realize()` began with `ggml_backend_sched_reset` — which clears the
/// tensor-to-backend map. The pin was erased and the node landed on the CPU,
/// reported only as `left: Some(1), right: Some(0)`. The pins are a parameter
/// of `realize_with` now, so that order cannot be written.
#[test]
fn pinning_a_node_is_honoured() {
    let _guard = one_at_a_time();
    let Some(index) = discrete_gpu() else {
        eprintln!("skipping: no discrete GPU");
        return;
    };
    let Ok(gpu) = Backend::open(index) else {
        eprintln!("skipping: device {index} would not initialise");
        return;
    };
    let cpu = Backend::cpu().expect("cpu backend");
    cpu.set_threads(1);

    let (k, m, n) = (4usize, 3usize, 2usize);
    let a: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| 2.0 - (i as f32) * 0.25).collect();

    let ctx = Context::new_no_alloc(16 * 1024 * 1024).expect("context");
    let ta = ctx.new_f32_2d(k as i64, m as i64).expect("a");
    let tb = ctx.new_f32_2d(k as i64, n as i64).expect("b");
    let out = ctx.mul_mat(&ta, &tb).expect("mul_mat");

    let mut a_bytes = aligned(&a);
    let host = HostBuffer::wrap(&mut a_bytes).expect("host buffer");
    host.attach(&ta, 0).expect("attach a");
    let mut b_bytes = aligned(&b);
    let host_b = HostBuffer::wrap(&mut b_bytes).expect("host buffer b");
    host_b.attach(&tb, 0).expect("attach b");

    let backends = [&gpu, &cpu];
    let sched = Scheduler::new(&backends, 2048, false).expect("scheduler");
    // Both operands are host-resident, so without an override the matmul stays
    // on the CPU. Pinned to the card, it must move.
    sched
        .realize_with(&ctx, &[&out], &[(&out, &gpu)])
        .expect("realize");

    let placed = sched.assignment_of(&out, &backends);
    assert_eq!(
        placed,
        Some(0),
        "pinned to the GPU (index 0 of the candidates) but landed at {placed:?}"
    );

    sched.run(&ctx, &[&out]).expect("run");
    let got = backend::download_f32(&out).expect("download");
    let want = reference_mul_mat(&a, &b, k, m, n);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-4, "element {i}: {g} vs {w}");
    }
}

/// **A minimal stand-in for the attention graph, which aborts in the runner.**
///
/// `--op-offload` dies on the third graph with
/// `ggml-alloc.c:623 GGML_ASSERT(buffer_id >= 0)` — a node the split left
/// unassigned. Attention is the only graph full of views, so this builds the
/// same shape and nothing else: a matmul, a permute, and a `cont`, across two
/// backends.
///
/// `#[ignore]`d because a ggml assert **aborts the process** and would take
/// every other result in this binary with it. Run it on its own:
///
/// ```text
/// cargo test --release -p chaos-ggml --test scheduler -- --ignored views
/// ```
#[test]
#[ignore = "aborts the whole binary if the bug reproduces; run alone"]
fn views_across_a_split_are_assigned_a_backend() {
    let _guard = one_at_a_time();
    let Some(index) = discrete_gpu() else {
        eprintln!("skipping: no discrete GPU");
        return;
    };
    let Ok(gpu) = Backend::open(index) else {
        eprintln!("skipping: device {index} would not initialise");
        return;
    };
    let cpu = Backend::cpu().expect("cpu backend");
    cpu.set_threads(1);

    let (k, m, n) = (4usize, 4usize, 4usize);
    let a: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| 2.0 - (i as f32) * 0.25).collect();

    let ctx = Context::new_no_alloc(16 * 1024 * 1024).expect("context");
    let ta = ctx.new_f32_2d(k as i64, m as i64).expect("a");
    let tb = ctx.new_f32_2d(k as i64, n as i64).expect("b");
    let product = ctx.mul_mat(&ta, &tb).expect("mul_mat");
    // The shape attention actually has: a 3-D reshape, a permute, and a cont.
    let three = ctx.reshape_3d(&product, 2, 2, n as i64).expect("reshape");
    let moved = ctx.permute(&three, [0, 2, 1, 3]).expect("permute");
    let out = ctx.cont(&moved).expect("cont");

    let mut a_bytes = aligned(&a);
    let host_a = HostBuffer::wrap(&mut a_bytes).expect("host buffer a");
    host_a.attach(&ta, 0).expect("attach a");
    let mut b_bytes = aligned(&b);
    let host_b = HostBuffer::wrap(&mut b_bytes).expect("host buffer b");
    host_b.attach(&tb, 0).expect("attach b");

    let backends = [&gpu, &cpu];
    let sched = Scheduler::new(&backends, 2048, true).expect("scheduler");
    sched.realize(&ctx, &[&out]).expect("realize");
    sched.run(&ctx, &[&out]).expect("run");

    let got = backend::download_f32(&out).expect("download");
    assert_eq!(
        got.len(),
        m * n,
        "wrong element count out of the view chain"
    );
}

/// The other half of the attention graph: the fused kernel and its F16 mask.
///
/// Views alone schedule fine (above), so if `--op-offload`'s abort reproduces
/// anywhere small it is here. Same `#[ignore]` reasoning: an assert kills the
/// binary.
///
/// ```text
/// cargo test --release -p chaos-ggml --test scheduler -- --ignored flash
/// ```
#[test]
#[ignore = "aborts the whole binary if the bug reproduces; run alone"]
fn a_flash_attention_graph_schedules() {
    let _guard = one_at_a_time();
    let Some(index) = discrete_gpu() else {
        eprintln!("skipping: no discrete GPU");
        return;
    };
    let Ok(gpu) = Backend::open(index) else {
        eprintln!("skipping: device {index} would not initialise");
        return;
    };
    let cpu = Backend::cpu().expect("cpu backend");
    cpu.set_threads(1);

    // The runner's shapes in miniature: head_dim 32, 2 heads, 4 positions.
    let (head_dim, n_head, n_tok) = (32i64, 2i64, 4i64);
    let ctx = Context::new_no_alloc(32 * 1024 * 1024).expect("context");
    let q = ctx.new_f32_3d(head_dim, n_head, n_tok).expect("q");
    let k = ctx.new_f16_3d(head_dim, n_head, n_tok).expect("k");
    let v = ctx.new_f16_3d(head_dim, n_head, n_tok).expect("v");
    // ggml wants [head_dim, n_tok, n_head] for the fused kernel.
    let qp = ctx.permute(&q, [0, 2, 1, 3]).expect("q permute");
    let qc = ctx.cont(&qp).expect("q cont");
    let kp = ctx.permute(&k, [0, 2, 1, 3]).expect("k permute");
    let kc = ctx.cont(&kp).expect("k cont");
    let vp = ctx.permute(&v, [0, 2, 1, 3]).expect("v permute");
    let vc = ctx.cont(&vp).expect("v cont");
    // **F16 and contiguous**, which the fused kernel asserts; the only values
    // are 0 and -inf so the bit patterns go in directly.
    let mask = ctx.new_f16_3d(n_tok, n_tok, 1).expect("mask");
    let attn = ctx
        .flash_attn_ext(&qc, &kc, &vc, &mask, 0.125, 0.0)
        .expect("flash_attn_ext");
    let out = ctx.cont(&attn).expect("out cont");

    // **The whole fix.** Without these the scheduler cannot place a bare leaf
    // and `ggml_gallocr_allocate_node` gets backend -1, which aborts.
    for t in [&q, &k, &v, &mask] {
        t.set_input();
    }

    let backends = [&gpu, &cpu];
    let sched = Scheduler::new(&backends, 2048, true).expect("scheduler");
    sched.realize(&ctx, &[&out]).expect("realize");

    // Inputs after allocation, per the ordering rule.
    let zeros_q = vec![0.0f32; (head_dim * n_head * n_tok) as usize];
    backend::upload_f32(&q, &zeros_q).expect("upload q");
    let zeros_kv = vec![0u8; (head_dim * n_head * n_tok) as usize * 2];
    backend::upload(&k, &zeros_kv).expect("upload k");
    backend::upload(&v, &zeros_kv).expect("upload v");
    let zeros_mask = vec![0u8; (n_tok * n_tok) as usize * 2];
    backend::upload(&mask, &zeros_mask).expect("upload mask");

    sched.run(&ctx, &[&out]).expect("run");
    let got = backend::download_f32(&out).expect("download");
    assert_eq!(got.len(), (head_dim * n_head * n_tok) as usize);
}
