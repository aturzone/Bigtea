//! **Where does the time in an expert read actually go?**
//!
//! A V4-Flash prefill spends 15.4s of ~31s reading 11.5 GiB of routed experts,
//! an apparent 1.06 GiB/s against an NVMe measured at 2.55 GB/s sequential. The
//! obvious reading is that the drive is the wall. The other reading is that the
//! bytes are being copied several times on their way in and the drive is idle
//! for half of it.
//!
//! This decides between them before any API is redesigned, because this project
//! has already reasoned ahead of measurement three times and been wrong each
//! time (parallel expert reads: slower; contextual sparsity: not present).
//!
//! Every byte of an expert slice is touched **four** times today:
//!
//! | # | where | what |
//! |---|---|---|
//! | 0 | `AlignedBuf::new` | `alloc_zeroed` — a full write of the span |
//! | 1 | `read_exact_at` | the actual disk transfer |
//! | 2 | `read_at` → `.to_vec()` | allocate again, copy |
//! | 3 | `bind_expert_slices` → `extend_from_slice` | copy into the stack |
//! | 4 | `WeightSet::bind` → `Arc<[u8]>::from` | allocate again, copy |
//!
//! Run with `cargo test -p bigtea-model --release --ignored expert_read_cost`.

use std::sync::Arc;
use std::time::Instant;

use bigtea_io::{AlignedBuf, SkewedBuf, ALIGN};
use bigtea_model::Model;

const SHARD: &str = r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf";

/// The tensor the runner reads most: one layer's stacked `ffn_up` experts.
const EXPERTS: &str = "blk.5.ffn_up_exps.weight";

/// Enough slices to be dominated by throughput rather than by the first fault.
const SLICES: u64 = 120;

fn open() -> Option<Model> {
    if !std::path::Path::new(SHARD).exists() {
        eprintln!("skipping: {SHARD} not present");
        return None;
    }
    Model::open_split(SHARD).ok()
}

fn gib(bytes: u64, secs: f64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64 / secs
}

#[test]
#[ignore = "reads gigabytes from a 144 GB container; a benchmark, not a correctness test"]
fn expert_read_cost_by_stage() {
    let Some(model) = open() else { return };
    let loc = model
        .location(EXPERTS)
        .expect("expert tensor present")
        .clone();
    let n_expert = *loc.dims.last().expect("stacked");
    let slice = loc.size / n_expert;
    let total = slice * SLICES;

    eprintln!("\n  {EXPERTS}");
    eprintln!(
        "  {n_expert} experts, slice {:.2} MiB, reading {SLICES} of them ({:.2} GiB)",
        slice as f64 / (1 << 20) as f64,
        total as f64 / (1u64 << 30) as f64
    );
    eprintln!(
        "  slice % {ALIGN} = {}, tensor offset % {ALIGN} = {}",
        slice % ALIGN as u64,
        loc.file_offset % ALIGN as u64
    );
    eprintln!("  io mode: {}\n", model.io_mode());

    // Never read the same expert twice: a repeat would measure the page cache
    // even in direct mode, because the drive has its own.
    let mut expert = 0u64;
    let next = move || {
        expert = (expert + 37) % n_expert;
        expert
    };

    // ---- (a) the whole of today's path, end to end ----
    // read_tensor_range -> extend_from_slice -> Arc::from, exactly as
    // `bind_expert_slices` does it.
    let mut nexta = next;
    let t = Instant::now();
    let mut stack = Vec::with_capacity(total as usize);
    let mut read_only = 0f64;
    for _ in 0..SLICES {
        let off = nexta() * slice;
        let r = Instant::now();
        let got = model.read_tensor_range(EXPERTS, off, slice).expect("read");
        read_only += r.elapsed().as_secs_f64();
        stack.extend_from_slice(&got);
    }
    let shared: Arc<[u8]> = stack.into();
    let today = t.elapsed().as_secs_f64();
    std::hint::black_box(&shared);
    drop(shared);

    // ---- (b) straight into a pre-allocated *aligned* stack ----
    // The obvious destination, and the wrong one: the tensor does not start on
    // a sector boundary, so nothing can land in place.
    let mut nextb = next;
    let t = Instant::now();
    let mut buf = AlignedBuf::new(total as usize);
    let mut copied_a = 0usize;
    for i in 0..SLICES {
        let off = nextb() * slice;
        let at = (i * slice) as usize;
        copied_a += model
            .read_range_into(EXPERTS, off, &mut buf[at..at + slice as usize])
            .expect("read into");
    }
    let into = t.elapsed().as_secs_f64();
    std::hint::black_box(&buf[0]);
    drop(buf);

    // ---- (c) straight into a stack skewed to match the tensor ----
    let skew = SkewedBuf::skew_for(loc.file_offset);
    let mut nextc = next;
    let t = Instant::now();
    let mut buf = SkewedBuf::new(total as usize, skew);
    let mut copied_s = 0usize;
    for i in 0..SLICES {
        let off = nextc() * slice;
        let at = (i * slice) as usize;
        copied_s += model
            .read_range_into(EXPERTS, off, &mut buf[at..at + slice as usize])
            .expect("read into");
    }
    let skewed = t.elapsed().as_secs_f64();
    std::hint::black_box(&buf[0]);

    let pct = |c: usize| c as f64 / total as f64 * 100.0;
    eprintln!(
        "  (a) read_tensor_range + extend + Arc   {today:6.2}s   {:5.2} GiB/s   3 copies",
        gib(total, today)
    );
    eprintln!(
        "        of which the read call itself    {read_only:6.2}s   {:5.2} GiB/s",
        gib(total, read_only)
    );
    eprintln!(
        "  (b) read_range_into, aligned stack     {into:6.2}s   {:5.2} GiB/s   {:.1}% copied",
        gib(total, into),
        pct(copied_a)
    );
    eprintln!(
        "  (c) read_range_into, skew {skew:<4}          {skewed:6.2}s   {:5.2} GiB/s   {:.2}% copied",
        gib(total, skewed),
        pct(copied_s)
    );
    eprintln!(
        "\n  VERDICT: killing the copies is worth {:.2}x on the expert path",
        today / skewed
    );
    eprintln!(
        "           copies are {:.0}% of the time attributed to \"disk\"",
        (1.0 - read_only.min(today) / today) * 100.0
    );

    // The measurement is worthless if the two paths disagree about the bytes.
    let check = model
        .read_tensor_range(EXPERTS, 0, slice.min(1 << 20))
        .expect("read");
    let mut mirror = vec![0u8; check.len()];
    model
        .read_range_into(EXPERTS, 0, &mut mirror)
        .expect("read into");
    assert_eq!(check, mirror, "the two read paths returned different bytes");
}
