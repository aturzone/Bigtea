//! Does a CPU graph allocator actually reuse buffers, and still give the right
//! answer?
//!
//! **An example, not a test.** `ggml_gallocr` on the host path is new here, and
//! ggml aborts rather than returning errors — a bad plan, a tensor written
//! before it has storage, or a context with no room for the work buffer all take
//! the process down. A crash here costs one `cargo run`.
//!
//! # What it has to prove
//!
//! A `Context`'s own arena allocates every tensor in a graph and frees none of
//! them, so a hundred-node graph pays for a hundred tensors. Almost all of them
//! are dead one step after they are written. `ggml_gallocr` plans the graph and
//! hands the same buffer to tensors whose lifetimes do not overlap.
//!
//! Two things are checked, and the second is the one that matters:
//!
//! 1. **The numbers are identical** to the same graph run the ordinary way.
//!    Reuse that changes an answer is not reuse, it is aliasing.
//! 2. **The plan is smaller** than the sum of the tensors in it.
//!
//! ```text
//! cargo run --release -p chaos-ggml --example try-cpu-gallocr
//! ```

#[cfg(not(have_ggml))]
fn main() {
    eprintln!("built without ggml -- set GGML_LIB_DIR and rebuild");
}

#[cfg(have_ggml)]
fn main() {
    use chaos_ggml::{Context, GraphAllocator};

    // A chain long enough that reuse has something to do: each step is a fresh
    // tensor and the one before it is dead immediately.
    const N: i64 = 1 << 18; // 256K floats = 1 MiB per tensor
    const STEPS: usize = 40;
    let input: Vec<f32> = (0..N).map(|i| (i as f32 * 0.001).sin()).collect();

    fn build(ctx: &Context) -> (chaos_ggml::Tensor<'_>, chaos_ggml::Tensor<'_>) {
        let x = ctx.new_f32_1d(N).expect("input");
        let mut h = x;
        for i in 0..STEPS {
            h = ctx.scale(&h, 1.0 + (i as f32) * 1e-6).expect("scale");
            h = ctx.silu(&h).expect("silu");
        }
        (x, h)
    }

    // -- the ordinary way: every tensor allocated, nothing reused -------------
    let plain = {
        let ctx = Context::new(512 << 20).expect("arena");
        let (x, out) = build(&ctx);
        x.set_f32(&input).expect("set input");
        ctx.compute(&out, 4).expect("compute");
        out.to_vec_f32()
    };
    let tensor_bytes = (STEPS * 2 + 1) * (N as usize) * 4;
    println!(
        "graph        {} tensors, {:.1} MiB if none are shared",
        STEPS * 2 + 1,
        tensor_bytes as f64 / (1 << 20) as f64
    );

    // -- with the planner ----------------------------------------------------
    // `new_no_alloc` still supplies tensor structs, the graph and the compute
    // work buffer; only tensor *data* comes from the plan.
    let ctx = Context::new_no_alloc(64 << 20).expect("metadata arena");
    let (x, out) = build(&ctx);

    let galloc = match GraphAllocator::for_cpu() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no CPU planner: {e:?}");
            std::process::exit(1);
        }
    };
    println!("reserving ...");
    if let Err(e) = galloc.reserve(&ctx, &[&out]) {
        eprintln!("reserve: {e:?}");
        std::process::exit(1);
    }
    println!("allocating ...");
    if let Err(e) = galloc.alloc(&ctx, &[&out]) {
        eprintln!("alloc: {e:?}");
        std::process::exit(1);
    }
    // **After alloc, never before**: the input has no storage until now.
    x.set_f32(&input).expect("set input");
    println!("computing ...");
    ctx.compute(&out, 4).expect("compute");
    let planned = out.to_vec_f32();

    let bytes = galloc.buffer_bytes();
    println!(
        "planned      {:.1} MiB, which is {:.1}x less",
        bytes as f64 / (1 << 20) as f64,
        tensor_bytes as f64 / bytes.max(1) as f64
    );

    // -- 1. the answer is the same -------------------------------------------
    let same = plain.len() == planned.len()
        && plain
            .iter()
            .zip(&planned)
            .all(|(a, b)| (a - b).abs() <= 1e-6 * a.abs().max(1.0));
    println!(
        "  {}",
        if same {
            "IDENTICAL to the unplanned graph"
        } else {
            "DIFFERS -- the reuse is aliasing live tensors"
        }
    );

    // -- 2. it is actually smaller -------------------------------------------
    let smaller = bytes < tensor_bytes / 2;
    println!(
        "  {}",
        if smaller {
            "and less than half the memory, so buffers really are shared"
        } else {
            "but no smaller -- nothing was shared"
        }
    );

    if !same || !smaller {
        std::process::exit(1);
    }
    println!("\nsurvived, exact, and smaller.");
}
