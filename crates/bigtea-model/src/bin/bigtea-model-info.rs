//! Resolve a split model and report its real layout.
//!
//! Usage: `bigtea-model-info <any-shard.gguf> [--budget GIB]`
//!
//! Works on a model that is still downloading: GGUF puts the index at the
//! start of each shard, so the full layout is knowable long before the weights
//! have landed.

use std::process::ExitCode;

use bigtea_model::Model;

const GIB: f64 = (1u64 << 30) as f64;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: bigtea-model-info <any-shard.gguf> [--budget GIB]");
        return ExitCode::from(2);
    };
    let mut budget_gib = 0.0f64;
    while let Some(a) = args.next() {
        if a == "--budget" {
            budget_gib = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        }
    }

    let model = match Model::open_split(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bigtea-model-info: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("architecture   {}", model.architecture());
    println!("shards         {}", model.shard_count());
    println!("io             {}", model.io_mode());

    let (available, total) = model.availability();
    match model.declared_tensor_count() {
        Some(declared) if declared as usize != total => println!(
            "tensors        {total} indexed of {declared} declared \
             ({} shard(s) still missing)",
            declared as usize - total
        ),
        _ => println!("tensors        {total}"),
    }
    println!(
        "on disk        {available} of {total} tensors readable now ({:.0}%)",
        available as f64 / total.max(1) as f64 * 100.0
    );

    let (expert, dense) = model.expert_vs_dense_bytes();
    let sum = expert + dense;
    println!("\nweight layout (indexed tensors):");
    println!("  always-read   {:>9.2} GiB   read on every token", dense as f64 / GIB);
    println!("  routed expert {:>9.2} GiB   read only when selected", expert as f64 / GIB);
    println!("  total         {:>9.2} GiB", sum as f64 / GIB);

    // The routing facts that turn a pool size into a per-token cost.
    if let (Some(n), Some(used)) = (model.arch_u64("expert_count"), model.arch_u64("expert_used_count")) {
        let per_token = expert / n.max(1) * used;
        println!(
            "\nrouting        {used} of {n} experts per token \
             -> {:.2} GiB of experts read per token",
            per_token as f64 / GIB
        );
        if budget_gib > 0.0 {
            let budget = (budget_gib * GIB) as u64;
            let resident = dense.min(budget);
            let shortfall = dense - resident;
            println!("\nwith a {budget_gib:.1} GiB weight budget:");
            println!("  dense resident  {:>8.2} GiB", resident as f64 / GIB);
            if shortfall > 0 {
                println!("  dense streaming {:>8.2} GiB  (re-read every token)", shortfall as f64 / GIB);
            }
            println!(
                "  read per token  {:>8.2} GiB",
                (shortfall + per_token) as f64 / GIB
            );
        }
    }

    // A tensor that is actually readable right now, proving the path end to end.
    if let Some(name) = model
        .tensor_names()
        .find(|n| model.is_available(n).unwrap_or(false))
        .map(str::to_string)
    {
        match model.read_tensor(&name) {
            Ok(bytes) => println!(
                "\nread check     {name}: {} bytes read successfully",
                bytes.len()
            ),
            Err(e) => println!("\nread check     {name}: FAILED: {e}"),
        }
    }

    ExitCode::SUCCESS
}
