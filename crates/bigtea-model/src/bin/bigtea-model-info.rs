//! Resolve a split model and report its real layout.
//!
//! Usage: `bigtea-model-info <any-shard.gguf> [--budget GIB]`
//!
//! Works on a model that is still downloading: GGUF puts the index at the
//! start of each shard, so the full layout is knowable long before the weights
//! have landed.

use std::process::ExitCode;

use bigtea_model::Model;
use bigtea_plan::{max_context_for_budget, overhead, AttentionShape, KV_BYTES_F16};

const GIB: f64 = (1u64 << 30) as f64;

/// Attention shape from the container's own metadata, for sizing the KV cache.
fn attention_shape(model: &Model) -> Option<AttentionShape> {
    let hidden = model.arch_u64("embedding_length")?;
    let n_layers = model.arch_u64("block_count")?;
    let n_heads = model.arch_u64("attention.head_count").unwrap_or(1).max(1);
    let n_kv_heads = model
        .arch_u64("attention.head_count_kv")
        .unwrap_or(n_heads)
        .max(1);
    // Prefer the declared key length; fall back to hidden/heads.
    let head_dim = model
        .arch_u64("attention.key_length")
        .unwrap_or(hidden / n_heads)
        .max(1);
    Some(AttentionShape {
        n_layers,
        n_kv_heads,
        head_dim,
        hidden_size: hidden,
        ffn_intermediate: model
            .arch_u64("expert_feed_forward_length")
            .or_else(|| model.arch_u64("feed_forward_length"))
            .unwrap_or(hidden),
    })
}

/// What a given amount of RAM actually buys, with the runtime cost computed
/// from the architecture rather than guessed.
fn report_budget(model: &Model, ram_gib: f64, dense: u64, per_token: u64) {
    let ram = (ram_gib * GIB) as u64;
    let Some(shape) = attention_shape(model) else {
        eprintln!("(cannot size KV cache: attention metadata missing)");
        return;
    };

    println!("\nwith {ram_gib:.1} GiB of available RAM:");
    for ctx in [4096u64, 32768, 131_072] {
        let ov = overhead(&shape, ctx, KV_BYTES_F16);
        let usable = ram.saturating_sub(ov.total());
        let resident = dense.min(usable);
        let shortfall = dense - resident;
        let per_tok = shortfall + per_token;
        let verdict = if shortfall == 0 {
            "dense resident".to_string()
        } else {
            format!("{:.2} GiB dense re-read/token", shortfall as f64 / GIB)
        };
        println!(
            "  {:>6} ctx  overhead {:>5.2} GiB  usable {:>5.2} GiB  \
             read/token {:>5.2} GiB  [{}]",
            ctx,
            ov.total() as f64 / GIB,
            usable as f64 / GIB,
            per_tok as f64 / GIB,
            verdict
        );
    }

    let max_ctx = max_context_for_budget(&shape, ram, dense, KV_BYTES_F16);
    if max_ctx > 0 {
        println!(
            "\n  longest context with all dense weights resident: {max_ctx} tokens"
        );
    } else {
        let ov = overhead(&shape, 4096, KV_BYTES_F16);
        let need = (dense + ov.total()) as f64 / GIB;
        println!(
            "\n  dense weights do not fit: needs about {need:.2} GiB available \
             RAM to keep them resident at 4K context"
        );
    }
}

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
            report_budget(&model, budget_gib, dense, per_token);
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
