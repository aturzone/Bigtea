//! Print what a GGUF container declares, without reading its weights.
//!
//! Usage: `gguf-info <file.gguf> [--tensors N]`

use std::io::Read;
use std::process::ExitCode;

use bigtea_gguf::{Gguf, TensorInfo, Value};

const GIB: f64 = (1u64 << 30) as f64;

/// Enough for the header, metadata and tensor index of any real model.
const HEADER_BUDGET: usize = 128 << 20;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: gguf-info <file.gguf> [--tensors N]");
        return ExitCode::from(2);
    };
    let mut show = 0usize;
    while let Some(a) = args.next() {
        if a == "--tensors" {
            show = args.next().and_then(|v| v.parse().ok()).unwrap_or(20);
        }
    }

    // Read only the head of the file — the point is not to touch the weights.
    let mut buf = Vec::new();
    match std::fs::File::open(&path) {
        Ok(f) => {
            if let Err(e) = f.take(HEADER_BUDGET as u64).read_to_end(&mut buf) {
                eprintln!("gguf-info: reading {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
        Err(e) => {
            eprintln!("gguf-info: opening {path}: {e}");
            return ExitCode::FAILURE;
        }
    }

    let gguf = match Gguf::parse(&buf) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gguf-info: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("version        {}", gguf.version);
    println!("architecture   {}", gguf.architecture().unwrap_or("?"));
    println!("metadata keys  {}", gguf.metadata.len());
    println!("tensors        {}", gguf.tensors.len());
    println!("data offset    {}", gguf.data_offset);

    if let (Some(no), Some(count)) = (gguf.get_u64("split.no"), gguf.get_u64("split.count")) {
        let total = gguf
            .get_u64("split.tensors.count")
            .map(|t| t.to_string())
            .unwrap_or_else(|| "?".into());
        println!("split          shard {} of {} ({} tensors overall)", no + 1, count, total);
    }

    // Architecture facts the model itself declares.
    let arch = gguf.architecture().unwrap_or("").to_string();
    let keys = [
        "block_count",
        "embedding_length",
        "expert_count",
        "expert_used_count",
        "expert_shared_count",
        "expert_feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "vocab_size",
        "context_length",
    ];
    println!("\ndeclared architecture:");
    for k in keys {
        let full = format!("{arch}.{k}");
        if let Some(v) = gguf.get_u64(&full) {
            println!("  {:<34} {}", k, v);
        }
    }

    let (expert, dense) = gguf.expert_vs_dense_bytes();
    let total = expert + dense;
    println!("\nthis shard's tensor bytes:");
    println!("  routed experts  {:>10.2} GiB", expert as f64 / GIB);
    println!("  everything else {:>10.2} GiB", dense as f64 / GIB);
    println!("  total           {:>10.2} GiB", total as f64 / GIB);

    // Quantization mix: which types carry the bytes.
    let mut by_type: Vec<(String, u64, u64)> = Vec::new();
    for t in &gguf.tensors {
        let Some(size) = t.size_bytes() else { continue };
        let name = t.ty.to_string();
        match by_type.iter_mut().find(|(n, ..)| *n == name) {
            Some(e) => {
                e.1 += size;
                e.2 += 1;
            }
            None => by_type.push((name, size, 1)),
        }
    }
    by_type.sort_by_key(|&(_, s, _)| std::cmp::Reverse(s));
    if !by_type.is_empty() {
        println!("\nquantization mix (this shard):");
        for (name, size, count) in &by_type {
            println!(
                "  {:<10} {:>10.2} GiB  across {:>5} tensors",
                name,
                *size as f64 / GIB,
                count
            );
        }
    }

    if show > 0 {
        println!("\nfirst {show} tensors:");
        for t in gguf.tensors.iter().take(show) {
            print_tensor(t);
        }
        if let Some(t) = gguf.tensors.iter().find(|t| t.is_routed_expert()) {
            println!("\nfirst routed-expert tensor:");
            print_tensor(t);
        }
    }

    // Anything the parser could not size is a correctness risk, so say so.
    let unknown: Vec<&TensorInfo> = gguf
        .tensors
        .iter()
        .filter(|t| t.size_bytes().is_none())
        .collect();
    if !unknown.is_empty() {
        println!("\n! {} tensor(s) had an unknown type or a partial block:", unknown.len());
        for t in unknown.iter().take(5) {
            println!("    {} ({}, {:?})", t.name, t.ty, t.dims);
        }
    }

    if let Some(Value::String(s)) = gguf.get("general.name") {
        println!("\nname           {s}");
    }
    ExitCode::SUCCESS
}

fn print_tensor(t: &TensorInfo) {
    let size = t
        .size_bytes()
        .map(|b| format!("{:.3} GiB", b as f64 / GIB))
        .unwrap_or_else(|| "?".into());
    println!(
        "  {:<44} {:<8} {:?} -> {}",
        t.name,
        t.ty.to_string(),
        t.dims,
        size
    );
}
