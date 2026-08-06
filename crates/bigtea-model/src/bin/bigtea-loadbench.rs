//! Measure how much read parallelism this machine's storage actually rewards.
//!
//! Usage: `bigtea-loadbench <any-shard.gguf> [--budget GIB]`
//!
//! A single synchronous read leaves an NVMe device mostly idle: it wants
//! several requests in flight to reach rated bandwidth. This sweeps thread
//! counts against the real model and reports where the returns stop, so the
//! streaming layer can be configured from a measurement rather than a guess.

use std::process::ExitCode;

use bigtea_model::{Model, ResidentSet};

const GIB: f64 = (1u64 << 30) as f64;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: bigtea-loadbench <any-shard.gguf> [--budget GIB]");
        return ExitCode::from(2);
    };
    let mut budget_gib = 3.0f64;
    while let Some(a) = args.next() {
        if a == "--budget" {
            budget_gib = args.next().and_then(|v| v.parse().ok()).unwrap_or(3.0);
        }
    }

    let model = match Model::open_split(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bigtea-loadbench: {e}");
            return ExitCode::FAILURE;
        }
    };
    let budget = (budget_gib * GIB) as u64;

    println!(
        "model    {}  ({} shards, {})",
        model.architecture(),
        model.shard_count(),
        model.io_mode()
    );
    println!("budget   {budget_gib:.1} GiB of always-read weights\n");
    println!(
        "{:>7}  {:>9}  {:>10}  {:>8}",
        "threads", "seconds", "GB/s", "speedup"
    );

    let mut baseline = 0.0f64;
    for threads in [1usize, 2, 4, 8, 12, 16] {
        match ResidentSet::load_parallel(&model, budget, threads) {
            Ok((set, report)) => {
                let gbps = report.bytes_per_sec() / 1e9;
                if threads == 1 {
                    baseline = gbps;
                }
                let speedup = if baseline > 0.0 { gbps / baseline } else { 1.0 };
                println!(
                    "{threads:>7}  {:>9.2}  {gbps:>10.2}  {speedup:>7.2}x",
                    report.seconds
                );
                // Drop before the next run so RAM is returned to the OS.
                drop(set);
            }
            Err(e) => {
                println!("{threads:>7}  failed: {e}");
                break;
            }
        }
    }

    println!(
        "\nNote: reads bypass the page cache, so repeated runs re-read from the\n\
         device rather than replaying RAM -- these are real disk numbers."
    );
    ExitCode::SUCCESS
}
