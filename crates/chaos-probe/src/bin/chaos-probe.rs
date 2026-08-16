//! Measure this machine.
//!
//! Usage: `chaos-probe [path] [--bandwidth]`
//!
//! The read-bandwidth benchmark is **off by default**: it writes a temporary
//! file larger than RAM, and this is the first command a new user runs — often
//! to check whether they have disk space at all. `--bandwidth` opts in;
//! `--quick` is still accepted and is now the default behaviour.
//! (it writes and reads a file larger than available RAM, deliberately, so the
//! page cache cannot hide the disk).

use std::process::ExitCode;

use chaos_probe::{processes, Machine};

fn main() -> ExitCode {
    let mut path = String::from(".");
    // **Quick by default.** The bandwidth benchmark writes a temporary file
    // larger than RAM, and this is the first command anyone runs — often to
    // decide whether they have the disk space for a model at all. Filling that
    // disk uninvited, with no confirmation, is the wrong first impression, and
    // it made an unarguable `chaos-probe` in CI a multi-minute disk hammer.
    // Opt in with `--bandwidth`; `--quick` stays accepted so existing scripts
    // and every doc that mentions it keep working.
    let mut quick = true;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--quick" | "-q" => quick = true,
            "--bandwidth" | "-b" => quick = false,
            "--processes" | "-p" => {
                dump_processes();
                return ExitCode::SUCCESS;
            }
            "-h" | "--help" => {
                println!("usage: chaos-probe [path] [--bandwidth] [--processes]");
                println!();
                println!("  --bandwidth  measure read speed. Writes a temporary file larger");
                println!("               than RAM, so it is off unless you ask for it.");
                println!("  --processes  what is holding RAM, and what closing it would free.");
                return ExitCode::SUCCESS;
            }
            other => path = other.to_string(),
        }
    }

    if !quick {
        eprintln!("measuring read bandwidth (writes a temporary file larger than RAM)...");
    }
    let machine_note = if quick {
        "
read       pass --bandwidth to measure it (writes a temp file > RAM)"
    } else {
        ""
    };
    let machine = Machine::probe(&path, !quick);
    println!("{machine}");
    if !machine_note.is_empty() {
        println!("{machine_note}");
    }

    // The number every plan hangs off, made explicit.
    let usable = machine.usable_ram_for_weights(OVERHEAD);
    println!(
        "\nusable for weights   {:.1} GiB   (available RAM minus a {:.0} GiB placeholder; \
         run chaos-model-info for the real figure)",
        chaos_probe::gib(usable),
        chaos_probe::gib(OVERHEAD)
    );

    report_reclaimable(usable);
    ExitCode::SUCCESS
}

/// Every process we can see, with whether we would touch it.
fn dump_processes() {
    let all = processes::list();
    println!("{} processes visible\n", all.len());
    println!("{:<34}{:>10}  status", "name", "rss");
    for p in all.iter().take(30) {
        println!(
            "{:<34}{:>9.0}M  {}",
            p.name,
            p.rss_bytes as f64 / (1 << 20) as f64,
            if p.protected {
                "protected"
            } else {
                "closeable"
            }
        );
    }
    let total: u64 = all.iter().map(|p| p.rss_bytes).sum();
    let closeable: u64 = all
        .iter()
        .filter(|p| !p.protected)
        .map(|p| p.rss_bytes)
        .sum();
    println!(
        "\ntotal {:.2} GiB, of which {:.2} GiB is closeable",
        chaos_probe::gib(total),
        chaos_probe::gib(closeable)
    );
}

/// Runtime cost that is *not* weights, when the model's shape is unknown.
///
/// A rough placeholder only: the real figure depends on attention shape and
/// context length and is computed per model by `chaos-plan`. Kept small
/// because `available` RAM already excludes the OS — charging 3 GiB here, as
/// this once did, double-counted it and threw away ~2 GiB of budget on a
/// machine with none to spare.
const OVERHEAD: u64 = 1 << 30;
/// Ignore anything smaller than this — closing a 64 MiB helper is disruption
/// for no benefit.
const MIN_WORTH_CLOSING: u64 = 128 << 20;

/// Show what is holding RAM and what closing it would actually buy.
///
/// On a machine this size that number is often the difference between the
/// dense weights being cached and being re-read every token, so it is worth
/// putting in front of the user before they start a run.
fn report_reclaimable(usable_now: u64) {
    let groups = processes::grouped(MIN_WORTH_CLOSING);
    if groups.is_empty() {
        return;
    }
    let total: u64 = groups.iter().map(|(_, b, _)| b).sum();

    println!("\nholding RAM (closeable, largest first):");
    for (name, bytes, count) in groups.iter().take(8) {
        let instances = if *count > 1 {
            format!("  ({count} processes)")
        } else {
            String::new()
        };
        println!(
            "  {:<28} {:>7.2} GiB{}",
            name,
            chaos_probe::gib(*bytes),
            instances
        );
    }
    println!(
        "\n  closing all of these would free up to {:.2} GiB,\n  \
         raising usable-for-weights from {:.1} to about {:.1} GiB.",
        chaos_probe::gib(total),
        chaos_probe::gib(usable_now),
        chaos_probe::gib(usable_now + total)
    );
    println!(
        "  (upper bound: processes share pages, and the OS may not return\n   \
         freed memory immediately. Nothing was closed -- this is a report.)"
    );
}
