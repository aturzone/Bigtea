//! Measure this machine.
//!
//! Usage: `bigtea-probe [path] [--quick]`
//!
//! `--quick` skips the read-bandwidth benchmark, which is the only slow step
//! (it writes and reads a file larger than available RAM, deliberately, so the
//! page cache cannot hide the disk).

use std::process::ExitCode;

use bigtea_probe::Machine;

fn main() -> ExitCode {
    let mut path = String::from(".");
    let mut quick = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--quick" | "-q" => quick = true,
            "-h" | "--help" => {
                println!("usage: bigtea-probe [path] [--quick]");
                return ExitCode::SUCCESS;
            }
            other => path = other.to_string(),
        }
    }

    if !quick {
        eprintln!("measuring read bandwidth (writes a temporary file larger than RAM)...");
    }
    let machine = Machine::probe(&path, !quick);
    println!("{machine}");

    // The number every plan hangs off, made explicit.
    let usable = machine.usable_ram_for_weights(3 << 30);
    println!(
        "\nusable for weights   {:.1} GiB   (available RAM minus 3 GiB for OS/KV/scratch)",
        bigtea_probe::gib(usable)
    );
    ExitCode::SUCCESS
}
