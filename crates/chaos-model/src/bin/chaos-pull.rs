//! Fetch a model by name, and say what it will cost before starting.
//!
//! Usage: `chaos-pull <model> [--quant NAME] [--dir PATH] [--yes] [--dry-run]`
//!
//! # What this is really for
//!
//! Not "download a file" — `curl` does that. It is for the question a user
//! cannot answer alone: **will this model run on this machine, and what will it
//! cost me to find out?** A 144 GB download is an afternoon and most of a disk.
//! Being told afterwards that the always-read set does not fit is the worst
//! possible time to learn it.
//!
//! So the plan is printed first, every time, and the answer that matters is not
//! "does the model fit in RAM" — it never does, that is the entire design — but
//! **does the always-read set fit**. Everything else streams.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use chaos_model::catalogue::{self, gib, Plan};

fn main() -> ExitCode {
    let mut model = String::new();
    let mut quant: Option<String> = None;
    let mut dir = PathBuf::from("models");
    let mut yes = false;
    let mut dry_run = false;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--quant" | "-q" => {
                quant = args.get(i + 1).cloned();
                i += 2;
            }
            "--dir" | "-d" => {
                if let Some(d) = args.get(i + 1) {
                    dir = PathBuf::from(d);
                }
                i += 2;
            }
            "--yes" | "-y" => {
                yes = true;
                i += 1;
            }
            "--dry-run" => {
                dry_run = true;
                i += 1;
            }
            "--list" | "-l" => {
                list();
                return ExitCode::SUCCESS;
            }
            "-h" | "--help" => {
                usage();
                return ExitCode::SUCCESS;
            }
            other => {
                // **A mistyped flag is an error, not a filename.** The same
                // catch-all in `chaos-serve` silently ate `-ngl`, `-c`, `--auto`
                // and `--force` for three releases while the app sent all four,
                // so every one of those settings did nothing and nothing said
                // so. There is no shared helper for this: the check is one
                // predicate with nothing to keep in sync, and an extra
                // dependency edge between leaf crates would cost more than it
                // saves.
                if other.starts_with('-') && other.len() > 1 {
                    eprintln!("chaos-pull: unknown option {other:?}");
                    eprintln!("            chaos-pull --help lists what it accepts");
                    return ExitCode::from(2);
                }
                if model.is_empty() {
                    model = other.to_string();
                }
                i += 1;
            }
        }
    }

    if model.is_empty() {
        usage();
        return ExitCode::from(2);
    }

    match run(&model, quant.as_deref(), &dir, yes, dry_run) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("chaos-pull: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    println!("usage: chaos-pull <model> [--quant NAME] [--dir PATH] [--yes] [--dry-run]");
    println!();
    println!("  --list      what Chaos can fetch");
    // "the largest that fits" was wrong twice over: the default picks the
    // largest *available*, and a quant that does not fit still runs here.
    println!("  --quant     which quantisation (default: the largest offered)");
    println!("  --dir       where to put it (default: ./models)");
    println!("  --dry-run   print the plan and stop");
    println!("  --yes       do not ask before downloading");
    println!();
    println!("Prints what the download costs and whether the result will run here,");
    println!("before fetching anything.");
}

fn list() {
    println!(
        "{:<16} {:<12} {:>9}  {:>9}  REPO",
        "MODEL", "QUANT", "SIZE", "RESIDENT"
    );
    for e in catalogue::CATALOGUE {
        for q in e.quants {
            // Marked in the list, not only at the download prompt: somebody
            // reading the catalogue should not have to start a fetch to find out.
            println!(
                "{:<16} {:<12} {:>7.1} GB  {:>6.2} GiB  {}{}",
                e.name,
                q.name,
                q.bytes as f64 / 1e9,
                gib(q.always_read_bytes),
                e.repo,
                if e.adult { "  [18+]" } else { "" }
            );
        }
    }
    println!();
    println!("RESIDENT is what must stay in RAM. The rest streams from disk, so it");
    println!("is the number that decides whether a model runs — not SIZE.");
    if catalogue::CATALOGUE.iter().any(|e| e.adult) {
        println!();
        println!("[18+] marks adult models. Fetching one asks you to confirm your age,");
        println!("and --yes does not skip that.");
    }
}

fn run(
    model: &str,
    quant: Option<&str>,
    dir: &Path,
    yes: bool,
    dry_run: bool,
) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let Some(entry) = catalogue::find(model) else {
        eprintln!("chaos-pull: no model called {model:?}. Known models:");
        list();
        return Ok(ExitCode::from(2));
    };
    let quant = match quant {
        Some(q) => entry
            .quant(q)
            .ok_or_else(|| format!("{} has no quant {q:?}", entry.name))?,
        None => entry.quants.first().ok_or("no quants in catalogue")?,
    };

    let files = entry.files(quant);
    std::fs::create_dir_all(dir)?;

    // Resume: a 144 GB download **will** be interrupted, so what is already
    // there is counted rather than re-fetched.
    let mut have = 0u64;
    for f in &files {
        if let Ok(md) = std::fs::metadata(dir.join(f)) {
            have += md.len();
        }
    }
    let remaining = quant.bytes.saturating_sub(have);

    let machine = chaos_probe::Machine::probe(dir, false);
    let plan = Plan {
        entry,
        quant,
        files: files.clone(),
        total_bytes: quant.bytes,
        remaining_bytes: remaining,
        disk_free_bytes: machine.storage.free_bytes,
        // Leave room for the compute arenas and the expert slices in flight.
        usable_ram_bytes: machine.usable_ram_for_weights(2 << 30),
    };

    print_plan(&plan, dir, have);

    if !plan.fits_on_disk() {
        eprintln!(
            "\nrefusing: {:.1} GB still to download, {:.1} GB free on {}.",
            plan.remaining_bytes as f64 / 1e9,
            plan.disk_free_bytes as f64 / 1e9,
            dir.display()
        );
        return Ok(ExitCode::FAILURE);
    }
    if remaining == 0 {
        println!("\nAlready complete. Nothing to download.");
        return Ok(ExitCode::SUCCESS);
    }
    if dry_run {
        return Ok(ExitCode::SUCCESS);
    }
    // **The age gate, and `--yes` does not skip it.**
    //
    // `--yes` means "do not ask me to confirm a 16 GB download"; it cannot mean
    // "I am over 18", because nobody typed that. A flag that waives an age check
    // is not an age check. Scripts and CI therefore cannot fetch these at all,
    // which is the correct outcome: there is no unattended context in which
    // consent has been given.
    if entry.adult && !adult_confirmed()? {
        println!("Cancelled.");
        return Ok(ExitCode::SUCCESS);
    }
    if !yes && !confirm()? {
        println!("Cancelled.");
        return Ok(ExitCode::SUCCESS);
    }

    fetch(entry, &files, dir)?;
    println!("\nDone. Run it with:");
    println!(
        "  chaos-run {} \"your prompt\" -n 32",
        dir.join(&files[0]).display()
    );
    Ok(ExitCode::SUCCESS)
}

fn print_plan(plan: &Plan, dir: &Path, have: u64) {
    let q = plan.quant;
    println!("model      {} ({})", plan.entry.name, plan.entry.arch);
    println!("quant      {}", q.name);
    println!("from       https://huggingface.co/{}", plan.entry.repo);
    println!("into       {}", dir.display());
    println!(
        "size       {:.1} GB across {} file{}",
        q.bytes as f64 / 1e9,
        q.shards,
        if q.shards == 1 { "" } else { "s" }
    );
    if have > 0 {
        println!(
            "have       {:.1} GB already, {:.1} GB to fetch",
            have as f64 / 1e9,
            plan.remaining_bytes as f64 / 1e9
        );
    }
    println!(
        "disk       {:.1} GB free{}",
        plan.disk_free_bytes as f64 / 1e9,
        if plan.fits_on_disk() {
            ""
        } else {
            "  <-- NOT ENOUGH"
        }
    );
    println!();

    // The number that actually decides whether this was worth downloading.
    println!(
        "resident   {:.2} GiB must stay in RAM; you have {:.2} GiB usable",
        gib(q.always_read_bytes),
        gib(plan.usable_ram_bytes)
    );
    if plan.always_read_fits() {
        println!(
            "           it fits — the other {:.0} GB streams from disk.",
            (q.bytes - q.always_read_bytes) as f64 / 1e9
        );
    } else {
        println!(
            "           SHORT BY {:.2} GiB. It will still run, but that much is",
            gib(plan.shortfall_bytes())
        );
        println!("           re-read from disk on every token, which is slow.");
        println!("           Close some applications, or pick a smaller quant.");
    }
}

/// Ask, in words, before fetching an adult model.
///
/// **Typed rather than a keypress.** `y` is muscle memory after the download
/// prompt above it; spelling something out is a deliberate act. The exact word
/// is echoed so there is no guessing what will be accepted.
///
/// Says what the model *is*, too. Two of these are LoRA adapters and one is a
/// diffusers directory, none of which this engine can run yet -- somebody about
/// to spend a gigabyte deserves to know that before the bar starts moving
/// rather than after.
fn adult_confirmed() -> Result<bool, Box<dyn std::error::Error>> {
    use std::io::{BufRead, Write};
    println!();
    println!("  +------------------------------------------------------------+");
    println!("  |  ADULT CONTENT -- 18+                                      |");
    println!("  +------------------------------------------------------------+");
    println!();
    println!("  This model is published for generating explicit imagery.");
    println!("  Chaos does not filter what a model produces.");
    println!();
    println!("  By continuing you confirm that you are at least 18 years old,");
    println!("  and that adult material is lawful where you are.");
    println!();
    print!("  Type I AM 18 to continue, or anything else to cancel: ");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(catalogue::says_i_am_18(&line))
}

fn confirm() -> Result<bool, Box<dyn std::error::Error>> {
    use std::io::{BufRead, Write};
    print!("\nDownload? [y/N] ");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

/// Fetch through `curl`.
///
/// Chaos has **no external Rust dependencies** — the whole workspace is path
/// crates plus a ggml FFI — and a download is not worth being the thing that
/// ends that. `curl` ships with Windows 10 1803+, macOS and essentially every
/// Linux, handles resume (`-C -`), redirects and progress, and is far better
/// tested than anything that would be written here.
///
/// If that trade stops being worth it, this is the one function to replace.
fn fetch(
    entry: &catalogue::Entry,
    files: &[String],
    dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if Command::new("curl").arg("--version").output().is_err() {
        return Err(
            "curl was not found on PATH, and it is how Chaos downloads. \
                    Install curl, or fetch the files listed above by hand."
                .into(),
        );
    }
    for (i, f) in files.iter().enumerate() {
        let url = entry.url(f);
        let out = dir.join(f);
        println!("\n[{}/{}] {f}", i + 1, files.len());

        let mut cmd = Command::new("curl");
        // `-C -` resumes; `-L` follows the CDN redirect HF issues; `--fail`
        // makes an HTTP error an error rather than a saved error page, which is
        // how a 401 becomes a corrupt .gguf.
        cmd.args([
            "-L",
            "--fail",
            "-C",
            "-",
            "--retry",
            "5",
            "--retry-delay",
            "5",
        ]);
        if let Ok(token) = std::env::var("HF_TOKEN") {
            cmd.arg("-H").arg(format!("Authorization: Bearer {token}"));
        }
        cmd.arg("-o").arg(&out).arg(&url);

        let status = cmd.status()?;
        if !status.success() {
            return Err(format!(
                "curl failed on {f} ({status}). Re-run the same command to resume; \
                 if this is a gated repo, set HF_TOKEN."
            )
            .into());
        }
    }
    Ok(())
}
