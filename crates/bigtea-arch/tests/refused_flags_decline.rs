//! Every flag the `REFUSED` table claims to decline must actually decline.
//!
//! # The bug this exists for
//!
//! `REFUSED` is consulted from the `other =>` fallback arm of the argument
//! match. So the moment a flag gains an *explicit* arm, that arm shadows its
//! `REFUSED` row and the row becomes unreachable — while still reading, to
//! anyone auditing the table, as a statement about what the binary does.
//!
//! `--jinja` sat in exactly that state: the row said "no Jinja engine;
//! templates are matched by family" while `crates/bigtea-jinja` was evaluating
//! templates one arm above it. Nothing failed. The table was simply wrong, the
//! doc generated from it was wrong, and the count of declined flags was one too
//! high in both.
//!
//! # Why it reads the source rather than a copy of the list
//!
//! A hardcoded list here would be a second copy of the same claim, and would
//! drift the same way. The table is extracted from `bigtea-run.rs` at test time,
//! so the test always checks the rows that actually exist today — including any
//! row added after this file was written.

use std::path::PathBuf;
use std::process::Command;

/// Flag names from the `REFUSED` table, read out of the binary's own source.
fn refused_flags() -> Vec<String> {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/bigtea-run.rs");
    let text = std::fs::read_to_string(&src)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", src.display()));

    let start = text
        .find("const REFUSED")
        .expect("REFUSED table not found — has it been renamed?");
    let rest = &text[start..];
    let end = rest.find("\n];").expect("REFUSED table has no terminator");
    let block = &rest[..end];

    // Every entry begins with the flag as a quoted literal. The `why` strings
    // can mention other flags in prose (`see --chat-template`), so matching a
    // WHOLE quoted string against the flag shape is what separates them.
    let mut flags = Vec::new();
    for piece in block.split('"').skip(1).step_by(2) {
        let looks_like_a_flag = piece.starts_with("--")
            && piece.len() > 2
            && piece[2..]
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        if looks_like_a_flag {
            flags.push(piece.to_string());
        }
    }
    flags
}

#[test]
fn declined_flags_actually_decline() {
    let flags = refused_flags();

    // The count is asserted BEFORE the loop, because a loop over an empty list
    // passes. That is the same shape as the jinja test that ran green on three
    // CI runners while checking nothing, and it is worth not repeating twice.
    assert!(
        flags.len() > 15,
        "extracted only {} flags from REFUSED — the parser is broken, not the table",
        flags.len()
    );

    let bin = env!("CARGO_BIN_EXE_bigtea-run");
    let mut alive = Vec::new();
    for flag in &flags {
        // A dummy value follows unconditionally. Refusal is decided on the flag
        // token in the fallback arm, before any value is consumed and before a
        // model is opened, so no container is needed and the value is never
        // read — which also means the arity in the table cannot make this wrong.
        let out = Command::new(bin)
            .args([flag.as_str(), "ignored"])
            .output()
            .unwrap_or_else(|e| panic!("cannot run {bin}: {e}"));

        // **The exit code alone does not discriminate**, which is the trap here.
        // A shadowed flag parses fine and the run then dies on the missing model
        // — `bigtea-run --jinja ignored` exits 2 with "no model given", the same
        // code a refusal uses. Checking only the status would have passed on the
        // exact bug this file was written for. The message is the evidence.
        let code = out.status.code();
        let err = String::from_utf8_lossy(&out.stderr);
        if code != Some(2) || !err.contains("is not supported") {
            alive.push(format!(
                "{flag}: exit {code:?}, stderr {:?}",
                err.lines().next().unwrap_or("")
            ));
        }
    }

    assert!(
        alive.is_empty(),
        "{} of {} REFUSED rows did not decline — an explicit match arm shadows \
         them, so the table describes a binary that no longer exists:\n  {}",
        alive.len(),
        flags.len(),
        alive.join("\n  ")
    );
}

#[test]
fn an_unknown_flag_is_an_error_rather_than_the_prompt() {
    // The bug: the fallback arm took ANY leftover token as the prompt, so a
    // mistyped or unimplemented flag was eaten and the real prompt discarded
    // with it. `bigtea-run -m m -fa off "hello"` set prompt = "-fa", exited 0,
    // and fluently completed the wrong text. Nothing in the output said so.
    let out = Command::new(env!("CARGO_BIN_EXE_bigtea-run"))
        .args(["-m", "nonexistent.gguf", "--not-a-real-flag", "hello"])
        .output()
        .expect("cannot run bigtea-run");

    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr was: {err}");
    assert!(
        err.contains("unknown flag"),
        "an unknown flag must say so: {err}"
    );
    // It must fail on the FLAG, not later on the missing model -- otherwise the
    // test passes on a build that still swallows the flag.
    assert!(
        !err.contains("no model given"),
        "reached model loading, so the flag was still swallowed: {err}"
    );
}

#[test]
fn a_double_dash_still_allows_a_prompt_that_starts_with_one() {
    // Making unknown flags an error takes away the only way to write a prompt
    // beginning with a dash, so the escape hatch has to work.
    let out = Command::new(env!("CARGO_BIN_EXE_bigtea-run"))
        .args(["--", "--not-a-real-flag"])
        .output()
        .expect("cannot run bigtea-run");

    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !err.contains("unknown flag"),
        "`--` did not stop flag parsing: {err}"
    );
    assert!(
        err.contains("no model given"),
        "expected it to get as far as needing a model: {err}"
    );
}

#[test]
fn flash_attn_off_is_refused_because_there_is_no_other_path() {
    // `-fa off` matters more than a normal unimplemented flag: it is one of the
    // no-op controls `scripts/parity-check.sh` passes to the REFERENCE to ask
    // whether it agrees with itself. Silently ignoring it on our side would
    // turn a parity check into a comparison of a run with itself.
    let bin = env!("CARGO_BIN_EXE_bigtea-run");
    let off = Command::new(bin)
        .args(["-m", "nonexistent.gguf", "-fa", "off"])
        .output()
        .expect("cannot run bigtea-run");
    let err = String::from_utf8_lossy(&off.stderr);
    assert_eq!(off.status.code(), Some(2), "stderr was: {err}");
    assert!(err.contains("one attention path"), "{err}");

    // `on` describes what this build actually does, so it is accepted and the
    // run proceeds to the next thing it lacks.
    let on = Command::new(bin)
        .args(["-fa", "on"])
        .output()
        .expect("cannot run bigtea-run");
    let on_err = String::from_utf8_lossy(&on.stderr);
    assert!(
        on_err.contains("no model given"),
        "-fa on should be accepted: {on_err}"
    );
}

#[test]
fn a_declined_flag_says_why_and_does_not_merely_exit() {
    // Declining is only better than ignoring if the message is actionable. An
    // exit code alone leaves the caller unable to tell a refusal from a crash.
    let out = Command::new(env!("CARGO_BIN_EXE_bigtea-run"))
        .args(["--gpu-layers", "32"])
        .output()
        .expect("cannot run bigtea-run");

    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr was: {err}");
    assert!(err.contains("--gpu-layers"), "message names no flag: {err}");
    assert!(
        err.contains("Drop the flag to continue"),
        "message gives the caller nothing to do: {err}"
    );
}
