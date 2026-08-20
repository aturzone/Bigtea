//! A template that *refuses* a system role gets the polyfill, not the fallback.
//!
//! # The bug
//!
//! Gemma's chat template calls `raise_exception('System role not supported')`.
//! Honouring that is correct — and it is not what the reference does with the
//! result. minja catches the exception, merges the system turn into the first
//! user turn, and re-renders. We dropped to the hardcoded family matcher
//! instead, which joins with `\n\n`:
//!
//! ```text
//! llama.cpp --jinja : <bos><start_of_turn>user\nSYS\nHI<end_of_turn>\n...
//! chaos            : <bos><start_of_turn>user\nSYS\n\nHI<end_of_turn>\n...
//! ```
//!
//! The merge already existed. It was reached only when the template never
//! mentions a system role *at all* — so Gemma, which mentions it in order to
//! reject it, was the one case the polyfill was written for and the one case it
//! could not see.
//!
//! # Why this runs the binary
//!
//! `[[bin]]` targets set `test = false`, so `render_with_jinja` has no unit-test
//! seam. Running the real binary against a real container also checks the thing
//! that actually matters — the tokens the model receives — rather than a helper
//! in isolation.

use std::path::Path;
use std::process::Command;

/// A container whose template rejects a system role. Skipped when absent.
const GEMMA2: &str = "C:/Projects/models/gemma2/gemma-2-2b-it-Q4_K_M.gguf";

#[test]
#[ignore = "needs the gemma-2 container"]
fn a_template_that_rejects_a_system_role_is_merged_rather_than_abandoned() {
    if !Path::new(GEMMA2).exists() {
        eprintln!("skipping: no model");
        return;
    }

    let out = Command::new(env!("CARGO_BIN_EXE_chaos-run"))
        .args([
            "-m",
            GEMMA2,
            "-sys",
            "SYS",
            "-p",
            "HI",
            "-n",
            "1",
            "--temp",
            "0",
            "--force",
            "-cnv",
            "--jinja",
            "--verbose-prompt",
        ])
        .output()
        .expect("cannot run chaos-run");

    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(
        !text.contains("falling back to the family matcher"),
        "the system-role rejection dropped to the family matcher instead of \
         merging:\n{text}"
    );
    // The rendered prompt is echoed by --verbose-prompt. `SYS\nHI` is the
    // reference's answer; `SYS\n\nHI` is the family matcher's, and is what this
    // test exists to catch.
    assert!(
        text.contains(r"SYS\nHI"),
        "expected the system merged with a single newline:\n{text}"
    );
    assert!(
        !text.contains(r"SYS\n\nHI"),
        "merged with two newlines -- that is the family matcher's spelling, \
         not the template's:\n{text}"
    );
}
