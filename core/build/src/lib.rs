//! Build-script helpers, shared so there is one copy of each.
//!
//! **A build dependency, not a runtime one.** Nothing here is linked into a
//! shipped binary; it runs on the build machine and prints `cargo:` directives.
//! The workspace's no-external-dependencies rule is intact — this crate has
//! none either, and it exists precisely so the same forty lines are not pasted
//! into six build scripts.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Give every binary in the calling crate the application icon.
///
/// Windows reads an executable's icon from a resource compiled into the PE file.
/// Without one, Explorer, the taskbar, Alt-Tab and the Start Menu show the blank
/// default — and **that was every binary except `chaos-app` and `chaos-setup`**,
/// because each of those two had its own private copy of this code and the other
/// four crates had none. `chaos-run.exe` shipped with a generic icon for eight
/// releases.
///
/// Done with `windres`, which comes with the MinGW toolchain this project
/// already requires on Windows, rather than a crate. `winres` and friends would
/// be the first build dependency in the workspace and they shell out to the same
/// tool.
///
/// **Only call this from a crate that produces at least one binary.**
/// `cargo:rustc-link-arg-bins` is rejected outright by cargo in a library-only
/// crate — "invalid instruction" — so adding this to one breaks its build rather
/// than being ignored. `chaos-image` hit exactly that.
///
/// **Never fatal.** A missing `windres`, a missing icon or a failed compile
/// leaves the binaries without an icon and the build succeeding. An icon is
/// worth having; it is not worth a build that cannot be made on a machine with a
/// slightly different toolchain, and every build script calling this is also
/// compiled on Linux CI.
pub fn embed_icon() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let Some(ico) = icon_path() else {
        println!("cargo:warning=assets/chaos.ico missing; run tools/make-ico.py");
        return;
    };
    println!("cargo:rerun-if-changed={}", ico.display());

    let Ok(out) = std::env::var("OUT_DIR") else {
        return;
    };
    let out = PathBuf::from(out);
    let rc = out.join("icon.rc");
    // `1` is the first icon resource, and Windows takes the lowest-numbered one
    // as the application icon. Forward slashes on purpose: `windres` reads a
    // backslash in a path as an escape and silently produces no icon.
    let path = ico.to_string_lossy().replace('\\', "/");
    if std::fs::write(&rc, format!("1 ICON \"{path}\"\n")).is_err() {
        return;
    }

    let res = out.join("icon.o");
    let ok = ["windres", "x86_64-w64-mingw32-windres"]
        .iter()
        .any(|tool| {
            Command::new(tool)
                .args([
                    "-i",
                    &rc.to_string_lossy(),
                    "-o",
                    &res.to_string_lossy(),
                    "-O",
                    "coff",
                ])
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        });

    if ok && res.exists() {
        // Handed straight to the linker: a `.o` from windres is an ordinary
        // object file holding the resource section. `-bins` rather than plain
        // `-arg` so a crate that also builds a library or tests is unaffected.
        println!("cargo:rustc-link-arg-bins={}", res.display());
    } else {
        println!("cargo:warning=windres unavailable; these binaries will have no icon");
    }
}

/// `assets/chaos.ico`, found by walking up from the crate being built.
///
/// Walked rather than assumed two levels up: the crates are not all at the same
/// depth, and a hard-coded `../..` is a silent failure the moment one moves.
fn icon_path() -> Option<PathBuf> {
    let start = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    let mut dir: &Path = &start;
    loop {
        let candidate = dir.join("assets").join("chaos.ico");
        if candidate.exists() {
            return Some(candidate);
        }
        dir = dir.parent()?;
    }
}
