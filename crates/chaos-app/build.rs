//! Give the executable an icon.
//!
//! Windows reads an application's icon from a resource compiled into the PE
//! file. Without one, Explorer, the taskbar, Alt-Tab and the Start Menu all
//! show the blank default -- which is what shipping "empty shapes" was.
//!
//! Done with `windres`, which comes with the MinGW toolchain this project
//! already requires to build at all, rather than a crate. `winres` and friends
//! would be the first build dependency in the workspace, and they shell out to
//! the same tool.
//!
//! **Never fatal.** A missing `windres`, a missing icon or a failed compile
//! leaves the binary without an icon and the build succeeding. An icon is worth
//! having; it is not worth a build that cannot be made on a machine with a
//! slightly different toolchain, and this file is compiled on Linux CI too.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let ico = root.join("assets").join("chaos.ico");
    println!("cargo:rerun-if-changed={}", ico.display());
    if !ico.exists() {
        println!("cargo:warning=assets/chaos.ico missing; run tools/make-ico.py");
        return;
    }

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let rc = out.join("icon.rc");
    // `1` is the first icon resource, and Windows takes the lowest-numbered one
    // as the application icon. Forward slashes on purpose: `windres` treats a
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
        // object file holding the resource section.
        println!("cargo:rustc-link-arg-bins={}", res.display());
    } else {
        println!("cargo:warning=windres unavailable; this binary will have no icon");
    }
}
