//! Link against a prebuilt `ggml`.
//!
//! `ggml` is the arithmetic we deliberately do not rewrite — quantized matmul
//! kernels are years of specialist SIMD work, and reimplementing them is not
//! where this project contributes. We own memory, residency, streaming and the
//! token loop; `ggml` owns the math.
//!
//! Point `GGML_LIB_DIR` at a directory containing `ggml-base.a`, `ggml-cpu.a`
//! and `ggml.a`. If it is unset, the build still succeeds but the crate
//! compiles with `ggml` unavailable, so the rest of the workspace keeps
//! building on a machine that has not built `ggml` yet.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=GGML_LIB_DIR");
    // Declare the cfg we set below, so `--cfg have_ggml` is a known name
    // rather than 10 "unexpected cfg condition" warnings in every build
    // that does not have ggml -- which is the build a newcomer runs first.
    println!("cargo::rustc-check-cfg=cfg(have_ggml)");

    let Some(dir) = std::env::var_os("GGML_LIB_DIR").map(PathBuf::from) else {
        println!(
            "cargo:warning=GGML_LIB_DIR not set; bigtea-ggml builds without ggml \
             (set it to a directory containing ggml-base.a, ggml-cpu.a, ggml.a)"
        );
        return;
    };

    let required = ["ggml-base", "ggml-cpu", "ggml"];
    let missing: Vec<_> = required
        .iter()
        .filter(|name| !dir.join(format!("{name}.a")).exists())
        .collect();
    if !missing.is_empty() {
        println!("cargo:warning=missing in GGML_LIB_DIR: {missing:?}; building without ggml");
        return;
    }

    // The GNU linker resolves `-lggml` to `libggml.a`, but ggml's own build
    // emits `ggml.a` with no prefix. Stage copies under the expected names
    // rather than asking the user to rename anything.
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"));
    let staged = out.join("ggml-libs");
    if let Err(e) = std::fs::create_dir_all(&staged) {
        println!("cargo:warning=cannot stage ggml libraries: {e}");
        return;
    }
    for name in required {
        let from = dir.join(format!("{name}.a"));
        let to = staged.join(format!("lib{name}.a"));
        // Re-copy every time: a rebuilt ggml must not be silently ignored.
        if let Err(e) = std::fs::copy(&from, &to) {
            println!("cargo:warning=cannot copy {}: {e}", from.display());
            return;
        }
        println!("cargo:rerun-if-changed={}", from.display());
    }

    println!("cargo:rustc-link-search=native={}", staged.display());
    // Order matters for static archives: dependents before dependencies.
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    // NOTE: `cfg!()` here would describe the *host*, not the target -- a
    // classic build-script trap that silently links the wrong runtime when
    // cross-compiling. Cargo passes the target's configuration in env vars.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // ggml has C++ translation units, so its runtime must come along.
    match (target_os.as_str(), target_env.as_str()) {
        ("macos", _) | ("ios", _) => println!("cargo:rustc-link-lib=dylib=c++"),
        (_, "msvc") => {} // MSVC links its runtime automatically
        _ => println!("cargo:rustc-link-lib=dylib=stdc++"),
    }

    // ggml's quantization kernels are built with OpenMP, so GOMP_* symbols
    // must resolve. Missing this fails with a wall of undefined references
    // from ggml-quants.c and nothing else.
    if target_env != "msvc" {
        println!("cargo:rustc-link-lib=dylib=gomp");
    }

    if target_os == "windows" {
        // ggml-cpu reads the registry to identify the CPU, which pulls in
        // advapi32 (Reg* functions). Without it the link fails on three
        // symbols and nothing else, which is easy to misread as a ggml
        // problem rather than a missing system library.
        println!("cargo:rustc-link-lib=dylib=advapi32");
    } else {
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=pthread");
    }

    println!("cargo:rustc-cfg=have_ggml");
    println!("cargo:rustc-check-cfg=cfg(have_ggml)");
}
