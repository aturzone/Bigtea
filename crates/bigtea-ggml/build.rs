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

    // ggml has C++ translation units, so its runtime must come along — and on
    // Windows-GNU it must come along *statically*.
    //
    // **This is the difference between a binary somebody can download and one
    // they cannot.** Linked dynamically, `bigtea-run.exe` needs
    // `libstdc++-6.dll` and `libgomp-1.dll` from MSYS2. Without them Windows
    // terminates the process with `0xC0000135` (STATUS_DLL_NOT_FOUND) *before*
    // `main`, so there is no message, no usage, nothing — it just exits.
    // Measured on a PATH with no MSYS2: dynamic exits -1073741515 in silence,
    // static prints its usage. Cost is about 0.7 MB of binary.
    //
    // Elsewhere dynamic is right: libstdc++ is a system library on Linux and
    // Apple's libc++ always is.
    //
    // `static=` makes *rustc* resolve the archive, so its directory has to be on
    // rustc's search path — the linker's own defaults are not enough, and
    // without it the build fails with "could not find native static library
    // `stdc++`". Ask the compiler that will do the linking where it keeps them,
    // rather than hardcoding an MSYS2 path that is wrong on CI and on everyone
    // else's machine. If that fails, stay dynamic: the build still works for
    // anyone who has MSYS2 on PATH, which is everyone building from source.
    let windows_gnu = target_os == "windows"
        && target_env == "gnu"
        && link_static_runtime_dir().is_some_and(|dir| {
            println!("cargo:rustc-link-search=native={dir}");
            true
        });
    match (target_os.as_str(), target_env.as_str()) {
        ("macos", _) | ("ios", _) => println!("cargo:rustc-link-lib=dylib=c++"),
        (_, "msvc") => {} // MSVC links its runtime automatically
        _ if windows_gnu => println!("cargo:rustc-link-lib=static=stdc++"),
        _ => println!("cargo:rustc-link-lib=dylib=stdc++"),
    }

    // On Apple platforms ggml's cmake enables Accelerate by default and calls
    // into vDSP for the vector kernels. Without the framework the link ends in
    // `Undefined symbols for architecture arm64: _vDSP_vadd, _vDSP_vmul, ...`
    // — a list that names Apple's library rather than ggml, which is why it is
    // easy to misread as a ggml problem.
    if matches!(target_os.as_str(), "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // ggml's quantization kernels *may* be built with OpenMP, in which case
    // GOMP_* symbols must resolve or the link fails with a wall of undefined
    // references from ggml-quants.c and nothing else.
    //
    // Whether they are is a property of how ggml was built, not of the target,
    // and it differs by platform: Apple's clang ships no OpenMP runtime, so
    // ggml's own cmake reports "OpenMP not found" and builds without it. Asking
    // for `-lgomp` there fails with `ld: library 'gomp' not found` — which was
    // exactly how this bug first showed up, on the very first macOS CI run.
    //
    // Default per platform, and let anyone with a non-default ggml override it.
    let openmp = match std::env::var("BIGTEA_GGML_OPENMP").as_deref() {
        Ok("1") | Ok("true") => true,
        Ok("0") | Ok("false") => false,
        // Apple: no OpenMP unless libomp was installed and ggml found it.
        // MSVC: links its own runtime.
        _ => !matches!(target_os.as_str(), "macos" | "ios") && target_env != "msvc",
    };
    if openmp {
        // Static on Windows-GNU for the same reason as libstdc++ above: a
        // downloaded binary has no MSYS2 to find `libgomp-1.dll` in.
        if windows_gnu {
            println!("cargo:rustc-link-lib=static=gomp");
        } else {
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    }
    println!("cargo:rerun-if-env-changed=BIGTEA_GGML_OPENMP");

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

/// Where the GNU toolchain keeps `libstdc++.a` and `libgomp.a`.
///
/// `gcc -print-file-name=X` returns the full path of the archive it would link,
/// or the bare name if it has none — which is the signal that static linking is
/// not available and the caller should stay dynamic.
fn link_static_runtime_dir() -> Option<String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "gcc".into());
    let mut dir = None;
    for lib in ["libstdc++.a", "libgomp.a"] {
        let out = std::process::Command::new(&cc)
            .arg(format!("-print-file-name={lib}"))
            .output()
            .ok()?;
        let path = String::from_utf8(out.stdout).ok()?;
        let path = std::path::Path::new(path.trim());
        // A bare filename back means gcc could not find it.
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty())?;
        if !path.exists() {
            return None;
        }
        dir = Some(parent.display().to_string());
    }
    dir
}
