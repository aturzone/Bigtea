//! Fail early, and in one sentence, when `ggml` is missing.
//!
//! `chaos-arch` is the only crate that cannot be built without `ggml`: it is
//! the forward pass, and the forward pass is `ggml` graphs. Every other crate in
//! the workspace builds fine without it, which is deliberate — the container,
//! probe and planning tools are useful on a machine that has never compiled a
//! line of C.
//!
//! Without this file the failure is a wall of `unresolved import
//! chaos_ggml::Context` errors, repeated once per module, which says nothing
//! about the actual problem: an environment variable is not set. That is the
//! first thing a new contributor sees, and "the build exploded" is a worse first
//! impression than "you need one more step, here it is".

use std::path::PathBuf;

const REQUIRED: [&str; 3] = ["ggml-base", "ggml-cpu", "ggml"];

fn main() {
    // Every binary in this crate -- chaos-run, chaos-serve and the benches --
    // shipped with the blank Windows default until the icon logic was shared.
    chaos_build::embed_icon();
    println!("cargo:rerun-if-env-changed=GGML_LIB_DIR");

    let Some(dir) = std::env::var_os("GGML_LIB_DIR").map(PathBuf::from) else {
        fail("GGML_LIB_DIR is not set.");
    };

    let missing: Vec<&str> = REQUIRED
        .iter()
        .copied()
        .filter(|name| !dir.join(format!("{name}.a")).exists())
        .collect();

    if !missing.is_empty() {
        fail(&format!(
            "GGML_LIB_DIR is {}, but it does not contain: {}",
            dir.display(),
            missing
                .iter()
                .map(|n| format!("{n}.a"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
}

fn fail(why: &str) -> ! {
    // `cargo:warning` lines survive into the error output and are not truncated
    // the way a panic message can be, so the instructions arrive intact.
    for line in [
        "".to_string(),
        format!("chaos-arch cannot build: {why}"),
        "".to_string(),
        "  chaos-arch is the inference engine and needs ggml's static libraries.".to_string(),
        "  Every other crate in this workspace builds without them.".to_string(),
        "".to_string(),
        "  1. Build ggml once:".to_string(),
        "       git clone https://github.com/ggml-org/llama.cpp".to_string(),
        "       cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release \\".to_string(),
        "         -DBUILD_SHARED_LIBS=OFF   # llama.cpp defaults to shared; we need .a".to_string(),
        "       cmake --build llama.cpp/build --config Release -j".to_string(),
        "".to_string(),
        "  2. Point Chaos at the result (the directory holding ggml-base.a,".to_string(),
        "     ggml-cpu.a and ggml.a -- usually llama.cpp/build/ggml/src):".to_string(),
        "       export GGML_LIB_DIR=/path/to/llama.cpp/build/ggml/src".to_string(),
        "       $env:GGML_LIB_DIR = \"C:/path/to/llama.cpp/build/ggml/src\"   # PowerShell"
            .to_string(),
        "".to_string(),
        "  Full instructions: https://github.com/aturzone/Chaos#building".to_string(),
        "".to_string(),
    ] {
        println!("cargo:warning={line}");
    }
    panic!("ggml not found -- see the instructions above");
}
