//! Every shipped binary answers `--version`.
//!
//! **Two of eleven did.** The rest either said `unknown option` or -- worse --
//! treated the flag as a path and reported "cannot find the file specified",
//! because the argument loop reached the filename before it reached the flag.
//! `chaos-gpubench --version` started benchmarking the GPU.
//!
//! `--version` is how a person checks whether an update landed. After
//! `Help > Install update` or `chaos-run --update`, the obvious thing to type
//! is `<whatever> --version`, and it has to answer on whichever binary that is.
//!
//! Checked by reading the sources rather than by running eleven processes: the
//! binaries live in five crates, a test cannot assume they have been built, and
//! running a benchmark to ask it its version is exactly the behaviour this is
//! here to prevent.
//!
//! **What this does not check is ordering** -- that the flag is handled before
//! an argument is taken as a filename. A byte-offset rule for that failed on
//! `chaos-run`, which is correct: it answers `--version` in an early scan over
//! every argument, 1500 bytes into `main` and still long before the positional
//! path. Ordering was verified the only way that means anything, by running all
//! eleven binaries and reading what they printed, and a test that fails on
//! correct code would have been worse than no test.

use std::path::{Path, PathBuf};

/// The workspace root, from this crate's manifest directory (`core/model`).
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("core/model is two levels below the workspace root")
        .to_path_buf()
}

/// Every `src/bin/*.rs` under `core/`, `cli/`, `gui/` and `network/`.
fn binaries() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for bucket in ["core", "cli", "gui", "network"] {
        let dir = root().join(bucket);
        let Ok(crates) = std::fs::read_dir(&dir) else {
            continue;
        };
        for c in crates.flatten() {
            let bin = c.path().join("src").join("bin");
            let Ok(files) = std::fs::read_dir(&bin) else {
                continue;
            };
            for f in files.flatten() {
                let p = f.path();
                if p.extension().is_some_and(|e| e == "rs") {
                    out.push(p);
                }
            }
        }
    }
    out
}

#[test]
fn every_binary_answers_version() {
    let bins = binaries();
    assert!(
        bins.len() >= 10,
        "found only {} binaries -- has the workspace layout moved again?",
        bins.len()
    );
    let mut missing = Vec::new();
    for p in &bins {
        let src = std::fs::read_to_string(p).expect("read a binary's source");
        // The flag, and the version it would print. Both, because a binary that
        // matches `--version` and then prints a usage line has not answered.
        if !(src.contains("\"--version\"") && src.contains("CARGO_PKG_VERSION")) {
            missing.push(p.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    assert!(
        missing.is_empty(),
        "these binaries do not answer --version: {missing:?}"
    );
}
