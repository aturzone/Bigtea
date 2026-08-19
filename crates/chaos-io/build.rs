//! Give this crate's binaries the application icon.
//!
//! chaos-iobench shipped with the blank Windows default for eight releases,
//! because the icon logic lived inside two other crates' build scripts and
//! nothing shared it. See `chaos_build::embed_icon`.

fn main() {
    chaos_build::embed_icon();
}
