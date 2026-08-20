//! Give the executable an icon.
//!
//! The implementation is `chaos_build::embed_icon`, shared with every other
//! crate that produces a binary. **It used to live here in full, and a second
//! copy lived in `chaos-setup/build.rs`** -- which is why the other four crates'
//! binaries had no icon at all: there was nothing to include.

fn main() {
    chaos_build::embed_icon();
}
