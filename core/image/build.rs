//! Put the Chaos icon on `chaos-draw`.
//!
//! The implementation is `chaos_build::embed_icon`, shared with every other
//! crate that produces a binary, so there is one definition of how the icon is
//! attached rather than one per crate that drifts.

fn main() {
    chaos_build::embed_icon();
}
