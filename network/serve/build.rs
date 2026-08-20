//! The icon, and nothing else.
//!
//! `chaos_build::embed_icon` is shared with every crate that produces a binary,
//! so there is one definition of how the icon is attached rather than one per
//! crate that drifts.

fn main() {
    chaos_build::embed_icon();
}
