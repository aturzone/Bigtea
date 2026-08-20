//! Whether a newer Chaos exists -- the shared answer, under the app's own name.
//!
//! The logic lives in [`chaos_model::release`] because a release ships eleven
//! binaries and the window is one of them: `chaos-run --update` has to give the
//! same answer, and `chaos-model` is the crate both depend on.
//!
//! This module is the app's name for it, so the window reads `update::decide`
//! rather than reaching across two crates in every call.

pub use chaos_model::release::*;
