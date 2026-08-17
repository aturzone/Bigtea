//! Everything in the app that is not a window.
//!
//! Split out from `main.rs` so it can be tested on any platform: the Win32 half
//! cannot run in CI, and the half that parses a byte stream and formats a
//! number is exactly where the bugs are. `main.rs` is windows-only; this is not.

pub mod art;
pub mod client;
pub mod models;
