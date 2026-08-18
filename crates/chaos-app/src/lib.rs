//! Everything in the app that is not a window.
//!
//! Split out from `main.rs` so it can be tested on any platform: the Win32 half
//! cannot run in CI, and the half that parses a byte stream and formats a
//! number is exactly where the bugs are. `main.rs` is windows-only; this is not.

pub mod art;
pub mod catalog;
/// Settings offered as choices computed from the machine, for the many users
/// who cannot be expected to know what a good thread count is.
pub mod choices;
pub mod client;
pub mod models;
/// Where every control lives: four pages, and the id of each thing on them.
pub mod nav;
pub mod settings;
/// The design tokens. Nothing outside this module names a colour.
pub mod theme;
/// Raw Win32, shared with `chaos-setup` so there is one set of declarations
/// rather than two that can drift apart.
#[cfg(windows)]
pub mod win32;
