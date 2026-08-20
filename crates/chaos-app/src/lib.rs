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
/// Watching a download by the bytes it puts on disk, since the downloader is
/// another process with no console.
pub mod download;
/// Watching a model load by the memory it takes, since "loading" with no
/// number is a window that looks broken.
pub mod loading;
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
