//! Predict whether a model runs on a machine, and how fast.
//!
//! # The model of the world
//!
//! Decoding a MoE model too large for RAM is **storage-bandwidth bound**, not
//! compute bound. Each token must read its activated slice of weights, so:
//!
//! ```text
//! tok/s = efficiency * bandwidth / (dense_shortfall + expert_bytes_per_token)
//! ```
//!
//! where `dense_shortfall` is the part of the always-read dense weights that
//! does not fit in RAM and is therefore re-read every single token.
//!
//! # Two properties that invert the obvious advice
//!
//! Both are asserted as tests, because both are easy to regress and each one
//! changes what a user should actually do:
//!
//! 1. **Spend spare RAM on the dense part, not on an expert cache.** Dense
//!    residency pays back linearly and unconditionally — every resident byte
//!    is a byte not re-read. An expert cache pays nothing at all until it can
//!    hold a whole token's working set, because below that entries are evicted
//!    before they are reused.
//! 2. **Pruning the expert pool does not make a model faster.** It shrinks the
//!    download and the disk footprint, but a token still routes to the same
//!    number of experts, so bytes-per-token is unchanged. Pruning buys a
//!    cheaper model to obtain, never a quicker one to run.

mod predict;
mod profile;
mod residency;

pub use predict::{Prediction, DEFAULT_EFFICIENCY, DEFAULT_OVERHEAD_BYTES};
pub use profile::{ModelProfile, ProfileError, ProfileSource};
pub use residency::{plan_from_gguf, plan_layout, Layout, Placed, Placement};

pub const GIB: u64 = 1 << 30;
