//! Model architectures — the part that is genuinely per-model.
//!
//! Everything below this crate is architecture-agnostic: containers, residency,
//! streaming, tokenization, the ggml graph API. This is where a specific
//! model's shape lives, and adding support for a new family means adding a
//! module here rather than touching the engine.
//!
//! # ggml's layout convention, because it is the main source of confusion
//!
//! `ne[0]` is the *fastest-moving* dimension. A weight that maps `n_in` to
//! `n_out` is therefore stored with `ne0 = n_in`, `ne1 = n_out`, and
//! `mul_mat(w, x)` with `x` shaped `[n_in, n_tokens]` yields
//! `[n_out, n_tokens]`. Reading these shapes as row-major — the intuition most
//! people bring — transposes every matrix and produces confident nonsense.

pub mod banner;
mod deepseek4;
mod deepseek4_forward;
mod expert_cache;
mod kv;
pub mod log;
mod logo_bitmap;
mod qwen3;
pub mod sample;
pub mod spectrum;
mod stream;
pub mod ui;

pub use deepseek4::{AttentionKind, Deepseek4Config, Deepseek4Model};
pub use deepseek4_forward::{
    forward, prefill, routing_last_token, routing_last_token_reset, routing_next_pass,
    routing_report, routing_weight_report, step, Deepseek4Cache, Deepseek4Forward, RepackedDense,
};
pub use expert_cache::{CacheStats, ExpertCache};
pub use kv::{KvCache, KvError, KvType};
pub use qwen3::{architecture_is_verified, Qwen3Config, Qwen3Model, VERIFIED_ARCHITECTURES};
pub use sample::{neg_log_prob, Sampler, SamplerConfig, SamplerStage};
pub use stream::{
    configured_threads, configured_threads_batch, StreamStats, StreamingRunner, TensorOverride,
};

use std::fmt;

#[derive(Debug)]
pub enum ArchError {
    /// The container declares an architecture we have no implementation for.
    Unsupported(String),
    /// A tensor the architecture requires is absent from the container.
    MissingTensor(String),
    /// Metadata needed to build the graph is absent.
    MissingMetadata(String),
    Model(chaos_model::Error),
    Ggml(chaos_ggml::GgmlError),
    /// The KV cache rejected an append — see [`kv::KvError`].
    Kv(kv::KvError),
    /// More tokens in **one pass** than the raw latent ring can hold at once.
    ///
    /// No longer a limit on the sequence: `limit` is the largest batch, and a
    /// longer prompt is chunked rather than refused.
    ContextTooLong {
        tokens: usize,
        limit: usize,
    },
    /// A path that is deliberately refused rather than silently approximated.
    Unimplemented(&'static str),
    /// A `--override-tensor` rule this build cannot honour, said out loud.
    ///
    /// Owned rather than `&'static str` because the useful message quotes the
    /// user's own pattern back at them: a rule that matches nothing looks
    /// identical to one that was never given.
    BadOverride(String),
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchError::Unsupported(a) => write!(
                f,
                "no implementation for architecture {a:?} (add one in chaos-arch)"
            ),
            ArchError::MissingTensor(t) => write!(f, "container has no tensor {t:?}"),
            ArchError::MissingMetadata(k) => write!(f, "container has no metadata key {k:?}"),
            ArchError::Model(e) => write!(f, "{e}"),
            ArchError::Ggml(e) => write!(f, "{e}"),
            ArchError::Kv(e) => write!(f, "{e}"),
            ArchError::Unimplemented(what) => write!(f, "not implemented: {what}"),
            ArchError::BadOverride(why) => write!(f, "--override-tensor {why}"),
            ArchError::ContextTooLong { tokens, limit } => write!(
                f,
                "{tokens} tokens in one pass; this path takes {limit} at a time. \
                 This is a limit on the batch, not on the sequence -- \
                 DeepSeek-V4-Flash holds its raw latents in a ring, so the \
                 conversation may be far longer than {limit}. Prefill in blocks \
                 of {limit} or fewer (-b)."
            ),
        }
    }
}

impl std::error::Error for ArchError {}

impl From<chaos_model::Error> for ArchError {
    fn from(e: chaos_model::Error) -> Self {
        ArchError::Model(e)
    }
}

impl From<kv::KvError> for ArchError {
    fn from(e: kv::KvError) -> Self {
        ArchError::Kv(e)
    }
}

impl From<chaos_ggml::GgmlError> for ArchError {
    fn from(e: chaos_ggml::GgmlError) -> Self {
        ArchError::Ggml(e)
    }
}

pub type Result<T> = std::result::Result<T, ArchError>;
