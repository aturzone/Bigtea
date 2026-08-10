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

mod deepseek4;
mod deepseek4_forward;
mod expert_cache;
mod kv;
mod qwen3;
pub mod sample;
pub mod spectrum;
mod stream;

pub use deepseek4::{AttentionKind, Deepseek4Config, Deepseek4Model};
pub use deepseek4_forward::{
    forward, prefill, routing_last_token, routing_last_token_reset, routing_next_pass,
    routing_report, routing_weight_report, step, Deepseek4Cache, Deepseek4Forward,
};
pub use expert_cache::{CacheStats, ExpertCache};
pub use kv::{KvCache, KvError};
pub use qwen3::{architecture_is_verified, Qwen3Config, Qwen3Model, VERIFIED_ARCHITECTURES};
pub use sample::{Sampler, SamplerConfig};
pub use stream::{StreamStats, StreamingRunner};

use std::fmt;

#[derive(Debug)]
pub enum ArchError {
    /// The container declares an architecture we have no implementation for.
    Unsupported(String),
    /// A tensor the architecture requires is absent from the container.
    MissingTensor(String),
    /// Metadata needed to build the graph is absent.
    MissingMetadata(String),
    Model(bigtea_model::Error),
    Ggml(bigtea_ggml::GgmlError),
    /// The KV cache rejected an append — see [`kv::KvError`].
    Kv(kv::KvError),
    /// More tokens than the attention window this architecture can hold.
    ContextTooLong {
        tokens: usize,
        limit: usize,
    },
    /// A path that is deliberately refused rather than silently approximated.
    Unimplemented(&'static str),
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchError::Unsupported(a) => write!(
                f,
                "no implementation for architecture {a:?} (add one in bigtea-arch)"
            ),
            ArchError::MissingTensor(t) => write!(f, "container has no tensor {t:?}"),
            ArchError::MissingMetadata(k) => write!(f, "container has no metadata key {k:?}"),
            ArchError::Model(e) => write!(f, "{e}"),
            ArchError::Ggml(e) => write!(f, "{e}"),
            ArchError::Kv(e) => write!(f, "{e}"),
            ArchError::Unimplemented(what) => write!(f, "not implemented: {what}"),
            ArchError::ContextTooLong { tokens, limit } => write!(
                f,
                "prompt is {tokens} tokens; this path holds {limit}. \
                 DeepSeek-V4-Flash builds its attention cache for the whole \
                 sequence at once, so {limit} is the ceiling until the KV cache \
                 lands. Shorten the prompt, or use -f with a smaller file."
            ),
        }
    }
}

impl std::error::Error for ArchError {}

impl From<bigtea_model::Error> for ArchError {
    fn from(e: bigtea_model::Error) -> Self {
        ArchError::Model(e)
    }
}

impl From<kv::KvError> for ArchError {
    fn from(e: kv::KvError) -> Self {
        ArchError::Kv(e)
    }
}

impl From<bigtea_ggml::GgmlError> for ArchError {
    fn from(e: bigtea_ggml::GgmlError) -> Self {
        ArchError::Ggml(e)
    }
}

pub type Result<T> = std::result::Result<T, ArchError>;
