//! What a model costs to run, reduced to the three numbers that decide it.
//!
//! Everything downstream needs exactly this much about a model:
//!
//! * `dense_bytes` — read on **every** token. Attention, routers, shared
//!   experts, embeddings. If this fits in RAM it is paid once; if not, the
//!   shortfall is re-read every token and usually dominates everything else.
//! * `expert_pool_bytes` — every routed expert. Sets the disk footprint but is
//!   never read in full.
//! * `expert_bytes_per_token` — the slice routing actually selects.
//!
//! A profile can be built two ways, and the distinction is recorded because it
//! changes how much to trust the answer:
//!
//! * [`ProfileSource::TensorIndex`] — summed from a real GGUF index. Exact.
//! * [`ProfileSource::Architecture`] — derived from published config metadata,
//!   before any weights exist locally. Approximate, and the only option when
//!   the point is to decide whether to download at all.

use bigtea_gguf::Gguf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileSource {
    /// Summed from a real container's tensor index — exact byte counts.
    TensorIndex,
    /// Derived from architecture metadata — no weights needed.
    Architecture,
}

impl std::fmt::Display for ProfileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileSource::TensorIndex => f.write_str("measured from tensor index"),
            ProfileSource::Architecture => f.write_str("estimated from architecture"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelProfile {
    pub name: String,
    /// Bytes read on every token regardless of routing.
    pub dense_bytes: u64,
    /// Bytes of every routed expert in the model.
    pub expert_pool_bytes: u64,
    /// Bytes of the experts one token selects.
    pub expert_bytes_per_token: u64,
    pub n_experts: u64,
    pub n_experts_used: u64,
    pub source: ProfileSource,
}

#[derive(Debug)]
pub enum ProfileError {
    /// The container did not declare how many experts exist or are used, so
    /// the per-token slice cannot be derived.
    MissingExpertCounts,
    /// Expert routing counts that cannot be true.
    ImpossibleRouting { used: u64, total: u64 },
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileError::MissingExpertCounts => f.write_str(
                "container declares no expert_count/expert_used_count, so \
                 per-token cost cannot be derived",
            ),
            ProfileError::ImpossibleRouting { used, total } => {
                write!(f, "routing declares {used} of {total} experts used")
            }
        }
    }
}

impl std::error::Error for ProfileError {}

impl ModelProfile {
    /// Total bytes on disk.
    pub fn container_bytes(&self) -> u64 {
        self.dense_bytes + self.expert_pool_bytes
    }

    /// Fraction of the model a single token touches. The lower this is, the
    /// more a machine far smaller than the model can still run it.
    pub fn sparsity(&self) -> f64 {
        let total = self.container_bytes();
        if total == 0 {
            return 0.0;
        }
        (self.dense_bytes + self.expert_bytes_per_token) as f64 / total as f64
    }

    /// Build from a parsed GGUF container.
    ///
    /// Uses real tensor sizes, so quantization block overheads are already
    /// included rather than approximated. Note that for a *sharded* model this
    /// reflects only the shards parsed so far — the caller should merge shards
    /// before trusting the totals.
    pub fn from_gguf(gguf: &Gguf, name: impl Into<String>) -> Result<Self, ProfileError> {
        let arch = gguf.architecture().unwrap_or("").to_string();
        let n_experts = gguf
            .get_u64(&format!("{arch}.expert_count"))
            .ok_or(ProfileError::MissingExpertCounts)?;
        let n_used = gguf
            .get_u64(&format!("{arch}.expert_used_count"))
            .ok_or(ProfileError::MissingExpertCounts)?;
        if n_experts == 0 || n_used == 0 || n_used > n_experts {
            return Err(ProfileError::ImpossibleRouting {
                used: n_used,
                total: n_experts,
            });
        }

        let (expert_pool_bytes, dense_bytes) = gguf.expert_vs_dense_bytes();
        // Experts within a layer are the same shape, so the per-token slice is
        // exactly the used/total fraction of the pool.
        let expert_bytes_per_token = expert_pool_bytes / n_experts * n_used;

        Ok(ModelProfile {
            name: name.into(),
            dense_bytes,
            expert_pool_bytes,
            expert_bytes_per_token,
            n_experts,
            n_experts_used: n_used,
            source: ProfileSource::TensorIndex,
        })
    }

    /// Build from architecture metadata plus a quantization, with no weights.
    ///
    /// This is the pre-download path: it answers "should I spend hours
    /// fetching this" from a few kilobytes of published config.
    ///
    /// `dense_bits` models "dynamic" quantizations, which deliberately keep
    /// attention and routers at higher precision than the routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn from_architecture(
        name: impl Into<String>,
        dense_params: u64,
        params_per_expert: u64,
        n_moe_layers: u64,
        n_experts: u64,
        n_experts_used: u64,
        expert_bits: f64,
        dense_bits: f64,
    ) -> Result<Self, ProfileError> {
        if n_experts == 0 || n_experts_used == 0 || n_experts_used > n_experts {
            return Err(ProfileError::ImpossibleRouting {
                used: n_experts_used,
                total: n_experts,
            });
        }
        let bytes = |params: u64, bits: f64| (params as f64 * bits / 8.0) as u64;
        let pool_params = n_moe_layers * n_experts * params_per_expert;
        let per_token_params = n_moe_layers * n_experts_used * params_per_expert;

        Ok(ModelProfile {
            name: name.into(),
            dense_bytes: bytes(dense_params, dense_bits),
            expert_pool_bytes: bytes(pool_params, expert_bits),
            expert_bytes_per_token: bytes(per_token_params, expert_bits),
            n_experts,
            n_experts_used,
            source: ProfileSource::Architecture,
        })
    }
}
