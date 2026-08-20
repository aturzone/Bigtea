//! What inference costs in RAM *besides* the weights.
//!
//! This used to be a flat 3 GiB constant, which was wrong twice over. It
//! double-counted the operating system — `available` RAM already excludes what
//! the OS holds — and it ignored the fact that the dominant term, the KV cache,
//! varies by three orders of magnitude with context length and attention shape.
//!
//! Charging a flat 3 GiB against a 16 GiB machine wastes ~2 GiB of budget, and
//! on a machine this size 2 GiB is the difference between the dense weights
//! being resident and being re-read on every token. So it is computed.
//!
//! Two terms:
//!
//! * **KV cache** — grows linearly with context. This is the term that
//!   actually matters, and it is why a model that runs comfortably at 4K
//!   context can be impossible at 128K on the same machine.
//! * **Scratch** — activation buffers for one token. Roughly constant, sized
//!   from the hidden and FFN dimensions.

use crate::GIB;

/// Bytes per element in the KV cache.
///
/// `f16` is the common default. Engines that quantize the KV cache to 8 or
/// even 4 bits change this directly, which is why it is a parameter rather
/// than baked in.
pub const KV_BYTES_F16: u64 = 2;

#[derive(Debug, Clone, Copy)]
pub struct Overhead {
    pub kv_cache_bytes: u64,
    pub scratch_bytes: u64,
}

impl Overhead {
    pub fn total(&self) -> u64 {
        self.kv_cache_bytes + self.scratch_bytes
    }
}

impl std::fmt::Display for Overhead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.2} GiB (kv {:.2} + scratch {:.2})",
            self.total() as f64 / GIB as f64,
            self.kv_cache_bytes as f64 / GIB as f64,
            self.scratch_bytes as f64 / GIB as f64
        )
    }
}

/// Shape inputs needed to size the runtime footprint.
#[derive(Debug, Clone, Copy)]
pub struct AttentionShape {
    pub n_layers: u64,
    /// Key/value heads. Grouped-query and multi-query attention shrink this
    /// dramatically — DeepSeek-V4-Flash uses **1**, which is why its KV cache
    /// is tiny for its size.
    pub n_kv_heads: u64,
    /// Dimension per key/value head.
    pub head_dim: u64,
    pub hidden_size: u64,
    /// FFN intermediate width, used only to size scratch.
    pub ffn_intermediate: u64,
}

/// KV cache bytes for `context_len` tokens.
///
/// Both a key and a value are stored per head per layer per token.
pub fn kv_cache_bytes(shape: &AttentionShape, context_len: u64, bytes_per_elem: u64) -> u64 {
    2 * shape.n_kv_heads * shape.head_dim * shape.n_layers * context_len * bytes_per_elem
}

/// Activation scratch for a single token.
///
/// Bounded by the widest intermediate tensors a layer materialises. Generous
/// by design — under-estimating scratch means an out-of-memory failure at the
/// worst possible moment, while over-estimating costs a little budget.
pub fn scratch_bytes(shape: &AttentionShape) -> u64 {
    // Gate, up and down projections of one MoE expert, plus room for the
    // attention intermediates, at f32.
    let ffn = 3 * shape.hidden_size.max(1) * shape.ffn_intermediate.max(1) * 4;
    let attn = 4 * shape.hidden_size.max(1) * shape.hidden_size.max(1) * 4;
    // Floor it: a tiny model still needs working room for buffers and I/O.
    (ffn + attn).max(256 << 20)
}

/// Full runtime overhead for a given context length.
pub fn overhead(shape: &AttentionShape, context_len: u64, kv_bytes_per_elem: u64) -> Overhead {
    Overhead {
        kv_cache_bytes: kv_cache_bytes(shape, context_len, kv_bytes_per_elem),
        scratch_bytes: scratch_bytes(shape),
    }
}

/// The largest context that leaves at least `weights_bytes` of the budget for
/// weights.
///
/// This inverts the usual question. Rather than "does 128K fit", it answers
/// "given that I want the dense weights resident, how much context can I
/// afford" — which is the trade a user on a small machine actually faces.
pub fn max_context_for_budget(
    shape: &AttentionShape,
    ram_budget_bytes: u64,
    weights_bytes: u64,
    kv_bytes_per_elem: u64,
) -> u64 {
    let scratch = scratch_bytes(shape);
    let Some(left) = ram_budget_bytes
        .checked_sub(weights_bytes)
        .and_then(|v| v.checked_sub(scratch))
    else {
        return 0;
    };
    let per_token = 2 * shape.n_kv_heads * shape.head_dim * shape.n_layers * kv_bytes_per_elem;
    if per_token == 0 {
        return 0;
    }
    left / per_token
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek-V4-Flash, from its published config.
    fn v4_flash() -> AttentionShape {
        AttentionShape {
            n_layers: 43,
            n_kv_heads: 1, // multi-query: the reason its KV cache is small
            head_dim: 512,
            hidden_size: 4096,
            ffn_intermediate: 2048,
        }
    }

    #[test]
    fn kv_cache_is_small_at_short_context() {
        // 2 * 1 head * 512 * 43 layers * 2 bytes = 88,064 bytes per token.
        let per_token = kv_cache_bytes(&v4_flash(), 1, KV_BYTES_F16);
        assert_eq!(per_token, 88_064);

        // At 4K context that is well under a gigabyte -- nothing like the flat
        // 3 GiB that used to be charged.
        let at_4k = kv_cache_bytes(&v4_flash(), 4096, KV_BYTES_F16);
        assert!(
            at_4k < GIB / 2,
            "4K KV was {:.2} GiB",
            at_4k as f64 / GIB as f64
        );
    }

    #[test]
    fn kv_cache_grows_linearly_and_dominates_at_long_context() {
        let shape = v4_flash();
        let at_4k = kv_cache_bytes(&shape, 4096, KV_BYTES_F16);
        let at_128k = kv_cache_bytes(&shape, 131_072, KV_BYTES_F16);
        assert_eq!(at_128k, at_4k * 32);
        // At 128K it is the single largest runtime cost on a 16 GiB machine.
        assert!(at_128k > 10 * GIB);
    }

    #[test]
    fn multi_query_attention_is_why_this_model_is_tractable() {
        let mqa = v4_flash();
        let mha = AttentionShape {
            n_kv_heads: 64,
            ..mqa
        };
        let a = kv_cache_bytes(&mqa, 8192, KV_BYTES_F16);
        let b = kv_cache_bytes(&mha, 8192, KV_BYTES_F16);
        assert_eq!(b, a * 64, "64 kv heads should cost 64x the cache");
    }

    #[test]
    fn quantizing_the_kv_cache_halves_it() {
        let shape = v4_flash();
        let f16 = kv_cache_bytes(&shape, 32768, 2);
        let q8 = kv_cache_bytes(&shape, 32768, 1);
        assert_eq!(f16, q8 * 2);
    }

    #[test]
    fn total_overhead_at_4k_is_far_below_the_old_flat_constant() {
        // The whole point of computing this: the old 3 GiB guess cost ~2 GiB
        // of weight budget on a machine that has none to spare.
        let o = overhead(&v4_flash(), 4096, KV_BYTES_F16);
        assert!(
            o.total() < 2 * GIB,
            "overhead at 4K was {o}, expected well under 2 GiB"
        );
    }

    #[test]
    fn max_context_answers_the_question_a_small_machine_actually_asks() {
        let shape = v4_flash();
        // 12 GiB budget, 7.07 GiB of dense weights resident.
        let weights = (7.07 * GIB as f64) as u64;
        let ctx = max_context_for_budget(&shape, 12 * GIB, weights, KV_BYTES_F16);
        assert!(ctx > 4096, "should afford more than 4K, got {ctx}");

        // If the weights alone exceed the budget, no context fits.
        assert_eq!(
            max_context_for_budget(&shape, 4 * GIB, 8 * GIB, KV_BYTES_F16),
            0
        );
    }

    #[test]
    fn scratch_has_a_floor_so_tiny_models_still_get_working_room() {
        let tiny = AttentionShape {
            n_layers: 2,
            n_kv_heads: 1,
            head_dim: 8,
            hidden_size: 16,
            ffn_intermediate: 16,
        };
        assert!(scratch_bytes(&tiny) >= 256 << 20);
    }
}
