//! Key/value cache — the difference between quadratic and linear generation.
//!
//! Without a cache, producing token *n* means recomputing attention for all
//! *n* previous positions, so generating a response costs O(n²). The
//! measurement that motivated this: 5 tokens from Qwen3-30B-A3B took 31,032
//! expert reads, against roughly 1,152 needed for a single position.
//!
//! Keys and values for a position never change once computed — they depend on
//! that token and its position, not on anything that follows. So they are
//! computed once and kept, and each new token attends over the stored history
//! while computing Q, K and V for itself alone.
//!
//! # What this costs
//!
//! Memory, linear in context length: `2 * n_kv_heads * head_dim * n_layer`
//! floats per position. For Qwen3-30B-A3B that is 48 layers x 4 kv heads x
//! 128 dims x 2 x 4 bytes = 196 KiB per token — trivial next to re-reading
//! gigabytes of experts, which is what it replaces.

/// Stored keys and values for every layer.
pub struct KvCache {
    /// Per layer, laid out `[head_dim * n_kv_heads]` per position, appended.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    n_positions: usize,
    per_position: usize,
}

impl KvCache {
    pub fn new(n_layer: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        KvCache {
            k: vec![Vec::new(); n_layer],
            v: vec![Vec::new(); n_layer],
            n_positions: 0,
            per_position: n_kv_heads * head_dim,
        }
    }

    /// Positions currently held.
    pub fn len(&self) -> usize {
        self.n_positions
    }

    pub fn is_empty(&self) -> bool {
        self.n_positions == 0
    }

    /// Floats stored per position per layer.
    pub fn per_position(&self) -> usize {
        self.per_position
    }

    /// Total bytes held, for reporting against the RAM budget.
    pub fn bytes(&self) -> usize {
        let f = std::mem::size_of::<f32>();
        self.k.iter().map(|v| v.len() * f).sum::<usize>()
            + self.v.iter().map(|v| v.len() * f).sum::<usize>()
    }

    /// Append one position's keys and values for `layer`.
    ///
    /// Returns an error rather than corrupting the cache when the slice is the
    /// wrong length — a short append would silently misalign every later
    /// position, and attention would read across position boundaries.
    pub fn push(&mut self, layer: usize, k: &[f32], v: &[f32]) -> Result<(), KvError> {
        if k.len() != self.per_position || v.len() != self.per_position {
            return Err(KvError::WrongSize {
                expected: self.per_position,
                got_k: k.len(),
                got_v: v.len(),
            });
        }
        self.k[layer].extend_from_slice(k);
        self.v[layer].extend_from_slice(v);
        Ok(())
    }

    /// Mark that a position has been appended to every layer.
    ///
    /// Separate from [`Self::push`] so the count advances once per token
    /// rather than once per layer.
    pub fn advance(&mut self) {
        self.n_positions += 1;
    }

    /// Advance by several positions at once — prefill appends a whole
    /// prompt before any token is generated.
    pub fn advance_by(&mut self, n: usize) {
        self.n_positions += n;
    }

    pub fn keys(&self, layer: usize) -> &[f32] {
        &self.k[layer]
    }

    pub fn values(&self, layer: usize) -> &[f32] {
        &self.v[layer]
    }

    /// Drop everything, for starting a new sequence.
    pub fn clear(&mut self) {
        for v in self.k.iter_mut().chain(self.v.iter_mut()) {
            v.clear();
        }
        self.n_positions = 0;
    }

    /// Whether every layer holds the same number of positions.
    ///
    /// A layer falling behind means some layer silently skipped its append,
    /// which would make attention read stale history for the rest of the run.
    pub fn is_consistent(&self) -> bool {
        let expected = self.n_positions * self.per_position;
        self.k.iter().all(|v| v.len() == expected) && self.v.iter().all(|v| v.len() == expected)
    }
}

#[derive(Debug)]
pub enum KvError {
    WrongSize {
        expected: usize,
        got_k: usize,
        got_v: usize,
    },
}

impl std::fmt::Display for KvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvError::WrongSize { expected, got_k, got_v } => write!(
                f,
                "kv cache expected {expected} floats per position, got k={got_k} v={got_v}"
            ),
        }
    }
}

impl std::error::Error for KvError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> KvCache {
        // 2 layers, 4 kv heads, head_dim 8 -> 32 floats per position.
        KvCache::new(2, 4, 8)
    }

    #[test]
    fn starts_empty_and_consistent() {
        let c = cache();
        assert!(c.is_empty());
        assert_eq!(c.per_position(), 32);
        assert!(c.is_consistent());
        assert_eq!(c.bytes(), 0);
    }

    #[test]
    fn appending_grows_each_layer_independently() {
        let mut c = cache();
        let k = vec![1.0f32; 32];
        let v = vec![2.0f32; 32];
        c.push(0, &k, &v).expect("layer 0");
        c.push(1, &k, &v).expect("layer 1");
        c.advance();

        assert_eq!(c.len(), 1);
        assert!(c.is_consistent());
        assert_eq!(c.keys(0).len(), 32);
        assert_eq!(c.values(1)[0], 2.0);
        // 2 layers * 2 tensors * 32 floats * 4 bytes
        assert_eq!(c.bytes(), 2 * 2 * 32 * 4);
    }

    #[test]
    fn history_accumulates_in_order() {
        let mut c = cache();
        for step in 0..3 {
            let k = vec![step as f32; 32];
            c.push(0, &k, &k).expect("push");
            c.push(1, &k, &k).expect("push");
            c.advance();
        }
        assert_eq!(c.len(), 3);
        let keys = c.keys(0);
        assert_eq!(keys.len(), 96);
        // Positions must stay in order; attention indexes by position.
        assert_eq!(keys[0], 0.0);
        assert_eq!(keys[32], 1.0);
        assert_eq!(keys[64], 2.0);
    }

    #[test]
    fn a_wrong_sized_append_is_refused() {
        // Accepting it would misalign every later position, and attention
        // would silently read across position boundaries.
        let mut c = cache();
        let short = vec![1.0f32; 16];
        assert!(c.push(0, &short, &short).is_err());
        assert!(c.is_consistent(), "a refused push must not mutate the cache");
    }

    #[test]
    fn inconsistency_is_detected() {
        let mut c = cache();
        let k = vec![1.0f32; 32];
        c.push(0, &k, &k).expect("layer 0 only");
        c.advance();
        // Layer 1 never received its position.
        assert!(!c.is_consistent(), "a lagging layer must be detected");
    }

    #[test]
    fn clearing_resets_for_a_new_sequence() {
        let mut c = cache();
        let k = vec![1.0f32; 32];
        c.push(0, &k, &k).unwrap();
        c.push(1, &k, &k).unwrap();
        c.advance();
        c.clear();
        assert!(c.is_empty());
        assert!(c.is_consistent());
        assert_eq!(c.bytes(), 0);
    }
}
