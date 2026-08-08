//! A frequency-gated cache for routed expert slices.
//!
//! # Why not recency
//!
//! Expert access is a **cyclic scan**: every block walks layer 0 to the last and
//! reads most of the experts each one routes to. When that cycle is larger than
//! the cache, recency is precisely the wrong signal — layer 0's slices are always
//! the oldest thing present when the last layer needs room, so they are evicted
//! immediately before the next block asks for them again. Measured on Qwen3, an
//! LRU-ish cache at 6.26 GiB returned a **17% hit rate with 20,975 evictions**,
//! worse than pinning an arbitrary fixed third would have given for free.
//! Frequency-gated admission took the same budget to **70%**.
//!
//! So admission is by frequency and nothing else: a newcomer must be wanted
//! **strictly more often** than the entry it would displace. That stops the churn
//! and lets the cache settle on genuinely hot experts.
//!
//! # Why it is worth caching at all
//!
//! Because the router is skewed and, more importantly, because the skew is
//! *stable within a prompt*. R0 measured that a hot set does **not** transfer
//! between prompts — pinned from one it covers 53.7% of another, and 37.5%
//! across subjects against 25% for caching at random — so nothing here should
//! ever be pre-loaded from a fixed list. R0.1 measured the regime this cache
//! actually runs in: a set warmed on the prompt covers **86.3%** of the routing
//! of the tokens generated after it. **Warm it; never pin it.**
//!
//! # Why it owns its memory
//!
//! Entries are heap allocations this process holds, not pages the kernel may
//! reclaim. That is deliberate and it is the whole architectural argument: past
//! ~6 GiB on Qwen3 an mmap-backed 71%-hit cache was the *slowest* configuration
//! measured, because cached bytes got paged out and a "hit" became a page fault
//! wearing a disguise. **A hit rate is not a tok/s**, and this module reports
//! both plus its footprint so the three are never confused.

use std::collections::HashMap;
use std::sync::Arc;

/// `(layer, tensor, expert)` packed into one integer.
///
/// A `(String, u32)` key would allocate on every lookup, and there are
/// `n_layer * 3 * n_expert_used` lookups per token — 774 on V4-Flash — on the
/// path whose largest measured cost was already memcpy.
pub type SliceKey = u64;

/// Pack a key. `tensor` distinguishes gate/up/down within a layer.
pub fn slice_key(layer: u32, tensor: u8, expert: u32) -> SliceKey {
    (u64::from(layer) << 40) | (u64::from(tensor) << 32) | u64::from(expert)
}

/// Hits, misses and footprint — reported together, never one alone.
#[derive(Debug, Default, Clone, Copy)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub admissions: u64,
    /// Bytes served from memory instead of read from disk.
    pub bytes_saved: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            0.0
        } else {
            self.hits as f64 / n as f64
        }
    }
}

pub struct ExpertCache {
    entries: HashMap<SliceKey, Arc<[u8]>>,
    /// How many *selections* each slice has served, hit or miss. Counting only
    /// hits would make an entry that is never admitted look unwanted forever;
    /// counting requests rather than selections would make every expert on a
    /// long prompt look equally wanted, because reads are deduplicated.
    freq: HashMap<SliceKey, u32>,
    /// Cached keys, for sampling eviction candidates by index. May hold keys
    /// already evicted; those are swap-removed when sampled.
    keys: Vec<SliceKey>,
    bytes: usize,
    budget: usize,
    rng: u64,
    stats: CacheStats,
}

impl ExpertCache {
    pub fn new(budget: usize) -> Self {
        ExpertCache {
            entries: HashMap::new(),
            freq: HashMap::new(),
            keys: Vec::new(),
            bytes: 0,
            budget,
            rng: 0x2545_F491_4F6C_DD1D,
            stats: CacheStats::default(),
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Ask for a slice, worth `weight` selections. Counted whether or not present.
    ///
    /// `weight` matters because expert reads are **deduplicated per block over
    /// the whole batch**: an expert chosen by ninety tokens and one chosen by a
    /// single token are both read once, so counting requests would score them
    /// equally. On a long prompt nearly every expert is requested in every pass,
    /// every count ties, and the cache freezes on whatever arrived first —
    /// keeping layer 0's slices forever because nothing can ever beat them.
    ///
    /// Weighting by selections is what makes the skew R0 measured visible to the
    /// policy at all.
    pub fn request(&mut self, key: SliceKey, weight: u32) -> Option<Arc<[u8]>> {
        *self.freq.entry(key).or_insert(0) += weight.max(1);
        match self.entries.get(&key) {
            Some(bytes) => {
                self.stats.hits += 1;
                self.stats.bytes_saved += bytes.len() as u64;
                Some(Arc::clone(bytes))
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Offer a freshly read slice, copying it **only if it is kept**.
    ///
    /// Taking `&[u8]` rather than an `Arc` matters: the caller reads straight
    /// into the packed, sector-skewed destination buffer — the arrangement that
    /// made reads 0.80 → 1.58 GiB/s — and most offers are refused once the cache
    /// is warm. Allocating first and deciding after would pay for every refusal.
    pub fn offer(&mut self, key: SliceKey, src: &[u8]) {
        if src.len() > self.budget {
            return;
        }
        let mine = self.freq.get(&key).copied().unwrap_or(1);

        while self.bytes + src.len() > self.budget {
            let Some(victim) = self.weakest(8) else {
                return;
            };
            let theirs = self.freq.get(&victim).copied().unwrap_or(1);
            if mine <= theirs {
                return; // the incumbent is wanted at least as often; leave it
            }
            if let Some(bytes) = self.entries.remove(&victim) {
                self.bytes -= bytes.len();
                self.stats.evictions += 1;
            } else {
                return;
            }
        }

        self.bytes += src.len();
        self.keys.push(key);
        self.entries.insert(key, Arc::from(src));
        self.stats.admissions += 1;
    }

    /// The least-wanted of a small random sample.
    ///
    /// Sampling reads `keys` by index. Iterating the map with `step_by` would
    /// look equivalent and be O(n) — `step_by` still calls `next` for every
    /// element it skips — which at one eviction per miss is a full scan of the
    /// cache thousands of times per token.
    fn weakest(&mut self, sample: usize) -> Option<SliceKey> {
        let mut best: Option<(SliceKey, u32)> = None;
        let mut tries = 0;
        while tries < sample * 2 && !self.keys.is_empty() {
            tries += 1;
            let i = (self.next_rand() as usize) % self.keys.len();
            let key = self.keys[i];
            if !self.entries.contains_key(&key) {
                self.keys.swap_remove(i); // stale: evicted earlier
                continue;
            }
            let uses = self.freq.get(&key).copied().unwrap_or(1);
            if best.is_none_or(|(_, b)| uses < b) {
                best = Some((key, uses));
            }
        }
        best.map(|(k, _)| k)
    }

    /// xorshift64: enough randomness to spread the sample, no dependency.
    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_do_not_collide_across_layers_tensors_or_experts() {
        let a = slice_key(3, 0, 7);
        assert_ne!(a, slice_key(3, 1, 7), "tensor must matter");
        assert_ne!(a, slice_key(4, 0, 7), "layer must matter");
        assert_ne!(a, slice_key(3, 0, 8), "expert must matter");
        // The packing must survive a realistic model's ranges.
        assert_ne!(slice_key(42, 2, 255), slice_key(42, 2, 254));
    }

    #[test]
    fn a_miss_then_an_offer_becomes_a_hit() {
        let mut c = ExpertCache::new(1024);
        let k = slice_key(0, 0, 1);
        assert!(c.request(k, 1).is_none());
        c.offer(k, &[7u8; 64]);
        assert_eq!(c.request(k, 1).as_deref(), Some(&[7u8; 64][..]));
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 1);
        assert_eq!(c.bytes(), 64);
    }

    /// The property that separates this from LRU: a slice asked for once must
    /// not evict one asked for many times, however recently it arrived.
    #[test]
    fn a_rare_newcomer_cannot_displace_a_frequent_incumbent() {
        let mut c = ExpertCache::new(64);
        let hot = slice_key(0, 0, 1);
        for _ in 0..10 {
            c.request(hot, 1);
        }
        c.offer(hot, &[1u8; 64]);
        assert_eq!(c.bytes(), 64, "cache is now full");

        let cold = slice_key(0, 0, 2);
        c.request(cold, 1);
        c.offer(cold, &[2u8; 64]);

        assert_eq!(c.request(hot, 1).as_deref(), Some(&[1u8; 64][..]));
        assert_eq!(c.stats().evictions, 0, "nothing should have been evicted");
    }

    /// ...and the converse, or the cache could never adapt to a new prompt.
    #[test]
    fn a_frequent_newcomer_does_displace_a_rare_incumbent() {
        let mut c = ExpertCache::new(64);
        let cold = slice_key(0, 0, 1);
        c.request(cold, 1);
        c.offer(cold, &[1u8; 64]);

        let hot = slice_key(0, 0, 2);
        for _ in 0..10 {
            c.request(hot, 1);
        }
        c.offer(hot, &[2u8; 64]);

        assert_eq!(c.request(hot, 1).as_deref(), Some(&[2u8; 64][..]));
        assert_eq!(c.stats().evictions, 1);
    }

    /// A budget smaller than one slice must refuse everything rather than
    /// evicting forever — the loop condition can never be satisfied.
    #[test]
    fn a_slice_larger_than_the_budget_is_refused_without_looping() {
        let mut c = ExpertCache::new(16);
        let k = slice_key(1, 1, 1);
        c.request(k, 1);
        c.offer(k, &[0u8; 64]);
        assert_eq!(c.bytes(), 0);
        assert_eq!(c.stats().admissions, 0);
    }

    #[test]
    fn footprint_never_exceeds_the_budget() {
        let mut c = ExpertCache::new(256);
        for e in 0..64u32 {
            let k = slice_key(0, 0, e);
            for _ in 0..=e {
                c.request(k, 1); // later experts are progressively hotter
            }
            c.offer(k, &[e as u8; 64]);
            assert!(c.bytes() <= c.budget(), "budget exceeded at expert {e}");
        }
        assert!(c.stats().hit_rate() >= 0.0);
    }
}

#[cfg(test)]
mod weighting_tests {
    use super::*;

    /// The case dedup creates: two experts, each requested once per pass, but
    /// one chosen by many tokens and the other by one. Counting requests would
    /// tie them and freeze the cache on whichever arrived first; counting
    /// selections lets the hot one take the slot.
    #[test]
    fn selections_not_requests_decide_who_stays() {
        let mut c = ExpertCache::new(64);
        let first = slice_key(0, 0, 1);
        let hot = slice_key(0, 0, 2);

        // One pass each. Same number of *requests*, very different weights.
        c.request(first, 1);
        c.offer(first, &[1u8; 64]);
        c.request(hot, 90);
        c.offer(hot, &[2u8; 64]);

        assert_eq!(
            c.request(hot, 1).as_deref(),
            Some(&[2u8; 64][..]),
            "the expert 90 tokens chose should have displaced the one 1 token chose"
        );
        assert_eq!(c.stats().evictions, 1);
    }

    /// A weight of zero must still count as one, or an expert could be requested
    /// forever and never become eligible for admission.
    #[test]
    fn a_zero_weight_still_counts_once() {
        let mut c = ExpertCache::new(64);
        let k = slice_key(0, 0, 1);
        c.request(k, 0);
        c.request(k, 0);
        c.offer(k, &[3u8; 64]);
        assert_eq!(c.request(k, 1).as_deref(), Some(&[3u8; 64][..]));
    }
}
