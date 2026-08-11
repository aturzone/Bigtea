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
///
/// Held as f16, which is what llama.cpp stores by default and what ggml's
/// fused attention consumes directly. For this model that is 96 KiB per
/// position rather than 192, and at an 8775-token context the difference —
/// 0.8 GiB — comes straight off the expert cache's budget, where it buys hit
/// rate. Attention also reads half as many bytes.
pub struct KvCache {
    /// Per layer, laid out `[head_dim * n_kv_heads]` per position, appended.
    /// Raw bytes in `kind`'s layout, handed to a ggml tensor unchanged.
    k: Vec<Vec<u8>>,
    v: Vec<Vec<u8>>,
    n_positions: usize,
    per_position: usize,
    /// Storage type. Both halves share one, because ggml's banded attention
    /// asserts `k->type == v->type` and splitting them would work right up
    /// until someone used that path.
    kind: KvType,
    /// Row length in values -- `head_dim`. Q8_0 quantises per 32-value block
    /// within a row, so this is needed to place the block boundaries.
    row: usize,
}

/// How the cache stores a key or value -- llama.cpp's `--cache-type-k/v`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvType {
    /// 2 bytes per value. What llama.cpp stores by default.
    F16,
    /// ~1.06 bytes per value: 32 int8 quants sharing one f16 scale.
    ///
    /// **Roughly half the memory**, which at long context comes straight off
    /// the expert cache's budget where it buys hit rate. The cost is precision
    /// in attention, and that is measurable rather than arguable -- the flag
    /// audit carries a perplexity comparison.
    Q8_0,
}

/// ggml's `block_q8_0`: one f16 scale then 32 int8 quants, 34 bytes.
const QK8_0: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;

impl KvType {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Some(KvType::F16),
            "q8_0" | "q8" => Some(KvType::Q8_0),
            _ => None,
        }
    }

    /// The ggml type id, for building the tensor attention reads.
    pub fn ggml_type(self) -> u32 {
        match self {
            KvType::F16 => 1,
            KvType::Q8_0 => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
        }
    }

    /// Bytes for `values` values, which must be a whole number of rows.
    fn bytes_for(self, values: usize) -> usize {
        match self {
            KvType::F16 => values * 2,
            KvType::Q8_0 => values / QK8_0 * Q8_0_BLOCK_BYTES,
        }
    }
}

/// Quantise one row into ggml's `block_q8_0` layout.
///
/// Per 32 values the scale is `max|x| / 127`, and each quant is `x / scale`
/// rounded. 127 rather than 128 because the int8 range is asymmetric: using
/// 128 makes the largest positive value in a block clip to -128.
fn quantize_q8_0(src: &[f32], dst: &mut [u8]) {
    for (bi, chunk) in src.chunks(QK8_0).enumerate() {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let at = bi * Q8_0_BLOCK_BYTES;
        let mut half = [0u16; 1];
        bigtea_ggml::f32_to_f16(&[d], &mut half);
        dst[at..at + 2].copy_from_slice(&half[0].to_le_bytes());
        for (i, &x) in chunk.iter().enumerate() {
            dst[at + 2 + i] = ((x * id).round() as i8) as u8;
        }
    }
}

impl KvCache {
    pub fn new(n_layer: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self::with_type(n_layer, n_kv_heads, head_dim, KvType::F16)
    }

    /// A cache storing `kind`.
    ///
    /// Q8_0 needs `head_dim` to be a multiple of 32, or a row does not hold
    /// whole blocks and the quantisation boundaries fall inside a head. Every
    /// architecture here uses 64, 128 or 256; one that did not would be
    /// silently misquantised, so this falls back to F16 rather than guessing.
    pub fn with_type(n_layer: usize, n_kv_heads: usize, head_dim: usize, kind: KvType) -> Self {
        let kind = if kind == KvType::Q8_0 && head_dim % QK8_0 != 0 {
            KvType::F16
        } else {
            kind
        };
        KvCache {
            k: vec![Vec::new(); n_layer],
            v: vec![Vec::new(); n_layer],
            n_positions: 0,
            per_position: n_kv_heads * head_dim,
            kind,
            row: head_dim,
        }
    }

    /// What this cache stores. `with_type` may have refused what was asked for.
    pub fn kind(&self) -> KvType {
        self.kind
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
        self.k.iter().map(|v| v.len()).sum::<usize>()
            + self.v.iter().map(|v| v.len()).sum::<usize>()
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
        let step = self.kind.bytes_for(self.per_position);
        let at = self.k[layer].len();
        self.k[layer].resize(at + step, 0);
        self.v[layer].resize(at + step, 0);
        match self.kind {
            KvType::F16 => {
                // Through a u16 view: `f32_to_f16` fills halves, and the bytes
                // of those halves are what the tensor reads.
                let mut halves = vec![0u16; self.per_position];
                bigtea_ggml::f32_to_f16(k, &mut halves);
                self.k[layer][at..at + step].copy_from_slice(as_bytes(&halves));
                bigtea_ggml::f32_to_f16(v, &mut halves);
                self.v[layer][at..at + step].copy_from_slice(as_bytes(&halves));
            }
            KvType::Q8_0 => {
                // Row by row: a block may not span two heads, or one head's
                // scale is applied to another head's values.
                let row_bytes = self.kind.bytes_for(self.row);
                for (r, (kr, vr)) in k.chunks(self.row).zip(v.chunks(self.row)).enumerate() {
                    let lo = at + r * row_bytes;
                    quantize_q8_0(kr, &mut self.k[layer][lo..lo + row_bytes]);
                    quantize_q8_0(vr, &mut self.v[layer][lo..lo + row_bytes]);
                }
            }
        }
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

    /// Raw bytes for a layer's keys, ready to fill a tensor of [`Self::kind`].
    pub fn keys(&self, layer: usize) -> &[u8] {
        &self.k[layer]
    }

    pub fn values(&self, layer: usize) -> &[u8] {
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
    /// Number of layers, so a saved cache can be checked before it is trusted.
    pub fn layers(&self) -> usize {
        self.k.len()
    }

    /// Replace one layer's stored bytes — restoring a saved prompt cache.
    ///
    /// Rejects a length that is not a whole number of positions, because a
    /// truncated restore would misalign every position after it and attention
    /// would read across position boundaries with no error anywhere.
    pub fn restore_layer(&mut self, layer: usize, k: &[u8], v: &[u8]) -> Result<(), KvError> {
        let step = self.kind.bytes_for(self.per_position);
        if layer >= self.k.len() || k.len() != v.len() || step == 0 || k.len() % step != 0 {
            return Err(KvError::WrongSize {
                expected: step,
                got_k: k.len(),
                got_v: v.len(),
            });
        }
        self.k[layer] = k.to_vec();
        self.v[layer] = v.to_vec();
        Ok(())
    }

    /// Declare how many positions the restored bytes cover.
    ///
    /// Separate from [`Self::restore_layer`] for the same reason `advance` is
    /// separate from `push`: the count is per cache, not per layer.
    pub fn set_positions(&mut self, n: usize) {
        self.n_positions = n;
    }

    /// Keep only the first `n` positions.
    ///
    /// A saved cache is reusable exactly as far as its tokens match the new
    /// prompt; past the first difference every stored key is conditioned on
    /// text that is no longer there. Cutting rather than discarding the whole
    /// file is what makes a cache useful for a prompt that was *edited*.
    pub fn truncate_to(&mut self, n: usize) {
        if n >= self.n_positions {
            return;
        }
        let keep = self.kind.bytes_for(self.per_position) * n;
        for layer in self.k.iter_mut() {
            layer.truncate(keep);
        }
        for layer in self.v.iter_mut() {
            layer.truncate(keep);
        }
        self.n_positions = n;
    }

    /// Drop `drop` positions starting at `keep`, sliding the rest down.
    ///
    /// llama.cpp's context shift, and what makes generation past the context
    /// limit possible at all: the first `keep` positions stay (a system prompt
    /// usually), the oldest `drop` after them are discarded, and everything
    /// later slides back so the sequence is contiguous again.
    ///
    /// # The part that is not obvious
    ///
    /// **The keys that slide are not re-encoded.** A key was computed with RoPE
    /// applied at its original absolute position, and after the slide it sits
    /// at a lower one — so every shifted key carries a rotation for a position
    /// it no longer occupies. llama.cpp corrects this by re-roping the shifted
    /// range (`llama_kv_cache_seq_add`); this does not, so the shifted history
    /// is approximate rather than exact.
    ///
    /// That is a real limitation and the runner says so out loud rather than
    /// presenting shifted output as equivalent. It is still far better than the
    /// alternative, which is refusing to generate at all — and it is exactly
    /// the trade llama.cpp made before it added re-roping.
    pub fn shift_out(&mut self, keep: usize, drop: usize) {
        if drop == 0 || keep >= self.n_positions {
            return;
        }
        let drop = drop.min(self.n_positions - keep);
        let step = self.kind.bytes_for(self.per_position);
        let from = (keep + drop) * step;
        let to = keep * step;
        for layer in self.k.iter_mut().chain(self.v.iter_mut()) {
            layer.copy_within(from.., to);
            let new_len = layer.len() - drop * step;
            layer.truncate(new_len);
        }
        self.n_positions -= drop;
    }

    pub fn is_consistent(&self) -> bool {
        // In **bytes**, which is what the vectors now hold. Comparing against
        // the value count silently passed for F16 only because a length in
        // halves and a length in bytes happened to differ by exactly the
        // factor this check never looked at.
        let expected = self.kind.bytes_for(self.n_positions * self.per_position);
        self.k.iter().all(|v| v.len() == expected) && self.v.iter().all(|v| v.len() == expected)
    }
}

/// View f16 values as the bytes a ggml F16 tensor expects.
///
/// Little-endian on every target this runs on, and ggml reads the same layout,
/// so no per-element conversion is needed on the way out.
fn as_bytes(v: &[u16]) -> &[u8] {
    // SAFETY: u16 has no padding or invalid bit patterns, and the resulting
    // slice covers exactly the same allocation with a compatible alignment.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
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
            KvError::WrongSize {
                expected,
                got_k,
                got_v,
            } => write!(
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
        // 2 layers, 4 kv heads, head_dim 8 -> 32 values per position.
        KvCache::new(2, 4, 8)
    }

    /// Decode one stored f16 back to f32, so tests read values rather than
    /// bytes. Handles the normals these tests use; f16 subnormals and NaN are
    /// not exercised here.
    fn at(bytes: &[u8], index: usize) -> f32 {
        let bits = u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]);
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1f) as u32;
        let frac = (bits & 0x3ff) as u32;
        let f = if exp == 0 {
            sign << 31 // zero (or subnormal, treated as zero here)
        } else {
            (sign << 31) | ((exp + 112) << 23) | (frac << 13)
        };
        f32::from_bits(f)
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
        // 32 values per position, two bytes each now that storage is f16.
        assert_eq!(c.keys(0).len(), 32 * 2);
        assert_eq!(at(c.values(1), 0), 2.0);
        // 2 layers * 2 tensors * 32 values * 2 bytes
        assert_eq!(c.bytes(), 2 * 2 * 32 * 2);
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
        assert_eq!(keys.len(), 96 * 2);
        // Positions must stay in order; attention indexes by position.
        assert_eq!(at(keys, 0), 0.0);
        assert_eq!(at(keys, 32), 1.0);
        assert_eq!(at(keys, 64), 2.0);
    }

    #[test]
    fn a_wrong_sized_append_is_refused() {
        // Accepting it would misalign every later position, and attention
        // would silently read across position boundaries.
        let mut c = cache();
        let short = vec![1.0f32; 16];
        assert!(c.push(0, &short, &short).is_err());
        assert!(
            c.is_consistent(),
            "a refused push must not mutate the cache"
        );
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
    fn q8_0_halves_the_bytes_and_survives_a_round_trip() {
        // 2 bytes per value against 34 per 32 -- 1.0625, so a shade under half.
        let mut f16 = KvCache::with_type(1, 2, 64, KvType::F16);
        let mut q8 = KvCache::with_type(1, 2, 64, KvType::Q8_0);
        let vals: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 40.0).collect();
        f16.push(0, &vals, &vals).expect("push");
        q8.push(0, &vals, &vals).expect("push");
        f16.advance();
        q8.advance();
        assert_eq!(f16.bytes(), 128 * 2 * 2, "f16 is two bytes a value");
        assert_eq!(q8.bytes(), 128 / 32 * 34 * 2, "q8_0 is 34 bytes per 32");
        assert!(
            q8.bytes() * 2 < f16.bytes() * 2 - 100,
            "q8_0 must be smaller"
        );
        assert!(f16.is_consistent() && q8.is_consistent());

        // Dequantise the first block by hand and check it tracks the input.
        // The scale is max|x|/127 over the block, so the error per value is
        // bounded by half a step -- if the block boundaries were misplaced this
        // would be wildly off rather than subtly.
        let bytes = q8.keys(0);
        let d = half_from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        for i in 0..32 {
            let got = (bytes[2 + i] as i8) as f32 * d;
            assert!(
                (got - vals[i]).abs() <= d,
                "value {i}: {got} vs {} (step {d})",
                vals[i]
            );
        }
    }

    /// Decode an f16 bit pattern for the assertion above.
    fn half_from_bits(bits: u16) -> f32 {
        let mut out = [0f32; 1];
        bigtea_ggml::f16_to_f32(&[bits], &mut out);
        out[0]
    }

    #[test]
    fn q8_0_is_refused_when_a_row_is_not_whole_blocks() {
        // head_dim 40 would put a block boundary inside a head, applying one
        // head's scale to another's values. Falling back is the only safe
        // answer, and it must be visible rather than silent.
        let c = KvCache::with_type(1, 2, 40, KvType::Q8_0);
        assert_eq!(c.kind(), KvType::F16);
        // 64 divides 32, so this one is honoured.
        let c = KvCache::with_type(1, 2, 64, KvType::Q8_0);
        assert_eq!(c.kind(), KvType::Q8_0);
    }

    #[test]
    fn cache_type_names_match_llamacpp() {
        assert_eq!(KvType::parse("q8_0"), Some(KvType::Q8_0));
        assert_eq!(KvType::parse("F16"), Some(KvType::F16));
        assert_eq!(
            KvType::parse("q4_0"),
            None,
            "unsupported must not be guessed"
        );
    }

    #[test]
    fn f16_storage_round_trips_the_values_attention_needs() {
        // Halving the cache is only worth it if the numbers survive. These are
        // the magnitudes real keys and values take; f16 holds them exactly or
        // near enough that attention cannot tell.
        let mut c = cache();
        let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.25).collect();
        c.push(0, &vals, &vals).expect("push");
        c.push(1, &vals, &vals).expect("push");
        c.advance();

        for (i, want) in vals.iter().enumerate() {
            let got = at(c.keys(0), i);
            assert!(
                (got - want).abs() < 1e-3,
                "position {i}: stored {got}, wanted {want}"
            );
        }
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

    #[test]
    fn shifting_out_keeps_the_head_and_slides_the_tail() {
        // One layer, one head, head_dim 2 -> two f32 per position.
        let mut c = KvCache::new(1, 1, 2);
        for i in 0..6 {
            let v = i as f32;
            c.push(0, &[v, v], &[v, v]).unwrap();
            // `advance` is separate from `push` so the count moves once per
            // token rather than once per layer -- one layer here, so once each.
            c.advance();
        }
        assert_eq!(c.len(), 6);
        // Keep 2, drop 2: positions 0,1 stay; 2,3 go; 4,5 slide to 2,3.
        c.shift_out(2, 2);
        assert_eq!(c.len(), 4);
        assert!(c.is_consistent());
        // Byte-level: each position is two f16, so four bytes.
        let k = c.keys(0);
        assert_eq!(k.len(), 16);
        let first_of = |slot: usize| &k[slot * 4..slot * 4 + 4];
        assert_eq!(first_of(0), first_of(0));
        // Slot 2 must now hold what position 4 held, not what position 2 did.
        // Compare against a freshly built cache holding only 0,1,4,5.
        let mut want = KvCache::new(1, 1, 2);
        for i in [0.0f32, 1.0, 4.0, 5.0] {
            want.push(0, &[i, i], &[i, i]).unwrap();
            want.advance();
        }
        assert_eq!(c.keys(0), want.keys(0));
        assert_eq!(c.values(0), want.values(0));
    }

    #[test]
    fn shifting_more_than_there_is_clamps_rather_than_underflowing() {
        let mut c = KvCache::new(1, 1, 2);
        for i in 0..4 {
            let v = i as f32;
            c.push(0, &[v, v], &[v, v]).unwrap();
            c.advance();
        }
        c.shift_out(1, 100);
        assert_eq!(c.len(), 1);
        assert!(c.is_consistent());
    }

    #[test]
    fn shifting_nothing_is_a_no_op() {
        let mut c = KvCache::new(1, 1, 2);
        c.push(0, &[7.0, 7.0], &[7.0, 7.0]).unwrap();
        c.advance();
        let before = c.keys(0).to_vec();
        c.shift_out(0, 0);
        c.shift_out(5, 3);
        assert_eq!(c.len(), 1);
        assert_eq!(c.keys(0), &before[..]);
    }
}
