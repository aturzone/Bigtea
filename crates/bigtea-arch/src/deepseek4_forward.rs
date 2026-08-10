//! The DeepSeek-V4-Flash forward pass, as library code rather than as a test.
//!
//! Every step here was built checkpoint-by-checkpoint against `llama.cpp`'s own
//! element-sums — see `tests/deepseek4_forward.rs`, which now checks *this*
//! code rather than a copy of it. That distinction matters: the port spent its
//! first weeks with the implementation living inside the test, which meant the
//! library shipped nothing and the verification proved only that the test
//! agreed with itself.
//!
//! # What is verified, and at which prompt length
//!
//! **Prompt length decides which code paths run**, because the compressed
//! attention builders are guarded on their compressed caches being non-empty.
//! The same layer runs different attention at different lengths:
//!
//! | tokens | layers 0-1 | even layers | odd layers |
//! |---|---|---|---|
//! | ≤3 | Raw | Raw (fallback) | Raw (fallback) |
//! | 5 | Raw | Compressed Sparse | Raw (fallback) |
//! | ≥128 | Raw | Compressed Sparse | Heavily Compressed |
//!
//! All three are checked against llama.cpp: the whole 43-block stack at 2
//! tokens, and layers 0-3 at 165 where both compressed kinds fire.
//!
//! # The one deliberate omission
//!
//! **The lightning indexer is not run**, and below ~2048 tokens that is exact
//! rather than approximate: `n_top_k = min(n_lid, indexer_top_k)` selects
//! *every* compressed slot, so the indexer's mask is precisely the visibility
//! mask and cannot change any output. Above that length this becomes an
//! approximation and [`Deepseek4Forward::indexer_is_exact`] returns false.

use bigtea_ggml::{Context, RopeParams, Tensor, WeightSet};
use bigtea_io::SkewedBuf;
use bigtea_model::{Model, ResidentSet};

use crate::expert_cache::{slice_key, ExpertCache, SliceKey};
use crate::{AttentionKind, Deepseek4Config, Result};

/// `LLAMA_ROPE_TYPE_NORM`: rotated pairs are adjacent, not offset by `n_rot/2`.
const ROPE_MODE_NORM: i32 = 0;

/// F16 `-inf`, written as bits. Mask values are only ever 0 or -inf, so writing
/// the pattern beats converting.
const F16_NEG_INF: [u8; 2] = [0x00, 0xFC];

/// When `compute` is actually needed.
///
/// `Context::compute` evaluates a tensor's **entire ancestor graph**, so calling
/// it on every intermediate does not merely dispatch more work — it *re-does*
/// the work, once per call, and pays a graph build and a threadpool cycle each
/// time. At a single token the ops are vectors and that overhead is most of the
/// cost: `layer_tail` plus `moe_routing` measured 0.06s per block for a handful
/// of normalisations and a top-6 sort.
///
/// So a value is computed only where the **CPU** must read it: before a
/// `to_vec_*` or a `set_*` that consumes it. Everything else stays a graph node
/// and is evaluated once, as part of whichever sync point subsumes it.
///
/// The genuine sync points in a block are: `kv_full` (attention builds an F16
/// cache from its values), the compressor's `kv`/`score`/output, the router's
/// `topk` (routing decides which expert slices to read from disk, so it cannot
/// be deferred), and the block's own output.
/// How often each expert of each layer is actually selected.
///
/// The whole streaming budget rests on an assumption nobody has checked: that a
/// token's 6-of-256 choice is spread evenly, so 137 GiB of experts are all
/// equally cold and none is worth holding in RAM. **If routing is skewed
/// instead — if a small hot set absorbs most selections — then that set is
/// cacheable and the bytes-per-token figure that bounds everything is wrong.**
///
/// Set `BIGTEA_ROUTING=1` and the runner prints the distribution at exit.
///
/// Indexed `[pass][layer][expert]`. The pass dimension exists because
/// generation here is **stateless** — every generated token re-runs prefill over
/// the whole sequence — so a single accumulated histogram counts the prompt once
/// per token. That silently inflated v0.0.2's chi-square by the pass count.
///
/// Keeping passes apart also turns the artefact into the measurement. The model
/// is causal, so token *i*'s routing is identical in every pass that contains it;
/// the difference between pass *k* and pass *k-1* is therefore exactly the
/// routing of the one token generated in between. That is the only way to ask
/// whether a cache warmed on the prompt predicts what generation goes on to need.
static ROUTING: std::sync::OnceLock<std::sync::Mutex<Vec<Vec<Vec<u32>>>>> =
    std::sync::OnceLock::new();

fn routing_log() -> &'static std::sync::Mutex<Vec<Vec<Vec<u32>>>> {
    ROUTING.get_or_init(|| std::sync::Mutex::new(vec![Vec::new()]))
}

fn record_routing(il: u32, n_expert: usize, ids: &[i32]) {
    let mut log = routing_log().lock().expect("routing histogram");
    record_into(&mut log, il, n_expert, ids);
}

/// The counting itself, split out from the global so it can be tested.
fn record_into(log: &mut [Vec<Vec<u32>>], il: u32, n_expert: usize, ids: &[i32]) {
    let pass = log.last_mut().expect("one pass always exists");
    while pass.len() <= il as usize {
        pass.push(vec![0u32; n_expert]);
    }
    for id in ids {
        if let Some(slot) = pass[il as usize].get_mut(*id as usize) {
            *slot += 1;
        }
    }
}

/// Start counting a new forward pass.
///
/// Call this before each re-prefill in the generation loop. Without it every
/// pass lands in one bin and the prompt is counted again per generated token.
pub fn routing_next_pass() {
    if std::env::var("BIGTEA_ROUTING").is_err() {
        return;
    }
    routing_log()
        .lock()
        .expect("routing histogram")
        .push(Vec::new());
}

/// What fraction of selections the hottest experts absorb, and what that would
/// cost to keep resident.
///
/// Prints nothing unless `BIGTEA_ROUTING` is set.
///
/// **Reported twice.** The first `hash_layers` blocks select by *token id* out
/// of `ffn_gate_tid2eid`, not by a learned gate, so their skew is the token
/// distribution wearing a router's clothes — a Zipfian prompt would look like a
/// skewed router. Only the `>= hash_layers` table says anything about gating,
/// and it is the one a cache should be sized from.
///
/// Set `BIGTEA_ROUTING_DUMP=<path>` to also write raw `layer,expert,count` rows,
/// which is what makes two runs comparable: the question R0 asks is not how
/// skewed one prompt is but whether two prompts are skewed toward the *same*
/// experts, and that cannot be read off a summary table.
pub fn routing_report(expert_gib_total: f64, hash_layers: u32) {
    let Some(log) = ROUTING.get() else { return };
    let log = log.lock().expect("routing histogram");
    if log.iter().all(|p| p.is_empty()) {
        return;
    }
    if let Ok(path) = std::env::var("BIGTEA_ROUTING_DUMP") {
        match dump_routing(&log, &path) {
            Ok(()) => eprintln!(
                "\nrouting histogram written to {path} ({} passes)",
                log.len()
            ),
            Err(e) => eprintln!("\nrouting histogram dump to {path} failed: {e}"),
        }
    }
    // The printed tables pool every pass, which is what the pre-existing report
    // did. Pooled counts are fine for *shares* — repeating a pass scales every
    // bin alike — and wrong for chi-square, which is why the dump keeps passes
    // apart and the report names its pass count.
    let hist = pool_passes(&log);
    if log.len() > 1 {
        eprintln!(
            "\nNOTE: {} forward passes pooled below. Generation re-runs prefill per\n\
             token, so the prompt is counted once per pass: shares are unaffected,\n\
             chi-square is inflated by roughly the pass count. Use -n 1 to measure.",
            log.len()
        );
    }
    let hash_layers = (hash_layers as usize).min(hist.len());
    routing_table(&hist, expert_gib_total, 0, "all layers");
    if hash_layers > 0 && hash_layers < hist.len() {
        routing_table(
            &hist,
            expert_gib_total,
            hash_layers,
            "learned-gating layers only",
        );
    }
}

/// One top-N table over `hist[from..]`.
fn routing_table(hist: &[Vec<u32>], expert_gib_total: f64, from: usize, label: &str) {
    let layers = &hist[from..];
    if layers.is_empty() {
        return;
    }
    let n_expert = layers[0].len();
    eprintln!(
        "\nrouting distribution — {label} ({} of {} layers, {n_expert} experts each)",
        layers.len(),
        hist.len()
    );
    eprintln!("  top-N experts per layer   share of selections   resident cost");

    for top in [1usize, 4, 8, 16, 32, 64, 128] {
        if top > n_expert {
            break;
        }
        let mut covered = 0u64;
        let mut total = 0u64;
        for layer in layers.iter() {
            let mut counts = layer.clone();
            counts.sort_unstable_by(|a, b| b.cmp(a));
            covered += counts.iter().take(top).map(|c| *c as u64).sum::<u64>();
            total += counts.iter().map(|c| *c as u64).sum::<u64>();
        }
        if total == 0 {
            return;
        }
        let share = covered as f64 / total as f64;
        let gib = expert_gib_total * top as f64 / n_expert as f64;
        eprintln!(
            "  {top:>3}   ({:>5.1}% of the model)   {:>6.1}%              {:>6.2} GiB",
            top as f64 / n_expert as f64 * 100.0,
            share * 100.0,
            gib
        );
    }

    // A perfectly uniform router would give exactly top/n_expert. Anything above
    // that is skew, and skew is the only thing that makes caching worth having.
    //
    // Two statistics, because v0.0.2 published the first one and it is the weaker
    // of the two. **Pooled** sums every layer's count for expert index i into one
    // bin, which asks whether an *index* is globally popular — but expert 7 of
    // layer 3 and expert 7 of layer 30 are unrelated weights, so a pooled figure
    // can be inflated by one layer or cancelled by two disagreeing ones.
    // **Per-layer** sums each layer's own chi-square, which is the question a
    // per-layer cache actually asks. Both are printed so the published 7805 stays
    // comparable rather than silently replaced.
    let mut pooled: Vec<u64> = vec![0; n_expert];
    for layer in layers.iter() {
        for (e, c) in layer.iter().enumerate() {
            pooled[e] += *c as u64;
        }
    }
    let total: u64 = pooled.iter().sum();
    let uniform = total as f64 / n_expert as f64;
    let chi_pooled: f64 = pooled
        .iter()
        .map(|c| (*c as f64 - uniform).powi(2) / uniform.max(1.0))
        .sum();

    let mut chi_layer = 0.0;
    let mut dof = 0usize;
    for layer in layers.iter() {
        let total: u64 = layer.iter().map(|c| *c as u64).sum();
        if total == 0 {
            continue;
        }
        let uniform = total as f64 / n_expert as f64;
        chi_layer += layer
            .iter()
            .map(|c| (*c as f64 - uniform).powi(2) / uniform.max(1e-9))
            .sum::<f64>();
        dof += n_expert - 1;
    }
    eprintln!(
        "  uniform routing would give top-16 = {:.1}%",
        16.0 / n_expert as f64 * 100.0
    );
    eprintln!(
        "  chi-square vs uniform: pooled {chi_pooled:.0} (d.o.f. {}), per-layer {chi_layer:.0} (d.o.f. {dof})",
        n_expert - 1
    );
}

/// Every pass summed into one `[layer][expert]` histogram.
fn pool_passes(log: &[Vec<Vec<u32>>]) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = Vec::new();
    for pass in log {
        for (il, layer) in pass.iter().enumerate() {
            if out.len() <= il {
                out.push(vec![0u32; layer.len()]);
            }
            for (e, c) in layer.iter().enumerate() {
                out[il][e] += c;
            }
        }
    }
    out
}

/// Raw `pass,layer,expert,count` rows, so two runs can be compared offline.
///
/// Zero counts are written too. The analysis needs a dense matrix, and 43 x 256
/// rows per pass is a rounding error next to the model.
fn dump_routing(log: &[Vec<Vec<u32>>], path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(out, "pass,layer,expert,count")?;
    for (p, pass) in log.iter().enumerate() {
        for (il, layer) in pass.iter().enumerate() {
            for (e, c) in layer.iter().enumerate() {
                writeln!(out, "{p},{il},{e},{c}")?;
            }
        }
    }
    out.flush()
}

/// The experts the **last token of a batch** selected, per layer.
///
/// The routing histogram cannot answer R3's question. It aggregates over every
/// token in the pass, so "did a cached step route the same way as a full
/// prefill" gets lost in the tokens they share. This records only the final
/// token's six-of-256, which is the one token both paths end on and therefore
/// the only fair comparison.
///
/// Enabled by `BIGTEA_ROUTING_LAST` so it costs nothing in a normal run.
static LAST_ROUTING: std::sync::OnceLock<std::sync::Mutex<Vec<Vec<i32>>>> =
    std::sync::OnceLock::new();

fn last_routing() -> &'static std::sync::Mutex<Vec<Vec<i32>>> {
    LAST_ROUTING.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

fn record_last_token(il: u32, n_used: usize, ids: &[i32]) {
    // `ids` is `n_used` per token, token-major, so the final token's selections
    // are the last `n_used` entries.
    let Some(tail) = ids.len().checked_sub(n_used) else {
        return;
    };
    let mut log = last_routing().lock().expect("last-token routing");
    while log.len() <= il as usize {
        log.push(Vec::new());
    }
    log[il as usize] = ids[tail..].to_vec();
}

/// Per-layer expert ids chosen by the last token of the most recent pass.
pub fn routing_last_token() -> Vec<Vec<i32>> {
    last_routing().lock().expect("last-token routing").clone()
}

/// Forget the recorded selections, so two passes can be compared cleanly.
pub fn routing_last_token_reset() {
    last_routing().lock().expect("last-token routing").clear();
}

/// How the renormalised router weight is spread across the six chosen experts.
///
/// # The question it answers
///
/// Every selected expert costs the same bytes to stream — 4.2 MiB — but they do
/// not contribute equally: the output is `Σ w_i · expert_i(x)` with `w`
/// renormalised over the six. If the top two carry most of the mass, the tail
/// is being paid for in full and returned at a discount, and **dropping it is a
/// byte reduction available at every batch size, cache state and RAM budget** —
/// the only lever measured so far with that property.
///
/// Accumulated as `[layer][rank]` sums of the weights sorted descending, plus a
/// count, so the report is a mean profile rather than one token's accident.
/// Sorting here rather than trusting selection order is deliberate: `top_k` does
/// not return indices in score order, and a profile built on that assumption
/// would look flat for a reason that has nothing to do with the model.
///
/// Enabled by `BIGTEA_ROUTING_WEIGHTS`, because reading the weights needs a
/// `compute()` that would otherwise re-evaluate the ancestor graph for nothing.
type WeightProfile = (Vec<Vec<f64>>, u64);
static ROUTING_WEIGHTS: std::sync::OnceLock<std::sync::Mutex<WeightProfile>> =
    std::sync::OnceLock::new();

fn routing_weights() -> &'static std::sync::Mutex<WeightProfile> {
    ROUTING_WEIGHTS.get_or_init(|| std::sync::Mutex::new((Vec::new(), 0)))
}

fn record_routing_weights(il: u32, n_used: usize, weights: &[f32]) {
    let mut log = routing_weights().lock().expect("routing weights");
    while log.0.len() <= il as usize {
        log.0.push(vec![0.0; n_used]);
    }
    let mut tokens = 0u64;
    for row in weights.chunks_exact(n_used) {
        let mut sorted: Vec<f32> = row.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).expect("finite router weight"));
        for (slot, v) in log.0[il as usize].iter_mut().zip(&sorted) {
            *slot += *v as f64;
        }
        tokens += 1;
    }
    // Counted once, on the first layer, so the divisor is tokens and not
    // tokens x layers.
    if il == 0 {
        log.1 += tokens;
    }
}

/// Print the mean router-weight profile, and what dropping the tail would save.
///
/// Prints nothing unless `BIGTEA_ROUTING_WEIGHTS` is set.
pub fn routing_weight_report(expert_gib_per_token: f64) {
    if std::env::var("BIGTEA_ROUTING_WEIGHTS").is_err() {
        return;
    }
    let log = routing_weights().lock().expect("routing weights");
    let (profile, tokens) = (&log.0, log.1);
    if profile.is_empty() || tokens == 0 {
        return;
    }
    let n_used = profile[0].len();
    // Mean over layers and tokens: one profile for the model, since the
    // decision — how many experts to read — is made the same way in every layer.
    let mut mean = vec![0.0f64; n_used];
    for layer in profile.iter() {
        for (m, v) in mean.iter_mut().zip(layer) {
            *m += v / (tokens as f64 * profile.len() as f64);
        }
    }
    let total: f64 = mean.iter().sum();

    println!();
    println!(
        "router weight profile  ({tokens} tokens, {} layers)",
        profile.len()
    );
    println!(
        "{:>5}  {:>8}  {:>10}  {:>10}  {:>12}",
        "KEEP", "WEIGHT", "CUMULATIVE", "GiB/token", "SPEEDUP"
    );
    let mut acc = 0.0;
    for (i, w) in mean.iter().enumerate() {
        acc += w;
        let keep = i + 1;
        let gib = expert_gib_per_token * keep as f64 / n_used as f64;
        println!(
            "{keep:>5}  {:>7.1}%  {:>9.1}%  {gib:>10.2}  {:>11.2}x",
            w / total * 100.0,
            acc / total * 100.0,
            n_used as f64 / keep as f64
        );
    }
    println!();
    println!("CUMULATIVE is the share of router weight kept, NOT the share of");
    println!("output preserved -- a dropped expert's contribution is its weight");
    println!("times its output, and the outputs are not equal. This bounds the");
    println!("idea; it does not decide it. Perplexity does.");
}

/// Attention state that must survive from one forward pass to the next.
///
/// # Why this exists
///
/// Without it, generating token *n* re-runs the whole prompt: every published
/// V4-Flash generation figure so far is the cost of re-prefilling the sequence,
/// not the cost of a token. It is also what makes the expert cache pay — a pass
/// over 166 tokens reads **122.8 distinct experts per layer (~66 GiB)**, and a
/// single-token step reads **6 (3.21 GiB)**. Nothing of that size is cacheable
/// until a step stops re-reading the sequence.
///
/// # Three structures, not two
///
/// The compressor's input ring is the one that is easy to miss, and missing it
/// does not fail — it summarises the wrong span, fluently. On a *prefill* the
/// previous window's rows are inside the batch being processed, so
/// [`compressor`] can front-pad with `state_rows` zeros and never read a ring.
/// In incremental decode those rows are in the past, and the zeros would be a
/// lie.
///
/// Roughly 24 MB across 43 layers. Memory is not the constraint here;
/// correctness is.
pub struct Deepseek4Cache {
    /// Raw KV latents, F16, `kv_lora_rank * N_KV` per layer.
    ///
    /// **Slot index is the absolute position**, which is what lets the mask stay
    /// simple arithmetic — and what a ring with wraparound would break, so the
    /// mask and any future ring must be rewritten together.
    raw: Vec<Vec<u16>>,
    /// Compressed summaries, F16, `kv_lora_rank * N_KV` per layer. Slot index is
    /// the **block** index, not the position.
    comp: Vec<Vec<u16>>,
    /// The compressor's input ring, per layer: the last `state_rows` rows of the
    /// `kv` and `score` projections, interleaved as `(kv, score)`.
    ///
    /// **This is the piece that is easy to miss.** On a prefill the previous
    /// window's rows are inside the batch being processed, so [`compressor`] can
    /// front-pad `state_rows` zeros and never read a ring — which is why the
    /// zeros were correct until now. In incremental decode those rows are in the
    /// past, and the zeros would summarise the wrong span **without failing**.
    ///
    /// Sized lazily: `state_rows * wide` depends on whether the layer's
    /// compressor overlaps, which is a property of the layer.
    ring: Vec<(Vec<f32>, Vec<f32>)>,
    /// Whether a layer's compressed half holds anything yet. The compressed
    /// builders are guarded on this, so the same layer takes a different path
    /// early in a sequence than later.
    comp_len: Vec<i64>,
    /// How many tokens this cache already describes: the absolute position the
    /// next step occupies.
    n_past: usize,
}

impl Deepseek4Cache {
    pub fn new(n_layer: u32, kv_lora_rank: u32) -> Self {
        let per_layer = kv_lora_rank as usize * N_KV as usize;
        Deepseek4Cache {
            raw: vec![vec![0u16; per_layer]; n_layer as usize],
            comp: vec![vec![0u16; per_layer]; n_layer as usize],
            ring: vec![(Vec::new(), Vec::new()); n_layer as usize],
            comp_len: vec![0; n_layer as usize],
            n_past: 0,
        }
    }

    /// Absolute position the next token will occupy.
    pub fn n_past(&self) -> usize {
        self.n_past
    }

    /// Forget everything, so the same cache can start a new sequence.
    pub fn clear(&mut self) {
        for layer in self.raw.iter_mut().chain(self.comp.iter_mut()) {
            layer.fill(0);
        }
        for (kv, sc) in self.ring.iter_mut() {
            kv.clear();
            sc.clear();
        }
        self.comp_len.fill(0);
        self.n_past = 0;
    }
}

/// One expert tensor's selected slices, packed, with the shape to bind them as.
///
/// The dims are not the tensor's: the last is the number of slices actually
/// read, so `[ne0, ne1, 6]` rather than `[ne0, ne1, 256]`.
type ExpertStack = (SkewedBuf, Vec<u64>);

/// Concurrent readers for a layer's expert slices.
///
/// Was four, because "no further gain at eight" — which was true, and was an
/// artefact of all four sharing one **synchronous** file handle, where the OS
/// serialises reads and the drive never leaves queue depth 1. With a handle per
/// reader ([`bigtea_model`]'s pool) the curve keeps climbing to eight:
///
/// ```text
/// threads      one shared handle      one handle each
///       4           2.01 GiB/s             2.65 GiB/s
///       8           2.05                   2.69
/// ```
///
/// Eight is where the per-handle curve flattens, and it must not exceed the
/// pool size or two readers would collide on one handle again.
const READERS: usize = 8;

/// The padded key window each half of the cache occupies.
const N_KV: i64 = 256;

/// Threads for every `ggml` graph evaluation in this file.
///
/// A constant here was a guess. `compute(&t, 0)` runs on **one** thread — the
/// count is floored at 1, not defaulted to all cores — so the number has to be
/// passed explicitly, and once it is passed explicitly it deserves to be
/// measured rather than assumed. `BIGTEA_THREADS` exists to measure it; the
/// default is chosen in `bigtea-run`, not here.
fn threads() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("BIGTEA_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&t: &usize| t > 0)
            // All cores, not a constant. 12 was this machine's count when the
            // constant was written and is wrong everywhere else.
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(4)
            })
    })
}

/// One block's forward pass, and the state it threads.
pub struct Deepseek4Forward<'m> {
    model: &'m Model,
    config: Deepseek4Config,
    /// Always-read weights held in RAM. `None` re-reads them per block, which
    /// is correct but costs 23% of a prefill and would cost it again on every
    /// generated token.
    resident: Option<&'m ResidentSet>,
    /// Routed expert slices held across blocks and tokens. `None` streams every
    /// slice from disk on every use.
    ///
    /// A `Mutex` rather than `&mut` threading: the read path is reached through
    /// several layers of `&self`, and the lock is taken once per block on a
    /// single thread, so it costs nothing measurable next to a 53 ms read.
    cache: Option<std::sync::Mutex<ExpertCache>>,
}

impl<'m> Deepseek4Forward<'m> {
    pub fn new(model: &'m Model, config: Deepseek4Config) -> Self {
        Deepseek4Forward {
            model,
            config,
            resident: None,
            cache: None,
        }
    }

    /// Serve always-read weights from `resident` instead of from disk.
    pub fn with_resident(mut self, resident: &'m ResidentSet) -> Self {
        self.resident = Some(resident);
        self
    }

    /// Hold routed expert slices in `budget` bytes of memory this process owns.
    ///
    /// Nothing is pre-loaded. R0 measured that a hot set chosen in advance
    /// covers only 37.5% of an unseen subject's routing, against 25% for caching
    /// at random — so the cache warms on the prompt it is given, which R0.1
    /// measured covers 86.3% of what that prompt goes on to generate.
    pub fn with_expert_cache(mut self, budget: usize) -> Self {
        self.cache = Some(std::sync::Mutex::new(ExpertCache::new(budget)));
        self
    }

    /// Hits, misses, evictions and footprint, or `None` without a cache.
    pub fn cache_stats(&self) -> Option<(crate::CacheStats, usize)> {
        self.cache.as_ref().map(|c| {
            let c = c.lock().expect("expert cache");
            (c.stats(), c.bytes())
        })
    }

    pub fn config(&self) -> &Deepseek4Config {
        &self.config
    }

    /// Whether skipping the lightning indexer is exact at this prompt length.
    ///
    /// It is, until the compressed cache holds more entries than the indexer
    /// would keep — `n_top_k = min(n_lid, indexer_top_k)`. Below that the
    /// indexer selects everything and changes nothing.
    pub fn indexer_is_exact(&self, n_tokens: usize) -> bool {
        let blocks = n_tokens as i64 / Deepseek4Config::CSA_RATIO;
        blocks.min(N_KV) <= self.config.indexer_top_k as i64
    }

    /// Tensor names one block needs, plus the globals for block 0.
    pub fn block_tensor_names(&self, il: u32) -> Vec<String> {
        let mut names = Vec::new();
        for suffix in [
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_attn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "hc_ffn_base",
            "attn_norm",
            "attn_q_a",
            "attn_q_a_norm",
            "attn_q_b",
            "attn_kv",
            "attn_kv_a_norm",
            "attn_sinks",
            "attn_output_a",
            "attn_output_b",
            "ffn_norm",
            "ffn_gate_inp",
            "ffn_gate_shexp",
            "ffn_up_shexp",
            "ffn_down_shexp",
        ] {
            names.push(format!("blk.{il}.{suffix}.weight"));
        }
        // Only some blocks carry these; the two routing schemes are mutually
        // exclusive and a compressor is absent on the two Raw layers.
        for suffix in [
            "ffn_gate_tid2eid.weight",
            "exp_probs_b.bias",
            "attn_compressor_kv.weight",
            "attn_compressor_gate.weight",
            "attn_compressor_ape.weight",
            "attn_compressor_norm.weight",
        ] {
            let n = format!("blk.{il}.{suffix}");
            if self.model.location(&n).is_some() {
                names.push(n);
            }
        }
        names
    }

    /// RoPE for `il`, from the shipped per-layer selection.
    fn rope(&self, il: u32) -> (RopeParams, i32) {
        let r = self.config.rope_for_layer(il);
        (r.params, r.n_ctx_orig)
    }
}

/// The four residual streams between blocks, as plain floats.
///
/// Handing the boundary across as a `Vec` is what lets each block own its arena
/// and drop it: freeing weights *inside* one `ggml` context is unsound, because
/// every `compute` rebuilds the graph through its sources and a dropped buffer
/// becomes a dangling pointer that reads freed memory successfully.
pub type Streams = Vec<f32>;

/// Build `hc_init`: the embedding repeated across the hyper-connection streams.
pub fn embed<'c>(
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    tokens: &[i32],
) -> Result<Tensor<'c>> {
    let nt = tokens.len() as i64;
    let hc = config.hc_mult as i64;
    let tok = ctx.new_i32_1d(nt)?;
    tok.set_i32(tokens)?;
    let embd = ctx.get_rows(weights.get("token_embd.weight").expect("bound"), &tok)?;
    let embd_r = ctx.reshape_3d(&embd, config.n_embd as i64, 1, nt)?;
    let shape = ctx.new_f32_3d(config.n_embd as i64, hc, nt)?;
    let hc_init = ctx.repeat(&embd_r, &shape)?;
    ctx.compute(&hc_init, threads())?;
    Ok(hc_init)
}

/// The three gates one `build_hc_pre` call produces, all from one mixes matmul.
struct HcGates<'c> {
    pre: Tensor<'c>,
    post: Tensor<'c>,
    comb: Tensor<'c>,
}

/// Slice the 24 mixes into the three gates.
///
/// Layout is `[0..hc]` pre, `[hc..2hc]` post, then the combination matrix, with
/// `hc_scale` indexed `[pre, post, comb]`. **Every one of those views is the
/// right size whichever slice you take**, so wrong offsets have no shape
/// consequence at all. `pre` ends with `scale_bias(x, 1, eps)` and `post` with
/// `scale(x, 2.0)` — different tails, same shape.
fn hc_gates<'c>(
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    prefix: &str,
    mixes: &Tensor<'c>,
    nt: i64,
) -> Result<HcGates<'c>> {
    let hc = config.hc_mult as i64;
    let f32_size = std::mem::size_of::<f32>();
    // The stride is the *source's* row — `(2 + hc) * hc` floats — not the 4 the
    // view is wide. At one token the stride is never traversed, so any value
    // passes; only a multi-token prompt pins it.
    let stride = ((2 + hc) * hc) as usize * f32_size;
    let scale_w = weights
        .get(&format!("{prefix}_scale.weight"))
        .expect("bound");
    let base_w = weights
        .get(&format!("{prefix}_base.weight"))
        .expect("bound");

    let gate = |mix_off: i64, scale_idx: usize, base_off: i64| -> Result<Tensor<'c>> {
        let view = ctx.view_2d(mixes, hc, nt, stride, mix_off as usize * f32_size)?;
        let s = ctx.view_1d(scale_w, 1, scale_idx * f32_size)?;
        let b = ctx.view_1d(base_w, hc, base_off as usize * f32_size)?;
        let scaled = ctx.mul(&view, &s)?;
        let biased = ctx.add(&scaled, &b)?;
        Ok(ctx.sigmoid(&biased)?)
    };

    let pre_gated = gate(0, 0, 0)?;
    let eps = ctx.new_f32_1d(hc)?;
    eps.set_f32(&vec![1e-6f32; hc as usize])?;
    let pre = ctx.add(&pre_gated, &eps)?;

    let post_gated = gate(hc, 1, hc)?;
    let post = ctx.scale(&post_gated, 2.0)?;

    let comb = ctx.dsv4_hc_comb(
        mixes,
        scale_w,
        base_w,
        1e-6,
        config.hc_sinkhorn_iterations as i32,
    )?;
    Ok(HcGates { pre, post, comb })
}

/// A block's entry: hyper-connection gates and `attn_norm`, from whatever
/// residual streams it was handed.
///
/// Block 0 reaches here from the embedding, every other block from the previous
/// block's output. **That is the only structural difference between the first
/// block and the rest.**
struct Entry<'c> {
    streams: Tensor<'c>,
    attn_norm: Tensor<'c>,
    gates: HcGates<'c>,
}

fn entry<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    streams: Tensor<'c>,
    nt: i64,
) -> Result<Entry<'c>> {
    let config = &fw.config;
    let flat = ctx.reshape_2d(&streams, config.hc_dim() as i64, nt)?;
    let normed = ctx.rms_norm(&flat, config.rms_eps)?;
    let mixes = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.hc_attn_fn.weight"))
            .expect("bound"),
        &normed,
    )?;
    let gates = hc_gates(
        ctx,
        weights,
        config,
        &format!("blk.{il}.hc_attn"),
        &mixes,
        nt,
    )?;

    let collapsed = ctx.dsv4_hc_pre(&streams, &gates.pre)?;
    let normed = ctx.rms_norm(&collapsed, config.rms_eps)?;
    let attn_norm = ctx.mul(
        &normed,
        weights
            .get(&format!("blk.{il}.attn_norm.weight"))
            .expect("bound"),
    )?;
    Ok(Entry {
        streams,
        attn_norm,
        gates,
    })
}

/// Q and KV, both low-rank, both with only their trailing `n_rot` dims rotated.
///
/// `kv` becomes K **and** V — there is no separate V projection, which is why
/// `head_count_kv` is 1. The per-head norm on `q` carries **no weight**, unlike
/// every other norm in this model.
fn q_and_kv<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    attn_norm: &Tensor<'c>,
    nt: i64,
    pos0: i64,
) -> Result<(Tensor<'c>, Tensor<'c>)> {
    let config = &fw.config;
    let head = config.kv_lora_rank as i64;
    let n_head = config.n_head as i64;
    let n_rot = config.n_rot as i64;
    let n_nope = config.n_rot_none() as i64;
    let f32_size = std::mem::size_of::<f32>();
    let hs = head as usize * f32_size;
    let (rope, rope_orig) = fw.rope(il);

    // Absolute positions. RoPE is applied *before* a value enters the cache, so
    // a cached entry must never be rotated again — which is why this is the
    // token's real position and not its index within the batch.
    let pos = ctx.new_i32_1d(nt)?;
    pos.set_i32(&(pos0 as i32..(pos0 + nt) as i32).collect::<Vec<i32>>())?;

    let qr = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.attn_q_a.weight"))
            .expect("bound"),
        attn_norm,
    )?;
    let qr = ctx.rms_norm(&qr, config.rms_eps)?;
    let qr = ctx.mul(
        &qr,
        weights
            .get(&format!("blk.{il}.attn_q_a_norm.weight"))
            .expect("bound"),
    )?;
    let q = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.attn_q_b.weight"))
            .expect("bound"),
        &qr,
    )?;
    let q = ctx.reshape_3d(&q, head, n_head, nt)?;
    let q = ctx.rms_norm(&q, config.rms_eps)?; // unweighted, deliberately

    let q_nope = ctx.view_3d(&q, n_nope, n_head, nt, hs, hs * n_head as usize, 0)?;
    let q_pe_in = ctx.view_3d(
        &q,
        n_rot,
        n_head,
        nt,
        hs,
        hs * n_head as usize,
        n_nope as usize * f32_size,
    )?;
    let q_pe = ctx.rope_ext(
        &q_pe_in,
        &pos,
        None,
        n_rot as i32,
        ROPE_MODE_NORM,
        rope_orig,
        rope,
    )?;
    let q_full = ctx.concat(&q_nope, &q_pe, 0)?;

    let kv = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.attn_kv.weight"))
            .expect("bound"),
        attn_norm,
    )?;
    let kv = ctx.rms_norm(&kv, config.rms_eps)?;
    let kv = ctx.mul(
        &kv,
        weights
            .get(&format!("blk.{il}.attn_kv_a_norm.weight"))
            .expect("bound"),
    )?;
    let kv = ctx.reshape_3d(&kv, head, 1, nt)?;
    let kv_nope = ctx.view_3d(&kv, n_nope, 1, nt, hs, hs, 0)?;
    let kv_pe_in = ctx.view_3d(&kv, n_rot, 1, nt, hs, hs, n_nope as usize * f32_size)?;
    let kv_pe = ctx.rope_ext(
        &kv_pe_in,
        &pos,
        None,
        n_rot as i32,
        ROPE_MODE_NORM,
        rope_orig,
        rope,
    )?;
    let kv_full = ctx.concat(&kv_nope, &kv_pe, 0)?;
    ctx.compute(&kv_full, threads())?;
    Ok((q_full, kv_full))
}

/// What one pass through a compressed layer produces:
/// `(ring kv before this batch, ring score before, this batch's kv, its score)`.
///
/// The "before" halves are what the batch front-pads with; returning them rather
/// than re-reading the cache is what stops a batch summarising itself.
type CompressorRows = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// The `kv` and `score` projections a compressed layer needs, and the ring slide.
///
/// Split out of [`compressor`] because it must run on **every** pass through a
/// compressed layer, while the summary itself is only built when a block
/// completes. A step that completes no block still contributes its row to the
/// window that the *next* completed block will summarise; skipping it would
/// leave a hole in the ring, and a hole does not fail — it summarises the wrong
/// span.
///
/// Returns the ring contents *as they were before this batch*, because that is
/// what the batch must front-pad with. Sliding first and reading second would
/// let a batch summarise itself.
#[allow(clippy::too_many_arguments)]
fn compressor_project<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    attn_norm: &Tensor<'c>,
    nt: i64,
    pos0: i64,
    overlap: bool,
    cache: &mut Deepseek4Cache,
) -> Result<CompressorRows> {
    let config = &fw.config;
    let head = config.kv_lora_rank as i64;
    let ratio = config.compress_block(il).expect("compressed layer");
    let wide = if overlap { 2 * head } else { head };
    let state_rows = if overlap { 8 } else { ratio };

    let kv = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.attn_compressor_kv.weight"))
            .expect("bound"),
        attn_norm,
    )?;
    let score = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.attn_compressor_gate.weight"))
            .expect("bound"),
        attn_norm,
    )?;
    // The gate's position embedding is indexed by the token's offset *within its
    // block*: `(pos0 + p) % ratio`, which equals `p % ratio` only at pos0 = 0.
    let pos_t = ctx.new_i32_1d(nt)?;
    pos_t.set_i32(
        &(0..nt)
            .map(|p| ((pos0 + p) % ratio) as i32)
            .collect::<Vec<i32>>(),
    )?;
    let ape = ctx.get_rows(
        weights
            .get(&format!("blk.{il}.attn_compressor_ape.weight"))
            .expect("bound"),
        &pos_t,
    )?;
    let score = ctx.add(&score, &ape)?;
    ctx.compute(&kv, threads())?;
    ctx.compute(&score, threads())?;

    let kv_vals = kv.to_vec_f32();
    let score_vals = score.to_vec_f32();

    let (ring_kv, ring_sc) = &mut cache.ring[il as usize];
    let prev_kv = ring_kv.clone();
    let prev_sc = ring_sc.clone();

    ring_kv.extend_from_slice(&kv_vals);
    ring_sc.extend_from_slice(&score_vals);
    let keep = (state_rows * wide) as usize;
    if ring_kv.len() > keep {
        ring_kv.drain(..ring_kv.len() - keep);
    }
    if ring_sc.len() > keep {
        ring_sc.drain(..ring_sc.len() - keep);
    }

    Ok((prev_kv, prev_sc, kv_vals, score_vals))
}

/// The overlap compressor (CSA) or the plain one (HCA), for a prefill.
///
/// Both summarise completed blocks of raw KV into one entry each. They differ
/// in more than a ratio: the overlap form keeps a state `2*n_embd_head` wide and
/// averages over **two** windows (`ratio` previous plus `ratio` current), while
/// the plain form is head-wide and uses the current window only.
///
/// The persistent ring llama.cpp maintains is not needed on a prefill:
/// `state_source_idx` resolves to an appended zero row for `pos < 0` and to the
/// current batch otherwise, so the ring is never read.
/// Argument count is high because the forward pass threads two `ggml`
/// contexts with *different* lifetimes -- a per-block compute arena and a
/// longer-lived weight context -- plus the model, the weight set and the
/// layer index. Bundling them into one struct is the obvious refactor and
/// the wrong one: it would force both contexts to share a lifetime, which
/// is exactly the invariant that keeps dropped weights from dangling.
#[allow(clippy::too_many_arguments)]
fn compressor<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    attn_norm: &Tensor<'c>,
    nt: i64,
    pos0: i64,
    overlap: bool,
    cache: &mut Deepseek4Cache,
) -> Result<Tensor<'c>> {
    let config = &fw.config;
    let head = config.kv_lora_rank as i64;
    let ratio = config.compress_block(il).expect("compressed layer");
    // Blocks are absolute. `b0` is the first block this batch completes and `b1`
    // one past the last; at pos0 = 0 that is `0..nt / ratio`, exactly what this
    // used to compute. A step summarises only the block it finishes, and the
    // rows of that block come partly from the ring and partly from the batch.
    let b0 = pos0 / ratio;
    let b1 = (pos0 + nt) / ratio;
    let n_blocks = b1 - b0;
    let wide = if overlap { 2 * head } else { head };
    let n_read = ratio * n_blocks;
    let state_rows = if overlap { 8 } else { ratio };

    let (prev_kv, prev_sc, kv_vals, score_vals) =
        compressor_project(fw, ctx, weights, il, attn_norm, nt, pos0, overlap, cache)?;

    let pad = if overlap { 1 } else { 0 };
    let total = state_rows + nt + pad;
    // Front-pad from the ring. At pos0 == 0 the ring is empty and this is the
    // block of zeros the verified prefill path has always used, so prefill stays
    // bit-identical; past that, these are the real preceding rows.
    let need = (state_rows * wide) as usize;
    let mut kv_buf = vec![0.0f32; need.saturating_sub(prev_kv.len())];
    kv_buf.extend_from_slice(&prev_kv);
    kv_buf.extend_from_slice(&kv_vals);
    kv_buf.extend(std::iter::repeat_n(0.0f32, (pad * wide) as usize));
    let kv_state = ctx.new_f32_2d(wide, total)?;
    kv_state.set_f32(&kv_buf)?;
    let mut sc_buf = vec![0.0f32; need.saturating_sub(prev_sc.len())];
    sc_buf.extend_from_slice(&prev_sc);
    sc_buf.extend_from_slice(&score_vals);
    // -inf so the softmax ignores the padding rather than averaging it in.
    sc_buf.extend(std::iter::repeat_n(
        f32::NEG_INFINITY,
        (pad * wide) as usize,
    ));
    let score_state = ctx.new_f32_2d(wide, total)?;
    score_state.set_f32(&sc_buf)?;

    let zero_row = (state_rows + nt) as i32;
    // The combined buffer is `state_rows` ring rows followed by this batch, so
    // absolute position `q` sits at `state_rows + (q - pos0)`. That is only
    // `state_rows + q` when pos0 is zero.
    //
    // The reach backwards is what fixes `state_rows`: the overlap half of block
    // `b0` reads from `b0 * ratio - ratio`, and with `b0 * ratio >= pos0 - ratio
    // + 1` that is at worst `pos0 - 2 * ratio + 1` — which is why 8 rows are kept
    // for a ratio of 4, and why a smaller ring would read past the front.
    let row_of = |q: i64| (state_rows + q - pos0) as i32;
    let mut idxs: Vec<i32> = Vec::new();
    if overlap {
        for b in b0..b1 {
            for j in 0..ratio {
                let p = b * ratio - ratio + j;
                idxs.push(if p < 0 { zero_row } else { row_of(p) });
            }
        }
    }
    for b in b0..b1 {
        for j in 0..ratio {
            idxs.push(row_of(b * ratio + j));
        }
    }
    debug_assert!(
        idxs.iter().all(|&i| i >= 0 && i <= zero_row),
        "compressor gathered outside the ring+batch buffer: pos0 {pos0}, blocks          {b0}..{b1}, state_rows {state_rows}"
    );
    let idx_t = ctx.new_i32_1d(idxs.len() as i64)?;
    idx_t.set_i32(&idxs)?;

    let f32_size = std::mem::size_of::<f32>();
    let row = wide as usize * f32_size;

    let mut halves = Vec::with_capacity(2);
    for src in [&kv_state, &score_state] {
        let rows = ctx.get_rows(src, &idx_t)?;
        let joined = if overlap {
            // The first `head` of one set of rows, and the *second* `head` of
            // the next: reading one entry per row summarises the wrong span.
            let prev = ctx.cont(&ctx.view_2d(&rows, head, n_read, row, 0)?)?;
            let cur = ctx.cont(&ctx.view_2d(
                &rows,
                head,
                n_read,
                row,
                n_read as usize * row + head as usize * f32_size,
            )?)?;
            let prev = ctx.reshape_3d(&prev, head, ratio, n_blocks)?;
            let cur = ctx.reshape_3d(&cur, head, ratio, n_blocks)?;
            ctx.concat(&prev, &cur, 1)?
        } else {
            ctx.reshape_3d(&rows, head, ratio, n_blocks)?
        };
        halves.push(ctx.cont(&ctx.permute(&joined, [1, 0, 2, 3])?)?);
    }
    let scores = halves.pop().expect("scores");
    let values = halves.pop().expect("values");

    let w = ctx.soft_max(&scores)?;
    let weighted = ctx.mul(&values, &w)?;
    let comp = ctx.sum_rows(&weighted)?;
    let comp = ctx.cont(&ctx.permute(&comp, [1, 0, 2, 3])?)?;
    let comp = ctx.rms_norm(&comp, config.rms_eps)?;
    let comp = ctx.mul(
        &comp,
        weights
            .get(&format!("blk.{il}.attn_compressor_norm.weight"))
            .expect("bound"),
    )?;

    // Rotated at the *block start* position, with the compressed base.
    let n_rot = config.n_rot as i64;
    let n_nope = config.n_rot_none() as i64;
    let hs = head as usize * f32_size;
    let nope = ctx.view_3d(&comp, n_nope, 1, n_blocks, hs, hs, 0)?;
    let pe_in = ctx.view_3d(
        &comp,
        n_rot,
        1,
        n_blocks,
        hs,
        hs,
        n_nope as usize * f32_size,
    )?;
    let comp_pos = ctx.new_i32_1d(n_blocks)?;
    comp_pos.set_i32(&(b0..b1).map(|b| (b * ratio) as i32).collect::<Vec<i32>>())?;
    let (rope, rope_orig) = fw.rope(il);
    let pe = ctx.rope_ext(
        &pe_in,
        &comp_pos,
        None,
        n_rot as i32,
        ROPE_MODE_NORM,
        rope_orig,
        rope,
    )?;
    let out = ctx.concat(&nope, &pe, 0)?;
    ctx.compute(&out, threads())?;
    Ok(out)
}

/// Attention over the raw window, and optionally the compressed summaries.
///
/// The raw half is causal **and sliding**: every layer's raw window is an SWA
/// window of `attention.sliding_window` (128). A plain causal mask passes on any
/// prompt shorter than the window and is wrong beyond it — which is exactly how
/// it went unnoticed until a 165-token capture. The compressed half is
/// visibility-limited instead: a token sees block `b` once that block is
/// complete and behind it.
/// Argument count is high because the forward pass threads two `ggml`
/// contexts with *different* lifetimes plus the model, the weight set and
/// the layer index. Bundling them would force both contexts to share a
/// lifetime, which is the invariant that stops dropped weights dangling.
#[allow(clippy::too_many_arguments)]
fn attention<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    q_full: &Tensor<'c>,
    kv_full: &Tensor<'c>,
    comp: Option<&Tensor<'c>>,
    nt: i64,
    pos0: i64,
    comp_block0: i64,
    cache: &mut Deepseek4Cache,
) -> Result<Tensor<'c>> {
    let config = &fw.config;
    let head = config.kv_lora_rank as i64;
    let n_head = config.n_head as i64;
    let groups = config.output_group_count as i64;
    let n_rot = config.n_rot as i64;
    let n_nope = config.n_rot_none() as i64;
    let f32_size = std::mem::size_of::<f32>();

    // Write this batch's latents into the persistent cache at their absolute
    // slots, then attend over the whole of it. A prefill starts at slot 0 and
    // fills 0..nt; a step at position p writes one row at p and reads 0..=p.
    // **There is deliberately no separate uncached path**: a `pos0 == 0` branch
    // that every existing test took would leave the incremental one unexercised,
    // and a wrong cache here returns fluent nonsense rather than an error.
    let kv_vals = kv_full.to_vec_f32();
    let raw = &mut cache.raw[il as usize];
    let at = (pos0 * head) as usize;
    bigtea_ggml::f32_to_f16(&kv_vals, &mut raw[at..at + kv_vals.len()]);

    let mut packed: Vec<u16> = raw.clone();
    if let Some(c) = comp {
        // The compressor returns only the blocks **this batch completed**, so
        // they append at their absolute index. Writing them from block 0 —
        // which was right while every pass started at position 0 — would make
        // a step overwrite the sequence's history with its own single block.
        let cv = c.to_vec_f32();
        let store = &mut cache.comp[il as usize];
        let at = (comp_block0 * head) as usize;
        bigtea_ggml::f32_to_f16(&cv, &mut store[at..at + cv.len()]);
        cache.comp_len[il as usize] = comp_block0 + cv.len() as i64 / head;
    }

    // The compressed half is present whenever the **sequence** has summaries, not
    // only when this batch produced some. Three steps in four complete no block,
    // and attending over the raw window alone on those would discard everything
    // the sequence had already compressed — silently, and only on the cached path.
    let has_comp = cache.comp_len[il as usize] > 0;
    let n_kv = if has_comp {
        packed.extend_from_slice(&cache.comp[il as usize]);
        2 * N_KV
    } else {
        N_KV
    };
    let k = ctx.new_f16_3d(head, n_kv, 1)?;
    let bytes: Vec<u8> = packed.iter().flat_map(|h| h.to_le_bytes()).collect();
    k.set_bytes(&bytes)?;

    let ratio = config.compress_block(il).unwrap_or(1);
    let window = config.sliding_window as i64;
    let mut mask = vec![0u8; (n_kv * nt) as usize * 2];
    for query in 0..nt {
        // The key axis is indexed by absolute position, so the query must be too
        // — otherwise a step at position 40 would mask everything before it.
        let q_abs = pos0 + query;
        let row = (query * n_kv) as usize * 2;
        for key in 0..N_KV {
            if key > q_abs || (window > 0 && q_abs - key >= window) {
                let at = row + key as usize * 2;
                mask[at..at + 2].copy_from_slice(&F16_NEG_INF);
            }
        }
        if has_comp {
            for blk in ((q_abs + 1) / ratio)..N_KV {
                let at = row + (N_KV + blk) as usize * 2;
                mask[at..at + 2].copy_from_slice(&F16_NEG_INF);
            }
        }
    }
    let mask_t = ctx.new_typed_2d(bigtea_gguf::GgmlType(1), n_kv, nt)?;
    mask_t.set_bytes(&mask)?;

    let q_perm = ctx.permute(q_full, [0, 2, 1, 3])?;
    let sinks = weights
        .get(&format!("blk.{il}.attn_sinks.weight"))
        .expect("bound");
    let scale = 1.0f32 / (head as f32).sqrt();
    let out = ctx.flash_attn_ext_with_sinks(&q_perm, &k, &k, &mask_t, sinks, scale)?;

    // The output is **de-roped** before projection. Skipping this leaves the
    // rotation baked into the residual stream, and no shape reveals it.
    let out = ctx.reshape_3d(&out, head, n_head, nt)?;
    let hs = head as usize * f32_size;
    let o_nope = ctx.view_3d(&out, n_nope, n_head, nt, hs, hs * n_head as usize, 0)?;
    let o_pe_in = ctx.view_3d(
        &out,
        n_rot,
        n_head,
        nt,
        hs,
        hs * n_head as usize,
        n_nope as usize * f32_size,
    )?;
    let pos = ctx.new_i32_1d(nt)?;
    pos.set_i32(&(pos0 as i32..(pos0 + nt) as i32).collect::<Vec<i32>>())?;
    let (rope, rope_orig) = fw.rope(il);
    let o_pe = ctx.rope_ext_back(
        &o_pe_in,
        &pos,
        None,
        n_rot as i32,
        ROPE_MODE_NORM,
        rope_orig,
        rope,
    )?;
    let out = ctx.concat(&o_nope, &o_pe, 0)?;

    // A batched matmul across `output_group_count` groups, not one matmul —
    // which is why the dimensions appear not to connect.
    let group_dim = n_head * head / groups;
    let out = ctx.reshape_3d(&out, group_dim, groups, nt)?;
    let out = ctx.cont(&ctx.permute(&out, [0, 2, 1, 3])?)?;
    let wo_a = weights
        .get(&format!("blk.{il}.attn_output_a.weight"))
        .expect("bound");
    let wo_a = ctx.reshape_3d(wo_a, group_dim, config.output_lora_rank as i64, groups)?;
    let oa = ctx.mul_mat(&wo_a, &out)?;
    let oa = ctx.cont(&ctx.permute(&oa, [0, 2, 1, 3])?)?;
    let oa = ctx.reshape_2d(&oa, config.output_lora_rank as i64 * groups, nt)?;
    let out = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.attn_output_b.weight"))
            .expect("bound"),
        &oa,
    )?;
    Ok(out)
}

/// The block tail: write attention back across the streams, then the FFN's own
/// gate block and `ffn_norm`.
///
/// A plain transformer does `x = x + f(x)`. This does
/// `x[dst] = f(x)*post[dst] + sum_src x[src]*comb[dst, src]`, with `comb` a
/// Sinkhorn-normalised `hc x hc`. None of that changes a shape.
///
/// The FFN's gates come from a **second, independent** mixes matmul against
/// `hc_ffn_fn` over the post-attention streams — reusing the attention block's
/// would be free of any error.
fn layer_tail<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    e: &Entry<'c>,
    attn_out: &Tensor<'c>,
    nt: i64,
) -> Result<(Tensor<'c>, Tensor<'c>, HcGates<'c>)> {
    let config = &fw.config;
    let streams = ctx.dsv4_hc_post(attn_out, &e.streams, &e.gates.post, &e.gates.comb)?;

    let flat = ctx.reshape_2d(&streams, config.hc_dim() as i64, nt)?;
    let normed = ctx.rms_norm(&flat, config.rms_eps)?;
    let mixes = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.hc_ffn_fn.weight"))
            .expect("bound"),
        &normed,
    )?;
    let gates = hc_gates(
        ctx,
        weights,
        config,
        &format!("blk.{il}.hc_ffn"),
        &mixes,
        nt,
    )?;

    let collapsed = ctx.dsv4_hc_pre(&streams, &gates.pre)?;
    let normed = ctx.rms_norm(&collapsed, config.rms_eps)?;
    let ffn_norm = ctx.mul(
        &normed,
        weights
            .get(&format!("blk.{il}.ffn_norm.weight"))
            .expect("bound"),
    )?;
    Ok((streams, ffn_norm, gates))
}

/// The router: probabilities, the six experts, and their normalised weights.
///
/// **Two entirely different selection schemes**, chosen by `hash_layer_count`.
/// The first three blocks look their experts up in `ffn_gate_tid2eid` by *token
/// id* — no top-k at all, and `exp_probs_b` unused. Every other block adds the
/// selection bias and takes `argsort_top_k`, where **the bias steers selection
/// only**: the weights are gathered from the *unbiased* probabilities.
fn moe_routing<'c>(
    fw: &Deepseek4Forward<'_>,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    il: u32,
    ffn_norm: &Tensor<'c>,
    tokens: &[i32],
) -> Result<(Tensor<'c>, Vec<i32>)> {
    let config = &fw.config;
    let nt = tokens.len() as i64;
    let n_expert = config.n_expert as i64;
    let n_used = config.n_expert_used as i64;

    let logits = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.ffn_gate_inp.weight"))
            .expect("bound"),
        ffn_norm,
    )?;
    // sqrt(softplus(x)) — `expert_gating_func 4`, neither softmax nor sigmoid.
    let probs = ctx.sqrt(&ctx.softplus(&logits)?)?;
    let probs3 = ctx.reshape_3d(&probs, 1, n_expert, nt)?;

    let topk = if il < config.hash_layer_count {
        let tok = ctx.new_i32_1d(nt)?;
        tok.set_i32(tokens)?;
        ctx.get_rows(
            weights
                .get(&format!("blk.{il}.ffn_gate_tid2eid.weight"))
                .expect("bound"),
            &tok,
        )?
    } else {
        let biased = ctx.add(
            &probs,
            weights
                .get(&format!("blk.{il}.exp_probs_b.bias"))
                .expect("bound"),
        )?;
        ctx.argsort_top_k(&biased, n_used as i32)?
    };
    ctx.compute(&topk, threads())?;
    let ids = topk.to_vec_i32();
    if std::env::var("BIGTEA_ROUTING").is_ok() {
        record_routing(il, n_expert as usize, &ids);
    }
    if std::env::var("BIGTEA_ROUTING_LAST").is_ok() {
        record_last_token(il, n_used as usize, &ids);
    }

    // Renormalised over the selected six only, then scaled. The divisor is
    // clamped at the smallest F16 normal, not at an epsilon.
    let w = ctx.get_rows(&probs3, &topk)?;
    let w2 = ctx.reshape_2d(&w, n_used, nt)?;
    let sum = ctx.clamp(&ctx.sum_rows(&w2)?, 6.103_515_6e-5, f32::INFINITY)?;
    let w_norm = ctx.div(&w2, &sum)?;
    if std::env::var("BIGTEA_ROUTING_WEIGHTS").is_ok() {
        ctx.compute(&w_norm, threads())?;
        record_routing_weights(il, n_used as usize, &w_norm.to_vec_f32());
    }
    let w3 = ctx.reshape_3d(&w_norm, 1, n_used, nt)?;
    let w_scaled = ctx.scale(&w3, config.expert_weights_scale)?;
    Ok((w_scaled, ids))
}

/// Read the expert slices these tokens route to, for **all three** expert
/// tensors of a layer at once, with several readers.
///
/// A stacked expert tensor is `[ne0, ne1, n_expert]` with equal slices, so slice
/// `i` starts at `i * size / n_expert`. Binding all 256 for one block is 3.19
/// GiB and does not fit this machine; the tokens' own selection is a fraction of
/// that.
///
/// # Why the destination is deliberately misaligned
///
/// Each slice is read straight into its final position in one stacked buffer,
/// so no byte is copied between the drive and `ggml`. That only works if the
/// memory address and the file offset agree modulo the sector size — and GGUF
/// pads tensor data to `general.alignment`, which is **32**, so V4-Flash's
/// experts sit at file offsets ≡ 2816 (mod 4096). A conventionally aligned
/// buffer can never match, and every byte bounces through a scratch.
///
/// The slices of one tensor are all the same size, and that size is a sector
/// multiple, so **one skew serves the whole stack**. Measured: 0.78 → 1.57
/// GiB/s, with 0.09% of bytes copied instead of 300%.
///
/// # Why all three tensors are read together
///
/// One reader cannot saturate an NVMe — the drive wants requests in flight, and
/// a single blocking read leaves most of it idle. Four readers measured 1.59 →
/// 1.99 GiB/s against a drive that does 2.37 GiB/s sequential.
///
/// An earlier attempt spawned readers **per tensor** and was *slower* than
/// serial: at one token that is only 6 slices per group, 129 groups per forward
/// pass, and the thread spawns cost more than the queue depth bought. Reading
/// gate, up and down together triples the work per group and cuts the groups to
/// 43, which is what makes the parallelism pay.
fn read_expert_slices(
    model: &Model,
    names: &[String],
    unique: &[i32],
    weights_of: &[u32],
    il: u32,
    cache: Option<&std::sync::Mutex<ExpertCache>>,
) -> Result<(Vec<ExpertStack>, u64)> {
    /// Where a slice's bytes come from. Both land in the same packed buffer, so
    /// the destination layout — and the sector skew that makes reads direct —
    /// is identical whether or not the cache is on.
    enum Src {
        Disk { offset: u64 },
        Memory(std::sync::Arc<[u8]>),
    }
    struct Job {
        name: usize,
        len: usize,
        src: Src,
    }

    let mut buffers = Vec::with_capacity(names.len());
    let mut total = 0u64;
    for name in names {
        let loc = model.location(name).expect("stacked tensor").clone();
        let n_expert = *loc.dims.last().expect("stacked");
        let slice = loc.size / n_expert;
        let bytes = unique.len() * slice as usize;
        let mut dims = loc.dims.clone();
        *dims.last_mut().expect("stacked") = unique.len() as u64;
        buffers.push((
            SkewedBuf::new(bytes, SkewedBuf::skew_for(loc.file_offset)),
            dims,
        ));
        total += bytes as u64;
    }

    // One job per slice per tensor, so every reader gets an equal share of the
    // bytes rather than an equal share of the tensors. A cached slice becomes a
    // copy job rather than disappearing, which keeps the destination spans
    // contiguous and lets the copies run on the same threads as the reads.
    let mut jobs = Vec::with_capacity(names.len() * unique.len());
    let mut misses: Vec<(usize, usize, SliceKey)> = Vec::new();
    let mut hit_bytes = 0u64;
    for (n, name) in names.iter().enumerate() {
        let loc = model.location(name).expect("stacked tensor");
        let slice = loc.size / *loc.dims.last().expect("stacked");
        for (p, e) in unique.iter().enumerate() {
            let key = slice_key(il, n as u8, *e as u32);
            let src = match cache {
                Some(c) => c.lock().expect("expert cache").request(key, weights_of[p]),
                None => None,
            };
            match src {
                Some(bytes) => {
                    hit_bytes += bytes.len() as u64;
                    jobs.push(Job {
                        name: n,
                        len: slice as usize,
                        src: Src::Memory(bytes),
                    });
                }
                None => {
                    misses.push((n, p, key));
                    jobs.push(Job {
                        name: n,
                        len: slice as usize,
                        src: Src::Disk {
                            offset: *e as u64 * slice,
                        },
                    });
                }
            }
        }
    }

    // Hand each reader disjoint destination spans *and its own file handle*.
    // Positioned reads need no locking in this code, but a synchronous handle is
    // serialised by the OS, so sharing one would leave the drive at queue depth
    // 1 no matter how many threads are spawned.
    let mut slots: Vec<Vec<(&Job, &mut [u8])>> = (0..READERS).map(|_| Vec::new()).collect();
    let mut cursors: Vec<&mut [u8]> = buffers.iter_mut().map(|(b, _)| &mut b[..]).collect();
    for (j, job) in jobs.iter().enumerate() {
        let cursor = std::mem::take(&mut cursors[job.name]);
        let (dst, rest) = cursor.split_at_mut(job.len);
        cursors[job.name] = rest;
        slots[j % READERS].push((job, dst));
    }

    let copied: usize = std::thread::scope(|scope| {
        let handles: Vec<_> = slots
            .into_iter()
            .enumerate()
            .map(|(slot, work)| {
                scope.spawn(move || {
                    let mut copied = 0usize;
                    for (job, dst) in work {
                        match &job.src {
                            Src::Disk { offset } => {
                                copied += model.read_range_into_via(
                                    &names[job.name],
                                    *offset,
                                    dst,
                                    slot,
                                )?;
                            }
                            Src::Memory(bytes) => dst.copy_from_slice(bytes),
                        }
                    }
                    Ok::<usize, crate::ArchError>(copied)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("reader thread did not panic"))
            .sum::<Result<usize>>()
    })?;

    // Offer what was actually read. The slices sit in the packed buffer in the
    // order of `unique`, so a miss's span is found from its position, and
    // `offer` copies only if it decides to keep — which past warm-up is rare.
    if let Some(c) = cache {
        let mut c = c.lock().expect("expert cache");
        for (n, p, key) in misses {
            let slice = buffers[n].0.len() / unique.len();
            c.offer(key, &buffers[n].0[p * slice..(p + 1) * slice]);
        }
    }

    if std::env::var("BIGTEA_IO_TIMING").is_ok() {
        eprintln!(
            "    io {} tensors x {} slices: {:.2}% copied, {:.0} MiB from cache",
            names.len(),
            unique.len(),
            copied as f64 / total.max(1) as f64 * 100.0,
            hit_bytes as f64 / (1 << 20) as f64,
        );
    }
    Ok((buffers, total))
}

/// The routed experts and the shared one, summed into the block's FFN output.
///
/// The shared expert runs for **every** token and is therefore resident weight;
/// confusing it with the 256 routed ones is the difference between a 7 GiB
/// resident set and a 144 GiB one. Both clamp their SwiGLU asymmetrically:
/// `(-inf, limit]` on the gate, `[-limit, limit]` on the up projection.
/// Argument count is high because the forward pass threads two `ggml`
/// contexts with *different* lifetimes -- a per-block compute arena and a
/// longer-lived weight context -- plus the model, the weight set and the
/// layer index. Bundling them into one struct is the obvious refactor and
/// the wrong one: it would force both contexts to share a lifetime, which
/// is exactly the invariant that keeps dropped weights from dangling.
#[allow(clippy::too_many_arguments)]
fn ffn<'c>(
    fw: &Deepseek4Forward<'_>,
    model: &Model,
    ctx: &'c Context,
    wctx: &'c Context,
    weights: &mut WeightSet<'c>,
    il: u32,
    ffn_norm: &Tensor<'c>,
    w_scaled: &Tensor<'c>,
    ids: &[i32],
    nt: i64,
) -> Result<Tensor<'c>> {
    let config = &fw.config;
    let n_embd = config.n_embd as i64;
    let n_used = config.n_expert_used as i64;
    let f32_size = std::mem::size_of::<f32>();
    let limit = config.swiglu_limit(il, false).unwrap_or(f32::INFINITY);
    let limit_sh = config.swiglu_limit(il, true).unwrap_or(f32::INFINITY);

    // ---- the shared expert ----
    let sh_gate = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.ffn_gate_shexp.weight"))
            .expect("bound"),
        ffn_norm,
    )?;
    let sh_gate = ctx.clamp(&sh_gate, f32::NEG_INFINITY, limit_sh)?;
    let sh_up = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.ffn_up_shexp.weight"))
            .expect("bound"),
        ffn_norm,
    )?;
    let sh_up = ctx.clamp(&sh_up, -limit_sh, limit_sh)?;
    let sh = ctx.mul_mat(
        weights
            .get(&format!("blk.{il}.ffn_down_shexp.weight"))
            .expect("bound"),
        &ctx.swiglu_split(&sh_gate, &sh_up)?,
    )?;

    // ---- the routed experts, read as slices ----
    let mut unique = ids.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let compact: Vec<i32> = ids
        .iter()
        .map(|e| unique.iter().position(|u| u == e).expect("in set") as i32)
        .collect();
    // How many of this block's tokens chose each unique expert. Reads are
    // deduplicated, so without this the cache cannot tell a hot expert from a
    // cold one — see `ExpertCache::request`.
    let mut selections = vec![0u32; unique.len()];
    for e in ids {
        if let Ok(p) = unique.binary_search(e) {
            selections[p] += 1;
        }
    }
    let mut dims_of = std::collections::HashMap::new();
    let t_exp = std::time::Instant::now();
    let names: Vec<String> = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
        .iter()
        .map(|s| format!("blk.{il}.{s}.weight"))
        .collect();
    let (buffers, exp_bytes) =
        read_expert_slices(model, &names, &unique, &selections, il, fw.cache.as_ref())?;
    for ((suffix, name), (buf, dims)) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
        .iter()
        .zip(&names)
        .zip(buffers)
    {
        let ty = model.location(name).expect("stacked tensor").ty;
        weights.bind(wctx, name, ty, &dims, buf)?;
        dims_of.insert(*suffix, dims);
    }
    if std::env::var("BIGTEA_BLOCK_TIMING").is_ok() {
        eprintln!(
            "  block {il:>2}  experts {:.2}s ({:.0} MiB, {} of {} slices)",
            t_exp.elapsed().as_secs_f64(),
            exp_bytes as f64 / (1 << 20) as f64,
            unique.len(),
            config.n_expert,
        );
    }
    let n_uniq = unique.len() as i64;
    let ids_t = ctx.new_i32_2d(n_used, nt)?;
    ids_t.set_i32(&compact)?;

    let stack = |suffix: &str| -> Result<Tensor<'c>> {
        let d = &dims_of[suffix];
        Ok(ctx.reshape_3d(
            weights
                .get(&format!("blk.{il}.{suffix}.weight"))
                .expect("bound"),
            d[0] as i64,
            d[1] as i64,
            n_uniq,
        )?)
    };

    let cur3 = ctx.reshape_3d(ffn_norm, n_embd, 1, nt)?;
    let gate = ctx.mul_mat_id(&stack("ffn_gate_exps")?, &cur3, &ids_t)?;
    let gate = ctx.clamp(&gate, f32::NEG_INFINITY, limit)?;
    let up = ctx.mul_mat_id(&stack("ffn_up_exps")?, &cur3, &ids_t)?;
    let up = ctx.clamp(&up, -limit, limit)?;
    let act = ctx.swiglu_split(&gate, &up)?;
    if std::env::var("BIGTEA_SPARSITY").is_ok() {
        // How much of the intermediate actually matters? The router picks 6 of
        // 256 experts; this asks how much of a CHOSEN expert is dead weight for
        // this token. Rows whose activation is negligible never reach the
        // output, so their  rows and  columns need not be read.
        ctx.compute(&act, threads())?;
        let v = act.to_vec_f32();
        let peak = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        let mut buckets = [0usize; 4]; // >1%, >0.1%, >0.01% of peak, and rest
        for x in &v {
            let r = x.abs() / peak.max(f32::MIN_POSITIVE);
            if r > 1e-2 {
                buckets[0] += 1
            } else if r > 1e-3 {
                buckets[1] += 1
            } else if r > 1e-4 {
                buckets[2] += 1
            } else {
                buckets[3] += 1
            }
        }
        let n = v.len() as f64;
        eprintln!(
            "  sparsity blk {il:>2}: >1% {:.1}%  >0.1% {:.1}%  >0.01% {:.1}%  negligible {:.1}%",
            100.0 * buckets[0] as f64 / n,
            100.0 * buckets[1] as f64 / n,
            100.0 * buckets[2] as f64 / n,
            100.0 * buckets[3] as f64 / n,
        );
    }
    let down = ctx.mul_mat_id(&stack("ffn_down_exps")?, &act, &ids_t)?;
    let weighted = ctx.mul(&down, w_scaled)?;

    // Sum across the six experts as six strided views and five adds, which is
    // the shape llama.cpp uses.
    let row = n_embd as usize * f32_size;
    let mut moe_out: Option<Tensor<'c>> = None;
    for j in 0..n_used as usize {
        let v = ctx.view_2d(&weighted, n_embd, nt, row * n_used as usize, j * row)?;
        moe_out = Some(match moe_out {
            None => v,
            Some(acc) => ctx.add(&acc, &v)?,
        });
    }
    let out = ctx.add(&moe_out.expect("experts"), &sh)?;
    Ok(out)
}

/// Bind one always-read tensor, from RAM if it is resident and from disk if it
/// is not. Returns its size, so a caller can report what it moved.
///
/// # Why residency is the difference between a demo and a runner
///
/// V4-Flash's always-read weights are 7.38 GiB and every one of them is touched
/// on **every token**. Read per block, they cost 7.1s of a 5-token prefill — 23%
/// — and a generation loop would pay that again for each token produced, forever.
/// Held in RAM they cost one read for the whole session.
///
/// Binding from the resident set is a refcount bump, not a copy: the same bytes
/// are pointed at by a fresh `ggml` tensor on every block of every token, and
/// copying 7.38 GiB per token to achieve that would defeat the purpose.
///
/// Falling back to disk is not a failure path but the design working: the
/// budget is a hard ceiling, and a machine too small for the whole set streams
/// the remainder rather than swapping. Swapping is slower than the streaming it
/// replaces.
fn bind_dense<'c>(
    fw: &Deepseek4Forward<'_>,
    wctx: &'c Context,
    weights: &mut WeightSet<'c>,
    name: &str,
    prefetched: &std::collections::HashMap<String, std::sync::Arc<SkewedBuf>>,
) -> Result<u64> {
    let loc = fw.model.location(name).expect("present").clone();
    if let Some(shared) = fw.resident.and_then(|r| r.get_shared(name)) {
        weights.bind_shared(wctx, name, loc.ty, &loc.dims, shared)?;
        return Ok(0);
    }
    // Read by `prefetch_dense` on several handles at once; falling back here
    // keeps the function correct if the prefetch was skipped or failed.
    let data = match prefetched.get(name) {
        Some(d) => d.clone(),
        None => fw.model.read_tensor_shared(name)?,
    };
    let n = data.len() as u64;
    weights.bind_shared(wctx, name, loc.ty, &loc.dims, data)?;
    Ok(n)
}

/// Read a block's non-resident always-read tensors in parallel, before binding.
///
/// # Why this is separate from binding
///
/// When the always-read set does not fit, every one of these is re-read on every
/// token — 147 MiB per block, **2.1 s per token** measured on a machine 3.1 GiB
/// short. That path read one tensor at a time through one file handle, which is
/// the worst case for an NVMe: serialised by the OS *and* at queue depth 1.
///
/// Binding cannot be parallelised — `ggml` contexts are not thread-safe and the
/// graph must be built in order — but reading can. So the reads are hoisted out,
/// spread across the shard's handle pool, and the bind loop that follows finds
/// its bytes already in memory.
///
/// Resident tensors are skipped entirely: `get_shared` is a refcount bump, and
/// prefetching them would read what is already in RAM.
fn prefetch_dense(
    fw: &Deepseek4Forward<'_>,
    names: &[String],
) -> Result<std::collections::HashMap<String, std::sync::Arc<SkewedBuf>>> {
    let missing: Vec<&String> = names
        .iter()
        .filter(|n| fw.resident.and_then(|r| r.get_shared(n)).is_none())
        .collect();
    if missing.len() < 2 {
        // One tensor has nothing to overlap with, and the common case — a fully
        // resident set — has none at all.
        return Ok(std::collections::HashMap::new());
    }

    let model = fw.model;
    let chunks: Vec<Vec<&String>> = (0..READERS)
        .map(|s| {
            missing
                .iter()
                .skip(s)
                .step_by(READERS)
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    let out = std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .into_iter()
            .enumerate()
            .map(|(slot, work)| {
                scope.spawn(move || {
                    let mut got = Vec::with_capacity(work.len());
                    for name in work {
                        let loc = model.location(name).expect("present");
                        let mut buf =
                            SkewedBuf::new(loc.size as usize, SkewedBuf::skew_for(loc.file_offset));
                        model.read_range_into_via(name, 0, &mut buf[..], slot)?;
                        got.push((name.clone(), std::sync::Arc::new(buf)));
                    }
                    Ok::<_, crate::ArchError>(got)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("dense prefetch thread did not panic"))
            .collect::<Result<Vec<_>>>()
    })?;
    Ok(out.into_iter().flatten().collect())
}

/// One whole block, in its own arena, streams in and streams out as floats.
///
/// Owning the arena per block is what makes depth free: chaining blocks inside
/// one `ggml` context costs hundreds of megabytes each. Freeing weights *inside*
/// a context instead would be unsound — every `compute` rebuilds the graph
/// through its sources, so a dropped buffer reads freed memory successfully.
pub fn block(
    fw: &Deepseek4Forward<'_>,
    cache: &mut Deepseek4Cache,
    il: u32,
    tokens: &[i32],
    pos0: i64,
    streams_in: Option<&[f32]>,
    arena: usize,
) -> Result<Streams> {
    let config = fw.config.clone();
    let nt = tokens.len() as i64;
    let t_block = std::time::Instant::now();
    let ctx = Context::new(arena)?;
    let wctx = Context::new_no_alloc(32 << 20)?;
    let arena_secs = t_block.elapsed().as_secs_f64();
    let mut weights = WeightSet::new();

    let mut names = fw.block_tensor_names(il);
    if il == 0 {
        names.push("token_embd.weight".to_string());
    }
    let t_bind = std::time::Instant::now();
    let prefetched = prefetch_dense(fw, &names)?;
    let mut dense_bytes = 0u64;
    for name in &names {
        dense_bytes += bind_dense(fw, &wctx, &mut weights, name, &prefetched)?;
    }
    let dense_secs = t_bind.elapsed().as_secs_f64();

    let streams = match streams_in {
        None => embed(&ctx, &weights, &config, tokens)?,
        Some(v) => {
            let t = ctx.new_f32_3d(config.n_embd as i64, config.hc_mult as i64, nt)?;
            t.set_f32(v)?;
            t
        }
    };

    let t_phase = std::time::Instant::now();
    let e = entry(fw, &ctx, &weights, il, streams, nt)?;
    let (q, kv) = q_and_kv(fw, &ctx, &weights, il, &e.attn_norm, nt, pos0)?;
    let qkv_secs = t_phase.elapsed().as_secs_f64();

    // Which attention runs is decided by the block's compression ratio *and*
    // whether a block has completed yet: below the first boundary a compressed
    // layer falls back to Raw, exactly as llama.cpp's guards do.
    let kind = config.attention_kind_from_ratio(il).expect("known ratio");
    // "Does this batch complete a block?" — absolute, not relative. `nt / r` is
    // zero for any single-token step, so a step would never build a summary and,
    // worse, would tell `attention` there was no compressed half at all.
    let fired = config
        .compress_block(il)
        .is_some_and(|r| (pos0 + nt) / r > pos0 / r);
    // The compressor front-pads `state_rows` zeros in place of a persistent ring,
    // which is exact only while the previous window is inside this batch. On an
    // incremental step it is in the past, and those zeros would summarise the
    // wrong span **without failing**. Refuse rather than return fluent nonsense;
    // the ring is the next piece of R3.
    let comp = match (kind, fired) {
        (AttentionKind::Raw, _) | (_, false) => None,
        (AttentionKind::CompressedSparse, true) => Some(compressor(
            fw,
            &ctx,
            &weights,
            il,
            &e.attn_norm,
            nt,
            pos0,
            true,
            cache,
        )?),
        (AttentionKind::HeavilyCompressed, true) => Some(compressor(
            fw,
            &ctx,
            &weights,
            il,
            &e.attn_norm,
            nt,
            pos0,
            false,
            cache,
        )?),
    };
    let t_phase = std::time::Instant::now();
    let attn_out = attention(
        fw,
        &ctx,
        &weights,
        il,
        &q,
        &kv,
        comp.as_ref(),
        nt,
        pos0,
        pos0 / config.compress_block(il).unwrap_or(1),
        cache,
    )?;
    let attn_secs = t_phase.elapsed().as_secs_f64();

    let t_phase = std::time::Instant::now();
    let (streams, ffn_norm, ffn_gates) = layer_tail(fw, &ctx, &weights, il, &e, &attn_out, nt)?;
    let (w_scaled, ids) = moe_routing(fw, &ctx, &weights, il, &ffn_norm, tokens)?;
    let tail_secs = t_phase.elapsed().as_secs_f64();

    let t_phase = std::time::Instant::now();
    let ffn_out = ffn(
        fw,
        fw.model,
        &ctx,
        &wctx,
        &mut weights,
        il,
        &ffn_norm,
        &w_scaled,
        &ids,
        nt,
    )?;
    let ffn_secs = t_phase.elapsed().as_secs_f64();

    let out = ctx.dsv4_hc_post(&ffn_out, &streams, &ffn_gates.post, &ffn_gates.comb)?;
    ctx.compute(&out, threads())?;

    if std::env::var("BIGTEA_BLOCK_TIMING").is_ok() {
        eprintln!(
            "  block {il:>2}  arena {arena_secs:.2}  dense {dense_secs:.2} ({:.0} MiB)               qkv {qkv_secs:.2}  attn {attn_secs:.2}  tail {tail_secs:.2}  ffn {ffn_secs:.2}               total {:.2}",
            dense_bytes as f64 / (1 << 20) as f64,
            t_block.elapsed().as_secs_f64(),
        );
    }
    Ok(out.to_vec_f32())
}

/// The output head: the **last** token's streams, collapsed and projected.
///
/// Its gate block is the `pre` half only — nothing writes back into the streams
/// after this, so there is no `post` and no combination matrix.
pub fn head(fw: &Deepseek4Forward<'_>, streams: &[f32], arena: usize) -> Result<Vec<f32>> {
    let config = &fw.config;
    let ctx = Context::new(arena)?;
    let wctx = Context::new_no_alloc(8 << 20)?;
    let mut weights = WeightSet::new();
    let names: Vec<String> = [
        "output_hc_fn.weight",
        "output_hc_scale.weight",
        "output_hc_base.weight",
        "output_norm.weight",
        "output.weight",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // The head runs once per pass, but `output.weight` alone is large enough
    // that reading it beside the others is worth the same parallelism.
    let prefetched = prefetch_dense(fw, &names)?;
    for name in &names {
        bind_dense(fw, &wctx, &mut weights, name, &prefetched)?;
    }

    let hc = config.hc_mult as i64;
    let n_embd = config.n_embd as i64;
    let hc_dim = config.hc_dim() as usize;
    let last = &streams[streams.len() - hc_dim..];

    let x = ctx.new_f32_3d(n_embd, hc, 1)?;
    x.set_f32(last)?;
    let flat = ctx.reshape_2d(&x, hc_dim as i64, 1)?;
    let normed = ctx.rms_norm(&flat, config.rms_eps)?;
    let mixes = ctx.mul_mat(weights.get("output_hc_fn.weight").expect("bound"), &normed)?;

    let scale = ctx.view_1d(weights.get("output_hc_scale.weight").expect("bound"), 1, 0)?;
    let base = ctx.view_1d(weights.get("output_hc_base.weight").expect("bound"), hc, 0)?;
    let gated = ctx.sigmoid(&ctx.add(&ctx.mul(&mixes, &scale)?, &base)?)?;
    let eps = ctx.new_f32_1d(hc)?;
    eps.set_f32(&vec![1e-6f32; hc as usize])?;
    let pre = ctx.add(&gated, &eps)?;

    let collapsed = ctx.dsv4_hc_pre(&x, &pre)?;
    let normed = ctx.rms_norm(&collapsed, config.rms_eps)?;
    let result = ctx.mul(&normed, weights.get("output_norm.weight").expect("bound"))?;
    let logits = ctx.mul_mat(weights.get("output.weight").expect("bound"), &result)?;
    ctx.compute(&logits, threads())?;
    Ok(logits.to_vec_f32())
}

/// Prefill: every block in order, then the head. Returns one logit per token id.
pub fn prefill(fw: &Deepseek4Forward<'_>, tokens: &[i32], arena: usize) -> Result<Vec<f32>> {
    // `attention` builds one F16 cache of `kv_lora_rank * N_KV` and indexes it by
    // absolute position, so a longer sequence used to run for eight seconds and
    // then panic on a slice range. Refuse it here, before any weight is read,
    // with a message that says what the limit is and why.
    if tokens.len() > N_KV as usize {
        return Err(crate::ArchError::ContextTooLong {
            tokens: tokens.len(),
            limit: N_KV as usize,
        });
    }
    let mut cache = Deepseek4Cache::new(fw.config.n_layer, fw.config.kv_lora_rank);
    forward(fw, &mut cache, tokens, arena)
}

/// One forward pass over `tokens`, appended to whatever `cache` already holds.
///
/// This is the single implementation behind both [`prefill`] and [`step`]: a
/// prefill is this against an empty cache, and a step is this with one token
/// against a full one. Keeping them one path is deliberate — a separate
/// uncached route would be the one every existing test took, leaving the
/// incremental one unexercised until a user found it.
pub fn forward(
    fw: &Deepseek4Forward<'_>,
    cache: &mut Deepseek4Cache,
    tokens: &[i32],
    arena: usize,
) -> Result<Vec<f32>> {
    let pos0 = cache.n_past as i64;
    if pos0 as usize + tokens.len() > N_KV as usize {
        return Err(crate::ArchError::ContextTooLong {
            tokens: pos0 as usize + tokens.len(),
            limit: N_KV as usize,
        });
    }
    let mut streams = block(fw, cache, 0, tokens, pos0, None, arena)?;
    for il in 1..fw.config.n_layer {
        streams = block(fw, cache, il, tokens, pos0, Some(&streams), arena)?;
    }
    cache.n_past += tokens.len();
    head(fw, &streams, arena)
}

/// Advance one token, reusing everything the cache already holds.
///
/// Costs one forward pass over a **single** token instead of over the whole
/// sequence. Both the arithmetic and the disk traffic collapse: a step selects
/// 6 distinct experts per layer where a 166-token pass selects 122.8.
pub fn step(
    fw: &Deepseek4Forward<'_>,
    cache: &mut Deepseek4Cache,
    token: i32,
    arena: usize,
) -> Result<Vec<f32>> {
    forward(fw, cache, &[token], arena)
}

#[cfg(test)]
mod routing_tests {
    use super::{pool_passes, record_into};

    /// Selections land in the newest pass, and pooling sums every pass.
    ///
    /// Pooling is what the printed report uses, and getting it wrong would look
    /// like a routing finding rather than a bug.
    #[test]
    fn passes_are_counted_separately_and_pool_correctly() {
        let mut log = vec![Vec::new()];
        record_into(&mut log, 0, 4, &[1, 1, 2]);
        log.push(Vec::new());
        record_into(&mut log, 0, 4, &[2, 3]);

        assert_eq!(log[0][0], vec![0, 2, 1, 0], "pass 0 keeps only its own");
        assert_eq!(log[1][0], vec![0, 0, 1, 1], "pass 1 starts from zero");
        assert_eq!(pool_passes(&log)[0], vec![0, 2, 2, 1]);
    }

    /// The property R0.1 rests on: because the model is causal, a later pass
    /// re-counts every earlier token, so `pass[k] - pass[k-1]` is exactly the
    /// token generated in between. A regression that carried counts forward, or
    /// reset them, would break the subtraction silently — the deltas would still
    /// be numbers, just the wrong ones.
    #[test]
    fn later_pass_minus_earlier_is_the_new_token() {
        let prompt = [3i32, 7, 3];
        let generated = [5i32];

        let mut log = vec![Vec::new()];
        record_into(&mut log, 0, 8, &prompt);
        log.push(Vec::new());
        record_into(&mut log, 0, 8, &prompt); // the re-prefill
        record_into(&mut log, 0, 8, &generated);

        let delta: Vec<i64> = log[1][0]
            .iter()
            .zip(&log[0][0])
            .map(|(b, a)| i64::from(*b) - i64::from(*a))
            .collect();
        assert!(
            delta.iter().all(|d| *d >= 0),
            "a delta must never go negative"
        );
        assert_eq!(delta.iter().sum::<i64>(), generated.len() as i64);
        assert_eq!(delta[5], 1, "the delta is the generated token alone");
    }

    /// A layer never selected leaves no row, and pooling must not invent one.
    #[test]
    fn pooling_tolerates_ragged_passes() {
        let mut log = vec![Vec::new()];
        record_into(&mut log, 2, 4, &[0]);
        log.push(Vec::new());
        record_into(&mut log, 0, 4, &[1]);

        let pooled = pool_passes(&log);
        assert_eq!(pooled.len(), 3);
        assert_eq!(pooled[0], vec![0, 1, 0, 0]);
        assert_eq!(pooled[2], vec![1, 0, 0, 0]);
    }
}
