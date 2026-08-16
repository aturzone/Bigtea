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
    /// Raw KV latents, F16, `kv_lora_rank * RAW_RING` per layer.
    ///
    /// **Slot index is `position % RAW_RING`.** It used to be the absolute
    /// position, which is why position 256 wrote past the end. The mask and the
    /// ring had to be rewritten together — the mask's key axis is no longer the
    /// slot index, it is a span of absolute positions gathered out of the ring,
    /// so the two cannot drift apart without the shapes disagreeing.
    ///
    /// Sound only because raw attention is **sliding**; see [`RAW_RING`].
    raw: Vec<Vec<u16>>,
    /// Compressed summaries, F16, `kv_lora_rank * blocks` per layer. Slot index
    /// is the **block** index, not the position.
    ///
    /// **Grows; it cannot be a ring.** The compressed half is
    /// visibility-limited rather than windowed — a token sees *every* block that
    /// is complete and behind it — so nothing here ever becomes unreachable. At
    /// ratio 4 that is one slot per 4 tokens, which is what a 1M context costs
    /// and why it is grown on demand rather than sized up front.
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
        let per_layer = kv_lora_rank as usize * RAW_RING as usize;
        Deepseek4Cache {
            raw: vec![vec![0u16; per_layer]; n_layer as usize],
            // Empty: grown to whatever the sequence reaches. A layer with no
            // compressor never grows one at all.
            comp: vec![Vec::new(); n_layer as usize],
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
        for layer in self.raw.iter_mut() {
            layer.fill(0);
        }
        // Released rather than zeroed: a new sequence regrows only what it
        // reaches, so a long one does not leave its footprint behind.
        for layer in self.comp.iter_mut() {
            layer.clear();
            layer.shrink_to_fit();
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

/// Reader slots handed to the **background** prefetch when R2 overlap is on.
///
/// # Why the pool has to be partitioned rather than shared
///
/// `read_range_into_via` requires concurrent readers to pass **distinct**
/// slots: a synchronous handle is serialised by the OS, so two threads on one
/// handle hold the drive at queue depth 1. That is the bug whose fix was worth
/// 1.32x on expert reads, and a background prefetch that reused slots the
/// foreground is already using would reintroduce it **by hand** — surfacing not
/// as an error but as "overlap does not help".
///
/// So the eight slots are split: the foreground keeps [`foreground_readers`]
/// of them and the prefetch gets the rest.
///
/// **The split is not free, and the cost is bandwidth rather than handles.**
/// Measured on V4-Flash with 3.10 GiB streaming, one session:
///
/// ```text
///                    dense    expert
/// no overlap         2.56s     7.02s
/// prefetch 2         0.02s     8.39s
/// prefetch 4         0.04s     8.43s
/// ```
///
/// Two readers hide the dense read as completely as four do, and four cost the
/// experts no more than two — so the toll is the *drive*, not the pool split,
/// and the smaller share is strictly better. `BIGTEA_PREFETCH_READERS`
/// overrides it, because the right number depends on the drive.
fn prefetch_readers() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("BIGTEA_PREFETCH_READERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0 && *n < READERS)
            .unwrap_or(2)
    })
}

/// Set for the duration of a pass that is **actually** prefetching.
///
/// Not the same as [`prefetch_overlap`]. With the always-read set fully
/// resident there is nothing to prefetch, and shrinking the foreground pool
/// anyway would be a pure loss — the expert reads would give up handles to a
/// thread that reads nothing. `forward` decides once per pass and sets this;
/// `read_expert_slices` reads it.
static PREFETCHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Slots the foreground (expert slices, and the in-block dense fallback) may
/// use. The whole pool unless a background prefetch is actually running.
fn foreground_readers() -> usize {
    if PREFETCHING.load(std::sync::atomic::Ordering::Relaxed) {
        READERS - prefetch_readers()
    } else {
        READERS
    }
}

/// Whether to read the next block's always-read weights while this one
/// computes.
///
/// **On by default, and that default is a measurement** — see
/// `docs/graph/research/r2-overlap-2026-08-11.md`. Worth ~1.09x on generation
/// with 3.10 GiB of the always-read set still streaming, and switched off
/// entirely when there is no shortfall, so the case it cannot help is a case it
/// does not touch. `BIGTEA_PREFETCH_OVERLAP=0` disables it.
fn prefetch_overlap() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("BIGTEA_PREFETCH_OVERLAP")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Positions the raw latent ring holds, and therefore the longest span a
/// single pass may need.
///
/// # Why a ring is exact here rather than an approximation
///
/// V4-Flash's raw attention is **sliding**, not merely causal: the container
/// declares `attention.sliding_window = 128`, and the mask drops any key more
/// than 128 positions behind its query. A position older than the window can
/// never be read again, so overwriting its slot loses nothing. That is what
/// makes wraparound sound, and it is a property of *this* model checked against
/// the container rather than an assumption — `sliding_window = 0` would mean
/// full causal attention, where a ring would quietly drop keys that are still
/// visible, so that case is refused.
///
/// The size is not the window, though. One pass covers queries
/// `pos0 ..= pos0 + nt - 1`, and the earliest of them still reaches `window - 1`
/// positions further back, so `window + nt - 1` distinct positions must be live
/// at once. 1024 leaves room for a 897-token batch, past any prefill block this
/// runner uses; beyond that [`forward`] refuses with both numbers rather than
/// wrapping over data the pass is still going to read.
///
/// 512 latents x 1024 positions x 2 bytes x 43 layers = 45 MB.
const RAW_RING: i64 = 1024;

/// Threads for every `ggml` graph evaluation in this file.
///
/// A constant here was a guess. `compute(&t, 0)` runs on **one** thread — the
/// count is floored at 1, not defaulted to all cores — so the number has to be
/// passed explicitly, and once it is passed explicitly it deserves to be
/// measured rather than assumed. `BIGTEA_THREADS` (i.e. `-t`) overrides it.
///
/// **Generation and prefill want opposite counts here, exactly as they do on
/// the dense path.** Both measured on V4-Flash:
///
/// ```text
/// generation (-n 4)      threads      1      2      4      8     20
///                        tok/s      0.331  0.378  0.380  0.346  0.296
///
/// prefill (180 tokens)   threads                   4            20
///                        tok/s                   2.54          3.27
/// ```
///
/// So a single-token step loses **1.28x** at every core, and a prefill loses
/// **1.29x** at four. One number cannot serve both, and capping this function
/// was tried first and would have traded one regression for the other.
///
/// A single-token step is a stack of matrix-vector products, and past a handful
/// of threads the per-node barrier costs more than the work it splits.
/// Qwen3-30B-A3B is the extreme case at **2.4x**, where the expert matmuls want
/// exactly one thread. A prefill block multiplies many columns at once and
/// scales with cores.
///
/// **This retires a claim that was in `CLAUDE.md`**: "4/12/20 threads all cost
/// the same on a V4-Flash prefill". That held at 5 tokens, where the pass is
/// almost entirely disk. At 180 it is 2.54 against 3.27.
/// Both counts are resolved **once**. `threads()` is called at every
/// `ctx.compute`, which is thousands of times per token, and the first version
/// of this split called `std::env::var` on each one — that locks the process
/// environment and allocates a `String`. It cost more than the split saved:
/// generation went to 0.267 tok/s, *below* the 0.296 it was meant to fix. The
/// per-call work is now an atomic load and a branch.
fn threads() -> usize {
    use std::sync::OnceLock;
    static GEN: OnceLock<usize> = OnceLock::new();
    static BAT: OnceLock<usize> = OnceLock::new();
    if BATCH.load(std::sync::atomic::Ordering::Relaxed) > 1 {
        *BAT.get_or_init(|| env_threads("BIGTEA_THREADS_BATCH").unwrap_or_else(all_cores))
    } else {
        // Cap at 4: the dense path measures its own peak per model with
        // `ThreadTuner`, and this file has no equivalent, so it takes the shape
        // of the curve rather than its exact peak. Erring low is cheap here —
        // one thread costs 13% — and erring high cost 1.28x.
        *GEN.get_or_init(|| env_threads("BIGTEA_THREADS").unwrap_or_else(|| all_cores().min(4)))
    }
}

/// Tokens in the pass currently being evaluated, set once by [`forward`].
///
/// The alternative is threading `n_tokens` through ten call sites across as
/// many functions, none of which otherwise care. This engine evaluates one pass
/// at a time — `bigtea-serve` serialises requests because the model has one KV
/// cache — so a process-wide value is accurate rather than merely convenient.
static BATCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

fn all_cores() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

fn env_threads(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&t| t > 0)
}

/// The dense tensors it is **safe** to rearrange, named one at a time.
///
/// # Why this is an allow-list and not an exclusion list
///
/// The dense path excludes three known-bad uses and repacks the rest, which is
/// right there because a new tensor arriving in a llama-family container is
/// almost certainly an ordinary `mul_mat` weight. Here the default has to be
/// the other way round.
///
/// Repacking interleaves rows. Every use except "first operand of `mul_mat`,
/// two-dimensional" reads the bytes by position — and **not one of them
/// fails**: `get_rows` returns the wrong row, a `view_1d` at a byte offset
/// returns the wrong slice, a `reshape_3d` cuts the matrix in the wrong places.
/// All three produce numbers, and numbers become fluent text. This
/// architecture's graph has four such uses among twenty-odd tensors, so
/// guessing from the name would be wrong roughly a fifth of the time, silently.
///
/// The audit behind the list, by what each tensor is actually passed to:
///
/// | tensor | use | repack |
/// |---|---|---|
/// | `token_embd` | `get_rows` by token id | **no** |
/// | `attn_compressor_ape` | `get_rows` by within-block position | **no** |
/// | `ffn_gate_tid2eid` | `get_rows` by token id | **no** |
/// | `*_hc_scale`, `*_hc_base` | `view_1d` at a byte offset | **no** |
/// | `attn_output_a` | `reshape_3d` into a grouped `mul_mat` | **no** |
/// | `attn_sinks` | sinks argument of `flash_attn_ext` | **no** |
/// | `*_norm` | `mul`, elementwise | no — and F32, so nothing to pack |
/// | `blk.*.ffn_*_exps` | **routed, streamed from disk** | **never** |
/// | the fourteen below | `mul_mat(w, x)`, 2-D | **yes** |
///
/// The routed experts are the load-bearing exclusion. They are bound zero-copy
/// from a pointer into the mapped container, one slice at a time, and never go
/// near [`bind_dense`] — that is what lets a 144 GB model run on a 15.7 GiB
/// machine at all. Repacking them would need the whole bank in RAM, which is
/// the thing this engine exists to avoid.
const REPACKABLE_DENSE: [&str; 14] = [
    "hc_attn_fn",
    "hc_ffn_fn",
    "attn_q_a",
    "attn_q_b",
    "attn_kv",
    // `attn_output_a` is deliberately absent — see the table above.
    "attn_output_b",
    "ffn_gate_inp",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
    "attn_compressor_kv",
    "attn_compressor_gate",
    "output_hc_fn",
    "output",
];

/// Whether `name` is one of the tensors [`REPACKABLE_DENSE`] allows.
///
/// Matches the suffix after the block prefix, so `blk.7.attn_kv.weight` and the
/// un-prefixed `output.weight` are both recognised — and
/// `blk.7.attn_kv_a_norm.weight` is not, because the comparison is on the whole
/// suffix rather than a prefix of it.
fn is_repackable_dense(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".weight") else {
        return false;
    };
    let suffix = match stem.strip_prefix("blk.") {
        Some(rest) => match rest.split_once('.') {
            Some((n, suffix)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => suffix,
            _ => return false,
        },
        None => stem,
    };
    REPACKABLE_DENSE.contains(&suffix)
}

/// Always-read weights rearranged into the CPU kernels' layout, once, at load.
///
/// # Why this is not simply `bind_repacked` inside [`bind_dense`]
///
/// Repacking is worth **1.42x on prefill** on the dense path, where
/// `load_resident` builds one `WeightSet` that lives for the session. V4-Flash
/// owns an arena **per block** — chaining 43 blocks into one `ggml` context
/// costs hundreds of megabytes each — so it builds a fresh context and a fresh
/// `WeightSet` for every block of every pass. Rearranging inside that loop
/// would re-do the whole always-read set 43 times per token: not a smaller win,
/// a large loss.
///
/// So the rearrangement happens once, here, and each block binds the result by
/// pointing a fresh tensor at bytes that are already in the right order.
///
/// # Why only tensors that are already resident
///
/// The rearranged copy lives in memory this process owns. Repacking a tensor
/// that was **not** resident would quietly spend RAM the residency budget had
/// already decided it did not have — and that budget exists because
/// over-committing makes the OS swap, which is slower than the streaming it
/// replaces.
///
/// For one that *is* resident the exchange is even: the original is taken out
/// of the set and dropped as soon as the rearranged copy exists, so the peak is
/// one tensor rather than a second whole set.
pub struct RepackedDense {
    tensors: std::collections::HashMap<String, std::sync::Arc<bigtea_ggml::Repacked>>,
    bytes: usize,
    declined: usize,
}

impl RepackedDense {
    /// Rearrange every resident tensor the allow-list permits.
    ///
    /// Takes each one out of `resident` as it goes, so the two copies never
    /// coexist beyond the tensor being converted. `BIGTEA_NO_REPACK` returns an
    /// empty set, which is the same switch the dense path reads.
    pub fn build(resident: &mut ResidentSet, model: &Model) -> Result<Self> {
        let mut tensors = std::collections::HashMap::new();
        let mut bytes = 0usize;
        let mut declined = 0usize;
        if std::env::var("BIGTEA_NO_REPACK").is_ok() {
            return Ok(RepackedDense {
                tensors,
                bytes,
                declined,
            });
        }
        // Only what is actually in RAM is a candidate, so the set walks itself.
        let names = resident.names();
        for name in &names {
            if !is_repackable_dense(name) {
                continue;
            }
            let Some(loc) = model.location(name) else {
                continue;
            };
            let (ty, dims) = (loc.ty, loc.dims.clone());
            let (ne0, ne1) = match dims.as_slice() {
                [a] => (*a as i64, 1i64),
                [a, b] => (*a as i64, *b as i64),
                // Nothing in the allow-list is higher rank, and guessing how to
                // flatten one would be exactly the silent mistake the list is
                // here to prevent.
                _ => continue,
            };
            if !bigtea_ggml::is_repackable(ty, ne0, ne1) {
                continue;
            }
            // Only if it is already in RAM — see the type's docs.
            let Some(original) = resident.take(name) else {
                continue;
            };
            match bigtea_ggml::Repacked::new(ty, ne0, ne1, &original[..]) {
                Ok(Some(repacked)) => {
                    bytes += repacked.bytes();
                    tensors.insert(name.clone(), std::sync::Arc::new(repacked));
                    // `original` is dropped here: the rearranged bytes replace
                    // it, and holding both is what this design exists to avoid.
                    drop(original);
                }
                // ggml declined after the shape check said it might not. Put it
                // back rather than leaving it to stream from disk every token.
                Ok(None) => {
                    declined += 1;
                    resident.put_back(name.clone(), original);
                }
                Err(e) => {
                    resident.put_back(name.clone(), original);
                    return Err(e.into());
                }
            }
        }
        Ok(RepackedDense {
            tensors,
            bytes,
            declined,
        })
    }

    fn get(&self, name: &str) -> Option<std::sync::Arc<bigtea_ggml::Repacked>> {
        self.tensors.get(name).cloned()
    }

    /// How many tensors were rearranged, the bytes they hold, and how many
    /// `ggml` declined after the shape check said it might not.
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.tensors.len(), self.bytes, self.declined)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// One block's forward pass, and the state it threads.
pub struct Deepseek4Forward<'m> {
    model: &'m Model,
    config: Deepseek4Config,
    /// Always-read weights held in RAM. `None` re-reads them per block, which
    /// is correct but costs 23% of a prefill and would cost it again on every
    /// generated token.
    resident: Option<&'m ResidentSet>,
    /// Always-read weights already rearranged into the CPU kernels' layout.
    /// Consulted **before** `resident`, because a repacked tensor was taken out
    /// of that set and only exists here.
    repacked: Option<&'m RepackedDense>,
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
            repacked: None,
            cache: None,
        }
    }

    /// Serve always-read weights from `resident` instead of from disk.
    pub fn with_resident(mut self, resident: &'m ResidentSet) -> Self {
        self.resident = Some(resident);
        self
    }

    /// Serve the tensors in `repacked` from their rearranged copies.
    ///
    /// Must be the same [`RepackedDense`] that was built from this forward's
    /// `ResidentSet`: `build` **removes** what it rearranges, so without this
    /// those tensors would fall back to streaming from disk on every block of
    /// every token — correct, and far slower than not repacking at all.
    pub fn with_repacked(mut self, repacked: &'m RepackedDense) -> Self {
        self.repacked = Some(repacked);
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
        // The `min(256)` this used to carry was the old fixed cache bound: the
        // compressed half could never hold more than 256 blocks, so it could
        // never exceed the indexer's budget either. Now that it grows, the
        // block count is the real quantity — and dropping the clamp is what
        // makes this report `false` at the length it actually stops being
        // exact instead of claiming exactness forever.
        blocks <= self.config.indexer_top_k as i64
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

/// The inclusive span of **absolute positions** one pass must be able to read.
///
/// `hi` is the last query. `lo` is how far the *earliest* query in the batch
/// still reaches — `window - 1` positions behind `pos0`, not behind `hi`, which
/// is the off-by-a-batch that would silently drop the keys the first rows of a
/// prefill need.
///
/// With no sliding window every position stays visible, so the span is the whole
/// sequence and a ring cannot serve it; [`forward`] refuses that case rather
/// than returning a span it cannot fill.
fn raw_span(pos0: i64, nt: i64, window: i64) -> (i64, i64) {
    let hi = pos0 + nt - 1;
    let lo = if window > 0 {
        (pos0 - window + 1).max(0)
    } else {
        0
    };
    (lo, hi)
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
    // One row per token, at `position % RAW_RING`. A batch that straddles the
    // wrap writes two runs, which is why this is a loop and not one copy.
    for i in 0..nt {
        let slot = ((pos0 + i) % RAW_RING) as usize;
        let src = (i * head) as usize;
        let dst = slot * head as usize;
        bigtea_ggml::f32_to_f16(
            &kv_vals[src..src + head as usize],
            &mut raw[dst..dst + head as usize],
        );
    }

    // The span of absolute positions this pass can actually read: the earliest
    // query reaches `window - 1` back, the latest is `pos0 + nt - 1`.
    //
    // **This is the key axis now** — a contiguous run of absolute positions
    // gathered out of the ring, not the ring's own slot order. Handing the mask
    // slot indices instead would attend to whatever happened to be `p % 1024`,
    // which is a wrong answer and not an error.
    let window = config.sliding_window as i64;
    let (lo, hi) = raw_span(pos0, nt, window);
    let n_raw = hi - lo + 1;

    let mut packed: Vec<u16> = Vec::with_capacity((n_raw * head) as usize);
    for p in lo..=hi {
        let slot = (p % RAW_RING) as usize;
        let at = slot * head as usize;
        packed.extend_from_slice(&raw[at..at + head as usize]);
    }

    if let Some(c) = comp {
        // The compressor returns only the blocks **this batch completed**, so
        // they append at their absolute index. Writing them from block 0 —
        // which was right while every pass started at position 0 — would make
        // a step overwrite the sequence's history with its own single block.
        let cv = c.to_vec_f32();
        let store = &mut cache.comp[il as usize];
        let at = (comp_block0 * head) as usize;
        // Grown here rather than sized up front: the compressed half cannot be
        // a ring — every complete block behind a token stays visible — so its
        // length is whatever the sequence has reached.
        if store.len() < at + cv.len() {
            store.resize(at + cv.len(), 0);
        }
        bigtea_ggml::f32_to_f16(&cv, &mut store[at..at + cv.len()]);
        cache.comp_len[il as usize] = comp_block0 + cv.len() as i64 / head;
    }

    // The compressed half is present whenever the **sequence** has summaries, not
    // only when this batch produced some. Three steps in four complete no block,
    // and attending over the raw window alone on those would discard everything
    // the sequence had already compressed — silently, and only on the cached path.
    let has_comp = cache.comp_len[il as usize] > 0;
    // Only the blocks the sequence has actually produced. This used to be the
    // whole 256-slot array, so the mask spent its time masking off zeros; now
    // the tensor is the size of the history, which is also what lets it grow.
    let n_comp = if has_comp {
        let blocks = cache.comp_len[il as usize];
        packed.extend_from_slice(&cache.comp[il as usize][..(blocks * head) as usize]);
        blocks
    } else {
        0
    };
    let n_kv = n_raw + n_comp;
    let k = ctx.new_f16_3d(head, n_kv, 1)?;
    let bytes: Vec<u8> = packed.iter().flat_map(|h| h.to_le_bytes()).collect();
    k.set_bytes(&bytes)?;

    let ratio = config.compress_block(il).unwrap_or(1);
    let mut mask = vec![0u8; (n_kv * nt) as usize * 2];
    for query in 0..nt {
        // The key axis is absolute positions `lo..=hi`, so the query must be
        // absolute too — otherwise a step at position 40 would mask everything
        // before it.
        let q_abs = pos0 + query;
        let row = (query * n_kv) as usize * 2;
        for j in 0..n_raw {
            // `j` is an offset into the gathered span, not a ring slot and not
            // a bare position. `lo + j` is the position it holds.
            let key = lo + j;
            if key > q_abs || (window > 0 && q_abs - key >= window) {
                let at = row + j as usize * 2;
                mask[at..at + 2].copy_from_slice(&F16_NEG_INF);
            }
        }
        if has_comp {
            for blk in ((q_abs + 1) / ratio)..n_comp {
                let at = row + (n_raw + blk) as usize * 2;
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
    //
    // Only the *foreground* share of the pool: with R2 overlap on, the last
    // `prefetch_readers()` handles belong to the thread reading the next block's
    // dense weights, and taking one of those back would put two threads on one
    // handle — which does not fail, it just serialises them.
    let readers = foreground_readers();
    let mut slots: Vec<Vec<(&Job, &mut [u8])>> = (0..readers).map(|_| Vec::new()).collect();
    let mut cursors: Vec<&mut [u8]> = buffers.iter_mut().map(|(b, _)| &mut b[..]).collect();
    for (j, job) in jobs.iter().enumerate() {
        let cursor = std::mem::take(&mut cursors[job.name]);
        let (dst, rest) = cursor.split_at_mut(job.len);
        cursors[job.name] = rest;
        slots[j % readers].push((job, dst));
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
    // **Do not parallelise across experts here.** `stream.rs` runs N whole
    // experts side by side with one ggml thread each and gains 1.29x on expert
    // compute; on this path the same change cannot pay, and the ceiling was
    // measured rather than argued. A throwaway build that kept the disk read
    // and dropped the three `mul_mat_id` calls below ran generation at 0.388
    // against 0.370 tok/s, and moved the block's `compute` phase by 0.01s of
    // 0.44 -- so **the entire routed expert arithmetic is under 5% of a
    // V4-Flash token**, against 67% for the slice read. There is also nothing
    // to gather: `read_expert_slices` packs the selected slices contiguously as
    // it reads them, so this path already has the batched form that cost
    // ~1.02 GB/token on the Qwen3 path. See
    // `research/parallel-experts-do-not-transfer-2026-08-16.md`.
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
    // Repacked first. `RepackedDense::build` takes what it rearranges out of
    // the resident set, so for those tensors this is the only place the bytes
    // exist — checking residency first would find nothing and stream from disk.
    if let Some(repacked) = fw.repacked.and_then(|r| r.get(name)) {
        weights.bind_repacked_shared(wctx, name, repacked)?;
        return Ok(0);
    }
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
    prefetch_dense_via(fw, names, 0, foreground_readers())
}

/// Is any of a block's always-read weights going to be read from disk?
///
/// Probed on **block 1** rather than by asking the resident set globally: what
/// matters is whether a *block* has missing tensors, and residency is filled
/// largest-first over the whole model, so a global "something was skipped" can
/// be true while every block this pass touches is fully resident. Block 0 is
/// the wrong probe because it also carries `token_embd`.
fn dense_shortfall(fw: &Deepseek4Forward<'_>) -> bool {
    if fw.config.n_layer < 2 {
        return false;
    }
    fw.block_tensor_names(1).iter().any(|n| {
        fw.repacked.and_then(|r| r.get(n)).is_none()
            && fw.resident.and_then(|r| r.get_shared(n)).is_none()
    })
}

/// [`prefetch_dense`], over reader slots `base..base + count`.
///
/// The slot range is a parameter because the foreground and the R2 background
/// prefetch must use **disjoint** handles — see [`prefetch_readers`].
fn prefetch_dense_via(
    fw: &Deepseek4Forward<'_>,
    names: &[String],
    base: usize,
    count: usize,
) -> Result<std::collections::HashMap<String, std::sync::Arc<SkewedBuf>>> {
    let count = count.max(1);
    let missing: Vec<&String> = names
        .iter()
        .filter(|n| {
            // A repacked tensor is in RAM too — it is simply held elsewhere,
            // having been taken out of the resident set. Without this it would
            // look absent and be re-read from disk on every block of every
            // token, which is the exact cost repacking was meant to avoid.
            fw.repacked.and_then(|r| r.get(n)).is_none()
                && fw.resident.and_then(|r| r.get_shared(n)).is_none()
        })
        .collect();
    if missing.len() < 2 {
        // One tensor has nothing to overlap with, and the common case — a fully
        // resident set — has none at all.
        return Ok(std::collections::HashMap::new());
    }

    let model = fw.model;
    let chunks: Vec<Vec<&String>> = (0..count)
        .map(|s| {
            missing
                .iter()
                .skip(s)
                .step_by(count)
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    let out = std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .into_iter()
            .enumerate()
            .map(|(i, work)| {
                let slot = base + i;
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
// Eight, and the alternative is worse. Six of them are what a block *is* —
// which layer, which tokens, where in the sequence, the arena, the streams in —
// and bundling them into a struct would add a type whose only job is to be
// unpacked immediately, on the one call site that exists.
#[allow(clippy::too_many_arguments)]
pub fn block(
    fw: &Deepseek4Forward<'_>,
    cache: &mut Deepseek4Cache,
    il: u32,
    tokens: &[i32],
    pos0: i64,
    streams_in: Option<&[f32]>,
    arena: usize,
    prefetched: Option<std::collections::HashMap<String, std::sync::Arc<SkewedBuf>>>,
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
    // Already read, while the previous block computed. `bind_dense` falls back
    // to reading anything the map is missing, so a partial or absent map is
    // slower and never wrong.
    let prefetched = match prefetched {
        Some(p) => p,
        None => prefetch_dense(fw, &names)?,
    };
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
    // The block builds one graph and evaluates it here, so every phase timer
    // above measures graph *construction* (plus, in `ffn`, the disk read). This
    // is the only line where arithmetic actually happens, and leaving it inside
    // the residual hid the fact that a V4-Flash token is 55% disk and 29% this.
    let t_phase = std::time::Instant::now();
    ctx.compute(&out, threads())?;
    let compute_secs = t_phase.elapsed().as_secs_f64();

    if std::env::var("BIGTEA_BLOCK_TIMING").is_ok() {
        eprintln!(
            "  block {il:>2}  arena {arena_secs:.2}  dense {dense_secs:.2} ({:.0} MiB)               qkv {qkv_secs:.2}  attn {attn_secs:.2}  tail {tail_secs:.2}  ffn {ffn_secs:.2}  compute {compute_secs:.2}               total {:.2}",
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
    // No sequence limit any more — the raw latents live in a ring and the
    // compressed half grows. What is still bounded is **one pass**, and
    // `forward` enforces that, so a prompt longer than a batch is chunked
    // rather than refused.
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
    // Every `threads()` call below this point reads it. See `threads`: a
    // one-token step and a prefill block want opposite counts, and this is the
    // single funnel both `prefill` and `step` pass through.
    BATCH.store(tokens.len(), std::sync::atomic::Ordering::Relaxed);
    let pos0 = cache.n_past as i64;

    // The limit is now on **one pass**, not on the sequence.
    //
    // A pass needs `window + nt - 1` distinct positions live at once: its
    // earliest query still reaches `window - 1` behind itself. Past that the
    // ring would wrap over rows this same pass is going to read — which is not
    // an error, it is attention over whatever those slots held before.
    //
    // With `sliding_window = 0` there is no window at all and raw attention is
    // full causal, so every position stays visible and a ring cannot hold them.
    // V4-Flash declares 128; anything else is refused rather than silently
    // dropping keys that are still in scope.
    let window = fw.config.sliding_window as i64;
    let (span, limit) = if window > 0 {
        // Reported as a *batch* limit, because that is what the caller can act
        // on: chunk the prompt. `span` is an internal quantity nobody can do
        // anything about.
        (
            window + tokens.len() as i64 - 1,
            (RAW_RING - window + 1).max(1),
        )
    } else {
        (pos0 + tokens.len() as i64, RAW_RING)
    };
    if span > RAW_RING {
        return Err(crate::ArchError::ContextTooLong {
            tokens: tokens.len(),
            limit: limit as usize,
        });
    }
    // R2: read block N+1's always-read weights while block N computes.
    //
    // Routing is data-dependent — block N+1's *experts* are chosen from block
    // N's output, so they cannot be known yet — but its **dense** tensors do not
    // depend on routing at all. That makes this exact rather than speculative:
    // a prefetch is never wrong, only wasted.
    //
    // It pays exactly when residency is short. `prefetch_dense` skips anything
    // already resident and returns an empty map when fewer than two tensors are
    // missing, so with the always-read set fully in RAM this costs one thread
    // spawn per block and reads nothing. **Quote no number from it without the
    // free RAM**; that is the axis these figures move along.
    let n_layer = fw.config.n_layer;

    // Decide once, because it decides how the reader pool is split for the
    // whole pass. **With no shortfall there is nothing to prefetch**, and
    // taking handles away from the expert reads to feed a thread that reads
    // nothing is a pure loss — so in that case the overlap is not merely
    // idle, it is off.
    let overlap = prefetch_overlap() && dense_shortfall(fw);
    PREFETCHING.store(overlap, std::sync::atomic::Ordering::Relaxed);
    // Restores the flag however this function leaves, including on `?`.
    struct ClearOnDrop;
    impl Drop for ClearOnDrop {
        fn drop(&mut self) {
            PREFETCHING.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let _clear = ClearOnDrop;

    let mut streams: Option<Streams> = None;
    let mut prefetched: Option<std::collections::HashMap<String, std::sync::Arc<SkewedBuf>>> = None;

    for il in 0..n_layer {
        let next: Option<Vec<String>> = if overlap && il + 1 < n_layer {
            Some(fw.block_tensor_names(il + 1))
        } else {
            None
        };

        let (out, ahead) = std::thread::scope(|scope| {
            let handle = next.as_ref().map(|names| {
                scope.spawn(move || {
                    // The tail of the pool, disjoint from what the expert reads
                    // in `block` are using on this thread.
                    prefetch_dense_via(fw, names, foreground_readers(), prefetch_readers())
                })
            });
            let out = block(
                fw,
                cache,
                il,
                tokens,
                pos0,
                streams.as_deref(),
                arena,
                prefetched.take(),
            );
            let ahead = match handle.map(|h| h.join()) {
                None => None,
                Some(Ok(Ok(map))) => Some(map),
                // An I/O error in the prefetch is not an error in the pass:
                // `block` reads what it needs. Losing the optimisation is the
                // right failure, and it must not be able to end a run.
                Some(Ok(Err(_))) => None,
                // A panic is a bug, not a slow disk. Do not swallow it.
                Some(Err(payload)) => std::panic::resume_unwind(payload),
            };
            (out, ahead)
        });
        streams = Some(out?);
        prefetched = ahead;
    }

    let streams = streams.expect("at least one block");
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
    use super::{is_repackable_dense, pool_passes, raw_span, record_into, RAW_RING};

    /// V4-Flash's declared window, from the container:
    /// `deepseek4.attention.sliding_window = 128`.
    const WINDOW: i64 = 128;

    /// A prefill from zero reaches no further back than zero.
    #[test]
    fn the_span_of_a_prefill_starts_at_the_beginning() {
        assert_eq!(raw_span(0, 1, WINDOW), (0, 0));
        assert_eq!(raw_span(0, 5, WINDOW), (0, 4));
        assert_eq!(raw_span(0, 165, WINDOW), (0, 164));
    }

    /// **The off-by-a-batch this is here to prevent.** The earliest query in the
    /// batch reaches `window - 1` behind `pos0`, not behind `hi`. Measuring from
    /// `hi` would drop exactly the keys the first rows of a prefill need — and
    /// attention over a short key set is fluent nonsense, not an error.
    #[test]
    fn the_span_reaches_back_from_the_first_query_not_the_last() {
        let (lo, hi) = raw_span(300, 64, WINDOW);
        assert_eq!(lo, 300 - WINDOW + 1, "measured from pos0");
        assert_eq!(hi, 363);
        // 128 for the window, plus the 63 later queries in the batch.
        assert_eq!(hi - lo + 1, WINDOW + 63);
    }

    /// A single cached step needs exactly one window.
    #[test]
    fn a_step_needs_exactly_the_window() {
        let (lo, hi) = raw_span(2000, 1, WINDOW);
        assert_eq!(hi - lo + 1, WINDOW);
        assert_eq!((lo, hi), (1873, 2000));
    }

    /// Every position in a span maps to a distinct ring slot — which is the
    /// whole safety argument for wraparound. If two positions in one span
    /// collided, one would silently read the other's latents.
    #[test]
    fn no_two_positions_in_one_span_share_a_ring_slot() {
        for &(pos0, nt) in &[(0, 1), (0, 897), (1100, 1), (5000, 400), (100_000, 64)] {
            let (lo, hi) = raw_span(pos0, nt, WINDOW);
            assert!(
                hi - lo < RAW_RING,
                "span {} exceeds the ring at pos0 {pos0}, nt {nt}",
                hi - lo + 1
            );
            let mut seen = std::collections::HashSet::new();
            for p in lo..=hi {
                assert!(
                    seen.insert(p % RAW_RING),
                    "positions collide in the ring at pos0 {pos0}, nt {nt}"
                );
            }
        }
    }

    /// The span is allowed to wrap, and must stay contiguous *in position
    /// order* when it does — the gather walks `lo..=hi`, not slot order.
    #[test]
    fn a_span_that_wraps_the_ring_is_still_read_in_position_order() {
        let (lo, hi) = raw_span(1100, 1, WINDOW);
        assert_eq!((lo, hi), (973, 1100));
        let slots: Vec<i64> = (lo..=hi).map(|p| p % RAW_RING).collect();
        assert_eq!(slots[0], 973);
        assert_eq!(*slots.last().expect("non-empty"), 1100 % RAW_RING);
        assert!(
            slots.windows(2).any(|w| w[1] < w[0]),
            "this case is chosen because it wraps; if it stopped wrapping the \
             test no longer covers wraparound"
        );
    }

    /// The largest batch the ring can serve, and one past it. `forward` refuses
    /// beyond this rather than wrapping over rows the same pass will read.
    #[test]
    fn the_batch_limit_is_the_ring_minus_the_window() {
        let max_batch = RAW_RING - WINDOW + 1;
        let (lo, hi) = raw_span(0, max_batch, WINDOW);
        assert!(hi - lo < RAW_RING);
        let (lo, hi) = raw_span(10_000, max_batch, WINDOW);
        assert_eq!(hi - lo + 1, RAW_RING, "exactly fills the ring");
        let (lo, hi) = raw_span(10_000, max_batch + 1, WINDOW);
        assert!(hi - lo + 1 > RAW_RING, "one past it must not fit");
    }

    /// With no sliding window the span is the whole sequence, which is what
    /// makes a ring unusable and why that configuration is refused.
    #[test]
    fn without_a_window_the_span_is_the_whole_sequence() {
        assert_eq!(raw_span(5000, 1, 0), (0, 5000));
    }

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

    /// The four uses that read repacked bytes **by position**, and produce
    /// confident nonsense rather than an error when they get them. This is the
    /// list the allow-list exists to enforce; a regression here is not a failing
    /// forward pass, it is a fluent wrong answer.
    #[test]
    fn tensors_read_by_position_are_never_repackable() {
        for name in [
            // `get_rows` — a repacked tensor's rows are interleaved, so row 5
            // is not where row 5 was.
            "token_embd.weight",
            "blk.0.attn_compressor_ape.weight",
            "blk.12.ffn_gate_tid2eid.weight",
            // `view_1d` at a byte offset, into the unpacked layout.
            "blk.3.hc_attn_scale.weight",
            "blk.3.hc_attn_base.weight",
            "blk.3.hc_ffn_scale.weight",
            "blk.3.hc_ffn_base.weight",
            "output_hc_scale.weight",
            "output_hc_base.weight",
            // `reshape_3d` into a grouped `mul_mat`, which cuts the matrix in
            // places the interleave has moved.
            "blk.7.attn_output_a.weight",
            // The sinks argument of `flash_attn_ext`, not a matmul weight.
            "blk.7.attn_sinks.weight",
        ] {
            assert!(!is_repackable_dense(name), "{name} must never be repacked");
        }
    }

    /// The routed experts are the exclusion that matters most: they stream from
    /// disk zero-copy, one slice at a time, and that is what lets a 144 GB model
    /// run on a 15.7 GiB machine. Repacking them would need the whole bank in
    /// RAM — the exact thing this engine exists to avoid.
    #[test]
    fn routed_experts_are_never_repackable() {
        for name in [
            "blk.5.ffn_gate_exps.weight",
            "blk.5.ffn_up_exps.weight",
            "blk.5.ffn_down_exps.weight",
        ] {
            assert!(
                !is_repackable_dense(name),
                "{name} streams; never repack it"
            );
        }
    }

    /// The shared expert is *not* routed — it runs for every token and is part
    /// of the always-read set, so it is both repackable and one of the largest
    /// wins available. The names differ by three characters (`shexp` against
    /// `exps`), which is exactly the kind of near-miss a substring rule gets
    /// wrong in one direction or the other.
    #[test]
    fn shared_experts_are_repackable_but_routed_ones_are_not() {
        assert!(is_repackable_dense("blk.5.ffn_gate_shexp.weight"));
        assert!(is_repackable_dense("blk.5.ffn_up_shexp.weight"));
        assert!(is_repackable_dense("blk.5.ffn_down_shexp.weight"));
        assert!(!is_repackable_dense("blk.5.ffn_gate_exps.weight"));
    }

    #[test]
    fn plain_matmul_weights_are_repackable() {
        for name in [
            "blk.0.attn_q_a.weight",
            "blk.41.attn_q_b.weight",
            "blk.9.attn_kv.weight",
            "blk.9.attn_output_b.weight",
            "blk.9.ffn_gate_inp.weight",
            "blk.9.attn_compressor_kv.weight",
            "blk.9.attn_compressor_gate.weight",
            "blk.9.hc_attn_fn.weight",
            "blk.9.hc_ffn_fn.weight",
            "output_hc_fn.weight",
            "output.weight",
        ] {
            assert!(
                is_repackable_dense(name),
                "{name} is a plain mul_mat weight"
            );
        }
    }

    /// The suffix is matched whole, not as a prefix. `attn_kv` is repackable and
    /// `attn_kv_a_norm` is not, and a `starts_with` rule would take both — the
    /// norm is F32 so `is_repackable` would decline it today, but that is luck
    /// rather than intent and would stop holding on a quantised norm.
    #[test]
    fn a_longer_name_that_starts_with_an_allowed_one_is_not_allowed() {
        assert!(is_repackable_dense("blk.2.attn_kv.weight"));
        assert!(!is_repackable_dense("blk.2.attn_kv_a_norm.weight"));
        assert!(!is_repackable_dense("blk.2.attn_q_a_norm.weight"));
        assert!(!is_repackable_dense("blk.2.attn_norm.weight"));
        assert!(!is_repackable_dense("blk.2.ffn_norm.weight"));
        assert!(!is_repackable_dense("blk.2.attn_compressor_norm.weight"));
    }

    /// Names that are not `blk.<digits>.<suffix>` or a known global must not be
    /// coaxed into matching — a bias is not a weight, and a non-numeric segment
    /// is not a block.
    #[test]
    fn malformed_and_unknown_names_are_refused() {
        assert!(!is_repackable_dense("blk.3.exp_probs_b.bias"));
        assert!(!is_repackable_dense("attn_kv.weight.extra"));
        assert!(!is_repackable_dense("blk.x.attn_kv.weight"));
        assert!(!is_repackable_dense("blk..attn_kv.weight"));
        assert!(!is_repackable_dense("blk.attn_kv.weight"));
        assert!(!is_repackable_dense("attn_kv"));
        assert!(!is_repackable_dense(""));
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
