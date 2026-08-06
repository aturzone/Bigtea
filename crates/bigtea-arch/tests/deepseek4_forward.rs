//! Build the V4-Flash forward pass against llama.cpp's numbers, one step at a
//! time.
//!
//! Every assertion here compares against a fixture captured from
//! `llama-eval-callback` on the real container. There are two, and the second
//! exists because the first has a hole:
//!
//! * `v4flash-layer0-oracle.txt` — prompt `"Hi"`, the single id 23166, no BOS.
//! * `v4flash-layer0-oracle-5tok.txt` — `"The capital of France is"`, five ids,
//!   no BOS, covering all of layer 0 through the MoE and shared expert.
//!
//! **The one-token trace cannot validate RoPE.** At position 0 the rotation is
//! the identity, so `q_pe` there has the same sum as its input and an
//! implementation that skipped the rotation entirely would pass. That is why
//! the five-token capture exists, and why the tests that check the rotation use
//! it. Both are kept: matching two independent inputs beats matching one.
//!
//! The element-sum is the check. It is unforgiving in the way this
//! architecture needs: a transposed matrix, an unrotated RoPE, a norm applied
//! with a weight it should not have — all change the sum, and none of them
//! raise an error or make the text obviously wrong. Building forward against
//! "looks like English" is how you get a model that is subtly broken on half
//! its layers.
//!
//! Ignored by default: these read weights out of a 144 GB container. Run with
//! `cargo test --release --test deepseek4_forward -- --ignored --nocapture`

use bigtea_arch::{Deepseek4Config, Deepseek4Model};
use bigtea_ggml::{Context, RopeParams, Tensor, WeightSet};
use bigtea_model::Model;

const SHARD: &str =
    r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf";

/// The prompt "Hi" as llama.cpp tokenized it: one token, no BOS.
const TOKEN: i32 = 23166;

/// "The capital of France is" — five tokens, no BOS, from the second capture.
///
/// The point of a longer prompt is positions 1..4: at position 0 alone RoPE is
/// the identity and cannot be checked at all.
const TOKENS_5: &[i32] = &[671, 6102, 294, 8760, 344];

/// "Hello there" — two tokens, from a second capture taken deliberately short.
///
/// **At two tokens every layer takes the Raw attention path.** The compressed
/// builders are guarded on their compressed caches being populated
/// (`deepseek4.cpp:1049-1063`), and at this length they are not, so layers 2+
/// fall through to `build_raw_attention` — code that is already built and
/// verified. That makes two things reachable that five tokens could not reach
/// without the indexer: the **compressed RoPE branch** (layers 2+ use it
/// regardless of which attention runs) and the **normal MoE routing path**
/// (layers 3+). Both were open holes.
const TOKENS_2: &[i32] = &[19923, 1031];

/// `LLAMA_ROPE_TYPE_NORM`. deepseek4 is in the NORM list in `llama-model.cpp`,
/// **not** the NEOX one — so rotated pairs are adjacent (`x[2i]`, `x[2i+1]`),
/// not offset by `n_rot/2`. Both conventions run, and one of them is wrong.
const ROPE_MODE_NORM: i32 = 0;

const SUMS_2TOK: &str = "tests/fixtures/v4flash-sums-2tok.txt";
const SUMS_5TOK: &str = "tests/fixtures/v4flash-sums-5tok.txt";

fn sums_2tok(il: u32) -> LayerSums {
    LayerSums::load(SUMS_2TOK, il, TOKENS_2)
}

fn sums_5tok(il: u32) -> LayerSums {
    LayerSums::load(SUMS_5TOK, il, TOKENS_5)
}

/// RoPE for `il`, from the shipped [`Deepseek4Config::rope_for_layer`].
///
/// Deliberately not a local copy of the rules. `rope_for_layer` is what a real
/// forward pass calls, so it is what these checkpoints have to exercise — a
/// helper written out again here would verify the test and not the code.
///
/// **This started out hardcoded to layer 0 and that was a real bug**: layers 2
/// and 3 take the compressed branch (base 160000, YaRN on, `n_ctx_orig` 65536)
/// and were being given layer 0's plain parameters with `n_ctx_orig` 0. It
/// survived because until layer 2 ran, every layer under test *was* layer 0's
/// branch. `rope_for_layer` itself was correct.
fn rope_for(config: &Deepseek4Config, il: u32) -> (RopeParams, i32) {
    let r = config.rope_for_layer(il);
    (r.params, r.n_ctx_orig)
}

/// Every checkpoint for one layer, keyed by a layer-agnostic label.
///
/// The forward helpers are written once and run for any layer; what changes is
/// this table and the block index. The numbers are **data in a fixture**, not
/// literals in the code, so a further layer costs a capture and nothing else —
/// which is what makes running all 43 of them feasible at all.
struct LayerSums {
    il: u32,
    /// The prompt these numbers were captured at. Two captures are in play and
    /// mixing them would compare a layer against the wrong run.
    tokens: &'static [i32],
    attn_gates: HcGateSums,
    ffn_gates: HcGateSums,
    /// Each routed expert's weighted contribution, checked individually so a
    /// mis-slotted expert cannot hide inside the total.
    weighted: Vec<f32>,
    rows: Vec<(String, f32)>,
}

impl LayerSums {
    /// Read one layer's rows out of a `layer|label|sum` fixture.
    fn load(path: &str, il: u32, tokens: &'static [i32]) -> LayerSums {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        let mut rows = Vec::new();
        let mut weighted = Vec::new();
        let mut attn = std::collections::HashMap::new();
        let mut ffn = std::collections::HashMap::new();
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let mut it = line.splitn(3, '|');
            let (Some(l), Some(label), Some(v)) = (it.next(), it.next(), it.next()) else {
                panic!("{path}: malformed row {line:?} (want layer|label|sum)");
            };
            if l.parse::<u32>().ok() != Some(il) {
                continue;
            }
            let v: f32 = v.trim().parse().expect("sum parses");
            if let Some(k) = label.strip_prefix("attn_gates.") {
                attn.insert(k.to_string(), v);
            } else if let Some(k) = label.strip_prefix("ffn_gates.") {
                ffn.insert(k.to_string(), v);
            } else if label.starts_with("weighted.") {
                weighted.push(v);
            } else {
                rows.push((label.to_string(), v));
            }
        }
        assert!(!rows.is_empty(), "{path} has no rows for layer {il}");
        LayerSums {
            il,
            tokens,
            attn_gates: HcGateSums::from_map(&attn),
            ffn_gates: HcGateSums::from_map(&ffn),
            weighted,
            rows,
        }
    }

    /// Panics on an unknown label rather than skipping the check. A typo that
    /// silently verified nothing would be worse than a failing test.
    fn get(&self, label: &str) -> f32 {
        self.try_get(label).unwrap_or_else(|| {
            panic!("no oracle row labelled {label:?} for layer {}", self.il)
        })
    }

    fn try_get(&self, label: &str) -> Option<f32> {
        self.rows.iter().find(|(k, _)| k == label).map(|(_, v)| *v)
    }
}

/// Assert one checkpoint against the layer's table.
fn check(s: &LayerSums, label: &str, got: f32) {
    assert_sum(&format!("{label}-{}", s.il), got, s.get(label));
}

/// For a checkpoint the *last* block does not have: `next_norm` is the norm of
/// the streams on the way into the next layer, and layer 42 has no next layer —
/// its final norm is `output_norm`, outside the block.
fn check_opt(s: &LayerSums, label: &str, got: f32) {
    if let Some(want) = s.try_get(label) {
        assert_sum(&format!("{label}-{}", s.il), got, want);
    }
}

/// Sums are accumulated over thousands of floats in a different order than
/// ggml uses, so exact equality is not the right bar. A wrong graph is off by
/// percent or more, never by 1e-3.
fn assert_sum(label: &str, got: f32, want: f32) {
    let tol = want.abs() * 1e-4 + 1e-3;
    assert!(
        (got - want).abs() <= tol,
        "{label}: got {got:.6}, llama.cpp got {want:.6} (tolerance {tol:.6})"
    );
    eprintln!("  {label:<24} {got:>14.6}  matches llama.cpp");
}

fn open() -> Option<Model> {
    if !std::path::Path::new(SHARD).exists() {
        eprintln!("skipping: {SHARD} not present");
        return None;
    }
    Model::open_split(SHARD).ok()
}

/// Bind a set of tensors by name into `weights`.
fn bind_all<'c>(
    model: &Model,
    ctx: &'c Context,
    weights: &mut WeightSet<'c>,
    names: &[String],
) {
    for name in names {
        let name = name.as_str();
        let loc = model
            .location(name)
            .unwrap_or_else(|| panic!("{name} present"))
            .clone();
        let data = model.read_tensor(name).expect("read tensor");
        weights.bind(ctx, name, loc.ty, &loc.dims, data).expect("bind");
    }
}

/// The prologue: embedding lookup, and the hyper-connection stream it seeds.
///
/// Oracle rows:
/// ```text
/// embd     GET_ROWS {4096, 1}      2.097937
/// hc_init  REPEAT   {4096, 4}      8.391747
/// node_4   RMS_NORM {16384, 1}    92.071121
/// ```
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn prologue_matches_llama_cpp() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");
    let _arch = Deepseek4Model::new(config.clone());

    let ctx = Context::new(64 << 20).expect("compute context");
    let mut weights = WeightSet::new();

    // Only the embedding table is needed for this step; loading the whole
    // resident set would cost 7.38 GiB to check one number.
    let loc = model
        .location("token_embd.weight")
        .expect("token_embd present")
        .clone();
    let data = model.read_tensor("token_embd.weight").expect("read embd");
    let embd_ctx = Context::new_no_alloc(4 << 20).expect("weight context");
    weights
        .bind(&embd_ctx, "token_embd.weight", loc.ty, &loc.dims, data)
        .expect("bind embd");

    let tok = ctx.new_i32_1d(1).expect("tok");
    tok.set_i32(&[TOKEN]).expect("set tok");
    let embd = ctx
        .get_rows(weights.get("token_embd.weight").expect("bound"), &tok)
        .expect("get_rows");
    ctx.compute(&embd, 12).expect("compute");

    let values = embd.to_vec_f32();
    assert_eq!(values.len(), config.n_embd as usize, "one row of n_embd");
    let sum: f32 = values.iter().sum();
    assert_sum("embd", sum, 2.097937);

    // hc_init repeats the embedding across the 4 hyper-connection streams, so
    // its sum is exactly four times. Cheap, but it pins hc_mult: getting the
    // stream count wrong would sail through every shape check.
    let hc_init = ctx
        .new_f32_2d(config.n_embd as i64, config.hc_mult as i64)
        .expect("hc_init");
    let mut repeated = Vec::with_capacity(values.len() * config.hc_mult as usize);
    for _ in 0..config.hc_mult {
        repeated.extend_from_slice(&values);
    }
    hc_init.set_f32(&repeated).expect("fill hc_init");
    assert_sum("hc_init", repeated.iter().sum::<f32>(), 8.391747);

    // node_4: the stream vector is RMS-normalised as one 16384-wide row, not
    // per stream. Normalising each stream separately gives a different number
    // and no error.
    let flat = ctx
        .reshape_2d(&hc_init, config.hc_dim() as i64, 1)
        .expect("flatten streams");
    let normed = ctx.rms_norm(&flat, config.rms_eps).expect("rms_norm");
    ctx.compute(&normed, 12).expect("compute norm");
    assert_sum("node_4 (rms_norm)", normed.to_vec_f32().iter().sum::<f32>(), 92.071121);
}

/// The hyper-connection mixing weights, and the collapse back to one vector.
///
/// Oracle rows:
/// ```text
/// hc_mixes-0     MUL_MAT      {24, 1}      -1121.066162
/// node_11        SIGMOID      {4, 1}           4.000000
/// hc_pre-0       SCALE        {4, 1}           4.000004
/// hc_attn_pre-0  DSV4_HC_PRE  {4096, 1}        8.391787
/// ```
///
/// This is the first exercise of `ggml_dsv4_hc_pre`, and the point where a
/// wrong reading of the hyper-connection block would first show up as a
/// number rather than as slightly-off text forty layers later.
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn hyper_connection_block_matches_llama_cpp() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    let ctx = Context::new(64 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(4 << 20).expect("weight context");
    let mut weights = WeightSet::new();

    for name in [
        "token_embd.weight",
        "blk.0.hc_attn_fn.weight",
        "blk.0.hc_attn_scale.weight",
        "blk.0.hc_attn_base.weight",
    ] {
        let loc = model.location(name).unwrap_or_else(|| panic!("{name} present")).clone();
        let data = model.read_tensor(name).expect("read");
        weights.bind(&wctx, name, loc.ty, &loc.dims, data).expect("bind");
    }

    let tok = ctx.new_i32_1d(1).expect("tok");
    tok.set_i32(&[TOKEN]).expect("set");
    let embd = ctx
        .get_rows(weights.get("token_embd.weight").expect("bound"), &tok)
        .expect("get_rows");
    ctx.compute(&embd, 12).expect("compute embd");
    let e = embd.to_vec_f32();

    let hc_init = ctx
        .new_f32_2d(config.n_embd as i64, config.hc_mult as i64)
        .expect("hc_init");
    let mut repeated = Vec::new();
    for _ in 0..config.hc_mult {
        repeated.extend_from_slice(&e);
    }
    hc_init.set_f32(&repeated).expect("fill");

    let flat = ctx.reshape_2d(&hc_init, config.hc_dim() as i64, 1).expect("flat");
    let normed = ctx.rms_norm(&flat, config.rms_eps).expect("norm");
    let mixes = ctx
        .mul_mat(weights.get("blk.0.hc_attn_fn.weight").expect("bound"), &normed)
        .expect("hc_mixes");
    ctx.compute(&mixes, 12).expect("compute mixes");
    assert_sum("hc_mixes", mixes.to_vec_f32().iter().sum::<f32>(), -1121.066162);

    // The 24 mixes are three groups of hc: [0..4] pre, [4..8] post, [8..24]
    // the 4x4 combination matrix. hc_scale is likewise [pre, post, comb] and
    // hc_base is [4 pre, 4 post, 16 comb]. Slicing these wrong is an easy
    // mistake with no shape consequence at all — every view is the right size.
    let hc = config.hc_mult as i64;
    let f32_size = std::mem::size_of::<f32>();
    let scale_pre = ctx
        .view_1d(weights.get("blk.0.hc_attn_scale.weight").expect("bound"), 1, 0)
        .expect("scale_pre");
    let base_pre = ctx
        .view_1d(weights.get("blk.0.hc_attn_base.weight").expect("bound"), hc, 0)
        .expect("base_pre");
    let pre = ctx.view_1d(&mixes, hc, 0).expect("mixes pre");

    // affine: mixes*scale + base, with scale a single broadcast value.
    let scaled = ctx.mul(&pre, &scale_pre).expect("mul scale");
    ctx.compute(&scaled, 12).expect("compute scaled");
    assert_sum("node_8 (mul)", scaled.to_vec_f32().iter().sum::<f32>(), 220.453522);

    let biased = ctx.add(&scaled, &base_pre).expect("add base");
    ctx.compute(&biased, 12).expect("compute biased");
    assert_sum("node_10 (add)", biased.to_vec_f32().iter().sum::<f32>(), 218.101593);

    let gated = ctx.sigmoid(&biased).expect("sigmoid");
    ctx.compute(&gated, 12).expect("compute sigmoid");
    assert_sum("node_11 (sigmoid)", gated.to_vec_f32().iter().sum::<f32>(), 4.000000);

    // scale_bias(pre, 1.0, hc_eps): the epsilon is what turns 4.000000 into
    // 4.000004, so this one number pins hyper_connection.epsilon at 1e-6.
    let eps_t = ctx.new_f32_1d(hc).expect("eps");
    eps_t.set_f32(&vec![1e-6f32; hc as usize]).expect("fill eps");
    let hc_pre_w = ctx.add(&gated, &eps_t).expect("add eps");
    ctx.compute(&hc_pre_w, 12).expect("compute hc_pre");
    assert_sum("hc_pre (scale_bias)", hc_pre_w.to_vec_f32().iter().sum::<f32>(), 4.000004);

    // The fused op itself: collapse [n_embd, hc] against the [hc] weights.
    let _ = f32_size;
    let collapsed = ctx.dsv4_hc_pre(&hc_init, &hc_pre_w).expect("dsv4_hc_pre");
    ctx.compute(&collapsed, 12).expect("compute hc_pre op");
    assert_sum(
        "hc_attn_pre (fused)",
        collapsed.to_vec_f32().iter().sum::<f32>(),
        8.391787,
    );
}

/// The Q path: down to `q_lora_rank`, back up, then the per-head norm.
///
/// Oracle rows:
/// ```text
/// norm-0     RMS_NORM {4096, 1}     23.019047
/// attn_norm  MUL      {4096, 1}      0.769727
/// qr-0       MUL_MAT  {1024, 1}     -1.006525
/// norm-0     RMS_NORM {1024, 1}    -13.669229
/// qr_norm-0  MUL      {1024, 1}     -0.573721
/// node_19    MUL_MAT  {32768, 1}     0.694762
/// q_norm-0   RMS_NORM {512, 64}     48.321102
/// ```
///
/// The last row is the one worth having: `q_norm` is an RMS norm with **no
/// weight**, applied per head across 512 dims. Applying `attn_q_a_norm` again
/// there is the natural-looking mistake and changes the number.
///
/// Note this trace cannot validate RoPE. The prompt is one token at position
/// 0, where the rotation is the identity — the oracle shows `q_pe` with the
/// same sum as its input. Checking RoPE needs a multi-token capture.
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn q_projection_matches_llama_cpp() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    let ctx = Context::new(512 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(8 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    bind_all(
        &model,
        &wctx,
        &mut weights,
        &[
            "token_embd.weight".to_string(),
            "blk.0.attn_norm.weight".to_string(),
            "blk.0.attn_q_a.weight".to_string(),
            "blk.0.attn_q_a_norm.weight".to_string(),
            "blk.0.attn_q_b.weight".to_string(),
        ],
    );

    // At layer 0 every hyper-connection stream is a copy of the embedding, so
    // the collapse is the embedding times the summed gate — 4.000004 — which
    // the previous test established against llama.cpp.
    let tok = ctx.new_i32_1d(1).expect("tok");
    tok.set_i32(&[TOKEN]).expect("set");
    let embd = ctx
        .get_rows(weights.get("token_embd.weight").expect("bound"), &tok)
        .expect("get_rows");
    ctx.compute(&embd, 12).expect("compute embd");
    let hc_out: Vec<f32> = embd.to_vec_f32().iter().map(|v| v * 4.000004).collect();

    let x = ctx.new_f32_2d(config.n_embd as i64, 1).expect("x");
    x.set_f32(&hc_out).expect("set x");
    assert_sum("hc_attn_pre", hc_out.iter().sum::<f32>(), 8.391787);

    let normed = ctx.rms_norm(&x, config.rms_eps).expect("norm");
    ctx.compute(&normed, 12).expect("compute");
    assert_sum("norm-0", normed.to_vec_f32().iter().sum::<f32>(), 23.019047);

    let attn_norm = ctx
        .mul(&normed, weights.get("blk.0.attn_norm.weight").expect("bound"))
        .expect("attn_norm");
    ctx.compute(&attn_norm, 12).expect("compute");
    assert_sum("attn_norm-0", attn_norm.to_vec_f32().iter().sum::<f32>(), 0.769727);

    let qr = ctx
        .mul_mat(weights.get("blk.0.attn_q_a.weight").expect("bound"), &attn_norm)
        .expect("qr");
    ctx.compute(&qr, 12).expect("compute");
    assert_sum("qr-0", qr.to_vec_f32().iter().sum::<f32>(), -1.006525);

    let qr_n = ctx.rms_norm(&qr, config.rms_eps).expect("qr rms");
    ctx.compute(&qr_n, 12).expect("compute");
    assert_sum("norm-0 (qr)", qr_n.to_vec_f32().iter().sum::<f32>(), -13.669229);

    let qr_norm = ctx
        .mul(&qr_n, weights.get("blk.0.attn_q_a_norm.weight").expect("bound"))
        .expect("qr_norm");
    ctx.compute(&qr_norm, 12).expect("compute");
    assert_sum("qr_norm-0", qr_norm.to_vec_f32().iter().sum::<f32>(), -0.573721);

    let q = ctx
        .mul_mat(weights.get("blk.0.attn_q_b.weight").expect("bound"), &qr_norm)
        .expect("q");
    ctx.compute(&q, 12).expect("compute");
    assert_sum("node_19 (q_b)", q.to_vec_f32().iter().sum::<f32>(), 0.694762);

    // Per head, and unweighted: this is the norm that has no learned scale.
    let q3 = ctx
        .reshape_3d(&q, config.kv_lora_rank as i64, config.n_head as i64, 1)
        .expect("reshape q");
    let q_norm = ctx.rms_norm(&q3, config.rms_eps).expect("q_norm");
    ctx.compute(&q_norm, 12).expect("compute");
    assert_sum("q_norm-0", q_norm.to_vec_f32().iter().sum::<f32>(), 48.321102);
}

/// The prologue at five tokens, up to `attn_norm`.
///
/// Shared by the multi-token tests below so the hyper-connection block is
/// exercised for real rather than shortcut. The single-token tests can multiply
/// the embedding by the summed gate because at layer 0 every stream is a copy
/// of it; at five tokens the gate differs per token and that shortcut is not
/// available — which is the better situation, since this runs the actual op.
fn prologue_5tok<'c>(
    s: &LayerSums,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
) -> Prologue5<'c> {
    let hc = config.hc_mult as i64;
    let nt = s.tokens.len() as i64;

    let tok = ctx.new_i32_1d(nt).expect("tok");
    tok.set_i32(s.tokens).expect("set");
    let embd = ctx
        .get_rows(weights.get("token_embd.weight").expect("bound"), &tok)
        .expect("get_rows");
    ctx.compute(&embd, 12).expect("compute embd");
    check(s, "embd", embd.to_vec_f32().iter().sum::<f32>());

    // [n_embd, tokens] -> [n_embd, 1, tokens] -> repeat to [n_embd, hc, tokens].
    let embd_r = ctx
        .reshape_3d(&embd, config.n_embd as i64, 1, nt)
        .expect("reshape embd");
    let shape = ctx
        .new_f32_3d(config.n_embd as i64, hc, nt)
        .expect("hc_init shape");
    let hc_init = ctx.repeat(&embd_r, &shape).expect("hc_init");
    ctx.compute(&hc_init, 12).expect("compute hc_init");
    check(s, "hc_init", hc_init.to_vec_f32().iter().sum::<f32>());

    let flat = ctx
        .reshape_2d(&hc_init, config.hc_dim() as i64, nt)
        .expect("flatten streams");
    let normed = ctx.rms_norm(&flat, config.rms_eps).expect("rms_norm");
    ctx.compute(&normed, 12).expect("compute norm");
    check(s, "hc_init_norm", normed.to_vec_f32().iter().sum::<f32>());

    layer_entry_5tok(s, ctx, weights, config, &hc_init)
}

/// A layer's entry: the attention gate block and `attn_norm`, from whatever
/// residual streams it was handed.
///
/// Layer 0 reaches here from the embedding; every other layer reaches it from
/// the previous layer's `l_last`. That is the *only* structural difference
/// between the first layer and the rest, which is why it is the seam.
///
/// Note the incoming streams are RMS-normed by the caller, because for layers
/// past the first that norm is the previous layer's last checkpoint
/// (`next_norm`) and is verified there rather than twice.
fn layer_entry_5tok<'c>(
    s: &LayerSums,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    streams: &Tensor<'c>,
) -> Prologue5<'c> {
    let nt = s.tokens.len() as i64;
    let flat = ctx
        .reshape_2d(streams, config.hc_dim() as i64, nt)
        .expect("flatten streams");
    let normed = ctx.rms_norm(&flat, config.rms_eps).expect("rms_norm");

    let mixes = ctx
        .mul_mat(weights.get(&format!("blk.{}.hc_attn_fn.weight", s.il)).expect("bound"), &normed)
        .expect("hc_mixes");
    ctx.compute(&mixes, 12).expect("compute mixes");
    check(s, "hc_mixes_attn", mixes.to_vec_f32().iter().sum::<f32>());

    let gates = hc_gates(ctx, weights, config, &format!("blk.{}.hc_attn", s.il), &mixes, s.attn_gates, nt);

    let collapsed = ctx.dsv4_hc_pre(streams, &gates.pre).expect("dsv4_hc_pre");
    ctx.compute(&collapsed, 12).expect("compute hc_pre op");
    check(s, "hc_attn_pre", collapsed.to_vec_f32().iter().sum::<f32>());

    let normed = ctx.rms_norm(&collapsed, config.rms_eps).expect("norm");
    ctx.compute(&normed, 12).expect("compute");
    check(s, "norm_attn", normed.to_vec_f32().iter().sum::<f32>());

    let attn_norm = ctx
        .mul(&normed, weights.get(&format!("blk.{}.attn_norm.weight", s.il)).expect("bound"))
        .expect("attn_norm");
    ctx.compute(&attn_norm, 12).expect("compute");
    check(s, "attn_norm", attn_norm.to_vec_f32().iter().sum::<f32>());

    Prologue5 { hc_init: *streams, attn_norm, gates }
}

/// What one `build_hc_pre` call produces.
///
/// All three gates come out of a **single** mixes matmul (`deepseek4.cpp:264`),
/// which is why they are returned together rather than recomputed: `pre`
/// collapses the streams on the way in, `post` and `comb` write the block's
/// output back on the way out.
struct HcGates<'c> {
    pre: Tensor<'c>,
    post: Tensor<'c>,
    comb: Tensor<'c>,
}

struct Prologue5<'c> {
    /// The four residual streams, `[n_embd, hc, tokens]`.
    hc_init: Tensor<'c>,
    /// Input to the Q and KV projections.
    attn_norm: Tensor<'c>,
    /// The attention block's gates, needed again after attention.
    gates: HcGates<'c>,
}

/// The oracle's numbers for one gate block. Two blocks per layer, same
/// structure, different weights and therefore different sums.
#[derive(Clone, Copy)]
struct HcGateSums {
    pre_view: f32,
    pre_scaled: f32,
    pre_biased: f32,
    pre_sigmoid: f32,
    pre: f32,
    post_view: f32,
    post_scaled: f32,
    post_biased: f32,
    post_sigmoid: f32,
    post: f32,
    comb: f32,
}

impl HcGateSums {
    fn from_map(m: &std::collections::HashMap<String, f32>) -> HcGateSums {
        let g = |k: &str| *m.get(k).unwrap_or(&f32::NAN);
        HcGateSums {
            pre_view: g("pre_view"),
            pre_scaled: g("pre_scaled"),
            pre_biased: g("pre_biased"),
            pre_sigmoid: g("pre_sigmoid"),
            pre: g("pre"),
            post_view: g("post_view"),
            post_scaled: g("post_scaled"),
            post_biased: g("post_biased"),
            post_sigmoid: g("post_sigmoid"),
            post: g("post"),
            comb: g("comb"),
        }
    }
}

/// Slice the 24 mixes into the three gates, exactly as `build_hc_pre` does.
///
/// The layout is `[0..hc]` pre, `[hc..2hc]` post, `[2hc..2hc+hc*hc]` the
/// combination matrix — with `hc_scale` indexed `[pre, post, comb]` and
/// `hc_base` `[hc pre, hc post, hc*hc comb]`. **Every one of those views is the
/// right size whichever slice you take**, so getting the offsets wrong has no
/// shape consequence at all; only these sums catch it.
///
/// `pre` and `post` differ in their tail: `pre` gets `scale_bias(x, 1, hc_eps)`
/// and `post` gets `scale(x, 2.0)` (`deepseek4.cpp:294, 300`). Swapping those is
/// another silent one.
fn hc_gates<'c>(
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    prefix: &str,
    mixes: &Tensor<'c>,
    want: HcGateSums,
    nt: i64,
) -> HcGates<'c> {
    let hc = config.hc_mult as i64;
    let f32_size = std::mem::size_of::<f32>();
    // The stride is the *source's* row, `hc_mix_dim = (2 + hc) * hc` = 24
    // floats (`deepseek4.cpp:277`, and `dsv4_view_2d` passes `t->nb[1]`), not
    // the 4 the view is wide. At one token the stride is never traversed and
    // any value for it passes, which is why the five-token capture is what
    // pinned this.
    let mix_stride = ((2 + hc) * hc) as usize * f32_size;

    let scale_w = weights.get(&format!("{prefix}_scale.weight")).expect("scale bound");
    let base_w = weights.get(&format!("{prefix}_base.weight")).expect("base bound");

    // One gate: view the mixes slice, affine it, sigmoid, then its own tail.
    let gate = |label: &str, mix_off: i64, scale_idx: i64, base_off: i64,
                sums: (f32, f32, f32, f32)| {
        let view = ctx
            .view_2d(mixes, hc, nt, mix_stride, mix_off as usize * f32_size)
            .expect("mixes view");
        ctx.compute(&view, 12).expect("compute view");
        assert_sum(&format!("{label} (view)"), view.to_vec_f32().iter().sum::<f32>(), sums.0);

        let s = ctx
            .view_1d(scale_w, 1, scale_idx as usize * f32_size)
            .expect("scale view");
        let b = ctx
            .view_1d(base_w, hc, base_off as usize * f32_size)
            .expect("base view");

        let scaled = ctx.mul(&view, &s).expect("mul scale");
        ctx.compute(&scaled, 12).expect("compute scaled");
        assert_sum(&format!("{label} (mul)"), scaled.to_vec_f32().iter().sum::<f32>(), sums.1);

        let biased = ctx.add(&scaled, &b).expect("add base");
        ctx.compute(&biased, 12).expect("compute biased");
        assert_sum(&format!("{label} (add)"), biased.to_vec_f32().iter().sum::<f32>(), sums.2);

        let gated = ctx.sigmoid(&biased).expect("sigmoid");
        ctx.compute(&gated, 12).expect("compute sigmoid");
        assert_sum(&format!("{label} (sigmoid)"), gated.to_vec_f32().iter().sum::<f32>(), sums.3);

        gated
    };

    let pre_gated = gate(
        "hc_pre",
        0,
        0,
        0,
        (want.pre_view, want.pre_scaled, want.pre_biased, want.pre_sigmoid),
    );
    // scale_bias(pre, 1.0, hc_eps): the epsilon is what turns 20.000000 into
    // 20.000015, so this one number pins hyper_connection.epsilon at 1e-6.
    let eps_t = ctx.new_f32_1d(hc).expect("eps");
    eps_t.set_f32(&vec![1e-6f32; hc as usize]).expect("fill eps");
    let pre = ctx.add(&pre_gated, &eps_t).expect("add eps");
    ctx.compute(&pre, 12).expect("compute pre");
    assert_sum("hc_pre (scale_bias)", pre.to_vec_f32().iter().sum::<f32>(), want.pre);

    let post_gated = gate(
        "hc_post",
        hc,
        1,
        hc,
        (want.post_view, want.post_scaled, want.post_biased, want.post_sigmoid),
    );
    let post = ctx.scale(&post_gated, 2.0).expect("scale post");
    ctx.compute(&post, 12).expect("compute post");
    assert_sum("hc_post (scale)", post.to_vec_f32().iter().sum::<f32>(), want.post);

    // The combination matrix is fused: ggml slices the mixes, applies the
    // affine and runs all 20 Sinkhorn iterations itself.
    let comb = ctx
        .dsv4_hc_comb(mixes, scale_w, base_w, 1e-6, config.hc_sinkhorn_iterations as i32)
        .expect("dsv4_hc_comb");
    ctx.compute(&comb, 12).expect("compute comb");
    assert_sum("hc_comb (DSV4_HC_COMB)", comb.to_vec_f32().iter().sum::<f32>(), want.comb);

    HcGates { pre, post, comb }
}

/// **The RoPE checkpoint.** Q and KV at five tokens, rotation included.
///
/// This exists because the one-token oracle could not check RoPE at all. At
/// position 0 the rotation is the identity, so `q_pe` there has exactly the
/// same sum as its input and an implementation that skipped the rotation
/// entirely would have passed. Positions 1..4 make the two numbers differ:
///
/// ```text
/// q_norm-0 (view)  {64, 64, 5}    695.835632
/// q_pe-0    ROPE   {64, 64, 5}   4082.126465   <- would still be 695.8 if unrotated
/// ```
///
/// Three things are under test here that nothing before could reach:
///
/// 1. **The rotation happens**, with this model's `freq_base` and no YaRN on
///    this layer.
/// 2. **Only 64 of each head's 512 dims are rotated.** `q_nope` is spliced back
///    unchanged, and `concat(q_nope, q_pe)` has to reproduce `q_norm`'s own sum
///    on the 448 dims while changing it on the 64.
/// 3. **The pairing convention is NORM, not NEOX.** Both produce a rotated
///    tensor of the right shape and a plausible-looking model.
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn rope_and_kv_match_llama_cpp_at_five_tokens() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    let ctx = Context::new(512 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(8 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    bind_all(&model, &wctx, &mut weights, &block_weights(0));
    bind_all(&model, &wctx, &mut weights, &optional_block_weights(&model, 0));

    let s = &sums_5tok(0);
    let p = prologue_5tok(s, &ctx, &weights, &config);
    let (q_full, kv_full) = q_and_kv_5tok(s, &ctx, &weights, &config, &p.attn_norm);
    check(s, "q", q_full.to_vec_f32().iter().sum::<f32>());
    check(s, "kv", kv_full.to_vec_f32().iter().sum::<f32>());
}

/// Every weight one block needs, for a full layer.
///
/// Built per layer rather than listed as `blk.0.*` constants, because running a
/// second layer is the whole point: the helpers take a block index and this
/// follows it.
fn block_weights(il: u32) -> Vec<String> {
    let mut names = vec!["token_embd.weight".to_string()];
    for suffix in [
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
        "attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b",
        "attn_kv", "attn_kv_a_norm", "attn_sinks",
        "attn_output_a", "attn_output_b",
        "ffn_norm", "ffn_gate_inp",
        "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
    ] {
        names.push(format!("blk.{il}.{suffix}.weight"));
    }
    names
}

/// Weights only some blocks carry, bound when they exist.
///
/// `ffn_gate_tid2eid` is on 3 blocks and `exp_probs_b` on 40 — and they are
/// mutually exclusive, being the two routing schemes. Binding blindly would
/// panic on whichever the block lacks.
fn optional_block_weights(model: &Model, il: u32) -> Vec<String> {
    ["ffn_gate_tid2eid.weight", "exp_probs_b.bias"]
        .iter()
        .map(|suffix| format!("blk.{il}.{suffix}"))
        .filter(|n| model.location(n).is_some())
        .collect()
}

/// Q and KV at five tokens, rotation included, checked step by step.
///
/// Returns `(q, kv)` shaped as llama.cpp leaves them: `q` is
/// `[head_dim, n_head, tokens]` and `kv` is `[head_dim, 1, tokens]` — one head,
/// and the same tensor will serve as both K and V.
fn q_and_kv_5tok<'c>(
    s: &LayerSums,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    attn_norm: &Tensor<'c>,
) -> (Tensor<'c>, Tensor<'c>) {
    let nt = s.tokens.len() as i64;
    let head_dim = config.kv_lora_rank as i64;
    let n_rot = config.n_rot as i64;
    let n_nope = config.n_rot_none() as i64;
    let f32_size = std::mem::size_of::<f32>();
    let head_stride = head_dim as usize * f32_size;
    let (rope, rope_n_ctx_orig) = rope_for(config, s.il);

    // Positions 0..4. This tensor is the whole difference from the one-token
    // capture: with a single zero in it, every assertion below still passes on
    // a forward pass that never rotates anything.
    let pos = ctx.new_i32_1d(nt).expect("pos");
    let positions: Vec<i32> = (0..nt as i32).collect();
    pos.set_i32(&positions).expect("set pos");

    // ---- Q ----
    let qr = ctx
        .mul_mat(weights.get(&format!("blk.{}.attn_q_a.weight", s.il)).expect("bound"), attn_norm)
        .expect("qr");
    ctx.compute(&qr, 12).expect("compute");
    check(s, "qr", qr.to_vec_f32().iter().sum::<f32>());

    let qr_n = ctx.rms_norm(&qr, config.rms_eps).expect("qr rms");
    ctx.compute(&qr_n, 12).expect("compute");
    check(s, "qr_rms", qr_n.to_vec_f32().iter().sum::<f32>());

    let qr_norm = ctx
        .mul(&qr_n, weights.get(&format!("blk.{}.attn_q_a_norm.weight", s.il)).expect("bound"))
        .expect("qr_norm");
    ctx.compute(&qr_norm, 12).expect("compute");
    check(s, "qr_norm", qr_norm.to_vec_f32().iter().sum::<f32>());

    let q = ctx
        .mul_mat(weights.get(&format!("blk.{}.attn_q_b.weight", s.il)).expect("bound"), &qr_norm)
        .expect("q");
    ctx.compute(&q, 12).expect("compute");
    check(s, "q_b", q.to_vec_f32().iter().sum::<f32>());

    let q3 = ctx
        .reshape_3d(&q, head_dim, config.n_head as i64, nt)
        .expect("reshape q");
    let q_norm = ctx.rms_norm(&q3, config.rms_eps).expect("q_norm");
    ctx.compute(&q_norm, 12).expect("compute");
    check(s, "q_norm", q_norm.to_vec_f32().iter().sum::<f32>());

    // The decoupled split. Both views keep the *source's* head stride, so they
    // interleave in memory rather than being two halves of a contiguous buffer;
    // that is why this needs view_3d and not reshape.
    let q_nope = ctx
        .view_3d(
            &q_norm,
            n_nope,
            config.n_head as i64,
            nt,
            head_stride,
            head_stride * config.n_head as usize,
            0,
        )
        .expect("q_nope");
    ctx.compute(&q_nope, 12).expect("compute q_nope");
    check(s, "q_nope", q_nope.to_vec_f32().iter().sum::<f32>());

    let q_pe_in = ctx
        .view_3d(
            &q_norm,
            n_rot,
            config.n_head as i64,
            nt,
            head_stride,
            head_stride * config.n_head as usize,
            n_nope as usize * f32_size,
        )
        .expect("q_pe view");
    ctx.compute(&q_pe_in, 12).expect("compute q_pe view");
    let unrotated: f32 = q_pe_in.to_vec_f32().iter().sum();
    check(s, "q_pe_in", unrotated);

    let q_pe = ctx
        .rope_ext(&q_pe_in, &pos, None, n_rot as i32, ROPE_MODE_NORM, rope_n_ctx_orig, rope)
        .expect("rope q_pe");
    ctx.compute(&q_pe, 12).expect("compute rope");
    let rotated: f32 = q_pe.to_vec_f32().iter().sum();
    check(s, "q_pe", rotated);

    // Guard the guard: if a future capture is taken at one token again, the two
    // sums collapse and this assertion is what says so instead of the suite
    // going quietly green.
    assert!(
        (rotated - unrotated).abs() > 1.0,
        "rope did not change the tensor ({unrotated:.6} -> {rotated:.6}); \
         the oracle cannot be validating the rotation"
    );

    let q_full = ctx.concat(&q_nope, &q_pe, 0).expect("concat q");
    ctx.compute(&q_full, 12).expect("compute q");

    // ---- KV ----
    // One head, and the same tensor serves as K *and* V (deepseek4.cpp:792).
    let kv = ctx
        .mul_mat(weights.get(&format!("blk.{}.attn_kv.weight", s.il)).expect("bound"), attn_norm)
        .expect("kv");
    ctx.compute(&kv, 12).expect("compute");
    check(s, "kv_a", kv.to_vec_f32().iter().sum::<f32>());

    let kv_n = ctx.rms_norm(&kv, config.rms_eps).expect("kv rms");
    ctx.compute(&kv_n, 12).expect("compute");
    check(s, "kv_rms", kv_n.to_vec_f32().iter().sum::<f32>());

    let kv_norm = ctx
        .mul(&kv_n, weights.get(&format!("blk.{}.attn_kv_a_norm.weight", s.il)).expect("bound"))
        .expect("kv_norm");
    ctx.compute(&kv_norm, 12).expect("compute");
    check(s, "kv_norm", kv_norm.to_vec_f32().iter().sum::<f32>());

    let kv3 = ctx.reshape_3d(&kv_norm, head_dim, 1, nt).expect("reshape kv");
    let kv_nope = ctx
        .view_3d(&kv3, n_nope, 1, nt, head_stride, head_stride, 0)
        .expect("kv_nope");
    ctx.compute(&kv_nope, 12).expect("compute kv_nope");
    check(s, "kv_nope", kv_nope.to_vec_f32().iter().sum::<f32>());

    let kv_pe_in = ctx
        .view_3d(
            &kv3,
            n_rot,
            1,
            nt,
            head_stride,
            head_stride,
            n_nope as usize * f32_size,
        )
        .expect("kv_pe view");
    ctx.compute(&kv_pe_in, 12).expect("compute kv_pe view");
    check(s, "kv_pe_in", kv_pe_in.to_vec_f32().iter().sum::<f32>());

    let kv_pe = ctx
        .rope_ext(&kv_pe_in, &pos, None, n_rot as i32, ROPE_MODE_NORM, rope_n_ctx_orig, rope)
        .expect("rope kv_pe");
    ctx.compute(&kv_pe, 12).expect("compute rope kv");
    check(s, "kv_pe", kv_pe.to_vec_f32().iter().sum::<f32>());

    let kv_full = ctx.concat(&kv_nope, &kv_pe, 0).expect("concat kv");
    ctx.compute(&kv_full, 12).expect("compute kv");

    (q_full, kv_full)
}

/// **Attention.** The fused kernel, the F16 cache, the mask and the sinks.
///
/// Oracle rows:
/// ```text
/// cache_k_l0 (view)  SET_ROWS        {512, 512}     63.123978
/// node_41            FLASH_ATTN_EXT  {512, 64, 5}  2879.606934
/// ```
///
/// Four things here are wrong-without-an-error, and the sum catches each:
///
/// 1. **`kv` is passed as K *and* V.** There is no separate V projection —
///    `build_raw_attention` calls `build_attn_mha(q, k, k, ...)`
///    (`deepseek4.cpp:792`). `head_count_kv` is 1 because of this.
/// 2. **The cache is F16**, so the reference sum after the cache write is
///    63.123978 where the tensor going in summed to 63.125298. That 1.3e-3 is
///    rounding; comparing the pre-cache number here would look like a near-miss
///    and be a different bug.
/// 3. **`n_kv` is padded to 256**, not the 5 tokens present. The 251 unused
///    slots are zero and must be masked to -inf, or they contribute a
///    `softmax(0)` share to every score.
/// 4. **Per-head sinks**, from `attn_sinks.weight`. A sink is an extra
///    always-attended logit with no value attached, so it changes only the
///    softmax denominator — omit it and every output is scaled slightly wrong,
///    with the right shape and no complaint.
///
/// Not covered, and deliberately not guessed at: raw layers are **sliding-window
/// layers** (`GGML_ASSERT(hparams.is_swa(il))`, window 128). At five tokens no
/// query reaches back 128 positions, so this capture cannot distinguish a
/// windowed mask from a plain causal one and the mask below is plain causal.
/// Verifying the window needs a capture longer than 128 tokens.
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn attention_matches_llama_cpp_at_five_tokens() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    let ctx = Context::new(512 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(8 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    bind_all(&model, &wctx, &mut weights, &block_weights(0));
    bind_all(&model, &wctx, &mut weights, &optional_block_weights(&model, 0));

    let p = prologue_5tok(&sums_5tok(0), &ctx, &weights, &config);
    let (q, kv) = q_and_kv_5tok(&sums_5tok(0), &ctx, &weights, &config, &p.attn_norm);
    attention_5tok(&sums_5tok(0), &ctx, &weights, &config, &q, &kv);
}

/// **The post hyper-connection**, and the second gate block that feeds the FFN.
///
/// Oracle rows:
/// ```text
/// hc_attn_post-0  DSV4_HC_POST  {4096, 4, 5}   -14.514359
/// node_65         RMS_NORM      {16384, 5}     -77.870285
/// hc_mixes-0      MUL_MAT       {24, 5}      -3608.835205
/// hc_comb-0       DSV4_HC_COMB  {4, 4, 5}       19.999979
/// hc_ffn_pre-0    DSV4_HC_PRE   {4096, 5}        1.926467
/// ffn_norm-0      MUL           {4096, 5}       11.634495
/// ```
///
/// This is where the residual stream is written *back*. A plain transformer does
/// `x = x + f(x)`; V4-Flash does
/// `x[dst] = f(x)*post[dst] + sum_src x[src]*comb[dst, src]`, with `comb` a
/// Sinkhorn-normalised 4x4 mixing matrix. Getting it wrong does not change any
/// shape.
///
/// Two things the sums pin that reading the code alone would not:
///
/// 1. **`post` and `pre` have different tails.** `pre` gets
///    `scale_bias(x, 1, hc_eps)` and `post` gets `scale(x, 2.0)`
///    (`deepseek4.cpp:294, 300`). The 2.0 is why `hc_post` sums to twice its
///    sigmoid.
/// 2. **The FFN's gates come from a second, independent mixes matmul** against
///    `hc_ffn_fn` over the *post-attention* stream — not from the attention
///    block's mixes. Reusing the first would be free of any error.
///
/// `hc_comb` summing to ~20.0 (5 tokens x 4 rows summing to 1) is the Sinkhorn
/// normalisation showing up: it is doubly stochastic, so this row would catch an
/// iteration count of 0 as easily as one of 19.
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn post_hyper_connection_matches_llama_cpp_at_five_tokens() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    let ctx = Context::new(512 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(8 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    bind_all(&model, &wctx, &mut weights, &block_weights(0));
    bind_all(&model, &wctx, &mut weights, &optional_block_weights(&model, 0));

    let p = prologue_5tok(&sums_5tok(0), &ctx, &weights, &config);
    let (q, kv) = q_and_kv_5tok(&sums_5tok(0), &ctx, &weights, &config, &p.attn_norm);
    let attn_out = attention_5tok(&sums_5tok(0), &ctx, &weights, &config, &q, &kv);

    let _ = layer_tail_5tok(&sums_5tok(0), &ctx, &weights, &config, &p, &attn_out);
}

/// `swiglu_clamp_exp[0]` and `swiglu_clamp_shexp[0]`, both 10 in this
/// container.
///
/// **These are per-layer arrays of 43 values and [`Deepseek4Config`] reads
/// neither.** Hardcoding index 0 is correct for this test and wrong for a real
/// forward pass; the config needs to carry them before any layer but 0 runs.
const SWIGLU_CLAMP_L0: f32 = 10.0;

/// **The MoE router and the shared expert.**
///
/// Oracle rows:
/// ```text
/// ffn_moe_logits-0            MUL_MAT   {256, 5}   -1176.607300
/// node_86                     SOFTPLUS  {256, 5}     587.096008
/// ffn_moe_probs-0             SQRT      {256, 5}     792.403992
/// ffn_moe_topk-0              GET_ROWS  {6, 5}      3688.000000
/// ffn_moe_weights-0           GET_ROWS  {1, 6, 5}     20.336262
/// ffn_moe_weights_norm-0      DIV       {6, 5}         5.000000
/// ffn_moe_weights_scaled-0    SCALE     {1, 6, 5}      7.500000
/// ffn_shexp-0                 MUL_MAT   {4096, 5}     16.228374
/// ```
///
/// The interesting rows are the ones that contradict what a DeepSeek MoE is
/// normally assumed to do:
///
/// 1. **`ffn_moe_topk-0` is a `GET_ROWS`, not a top-k.** Layers 0-2 are the
///    `hash_layer_count` layers: their six experts come from
///    `ffn_gate_tid2eid`, a `[6, vocab]` I32 table indexed by *token id*. The
///    router probabilities are still computed, but only to weight experts that
///    were already chosen. For a streaming runner this is worth more than a
///    correctness check — on these layers the expert set is knowable before any
///    compute happens.
/// 2. **The gate is `sqrt(softplus(x))`.** `expert_gating_func 4`.
/// 3. **Weights are renormalised over the selected six only**, then scaled by
///    1.5. `ffn_moe_weights_norm` summing to exactly 5.0 across 5 tokens is
///    that renormalisation, and 7.5 is the scale.
/// 4. **The SwiGLU clamp is asymmetric on the gate**: `(-inf, 10]` for the
///    gate, `[-10, 10]` for the up projection, in a `LLM_ARCH_DEEPSEEK4` branch
///    (`llama-graph.cpp:2050-2057`). At five tokens neither bound is actually
///    reached — the clamped sums equal the unclamped ones — so this capture
///    confirms the *shape* of the computation but **not** the bounds. Noted
///    rather than claimed.
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn moe_router_and_shared_expert_match_llama_cpp_at_five_tokens() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    let ctx = Context::new(512 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(8 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    bind_all(&model, &wctx, &mut weights, &block_weights(0));
    bind_all(&model, &wctx, &mut weights, &optional_block_weights(&model, 0));

    let p = prologue_5tok(&sums_5tok(0), &ctx, &weights, &config);
    let (q, kv) = q_and_kv_5tok(&sums_5tok(0), &ctx, &weights, &config, &p.attn_norm);
    let attn_out = attention_5tok(&sums_5tok(0), &ctx, &weights, &config, &q, &kv);
    let (_streams, ffn_norm, _gates) = layer_tail_5tok(&sums_5tok(0), &ctx, &weights, &config, &p, &attn_out);

    let _ = moe_routing_5tok(&sums_5tok(0), &ctx, &weights, &config, &ffn_norm);
    let _ = shared_expert_5tok(&sums_5tok(0), &ctx, &weights, &ffn_norm);
}

/// The router: probabilities, the six experts, and their normalised weights.
///
/// Returns `(weights, ids)` — the scaled weights shaped `[1, n_used, tokens]`
/// ready to multiply the expert outputs, and the expert ids as `mul_mat_id`
/// wants them.
fn moe_routing_5tok<'c>(
    s: &LayerSums,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    ffn_norm: &Tensor<'c>,
) -> (Tensor<'c>, Tensor<'c>) {
    let nt = s.tokens.len() as i64;
    let n_expert = config.n_expert as i64;
    let n_used = config.n_expert_used as i64;

    // ---- routing ----
    let logits = ctx
        .mul_mat(weights.get(&format!("blk.{}.ffn_gate_inp.weight", s.il)).expect("bound"), ffn_norm)
        .expect("logits");
    ctx.compute(&logits, 12).expect("compute logits");
    check(s, "moe_logits", logits.to_vec_f32().iter().sum::<f32>());

    let sp = ctx.softplus(&logits).expect("softplus");
    ctx.compute(&sp, 12).expect("compute softplus");
    check(s, "moe_softplus", sp.to_vec_f32().iter().sum::<f32>());

    let probs = ctx.sqrt(&sp).expect("sqrt");
    ctx.compute(&probs, 12).expect("compute sqrt");
    check(s, "moe_probs", probs.to_vec_f32().iter().sum::<f32>());

    let probs3 = ctx.reshape_3d(&probs, 1, n_expert, nt).expect("reshape probs");

    // Two entirely different ways of choosing experts, and which one a layer
    // uses is decided by `hash_layer_count`.
    let topk = if s.il < config.hash_layer_count {
        // Hash routing (layers 0-2): the six experts are a lookup on the token
        // id and the router never picks anything. `exp_probs_b` is not applied
        // at all on these layers.
        let tok = ctx.new_i32_1d(nt).expect("tok");
        tok.set_i32(s.tokens).expect("set");
        ctx.get_rows(
            weights.get(&format!("blk.{}.ffn_gate_tid2eid.weight", s.il)).expect("bound"),
            &tok,
        )
        .expect("topk")
    } else {
        // The path the other 40 layers use. The selection bias is added to a
        // *copy* — llama.cpp's own comment is "leave probs unbiased as it's
        // later used to get expert weights" (`llama-graph.cpp:1885`). Biasing
        // the weights too is the natural mistake: it changes every expert
        // weight and no shape.
        let biased = ctx
            .add(&probs, weights.get(&format!("blk.{}.exp_probs_b.bias", s.il)).expect("bound"))
            .expect("probs_biased");
        ctx.compute(&biased, 12).expect("compute biased");
        check(s, "moe_probs_biased", biased.to_vec_f32().iter().sum::<f32>());

        // argsort_top_k, not top_k: this one's indices *are* in score order.
        let sel = ctx.argsort_top_k(&biased, n_used as i32).expect("argsort_top_k");
        ctx.compute(&sel, 12).expect("compute argsort");
        sel
    };
    ctx.compute(&topk, 12).expect("compute topk");
    let ids = topk.to_vec_i32();
    assert_eq!(ids.len(), (n_used * nt) as usize, "six experts per token");
    check(s, "moe_topk", ids.iter().sum::<i32>() as f32);

    // Always from the unbiased probabilities, on both routing paths.
    let w = ctx.get_rows(&probs3, &topk).expect("weights");
    ctx.compute(&w, 12).expect("compute weights");
    check(s, "moe_weights", w.to_vec_f32().iter().sum::<f32>());

    // Renormalise over the *selected* six, not over all 256. This is the step
    // whose absence is invisible: the weights still sum to something, the model
    // still speaks.
    let w2 = ctx.reshape_2d(&w, n_used, nt).expect("reshape weights");
    let sum = ctx.sum_rows(&w2).expect("sum_rows");
    ctx.compute(&sum, 12).expect("compute sum");
    check(s, "moe_weights_sum", sum.to_vec_f32().iter().sum::<f32>());

    // Clamped away from zero at the smallest F16 normal, not at some epsilon.
    let sum_c = ctx.clamp(&sum, 6.103515625e-5, f32::INFINITY).expect("clamp sum");
    ctx.compute(&sum_c, 12).expect("compute clamped sum");
    check(s, "moe_weights_sum_clamped", sum_c.to_vec_f32().iter().sum::<f32>());

    let w_norm = ctx.div(&w2, &sum_c).expect("div");
    ctx.compute(&w_norm, 12).expect("compute norm");
    check(s, "moe_weights_norm", w_norm.to_vec_f32().iter().sum::<f32>());

    // Reshaped back to [1, n_used, tokens] *before* the scale, so it can
    // broadcast over each expert's [n_embd] output later.
    let w3 = ctx
        .reshape_3d(&w_norm, 1, n_used, nt)
        .expect("reshape weights");
    let w_scaled = ctx
        .scale(&w3, config.expert_weights_scale)
        .expect("scale weights");
    ctx.compute(&w_scaled, 12).expect("compute scaled");
    check(s, "moe_weights_scaled", w_scaled.to_vec_f32().iter().sum::<f32>());

    (w_scaled, topk)
}

/// **The routed experts, and the rest of layer 0.**
///
/// Oracle rows:
/// ```text
/// ffn_moe_gate-0            MUL_MAT_ID  {2048, 6, 5}  -6601.376953
/// ffn_moe_up-0              MUL_MAT_ID  {2048, 6, 5}     -8.613072
/// ffn_moe_swiglu_limited-0  SWIGLU      {2048, 6, 5}     11.649389
/// ffn_moe_down-0            MUL_MAT_ID  {4096, 6, 5}     89.523140
/// ffn_moe_out-0             ADD         {4096, 5}        18.572350
/// ffn_out-0                 ADD         {4096, 5}        34.800404
/// l_last-0                  DSV4_HC_POST {4096, 4, 5}     6.733532
/// node_125                  RMS_NORM    {16384, 5}        1.599161
/// ```
///
/// **Only the selected expert slices are read.** llama.cpp mmaps all 256 and
/// lets `mul_mat_id` index into them; binding layer 0's three stacked tensors
/// that way is ~3.2 GiB, which does not fit on this machine. Instead the unique
/// experts the five tokens actually route to are read individually with
/// `read_tensor_range` and packed into a compact stack, with the ids remapped
/// to match — around a tenth of the bytes, and the same arithmetic.
///
/// That is not a shortcut taken for the test's convenience: **it is what the
/// runner has to do anyway.** This is the first time the port has exercised the
/// partial-read path against a reference, and the sums say the packing and the
/// id remapping are both right — a remap that scrambled experts would still
/// produce a full-rank result of exactly the right shape.
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn routed_experts_and_layer_output_match_llama_cpp_at_five_tokens() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    let ctx = Context::new(512 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(16 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    bind_all(&model, &wctx, &mut weights, &block_weights(0));
    bind_all(&model, &wctx, &mut weights, &optional_block_weights(&model, 0));

    let _ = layer0_5tok(&sums_5tok(0), &model, &ctx, &wctx, &mut weights, &config);
}

/// The whole of layer 0, returning `l_last-0` — the four residual streams layer
/// 1 receives.
fn layer0_5tok<'c>(
    s: &LayerSums,
    model: &Model,
    ctx: &'c Context,
    wctx: &'c Context,
    weights: &mut WeightSet<'c>,
    config: &Deepseek4Config,
) -> Tensor<'c> {
    let p = prologue_5tok(s, ctx, weights, config);
    layer_body_5tok(s, model, ctx, wctx, weights, config, p)
}

/// A whole layer that is *not* the first: no embedding, no `hc_init`. It takes
/// the previous layer's `l_last` as its residual streams and returns its own.
///
/// This is what a real forward pass runs 42 times.
fn layer_5tok<'c>(
    s: &LayerSums,
    model: &Model,
    ctx: &'c Context,
    wctx: &'c Context,
    weights: &mut WeightSet<'c>,
    config: &Deepseek4Config,
    streams: &Tensor<'c>,
) -> Tensor<'c> {
    let p = layer_entry_5tok(s, ctx, weights, config, streams);
    layer_body_5tok(s, model, ctx, wctx, weights, config, p)
}

/// Everything after the residual streams exist, shared by every layer.
fn layer_body_5tok<'c>(
    s: &LayerSums,
    model: &Model,
    ctx: &'c Context,
    wctx: &'c Context,
    weights: &mut WeightSet<'c>,
    config: &Deepseek4Config,
    p: Prologue5<'c>,
) -> Tensor<'c> {
    let (q, kv) = q_and_kv_5tok(s, ctx, weights, config, &p.attn_norm);
    let attn_out = attention_5tok(s, ctx, weights, config, &q, &kv);
    let (streams, ffn_norm, gates) = layer_tail_5tok(s, ctx, weights, config, &p, &attn_out);
    let (w_scaled, topk) = moe_routing_5tok(s, ctx, weights, config, &ffn_norm);
    let shexp = shared_expert_5tok(s, ctx, weights, &ffn_norm);

    let nt = s.tokens.len() as i64;
    let n_embd = config.n_embd as i64;
    let n_used = config.n_expert_used as i64;
    let f32_size = std::mem::size_of::<f32>();

    // ---- read only the experts these five tokens route to ----
    let ids = topk.to_vec_i32();
    let mut unique = ids.clone();
    unique.sort_unstable();
    unique.dedup();
    let position = |e: i32| unique.iter().position(|u| *u == e).expect("in set") as i32;
    let compact: Vec<i32> = ids.iter().map(|e| position(*e)).collect();
    eprintln!(
        "  {:<24} {} of {} experts, {} slots",
        "routed",
        unique.len(),
        config.n_expert,
        ids.len()
    );

    let mut read_bytes = 0u64;
    let mut dims_of = std::collections::HashMap::new();
    for suffix in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
        let name = format!("blk.{}.{suffix}.weight", s.il);
        let (bytes, dims) = bind_expert_slices(model, wctx, weights, &name, &unique);
        read_bytes += bytes;
        dims_of.insert(suffix, dims);
    }
    eprintln!(
        "  {:<24} {:.2} GiB read (all 256 would be {:.2} GiB)",
        "expert slices",
        read_bytes as f64 / (1 << 30) as f64,
        read_bytes as f64 / unique.len() as f64 * config.n_expert as f64 / (1 << 30) as f64
    );

    let n_uniq = unique.len() as i64;
    let ids_t = ctx.new_i32_2d(n_used, nt).expect("ids");
    ids_t.set_i32(&compact).expect("set ids");

    let stack = |suffix: &str| {
        let d = &dims_of[suffix];
        ctx.reshape_3d(
            weights
                .get(&format!("blk.{}.{suffix}.weight", s.il))
                .expect("expert stack bound"),
            d[0] as i64,
            d[1] as i64,
            n_uniq,
        )
        .expect("reshape experts")
    };

    // ---- the expert FFN ----
    let cur3 = ctx.reshape_3d(&ffn_norm, n_embd, 1, nt).expect("reshape cur");

    let gate = ctx
        .mul_mat_id(&stack("ffn_gate_exps"), &cur3, &ids_t)
        .expect("moe gate");
    ctx.compute(&gate, 12).expect("compute moe gate");
    check(s, "moe_gate", gate.to_vec_f32().iter().sum::<f32>());

    let gate_c = ctx
        .clamp(&gate, f32::NEG_INFINITY, SWIGLU_CLAMP_L0)
        .expect("clamp gate");

    let up = ctx
        .mul_mat_id(&stack("ffn_up_exps"), &cur3, &ids_t)
        .expect("moe up");
    ctx.compute(&up, 12).expect("compute moe up");
    check(s, "moe_up", up.to_vec_f32().iter().sum::<f32>());

    let up_c = ctx
        .clamp(&up, -SWIGLU_CLAMP_L0, SWIGLU_CLAMP_L0)
        .expect("clamp up");

    let act = ctx.swiglu_split(&gate_c, &up_c).expect("swiglu");
    ctx.compute(&act, 12).expect("compute swiglu");
    check(s, "moe_swiglu", act.to_vec_f32().iter().sum::<f32>());

    let down = ctx
        .mul_mat_id(&stack("ffn_down_exps"), &act, &ids_t)
        .expect("moe down");
    ctx.compute(&down, 12).expect("compute moe down");
    check(s, "moe_down", down.to_vec_f32().iter().sum::<f32>());

    // Each expert's output scaled by its router weight, then summed across the
    // six. llama.cpp does this as six strided views and five adds rather than a
    // reduction, so the same shape is used here.
    let weighted = ctx.mul(&down, &w_scaled).expect("weight experts");
    ctx.compute(&weighted, 12).expect("compute weighted");
    check(s, "moe_weighted", weighted.to_vec_f32().iter().sum::<f32>());

    let row = n_embd as usize * f32_size;
    let mut moe_out: Option<Tensor> = None;
    for (j, want) in s.weighted.iter().enumerate() {
        let v = ctx
            .view_2d(&weighted, n_embd, nt, row * n_used as usize, j * row)
            .expect("expert view");
        ctx.compute(&v, 12).expect("compute view");
        assert_sum(
            &format!("moe_weighted-{}[{j}]", s.il),
            v.to_vec_f32().iter().sum::<f32>(),
            *want,
        );
        moe_out = Some(match moe_out {
            None => v,
            Some(acc) => ctx.add(&acc, &v).expect("add expert"),
        });
    }
    let moe_out = moe_out.expect("six experts");
    ctx.compute(&moe_out, 12).expect("compute moe_out");
    check(s, "moe_out", moe_out.to_vec_f32().iter().sum::<f32>());

    // ---- the shared expert joins, and the layer closes ----
    let ffn_out = ctx.add(&moe_out, &shexp).expect("ffn_out");
    ctx.compute(&ffn_out, 12).expect("compute ffn_out");
    check(s, "ffn_out", ffn_out.to_vec_f32().iter().sum::<f32>());

    // The layer's second hyper-connection write-back, using the FFN block's
    // gates — not the attention block's.
    let l_last = ctx
        .dsv4_hc_post(&ffn_out, &streams, &gates.post, &gates.comb)
        .expect("dsv4_hc_post");
    ctx.compute(&l_last, 12).expect("compute l_last");
    check(s, "l_last", l_last.to_vec_f32().iter().sum::<f32>());

    // What layer 1 sees. Matching here means the whole of layer 0 is right,
    // since every earlier error would have to cancel exactly to arrive at it.
    let flat = ctx
        .reshape_2d(&l_last, config.hc_dim() as i64, nt)
        .expect("flatten");
    let normed = ctx.rms_norm(&flat, config.rms_eps).expect("rms_norm");
    ctx.compute(&normed, 12).expect("compute node_125");
    check_opt(s, "next_norm", normed.to_vec_f32().iter().sum::<f32>());

    l_last
}

/// **Layer 1 begins, which is the only evidence that layers compose.**
///
/// Oracle rows, from `v4flash-layer1-oracle-5tok.txt`:
/// ```text
/// hc_mixes-1     MUL_MAT      {24, 5}     -3428.892578
/// hc_pre-1       SCALE        {4, 5}          5.607770
/// hc_attn_pre-1  DSV4_HC_PRE  {4096, 5}      -0.132875
/// norm-1         RMS_NORM     {4096, 5}       7.388832
/// attn_norm-1    MUL          {4096, 5}      -0.242196
/// qr-1           MUL_MAT      {1024, 5}       6.841653
/// ```
///
/// Everything before this test verified *layer 0*. That is a weaker claim than
/// it sounds, for two reasons this test closes:
///
/// 1. **Composition.** Layer 1 has no embedding and no `hc_init` — it consumes
///    layer 0's `l_last-0` directly. Its first matmul is against `node_125`,
///    the RMS-norm of that. If layer 0's output were wrong anywhere, nothing
///    here could match, so `hc_mixes-1` is a single number standing in for the
///    correctness of the entire preceding layer.
/// 2. **The code is not fitted to layer 0's weights.** Layer 1 is the *second*
///    `Raw` layer — confirmed two ways, `compress_ratios` having exactly two
///    zeros and exactly two blocks carrying neither compressor nor indexer — so
///    it runs the same path with entirely different numbers. An implementation
///    that happened to suit layer 0 has nowhere to hide.
///
/// Layer 1's own `hc_attn_scale` is 0.090787 against layer 0's 2.076026, and
/// its gates sum to 5.607770 rather than 20.000015, so these really are
/// different weights and not the same rows under another name.
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn layers_compose_through_the_first_compressed_layer() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    // Two full layers in one arena. ggml ABORTS the process when its pool is
    // exhausted -- it does not return an error -- so this is sized up front,
    // and 512 MiB (enough for one layer) dies partway through layer 1's Q.
    let ctx = Context::new(1536 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(16 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    bind_all(&model, &wctx, &mut weights, &block_weights(0));
    bind_all(&model, &wctx, &mut weights, &optional_block_weights(&model, 0));

    bind_all(&model, &wctx, &mut weights, &block_weights(1));
    bind_all(&model, &wctx, &mut weights, &optional_block_weights(&model, 1));

    let l_last = layer0_5tok(&sums_5tok(0), &model, &ctx, &wctx, &mut weights, &config);

    // Layer 1 is Raw too, so it must take the uncompressed RoPE branch as
    // well. Asserted rather than assumed: it is the last layer for which that
    // is true, and every layer after it is compressed.
    assert!(
        !config.uses_compress_rope(1),
        "layer 1 is the second Raw layer and must be uncompressed"
    );
    assert_eq!(
        config.attention_kind_from_ratio(1),
        Some(bigtea_arch::AttentionKind::Raw)
    );

    let l_last = layer_5tok(&sums_5tok(1), &model, &ctx, &wctx, &mut weights, &config, &l_last);

    // Into the first *compressed* layer. Its attention is Compressed Sparse and
    // is not built — but a layer's entry does not depend on which attention
    // follows, so the seam is checkable now. When CSA is built, only the
    // attention itself will be new ground.
    assert_eq!(
        config.attention_kind_from_ratio(2),
        Some(bigtea_arch::AttentionKind::CompressedSparse),
        "layer 2 is where the compressed layers begin"
    );
    bind_all(&model, &wctx, &mut weights, &block_weights(2));
    bind_all(&model, &wctx, &mut weights, &optional_block_weights(&model, 2));
    let _ = layer_entry_5tok(&sums_5tok(2), &ctx, &weights, &config, &l_last);
}


/// One layer, in its **own context**, seeded from and returning plain `Vec<f32>`.
///
/// This is the shape that makes depth free. Chaining layers inside a single
/// `ggml` context costs ~640 MiB of arena each — four layers needed 2.5 GiB and
/// eight would have wanted more than this machine has. Here every layer builds
/// its arena and `WeightSet`, runs, hands out its residual streams as ordinary
/// floats, and drops the lot.
///
/// Freeing weights *without* the fresh context would not be safe: every
/// `compute` rebuilds the graph back through its sources, so a dropped weight
/// buffer leaves a dangling pointer that reads freed memory **successfully**,
/// yielding plausible numbers. Handing the boundary across as a `Vec` is what
/// makes the drop sound.
///
/// It is also what the streaming runner has to do, so this is not scaffolding.
fn layer_owned(
    s: &LayerSums,
    model: &Model,
    config: &Deepseek4Config,
    streams_in: Option<&[f32]>,
) -> Vec<f32> {
    let ctx = Context::new(1024 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(16 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    bind_all(model, &wctx, &mut weights, &block_weights(s.il));
    bind_all(model, &wctx, &mut weights, &optional_block_weights(model, s.il));

    let out = match streams_in {
        None => layer0_5tok(s, model, &ctx, &wctx, &mut weights, config),
        Some(v) => {
            let nt = s.tokens.len() as i64;
            let t = ctx
                .new_f32_3d(config.n_embd as i64, config.hc_mult as i64, nt)
                .expect("streams");
            t.set_f32(v).expect("fill streams");
            layer_5tok(s, model, &ctx, &wctx, &mut weights, config, &t)
        }
    };
    out.to_vec_f32()
}

/// **The whole model: all 43 blocks, at two tokens.**
///
/// Every block of DeepSeek-V4-Flash, from the embedding to the residual streams
/// `output_norm` would consume, checked against llama.cpp at roughly sixty
/// points per layer.
///
/// The reason a *short* prompt gets here is the guard on the compressed
/// attention builders (`deepseek4.cpp:1049-1063`): they need their compressed
/// caches populated, and at two tokens those caches are empty, so **every layer
/// falls through to `build_raw_attention`**. The compressor projections still
/// run; nothing reads them.
///
/// So this covers, on all 43 blocks: hyper-connections both halves, Q and KV,
/// RoPE — plain on layers 0-1 and **compressed/YaRN on 2-42** — fused attention
/// with sinks, de-rope, the grouped output projection, **both** MoE routing
/// schemes, the routed experts via partial reads, and the shared expert.
///
/// What it does **not** cover, and no amount of layers here would: the
/// compressors themselves, the lightning indexer, the sliding window (128,
/// never reached at this length) and the SwiGLU clamp bounds (never hit). Those
/// need longer prompts and code that does not exist yet.
#[test]
#[ignore = "reads weights from a 144 GB container, 43 layers"]
fn every_layer_runs_at_two_tokens() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    // The premise, asserted rather than assumed.
    assert!(!config.uses_compress_rope(1), "layers 0-1 are the plain-RoPE ones");
    assert!(config.uses_compress_rope(2), "layers 2+ take the YaRN branch");
    assert_eq!(config.hash_layer_count, 3, "only the first 3 blocks hash-route");

    let mut streams = layer_owned(&sums_2tok(0), &model, &config, None);
    for il in 1..config.n_layer {
        streams = layer_owned(&sums_2tok(il), &model, &config, Some(&streams));
        eprintln!("  ---- layer {il} of {} matches ----", config.n_layer - 1);
    }

    // The last block has no `next_norm`, so its `l_last` is the final check.
    let want = LayerSums::load(SUMS_2TOK, config.n_layer - 1, TOKENS_2).get("l_last");
    assert_sum("l_last (final block)", streams.iter().sum::<f32>(), want);

    // ...and then the head, which makes this a complete forward pass: from a
    // token id to a logit per vocabulary entry.
    let head = LayerSums::load(SUMS_2TOK, 43, TOKENS_2);
    let logits = output_head(&head, &model, &config, &streams);

    // The argmax is the token this model would actually emit. Not in the
    // trace — llama.cpp prints sums, not picks — so it is reported rather than
    // asserted, and it is the first end-to-end *output* this port has produced.
    let best = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .expect("non-empty logits");
    // Decoded, because an id is not evidence and a word is. If the whole
    // 43-layer stack were subtly wrong this would still be *a* token.
    let tok = bigtea_tokenizer::Tokenizer::from_metadata(model.metadata()).ok();
    let text = tok
        .as_ref()
        .and_then(|t| t.token_text(best.0 as u32))
        .unwrap_or("<undecodable>")
        .to_string();
    eprintln!(
        "  {:<24} id {} = {:?} at {:.4}   (prompt: {:?})",
        "argmax", best.0, text, best.1, "Hello there"
    );
}


/// **The output head**, which turns the last block's streams into logits.
///
/// Oracle rows (pseudo-layer 43 in the fixture):
/// ```text
/// node_8112       GET_ROWS  {16384, 1}       -1219.583618
/// hc_head_mixes   MUL_MAT   {4, 1}               1.332416
/// hc_head         ADD       {4096, 1}         -162.383377
/// result_norm     MUL       {4096, 1}          -22.952126
/// result_output   MUL_MAT   {129280, 1}     437389.187500
/// ```
///
/// **Only the last token reaches here.** `inp_out_ids` selects one row before
/// anything else runs, so the head is single-token whatever the prompt length —
/// which is why generating a token costs one 129280-wide matmul and not `nt` of
/// them.
///
/// The head has a hyper-connection collapse of its own, with its own
/// `output_hc_*` weights: the four residual streams are still four here and
/// something has to fold them into one. llama.cpp writes that out as four
/// multiplies and three adds rather than calling the fused op; `dsv4_hc_pre`
/// computes the same thing and is used here.
fn output_head(
    s: &LayerSums,
    model: &Model,
    config: &Deepseek4Config,
    streams: &[f32],
) -> Vec<f32> {
    let ctx = Context::new(1024 << 20).expect("head context");
    let wctx = Context::new_no_alloc(8 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    bind_all(
        model,
        &wctx,
        &mut weights,
        &[
            "output_hc_fn.weight".to_string(),
            "output_hc_scale.weight".to_string(),
            "output_hc_base.weight".to_string(),
            "output_norm.weight".to_string(),
            "output.weight".to_string(),
        ],
    );

    let hc = config.hc_mult as i64;
    let n_embd = config.n_embd as i64;
    let hc_dim = config.hc_dim() as i64;
    let nt = s.tokens.len();

    // inp_out_ids: the last token's streams, and only those.
    let last = &streams[(nt - 1) * hc_dim as usize..];
    let x = ctx.new_f32_3d(n_embd, hc, 1).expect("head streams");
    x.set_f32(last).expect("fill head streams");
    check(s, "head_select", last.iter().sum::<f32>());

    let flat = ctx.reshape_2d(&x, hc_dim, 1).expect("flatten");
    let normed = ctx.rms_norm(&flat, config.rms_eps).expect("head rms");
    ctx.compute(&normed, 12).expect("compute head norm");
    check(s, "head_norm", normed.to_vec_f32().iter().sum::<f32>());

    let mixes = ctx
        .mul_mat(weights.get("output_hc_fn.weight").expect("bound"), &normed)
        .expect("head mixes");
    ctx.compute(&mixes, 12).expect("compute head mixes");
    check(s, "head_mixes", mixes.to_vec_f32().iter().sum::<f32>());

    // The head's gate block is the `pre` half only — there is no `post` here,
    // because nothing writes back into the streams after this.
    let f32_size = std::mem::size_of::<f32>();
    let scale = ctx
        .view_1d(weights.get("output_hc_scale.weight").expect("bound"), 1, 0)
        .expect("head scale");
    let base = ctx
        .view_1d(weights.get("output_hc_base.weight").expect("bound"), hc, 0)
        .expect("head base");
    let _ = f32_size;

    let scaled = ctx.mul(&mixes, &scale).expect("head mul");
    ctx.compute(&scaled, 12).expect("compute");
    check(s, "head_mul", scaled.to_vec_f32().iter().sum::<f32>());

    let biased = ctx.add(&scaled, &base).expect("head add");
    ctx.compute(&biased, 12).expect("compute");
    check(s, "head_add", biased.to_vec_f32().iter().sum::<f32>());

    let gated = ctx.sigmoid(&biased).expect("head sigmoid");
    ctx.compute(&gated, 12).expect("compute");
    check(s, "head_sigmoid", gated.to_vec_f32().iter().sum::<f32>());

    let eps_t = ctx.new_f32_1d(hc).expect("eps");
    eps_t.set_f32(&vec![1e-6f32; hc as usize]).expect("fill eps");
    let pre = ctx.add(&gated, &eps_t).expect("head pre");
    ctx.compute(&pre, 12).expect("compute");
    check(s, "head_pre", pre.to_vec_f32().iter().sum::<f32>());

    let collapsed = ctx.dsv4_hc_pre(&x, &pre).expect("head hc_pre");
    ctx.compute(&collapsed, 12).expect("compute head hc");
    check(s, "head_hc", collapsed.to_vec_f32().iter().sum::<f32>());

    let normed = ctx.rms_norm(&collapsed, config.rms_eps).expect("final rms");
    ctx.compute(&normed, 12).expect("compute");
    check(s, "head_rms", normed.to_vec_f32().iter().sum::<f32>());

    let result_norm = ctx
        .mul(&normed, weights.get("output_norm.weight").expect("bound"))
        .expect("result_norm");
    ctx.compute(&result_norm, 12).expect("compute");
    check(s, "result_norm", result_norm.to_vec_f32().iter().sum::<f32>());

    let logits = ctx
        .mul_mat(weights.get("output.weight").expect("bound"), &result_norm)
        .expect("result_output");
    ctx.compute(&logits, 12).expect("compute logits");
    let out = logits.to_vec_f32();
    assert_eq!(out.len(), config.vocab_size as usize, "one logit per token id");
    check(s, "result_output", out.iter().sum::<f32>());
    out
}

/// **The CSA compressor**, for a prefill from an empty cache.
///
/// Every compressed-sparse layer keeps a running summary of the raw KV it has
/// seen: one entry per completed block of `ratio` positions, each a
/// score-weighted average over `2*ratio` raw positions — two windows, which is
/// what "overlap" means. Attention then attends to the raw window *and* these
/// summaries.
///
/// # Why the persistent cache is not needed here
///
/// llama.cpp reads this state through index tensors computed in
/// `llama-kv-cache-dsv4.cpp` (1978 lines). But `state_source_idx` resolves to an
/// appended zero row when `pos < 0` and to the **current ubatch** otherwise, so
/// on a prefill from empty the persistent ring is never read. The indices are
/// then constructible directly, which is what this does. Generation, where
/// earlier ubatches are read back, will need the real ring.
///
/// At five tokens with `ratio = 4`: one completed block, so `n_blocks = 1`, the
/// previous window is four copies of the zero row and the current window is
/// positions 0-3. Position 4 belongs to no completed block yet.
///
/// # The shape that hides a bug
///
/// The state is `2*n_embd_head` wide — **two entries per row**. `kv_prev` reads
/// the first 512 of one set of rows and `kv_cur` the *second* 512 of the next
/// set. Reading it as one entry per row gives a correctly-shaped compressor
/// summarising the wrong span, with no error.
fn overlap_compressor_5tok<'c>(
    s: &LayerSums,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    attn_norm: &Tensor<'c>,
    // head: 512 for the attention compressor, 128 for the indexer's.
    head: i64,
    // wp: "attn_compressor" or "indexer_compressor".
    wp: &str,
    // lp: "csa" or "lid", the label prefix in the fixture.
    lp: &str,
) -> Tensor<'c> {
    let il = s.il;
    let nt = s.tokens.len() as i64;
    let ratio = Deepseek4Config::CSA_RATIO;
    let wide = 2 * head; // the state's row width: two entries per row
    let n_blocks = nt / ratio;
    assert!(n_blocks > 0, "no block completes at {nt} tokens");
    let n_read = ratio * n_blocks;
    // The persistent CSA state, all zeros on a prefill. Its height only has to
    // be large enough for the indices below to land inside it.
    let state_rows = 8i64;

    let kv = ctx
        .mul_mat(
            weights
                .get(&format!("blk.{il}.{wp}_kv.weight"))
                .expect("bound"),
            attn_norm,
        )
        .expect("csa_state_kv");
    ctx.compute(&kv, 12).expect("compute");
    check(s, &format!("{lp}_state_kv"), kv.to_vec_f32().iter().sum::<f32>());

    let score = ctx
        .mul_mat(
            weights
                .get(&format!("blk.{il}.{wp}_gate.weight"))
                .expect("bound"),
            attn_norm,
        )
        .expect("csa_state_score");
    ctx.compute(&score, 12).expect("compute");
    check(s, &format!("{lp}_state_score"), score.to_vec_f32().iter().sum::<f32>());

    // The gate gets an absolute-position embedding indexed by the token's
    // offset *within its block*, not by its absolute position.
    let state_pos: Vec<i32> = (0..nt).map(|p| (p % ratio) as i32).collect();
    let pos_t = ctx.new_i32_1d(nt).expect("state_pos");
    pos_t.set_i32(&state_pos).expect("set");
    let ape = ctx
        .get_rows(
            weights
                .get(&format!("blk.{il}.{wp}_ape.weight"))
                .expect("bound"),
            &pos_t,
        )
        .expect("ape rows");
    ctx.compute(&ape, 12).expect("compute");
    check(s, &format!("{lp}_ape_rows"), ape.to_vec_f32().iter().sum::<f32>());

    let score = ctx.add(&score, &ape).expect("score + ape");
    ctx.compute(&score, 12).expect("compute");
    check(s, &format!("{lp}_state_score_ape"), score.to_vec_f32().iter().sum::<f32>());

    // Assemble the state as llama.cpp's graph does: [empty ring | this ubatch |
    // one appended row]. The appended row is zero for values and -inf for
    // scores, so the softmax below ignores the padding rather than averaging
    // it in.
    let total = state_rows + nt + 1;
    let kv_state = {
        let mut v = vec![0.0f32; (state_rows * wide) as usize];
        v.extend_from_slice(&kv.to_vec_f32());
        v.extend(std::iter::repeat(0.0f32).take(wide as usize));
        let t = ctx.new_f32_2d(wide, total).expect("kv state");
        t.set_f32(&v).expect("fill kv state");
        t
    };
    let score_state = {
        let mut v = vec![0.0f32; (state_rows * wide) as usize];
        v.extend_from_slice(&score.to_vec_f32());
        v.extend(std::iter::repeat(f32::NEG_INFINITY).take(wide as usize));
        let t = ctx.new_f32_2d(wide, total).expect("score state");
        t.set_f32(&v).expect("fill score state");
        t
    };

    // The read indices: every block's previous window first, then every
    // block's current window. A negative position means the zero row.
    let zero_row = (state_rows + nt) as i32;
    let mut idxs: Vec<i32> = Vec::with_capacity((2 * n_read) as usize);
    for b in 0..n_blocks {
        let start = b * ratio - ratio;
        for j in 0..ratio {
            let p = start + j;
            idxs.push(if p < 0 { zero_row } else { (state_rows + p) as i32 });
        }
    }
    for b in 0..n_blocks {
        let start = b * ratio;
        for j in 0..ratio {
            idxs.push((state_rows + start + j) as i32);
        }
    }
    let idx_t = ctx.new_i32_1d(2 * n_read).expect("idxs");
    idx_t.set_i32(&idxs).expect("set idxs");

    let f32_size = std::mem::size_of::<f32>();
    let row = wide as usize * f32_size;

    let split = |src: &Tensor<'c>, is_kv: bool| -> Tensor<'c> {
        let rows = ctx.get_rows(src, &idx_t).expect("gather");
        ctx.compute(&rows, 12).expect("compute");
        if is_kv {
            check(s, &format!("{lp}_gathered"), rows.to_vec_f32().iter().sum::<f32>());
        }
        // First 512 of the first n_read rows; second 512 of the next n_read.
        let prev = ctx
            .cont(&ctx.view_2d(&rows, head, n_read, row, 0).expect("prev view"))
            .expect("prev");
        let cur = ctx
            .cont(
                &ctx.view_2d(
                    &rows,
                    head,
                    n_read,
                    row,
                    n_read as usize * row + head as usize * f32_size,
                )
                .expect("cur view"),
            )
            .expect("cur");
        let prev = ctx.reshape_3d(&prev, head, ratio, n_blocks).expect("prev 3d");
        let cur = ctx.reshape_3d(&cur, head, ratio, n_blocks).expect("cur 3d");
        if is_kv {
            ctx.compute(&prev, 12).expect("compute");
            check(s, &format!("{lp}_kv_prev"), prev.to_vec_f32().iter().sum::<f32>());
            ctx.compute(&cur, 12).expect("compute");
            check(s, &format!("{lp}_kv_cur"), cur.to_vec_f32().iter().sum::<f32>());
        }
        let joined = ctx.concat(&prev, &cur, 1).expect("concat windows");
        ctx.cont(&ctx.permute(&joined, [1, 0, 2, 3]).expect("permute"))
            .expect("cont")
    };

    let values = split(&kv_state, true);
    ctx.compute(&values, 12).expect("compute values");
    check(s, &format!("{lp}_values_perm"), values.to_vec_f32().iter().sum::<f32>());
    let scores = split(&score_state, false);

    let w = ctx.soft_max(&scores).expect("softmax scores");
    ctx.compute(&w, 12).expect("compute weights");
    check(s, &format!("{lp}_comp_weights"), w.to_vec_f32().iter().sum::<f32>());

    let weighted = ctx.mul(&values, &w).expect("weight values");
    ctx.compute(&weighted, 12).expect("compute weighted");
    check(s, &format!("{lp}_weighted"), weighted.to_vec_f32().iter().sum::<f32>());

    let summed = ctx.sum_rows(&weighted).expect("sum_rows");
    ctx.compute(&summed, 12).expect("compute summed");
    check(s, &format!("{lp}_summed"), summed.to_vec_f32().iter().sum::<f32>());

    let comp = ctx
        .cont(&ctx.permute(&summed, [1, 0, 2, 3]).expect("permute back"))
        .expect("cont");
    ctx.compute(&comp, 12).expect("compute comp");
    check(s, &format!("{lp}_comp_raw"), comp.to_vec_f32().iter().sum::<f32>());

    let normed = ctx.rms_norm(&comp, config.rms_eps).expect("comp rms");
    ctx.compute(&normed, 12).expect("compute");
    check(s, &format!("{lp}_comp_rms"), normed.to_vec_f32().iter().sum::<f32>());

    let comp = ctx
        .mul(
            &normed,
            weights
                .get(&format!("blk.{il}.{wp}_norm.weight"))
                .expect("bound"),
        )
        .expect("comp norm");
    ctx.compute(&comp, 12).expect("compute");
    check(s, &format!("{lp}_comp_normed"), comp.to_vec_f32().iter().sum::<f32>());

    // Rotated at the *block* position, which is 0 for the first block — so at
    // five tokens this rotation is the identity and is NOT verified here. Same
    // shape of hole as the one-token capture had for the main RoPE.
    let n_rot = config.n_rot as i64;
    // Relative to THIS compressor's head, not the attention head:
    // Deepseek4Config::n_rot_none() is 512-64, and the indexer's is 128-64.
    let n_nope = head - n_rot;
    let head_stride = head as usize * f32_size;
    let nope = ctx
        .view_3d(&comp, n_nope, 1, n_blocks, head_stride, head_stride, 0)
        .expect("comp nope");
    ctx.compute(&nope, 12).expect("compute");
    check(s, &format!("{lp}_comp_nope"), nope.to_vec_f32().iter().sum::<f32>());

    let pe_in = ctx
        .view_3d(
            &comp,
            n_rot,
            1,
            n_blocks,
            head_stride,
            head_stride,
            n_nope as usize * f32_size,
        )
        .expect("comp pe view");
    ctx.compute(&pe_in, 12).expect("compute");
    check(s, &format!("{lp}_comp_pe_in"), pe_in.to_vec_f32().iter().sum::<f32>());

    let comp_pos = ctx.new_i32_1d(n_blocks).expect("comp_pos");
    let block_pos: Vec<i32> = (0..n_blocks).map(|b| (b * ratio) as i32).collect();
    comp_pos.set_i32(&block_pos).expect("set comp_pos");
    let (rope, rope_orig) = rope_for(config, il);
    let pe = ctx
        .rope_ext(
            &pe_in,
            &comp_pos,
            None,
            n_rot as i32,
            ROPE_MODE_NORM,
            rope_orig,
            rope,
        )
        .expect("comp rope");
    ctx.compute(&pe, 12).expect("compute");
    check(s, &format!("{lp}_comp_pe"), pe.to_vec_f32().iter().sum::<f32>());

    let out = ctx.concat(&nope, &pe, 0).expect("concat comp");
    ctx.compute(&out, 12).expect("compute comp out");
    check(s, &format!("{lp}_comp"), out.to_vec_f32().iter().sum::<f32>());
    out
}

/// The orthonormal Walsh-Hadamard rotation the lightning indexer runs its keys
/// and queries through.
///
/// Sylvester's construction scaled by `1/sqrt(n)`, transcribed from
/// `llama-kv-cache.cpp:22`. It is its own inverse (`H² = I`), which is why the
/// same matrix un-rotates the attention output.
///
/// **It is generated, not stored.** Nothing in the container holds it — a port
/// that goes looking for a `*_rot` tensor will not find one and may conclude
/// the rotation is optional. llama.cpp builds it at cache-init time for
/// DeepSeek indexers unconditionally (`llama-kv-cache.cpp:352`).
fn walsh_hadamard(n: usize) -> Vec<f32> {
    assert!(n.is_power_of_two(), "Walsh-Hadamard needs a power of two");
    let mut m = vec![0.0f32; n * n];
    m[0] = 1.0 / (n as f32).sqrt();
    let mut s = 1usize;
    while s < n {
        for i in 0..s {
            for j in 0..s {
                let v = m[i * n + j];
                m[(i + s) * n + j] = v;
                m[i * n + (j + s)] = v;
                m[(i + s) * n + (j + s)] = -v;
            }
        }
        s *= 2;
    }
    m
}

/// Apply the Hadamard rotation: reshape to `[n, ..]`, matmul, reshape back.
///
/// Mirrors `llama_mul_mat_hadamard` (`llama-impl.h:57`).
fn hadamard_rotate<'c>(ctx: &'c Context, x: &Tensor<'c>, rot: &Tensor<'c>, n: i64) -> Tensor<'c> {
    let total = x.len();
    let flat = if x.is_contiguous() {
        ctx.reshape_2d(x, n, total / n).expect("reshape for hadamard")
    } else {
        let c = ctx.cont(x).expect("cont for hadamard");
        ctx.reshape_2d(&c, n, total / n).expect("reshape for hadamard")
    };
    ctx.mul_mat(rot, &flat).expect("hadamard matmul")
}

/// The lightning indexer's **query path**: its own low-rank Q, its own RoPE,
/// and the Hadamard rotation.
///
/// Returns `(q_rot, weights)` — the rotated indexer queries shaped
/// `[indexer_head, n_indexer_head, tokens]` and the per-head scores scale that
/// `ggml_lightning_indexer` multiplies by.
///
/// Three things here are not the attention path's, despite looking like it:
///
/// 1. **It reuses `qr`**, the shared Q down-projection, but has its own up
///    projection (`indexer.attn_q_b`) into a 128-wide head, not 512.
/// 2. **Its RoPE always uses the compressed base**, unconditionally
///    (`deepseek4.cpp:555`) — not the per-layer choice the attention path
///    makes. That happens to agree here because every layer with an indexer is
///    a compressed layer, but the code says "always", not "per layer".
/// 3. **The rotation is real at this length.** Unlike the compressor's, whose
///    `comp_pos` is 0 for the first block, this one rotates by token position:
///    21.281902 goes to 17.747589.
fn lid_query_5tok<'c>(
    s: &LayerSums,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    attn_norm: &Tensor<'c>,
    rot: &Tensor<'c>,
) -> (Tensor<'c>, Tensor<'c>) {
    let il = s.il;
    let nt = s.tokens.len() as i64;
    let head = config.indexer_key_length as i64;
    let n_head = config.indexer_head_count as i64;
    let n_rot = config.n_rot as i64;
    let n_nope = head - n_rot;
    let f32_size = std::mem::size_of::<f32>();
    let head_stride = head as usize * f32_size;

    // The shared Q down-projection, which the indexer borrows.
    let qr = ctx
        .mul_mat(
            weights.get(&format!("blk.{il}.attn_q_a.weight")).expect("bound"),
            attn_norm,
        )
        .expect("qr");
    let qr = ctx.rms_norm(&qr, config.rms_eps).expect("qr rms");
    let qr = ctx
        .mul(
            &qr,
            weights.get(&format!("blk.{il}.attn_q_a_norm.weight")).expect("bound"),
        )
        .expect("qr_norm");

    let q = ctx
        .mul_mat(
            weights.get(&format!("blk.{il}.indexer.attn_q_b.weight")).expect("bound"),
            &qr,
        )
        .expect("lid_q");
    let q = ctx.reshape_3d(&q, head, n_head, nt).expect("reshape lid_q");
    ctx.compute(&q, 12).expect("compute lid_q");
    check(s, "lid_q", q.to_vec_f32().iter().sum::<f32>());

    let nope = ctx
        .view_3d(&q, n_nope, n_head, nt, head_stride, head_stride * n_head as usize, 0)
        .expect("lid_q nope");
    ctx.compute(&nope, 12).expect("compute");
    check(s, "lid_q_nope", nope.to_vec_f32().iter().sum::<f32>());

    let pe_in = ctx
        .view_3d(
            &q,
            n_rot,
            n_head,
            nt,
            head_stride,
            head_stride * n_head as usize,
            n_nope as usize * f32_size,
        )
        .expect("lid_q pe view");
    ctx.compute(&pe_in, 12).expect("compute");
    check(s, "lid_q_pe_in", pe_in.to_vec_f32().iter().sum::<f32>());

    let pos = ctx.new_i32_1d(nt).expect("pos");
    let positions: Vec<i32> = (0..nt as i32).collect();
    pos.set_i32(&positions).expect("set pos");
    // The compressed parameters, always — see the note above.
    let (rope, rope_orig) = rope_for(config, il);
    let pe = ctx
        .rope_ext(&pe_in, &pos, None, n_rot as i32, ROPE_MODE_NORM, rope_orig, rope)
        .expect("lid_q_pe");
    ctx.compute(&pe, 12).expect("compute");
    check(s, "lid_q_pe", pe.to_vec_f32().iter().sum::<f32>());

    let q = ctx.concat(&nope, &pe, 0).expect("concat lid_q");
    let q_rot = hadamard_rotate(ctx, &q, rot, head);
    let q_rot = ctx.reshape_3d(&q_rot, head, n_head, nt).expect("reshape q_rot");
    ctx.compute(&q_rot, 12).expect("compute q_rot");
    check(s, "lid_q_rot", q_rot.to_vec_f32().iter().sum::<f32>());

    // One score weight per indexer head, scaled by the geometric mean of the
    // indexer's two dimensions rather than by the head width alone.
    let w = ctx
        .mul_mat(
            weights.get(&format!("blk.{il}.indexer.proj.weight")).expect("bound"),
            attn_norm,
        )
        .expect("lid_weights");
    let scale = 1.0f32 / ((head * n_head) as f32).sqrt();
    let w = ctx.scale(&w, scale).expect("scale lid_weights");
    ctx.compute(&w, 12).expect("compute lid_weights");
    check(s, "lid_weights", w.to_vec_f32().iter().sum::<f32>());

    (q_rot, w)
}

/// Both compressors of a Compressed Sparse layer, and the indexer's rotation.
///
/// A CSA layer runs the overlap compressor **twice** with different weights and
/// different widths: once at 512 for attention, once at 128 for the lightning
/// indexer. The indexer's output is then put through the Walsh-Hadamard
/// rotation; the attention one is not.
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn csa_compressor_matches_llama_cpp() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    let ctx = Context::new(2048 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(32 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    for il in 0..3u32 {
        bind_all(&model, &wctx, &mut weights, &block_weights(il));
        bind_all(&model, &wctx, &mut weights, &optional_block_weights(&model, il));
    }
    let comp: Vec<String> = [
        "attn_compressor_kv",
        "attn_compressor_gate",
        "attn_compressor_ape",
        "attn_compressor_norm",
        "indexer_compressor_kv",
        "indexer_compressor_gate",
        "indexer_compressor_ape",
        "indexer_compressor_norm",
        "indexer.attn_q_b",
        "indexer.proj",
    ]
    .iter()
    .map(|x| format!("blk.2.{x}.weight"))
    .collect();
    bind_all(&model, &wctx, &mut weights, &comp);

    let l0 = layer0_5tok(&sums_5tok(0), &model, &ctx, &wctx, &mut weights, &config);
    let l1 = layer_5tok(&sums_5tok(1), &model, &ctx, &wctx, &mut weights, &config, &l0);

    let s = sums_5tok(2);
    let p = layer_entry_5tok(&s, &ctx, &weights, &config, &l1);

    // The attention compressor, at the full head width.
    let _csa = overlap_compressor_5tok(
        &s,
        &ctx,
        &weights,
        &config,
        &p.attn_norm,
        config.kv_lora_rank as i64,
        "attn_compressor",
        "csa",
    );

    // The indexer's own compressor, at its narrower head.
    let head_lid = config.indexer_key_length as i64;
    let lid = overlap_compressor_5tok(
        &s,
        &ctx,
        &weights,
        &config,
        &p.attn_norm,
        head_lid,
        "indexer_compressor",
        "lid",
    );

    // ...then the Hadamard, which only the indexer applies.
    let rot = ctx
        .new_f32_2d(head_lid, head_lid)
        .expect("hadamard");
    rot.set_f32(&walsh_hadamard(head_lid as usize)).expect("fill hadamard");
    let rotated = hadamard_rotate(&ctx, &lid, &rot, head_lid);
    ctx.compute(&rotated, 12).expect("compute hadamard");
    check(&s, "lid_comp_rot", rotated.to_vec_f32().iter().sum::<f32>());

    // The indexer's query side, which shares the rotation but not much else.
    let _ = lid_query_5tok(&s, &ctx, &weights, &config, &p.attn_norm, &rot);
}

/// Read just the named experts out of a stacked tensor and bind them as a
/// compact stack, returning `(bytes read, the compact dims)`.
///
/// A stacked expert tensor is `[ne0, ne1, n_expert]` with every slice the same
/// size, so slice `i` starts at `i * size / n_expert` — the same arithmetic
/// `stream.rs` uses to fetch one expert.
fn bind_expert_slices<'c>(
    model: &Model,
    ctx: &'c Context,
    weights: &mut WeightSet<'c>,
    name: &str,
    unique: &[i32],
) -> (u64, Vec<u64>) {
    let loc = model.location(name).unwrap_or_else(|| panic!("{name} present")).clone();
    let n_expert = *loc.dims.last().expect("stacked tensor");
    let slice = loc.size / n_expert;

    let mut buf = Vec::with_capacity(unique.len() * slice as usize);
    for e in unique {
        let bytes = model
            .read_tensor_range(name, *e as u64 * slice, slice)
            .expect("read expert slice");
        buf.extend_from_slice(&bytes);
    }

    let mut dims = loc.dims.clone();
    *dims.last_mut().expect("stacked") = unique.len() as u64;
    let read = buf.len() as u64;
    weights.bind(ctx, name, loc.ty, &dims, buf).expect("bind experts");
    (read, dims)
}

/// The shared expert: always active, and therefore resident weight.
///
/// Confusing it with the 256 routed ones is the difference between a 7 GiB
/// resident set and a 144 GiB one.
fn shared_expert_5tok<'c>(
    s: &LayerSums,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    ffn_norm: &Tensor<'c>,
) -> Tensor<'c> {
    let gate = ctx
        .mul_mat(weights.get(&format!("blk.{}.ffn_gate_shexp.weight", s.il)).expect("bound"), ffn_norm)
        .expect("gate");
    ctx.compute(&gate, 12).expect("compute gate");
    check(s, "shexp_gate", gate.to_vec_f32().iter().sum::<f32>());

    // Asymmetric on purpose: upper bound only.
    let gate_c = ctx
        .clamp(&gate, f32::NEG_INFINITY, SWIGLU_CLAMP_L0)
        .expect("clamp gate");
    ctx.compute(&gate_c, 12).expect("compute gate clamped");
    check(s, "shexp_gate_clamped", gate_c.to_vec_f32().iter().sum::<f32>());

    let up = ctx
        .mul_mat(weights.get(&format!("blk.{}.ffn_up_shexp.weight", s.il)).expect("bound"), ffn_norm)
        .expect("up");
    ctx.compute(&up, 12).expect("compute up");
    check(s, "shexp_up", up.to_vec_f32().iter().sum::<f32>());

    let up_c = ctx
        .clamp(&up, -SWIGLU_CLAMP_L0, SWIGLU_CLAMP_L0)
        .expect("clamp up");
    ctx.compute(&up_c, 12).expect("compute up clamped");
    check(s, "shexp_up_clamped", up_c.to_vec_f32().iter().sum::<f32>());

    let act = ctx.swiglu_split(&gate_c, &up_c).expect("swiglu");
    ctx.compute(&act, 12).expect("compute swiglu");
    check(s, "shexp_swiglu", act.to_vec_f32().iter().sum::<f32>());

    let shexp = ctx
        .mul_mat(weights.get(&format!("blk.{}.ffn_down_shexp.weight", s.il)).expect("bound"), &act)
        .expect("shexp");
    ctx.compute(&shexp, 12).expect("compute shexp");
    check(s, "shexp", shexp.to_vec_f32().iter().sum::<f32>());

    shexp
}

/// Post hyper-connection, the FFN gate block, and `ffn_norm`.
///
/// Returns `(streams, ffn_norm)`: the updated 4-stream residual, which the
/// layer's second `hc_post` will need, and the normalised input the MoE and the
/// shared expert both consume.
fn layer_tail_5tok<'c>(
    s: &LayerSums,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    p: &Prologue5<'c>,
    attn_out: &Tensor<'c>,
) -> (Tensor<'c>, Tensor<'c>, HcGates<'c>) {
    let nt = s.tokens.len() as i64;

    // x = attn_out, residual = the streams as they were *before* attention.
    let streams = ctx
        .dsv4_hc_post(attn_out, &p.hc_init, &p.gates.post, &p.gates.comb)
        .expect("dsv4_hc_post");
    ctx.compute(&streams, 12).expect("compute hc_post");
    check(s, "hc_attn_post", streams.to_vec_f32().iter().sum::<f32>());

    // The FFN's gates come from their own matmul over the post-attention
    // stream, against hc_ffn_fn rather than hc_attn_fn.
    let flat = ctx
        .reshape_2d(&streams, config.hc_dim() as i64, nt)
        .expect("flatten streams");
    let normed = ctx.rms_norm(&flat, config.rms_eps).expect("rms_norm");
    ctx.compute(&normed, 12).expect("compute norm");
    check(s, "post_norm", normed.to_vec_f32().iter().sum::<f32>());

    let mixes = ctx
        .mul_mat(weights.get(&format!("blk.{}.hc_ffn_fn.weight", s.il)).expect("bound"), &normed)
        .expect("hc_mixes ffn");
    ctx.compute(&mixes, 12).expect("compute mixes");
    check(s, "hc_mixes_ffn", mixes.to_vec_f32().iter().sum::<f32>());

    let gates = hc_gates(ctx, weights, config, &format!("blk.{}.hc_ffn", s.il), &mixes, s.ffn_gates, nt);

    let collapsed = ctx.dsv4_hc_pre(&streams, &gates.pre).expect("dsv4_hc_pre");
    ctx.compute(&collapsed, 12).expect("compute hc_ffn_pre");
    check(s, "hc_ffn_pre", collapsed.to_vec_f32().iter().sum::<f32>());

    let normed = ctx.rms_norm(&collapsed, config.rms_eps).expect("norm");
    ctx.compute(&normed, 12).expect("compute");
    check(s, "norm_ffn", normed.to_vec_f32().iter().sum::<f32>());

    let ffn_norm = ctx
        .mul(&normed, weights.get(&format!("blk.{}.ffn_norm.weight", s.il)).expect("bound"))
        .expect("ffn_norm");
    ctx.compute(&ffn_norm, 12).expect("compute");
    check(s, "ffn_norm", ffn_norm.to_vec_f32().iter().sum::<f32>());

    (streams, ffn_norm, gates)
}

/// Attention through to `attn_out`, returning the block's output.
///
/// Shared with the layer-tail test, which needs `attn_out` to feed the post
/// hyper-connection.
fn attention_5tok<'c>(
    s: &LayerSums,
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    q_full: &Tensor<'c>,
    kv_full: &Tensor<'c>,
) -> Tensor<'c> {
    let nt = s.tokens.len() as i64;
    let head_dim = config.kv_lora_rank as i64;
    let n_head = config.n_head as i64;
    // The padded cache window llama.cpp's trace shows: cache_k_l0 is viewed as
    // {512, 1, 256} and permuted to {512, 256, 1}.
    const N_KV: i64 = 256;

    let sinks = weights.get(&format!("blk.{}.attn_sinks.weight", s.il)).expect("bound");
    assert_eq!(sinks.len(), n_head, "one sink per head");

    // ---- the K cache, in F16 ----
    // Positions 0..4 hold the compressed KV; 5..255 stay zero and are masked.
    let kv_values = kv_full.to_vec_f32();
    let mut cache_f16 = vec![0u16; (head_dim * N_KV) as usize];
    bigtea_ggml::f32_to_f16(&kv_values, &mut cache_f16[..kv_values.len()]);

    // Check the cache against llama.cpp *after* rounding, which is the number
    // its trace reports.
    let mut round_tripped = vec![0f32; cache_f16.len()];
    bigtea_ggml::f16_to_f32(&cache_f16, &mut round_tripped);
    check(s, "cache_k", round_tripped.iter().sum::<f32>());

    let k = ctx.new_f16_3d(head_dim, N_KV, 1).expect("k cache");
    let bytes: Vec<u8> = cache_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    k.set_bytes(&bytes).expect("fill k");

    // ---- the mask ----
    // [n_kv, n_tokens], F16 and contiguous — ggml asserts both. Only 0 and -inf
    // occur, so the bit patterns go in directly: -inf must stay exactly -inf.
    const F16_NEG_INF: [u8; 2] = [0x00, 0xFC];
    let mut mask_bytes = vec![0u8; (N_KV * nt) as usize * 2];
    for query in 0..nt {
        let row = (query * N_KV) as usize * 2;
        // Everything strictly after this query's own position, which includes
        // all 251 unused cache slots.
        for key in (query + 1)..N_KV {
            let at = row + key as usize * 2;
            mask_bytes[at..at + 2].copy_from_slice(&F16_NEG_INF);
        }
    }
    let mask = ctx
        .new_typed_2d(bigtea_gguf::GgmlType(1), N_KV, nt)
        .expect("mask");
    mask.set_bytes(&mask_bytes).expect("fill mask");

    // ---- the fused kernel ----
    // q arrives as [head_dim, n_head, tokens] and the kernel wants
    // [head_dim, tokens, n_head], so dims 1 and 2 swap.
    let q_perm = ctx.permute(q_full, [0, 2, 1, 3]).expect("permute q");
    ctx.compute(&q_perm, 12).expect("compute q_perm");
    check(s, "q_perm", q_perm.to_vec_f32().iter().sum::<f32>());

    // scale is 1/sqrt(n_embd_head) over the *full* 512, not the 448 unrotated
    // dims (deepseek4.cpp:1063).
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let out = ctx
        .flash_attn_ext_with_sinks(&q_perm, &k, &k, &mask, sinks, scale)
        .expect("flash_attn_ext");
    ctx.compute(&out, 12).expect("compute attention");
    check(s, "flash_attn", out.to_vec_f32().iter().sum::<f32>());

    // Prove the sinks are load-bearing rather than assume it. `add_sinks`
    // mutates a node and returns nothing, so a binding that silently did
    // nothing would leave every assertion above still passing — the same shape
    // of hole the one-token RoPE capture had. Running the kernel without them
    // must give a different number.
    let no_sinks = ctx
        .flash_attn_ext(&q_perm, &k, &k, &mask, scale)
        .expect("flash_attn_ext without sinks");
    ctx.compute(&no_sinks, 12).expect("compute attention");
    let without: f32 = no_sinks.to_vec_f32().iter().sum();
    assert!(
        (without - 2879.606934).abs() > 1.0,
        "attention without sinks gave {without:.6}, which matches the reference \
         anyway — the sinks are not reaching the kernel"
    );
    eprintln!("  {:<24} {:>14.6}  (differs, as it must)", "without sinks", without);

    // ---- de-rope, then the grouped output projection ----
    // Oracle rows:
    //   attn_raw (view)   {64, 64, 5}  3432.786621
    //   node_47 ROPE_BACK {64, 64, 5}    28.466785
    //   attn_derope-0     {512, 64, 5} -524.695190
    //   attn_wo_a-0       {1024, 5, 8}  134.724960
    //   attn_out-0        {4096, 5}     255.856689
    let n_nope = config.n_rot_none() as i64;
    let n_rot = config.n_rot as i64;
    let f32_size = std::mem::size_of::<f32>();
    let head_stride = head_dim as usize * f32_size;
    let (rope, rope_n_ctx_orig) = rope_for(config, s.il);
    let pos = ctx.new_i32_1d(nt).expect("pos");
    let positions: Vec<i32> = (0..nt as i32).collect();
    pos.set_i32(&positions).expect("set pos");

    // flash_attn_ext returns [head_dim, n_head, tokens] already, so this
    // reshape is a no-op on the layout and only re-labels it.
    let out3 = ctx.reshape_3d(&out, head_dim, n_head, nt).expect("reshape out");
    let out_nope = ctx
        .view_3d(&out3, n_nope, n_head, nt, head_stride, head_stride * n_head as usize, 0)
        .expect("out_nope");
    ctx.compute(&out_nope, 12).expect("compute out_nope");
    check(s, "out_nope", out_nope.to_vec_f32().iter().sum::<f32>());

    let out_pe_in = ctx
        .view_3d(
            &out3,
            n_rot,
            n_head,
            nt,
            head_stride,
            head_stride * n_head as usize,
            n_nope as usize * f32_size,
        )
        .expect("out_pe view");
    ctx.compute(&out_pe_in, 12).expect("compute out_pe view");
    check(s, "out_pe_in", out_pe_in.to_vec_f32().iter().sum::<f32>());

    // The inverse rotation, with exactly the parameters the forward one used.
    // 3432.8 -> 28.5 is not a small correction, and a forward rope here instead
    // of a backward one would be neither an error nor obviously wrong.
    let out_pe = ctx
        .rope_ext_back(&out_pe_in, &pos, None, n_rot as i32, ROPE_MODE_NORM, rope_n_ctx_orig, rope)
        .expect("rope_back");
    ctx.compute(&out_pe, 12).expect("compute rope_back");
    check(s, "rope_back", out_pe.to_vec_f32().iter().sum::<f32>());

    let derope = ctx.concat(&out_nope, &out_pe, 0).expect("concat derope");
    ctx.compute(&derope, 12).expect("compute derope");
    check(s, "attn_derope", derope.to_vec_f32().iter().sum::<f32>());

    // The output projection is grouped: `attn_output_a` ships 2-D as
    // [4096, 8192] and is *used* as [4096, o_lora_rank, 8] — a batched matmul
    // over 8 groups of 8 heads. Reading the shapes alone suggests the
    // dimensions do not connect, which is why this is transcribed from
    // deepseek4.cpp:1079-1084 rather than derived.
    let n_groups = config.output_group_count as i64;
    let o_lora = config.output_lora_rank as i64;
    let group_dim = (n_head / n_groups) * head_dim;
    let derope_g = ctx
        .reshape_3d(&derope, group_dim, n_groups, nt)
        .expect("reshape groups");
    let derope_p = ctx.permute(&derope_g, [0, 2, 1, 3]).expect("permute groups");
    ctx.compute(&derope_p, 12).expect("compute permuted");
    check(s, "attn_derope_perm", derope_p.to_vec_f32().iter().sum::<f32>());

    let wo_a = weights.get(&format!("blk.{}.attn_output_a.weight", s.il)).expect("bound");
    let wo_a3 = ctx
        .reshape_3d(wo_a, group_dim, o_lora, n_groups)
        .expect("reshape wo_a");
    let oa = ctx.mul_mat(&wo_a3, &derope_p).expect("attn_wo_a");
    ctx.compute(&oa, 12).expect("compute wo_a");
    check(s, "attn_wo_a", oa.to_vec_f32().iter().sum::<f32>());

    let oa_p = ctx.permute(&oa, [0, 2, 1, 3]).expect("permute oa");
    let oa_c = ctx.cont(&oa_p).expect("cont oa");
    let oa_2d = ctx
        .reshape_2d(&oa_c, o_lora * n_groups, nt)
        .expect("flatten oa");
    ctx.compute(&oa_2d, 12).expect("compute oa");
    check(s, "attn_wo_a_cont", oa_2d.to_vec_f32().iter().sum::<f32>());

    let attn_out = ctx
        .mul_mat(weights.get(&format!("blk.{}.attn_output_b.weight", s.il)).expect("bound"), &oa_2d)
        .expect("attn_out");
    ctx.compute(&attn_out, 12).expect("compute attn_out");
    check(s, "attn_out", attn_out.to_vec_f32().iter().sum::<f32>());

    attn_out
}
