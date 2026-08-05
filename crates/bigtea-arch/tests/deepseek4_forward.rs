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
const TOKENS_5: [i32; 5] = [671, 6102, 294, 8760, 344];

/// `LLAMA_ROPE_TYPE_NORM`. deepseek4 is in the NORM list in `llama-model.cpp`,
/// **not** the NEOX one — so rotated pairs are adjacent (`x[2i]`, `x[2i+1]`),
/// not offset by `n_rot/2`. Both conventions run, and one of them is wrong.
const ROPE_MODE_NORM: i32 = 0;

/// RoPE for layer 0, from the shipped [`Deepseek4Config::rope_for_layer`].
///
/// Deliberately not a local copy of the rules. `rope_for_layer` is what a real
/// forward pass will call, so it is what these checkpoints have to exercise —
/// a helper written out again here would verify the test and not the code.
///
/// It also asserts layer 0 is the *uncompressed* branch. `deepseek4.cpp:822-829`
/// picks per layer, and the container's YaRN settings (factor 16, original
/// context 65536, beta_fast 32, beta_slow 1) belong to the other 41 layers;
/// applying them here would be wrong in a way nothing reports.
fn rope_layer_0(config: &Deepseek4Config) -> RopeParams {
    assert!(
        !config.uses_compress_rope(0),
        "layer 0 must be uncompressed; compress_ratios[0] = {:?}",
        config.compress_ratios.first()
    );
    let rope = config.rope_for_layer(0);
    assert_eq!(rope.n_ctx_orig, 0, "no context extension on an uncompressed layer");
    rope.params
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
    names: &[&str],
) {
    for name in names {
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
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q_a.weight",
            "blk.0.attn_q_a_norm.weight",
            "blk.0.attn_q_b.weight",
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
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
) -> Prologue5<'c> {
    let hc = config.hc_mult as i64;
    let nt = TOKENS_5.len() as i64;

    let tok = ctx.new_i32_1d(nt).expect("tok");
    tok.set_i32(&TOKENS_5).expect("set");
    let embd = ctx
        .get_rows(weights.get("token_embd.weight").expect("bound"), &tok)
        .expect("get_rows");
    ctx.compute(&embd, 12).expect("compute embd");
    assert_sum("embd", embd.to_vec_f32().iter().sum::<f32>(), -5.680017);

    // [n_embd, tokens] -> [n_embd, 1, tokens] -> repeat to [n_embd, hc, tokens].
    let embd_r = ctx
        .reshape_3d(&embd, config.n_embd as i64, 1, nt)
        .expect("reshape embd");
    let shape = ctx
        .new_f32_3d(config.n_embd as i64, hc, nt)
        .expect("hc_init shape");
    let hc_init = ctx.repeat(&embd_r, &shape).expect("hc_init");
    ctx.compute(&hc_init, 12).expect("compute hc_init");
    assert_sum("hc_init", hc_init.to_vec_f32().iter().sum::<f32>(), -22.719982);

    let flat = ctx
        .reshape_2d(&hc_init, config.hc_dim() as i64, nt)
        .expect("flatten streams");
    let normed = ctx.rms_norm(&flat, config.rms_eps).expect("rms_norm");
    ctx.compute(&normed, 12).expect("compute norm");
    assert_sum("node_4 (rms_norm)", normed.to_vec_f32().iter().sum::<f32>(), -240.102188);

    let mixes = ctx
        .mul_mat(weights.get("blk.0.hc_attn_fn.weight").expect("bound"), &normed)
        .expect("hc_mixes");
    ctx.compute(&mixes, 12).expect("compute mixes");
    assert_sum("hc_mixes", mixes.to_vec_f32().iter().sum::<f32>(), -7549.175781);

    let gates = hc_gates(ctx, weights, config, "hc_attn", &mixes, ATTN_GATE_SUMS);

    let collapsed = ctx.dsv4_hc_pre(&hc_init, &gates.pre).expect("dsv4_hc_pre");
    ctx.compute(&collapsed, 12).expect("compute hc_pre op");
    assert_sum(
        "hc_attn_pre (fused)",
        collapsed.to_vec_f32().iter().sum::<f32>(),
        -22.720131,
    );

    let normed = ctx.rms_norm(&collapsed, config.rms_eps).expect("norm");
    ctx.compute(&normed, 12).expect("compute");
    assert_sum("norm-0", normed.to_vec_f32().iter().sum::<f32>(), -59.969891);

    let attn_norm = ctx
        .mul(&normed, weights.get("blk.0.attn_norm.weight").expect("bound"))
        .expect("attn_norm");
    ctx.compute(&attn_norm, 12).expect("compute");
    assert_sum("attn_norm-0", attn_norm.to_vec_f32().iter().sum::<f32>(), -1.357153);

    Prologue5 { hc_init, attn_norm, gates }
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

const ATTN_GATE_SUMS: HcGateSums = HcGateSums {
    pre_view: 516.695312,
    pre_scaled: 1072.673096,
    pre_biased: 1060.913452,
    pre_sigmoid: 20.000000,
    pre: 20.000015,
    post_view: -8064.689453,
    post_scaled: -151.041122,
    post_biased: -291.603790,
    post_sigmoid: 0.078218,
    post: 0.156435,
    comb: 19.999973,
};

const FFN_GATE_SUMS: HcGateSums = HcGateSums {
    pre_view: -22.430878,
    pre_scaled: -2.542310,
    pre_biased: -21.574642,
    pre_sigmoid: 6.341424,
    pre: 6.341444,
    post_view: -3306.610107,
    post_scaled: -118.711212,
    post_biased: -172.717499,
    post_sigmoid: 1.477709,
    post: 2.955418,
    comb: 19.999979,
};

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
    block: &str,
    mixes: &Tensor<'c>,
    want: HcGateSums,
) -> HcGates<'c> {
    let hc = config.hc_mult as i64;
    let nt = TOKENS_5.len() as i64;
    let f32_size = std::mem::size_of::<f32>();
    // The stride is the *source's* row, `hc_mix_dim = (2 + hc) * hc` = 24
    // floats (`deepseek4.cpp:277`, and `dsv4_view_2d` passes `t->nb[1]`), not
    // the 4 the view is wide. At one token the stride is never traversed and
    // any value for it passes, which is why the five-token capture is what
    // pinned this.
    let mix_stride = ((2 + hc) * hc) as usize * f32_size;

    let scale_w = weights.get(&format!("blk.0.{block}_scale.weight")).expect("scale bound");
    let base_w = weights.get(&format!("blk.0.{block}_base.weight")).expect("base bound");

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
    bind_all(&model, &wctx, &mut weights, ATTENTION_WEIGHTS);

    let p = prologue_5tok(&ctx, &weights, &config);
    let (q_full, kv_full) = q_and_kv_5tok(&ctx, &weights, &config, &p.attn_norm);
    assert_sum("q-0 (CONCAT)", q_full.to_vec_f32().iter().sum::<f32>(), 3544.263184);
    assert_sum("kv-0 (CONCAT)", kv_full.to_vec_f32().iter().sum::<f32>(), 63.125298);
}

/// Weights the attention path of layer 0 needs, in one list so the two
/// multi-token tests cannot drift apart.
const ATTENTION_WEIGHTS: &[&str] = &[
    "token_embd.weight",
    "blk.0.hc_attn_fn.weight",
    "blk.0.hc_attn_scale.weight",
    "blk.0.hc_attn_base.weight",
    "blk.0.attn_norm.weight",
    "blk.0.attn_q_a.weight",
    "blk.0.attn_q_a_norm.weight",
    "blk.0.attn_q_b.weight",
    "blk.0.attn_kv.weight",
    "blk.0.attn_kv_a_norm.weight",
    "blk.0.attn_sinks.weight",
    "blk.0.attn_output_a.weight",
    "blk.0.attn_output_b.weight",
];

/// Q and KV at five tokens, rotation included, checked step by step.
///
/// Returns `(q, kv)` shaped as llama.cpp leaves them: `q` is
/// `[head_dim, n_head, tokens]` and `kv` is `[head_dim, 1, tokens]` — one head,
/// and the same tensor will serve as both K and V.
fn q_and_kv_5tok<'c>(
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    attn_norm: &Tensor<'c>,
) -> (Tensor<'c>, Tensor<'c>) {
    let nt = TOKENS_5.len() as i64;
    let head_dim = config.kv_lora_rank as i64;
    let n_rot = config.n_rot as i64;
    let n_nope = config.n_rot_none() as i64;
    let f32_size = std::mem::size_of::<f32>();
    let head_stride = head_dim as usize * f32_size;
    let rope = rope_layer_0(config);

    // Positions 0..4. This tensor is the whole difference from the one-token
    // capture: with a single zero in it, every assertion below still passes on
    // a forward pass that never rotates anything.
    let pos = ctx.new_i32_1d(nt).expect("pos");
    pos.set_i32(&[0, 1, 2, 3, 4]).expect("set pos");

    // ---- Q ----
    let qr = ctx
        .mul_mat(weights.get("blk.0.attn_q_a.weight").expect("bound"), attn_norm)
        .expect("qr");
    ctx.compute(&qr, 12).expect("compute");
    assert_sum("qr-0", qr.to_vec_f32().iter().sum::<f32>(), 0.811234);

    let qr_n = ctx.rms_norm(&qr, config.rms_eps).expect("qr rms");
    ctx.compute(&qr_n, 12).expect("compute");
    assert_sum("norm-0 (qr)", qr_n.to_vec_f32().iter().sum::<f32>(), 7.035826);

    let qr_norm = ctx
        .mul(&qr_n, weights.get("blk.0.attn_q_a_norm.weight").expect("bound"))
        .expect("qr_norm");
    ctx.compute(&qr_norm, 12).expect("compute");
    assert_sum("qr_norm-0", qr_norm.to_vec_f32().iter().sum::<f32>(), -0.110477);

    let q = ctx
        .mul_mat(weights.get("blk.0.attn_q_b.weight").expect("bound"), &qr_norm)
        .expect("q");
    ctx.compute(&q, 12).expect("compute");
    assert_sum("node_19 (q_b)", q.to_vec_f32().iter().sum::<f32>(), 3.458504);

    let q3 = ctx
        .reshape_3d(&q, head_dim, config.n_head as i64, nt)
        .expect("reshape q");
    let q_norm = ctx.rms_norm(&q3, config.rms_eps).expect("q_norm");
    ctx.compute(&q_norm, 12).expect("compute");
    assert_sum("q_norm-0", q_norm.to_vec_f32().iter().sum::<f32>(), 157.955597);

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
    assert_sum("q_norm-0 (nope view)", q_nope.to_vec_f32().iter().sum::<f32>(), -537.884277);

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
    assert_sum("q_norm-0 (pe view)", unrotated, 695.835632);

    let q_pe = ctx
        .rope_ext(&q_pe_in, &pos, None, n_rot as i32, ROPE_MODE_NORM, 0, rope)
        .expect("rope q_pe");
    ctx.compute(&q_pe, 12).expect("compute rope");
    let rotated: f32 = q_pe.to_vec_f32().iter().sum();
    assert_sum("q_pe-0 (ROPE)", rotated, 4082.126465);

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
        .mul_mat(weights.get("blk.0.attn_kv.weight").expect("bound"), attn_norm)
        .expect("kv");
    ctx.compute(&kv, 12).expect("compute");
    assert_sum("node_26 (kv_a)", kv.to_vec_f32().iter().sum::<f32>(), 1.867785);

    let kv_n = ctx.rms_norm(&kv, config.rms_eps).expect("kv rms");
    ctx.compute(&kv_n, 12).expect("compute");
    assert_sum("norm-0 (kv)", kv_n.to_vec_f32().iter().sum::<f32>(), 19.056290);

    let kv_norm = ctx
        .mul(&kv_n, weights.get("blk.0.attn_kv_a_norm.weight").expect("bound"))
        .expect("kv_norm");
    ctx.compute(&kv_norm, 12).expect("compute");
    assert_sum("node_28 (kv_norm)", kv_norm.to_vec_f32().iter().sum::<f32>(), 10.532839);

    let kv3 = ctx.reshape_3d(&kv_norm, head_dim, 1, nt).expect("reshape kv");
    let kv_nope = ctx
        .view_3d(&kv3, n_nope, 1, nt, head_stride, head_stride, 0)
        .expect("kv_nope");
    ctx.compute(&kv_nope, 12).expect("compute kv_nope");
    assert_sum("kv_norm-0 (nope view)", kv_nope.to_vec_f32().iter().sum::<f32>(), -13.516478);

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
    assert_sum("kv_norm-0 (pe view)", kv_pe_in.to_vec_f32().iter().sum::<f32>(), 24.049295);

    let kv_pe = ctx
        .rope_ext(&kv_pe_in, &pos, None, n_rot as i32, ROPE_MODE_NORM, 0, rope)
        .expect("rope kv_pe");
    ctx.compute(&kv_pe, 12).expect("compute rope kv");
    assert_sum("kv_pe-0 (ROPE)", kv_pe.to_vec_f32().iter().sum::<f32>(), 76.641815);

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
    bind_all(&model, &wctx, &mut weights, ATTENTION_WEIGHTS);

    let p = prologue_5tok(&ctx, &weights, &config);
    let (q, kv) = q_and_kv_5tok(&ctx, &weights, &config, &p.attn_norm);
    attention_5tok(&ctx, &weights, &config, &q, &kv);
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
    let mut names: Vec<&str> = ATTENTION_WEIGHTS.to_vec();
    names.extend_from_slice(FFN_WEIGHTS);
    bind_all(&model, &wctx, &mut weights, &names);

    let p = prologue_5tok(&ctx, &weights, &config);
    let (q, kv) = q_and_kv_5tok(&ctx, &weights, &config, &p.attn_norm);
    let attn_out = attention_5tok(&ctx, &weights, &config, &q, &kv);

    let _ = layer_tail_5tok(&ctx, &weights, &config, &p, &attn_out);
}

/// Weights for the FFN half of the block.
const FFN_WEIGHTS: &[&str] = &[
    "blk.0.hc_ffn_fn.weight",
    "blk.0.hc_ffn_scale.weight",
    "blk.0.hc_ffn_base.weight",
    "blk.0.ffn_norm.weight",
];

/// Router and shared expert. All small — the 256 routed experts are not here.
const MOE_WEIGHTS: &[&str] = &[
    "blk.0.ffn_gate_inp.weight",
    "blk.0.ffn_gate_tid2eid.weight",
    "blk.0.ffn_gate_shexp.weight",
    "blk.0.ffn_up_shexp.weight",
    "blk.0.ffn_down_shexp.weight",
];

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
    let mut names: Vec<&str> = ATTENTION_WEIGHTS.to_vec();
    names.extend_from_slice(FFN_WEIGHTS);
    names.extend_from_slice(MOE_WEIGHTS);
    bind_all(&model, &wctx, &mut weights, &names);

    let p = prologue_5tok(&ctx, &weights, &config);
    let (q, kv) = q_and_kv_5tok(&ctx, &weights, &config, &p.attn_norm);
    let attn_out = attention_5tok(&ctx, &weights, &config, &q, &kv);
    let (_streams, ffn_norm, _gates) = layer_tail_5tok(&ctx, &weights, &config, &p, &attn_out);

    let _ = moe_routing_5tok(&ctx, &weights, &config, &ffn_norm);
    let _ = shared_expert_5tok(&ctx, &weights, &ffn_norm);
}

/// The router: probabilities, the six experts, and their normalised weights.
///
/// Returns `(weights, ids)` — the scaled weights shaped `[1, n_used, tokens]`
/// ready to multiply the expert outputs, and the expert ids as `mul_mat_id`
/// wants them.
fn moe_routing_5tok<'c>(
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    ffn_norm: &Tensor<'c>,
) -> (Tensor<'c>, Tensor<'c>) {
    let nt = TOKENS_5.len() as i64;
    let n_expert = config.n_expert as i64;
    let n_used = config.n_expert_used as i64;

    // ---- routing ----
    let logits = ctx
        .mul_mat(weights.get("blk.0.ffn_gate_inp.weight").expect("bound"), ffn_norm)
        .expect("logits");
    ctx.compute(&logits, 12).expect("compute logits");
    assert_sum(
        "ffn_moe_logits-0",
        logits.to_vec_f32().iter().sum::<f32>(),
        -1176.607300,
    );

    let sp = ctx.softplus(&logits).expect("softplus");
    ctx.compute(&sp, 12).expect("compute softplus");
    assert_sum("node_86 (SOFTPLUS)", sp.to_vec_f32().iter().sum::<f32>(), 587.096008);

    let probs = ctx.sqrt(&sp).expect("sqrt");
    ctx.compute(&probs, 12).expect("compute sqrt");
    assert_sum("ffn_moe_probs-0", probs.to_vec_f32().iter().sum::<f32>(), 792.403992);

    // Hash routing: the six experts are a lookup on the token id, and the
    // router never picks anything.
    let probs3 = ctx.reshape_3d(&probs, 1, n_expert, nt).expect("reshape probs");
    let tok = ctx.new_i32_1d(nt).expect("tok");
    tok.set_i32(&TOKENS_5).expect("set");
    let topk = ctx
        .get_rows(weights.get("blk.0.ffn_gate_tid2eid.weight").expect("bound"), &tok)
        .expect("topk");
    ctx.compute(&topk, 12).expect("compute topk");
    let ids = topk.to_vec_i32();
    assert_eq!(ids.len(), (n_used * nt) as usize, "six experts per token");
    assert_sum("ffn_moe_topk-0", ids.iter().sum::<i32>() as f32, 3688.0);

    let w = ctx.get_rows(&probs3, &topk).expect("weights");
    ctx.compute(&w, 12).expect("compute weights");
    assert_sum("ffn_moe_weights-0", w.to_vec_f32().iter().sum::<f32>(), 20.336262);

    // Renormalise over the *selected* six, not over all 256. This is the step
    // whose absence is invisible: the weights still sum to something, the model
    // still speaks.
    let w2 = ctx.reshape_2d(&w, n_used, nt).expect("reshape weights");
    let sum = ctx.sum_rows(&w2).expect("sum_rows");
    ctx.compute(&sum, 12).expect("compute sum");
    assert_sum("ffn_moe_weights_sum-0", sum.to_vec_f32().iter().sum::<f32>(), 20.336262);

    // Clamped away from zero at the smallest F16 normal, not at some epsilon.
    let sum_c = ctx.clamp(&sum, 6.103515625e-5, f32::INFINITY).expect("clamp sum");
    ctx.compute(&sum_c, 12).expect("compute clamped sum");
    assert_sum(
        "ffn_moe_weights_sum_clamped-0",
        sum_c.to_vec_f32().iter().sum::<f32>(),
        20.336262,
    );

    let w_norm = ctx.div(&w2, &sum_c).expect("div");
    ctx.compute(&w_norm, 12).expect("compute norm");
    assert_sum(
        "ffn_moe_weights_norm-0",
        w_norm.to_vec_f32().iter().sum::<f32>(),
        5.000000,
    );

    // Reshaped back to [1, n_used, tokens] *before* the scale, so it can
    // broadcast over each expert's [n_embd] output later.
    let w3 = ctx
        .reshape_3d(&w_norm, 1, n_used, nt)
        .expect("reshape weights");
    let w_scaled = ctx
        .scale(&w3, config.expert_weights_scale)
        .expect("scale weights");
    ctx.compute(&w_scaled, 12).expect("compute scaled");
    assert_sum(
        "ffn_moe_weights_scaled-0",
        w_scaled.to_vec_f32().iter().sum::<f32>(),
        7.500000,
    );

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
    let mut names: Vec<&str> = ATTENTION_WEIGHTS.to_vec();
    names.extend_from_slice(FFN_WEIGHTS);
    names.extend_from_slice(MOE_WEIGHTS);
    bind_all(&model, &wctx, &mut weights, &names);

    let p = prologue_5tok(&ctx, &weights, &config);
    let (q, kv) = q_and_kv_5tok(&ctx, &weights, &config, &p.attn_norm);
    let attn_out = attention_5tok(&ctx, &weights, &config, &q, &kv);
    let (streams, ffn_norm, gates) = layer_tail_5tok(&ctx, &weights, &config, &p, &attn_out);
    let (w_scaled, topk) = moe_routing_5tok(&ctx, &weights, &config, &ffn_norm);
    let shexp = shared_expert_5tok(&ctx, &weights, &ffn_norm);

    let nt = TOKENS_5.len() as i64;
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
        let name = format!("blk.0.{suffix}.weight");
        let (bytes, dims) = bind_expert_slices(&model, &wctx, &mut weights, &name, &unique);
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
            weights.get(&format!("blk.0.{suffix}.weight")).expect("bound"),
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
    assert_sum(
        "ffn_moe_gate-0",
        gate.to_vec_f32().iter().sum::<f32>(),
        -6601.376953,
    );

    let gate_c = ctx
        .clamp(&gate, f32::NEG_INFINITY, SWIGLU_CLAMP_L0)
        .expect("clamp gate");

    let up = ctx
        .mul_mat_id(&stack("ffn_up_exps"), &cur3, &ids_t)
        .expect("moe up");
    ctx.compute(&up, 12).expect("compute moe up");
    assert_sum("ffn_moe_up-0", up.to_vec_f32().iter().sum::<f32>(), -8.613072);

    let up_c = ctx
        .clamp(&up, -SWIGLU_CLAMP_L0, SWIGLU_CLAMP_L0)
        .expect("clamp up");

    let act = ctx.swiglu_split(&gate_c, &up_c).expect("swiglu");
    ctx.compute(&act, 12).expect("compute swiglu");
    assert_sum(
        "ffn_moe_swiglu_limited-0",
        act.to_vec_f32().iter().sum::<f32>(),
        11.649389,
    );

    let down = ctx
        .mul_mat_id(&stack("ffn_down_exps"), &act, &ids_t)
        .expect("moe down");
    ctx.compute(&down, 12).expect("compute moe down");
    assert_sum(
        "ffn_moe_down-0",
        down.to_vec_f32().iter().sum::<f32>(),
        89.523140,
    );

    // Each expert's output scaled by its router weight, then summed across the
    // six. llama.cpp does this as six strided views and five adds rather than a
    // reduction, so the same shape is used here.
    let weighted = ctx.mul(&down, &w_scaled).expect("weight experts");
    ctx.compute(&weighted, 12).expect("compute weighted");
    assert_sum(
        "ffn_moe_weighted-0",
        weighted.to_vec_f32().iter().sum::<f32>(),
        18.572262,
    );

    const PER_EXPERT: [f32; 6] = [
        1.238907, 6.887056, 6.492266, 3.250872, -5.103240, 5.806348,
    ];
    let row = n_embd as usize * f32_size;
    let mut moe_out: Option<Tensor> = None;
    for (j, want) in PER_EXPERT.iter().enumerate() {
        let v = ctx
            .view_2d(&weighted, n_embd, nt, row * n_used as usize, j * row)
            .expect("expert view");
        ctx.compute(&v, 12).expect("compute view");
        assert_sum(
            &format!("ffn_moe_weighted-0 [{j}]"),
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
    assert_sum(
        "ffn_moe_out-0",
        moe_out.to_vec_f32().iter().sum::<f32>(),
        18.572350,
    );

    // ---- the shared expert joins, and the layer closes ----
    let ffn_out = ctx.add(&moe_out, &shexp).expect("ffn_out");
    ctx.compute(&ffn_out, 12).expect("compute ffn_out");
    assert_sum(
        "ffn_out-0",
        ffn_out.to_vec_f32().iter().sum::<f32>(),
        34.800404,
    );

    // The layer's second hyper-connection write-back, using the FFN block's
    // gates — not the attention block's.
    let l_last = ctx
        .dsv4_hc_post(&ffn_out, &streams, &gates.post, &gates.comb)
        .expect("dsv4_hc_post");
    ctx.compute(&l_last, 12).expect("compute l_last");
    assert_sum("l_last-0", l_last.to_vec_f32().iter().sum::<f32>(), 6.733532);

    // What layer 1 sees. Matching here means the whole of layer 0 is right,
    // since every earlier error would have to cancel exactly to arrive at it.
    let flat = ctx
        .reshape_2d(&l_last, config.hc_dim() as i64, nt)
        .expect("flatten");
    let normed = ctx.rms_norm(&flat, config.rms_eps).expect("rms_norm");
    ctx.compute(&normed, 12).expect("compute node_125");
    assert_sum(
        "node_125 (into layer 1)",
        normed.to_vec_f32().iter().sum::<f32>(),
        1.599161,
    );
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
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    ffn_norm: &Tensor<'c>,
) -> Tensor<'c> {
    let gate = ctx
        .mul_mat(weights.get("blk.0.ffn_gate_shexp.weight").expect("bound"), ffn_norm)
        .expect("gate");
    ctx.compute(&gate, 12).expect("compute gate");
    assert_sum("ffn_gate-0", gate.to_vec_f32().iter().sum::<f32>(), 2518.574707);

    // Asymmetric on purpose: upper bound only.
    let gate_c = ctx
        .clamp(&gate, f32::NEG_INFINITY, SWIGLU_CLAMP_L0)
        .expect("clamp gate");
    ctx.compute(&gate_c, 12).expect("compute gate clamped");
    assert_sum(
        "ffn_gate_clamped-0",
        gate_c.to_vec_f32().iter().sum::<f32>(),
        2518.574707,
    );

    let up = ctx
        .mul_mat(weights.get("blk.0.ffn_up_shexp.weight").expect("bound"), ffn_norm)
        .expect("up");
    ctx.compute(&up, 12).expect("compute up");
    assert_sum("ffn_up-0", up.to_vec_f32().iter().sum::<f32>(), 36.162750);

    let up_c = ctx
        .clamp(&up, -SWIGLU_CLAMP_L0, SWIGLU_CLAMP_L0)
        .expect("clamp up");
    ctx.compute(&up_c, 12).expect("compute up clamped");
    assert_sum("ffn_up_clamped-0", up_c.to_vec_f32().iter().sum::<f32>(), 36.162750);

    let act = ctx.swiglu_split(&gate_c, &up_c).expect("swiglu");
    ctx.compute(&act, 12).expect("compute swiglu");
    assert_sum(
        "ffn_swiglu_limited-0",
        act.to_vec_f32().iter().sum::<f32>(),
        -39.681740,
    );

    let shexp = ctx
        .mul_mat(weights.get("blk.0.ffn_down_shexp.weight").expect("bound"), &act)
        .expect("shexp");
    ctx.compute(&shexp, 12).expect("compute shexp");
    assert_sum("ffn_shexp-0", shexp.to_vec_f32().iter().sum::<f32>(), 16.228374);

    shexp
}

/// Post hyper-connection, the FFN gate block, and `ffn_norm`.
///
/// Returns `(streams, ffn_norm)`: the updated 4-stream residual, which the
/// layer's second `hc_post` will need, and the normalised input the MoE and the
/// shared expert both consume.
fn layer_tail_5tok<'c>(
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    p: &Prologue5<'c>,
    attn_out: &Tensor<'c>,
) -> (Tensor<'c>, Tensor<'c>, HcGates<'c>) {
    let nt = TOKENS_5.len() as i64;

    // x = attn_out, residual = the streams as they were *before* attention.
    let streams = ctx
        .dsv4_hc_post(attn_out, &p.hc_init, &p.gates.post, &p.gates.comb)
        .expect("dsv4_hc_post");
    ctx.compute(&streams, 12).expect("compute hc_post");
    assert_sum(
        "hc_attn_post-0",
        streams.to_vec_f32().iter().sum::<f32>(),
        -14.514359,
    );

    // The FFN's gates come from their own matmul over the post-attention
    // stream, against hc_ffn_fn rather than hc_attn_fn.
    let flat = ctx
        .reshape_2d(&streams, config.hc_dim() as i64, nt)
        .expect("flatten streams");
    let normed = ctx.rms_norm(&flat, config.rms_eps).expect("rms_norm");
    ctx.compute(&normed, 12).expect("compute norm");
    assert_sum("node_65 (rms_norm)", normed.to_vec_f32().iter().sum::<f32>(), -77.870285);

    let mixes = ctx
        .mul_mat(weights.get("blk.0.hc_ffn_fn.weight").expect("bound"), &normed)
        .expect("hc_mixes ffn");
    ctx.compute(&mixes, 12).expect("compute mixes");
    assert_sum("hc_mixes (ffn)", mixes.to_vec_f32().iter().sum::<f32>(), -3608.835205);

    let gates = hc_gates(ctx, weights, config, "hc_ffn", &mixes, FFN_GATE_SUMS);

    let collapsed = ctx.dsv4_hc_pre(&streams, &gates.pre).expect("dsv4_hc_pre");
    ctx.compute(&collapsed, 12).expect("compute hc_ffn_pre");
    assert_sum(
        "hc_ffn_pre-0",
        collapsed.to_vec_f32().iter().sum::<f32>(),
        1.926467,
    );

    let normed = ctx.rms_norm(&collapsed, config.rms_eps).expect("norm");
    ctx.compute(&normed, 12).expect("compute");
    assert_sum("norm-0 (ffn)", normed.to_vec_f32().iter().sum::<f32>(), 61.501854);

    let ffn_norm = ctx
        .mul(&normed, weights.get("blk.0.ffn_norm.weight").expect("bound"))
        .expect("ffn_norm");
    ctx.compute(&ffn_norm, 12).expect("compute");
    assert_sum("ffn_norm-0", ffn_norm.to_vec_f32().iter().sum::<f32>(), 11.634495);

    (streams, ffn_norm, gates)
}

/// Attention through to `attn_out`, returning the block's output.
///
/// Shared with the layer-tail test, which needs `attn_out` to feed the post
/// hyper-connection.
fn attention_5tok<'c>(
    ctx: &'c Context,
    weights: &WeightSet<'c>,
    config: &Deepseek4Config,
    q_full: &Tensor<'c>,
    kv_full: &Tensor<'c>,
) -> Tensor<'c> {
    let nt = TOKENS_5.len() as i64;
    let head_dim = config.kv_lora_rank as i64;
    let n_head = config.n_head as i64;
    // The padded cache window llama.cpp's trace shows: cache_k_l0 is viewed as
    // {512, 1, 256} and permuted to {512, 256, 1}.
    const N_KV: i64 = 256;

    let sinks = weights.get("blk.0.attn_sinks.weight").expect("bound");
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
    assert_sum(
        "cache_k_l0 (F16)",
        round_tripped.iter().sum::<f32>(),
        63.123978,
    );

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
    assert_sum(
        "q-0 (permuted)",
        q_perm.to_vec_f32().iter().sum::<f32>(),
        3544.223633,
    );

    // scale is 1/sqrt(n_embd_head) over the *full* 512, not the 448 unrotated
    // dims (deepseek4.cpp:1063).
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let out = ctx
        .flash_attn_ext_with_sinks(&q_perm, &k, &k, &mask, sinks, scale)
        .expect("flash_attn_ext");
    ctx.compute(&out, 12).expect("compute attention");
    assert_sum(
        "node_41 (FLASH_ATTN_EXT)",
        out.to_vec_f32().iter().sum::<f32>(),
        2879.606934,
    );

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
    let rope = rope_layer_0(config);
    let pos = ctx.new_i32_1d(nt).expect("pos");
    pos.set_i32(&[0, 1, 2, 3, 4]).expect("set pos");

    // flash_attn_ext returns [head_dim, n_head, tokens] already, so this
    // reshape is a no-op on the layout and only re-labels it.
    let out3 = ctx.reshape_3d(&out, head_dim, n_head, nt).expect("reshape out");
    let out_nope = ctx
        .view_3d(&out3, n_nope, n_head, nt, head_stride, head_stride * n_head as usize, 0)
        .expect("out_nope");
    ctx.compute(&out_nope, 12).expect("compute out_nope");
    assert_sum(
        "attn_raw-0 (nope view)",
        out_nope.to_vec_f32().iter().sum::<f32>(),
        -553.160217,
    );

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
    assert_sum(
        "attn_raw-0 (pe view)",
        out_pe_in.to_vec_f32().iter().sum::<f32>(),
        3432.786621,
    );

    // The inverse rotation, with exactly the parameters the forward one used.
    // 3432.8 -> 28.5 is not a small correction, and a forward rope here instead
    // of a backward one would be neither an error nor obviously wrong.
    let out_pe = ctx
        .rope_ext_back(&out_pe_in, &pos, None, n_rot as i32, ROPE_MODE_NORM, 0, rope)
        .expect("rope_back");
    ctx.compute(&out_pe, 12).expect("compute rope_back");
    assert_sum(
        "node_47 (ROPE_BACK)",
        out_pe.to_vec_f32().iter().sum::<f32>(),
        28.466785,
    );

    let derope = ctx.concat(&out_nope, &out_pe, 0).expect("concat derope");
    ctx.compute(&derope, 12).expect("compute derope");
    assert_sum(
        "attn_derope-0",
        derope.to_vec_f32().iter().sum::<f32>(),
        -524.695190,
    );

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
    assert_sum(
        "attn_derope-0 (permuted)",
        derope_p.to_vec_f32().iter().sum::<f32>(),
        -524.691833,
    );

    let wo_a = weights.get("blk.0.attn_output_a.weight").expect("bound");
    let wo_a3 = ctx
        .reshape_3d(wo_a, group_dim, o_lora, n_groups)
        .expect("reshape wo_a");
    let oa = ctx.mul_mat(&wo_a3, &derope_p).expect("attn_wo_a");
    ctx.compute(&oa, 12).expect("compute wo_a");
    assert_sum("attn_wo_a-0", oa.to_vec_f32().iter().sum::<f32>(), 134.724960);

    let oa_p = ctx.permute(&oa, [0, 2, 1, 3]).expect("permute oa");
    let oa_c = ctx.cont(&oa_p).expect("cont oa");
    let oa_2d = ctx
        .reshape_2d(&oa_c, o_lora * n_groups, nt)
        .expect("flatten oa");
    ctx.compute(&oa_2d, 12).expect("compute oa");
    assert_sum(
        "attn_wo_a-0 (cont)",
        oa_2d.to_vec_f32().iter().sum::<f32>(),
        134.724960,
    );

    let attn_out = ctx
        .mul_mat(weights.get("blk.0.attn_output_b.weight").expect("bound"), &oa_2d)
        .expect("attn_out");
    ctx.compute(&attn_out, 12).expect("compute attn_out");
    assert_sum(
        "attn_out-0",
        attn_out.to_vec_f32().iter().sum::<f32>(),
        255.856689,
    );

    attn_out
}
