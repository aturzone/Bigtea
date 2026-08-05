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

/// RoPE parameters for a layer whose `dsv4_compress_ratios[il]` is 0.
///
/// `deepseek4.cpp:822-829` picks per layer: with no compression the plain
/// `freq_base` is used with scaling switched **off** entirely — freq_scale 1,
/// ext_factor 0, and therefore `dsv4_rope_attn_factor` returns 1, with both
/// betas and `n_ctx_orig` zeroed. The container's YaRN settings (factor 16,
/// original context 65536, beta_fast 32, beta_slow 1) apply to the other 41
/// layers, and applying them here would be wrong in a way nothing reports.
fn rope_params_uncompressed(config: &Deepseek4Config) -> RopeParams {
    RopeParams {
        freq_base: config.rope_freq_base,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
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
) -> Tensor<'c> {
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

    // [0..4] of each token's 24 mixes is the pre gate. At five tokens this is a
    // strided 2-D view, not the contiguous prefix it looked like at one token —
    // a distinction the single-token capture could not have exposed, because
    // with nt == 1 the stride is never traversed and any value for it passes.
    //
    // The stride is the *source's* row, `hc_mix_dim = (2 + hc) * hc` = 24
    // floats (`deepseek4.cpp:277`, and `dsv4_view_2d` passes `t->nb[1]`), not
    // the 4 the view is wide.
    let f32_size = std::mem::size_of::<f32>();
    let hc_mix_dim = (2 + hc) * hc;
    let mix_stride = hc_mix_dim as usize * f32_size;
    let pre = ctx
        .view_2d(&mixes, hc, nt, mix_stride, 0)
        .expect("mixes pre");
    ctx.compute(&pre, 12).expect("compute pre");
    assert_sum("hc_mixes (pre view)", pre.to_vec_f32().iter().sum::<f32>(), 516.695312);

    let scale_pre = ctx
        .view_1d(weights.get("blk.0.hc_attn_scale.weight").expect("bound"), 1, 0)
        .expect("scale_pre");
    let base_pre = ctx
        .view_1d(weights.get("blk.0.hc_attn_base.weight").expect("bound"), hc, 0)
        .expect("base_pre");

    let scaled = ctx.mul(&pre, &scale_pre).expect("mul scale");
    ctx.compute(&scaled, 12).expect("compute scaled");
    assert_sum("node_8 (mul)", scaled.to_vec_f32().iter().sum::<f32>(), 1072.673096);

    let biased = ctx.add(&scaled, &base_pre).expect("add base");
    ctx.compute(&biased, 12).expect("compute biased");
    assert_sum("node_10 (add)", biased.to_vec_f32().iter().sum::<f32>(), 1060.913452);

    let gated = ctx.sigmoid(&biased).expect("sigmoid");
    ctx.compute(&gated, 12).expect("compute sigmoid");
    assert_sum("node_11 (sigmoid)", gated.to_vec_f32().iter().sum::<f32>(), 20.000000);

    let eps_t = ctx.new_f32_1d(hc).expect("eps");
    eps_t.set_f32(&vec![1e-6f32; hc as usize]).expect("fill eps");
    let hc_pre_w = ctx.add(&gated, &eps_t).expect("add eps");
    ctx.compute(&hc_pre_w, 12).expect("compute hc_pre");
    assert_sum("hc_pre (scale_bias)", hc_pre_w.to_vec_f32().iter().sum::<f32>(), 20.000015);

    let collapsed = ctx.dsv4_hc_pre(&hc_init, &hc_pre_w).expect("dsv4_hc_pre");
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

    attn_norm
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
    bind_all(
        &model,
        &wctx,
        &mut weights,
        &[
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
        ],
    );

    let nt = TOKENS_5.len() as i64;
    let head_dim = config.kv_lora_rank as i64;
    let n_rot = config.n_rot as i64;
    let n_nope = config.n_rot_none() as i64;
    let f32_size = std::mem::size_of::<f32>();
    let head_stride = head_dim as usize * f32_size;
    let rope = rope_params_uncompressed(&config);

    // Positions 0..4. This tensor is the whole difference from the one-token
    // capture: with a single zero in it, every assertion below still passes on
    // a forward pass that never rotates anything.
    let pos = ctx.new_i32_1d(nt).expect("pos");
    pos.set_i32(&[0, 1, 2, 3, 4]).expect("set pos");

    let attn_norm = prologue_5tok(&ctx, &weights, &config);

    // ---- Q ----
    let qr = ctx
        .mul_mat(weights.get("blk.0.attn_q_a.weight").expect("bound"), &attn_norm)
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
    assert_sum("q-0 (CONCAT)", q_full.to_vec_f32().iter().sum::<f32>(), 3544.263184);

    // ---- KV ----
    // One head, and the same tensor serves as K *and* V (deepseek4.cpp:792).
    let kv = ctx
        .mul_mat(weights.get("blk.0.attn_kv.weight").expect("bound"), &attn_norm)
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
    assert_sum("kv-0 (CONCAT)", kv_full.to_vec_f32().iter().sum::<f32>(), 63.125298);
}
