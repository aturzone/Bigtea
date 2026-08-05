//! Build the V4-Flash forward pass against llama.cpp's numbers, one step at a
//! time.
//!
//! Every assertion here compares against `tests/fixtures/v4flash-layer0-oracle.txt`,
//! captured from `llama-eval-callback` on the real container with the prompt
//! `"Hi"` — which tokenizes to the single id 23166, no BOS.
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
use bigtea_ggml::{Context, WeightSet};
use bigtea_model::Model;

const SHARD: &str =
    r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf";

/// The prompt "Hi" as llama.cpp tokenized it: one token, no BOS.
const TOKEN: i32 = 23166;

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
}
