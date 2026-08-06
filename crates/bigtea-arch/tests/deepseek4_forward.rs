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
/// this table and the block index. Keeping the numbers as data rather than as
/// literals inside the helpers is what makes a second layer nearly free — and
/// running a second layer is the only way to tell a correct implementation from
/// one accidentally fitted to layer 0's weights.
struct LayerSums {
    il: u32,
    /// The prompt this layer's numbers were captured at. Two captures are in
    /// play and mixing them would compare a layer against the wrong run.
    tokens: &'static [i32],
    attn_gates: HcGateSums,
    ffn_gates: HcGateSums,
    /// Each routed expert's weighted contribution, checked individually so a
    /// mis-slotted expert cannot hide inside the total.
    weighted: [f32; 6],
    rows: &'static [(&'static str, f32)],
}

impl LayerSums {
    /// Panics on an unknown label rather than skipping the check. A typo that
    /// silently verified nothing would be worse than a failing test.
    fn get(&self, label: &str) -> f32 {
        self.rows
            .iter()
            .find(|(k, _)| *k == label)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| {
                panic!("no oracle row labelled {label:?} for layer {}", self.il)
            })
    }
}

/// Assert one checkpoint against the layer's table.
fn check(s: &LayerSums, label: &str, got: f32) {
    assert_sum(&format!("{label}-{}", s.il), got, s.get(label));
}

/// Layer 0, from `v4flash-layer0-oracle-5tok.txt`.
const LAYER0: LayerSums = LayerSums {
    il: 0,
    tokens: TOKENS_5,
    attn_gates: ATTN_GATE_SUMS,
    ffn_gates: FFN_GATE_SUMS,
    weighted: [1.238907, 6.887056, 6.492266, 3.250872, -5.103240, 5.806348],
    rows: &[("embd", -5.680017), ("hc_init", -22.719982), ("hc_init_norm", -240.102188),
("hc_mixes_attn", -7549.175781), ("hc_attn_pre", -22.720131), ("norm_attn", -59.969891),
("attn_norm", -1.357153), ("qr", 0.811234), ("qr_rms", 7.035826), ("qr_norm", -0.110477),
("q_b", 3.458504), ("q_norm", 157.955597), ("q_nope", -537.884277), ("q_pe_in", 695.835632),
("q_pe", 4082.126465), ("q", 3544.263184), ("kv_a", 1.867785), ("kv_rms", 19.056290),
("kv_norm", 10.532839), ("kv_nope", -13.516478), ("kv_pe_in", 24.049295), ("kv_pe", 76.641815),
("kv", 63.125298), ("cache_k", 63.123978), ("q_perm", 3544.223633), ("flash_attn", 2879.606934),
("out_nope", -553.160217), ("out_pe_in", 3432.786621), ("rope_back", 28.466785),
("attn_derope", -524.695190), ("attn_derope_perm", -524.691833), ("attn_wo_a", 134.724960),
("attn_wo_a_cont", 134.724960), ("attn_out", 255.856689), ("hc_attn_post", -14.514359),
("post_norm", -77.870285), ("hc_mixes_ffn", -3608.835205), ("hc_ffn_pre", 1.926467),
("norm_ffn", 61.501854), ("ffn_norm", 11.634495), ("moe_logits", -1176.607300),
("moe_softplus", 587.096008), ("moe_probs", 792.403992), ("moe_topk", 3688.0),
("moe_weights", 20.336262), ("moe_weights_sum", 20.336262),
("moe_weights_sum_clamped", 20.336262), ("moe_weights_norm", 5.000000),
("moe_weights_scaled", 7.500000), ("shexp_gate", 2518.574707),
("shexp_gate_clamped", 2518.574707), ("shexp_up", 36.162750), ("shexp_up_clamped", 36.162750),
("shexp_swiglu", -39.681740), ("shexp", 16.228374), ("moe_gate", -6601.376953),
("moe_up", -8.613072), ("moe_swiglu", 11.649389), ("moe_down", 89.523140),
("moe_weighted", 18.572262), ("moe_out", 18.572350), ("ffn_out", 34.800404),
("l_last", 6.733532), ("next_norm", 1.599161),],
};

/// Layer 1, from `v4flash-layer1-oracle-5tok.txt`.
///
/// No `embd`/`hc_init` rows: layer 1 has no prologue, it consumes layer 0's
/// `l_last` directly. Asking for one panics, which is the right outcome.
const LAYER1: LayerSums = LayerSums {
    il: 1,
    tokens: TOKENS_5,
    attn_gates: LAYER1_ATTN_GATE_SUMS,
    ffn_gates: LAYER1_FFN_GATE_SUMS,
    weighted: [-3.130055, 3.890552, -1.967723, -5.374112, -17.948400, 6.997021],
    rows: &[("hc_mixes_attn", -3428.892578), ("hc_attn_pre", -0.132875), ("norm_attn", 7.388832),
("attn_norm", -0.242196), ("qr", 6.841653), ("qr_rms", 69.162224), ("qr_norm", 2.789355),
("q_b", -33.388527), ("q_norm", -788.320190), ("q_nope", 2242.772461),
("q_pe_in", -3031.105957), ("q_pe", -8.882785), ("q", 2233.875488), ("kv_a", -15.869817),
("kv_rms", -126.332497), ("kv_norm", -106.741287), ("kv_nope", -84.832062),
("kv_pe_in", -21.909225), ("kv_pe", -28.781113), ("kv", -113.613174),
("cache_k", -113.621475), ("q_perm", 2233.874023), ("flash_attn", -2656.343018),
("out_nope", -2504.559814), ("out_pe_in", -151.774292), ("rope_back", -1608.978394),
("attn_derope", -4113.550781), ("attn_derope_perm", -4113.518555), ("attn_wo_a", 25.395323),
("attn_wo_a_cont", 25.395643), ("attn_out", 165.457382), ("hc_attn_post", 18.605337),
("post_norm", 110.003967), ("hc_mixes_ffn", -2718.131592), ("hc_ffn_pre", 9.309790),
("norm_ffn", 107.963402), ("ffn_norm", 27.486490), ("moe_logits", 23.125708),
("moe_softplus", 984.889099), ("moe_probs", 1087.419312), ("moe_topk", 3951.0),
("moe_weights", 25.273024), ("moe_weights_sum", 25.273022),
("moe_weights_sum_clamped", 25.273022), ("moe_weights_norm", 5.000000),
("moe_weights_scaled", 7.500000), ("shexp_gate", -1759.062988),
("shexp_gate_clamped", -1759.062988), ("shexp_up", 12.071652),
("shexp_up_clamped", 12.071652), ("shexp_swiglu", -57.020535), ("shexp", 36.638653),
("moe_gate", -3060.197266), ("moe_up", -120.346321), ("moe_swiglu", 11.324850),
("moe_down", -26.838915), ("moe_weighted", -17.532829), ("moe_out", -17.532742),
("ffn_out", 19.105911), ("l_last", 23.754854), ("next_norm", 98.961739),],
};

/// Layer 2's entry only — its attention is Compressed Sparse and is not built.
///
/// A layer's *entry* is architecture-independent: the hyper-connection gate
/// block and `attn_norm` are identical whatever attention follows. So the seam
/// into the first compressed layer is checkable now, and when CSA is built only
/// the attention itself will be new.
const LAYER2_ENTRY: LayerSums = LayerSums {
    il: 2,
    tokens: TOKENS_5,
    attn_gates: LAYER2_ATTN_GATE_SUMS,
    ffn_gates: LAYER2_ATTN_GATE_SUMS, // unused: the FFN half is not reached
    weighted: [0.0; 6],
    rows: &[
        ("hc_mixes_attn", -3056.212402),
        ("hc_attn_pre", 15.256248),
        ("norm_attn", 81.159294),
        ("attn_norm", 5.640476),
    ],
};

const LAYER2_ATTN_GATE_SUMS: HcGateSums = HcGateSums {
    pre_view: -287.356506,
    pre_scaled: -12.378065,
    pre_biased: -45.173386,
    pre_sigmoid: 5.595741,
    pre: 5.595761,
    post_view: -2590.740479,
    post_scaled: -70.598717,
    post_biased: -316.790985,
    post_sigmoid: 0.108592,
    post: 0.217185,
    comb: 19.999975,
};

const L0T2_ATTN_GATES: HcGateSums = HcGateSums {
    pre_view: 209.370789,
    pre_scaled: 434.659210,
    pre_biased: 429.955353,
    pre_sigmoid: 8.000000,
    pre: 8.000008,
    post_view: -2760.560059,
    post_scaled: -51.701687,
    post_biased: -107.926743,
    post_sigmoid: 0.049193,
    post: 0.098386,
    comb: 7.999992,
};
const L0T2_FFN_GATES: HcGateSums = HcGateSums {
    pre_view: -46.849030,
    pre_scaled: -5.309858,
    pre_biased: -12.922791,
    pre_sigmoid: 2.537631,
    pre: 2.537638,
    post_view: -1059.511353,
    post_scaled: -38.037712,
    post_biased: -59.640213,
    post_sigmoid: 1.009126,
    post: 2.018253,
    comb: 7.999992,
};
const L0T2: LayerSums = LayerSums {
    il: 0,
    tokens: TOKENS_2,
    attn_gates: L0T2_ATTN_GATES,
    ffn_gates: L0T2_FFN_GATES,
    weighted: [2.690806, 3.103284, 1.483350, 4.910249, -1.308418, 0.460610],
    rows: &[
        ("hc_mixes_attn", -2551.319580),
        ("hc_attn_pre", 6.423793),
        ("norm_attn", -1.717879),
        ("attn_norm", 0.001301),
        ("qr", -1.779395),
        ("qr_rms", -24.776793),
        ("qr_norm", -0.961105),
        ("q_b", 0.905390),
        ("q_norm", 44.419037),
        ("q_nope", 97.833832),
        ("q_pe_in", -53.416119),
        ("q_pe", 237.916412),
        ("q", 335.751648),
        ("kv_a", 6.459026),
        ("kv_rms", 70.477623),
        ("kv_norm", 45.408470),
        ("kv_nope", 27.834160),
        ("kv_pe_in", 17.574301),
        ("kv_pe", 27.742146),
        ("kv", 55.576317),
        ("cache_k", 55.579575),
        ("q_perm", 335.750671),
        ("flash_attn", 1372.857910),
        ("out_nope", 624.717163),
        ("out_pe_in", 748.141052),
        ("rope_back", 218.285583),
        ("attn_derope", 843.000610),
        ("attn_derope_perm", 842.997192),
        ("attn_wo_a", -129.421265),
        ("attn_wo_a_cont", -129.421600),
        ("attn_out", 189.549622),
        ("hc_attn_post", 14.272767),
        ("post_norm", 115.687309),
        ("hc_mixes_ffn", -1174.730103),
        ("hc_ffn_pre", 9.980318),
        ("norm_ffn", 74.270218),
        ("ffn_norm", 10.745688),
        ("moe_logits", -239.519104),
        ("moe_softplus", 380.650543),
        ("moe_probs", 389.258514),
        ("moe_topk", 1489.000000),
        ("moe_weights", 8.878812),
        ("moe_weights_sum", 8.878812),
        ("moe_weights_sum_clamped", 8.878812),
        ("moe_weights_norm", 2.000000),
        ("moe_weights_scaled", 3.000000),
        ("shexp_gate", 1304.182739),
        ("shexp_gate_clamped", 1304.182739),
        ("shexp_up", 46.957611),
        ("shexp_up_clamped", 46.957611),
        ("shexp_swiglu", -17.843456),
        ("shexp", 23.385096),
        ("moe_gate", -996.341431),
        ("moe_up", 32.241657),
        ("moe_swiglu", 28.105509),
        ("moe_down", 40.416405),
        ("moe_weighted", 11.339972),
        ("moe_out", 11.339869),
        ("ffn_out", 34.725060),
        ("l_last", 48.880856),
        ("next_norm", 278.791321),
        ("embd", 1.605949),
        ("hc_init", 6.423725),
        ("hc_init_norm", -6.837732),
    ],
};

const L1T2_ATTN_GATES: HcGateSums = HcGateSums {
    pre_view: -19.943457,
    pre_scaled: -1.810599,
    pre_biased: -11.711184,
    pre_sigmoid: 2.078893,
    pre: 2.078902,
    post_view: -928.173157,
    post_scaled: -28.170483,
    post_biased: -138.429504,
    post_sigmoid: 0.074534,
    post: 0.149069,
    comb: 7.999992,
};
const L1T2_FFN_GATES: HcGateSums = HcGateSums {
    pre_view: -245.671432,
    pre_scaled: -21.606079,
    pre_biased: -36.009747,
    pre_sigmoid: 1.643103,
    pre: 1.643111,
    post_view: -576.170471,
    post_scaled: -33.862514,
    post_biased: -41.283554,
    post_sigmoid: 0.882652,
    post: 1.765305,
    comb: 7.999992,
};
const L1T2: LayerSums = LayerSums {
    il: 1,
    tokens: TOKENS_2,
    attn_gates: L1T2_ATTN_GATES,
    ffn_gates: L1T2_FFN_GATES,
    weighted: [-3.639457, 0.028346, -4.889457, -3.136386, 1.335585, -0.239741],
    rows: &[
        ("hc_mixes_attn", -1037.038818),
        ("hc_attn_pre", 10.950562),
        ("norm_attn", 114.471718),
        ("attn_norm", 4.476905),
        ("qr", 0.608606),
        ("qr_rms", 7.972394),
        ("qr_norm", 0.414442),
        ("q_b", -5.684414),
        ("q_norm", -129.545334),
        ("q_nope", 1134.758911),
        ("q_pe_in", -1264.305176),
        ("q_pe", -698.882568),
        ("q", 435.879791),
        ("kv_a", -4.307814),
        ("kv_rms", -38.171127),
        ("kv_norm", -38.070057),
        ("kv_nope", -34.932976),
        ("kv_pe_in", -3.137122),
        ("kv_pe", -0.894517),
        ("kv", -35.827446),
        ("cache_k", -35.825783),
        ("q_perm", 435.880096),
        ("flash_attn", -339.987518),
        ("out_nope", -135.980682),
        ("out_pe_in", -204.004913),
        ("rope_back", -335.867615),
        ("attn_derope", -471.849945),
        ("attn_derope_perm", -471.850739),
        ("attn_wo_a", 11.147385),
        ("attn_wo_a_cont", 11.147484),
        ("attn_out", -1.024545),
        ("hc_attn_post", 48.594090),
        ("post_norm", 289.686951),
        ("hc_mixes_ffn", -800.463989),
        ("hc_ffn_pre", 7.422015),
        ("norm_ffn", 88.181358),
        ("ffn_norm", 22.388680),
        ("moe_logits", 112.637177),
        ("moe_softplus", 433.809448),
        ("moe_probs", 463.523590),
        ("moe_topk", 1424.000000),
        ("moe_weights", 10.837965),
        ("moe_weights_sum", 10.837965),
        ("moe_weights_sum_clamped", 10.837965),
        ("moe_weights_norm", 2.000000),
        ("moe_weights_scaled", 3.000000),
        ("shexp_gate", -478.736237),
        ("shexp_gate_clamped", -478.736237),
        ("shexp_up", -7.797753),
        ("shexp_up_clamped", -7.797753),
        ("shexp_swiglu", -29.932285),
        ("shexp", 9.406198),
        ("moe_gate", -91.637253),
        ("moe_up", -66.311356),
        ("moe_swiglu", -13.228956),
        ("moe_down", -37.769482),
        ("moe_weighted", -10.541123),
        ("moe_out", -10.541110),
        ("ffn_out", -1.134963),
        ("l_last", 48.674198),
        ("next_norm", 202.615189),
    ],
};

const L2T2_ATTN_GATES: HcGateSums = HcGateSums {
    pre_view: -66.417557,
    pre_scaled: -2.860979,
    pre_biased: -15.979107,
    pre_sigmoid: 2.423081,
    pre: 2.423090,
    post_view: -697.689209,
    post_scaled: -19.012312,
    post_biased: -117.489227,
    post_sigmoid: 0.065951,
    post: 0.131902,
    comb: 7.999992,
};
const L2T2_FFN_GATES: HcGateSums = HcGateSums {
    pre_view: 63.886581,
    pre_scaled: 9.101670,
    pre_biased: -4.527986,
    pre_sigmoid: 2.914737,
    pre: 2.914745,
    post_view: -211.144928,
    post_scaled: -17.000410,
    post_biased: -23.442335,
    post_sigmoid: 0.581446,
    post: 1.162891,
    comb: 7.999992,
};
const L2T2: LayerSums = LayerSums {
    il: 2,
    tokens: TOKENS_2,
    attn_gates: L2T2_ATTN_GATES,
    ffn_gates: L2T2_FFN_GATES,
    weighted: [3.269274, 2.185421, 1.311456, 1.190008, -3.560312, 0.731329],
    rows: &[
        ("hc_mixes_attn", -773.708740),
        ("hc_attn_pre", 13.370720),
        ("norm_attn", 43.803864),
        ("attn_norm", 5.764471),
        ("qr", 5.323662),
        ("qr_rms", 44.813297),
        ("qr_norm", 1.362917),
        ("q_b", 31.656322),
        ("q_norm", 1072.781616),
        ("q_nope", 1477.194702),
        ("q_pe_in", -404.419312),
        ("q_pe", -564.463867),
        ("q", 912.738342),
        ("kv_a", 3.730389),
        ("kv_rms", 16.242336),
        ("kv_norm", 6.633435),
        ("kv_nope", 17.739407),
        ("kv_pe_in", -11.105971),
        ("kv_pe", -10.623191),
        ("kv", 7.116213),
        ("cache_k", 7.117183),
        ("q_perm", 912.735718),
        ("flash_attn", 1227.910034),
        ("out_nope", 1378.445557),
        ("out_pe_in", -150.534897),
        ("rope_back", -126.964722),
        ("attn_derope", 1251.483276),
        ("attn_derope_perm", 1251.483521),
        ("attn_wo_a", 49.200817),
        ("attn_wo_a_cont", 49.200695),
        ("attn_out", -4.513974),
        ("hc_attn_post", 47.364868),
        ("post_norm", 203.504044),
        ("hc_mixes_ffn", -129.856979),
        ("hc_ffn_pre", 9.273748),
        ("norm_ffn", 54.077953),
        ("ffn_norm", 16.603214),
        ("moe_logits", 195.286179),
        ("moe_softplus", 485.156067),
        ("moe_probs", 488.993378),
        ("moe_topk", 1127.000000),
        ("moe_weights", 11.893110),
        ("moe_weights_sum", 11.893110),
        ("moe_weights_sum_clamped", 11.893110),
        ("moe_weights_norm", 2.000000),
        ("moe_weights_scaled", 3.000000),
        ("shexp_gate", -519.246338),
        ("shexp_gate_clamped", -519.246338),
        ("shexp_up", 2.633580),
        ("shexp_up_clamped", 2.633580),
        ("shexp_swiglu", 3.505288),
        ("shexp", 3.952679),
        ("moe_gate", -797.049194),
        ("moe_up", 49.092308),
        ("moe_swiglu", 7.136783),
        ("moe_down", 26.848846),
        ("moe_weighted", 5.127125),
        ("moe_out", 5.127162),
        ("ffn_out", 9.079841),
        ("l_last", 52.540024),
        ("next_norm", 220.934692),
    ],
};

const L3T2_ATTN_GATES: HcGateSums = HcGateSums {
    pre_view: 118.016075,
    pre_scaled: 9.362187,
    pre_biased: -1.522255,
    pre_sigmoid: 3.695869,
    pre: 3.695877,
    post_view: -708.020691,
    post_scaled: -28.494911,
    post_biased: -45.285225,
    post_sigmoid: 0.111723,
    post: 0.223446,
    comb: 7.999992,
};
const L3T2_FFN_GATES: HcGateSums = HcGateSums {
    pre_view: 84.676689,
    pre_scaled: 7.280041,
    pre_biased: -4.989226,
    pre_sigmoid: 2.965539,
    pre: 2.965547,
    post_view: -285.240814,
    post_scaled: -16.522358,
    post_biased: -24.049160,
    post_sigmoid: 0.857103,
    post: 1.714206,
    comb: 7.999993,
};
const L3T2: LayerSums = LayerSums {
    il: 3,
    tokens: TOKENS_2,
    attn_gates: L3T2_ATTN_GATES,
    ffn_gates: L3T2_FFN_GATES,
    weighted: [59.109970, 3.667929, -1.341033, 0.643734, 1.267123, -0.377637],
    rows: &[
        ("hc_mixes_attn", -598.809387),
        ("hc_attn_pre", 24.208088),
        ("norm_attn", 79.563873),
        ("attn_norm", 4.056807),
        ("qr", 3.573563),
        ("qr_rms", 35.861706),
        ("qr_norm", 0.928110),
        ("q_b", -12.809828),
        ("q_norm", -522.932495),
        ("q_nope", -2075.611328),
        ("q_pe_in", 1552.682373),
        ("q_pe", 1503.679077),
        ("q", -571.931274),
        ("kv_a", 3.354357),
        ("kv_rms", 26.248386),
        ("kv_norm", 10.826424),
        ("kv_nope", 44.097580),
        ("kv_pe_in", -33.271152),
        ("kv_pe", -37.037834),
        ("kv", 7.059739),
        ("cache_k", 7.060322),
        ("q_perm", -571.931824),
        ("flash_attn", -101.196808),
        ("out_nope", 667.009460),
        ("out_pe_in", -768.201477),
        ("rope_back", -726.776123),
        ("attn_derope", -59.771351),
        ("attn_derope_perm", -59.771358),
        ("attn_wo_a", -104.886375),
        ("attn_wo_a_cont", -104.886482),
        ("attn_out", 27.662806),
        ("hc_attn_post", 55.305370),
        ("post_norm", 220.449265),
        ("hc_mixes_ffn", -228.237274),
        ("hc_ffn_pre", 11.994062),
        ("norm_ffn", 69.295235),
        ("ffn_norm", 18.009119),
        ("moe_logits", -1966.536499),
        ("moe_softplus", 18.498667),
        ("moe_probs", 83.329803),
        ("moe_probs_biased", 4685.573242),
        ("moe_argsort", 65280.000000),
        ("moe_topk", 1634.000000),
        ("moe_weights", 5.837240),
        ("moe_weights_sum", 5.837240),
        ("moe_weights_sum_clamped", 5.837240),
        ("moe_weights_norm", 2.000000),
        ("moe_weights_scaled", 3.000000),
        ("shexp_gate", -1346.839722),
        ("shexp_gate_clamped", -1346.839722),
        ("shexp_up", -9.172648),
        ("shexp_up_clamped", -9.172648),
        ("shexp_swiglu", -0.316023),
        ("shexp", 6.563107),
        ("moe_gate", -2352.544434),
        ("moe_up", 85.588280),
        ("moe_swiglu", 5.400528),
        ("moe_down", 105.462456),
        ("moe_weighted", 62.970425),
        ("moe_out", 62.970127),
        ("ffn_out", 69.533234),
        ("l_last", 113.606995),
        ("next_norm", 427.686554),
    ],
};


/// Layer 1's FFN gate block.
const LAYER1_FFN_GATE_SUMS: HcGateSums = HcGateSums {
    pre_view: -916.991821,
    pre_scaled: -80.646713,
    pre_biased: -116.655891,
    pre_sigmoid: 4.978212,
    pre: 4.978231,
    post_view: -1807.392578,
    post_scaled: -106.223541,
    post_biased: -124.776138,
    post_sigmoid: 1.647393,
    post: 3.294786,
    comb: 19.999975,
};

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

    let s = &LAYER0;
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

    let p = prologue_5tok(&LAYER0, &ctx, &weights, &config);
    let (q, kv) = q_and_kv_5tok(&LAYER0, &ctx, &weights, &config, &p.attn_norm);
    attention_5tok(&LAYER0, &ctx, &weights, &config, &q, &kv);
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

    let p = prologue_5tok(&LAYER0, &ctx, &weights, &config);
    let (q, kv) = q_and_kv_5tok(&LAYER0, &ctx, &weights, &config, &p.attn_norm);
    let attn_out = attention_5tok(&LAYER0, &ctx, &weights, &config, &q, &kv);

    let _ = layer_tail_5tok(&LAYER0, &ctx, &weights, &config, &p, &attn_out);
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

    let p = prologue_5tok(&LAYER0, &ctx, &weights, &config);
    let (q, kv) = q_and_kv_5tok(&LAYER0, &ctx, &weights, &config, &p.attn_norm);
    let attn_out = attention_5tok(&LAYER0, &ctx, &weights, &config, &q, &kv);
    let (_streams, ffn_norm, _gates) = layer_tail_5tok(&LAYER0, &ctx, &weights, &config, &p, &attn_out);

    let _ = moe_routing_5tok(&LAYER0, &ctx, &weights, &config, &ffn_norm);
    let _ = shared_expert_5tok(&LAYER0, &ctx, &weights, &ffn_norm);
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

    let _ = layer0_5tok(&LAYER0, &model, &ctx, &wctx, &mut weights, &config);
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
    check(s, "next_norm", normed.to_vec_f32().iter().sum::<f32>());

    l_last
}

/// Layer 1's own attention gates, from `v4flash-layer1-oracle-5tok.txt`.
const LAYER1_ATTN_GATE_SUMS: HcGateSums = HcGateSums {
    pre_view: -11.962343,
    pre_scaled: -1.086020,
    pre_biased: -25.837482,
    pre_sigmoid: 5.607750,
    pre: 5.607770,
    post_view: -3115.966064,
    post_scaled: -94.570992,
    post_biased: -370.218567,
    post_sigmoid: 0.104725,
    post: 0.209451,
    comb: 19.999981,
};

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

    let l_last = layer0_5tok(&LAYER0, &model, &ctx, &wctx, &mut weights, &config);

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

    let l_last = layer_5tok(&LAYER1, &model, &ctx, &wctx, &mut weights, &config, &l_last);

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
    let _ = layer_entry_5tok(&LAYER2_ENTRY, &ctx, &weights, &config, &l_last);
}


/// **Four layers, and the two remaining holes closed.**
///
/// Runs layers 0-3 end to end at a *two-token* prompt, ending on the exact
/// tensor layer 4 would receive.
///
/// The reason a shorter prompt reaches further than a longer one is the guard
/// on the compressed attention builders (`deepseek4.cpp:1049-1063`): they need
/// their compressed caches populated, and at two tokens those caches are empty,
/// so **layers 2 and 3 fall through to `build_raw_attention`** — already built
/// and already verified. Their compressor projections still run, but nothing
/// reads them at this length.
///
/// That makes two things checkable that five tokens could not reach without
/// first building the lightning indexer:
///
/// 1. **The compressed RoPE branch.** `compress_ratios[il] != 0` from layer 2
///    on, and that choice is independent of which attention builder runs — so
///    `rope_for_layer`'s YaRN branch, transcribed from source and until now
///    checked against nothing, finally executes. `q_pe-2` and `q_pe-3` both
///    rotate against base 160000, not 10000.
/// 2. **The normal MoE routing path**, which 40 of 43 layers use. Layers 0-2
///    are the `hash_layer_count` layers and select by token-id lookup; layer 3
///    is the first to do `probs + exp_probs_b -> argsort_top_k`. The bias
///    steers *selection only* — the weights are gathered from the unbiased
///    probabilities, which llama.cpp spells out at `llama-graph.cpp:1885` and
///    which changes every expert weight if got wrong, with no shape to catch it.
///
/// It is also a third independent input for layers 0 and 1, whose numbers here
/// share nothing with either earlier capture.
#[test]
#[ignore = "reads weights from a 144 GB container"]
fn four_layers_at_two_tokens_close_the_compressed_rope_and_routing_holes() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    // Four layers in one arena; ggml aborts rather than erroring if this is
    // short, so it is sized for the whole chain up front.
    let ctx = Context::new(2560 << 20).expect("compute context");
    let wctx = Context::new_no_alloc(32 << 20).expect("weight context");
    let mut weights = WeightSet::new();
    for il in 0..4u32 {
        bind_all(&model, &wctx, &mut weights, &block_weights(il));
        bind_all(&model, &wctx, &mut weights, &optional_block_weights(&model, il));
    }

    // The premise, asserted rather than assumed: layer 0-1 uncompressed, 2-3
    // compressed, and only 0-2 hash-routed.
    assert!(!config.uses_compress_rope(1));
    assert!(config.uses_compress_rope(2), "layer 2 must take the YaRN branch");
    assert!(config.uses_compress_rope(3));
    assert_eq!(config.hash_layer_count, 3);

    let mut streams = layer0_5tok(&L0T2, &model, &ctx, &wctx, &mut weights, &config);
    for s in [&L1T2, &L2T2, &L3T2] {
        streams = layer_5tok(s, &model, &ctx, &wctx, &mut weights, &config, &streams);
    }
    let _ = streams;
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
