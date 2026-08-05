//! Verify the V4-Flash manifest against the real 144 GB container.
//!
//! Step one of the port: before any arithmetic, prove we can name every tensor
//! the architecture needs across five shards, and that the resident/streamed
//! split lands where `docs/graph/research/v4flash-port-recon.md` predicted —
//! 7.38 GiB always-read against 137.06 GiB routed. If that split is wrong the
//! model does not run at all, and it is far cheaper to find out here than
//! after a two-minute load.
//!
//! Skips when the model is absent, so the suite still passes on a machine
//! without 144 GB to spare.

use bigtea_arch::{AttentionKind, Deepseek4Config, Deepseek4Model};
use bigtea_model::Model;

const SHARD: &str =
    r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf";

const GIB: f64 = (1u64 << 30) as f64;

fn open() -> Option<Model> {
    if !std::path::Path::new(SHARD).exists() {
        eprintln!("skipping: {SHARD} not present");
        return None;
    }
    Model::open_split(SHARD).ok()
}

#[test]
fn manifest_matches_the_container() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config from container");

    // Read from the container, not assumed.
    assert_eq!(config.n_layer, 43);
    assert_eq!(config.n_embd, 4096);
    assert_eq!(config.n_head, 64);
    assert_eq!(config.n_head_kv, 1, "MLA: one KV head, not per-head K/V");
    assert_eq!(config.n_expert, 256);
    assert_eq!(config.n_expert_used, 6);
    assert_eq!(config.n_expert_shared, 1, "a shared expert runs every token");
    assert_eq!(config.q_lora_rank, 1024);
    assert_eq!(config.kv_lora_rank, 512);

    // Per-layer arrays, which are the ones a scalar read would quietly ruin.
    assert_eq!(
        config.swiglu_clamp_exp.len(),
        config.n_layer as usize,
        "one SwiGLU clamp limit per block"
    );
    assert_eq!(config.swiglu_clamp_shexp.len(), config.n_layer as usize);
    assert_eq!(config.swiglu_limit(0, false), Some(10.0));

    // 44 ratios for 43 blocks, and that is not an off-by-one: only the first
    // 43 are ever consulted. Layers 0-1 are the two uncompressed ones, which
    // is also what the attention plan says independently.
    assert_eq!(config.compress_ratios.len(), 44);
    assert!(!config.uses_compress_rope(0), "layer 0 is uncompressed");
    let uncompressed = (0..config.n_layer)
        .filter(|il| !config.uses_compress_rope(*il))
        .count();
    assert_eq!(uncompressed, 2, "exactly two layers skip the compressed RoPE");

    // The RoPE layer 0 actually gets, which is the one the forward tests
    // verify against llama.cpp's trace.
    let rope0 = config.rope_for_layer(0);
    assert_eq!(rope0.params.freq_base, 10_000.0);
    assert_eq!(rope0.params.freq_scale, 1.0);
    assert_eq!(rope0.params.ext_factor, 0.0, "no YaRN on an uncompressed layer");
    assert_eq!(rope0.n_ctx_orig, 0);

    // And a compressed one, which uses a different base entirely. Transcribed
    // from deepseek4.cpp and NOT verified against a capture — the oracle stops
    // at the end of layer 0.
    let compressed = (0..config.n_layer)
        .find(|il| config.uses_compress_rope(*il))
        .expect("41 compressed layers");
    let rope_c = config.rope_for_layer(compressed);
    assert_eq!(rope_c.params.freq_base, 160_000.0);
    assert_eq!(rope_c.n_ctx_orig, 65_536);
    assert!(rope_c.params.attn_factor < 1.0);

    let arch = Deepseek4Model::new(config);
    arch.verify(&model).expect("every named tensor is present");

    // The two uncompressed layers should be the two `Raw` attention layers —
    // two facts read from different places in the container agreeing.
    let plan = arch.attention_plan(&model);
    let raw = plan.iter().filter(|k| **k == AttentionKind::Raw).count();
    assert_eq!(raw, 2, "compress_ratios and the tensor manifest must agree");
}

/// The layers are not uniform, and pinning the counts is what stops a future
/// change from silently skipping part of the architecture.
#[test]
fn optional_tensors_appear_on_exactly_the_layers_they_should() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");
    let arch = Deepseek4Model::new(config);

    let counts = arch.optional_layer_counts(&model);
    let get = |name: &str| counts.iter().find(|(s, _)| *s == name).map(|(_, n)| *n);

    assert_eq!(get("attn_compressor_kv.weight"), Some(41));
    assert_eq!(get("exp_probs_b.bias"), Some(40));
    assert_eq!(get("indexer.proj.weight"), Some(21));
    // hash_layer_count in the metadata is 3, and these are those three layers.
    assert_eq!(get("ffn_gate_tid2eid.weight"), Some(3));

    // Nothing in the container goes unread. A tensor nobody names is a piece
    // of the architecture that has been skipped, which degrades output rather
    // than raising an error.
    let unclaimed = arch.unclaimed_tensors(&model);
    assert!(
        unclaimed.is_empty(),
        "{} tensors unaccounted for, first few: {:?}",
        unclaimed.len(),
        &unclaimed[..unclaimed.len().min(8)]
    );
}

/// The model uses three different attentions, and which one is per layer.
///
/// Implementing one and applying it everywhere would give fluent output that
/// is wrong on half the layers — so the plan is derived from the container and
/// pinned here.
#[test]
fn attention_kind_is_decided_per_layer() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");
    let arch = Deepseek4Model::new(config);
    let plan = arch.attention_plan(&model);

    let count = |k: AttentionKind| plan.iter().filter(|p| **p == k).count();
    let raw = count(AttentionKind::Raw);
    let hca = count(AttentionKind::HeavilyCompressed);
    let csa = count(AttentionKind::CompressedSparse);
    eprintln!("attention plan: {raw} raw, {hca} heavily-compressed, {csa} compressed-sparse");

    assert_eq!(plan.len(), 43);
    assert_eq!(raw + hca + csa, 43, "every layer classified");
    assert_eq!(csa, 21, "21 layers carry the lightning indexer");
    assert_eq!(hca, 20, "20 carry a compressor but no indexer");
    assert_eq!(raw, 2, "2 carry neither");
}

/// Shapes that only make sense once read from llama.cpp's loader, pinned so a
/// future misreading fails here rather than as plausible-looking output.
#[test]
fn derived_shapes_match_the_container() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");

    // Only 64 of each 512-wide head is rotated; the rest carries no position.
    assert_eq!(config.n_rot, 64);
    assert_eq!(config.n_rot_none(), 448);

    // The hyper-connection block runs on 4 parallel streams of n_embd.
    assert_eq!(config.hc_mult, 4);
    assert_eq!(config.hc_dim(), 16384);
    let hc_fn = model
        .location("blk.0.hc_attn_fn.weight")
        .expect("hc_attn_fn present");
    assert_eq!(hc_fn.dims[0], config.hc_dim() as u64, "hc_attn_fn is [hc_dim, mix]");

    // attn_output_a ships 2-D but is used as [n_head*head_dim/groups, rank, groups].
    assert_eq!(config.output_group_count, 8);
    let wo_a = model.location("blk.0.attn_output_a.weight").expect("wo_a");
    let expect_ne0 =
        (config.n_head as u64 * config.kv_lora_rank as u64) / config.output_group_count as u64;
    assert_eq!(wo_a.dims[0], expect_ne0, "grouped output projection ne0");
    assert_eq!(
        wo_a.dims[1],
        config.output_lora_rank as u64 * config.output_group_count as u64,
        "grouped output projection ne1"
    );

    // Q goes down to q_lora_rank then up to n_head * head_dim.
    let wq_b = model.location("blk.0.attn_q_b.weight").expect("wq_b");
    assert_eq!(wq_b.dims[0], config.q_lora_rank as u64);
    assert_eq!(wq_b.dims[1], config.n_head as u64 * config.kv_lora_rank as u64);

    // K and V are one shared compressed head, not per-head tensors.
    let wkv = model.location("blk.0.attn_kv.weight").expect("wkv");
    assert_eq!(wkv.dims[0], config.n_embd as u64);
    assert_eq!(wkv.dims[1], config.kv_lora_rank as u64);
}

/// Actually load the 7.38 GiB resident set off five shards and bind it.
///
/// Step two of the port. No arithmetic yet — this proves the parts that fail
/// silently or expensively later: that shard resolution works across all five
/// files, that every resident tensor's declared size matches what ggml expects
/// for its type (Q8_0, F32, BF16 and I32 all appear here, and a block-size
/// table missing one of them would surface as a size mismatch), and that
/// 7.38 GiB really does fit alongside everything else on a 15.7 GiB machine.
///
/// Ignored by default because it reads 7.38 GiB. Run it deliberately:
/// `cargo test --release --test deepseek4_container -- --ignored --nocapture`
#[test]
#[ignore = "loads 7.38 GiB of weights"]
fn resident_set_loads_and_binds() {
    use bigtea_ggml::{Context, WeightSet};

    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");
    let arch = Deepseek4Model::new(config);
    let names = arch.resident_tensor_names(&model);

    // Metadata only: the data pointers reference buffers we own, so this arena
    // holds tensor structs rather than weights.
    let ctx = Context::new_no_alloc(256 << 20).expect("context");
    let mut weights = WeightSet::new();
    let start = std::time::Instant::now();
    let mut bound = 0u64;

    for name in &names {
        let loc = model.location(name).expect("verified present").clone();
        let data = model
            .read_tensor(name)
            .unwrap_or_else(|e| panic!("reading {name}: {e}"));
        bound += data.len() as u64;
        weights
            .bind(&ctx, name, loc.ty, &loc.dims, data)
            .unwrap_or_else(|e| panic!("binding {name} (type {:?}): {e}", loc.ty));
    }

    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "bound {} tensors, {:.2} GiB in {secs:.1}s ({:.2} GB/s) across {} shards",
        weights.len(),
        bound as f64 / GIB,
        bound as f64 / 1e9 / secs.max(1e-9),
        model.shard_count()
    );

    assert_eq!(weights.len(), names.len(), "every resident tensor bound");
    assert!(
        bound as f64 / GIB > 7.0,
        "expected ~7.38 GiB resident, got {:.2}",
        bound as f64 / GIB
    );
}

#[test]
fn residency_split_is_what_the_design_depends_on() {
    let Some(model) = open() else { return };
    let config = Deepseek4Config::from_model(&model).expect("config");
    let per_token = config.expert_bytes_per_token(&model);
    let arch = Deepseek4Model::new(config);

    let resident = arch.resident_bytes(&model) as f64 / GIB;
    let per_token_gib = per_token as f64 / GIB;
    eprintln!("resident {resident:.2} GiB, {per_token_gib:.2} GiB of experts per token");

    // The premise: always-read weights fit a 15.7 GiB machine, experts do not.
    // Recon measured 7.38 GiB; allow room for the shared expert being counted
    // here, but fail loudly if it drifts into "does not fit" territory.
    assert!(
        resident > 5.0 && resident < 12.0,
        "resident set {resident:.2} GiB is outside the range this design assumes"
    );
    assert!(
        per_token_gib > 2.0 && per_token_gib < 5.0,
        "experts per token {per_token_gib:.2} GiB does not match the 3.21 GiB measured"
    );
}
