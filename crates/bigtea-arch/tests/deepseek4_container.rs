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

use bigtea_arch::{Deepseek4Config, Deepseek4Model};
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

    let arch = Deepseek4Model::new(config);
    arch.verify(&model).expect("every named tensor is present");
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
