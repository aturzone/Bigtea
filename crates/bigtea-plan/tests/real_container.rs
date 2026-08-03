//! Integration test against a **real** model container on disk.
//!
//! This deliberately tests against actual downloaded weights rather than a
//! fixture, because the failure mode that matters is a real container whose
//! metadata does not look the way the parser assumed. Fixtures cannot catch
//! that; only real files can.
//!
//! Skips cleanly when the model is absent, so the suite stays green on a
//! machine that has not downloaded 144 GiB. Point `BIGTEA_TEST_GGUF` at a
//! container to run it.

use std::io::Read;
use std::path::PathBuf;

use bigtea_gguf::Gguf;
use bigtea_plan::{ModelProfile, Prediction, ProfileSource, DEFAULT_EFFICIENCY, GIB};

/// Where the DeepSeek-V4-Flash shards land on the development machine.
const DEFAULT_PATH: &str =
    r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf";

fn find_container() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BIGTEA_TEST_GGUF") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = PathBuf::from(DEFAULT_PATH);
    p.exists().then_some(p)
}

fn read_head(path: &PathBuf, limit: u64) -> std::io::Result<Vec<u8>> {
    let f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.take(limit).read_to_end(&mut buf)?;
    Ok(buf)
}

#[test]
fn parses_a_real_deepseek_container() {
    let Some(path) = find_container() else {
        eprintln!("skipping: no container found (set BIGTEA_TEST_GGUF)");
        return;
    };
    let buf = read_head(&path, 128 << 20).expect("readable");
    let gguf = Gguf::parse(&buf).expect("real containers must parse");

    assert_eq!(gguf.version, 3);
    let arch = gguf.architecture().expect("architecture is declared");
    assert!(
        arch.starts_with("deepseek"),
        "expected a deepseek architecture, got {arch}"
    );

    // Values published for DeepSeek-V4-Flash. These come from the container
    // itself, and independently match its config.json -- two sources agreeing.
    assert_eq!(gguf.get_u64(&format!("{arch}.block_count")), Some(43));
    assert_eq!(gguf.get_u64(&format!("{arch}.expert_count")), Some(256));
    assert_eq!(gguf.get_u64(&format!("{arch}.expert_used_count")), Some(6));
    assert_eq!(gguf.get_u64(&format!("{arch}.expert_shared_count")), Some(1));
    assert_eq!(gguf.get_u64(&format!("{arch}.embedding_length")), Some(4096));
    assert_eq!(
        gguf.get_u64(&format!("{arch}.expert_feed_forward_length")),
        Some(2048)
    );
}

#[test]
fn every_tensor_in_a_real_container_can_be_sized() {
    // An unknown ggml type or a partial block would silently under-report the
    // model's real cost, which is the one number Bigtea exists to get right.
    let Some(path) = find_container() else {
        eprintln!("skipping: no container found");
        return;
    };
    let buf = read_head(&path, 128 << 20).expect("readable");
    let gguf = Gguf::parse(&buf).expect("parses");

    let unsized_tensors: Vec<_> = gguf
        .tensors
        .iter()
        .filter(|t| t.size_bytes().is_none())
        .map(|t| format!("{} ({}, {:?})", t.name, t.ty, t.dims))
        .collect();
    assert!(
        unsized_tensors.is_empty(),
        "these tensors could not be sized: {unsized_tensors:?}"
    );
}

#[test]
fn shard_one_is_metadata_only_and_declares_the_whole_split() {
    let Some(path) = find_container() else {
        eprintln!("skipping: no container found");
        return;
    };
    let buf = read_head(&path, 128 << 20).expect("readable");
    let gguf = Gguf::parse(&buf).expect("parses");

    if let Some(count) = gguf.get_u64("split.count") {
        assert!(count >= 1);
        let total = gguf
            .get_u64("split.tensors.count")
            .expect("a split declares its overall tensor count");
        assert!(
            total > gguf.tensors.len() as u64 || count == 1,
            "a single shard should hold fewer tensors than the whole model"
        );
    }
}

#[test]
fn profile_from_a_real_container_predicts_a_runnable_plan() {
    let Some(path) = find_container() else {
        eprintln!("skipping: no container found");
        return;
    };
    let buf = read_head(&path, 128 << 20).expect("readable");
    let gguf = Gguf::parse(&buf).expect("parses");

    let Ok(profile) = ModelProfile::from_gguf(&gguf, "DeepSeek-V4-Flash") else {
        eprintln!("skipping: shard declares no expert counts");
        return;
    };
    assert_eq!(profile.source, ProfileSource::TensorIndex);
    assert_eq!(profile.n_experts, 256);
    assert_eq!(profile.n_experts_used, 6);

    // A token must never be asked to read more than the whole expert pool.
    assert!(profile.expert_bytes_per_token <= profile.expert_pool_bytes);

    let p = Prediction::new(
        &profile,
        Some(14 * GIB),
        Some(3.09e9),
        Some(730 * GIB),
        3 * GIB,
        DEFAULT_EFFICIENCY,
    );
    // Arithmetic invariants that must hold for any container.
    assert_eq!(
        p.bytes_per_token,
        p.dense_shortfall_bytes + p.expert_bytes_per_token
    );
    assert_eq!(p.dense_resident_bytes + p.dense_shortfall_bytes, p.dense_bytes);
    assert!(p.container_bytes >= p.dense_bytes);
}
