//! The model resolver against the real, still-downloading DeepSeek container.
//!
//! Skips cleanly when the model is absent; set `CHAOS_TEST_GGUF` to point at
//! any shard.

use std::path::PathBuf;

use chaos_io::IoMode;
use chaos_model::{discover_shards, Error, Model};

const DEFAULT_PATH: &str =
    r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf";

fn shard_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CHAOS_TEST_GGUF") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = PathBuf::from(DEFAULT_PATH);
    p.exists().then_some(p)
}

fn model() -> Option<Model> {
    let path = shard_path()?;
    Some(Model::open_split(&path).expect("real model must open"))
}

#[test]
fn discovers_sibling_shards_from_one_path() {
    let Some(path) = shard_path() else {
        eprintln!("skipping: no container");
        return;
    };
    let shards = discover_shards(&path);
    assert!(
        shards.len() > 1,
        "a split model should discover its siblings, found {}",
        shards.len()
    );
    // Discovered in shard order, so index N is shard N.
    for pair in shards.windows(2) {
        assert!(pair[0] < pair[1], "shards must come back in order");
    }
}

#[test]
fn resolves_a_partially_downloaded_model() {
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    assert_eq!(m.architecture(), "deepseek4");
    assert!(m.tensor_count() > 1000, "expected the full tensor index");

    // The key property: an incomplete download still yields a complete index
    // for the shards present, because GGUF puts the index at the front.
    let (available, total) = m.availability();
    assert!(available <= total);
    assert!(total > 0);
}

#[test]
fn routing_metadata_matches_the_published_architecture() {
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    assert_eq!(m.arch_u64("expert_count"), Some(256));
    assert_eq!(m.arch_u64("expert_used_count"), Some(6));
    assert_eq!(m.arch_u64("expert_shared_count"), Some(1));
    assert_eq!(m.arch_u64("block_count"), Some(43));
}

#[test]
fn experts_dominate_the_container_and_dense_does_not() {
    // This ratio is the entire reason a 138 GiB model can run on 16 GiB: the
    // part that must be resident is small, and the huge part is read sparsely.
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    let (expert, dense) = m.expert_vs_dense_bytes();
    assert!(expert > 0 && dense > 0);
    assert!(
        expert > dense * 10,
        "experts ({expert}) should dwarf dense ({dense})"
    );
}

#[test]
fn reads_a_real_tensor_and_gets_its_exact_size() {
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    let Some(name) = m
        .tensor_names()
        .find(|n| m.is_available(n).unwrap_or(false))
        .map(str::to_string)
    else {
        eprintln!("skipping: no tensor fully downloaded yet");
        return;
    };

    let loc = m.location(&name).expect("located").clone();
    let bytes = m.read_tensor(&name).expect("read");
    assert_eq!(
        bytes.len() as u64,
        loc.size,
        "read must return exactly the tensor's stored size"
    );
    assert!(
        !bytes.iter().all(|&b| b == 0),
        "a real tensor should not be all zeros"
    );
}

#[test]
fn reading_an_undownloaded_tensor_fails_loudly() {
    // Returning zeros or truncated data here would corrupt inference silently,
    // which is far worse than refusing.
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    let Some(name) = m
        .tensor_names()
        .find(|n| !m.is_available(n).unwrap_or(true))
        .map(str::to_string)
    else {
        eprintln!("skipping: everything is downloaded");
        return;
    };
    match m.read_tensor(&name) {
        Err(Error::NotDownloaded { .. }) => {}
        Err(other) => panic!("wrong error for missing data: {other}"),
        Ok(_) => panic!("read succeeded for a tensor that is not on disk"),
    }
}

#[test]
fn unknown_tensor_names_are_rejected() {
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    assert!(matches!(
        m.read_tensor("no.such.tensor"),
        Err(Error::UnknownTensor(_))
    ));
}

#[test]
fn direct_io_is_used_for_the_real_model() {
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    assert_eq!(
        m.io_mode(),
        IoMode::Direct,
        "streaming a model larger than RAM through the page cache would \
         double-buffer and make every cache measurement the kernel's"
    );
}

#[test]
fn every_tensor_location_is_inside_its_shard_index() {
    // A bad data_offset would put every read at the wrong place -- silently, if
    // the bytes happen to parse. Offsets must at least be plausible.
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    for name in m.tensor_names().take(200) {
        let loc = m.location(name).expect("located");
        assert!(loc.size > 0, "{name} has zero size");
        assert!(
            loc.shard < m.shard_count(),
            "{name} points at a missing shard"
        );
        assert!(loc.file_offset > 0, "{name} starts before the data section");
    }
}
