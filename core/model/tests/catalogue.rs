//! The catalogue is data, and wrong data here is a download button that 404s
//! or a size that lies about what will fit. None of that shows up as a compile
//! error, so it is asserted.

use chaos_model::catalogue::{find, CATALOGUE};

/// Names are what a user types. Two entries answering to one name means
/// `chaos-pull <name>` is a coin toss.
#[test]
fn every_name_is_unique() {
    let mut seen: Vec<&str> = Vec::new();
    for e in CATALOGUE {
        assert!(!seen.contains(&e.name), "two entries are called {}", e.name);
        seen.push(e.name);
    }
}

#[test]
fn every_name_resolves() {
    for e in CATALOGUE {
        assert!(
            find(e.name).is_some(),
            "{} is in the catalogue but find() misses it",
            e.name
        );
    }
}

/// A stem offering more than one quantisation must carry `{quant}`, or they all
/// resolve to the same filename and the wrong one is downloaded.
///
/// **Only when there is a choice to make.** The FLUX.2 autoencoder is one file
/// with one form -- `split_files/vae/flux2-vae.safetensors` -- and demanding a
/// `{quant}` placeholder in it would mean inventing a quantisation that does not
/// exist. The check that matters is that no two quants of the same entry can
/// collide, and with one quant there is nothing to collide with.
#[test]
fn a_stem_varies_by_quant_whenever_there_is_a_choice() {
    for e in CATALOGUE {
        if e.quants.len() > 1 {
            assert!(
                e.stem.contains("{quant}"),
                "{} offers {} quantisations but its stem {:?} does not vary with them",
                e.name,
                e.quants.len(),
                e.stem
            );
        }
        // And whatever the count, distinct quants must give distinct filenames.
        let mut seen: Vec<Vec<String>> = Vec::new();
        for q in e.quants {
            let files = e.files(q);
            assert!(
                !seen.contains(&files),
                "{}: two quantisations resolve to the same files {:?}",
                e.name,
                files
            );
            seen.push(files);
        }
    }
}

/// Filenames must actually differ between quantisations of one model.
#[test]
fn quants_of_one_model_produce_different_files() {
    for e in CATALOGUE {
        let mut names: Vec<String> = Vec::new();
        for q in e.quants {
            let f = e.files(q).join(",");
            assert!(
                !names.contains(&f),
                "{} produces {f} for two quants",
                e.name
            );
            names.push(f);
        }
    }
}

/// A shard count must agree with how many filenames are generated, or the
/// downloader fetches a container it cannot open.
#[test]
fn shard_counts_match_the_file_list() {
    for e in CATALOGUE {
        for q in e.quants {
            let files = e.files(q);
            let expected = q.shards.max(1) as usize;
            assert_eq!(
                files.len(),
                expected,
                "{} {} lists {} files for {} shards",
                e.name,
                q.name,
                files.len(),
                q.shards
            );
            if q.shards > 1 {
                assert!(
                    files[0].contains(&format!("-00001-of-{:05}", q.shards)),
                    "{}",
                    files[0]
                );
            }
        }
    }
}

/// **The number that decides whether a model runs.** It can never exceed the
/// download, and zero would make the app claim anything fits.
#[test]
fn the_always_read_set_is_sane() {
    for e in CATALOGUE {
        for q in e.quants {
            assert!(q.bytes > 0, "{} {} has no size", e.name, q.name);
            assert!(
                q.always_read_bytes > 0,
                "{} {} claims nothing has to stay resident",
                e.name,
                q.name
            );
            assert!(
                q.always_read_bytes <= q.bytes,
                "{} {} needs {} resident out of a {} download",
                e.name,
                q.name,
                q.always_read_bytes,
                q.bytes
            );
        }
    }
}

/// A dense model has no routed experts, so all of it is always-read. Claiming
/// otherwise would promise streaming that cannot happen.
#[test]
fn dense_models_are_entirely_resident() {
    for e in CATALOGUE {
        if e.arch.contains("moe") || e.arch == "deepseek4" {
            continue;
        }
        for q in e.quants {
            assert_eq!(
                q.always_read_bytes, q.bytes,
                "{} is dense ({}), so every byte is always-read",
                e.name, e.arch
            );
        }
    }
}

/// Only the MoE entries may claim a resident set smaller than the download --
/// and they must, or there is no reason to stream them.
#[test]
fn the_moe_entries_actually_stream() {
    for e in CATALOGUE {
        if !(e.arch.contains("moe") || e.arch == "deepseek4") {
            continue;
        }
        for q in e.quants {
            assert!(
                q.always_read_bytes < q.bytes,
                "{} is MoE but claims to need all {} bytes resident",
                e.name,
                q.bytes
            );
        }
    }
}

/// Repos are `owner/name` on Hugging Face; a bare name silently builds a URL
/// that cannot resolve.
#[test]
fn repos_look_like_hugging_face_repos() {
    for e in CATALOGUE {
        assert_eq!(
            e.repo.matches('/').count(),
            1,
            "{} has repo {:?}, which is not owner/name",
            e.name,
            e.repo
        );
        assert!(
            !e.repo.starts_with("http"),
            "{} should be owner/name, not a URL",
            e.name
        );
    }
}

/// Enough models to be worth calling a catalogue, and the two the project is
/// actually about are present.
#[test]
fn the_headline_models_are_offered() {
    assert!(
        CATALOGUE.len() >= 10,
        "only {} models offered",
        CATALOGUE.len()
    );
    assert!(find("v4flash").is_some());
    assert!(find("qwen3-32b").is_some());
    assert!(find("gemma3-27b").is_some());
}
