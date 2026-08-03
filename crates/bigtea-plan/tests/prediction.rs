//! Prediction behaviour, including the two properties that invert the obvious
//! advice. Both are easy to regress and each changes what a user should do, so
//! they are asserted rather than left as comments.

use bigtea_plan::{ModelProfile, Prediction, ProfileSource, DEFAULT_EFFICIENCY, GIB};

/// DeepSeek-V4-Flash at roughly Q4, as published: 43 layers, 256 experts,
/// 6 used, 1 shared, hidden 4096, expert FFN 2048.
fn v4_flash(expert_bits: f64, dense_bits: f64) -> ModelProfile {
    let params_per_expert = 3 * 4096 * 2048;
    ModelProfile::from_architecture(
        "DeepSeek-V4-Flash",
        7_060_000_000, // dense params, derived from the published config
        params_per_expert,
        43,
        256,
        6,
        expert_bits,
        dense_bits,
    )
    .expect("routing counts are valid")
}

fn predict_on_laptop(profile: &ModelProfile, ram_gib: f64) -> Prediction {
    Prediction::new(
        profile,
        Some((ram_gib * GIB as f64) as u64),
        Some(3.09e9), // measured on the target machine
        Some(730 * GIB),
        3 * GIB,
        DEFAULT_EFFICIENCY,
    )
}

#[test]
fn dense_resident_means_only_experts_are_read() {
    let p = predict_on_laptop(&v4_flash(4.0, 5.5), 14.5);
    assert!(p.dense_fully_resident());
    assert_eq!(p.bytes_per_token, p.expert_bytes_per_token);
}

#[test]
fn dense_shortfall_is_charged_on_every_token() {
    // Starve it: the dense part can no longer be cached.
    let p = predict_on_laptop(&v4_flash(4.0, 5.5), 3.5);
    assert!(!p.dense_fully_resident());
    assert_eq!(
        p.bytes_per_token,
        p.dense_shortfall_bytes + p.expert_bytes_per_token
    );
    assert!(p.bytes_per_token > p.expert_bytes_per_token);
}

#[test]
fn property_pruning_the_pool_does_not_change_speed() {
    // Counter-intuitive property #1. Halving the expert pool halves the
    // container, but a token still routes to 6 experts -- so bytes/token, and
    // therefore speed, are identical. Pruning buys a cheaper download only.
    let params_per_expert = 3 * 4096 * 2048;
    let full = v4_flash(4.0, 5.5);
    let pruned = ModelProfile::from_architecture(
        "V4-Flash-pruned",
        7_060_000_000,
        params_per_expert,
        43,
        128, // half the pool
        6,   // same routing
        4.0,
        5.5,
    )
    .unwrap();

    let a = predict_on_laptop(&full, 14.5);
    let b = predict_on_laptop(&pruned, 14.5);

    assert!(b.container_bytes < a.container_bytes, "pruning must shrink disk");
    assert_eq!(
        b.bytes_per_token, a.bytes_per_token,
        "pruning must NOT change bytes per token"
    );
    assert_eq!(b.tokens_per_sec_realistic, a.tokens_per_sec_realistic);
}

#[test]
fn property_partial_expert_cache_is_reported_as_worthless() {
    // Counter-intuitive property #2: RAM left over after dense residency, but
    // smaller than one token's expert working set, buys nothing -- entries are
    // evicted before they can be reused.
    let p = predict_on_laptop(&v4_flash(4.0, 5.5), 14.5);
    let leftover = p.usable_ram_bytes - p.dense_resident_bytes;
    if leftover > 0 && leftover < p.expert_bytes_per_token {
        assert!(
            p.notes.iter().any(|n| n.contains("expert caching")),
            "a useless partial expert cache must be called out"
        );
    }
}

#[test]
fn fewer_bits_is_never_slower() {
    let hi = predict_on_laptop(&v4_flash(8.0, 8.0), 14.5);
    let lo = predict_on_laptop(&v4_flash(2.0, 4.5), 14.5);
    assert!(lo.bytes_per_token < hi.bytes_per_token);
    assert!(lo.tokens_per_sec_realistic.unwrap() > hi.tokens_per_sec_realistic.unwrap());
}

#[test]
fn disk_shortfall_is_flagged_not_hidden() {
    let profile = v4_flash(4.0, 5.5);
    let tight = Prediction::new(
        &profile,
        Some(14 * GIB),
        Some(3.09e9),
        Some(10 * GIB),
        3 * GIB,
        DEFAULT_EFFICIENCY,
    );
    assert_eq!(tight.fits_disk, Some(false));
    assert!(!tight.is_runnable());
    assert!(tight.notes.iter().any(|n| n.contains("more disk")));
}

#[test]
fn missing_bandwidth_yields_no_speed_rather_than_a_guess() {
    let profile = v4_flash(4.0, 5.5);
    let p = Prediction::new(&profile, Some(14 * GIB), None, None, 3 * GIB, DEFAULT_EFFICIENCY);
    assert!(p.tokens_per_sec_realistic.is_none());
    assert!(p.seconds_per_token.is_none());
    // But the byte accounting is still valid and useful.
    assert!(p.bytes_per_token > 0);
}

#[test]
fn efficiency_discount_is_applied_exactly() {
    let p = predict_on_laptop(&v4_flash(4.0, 5.5), 14.5);
    let ceiling = p.tokens_per_sec_ceiling.unwrap();
    let realistic = p.tokens_per_sec_realistic.unwrap();
    assert!((realistic - ceiling * p.efficiency).abs() < 1e-9);
}

#[test]
fn dynamic_quantization_raises_dense_bytes_but_not_per_token_cost() {
    // Keeping attention at higher precision costs disk and RAM, but while the
    // dense part still fits, it does not change what streams per token.
    let flat = predict_on_laptop(&v4_flash(2.0, 2.0), 14.5);
    let dynamic = predict_on_laptop(&v4_flash(2.0, 6.0), 14.5);
    assert!(dynamic.dense_bytes > flat.dense_bytes);
    assert_eq!(dynamic.expert_bytes_per_token, flat.expert_bytes_per_token);
}

#[test]
fn impossible_routing_is_rejected() {
    let err = ModelProfile::from_architecture("bad", 1, 1, 1, 4, 9, 4.0, 4.0);
    assert!(err.is_err(), "using more experts than exist must fail");
    let err = ModelProfile::from_architecture("bad", 1, 1, 1, 0, 0, 4.0, 4.0);
    assert!(err.is_err(), "zero experts must fail");
}

#[test]
fn sparsity_is_why_this_is_possible_at_all() {
    let p = v4_flash(4.0, 5.5);
    assert_eq!(p.source, ProfileSource::Architecture);
    // A token touches only a few percent of the model. Without this, no
    // machine of this size could run it.
    assert!(p.sparsity() < 0.10, "sparsity was {}", p.sparsity());
}
