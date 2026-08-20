//! `--override-tensor`: what it matches, and what it refuses to guess at.
//!
//! These run without a GPU, which is the point — the parsing and the pattern
//! are where a placement rule goes wrong, and both are pure. The residency they
//! feed is exercised on hardware in `scripts/parity-check.sh` and by hand;
//! nothing here needs a card.
//!
//! **The refusals matter more than the matches.** llama.cpp's `-ot` takes a
//! `std::regex` and this build matches substrings with `*`. A user's regex
//! treated as a literal would match nothing, the flag would appear to work, and
//! the model would load exactly where it would have anyway — a flag accepted
//! and ignored, which is the failure this project's declined-flag table exists
//! to prevent. So a regex character is an error with the pattern quoted back.

use chaos_arch::TensorOverride;

fn host(pattern: &str) -> TensorOverride {
    TensorOverride::parse(&format!("{pattern}=CPU")).expect("should parse")
}

#[test]
fn a_bare_substring_matches_anywhere_in_the_name() {
    let rule = host("*_exps*");
    assert!(rule.matches_name("blk.3.ffn_gate_exps.weight"));
    assert!(!rule.matches_name("blk.3.ffn_gate.weight"));
}

#[test]
fn a_leading_segment_must_start_the_name() {
    // `blk.1*` is the common shape: "block 1, and 10-15 too". It must NOT
    // match `attn.blk.1...`, which is what a plain `contains` would do.
    let rule = host("blk.1*");
    assert!(rule.matches_name("blk.1.attn_q.weight"));
    assert!(rule.matches_name("blk.14.ffn_up.weight"));
    assert!(!rule.matches_name("blk.2.attn_q.weight"));
    assert!(!rule.matches_name("output.blk.1.weight"));
}

#[test]
fn a_trailing_segment_must_end_the_name() {
    let rule = host("*.weight");
    assert!(rule.matches_name("blk.0.attn_q.weight"));
    assert!(!rule.matches_name("blk.0.attn_q.weight.extra"));
}

#[test]
fn a_pattern_with_no_wildcard_is_an_exact_name() {
    let rule = host("token_embd.weight");
    assert!(rule.matches_name("token_embd.weight"));
    assert!(!rule.matches_name("token_embd.weight.2"));
    assert!(!rule.matches_name("x.token_embd.weight"));
}

#[test]
fn a_lone_star_matches_everything() {
    let rule = host("*");
    assert!(rule.matches_name("blk.0.attn_q.weight"));
    assert!(rule.matches_name("output.weight"));
}

/// **The refusal that keeps the flag honest.**
///
/// Every character here means something in `std::regex` and nothing here.
/// Silently matching zero tensors would be indistinguishable from not passing
/// the flag at all.
#[test]
fn a_regex_pattern_is_refused_rather_than_matched_literally() {
    for pattern in [
        r"blk\.(1[0-9])\..*_exps",
        "blk.[0-9]+.ffn",
        "^blk",
        "ffn$",
        "(a|b)",
        "a+b",
    ] {
        let err = TensorOverride::parse(&format!("{pattern}=CPU"))
            .expect_err(&format!("{pattern:?} should be refused"));
        let text = err.to_string();
        assert!(
            text.contains("regex"),
            "{pattern:?}: message does not say why: {text}"
        );
        // Compared against the DEBUG form, because that is what the message
        // prints: a backslash pattern comes back escaped, and asserting on the
        // raw text fails on exactly the patterns this refusal exists for.
        assert!(
            text.contains(&format!("{pattern:?}")),
            "{pattern:?}: message does not quote the pattern back: {text}"
        );
    }
}

#[test]
fn a_rule_without_a_target_is_refused() {
    let err = TensorOverride::parse("*_exps").expect_err("no `=` should be refused");
    assert!(err.to_string().contains("CPU"), "{err}");
}

#[test]
fn an_unknown_target_names_the_two_that_work() {
    let err = TensorOverride::parse("*_exps=Vulkan1").expect_err("device names are not supported");
    let text = err.to_string();
    assert!(text.contains("CPU"), "{text}");
    // Naming a specific device is `--split-mode`'s job, and that is declined.
    assert!(text.contains("--split-mode"), "{text}");
}

#[test]
fn an_empty_pattern_is_refused() {
    assert!(TensorOverride::parse("=CPU").is_err());
}

#[test]
fn both_target_spellings_parse() {
    assert!(!TensorOverride::parse("*=CPU").expect("cpu").on_device);
    assert!(!TensorOverride::parse("*=host").expect("host").on_device);
    assert!(TensorOverride::parse("*=GPU").expect("gpu").on_device);
    assert!(TensorOverride::parse("*=device").expect("device").on_device);
}
