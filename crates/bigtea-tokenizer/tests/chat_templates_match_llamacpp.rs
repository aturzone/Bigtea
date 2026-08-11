//! Every chat template, rendered and compared against llama.cpp's own output.
//!
//! # Why a captured fixture and not a hand-written expectation
//!
//! A chat template that is subtly wrong does not fail. The model answers, in
//! fluent English, having been handed a framing it was never trained on — it
//! comments on the question instead of answering it, or answers the system
//! prompt. That is this project's most expensive failure mode and it is
//! invisible to any test that checks "did it produce a string".
//!
//! So the expectation is not mine. `scripts/capture-chat-templates.py` runs
//! llama.cpp with `--verbose-prompt` and reconstructs, token by token, the
//! exact prompt it builds for every template it knows. This test replays those
//! renderings. "Bigtea supports `gpt-oss`" then means "byte-identical to what
//! llama.cpp produced, on a recorded command line" rather than "it looked
//! right".
//!
//! Regenerate with:
//!
//! ```text
//! python scripts/capture-chat-templates.py > crates/bigtea-tokenizer/tests/chat-templates.txt
//! ```

use bigtea_tokenizer::chat::{ChatFormat, Message};

/// The capture used these two messages; the fixture is meaningless with others.
const SYSTEM: &str = "SYS";
const USER: &str = "HI";

fn fixture() -> Vec<(String, String)> {
    include_str!("chat-templates.txt")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .filter_map(|l| l.split_once('\t'))
        .map(|(name, body)| (name.to_string(), unescape(body)))
        .collect()
}

/// Undo the fixture's escaping. Backslash last would double-unescape a literal
/// `\\n`, so the scan is single-pass.
fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[test]
fn the_fixture_is_present_and_covers_llamacpp() {
    let f = fixture();
    // 54 is what llama.cpp's table held when this was captured. A smaller
    // number means the capture failed partway and wrote a truncated file,
    // which would make every missing template silently "pass".
    assert_eq!(
        f.len(),
        54,
        "fixture is truncated -- re-run scripts/capture-chat-templates.py"
    );
    assert!(f.iter().any(|(n, _)| n == "chatml"));
}

#[test]
fn every_template_bigtea_claims_matches_llamacpp_exactly() {
    let messages = [Message::new("system", SYSTEM), Message::new("user", USER)];

    // Two of llama.cpp's templates use bytes the capture model's tokenizer
    // cannot round-trip, so the fixture holds U+FFFD where the real template
    // has a private-use byte. Comparing against a corrupted expectation would
    // be worse than not comparing: it would bake the corruption in.
    let mangled = |body: &str| body.contains('\u{fffd}');

    let mut checked = 0;
    let mut unsupported: Vec<String> = Vec::new();
    let mut wrong: Vec<String> = Vec::new();

    for (name, expected) in fixture() {
        if expected.starts_with("DECLINED") || mangled(&expected) {
            continue;
        }
        let Some(fmt) = ChatFormat::from_name(&name) else {
            unsupported.push(name);
            continue;
        };
        // llama.cpp's hardcoded renderers have no access to the vocabulary, so
        // every separator they emit is a literal. `eos` is therefore only
        // consulted by the families that genuinely embed one, and the fixture
        // is what decides which those are.
        let got = fmt.apply(&messages, "", true);
        if got != expected {
            wrong.push(format!(
                "\n  {name}\n    want {expected:?}\n    got  {got:?}"
            ));
        }
        checked += 1;
    }

    assert!(
        wrong.is_empty(),
        "{} of {checked} templates disagree with llama.cpp:{}",
        wrong.len(),
        wrong.join("")
    );
    // Not an assertion that the list is empty -- it is a record of the gap, and
    // it shrinks as families land. A regression that *removes* support shows up
    // as this number growing.
    assert!(
        unsupported.len() <= 2,
        "unsupported templates grew to {}: {unsupported:?}",
        unsupported.len()
    );
}
