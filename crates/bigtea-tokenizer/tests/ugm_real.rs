//! Unigram against a real T5 container, checked against llama.cpp.
//!
//! Captured with `llama-tokenize -m flan-t5-small.Q8_0.gguf -p "<text>"`.
//!
//! **llama-tokenize prints no trailing `</s>`**, while this container declares
//! `add_eos_token = true` and Bigtea honours it — T5's encoder input ends with
//! `</s>`, which is how the model was trained. So the expectations below are the
//! oracle's ids plus id 1, and the difference is stated here rather than
//! silently absorbed by loosening the test.

use std::path::PathBuf;

use bigtea_model::Model;
use bigtea_tokenizer::{Kind, Tokenizer};

const DEFAULT_PATH: &str = r"C:\Projects\models\t5\flan-t5-small.Q8_0.gguf";
const EOS: u32 = 1;

fn tokenizer() -> Option<Tokenizer> {
    let p = std::env::var("BIGTEA_TEST_T5")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_PATH));
    if !p.exists() {
        return None;
    }
    let model = Model::open_split(&p).expect("open model");
    Some(Tokenizer::from_metadata(model.metadata()).expect("build tokenizer"))
}

#[test]
fn loads_a_t5_container_as_unigram() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no T5 model");
        return;
    };
    assert_eq!(tk.kind(), Kind::Ugm);
    assert_eq!(tk.vocab_size(), 32_128);
    assert_eq!(tk.eos, Some(EOS));
    assert!(!tk.add_bos, "T5 declares add_bos_token = false");
    assert!(tk.add_eos, "...and add_eos_token = true");
}

#[test]
fn matches_llama_cpp_token_for_token() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no T5 model");
        return;
    };
    let cases: &[(&str, &[u32])] = &[
        (
            "The capital of France is Paris.",
            &[37, 1784, 13, 1410, 19, 1919, 5, EOS],
        ),
        (
            "translate English to German: How old are you?",
            &[13959, 1566, 12, 2968, 10, 571, 625, 33, 25, 58, EOS],
        ),
        ("tokenization", &[14145, 1707, EOS]),
        ("Hello, World! 42", &[8774, 6, 1150, 55, 6426, EOS]),
        // Runs of whitespace collapse, and trailing whitespace adds no token.
        ("  spaced   out  ", &[628, 26, 91, EOS]),
    ];
    let mut wrong = Vec::new();
    for (text, want) in cases {
        let got = tk.encode(text);
        if got != *want {
            wrong.push(format!("  {text:?}\n    want {want:?}\n    got  {got:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "diverged from llama.cpp on {} of {} cases:\n{}",
        wrong.len(),
        cases.len(),
        wrong.join("\n")
    );
}

/// Unigram keeps the text; only the leading boundary and collapsed runs go.
#[test]
fn round_trips_ordinary_text() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no T5 model");
        return;
    };
    for text in [
        "The capital of France is Paris.",
        "translate English to German: How old are you?",
        "Hello, World! 42",
    ] {
        assert_eq!(tk.decode(&tk.encode(text)), text, "round trip changed it");
    }
}
