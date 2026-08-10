//! WordPiece against a real BERT container, checked token-for-token against
//! llama.cpp.
//!
//! Unit tests in `wpm.rs` cover the pieces. This covers the thing that decides
//! whether the implementation is right: **the exact ids llama.cpp produces**,
//! from `llama-tokenize -m all-MiniLM-L6-v2.Q8_0.gguf -p "<text>"`.
//!
//! A tokenizer that is subtly wrong never crashes. It shifts a boundary, the
//! model predicts a fluent continuation of the wrong tokens, and the output
//! looks like a broken forward pass. Every case below was chosen because it can
//! only pass if a specific rule is right:
//!
//! | text | what it pins |
//! |---|---|
//! | `Paris.` | lowercasing, and `.` split off |
//! | `tokenization` | `##` continuation |
//! | `café naïve` | accents dropped, not turned into `[UNK]` |
//! | `北京大学` | one word per CJK character |
//! | `hello 🦄 world` | an uncoverable word is **one** `[UNK]` |
//! | `Ω≈ç√` | non-ASCII symbols do *not* split |
//! | `3.14159` | digits split on the punctuation, not per digit |

use std::path::PathBuf;

use bigtea_model::Model;
use bigtea_tokenizer::{Kind, Tokenizer};

const DEFAULT_PATH: &str = r"C:\Projects\models\bert\all-MiniLM-L6-v2.Q8_0.gguf";

fn tokenizer() -> Option<Tokenizer> {
    let p = std::env::var("BIGTEA_TEST_BERT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_PATH));
    if !p.exists() {
        return None;
    }
    let model = Model::open_split(&p).expect("open model");
    Some(Tokenizer::from_metadata(model.metadata()).expect("build tokenizer"))
}

#[test]
fn loads_a_bert_container_as_wordpiece() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no BERT model");
        return;
    };
    assert_eq!(tk.kind(), Kind::Wpm);
    assert_eq!(tk.vocab_size(), 30_522);
    assert_eq!(tk.bos, Some(101), "[CLS]");
    assert_eq!(tk.eos, Some(102), "[SEP]");
    // Neither flag is in the container, and both must still be on.
    assert!(tk.add_bos, "BERT wraps every sequence in [CLS]");
    assert!(tk.add_eos, "...and closes it with [SEP]");
}

#[test]
fn matches_llama_cpp_token_for_token() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no BERT model");
        return;
    };
    // Captured from llama-tokenize on this exact container.
    let cases: &[(&str, &[u32])] = &[
        (
            "The capital of France is Paris.",
            &[101, 1996, 3007, 1997, 2605, 2003, 3000, 1012, 102],
        ),
        ("tokenization", &[101, 19204, 3989, 102]),
        ("unbelievability", &[101, 4895, 8671, 2666, 3567, 8553, 102]),
        (
            "Hello, World! 42 items.",
            &[101, 7592, 1010, 2088, 999, 4413, 5167, 1012, 102],
        ),
        ("café naïve", &[101, 7668, 15743, 102]),
        ("北京大学", &[101, 1781, 1755, 1810, 1817, 102]),
        ("don't", &[101, 2123, 1005, 1056, 102]),
        (
            "supercalifragilisticexpialidocious",
            &[
                101, 3565, 9289, 10128, 29181, 24411, 4588, 10288, 19312, 21273, 10085, 6313, 102,
            ],
        ),
        ("  spaced   out  ", &[101, 19835, 2041, 102]),
        ("hello 🦄 world", &[101, 7592, 100, 2088, 102]),
        ("Ω≈ç√", &[101, 1179, 30133, 2278, 30127, 102]),
        ("ABC", &[101, 5925, 102]),
        ("3.14159", &[101, 1017, 1012, 15471, 28154, 102]),
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

/// WordPiece destroys case and accents, so the round trip is lossy **by
/// construction**. Asserting equality here would be asserting the algorithm is
/// something other than what it is; what must hold is that the words survive.
#[test]
fn decodes_to_the_normalised_text() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no BERT model");
        return;
    };
    assert_eq!(
        tk.decode(&tk.encode("The capital of France is Paris.")),
        "the capital of france is paris ."
    );
    assert_eq!(tk.decode(&tk.encode("tokenization")), "tokenization");
}
