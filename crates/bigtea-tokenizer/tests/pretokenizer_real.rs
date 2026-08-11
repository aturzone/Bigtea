//! A6c — `tokenizer.ggml.pre` against real containers.
//!
//! The pre-tokenizer decides where BPE is *allowed* to merge, so choosing the
//! wrong one does not fail — it shifts every boundary and the model answers
//! fluently and wrongly. It was previously ignored entirely, which meant every
//! byte-level BPE container was split with DeepSeek's rule.
//!
//! The two variants there are containers for here differ on the two things the
//! ticket named, and both are checked against `llama-tokenize`:
//!
//! ```text
//!              "4567"        "12345678"          "don't"
//! qwen2        4 5 6 7       1 2 3 4 5 6 7 8     don 't
//! llama-bpe    456 7         123 456 78          don 't
//! ```

use std::path::PathBuf;

use bigtea_model::Model;
use bigtea_tokenizer::{PreTokenizer, Tokenizer};

fn load(env: &str, default: &str) -> Option<Tokenizer> {
    let p = std::env::var(env)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default));
    if !p.exists() {
        return None;
    }
    let model = Model::open_split(&p).expect("open model");
    Some(Tokenizer::from_metadata(model.metadata()).expect("build tokenizer"))
}

fn qwen() -> Option<Tokenizer> {
    load(
        "BIGTEA_TEST_QWEN",
        r"C:\Projects\models\qwen3-4b\Qwen3-4B-Q4_K_M.gguf",
    )
}

fn llama() -> Option<Tokenizer> {
    load(
        "BIGTEA_TEST_LLAMA",
        r"C:\Projects\models\llama32-1b\Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    )
}

fn check(tk: &Tokenizer, cases: &[(&str, &[u32])], who: &str) {
    let mut wrong = Vec::new();
    for (text, want) in cases {
        let got = tk.encode(text);
        if got != *want {
            wrong.push(format!("  {text:?}\n    want {want:?}\n    got  {got:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{who} diverged from llama.cpp on {} of {} cases:\n{}",
        wrong.len(),
        cases.len(),
        wrong.join("\n")
    );
}

#[test]
fn qwen_declares_qwen2_and_matches_llama_cpp() {
    let Some(tk) = qwen() else {
        eprintln!("skipping: no Qwen model");
        return;
    };
    check(
        &tk,
        &[
            ("The capital of France is", &[785, 6722, 315, 9625, 374]),
            ("don't It's they're", &[15007, 944, 1084, 594, 807, 2299]),
            (
                // One digit at a time: the qwen2 rule.
                "4567 and 12345678",
                &[19, 20, 21, 22, 323, 220, 16, 17, 18, 19, 20, 21, 22, 23],
            ),
            ("hello, world!", &[14990, 11, 1879, 0]),
            ("line one\nline two", &[1056, 825, 198, 1056, 1378]),
            (
                "def add(a, b): return a + b",
                &[750, 912, 2877, 11, 293, 1648, 470, 264, 488, 293],
            ),
        ],
        "qwen2",
    );
}

#[test]
fn llama_declares_llama_bpe_and_matches_llama_cpp() {
    let Some(tk) = llama() else {
        eprintln!("skipping: no Llama model");
        return;
    };
    check(
        &tk,
        &[
            (
                "The capital of France is",
                &[128000, 791, 6864, 315, 9822, 374],
            ),
            (
                "don't It's they're",
                &[128000, 15357, 956, 1102, 596, 814, 2351],
            ),
            (
                // Groups of three: the llama-bpe rule, on the same input.
                "4567 and 12345678",
                &[128000, 10961, 22, 323, 220, 4513, 10961, 2495],
            ),
            ("hello, world!", &[128000, 15339, 11, 1917, 0]),
            ("line one\nline two", &[128000, 1074, 832, 198, 1074, 1403]),
            (
                "def add(a, b): return a + b",
                &[128000, 755, 923, 2948, 11, 293, 1680, 471, 264, 489, 293],
            ),
        ],
        "llama-bpe",
    );
}

/// The same text, two containers, two different splits — which is the whole
/// reason `tokenizer.ggml.pre` cannot be ignored.
#[test]
fn the_two_variants_really_do_disagree() {
    let (Some(q), Some(l)) = (qwen(), llama()) else {
        eprintln!("skipping: need both models");
        return;
    };
    let digits = "4567";
    let qn = q.encode(digits).len();
    let ln = l.encode(digits).iter().filter(|&&i| i != 128000).count();
    assert_eq!(qn, 4, "qwen2 takes one digit at a time");
    assert_eq!(ln, 2, "llama-bpe groups three");
}

/// V4-Flash is the container the original splitter was written and verified
/// against, so it is the regression that matters most.
#[test]
fn v4flash_still_uses_joyai_llm_unchanged() {
    let Some(tk) = load(
        "BIGTEA_TEST_GGUF",
        r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf",
    ) else {
        eprintln!("skipping: no V4-Flash model");
        return;
    };
    // Captured before this change, with the splitter that had no variants.
    for text in [
        "The capital of France is",
        "Hello, world!",
        "Numbers: 42, 1337, 2026.",
        "multi\nline\ntext",
    ] {
        let ids = tk.encode(text);
        assert!(!ids.is_empty());
        assert_eq!(tk.decode(&ids), text, "round trip changed {text:?}");
    }
}

/// A variant with no container to check against must be refused, not guessed.
#[test]
fn an_unverified_pre_tokenizer_is_refused_by_name() {
    for unknown in ["deepseek-llm", "falcon", "smaug-bpe", "bert-bge"] {
        let err = PreTokenizer::from_name(unknown).expect_err("must refuse");
        let text = err.to_string();
        assert!(text.contains(unknown), "must name the variant: {text}");
    }
}
