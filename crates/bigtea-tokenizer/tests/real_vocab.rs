//! The tokenizer against the model's own vocabulary.
//!
//! Unit tests cover the pieces; this checks the thing that actually matters —
//! that real text encodes to sensible ids against a real 129,280-token
//! vocabulary and decodes back unchanged. A tokenizer that is subtly wrong
//! does not fail loudly: it yields fluent nonsense that looks like a broken
//! forward pass, so it is worth pinning down before the forward pass exists.

use std::path::PathBuf;

use bigtea_model::Model;
use bigtea_tokenizer::Tokenizer;

const DEFAULT_PATH: &str =
    r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf";

fn tokenizer() -> Option<Tokenizer> {
    let p = std::env::var("BIGTEA_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_PATH));
    if !p.exists() {
        return None;
    }
    let model = Model::open_split(&p).expect("open model");
    Some(Tokenizer::from_metadata(model.metadata()).expect("build tokenizer"))
}

#[test]
fn loads_the_real_vocabulary() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no model");
        return;
    };
    assert_eq!(tk.vocab_size(), 129_280, "unexpected vocabulary size");
    assert_eq!(tk.bos, Some(0));
    assert_eq!(tk.eos, Some(1));
    // This model declares both false; honouring them matters because adding an
    // unwanted BOS shifts every position by one.
    assert!(!tk.add_bos);
    assert!(!tk.add_eos);
}

#[test]
fn round_trips_real_text() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no model");
        return;
    };
    for text in [
        "The capital of France is",
        "Hello, world!",
        "def fibonacci(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
        "Numbers: 42, 1337, 2026.",
        "  leading spaces",
        "multi\nline\ntext",
    ] {
        let ids = tk.encode(text);
        assert!(!ids.is_empty(), "encoded {text:?} to nothing");
        assert!(
            ids.iter().all(|&id| (id as usize) < tk.vocab_size()),
            "produced an out-of-range id for {text:?}"
        );
        let decoded = tk.decode(&ids);
        assert_eq!(decoded, text, "round trip changed {text:?}");
    }
}

#[test]
fn encoding_is_compact_not_byte_per_token() {
    // The whole point of BPE: common words become single tokens. If merges
    // were being missed, this would degenerate to roughly one token per byte.
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no model");
        return;
    };
    let text = "The capital of France is Paris and the capital of Germany is Berlin";
    let ids = tk.encode(text);
    eprintln!("{} chars -> {} tokens", text.len(), ids.len());
    assert!(
        ids.len() < text.len() / 3,
        "expected strong compression, got {} tokens for {} chars",
        ids.len(),
        text.len()
    );
}

#[test]
fn common_words_are_single_tokens() {
    // A direct check that merges are being applied: " the" should exist in the
    // vocabulary as one token, and encode as one id.
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no model");
        return;
    };
    let ids = tk.encode(" the");
    assert_eq!(
        ids.len(),
        1,
        "' the' encoded to {ids:?}, expected one token"
    );
}

#[test]
fn unicode_survives_the_round_trip() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no model");
        return;
    };
    for text in ["café", "naïve — dash", "日本語", "emoji: 🚀"] {
        let ids = tk.encode(text);
        assert_eq!(tk.decode(&ids), text, "round trip failed for {text:?}");
    }
}

#[test]
fn special_tokens_are_present_and_addressable() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: no model");
        return;
    };
    let bos = tk.bos.expect("bos declared");
    let text = tk.token_text(bos).expect("bos has text");
    assert!(
        text.contains("begin") || text.contains('<'),
        "bos token text looks wrong: {text:?}"
    );
}

/// SentencePiece round trip against the real TinyLlama vocabulary.
///
/// A wrong tokenizer never crashes, so the only useful check is that text
/// survives a round trip and that streaming decode matches whole-sequence
/// decode. Ignored by default: it needs a container on disk.
#[test]
#[ignore = "needs models/tinyllama"]
fn spm_round_trips_real_text_and_streams_the_same() {
    let path =
        std::path::Path::new("C:/Projects/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let model = bigtea_model::Model::open_split(path).expect("open");
    let tk = bigtea_tokenizer::Tokenizer::from_metadata(model.metadata()).expect("tokenizer");
    assert_eq!(tk.kind(), bigtea_tokenizer::Kind::Spm);

    for text in [
        "The capital of France is Paris.",
        "hello world",
        "a b  c",
        "def fib(n):\n    return n",
        "Ünïcödé and emoji \u{1F600}",
    ] {
        let ids = tk.encode(text);
        assert!(!ids.is_empty(), "{text:?} encoded to nothing");
        let back = tk.decode(&ids);
        assert_eq!(back, text, "round trip failed for {text:?} -> {ids:?}");

        // Decoding one token at a time -- what generation does -- must
        // concatenate to the same text. The invariant is on BYTES, not on
        // Strings: one character is often several tokens, so a per-token
        // String conversion would replace every incomplete fragment with a
        // replacement character and lose it permanently.
        let mut streamed = Vec::new();
        for &id in &ids {
            streamed.extend(tk.decode_bytes(std::slice::from_ref(&id)));
        }
        let streamed = String::from_utf8(streamed).expect("streamed bytes are valid UTF-8");
        assert_eq!(
            streamed.strip_prefix(' ').unwrap_or(&streamed),
            text,
            "streaming decode disagreed with whole-sequence decode for {text:?}"
        );
    }
}
