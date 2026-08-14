//! RWKV through the public API, on a hand-built vocabulary.
//!
//! # Why hand-built
//!
//! There is no RWKV container on this machine. The alternative to a synthetic
//! vocabulary is claiming a tokenizer family nobody has exercised — which this
//! project has already done once, with `gemma2` sitting in
//! `VERIFIED_ARCHITECTURES` while running the wrong activation.
//!
//! So this goes through `from_metadata`, the same path a real container takes,
//! rather than poking at internals. What it cannot prove is agreement with
//! llama.cpp on a real RWKV model; that needs the container, and until someone
//! runs it the family is *implemented*, not *verified*.

use std::collections::BTreeMap;

use bigtea_gguf::Value;
use bigtea_tokenizer::Tokenizer;

/// A container's metadata, with the vocabulary stored **escaped** as RWKV does.
fn meta(tokens: &[&str]) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert(
        "tokenizer.ggml.model".to_string(),
        Value::String("rwkv".into()),
    );
    m.insert(
        "tokenizer.ggml.tokens".to_string(),
        Value::Array(tokens.iter().map(|t| Value::String((*t).into())).collect()),
    );
    m
}

#[test]
fn a_container_declaring_rwkv_builds() {
    let t = Tokenizer::from_metadata(&meta(&["a", "b"])).unwrap();
    assert_eq!(t.vocab_size(), 2);
}

#[test]
fn an_escaped_newline_in_the_vocabulary_matches_a_real_one_in_the_text() {
    // The failure this guards: a loader that keeps the stored text builds a
    // trie keyed on a backslash and an `n`, so a real newline never matches
    // and every line break becomes an unknown token. Fluent, different, silent.
    let t = Tokenizer::from_metadata(&meta(&["hello", "\\n", "world"])).unwrap();
    let ids = t.encode("hello\nworld");
    assert_eq!(
        ids,
        vec![0, 1, 2],
        "escaped newline did not match a real one"
    );
}

#[test]
fn decoding_unescapes_too() {
    // The inverse mistake: emitting the stored text would put a literal
    // backslash-n in the output where the model produced a newline.
    let t = Tokenizer::from_metadata(&meta(&["hello", "\\n", "world"])).unwrap();
    let out = String::from_utf8(t.decode_bytes(&[0, 1, 2])).unwrap();
    assert_eq!(out, "hello\nworld");
}

#[test]
fn the_longest_entry_wins_at_each_position() {
    // Greedy longest match is the whole algorithm; taking the first match
    // instead would tokenize "abc" as three tokens and still look reasonable.
    let t = Tokenizer::from_metadata(&meta(&["a", "ab", "abc"])).unwrap();
    assert_eq!(t.encode("abc"), vec![2]);
    assert_eq!(t.encode("abx"), vec![1]);
}

#[test]
fn a_hex_escape_denotes_a_byte_that_is_not_valid_utf8_alone() {
    // `\xff` cannot be a `char`, which is why the unescape works in bytes.
    // A String-based implementation would lose it or replace it with U+FFFD.
    let t = Tokenizer::from_metadata(&meta(&["\\xff\\xfe"])).unwrap();
    assert_eq!(t.decode_bytes(&[0]), vec![0xff, 0xfe]);
}

#[test]
fn round_tripping_multibyte_text_is_byte_exact() {
    let t = Tokenizer::from_metadata(&meta(&["\u{4f60}\u{597d}", "!"])).unwrap();
    let ids = t.encode("\u{4f60}\u{597d}!");
    assert_eq!(ids, vec![0, 1]);
    assert_eq!(
        String::from_utf8(t.decode_bytes(&ids)).unwrap(),
        "\u{4f60}\u{597d}!"
    );
}
