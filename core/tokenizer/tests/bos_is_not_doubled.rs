//! A template that writes its own BOS must not get a second one.
//!
//! # The bug
//!
//! `encode` prepended BOS whenever the container declared `add_bos_token`, and
//! `partition_specials` separately mapped a literal `<bos>` in the text to its
//! own id. A chat template evaluated by `--jinja` usually contains that literal
//! — Gemma's starts with `<bos>`, Llama-3's with `<|begin_of_text|>` — so the
//! model was prefilled a token **long**:
//!
//! ```text
//! chaos --jinja : [2, 2, 105, 2364, ...]
//! llama.cpp      : [2,    105, 2364, ...]
//! ```
//!
//! Measured on gemma-3, Llama-3.2, internlm2 and Phi-3. It is the mirror of the
//! Falcon3 bug, which was prefilled a token *short*, and just as quiet: nothing
//! raises, and the model answers fluently from a position nobody trained.
//!
//! The hardcoded family renderers never emitted the BOS text, so this could only
//! appear once a real Jinja engine started evaluating the container's own
//! template — the feature and the bug arrived together.

use std::collections::BTreeMap;

use chaos_gguf::Value;
use chaos_tokenizer::Tokenizer;

/// A minimal SPM-ish container that declares a BOS and asks for it.
fn meta(add_bos: bool) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert(
        "tokenizer.ggml.model".to_string(),
        Value::String("llama".into()),
    );
    // id 0 `<unk>`, id 1 `<bos>`, then ordinary pieces.
    let tokens = ["<unk>", "<bos>", "hi", "there"];
    m.insert(
        "tokenizer.ggml.tokens".to_string(),
        Value::Array(tokens.iter().map(|t| Value::String((*t).into())).collect()),
    );
    m.insert(
        "tokenizer.ggml.token_type".to_string(),
        // 3 = CONTROL for the two specials, 1 = NORMAL for the rest.
        Value::Array(vec![
            Value::I32(3),
            Value::I32(3),
            Value::I32(1),
            Value::I32(1),
        ]),
    );
    // SPM picks its merge by score, so the loader requires one per token.
    m.insert(
        "tokenizer.ggml.scores".to_string(),
        Value::Array(vec![
            Value::F32(0.0),
            Value::F32(0.0),
            Value::F32(-1.0),
            Value::F32(-2.0),
        ]),
    );
    m.insert("tokenizer.ggml.bos_token_id".to_string(), Value::U32(1));
    m.insert(
        "tokenizer.ggml.add_bos_token".to_string(),
        Value::Bool(add_bos),
    );
    m
}

#[test]
fn a_text_opening_with_the_bos_token_gets_exactly_one() {
    let t = Tokenizer::from_metadata(&meta(true)).unwrap();
    let ids = t.encode("<bos>hi");
    assert_eq!(
        ids.iter().filter(|&&id| id == 1).count(),
        1,
        "BOS appears {} times in {ids:?}",
        ids.iter().filter(|&&id| id == 1).count()
    );
    assert_eq!(ids.first(), Some(&1), "BOS is not first: {ids:?}");
}

#[test]
fn a_text_without_it_still_gets_one() {
    // The guard must not cost the ordinary case its BOS -- that is the Falcon3
    // bug, which was a token short and equally silent.
    let t = Tokenizer::from_metadata(&meta(true)).unwrap();
    let ids = t.encode("hi");
    assert_eq!(ids.first(), Some(&1), "BOS was dropped: {ids:?}");
}

#[test]
fn a_container_that_declines_bos_never_gets_one_added() {
    let t = Tokenizer::from_metadata(&meta(false)).unwrap();
    // The literal in the text is still honoured -- it is a control token, and
    // suppressing it would be a different bug.
    assert_eq!(t.encode("<bos>hi").first(), Some(&1));
    assert_ne!(t.encode("hi").first(), Some(&1));
}
