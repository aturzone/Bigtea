//! Whitespace after a special token is dropped — for the models llama.cpp does
//! it for, and only those.
//!
//! # The bug, and why it hid
//!
//! A chat template writes `<|user|>\n` before the message. We kept that newline;
//! llama.cpp drops it. SPM then prefixes the following fragment with `▁`, so the
//! *next word* tokenizes differently, and one short Phi-3 turn came out as
//! fourteen tokens against llama.cpp's eight:
//!
//! ```text
//! <|user|>\nSYS\nHI<|end|>\n<|assistant|>\n
//!   chaos   : [1, 32010, 29871, 13, 14816, 29903, 13, 17628, 32007, 29871, 13, 32001, 29871, 13]
//!   llama.cpp: [1, 32010, 317, 21554, 13, 17628, 32007, 32001]
//! ```
//!
//! The parity sweep never saw it because that sweep uses plain prompts with no
//! special tokens in them. It only appears once something applies a chat
//! template — which is every real use of the server.
//!
//! # Keyed on the model's NAME, which is the part worth remembering
//!
//! `LLAMA_TOKEN_ATTR_RSTRIP` is **not in the container**. llama.cpp sets it in
//! `llama-vocab.cpp` from `_contains_any(model_name, {"phi-3", "phi3"})`, beside
//! `<mask>` LSTRIP rules keyed on `jina-v2-*` and `modern-bert`. Nothing in the
//! tokenizer metadata separates a Phi-3 vocabulary from any other SPM one, so
//! matching the reference means keying on the same string it keys on.

use std::collections::BTreeMap;

use chaos_gguf::Value;
use chaos_tokenizer::Tokenizer;

fn meta(name: &str) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert(
        "tokenizer.ggml.model".to_string(),
        Value::String("llama".into()),
    );
    let tokens = ["<unk>", "<s>", "<|user|>", "hi", "\n"];
    m.insert(
        "tokenizer.ggml.tokens".to_string(),
        Value::Array(tokens.iter().map(|t| Value::String((*t).into())).collect()),
    );
    m.insert(
        "tokenizer.ggml.token_type".to_string(),
        Value::Array(vec![
            Value::I32(3),
            Value::I32(3),
            Value::I32(3),
            Value::I32(1),
            Value::I32(1),
        ]),
    );
    m.insert(
        "tokenizer.ggml.scores".to_string(),
        Value::Array((0..5).map(|i| Value::F32(-(i as f32))).collect()),
    );
    m.insert(
        "tokenizer.ggml.add_bos_token".to_string(),
        Value::Bool(false),
    );
    m.insert("general.name".to_string(), Value::String(name.into()));
    m
}

/// The id of the newline token in the fixture above.
const NEWLINE: u32 = 4;

#[test]
fn a_phi3_named_model_drops_the_newline_after_a_special_token() {
    let t = Tokenizer::from_metadata(&meta("Phi-3-mini-4k-instruct")).unwrap();
    let ids = t.encode("<|user|>\nhi");
    assert!(
        !ids.contains(&NEWLINE),
        "the newline after <|user|> survived: {ids:?}"
    );
}

#[test]
fn any_other_model_keeps_it() {
    // The rule is a per-model quirk in the reference, not general behaviour.
    // Applying it everywhere would silently change every SPM chat model.
    let t = Tokenizer::from_metadata(&meta("Llama-3.2-1B-Instruct")).unwrap();
    let ids = t.encode("<|user|>\nhi");
    assert!(
        ids.contains(&NEWLINE),
        "the newline was stripped on a model llama.cpp does not strip for: {ids:?}"
    );
}

#[test]
fn the_three_exempt_specials_are_exempt() {
    // llama.cpp sets RSTRIP on every Phi-3 special and then turns it back off
    // for these three. `<s>` matters most: it opens the prompt, and stripping
    // after it would eat the whitespace SPM's dummy prefix depends on.
    let t = Tokenizer::from_metadata(&meta("phi3")).unwrap();
    let ids = t.encode("<s>\nhi");
    assert!(
        ids.contains(&NEWLINE),
        "whitespace after <s> was stripped, and it must not be: {ids:?}"
    );
}

#[test]
fn the_name_match_is_case_insensitive() {
    // Containers spell it `Phi-3-mini-4k-instruct`, `phi3`, and `PHI-3`.
    // llama.cpp lowercases the name before comparing; so does this.
    for name in ["PHI-3-Mini", "phi-3", "Phi3-medium"] {
        let t = Tokenizer::from_metadata(&meta(name)).unwrap();
        assert!(
            !t.encode("<|user|>\nhi").contains(&NEWLINE),
            "{name} was not recognised as a Phi-3"
        );
    }
}
