//! A vocabulary that stores a control character **raw** must still tokenize it.
//!
//! # The bug
//!
//! Falcon3 holds a literal newline at id 12. A byte-level BPE vocabulary would
//! hold `Ċ` instead, so encoding the text first and looking up `Ċ` found
//! nothing — and the per-character fallback looked up the same missing key.
//! **Every newline was silently dropped:**
//!
//! ```text
//! a\nb    ours [11, 2088, 2089]    llama.cpp [11, 2088, 12, 2089]
//! ```
//!
//! It passed 8/8 parity because none of those eight prompts contains a newline,
//! and it broke every chat template on the model, all of which do.
//!
//! # Why the existing machinery missed it
//!
//! These tokens are USER_DEFINED and would have been partitioned out before BPE
//! ran — except `specials` excludes anything under three bytes, deliberately,
//! because matching a one-character marker would slice ordinary text apart. The
//! guard that stops short markers from cutting text is the same guard that loses
//! this one, so the fix belongs in the encoder rather than in that guard.

use std::collections::BTreeMap;

use bigtea_gguf::Value;
use bigtea_tokenizer::Tokenizer;

/// A byte-level BPE vocabulary that stores its newline **raw** rather than as
/// `Ċ`, the way Falcon3 does.
fn meta() -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert(
        "tokenizer.ggml.model".to_string(),
        Value::String("gpt2".into()),
    );
    // id 0 `a`, id 1 `b`, id 2 a RAW newline, id 3 the byte-encoded space form,
    // id 4 a RAW tab — the two whitespace characters Falcon3 stores unencoded.
    let tokens = ["a", "b", "\n", "Ġc", "\t"];
    m.insert(
        "tokenizer.ggml.tokens".to_string(),
        Value::Array(tokens.iter().map(|t| Value::String((*t).into())).collect()),
    );
    m.insert(
        "tokenizer.ggml.token_type".to_string(),
        // 4 = USER_DEFINED for the raw whitespace, 1 = NORMAL for the rest.
        Value::Array(vec![
            Value::I32(1),
            Value::I32(1),
            Value::I32(4),
            Value::I32(1),
            Value::I32(4),
        ]),
    );
    m.insert(
        "tokenizer.ggml.add_bos_token".to_string(),
        Value::Bool(false),
    );
    // Named explicitly: the pre-tokenizer decides where the pieces fall, so a
    // fixture that leaves it to a default is testing whichever default is
    // current rather than the thing it means to.
    m.insert(
        "tokenizer.ggml.pre".to_string(),
        Value::String("gpt2".into()),
    );
    // Without a merge, `Ġ` and `c` never join and `Ġc` is unreachable however
    // the lookup is written — the fixture would "pass" for the wrong reason.
    m.insert(
        "tokenizer.ggml.merges".to_string(),
        Value::Array(vec![Value::String("Ġ c".into())]),
    );
    m
}

#[test]
fn a_raw_newline_in_the_vocabulary_is_found_rather_than_dropped() {
    let t = Tokenizer::from_metadata(&meta()).unwrap();
    let ids = t.encode("a\nb");
    assert!(
        ids.contains(&2),
        "the raw newline token was dropped: {ids:?}"
    );
    assert_eq!(ids, vec![0, 2, 1], "{ids:?}");
}

#[test]
fn the_byte_encoded_form_still_wins_when_the_vocabulary_has_it() {
    // The raw lookup is a FALLBACK, consulted only when the byte-encoded form is
    // absent. A vocabulary holding `Ġc` must keep using it, or every model that
    // already tokenizes correctly would change.
    let t = Tokenizer::from_metadata(&meta()).unwrap();
    let ids = t.encode(" c");
    assert_eq!(ids, vec![3], "byte-encoded form was bypassed: {ids:?}");
}

#[test]
fn a_run_of_them_is_not_lost_the_way_a_single_one_was_fixed() {
    // The first fix resolved the whole PIECE against the vocabulary, which
    // handled `a\nb` and nothing else: the piece a pre-tokenizer hands over is
    // rarely the bare character. `a\n\nb`, `a\tb`, a CRLF and every indented
    // code block still lost their whitespace, on a model that reported the
    // single-newline case as fixed.
    let t = Tokenizer::from_metadata(&meta()).unwrap();
    assert_eq!(t.encode("a\n\nb"), vec![0, 2, 2, 1], "a run was collapsed");
    assert_eq!(t.encode("a\tb"), vec![0, 4, 1], "the tab was dropped");
    assert_eq!(
        t.encode("a\n\tb"),
        vec![0, 2, 4, 1],
        "mixed whitespace was dropped"
    );
}

#[test]
fn text_with_no_control_characters_is_untouched() {
    let t = Tokenizer::from_metadata(&meta()).unwrap();
    assert_eq!(t.encode("ab"), vec![0, 1]);
}

/// A vocabulary that gives runs of spaces their own ids, the way OLMo does.
fn runs_meta() -> BTreeMap<String, Value> {
    let mut m = meta();
    let tokens = ["a", "b", "\n", "Ġc", "\t", "  ", "   "];
    m.insert(
        "tokenizer.ggml.tokens".to_string(),
        Value::Array(tokens.iter().map(|t| Value::String((*t).into())).collect()),
    );
    m.insert(
        "tokenizer.ggml.token_type".to_string(),
        Value::Array(vec![
            Value::I32(1),
            Value::I32(1),
            Value::I32(4),
            Value::I32(1),
            Value::I32(4),
            // The run tokens are USER_DEFINED, as they are in OLMo.
            Value::I32(4),
            Value::I32(4),
        ]),
    );
    m
}

#[test]
fn a_short_whitespace_run_with_its_own_id_is_matched_not_split() {
    // `specials` excluded anything under three bytes, so a TWO-space token was
    // skipped while three, four and five were kept. On OLMo that showed as
    // exactly one broken length:
    //
    //   a  b   ours [66, 245, 67]     llama [66, 50276, 67]
    //   a   b  ours [66, 50275, 67]   llama [66, 50275, 67]   (already right)
    //
    // Whitespace cannot slice ordinary text apart -- it is already a boundary --
    // which is why the length guard is relaxed for it and nothing else.
    let t = Tokenizer::from_metadata(&runs_meta()).unwrap();
    assert_eq!(t.encode("a  b"), vec![0, 5, 1], "two-space run was split");
    assert_eq!(
        t.encode("a   b"),
        vec![0, 6, 1],
        "three-space run was split"
    );
}

#[test]
fn a_single_space_is_not_turned_into_a_special() {
    // The relaxation must not promote every space: a lone one has no id of its
    // own here and must still go through the ordinary path.
    let t = Tokenizer::from_metadata(&runs_meta()).unwrap();
    let ids = t.encode("a b");
    assert!(!ids.contains(&5) && !ids.contains(&6), "{ids:?}");
}
