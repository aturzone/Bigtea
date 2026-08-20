//! D2 — GGUF v2 and v3, built in memory rather than downloaded.
//!
//! # The ticket's premise was wrong, and this is the evidence
//!
//! D2 was written as "v2 writes array lengths as `u32` where v3 uses `u64`".
//! **That change was v1 to v2, not v2 to v3.** llama.cpp's own reader
//! (`ggml/src/gguf.cpp`) has no width branch anywhere: it reads every length as
//! 64-bit, refuses v1 outright with *"GGUFv1 is no longer supported"*, and
//! treats v2 and v3 identically. v3's addition was big-endian support.
//!
//! So the test is not "parse a narrower field" — it is **that byte-identical
//! payloads differing only in the version field produce identical parses**,
//! which is the actual compatibility claim, and that v1 and the future are
//! refused clearly.

mod common;

use chaos_gguf::{Error, Gguf, Value};
use common::{valid_container, Builder};

#[test]
fn v2_and_v3_parse_identically() {
    let v2 = Gguf::parse(&valid_container(2)).expect("v2 must parse");
    let v3 = Gguf::parse(&valid_container(3)).expect("v3 must parse");

    assert_eq!(v2.version, 2);
    assert_eq!(v3.version, 3);
    // Everything except the version field must be indistinguishable.
    assert_eq!(
        v2.metadata, v3.metadata,
        "metadata differed between versions"
    );
    assert_eq!(v2.data_offset, v3.data_offset);
    assert_eq!(v2.tensors.len(), v3.tensors.len());
    assert_eq!(v2.tensors[0].name, v3.tensors[0].name);
    assert_eq!(v2.tensors[0].dims, v3.tensors[0].dims);
}

#[test]
fn a_v2_container_yields_the_values_it_declared() {
    let g = Gguf::parse(&valid_container(2)).expect("v2 must parse");
    assert_eq!(g.architecture(), Some("llama"));
    assert_eq!(g.get_u64("llama.block_count"), Some(32));
    let Some(Value::Array(tokens)) = g.get("tokenizer.ggml.tokens") else {
        panic!("tokens array missing");
    };
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[1].as_str(), Some("hello"));
    let Some(Value::Array(scores)) = g.get("tokenizer.ggml.scores") else {
        panic!("scores array missing");
    };
    assert_eq!(scores.len(), 3);
    assert_eq!(scores[2].as_f32(), Some(-2.5));
    assert_eq!(g.tensors[0].elements(), 4096 * 32000);
}

/// llama.cpp refuses v1 rather than reading it. Silently accepting one would
/// mean reading `u32` lengths as `u64` and getting nonsense counts.
#[test]
fn v1_is_refused() {
    let err = Gguf::parse(&valid_container(1)).expect_err("v1 must be refused");
    assert!(
        matches!(err, Error::UnsupportedVersion(1)),
        "expected UnsupportedVersion(1), got {err:?}"
    );
}

#[test]
fn a_future_version_is_refused_rather_than_guessed() {
    let err = Gguf::parse(&valid_container(4)).expect_err("v4 must be refused");
    assert!(matches!(err, Error::UnsupportedVersion(4)), "got {err:?}");
}

/// A v3 header written big-endian reads as `0x03000000` on a little-endian
/// host. Reporting "unsupported version 50331648" is true and useless; it reads
/// like corruption and sends the reader to the wrong place.
#[test]
fn a_byte_swapped_version_is_named_as_an_endianness_problem() {
    let mut b = Builder::default();
    b.u32(common::MAGIC).u32(3u32.swap_bytes()).u64(0).u64(0);
    let err = Gguf::parse(b.bytes()).expect_err("must be refused");
    match err {
        Error::ByteOrderMismatch { found } => assert_eq!(found, 0x0300_0000),
        other => panic!("expected ByteOrderMismatch, got {other:?}"),
    }
    // The message has to say so, since that is the whole point of the variant.
    let text = Gguf::parse(b.bytes()).unwrap_err().to_string();
    assert!(
        text.contains("endian"),
        "the error must name endianness, said: {text}"
    );
}

#[test]
fn a_file_that_is_not_gguf_at_all_is_refused_by_magic() {
    let mut b = Builder::default();
    b.u32(0x1234_5678).u32(3).u64(0).u64(0);
    assert!(matches!(
        Gguf::parse(b.bytes()),
        Err(Error::BadMagic { found: 0x1234_5678 })
    ));
}

/// `general.alignment` moves where tensor data begins, and a container that
/// declares a non-power-of-two must not be believed.
#[test]
fn alignment_is_honoured_when_sane_and_ignored_when_not() {
    for (declared, expect_align) in [(64u64, 64u64), (32, 32), (0, 32), (33, 32)] {
        let mut b = Builder::header(3, 1, 0);
        b.kv_u32("general.alignment", declared as u32);
        let g = Gguf::parse(b.bytes()).expect("parses");
        assert_eq!(
            g.data_offset % expect_align,
            0,
            "declared alignment {declared} produced offset {}",
            g.data_offset
        );
        assert!(g.data_offset >= b.bytes().len() as u64);
    }
}
