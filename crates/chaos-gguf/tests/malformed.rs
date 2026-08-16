//! D1 — a malformed container must produce an `Err`, never a panic and never a
//! silently wrong value.
//!
//! GGUF files are third-party input measured in hundreds of gigabytes, and every
//! length field in one drives an allocation. There is no fuzzing crate here —
//! the workspace has no external dependencies and that is deliberate — so this
//! is a hand-written corpus plus two exhaustive sweeps that need no crate:
//!
//! - **every prefix** of a valid container, which is truncation at every field
//!   boundary and every offset in between;
//! - **every single-byte corruption** at a stride across the header, which is
//!   where a length field turns into an enormous number.
//!
//! Both assert only one thing: the call returns. A panic in a library that
//! parses hostile input is a denial of service in the server that embeds it, and
//! `parse` takes `&[u8]` precisely so a caller can hand it a few megabytes of a
//! 144 GB file it has not validated.
//!
//! The third property is the quiet one: **no wrong answers**. A duplicate key
//! used to overwrite silently, so a container with two `general.architecture`
//! entries loaded as whichever came second — no error, no crash, just a
//! different model than the file describes.

mod common;

use chaos_gguf::{Error, Gguf};
use common::{valid_container, Builder, T_ARRAY, T_F32, T_STRING};

/// A length that claims more bytes than the file holds.
#[test]
fn a_string_longer_than_the_file_is_truncation_not_a_panic() {
    let mut b = Builder::header(3, 1, 0);
    b.string_with_declared_len(9_999, "short");
    match Gguf::parse(b.bytes()) {
        Err(Error::Truncated { .. }) => {}
        other => panic!("expected Truncated, got {other:?}"),
    }
}

/// A string length so large it would be an allocation attack.
#[test]
fn an_enormous_string_length_is_refused_before_allocating() {
    let mut b = Builder::header(3, 1, 0);
    b.string_with_declared_len(u64::MAX, "x");
    match Gguf::parse(b.bytes()) {
        Err(Error::ImplausibleCount { what, .. }) => assert_eq!(what, "string length"),
        other => panic!("expected ImplausibleCount, got {other:?}"),
    }
}

/// An array that declares more elements than follow it.
#[test]
fn an_array_length_past_the_end_is_truncation() {
    let mut b = Builder::header(3, 1, 0);
    b.kv_array_with_declared_len("scores", T_F32, 1_000, &[1.0, 2.0, 3.0]);
    match Gguf::parse(b.bytes()) {
        Err(Error::Truncated { .. }) => {}
        other => panic!("expected Truncated, got {other:?}"),
    }
}

#[test]
fn an_enormous_array_length_is_refused_before_allocating() {
    let mut b = Builder::header(3, 1, 0);
    b.kv_array_with_declared_len("scores", T_F32, u64::MAX, &[1.0]);
    match Gguf::parse(b.bytes()) {
        Err(Error::ImplausibleCount { what, .. }) => assert_eq!(what, "array length"),
        other => panic!("expected ImplausibleCount, got {other:?}"),
    }
}

#[test]
fn an_unknown_value_type_is_refused() {
    let mut b = Builder::header(3, 1, 0);
    b.string("weird").u32(99);
    match Gguf::parse(b.bytes()) {
        Err(Error::UnknownValueType(99)) => {}
        other => panic!("expected UnknownValueType(99), got {other:?}"),
    }
}

/// Arrays of arrays are not in the format. Refusing keeps the value parser
/// non-recursive, so a nested container cannot overflow the stack.
#[test]
fn an_array_of_arrays_is_refused_rather_than_recursed() {
    let mut b = Builder::header(3, 1, 0);
    b.string("nested").u32(T_ARRAY).u32(T_ARRAY).u64(8);
    match Gguf::parse(b.bytes()) {
        Err(Error::UnknownValueType(t)) => assert_eq!(t, T_ARRAY),
        other => panic!("expected the nested array to be refused, got {other:?}"),
    }
}

/// **The silent-wrong-value case.** Two values, one key: the container does not
/// say which is meant, and taking the last one quietly is the worst option.
#[test]
fn a_duplicate_key_is_refused_not_overwritten() {
    let mut b = Builder::header(3, 2, 0);
    b.kv_string("general.architecture", "llama")
        .kv_string("general.architecture", "bert");
    match Gguf::parse(b.bytes()) {
        Err(Error::DuplicateKey(k)) => assert_eq!(k, "general.architecture"),
        other => panic!("expected DuplicateKey, got {other:?}"),
    }
}

#[test]
fn an_empty_key_is_refused() {
    let mut b = Builder::header(3, 1, 0);
    b.kv_string("", "value");
    assert!(matches!(Gguf::parse(b.bytes()), Err(Error::EmptyKey)));
}

#[test]
fn two_tensors_with_one_name_are_refused() {
    let mut b = Builder::header(3, 0, 2);
    b.tensor("token_embd.weight", &[4, 4], 0, 0)
        .tensor("token_embd.weight", &[8, 8], 0, 64);
    match Gguf::parse(b.bytes()) {
        Err(Error::DuplicateTensor(n)) => assert_eq!(n, "token_embd.weight"),
        other => panic!("expected DuplicateTensor, got {other:?}"),
    }
}

#[test]
fn an_absurd_tensor_rank_is_refused() {
    let mut b = Builder::header(3, 0, 1);
    b.tensor_with_declared_rank("t", 99, &[4], 0, 0);
    match Gguf::parse(b.bytes()) {
        Err(Error::ImplausibleCount { what, value }) => {
            assert_eq!(what, "tensor rank");
            assert_eq!(value, 99);
        }
        other => panic!("expected ImplausibleCount, got {other:?}"),
    }
}

#[test]
fn absurd_tensor_and_metadata_counts_are_refused_before_allocating() {
    let mut b = Builder::default();
    b.u32(common::MAGIC).u32(3).u64(u64::MAX).u64(0);
    assert!(matches!(
        Gguf::parse(b.bytes()),
        Err(Error::ImplausibleCount { what: "tensor", .. })
    ));

    let mut b = Builder::default();
    b.u32(common::MAGIC).u32(3).u64(0).u64(u64::MAX);
    assert!(matches!(
        Gguf::parse(b.bytes()),
        Err(Error::ImplausibleCount {
            what: "metadata",
            ..
        })
    ));
}

#[test]
fn a_string_that_is_not_utf8_is_refused() {
    let mut b = Builder::header(3, 1, 0);
    // Declare four bytes, then write an invalid sequence.
    b.u64(4);
    b.buf.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC]);
    assert!(matches!(Gguf::parse(b.bytes()), Err(Error::BadUtf8)));
}

#[test]
fn an_empty_buffer_is_truncation() {
    assert!(matches!(Gguf::parse(&[]), Err(Error::Truncated { .. })));
}

/// Truncation at **every** offset. This is the sweep that replaces a fuzzer for
/// the one failure mode fuzzing would most likely find here: a length read from
/// bytes that exist, pointing at bytes that do not.
#[test]
fn no_prefix_of_a_valid_container_panics() {
    let full = valid_container(3);
    for cut in 0..full.len() {
        let result = Gguf::parse(&full[..cut]);
        assert!(
            result.is_err(),
            "a {cut}-byte prefix of a {}-byte container parsed as valid",
            full.len()
        );
    }
    // ...and the whole thing must still parse, or the sweep proves nothing.
    assert!(Gguf::parse(&full).is_ok(), "the full container must parse");
}

/// Single-byte corruption across the header. Most land in a length or a type
/// tag, which is exactly where a parser turns a small mistake into a huge
/// allocation or an out-of-bounds slice.
///
/// Only that it **returns** is asserted: many corruptions produce a container
/// that is still structurally valid, and demanding an error would be demanding
/// the parser detect corruption it cannot see.
#[test]
fn no_single_byte_corruption_panics() {
    let full = valid_container(3);
    let mut checked = 0;
    for at in 0..full.len() {
        for patch in [0x00u8, 0x01, 0x7F, 0x80, 0xFF] {
            let mut broken = full.clone();
            if broken[at] == patch {
                continue;
            }
            broken[at] = patch;
            // The assertion is that this line is reached at all.
            let _ = Gguf::parse(&broken);
            checked += 1;
        }
    }
    assert!(checked > 1_000, "expected a real sweep, did {checked}");
}

/// Corruption of the *lengths* specifically, which the byte sweep only reaches
/// by luck. Every declared length in a valid container is replaced with a value
/// that overruns, and each must be an error rather than a read past the end.
#[test]
fn overrunning_lengths_are_errors_at_every_field() {
    // string value, string array, f32 array, tensor name, tensor dims.
    let cases: Vec<Vec<u8>> = vec![
        {
            let mut b = Builder::header(3, 1, 0);
            b.string("k")
                .u32(T_STRING)
                .string_with_declared_len(500, "v");
            b.into_bytes()
        },
        {
            let mut b = Builder::header(3, 1, 0);
            b.string("k").u32(T_ARRAY).u32(T_STRING).u64(500);
            b.string("only-one");
            b.into_bytes()
        },
        {
            let mut b = Builder::header(3, 0, 1);
            b.string_with_declared_len(500, "t")
                .u32(1)
                .u64(4)
                .u32(0)
                .u64(0);
            b.into_bytes()
        },
        {
            let mut b = Builder::header(3, 0, 1);
            b.tensor_with_declared_rank("t", 4, &[4], 0, 0);
            b.into_bytes()
        },
    ];
    for (i, bytes) in cases.iter().enumerate() {
        assert!(
            Gguf::parse(bytes).is_err(),
            "case {i} declared more than it held and parsed as valid"
        );
    }
}
