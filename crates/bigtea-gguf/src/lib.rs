//! GGUF container parsing: metadata and the tensor index, without touching weights.
//!
//! Bigtea's whole premise is deciding what to do with a model *before* paying
//! to read it, so this parser reads the header and index and stops. On a
//! sharded model that is the first few megabytes of a multi-hundred-gigabyte
//! container.
//!
//! The parts that matter downstream:
//!
//! * **Tensor names and shapes** — which tell us, per layer, what is dense
//!   (read every token) and what is a routed expert (read only when selected).
//!   That split decides whether a model runs at all on a small machine.
//! * **Tensor offsets and quantized sizes** — the real byte cost of a read,
//!   rather than a parameter-count estimate.
//!
//! Format reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;

mod reader;
mod tensor;

pub use reader::{Gguf, TensorInfo};
pub use tensor::GgmlType;

/// Magic bytes at the start of every GGUF file: `"GGUF"`, little-endian.
pub const MAGIC: u32 = 0x4655_4747;

#[derive(Debug)]
pub enum Error {
    /// Not a GGUF file at all — wrong magic.
    BadMagic { found: u32 },
    /// A GGUF version this parser does not implement.
    UnsupportedVersion(u32),
    /// The file ended in the middle of a field.
    Truncated {
        needed: usize,
        available: usize,
        context: &'static str,
    },
    /// A metadata value carried an unknown type tag.
    UnknownValueType(u32),
    /// A tensor declared a `ggml` type this build does not know.
    UnknownTensorType(u32),
    /// A string field was not valid UTF-8.
    BadUtf8,
    /// A declared count is implausible — treated as corruption rather than
    /// trusted, since these drive allocations.
    ImplausibleCount { what: &'static str, value: u64 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadMagic { found } => write!(
                f,
                "not a GGUF file (magic was {found:#010x}, expected {MAGIC:#010x})"
            ),
            Error::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            Error::Truncated {
                needed,
                available,
                context,
            } => write!(
                f,
                "truncated while reading {context}: needed {needed} bytes, {available} left"
            ),
            Error::UnknownValueType(t) => write!(f, "unknown metadata value type {t}"),
            Error::UnknownTensorType(t) => write!(f, "unknown tensor type {t}"),
            Error::BadUtf8 => write!(f, "string field was not valid UTF-8"),
            Error::ImplausibleCount { what, value } => {
                write!(
                    f,
                    "implausible {what} count {value} — file is likely corrupt"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// A metadata value. GGUF's type system, flattened to what callers actually use.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    /// Best-effort integer view. Returns `None` for values that are not
    /// integral, rather than coercing and hiding a type surprise.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// Best-effort float view.
    ///
    /// Integers convert because GGUF writers are inconsistent about it: this
    /// container stores `swiglu_clamp_exp` as whole numbers, and refusing them
    /// would mean refusing a value that is plainly there.
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            Value::U8(v) => Some(v as f32),
            Value::U16(v) => Some(v as f32),
            Value::U32(v) => Some(v as f32),
            Value::U64(v) => Some(v as f32),
            Value::I8(v) => Some(v as f32),
            Value::I16(v) => Some(v as f32),
            Value::I32(v) => Some(v as f32),
            Value::I64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// Metadata key/value store, ordered so listings are stable across runs.
pub type Metadata = BTreeMap<String, Value>;
