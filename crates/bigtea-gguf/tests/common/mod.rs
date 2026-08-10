//! A GGUF header builder, so container tests need no downloads.
//!
//! Every field is written by hand and every length is settable independently of
//! the bytes that follow it. That is the point: the interesting cases are the
//! ones where a declared length and the actual data *disagree*, which no real
//! container will ever give us and which a hostile or truncated one will.

#![allow(dead_code)]

pub const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

/// Metadata value type tags, as GGUF numbers them.
pub const T_U32: u32 = 4;
pub const T_F32: u32 = 6;
pub const T_BOOL: u32 = 7;
pub const T_STRING: u32 = 8;
pub const T_ARRAY: u32 = 9;
pub const T_U64: u32 = 10;

#[derive(Default)]
pub struct Builder {
    pub buf: Vec<u8>,
}

impl Builder {
    /// A well-formed header at `version`, with `n_kv` metadata pairs and
    /// `n_tensors` tensor entries still to be appended.
    pub fn header(version: u32, n_kv: u64, n_tensors: u64) -> Self {
        let mut b = Builder::default();
        b.u32(MAGIC).u32(version).u64(n_tensors).u64(n_kv);
        b
    }

    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// A length-prefixed string, as GGUF v2/v3 write one: u64 length, then bytes.
    pub fn string(&mut self, s: &str) -> &mut Self {
        self.u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
        self
    }

    /// A string whose *declared* length is not the number of bytes that follow.
    pub fn string_with_declared_len(&mut self, declared: u64, actual: &str) -> &mut Self {
        self.u64(declared);
        self.buf.extend_from_slice(actual.as_bytes());
        self
    }

    /// One metadata pair whose value is a string.
    pub fn kv_string(&mut self, key: &str, value: &str) -> &mut Self {
        self.string(key).u32(T_STRING).string(value)
    }

    pub fn kv_u32(&mut self, key: &str, value: u32) -> &mut Self {
        self.string(key).u32(T_U32).u32(value)
    }

    pub fn kv_u64(&mut self, key: &str, value: u64) -> &mut Self {
        self.string(key).u32(T_U64).u64(value)
    }

    pub fn kv_bool(&mut self, key: &str, value: bool) -> &mut Self {
        self.string(key).u32(T_BOOL).u8(u8::from(value))
    }

    /// A metadata pair holding an array of f32.
    pub fn kv_f32_array(&mut self, key: &str, values: &[f32]) -> &mut Self {
        self.string(key)
            .u32(T_ARRAY)
            .u32(T_F32)
            .u64(values.len() as u64);
        for v in values {
            self.f32(*v);
        }
        self
    }

    /// A metadata pair holding an array of strings.
    pub fn kv_string_array(&mut self, key: &str, values: &[&str]) -> &mut Self {
        self.string(key)
            .u32(T_ARRAY)
            .u32(T_STRING)
            .u64(values.len() as u64);
        for v in values {
            self.string(v);
        }
        self
    }

    /// An array whose declared element count is not what follows it.
    pub fn kv_array_with_declared_len(
        &mut self,
        key: &str,
        elem_ty: u32,
        declared: u64,
        actual: &[f32],
    ) -> &mut Self {
        self.string(key).u32(T_ARRAY).u32(elem_ty).u64(declared);
        for v in actual {
            self.f32(*v);
        }
        self
    }

    /// One tensor index entry: name, rank, dims, type, offset.
    pub fn tensor(&mut self, name: &str, dims: &[u64], ty: u32, offset: u64) -> &mut Self {
        self.string(name).u32(dims.len() as u32);
        for d in dims {
            self.u64(*d);
        }
        self.u32(ty).u64(offset)
    }

    /// A tensor entry whose declared rank is not the number of dims that follow.
    pub fn tensor_with_declared_rank(
        &mut self,
        name: &str,
        declared_rank: u32,
        dims: &[u64],
        ty: u32,
        offset: u64,
    ) -> &mut Self {
        self.string(name).u32(declared_rank);
        for d in dims {
            self.u64(*d);
        }
        self.u32(ty).u64(offset)
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// A complete, valid container header at the given version: two metadata pairs
/// and one tensor. Used to show v2 and v3 parse identically.
pub fn valid_container(version: u32) -> Vec<u8> {
    let mut b = Builder::header(version, 4, 1);
    b.kv_string("general.architecture", "llama")
        .kv_u32("llama.block_count", 32)
        .kv_string_array("tokenizer.ggml.tokens", &["<s>", "hello", "world"])
        .kv_f32_array("tokenizer.ggml.scores", &[0.0, -1.5, -2.5])
        .tensor("token_embd.weight", &[4096, 32000], 0, 0);
    b.into_bytes()
}
