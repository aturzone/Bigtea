//! Reading the GGUF header, metadata block and tensor index.
//!
//! Everything here is bounds-checked against the buffer it was given. GGUF
//! files are third-party input measured in hundreds of gigabytes, and every
//! length field in them drives an allocation — so a corrupt or hostile file
//! must produce an `Error`, never a panic and never a huge allocation.

use std::collections::BTreeMap;

use crate::{Error, GgmlType, Metadata, Result, Value, MAGIC};

/// Refuse counts beyond this. Real models are far below it; anything above is
/// corruption, and believing it would mean pre-allocating gigabytes.
const MAX_COUNT: u64 = 1 << 24;
/// Longest string field we will accept.
const MAX_STR: u64 = 1 << 20;

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize, context: &'static str) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated {
            needed: n,
            available: self.buf.len().saturating_sub(self.pos),
            context,
        })?;
        if end > self.buf.len() {
            return Err(Error::Truncated {
                needed: n,
                available: self.buf.len() - self.pos,
                context,
            });
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self, ctx: &'static str) -> Result<u32> {
        let b = self.take(4, ctx)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self, ctx: &'static str) -> Result<u64> {
        let b = self.take(8, ctx)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn string(&mut self, ctx: &'static str) -> Result<String> {
        let len = self.u64(ctx)?;
        if len > MAX_STR {
            return Err(Error::ImplausibleCount {
                what: "string length",
                value: len,
            });
        }
        let bytes = self.take(len as usize, ctx)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::BadUtf8)
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1, "u8")?[0]),
            1 => Value::I8(self.take(1, "i8")?[0] as i8),
            2 => {
                let b = self.take(2, "u16")?;
                Value::U16(u16::from_le_bytes([b[0], b[1]]))
            }
            3 => {
                let b = self.take(2, "i16")?;
                Value::I16(i16::from_le_bytes([b[0], b[1]]))
            }
            4 => Value::U32(self.u32("u32")?),
            5 => Value::I32(self.u32("i32")? as i32),
            6 => Value::F32(f32::from_bits(self.u32("f32")?)),
            7 => Value::Bool(self.take(1, "bool")?[0] != 0),
            8 => Value::String(self.string("string")?),
            9 => {
                let elem_ty = self.u32("array type")?;
                let n = self.u64("array length")?;
                if n > MAX_COUNT {
                    return Err(Error::ImplausibleCount {
                        what: "array length",
                        value: n,
                    });
                }
                // Arrays of arrays are not in the format; rejecting keeps this
                // non-recursive and therefore not stack-overflowable.
                if elem_ty == 9 {
                    return Err(Error::UnknownValueType(elem_ty));
                }
                let mut items = Vec::with_capacity(n.min(4096) as usize);
                for _ in 0..n {
                    items.push(self.value(elem_ty)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64("u64")?),
            11 => Value::I64(self.u64("i64")? as i64),
            12 => {
                let b = self.take(8, "f64")?;
                Value::F64(f64::from_le_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ]))
            }
            other => return Err(Error::UnknownValueType(other)),
        })
    }
}

/// One tensor's entry in the index. Shape and location only — no weights.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset from the start of the tensor-data section, not the file.
    pub offset: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product::<u64>()
    }

    /// Bytes this tensor occupies on disk, or `None` if its type is unknown.
    pub fn size_bytes(&self) -> Option<u64> {
        self.ty.size_of(self.elements())
    }

    /// True for a routed-expert tensor — the ones read only when selected.
    ///
    /// llama.cpp names these `blk.N.ffn_{gate,up,down}_exps.weight`, keeping
    /// every expert of a layer in one stacked tensor. The `_exps` suffix is
    /// what distinguishes them from the always-read shared/dense FFN.
    pub fn is_routed_expert(&self) -> bool {
        self.name.contains("_exps")
    }

    /// Layer index parsed from a `blk.N.` prefix, if present.
    pub fn layer(&self) -> Option<u32> {
        let rest = self.name.strip_prefix("blk.")?;
        let (num, _) = rest.split_once('.')?;
        num.parse().ok()
    }
}

/// A parsed GGUF header: metadata plus the tensor index.
#[derive(Debug, Clone)]
pub struct Gguf {
    pub version: u32,
    pub metadata: Metadata,
    pub tensors: Vec<TensorInfo>,
    /// Byte offset where the tensor-data section begins.
    pub data_offset: u64,
}

impl Gguf {
    /// Parse from a buffer holding at least the header, metadata and index.
    ///
    /// For a sharded model that is the first shard, which is why this takes a
    /// slice rather than a path: the caller may have only a few megabytes of a
    /// container that is hundreds of gigabytes.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        let mut cur = Cursor::new(buf);

        let magic = cur.u32("magic")?;
        if magic != MAGIC {
            return Err(Error::BadMagic { found: magic });
        }
        let version = cur.u32("version")?;
        // A version whose low half is zero is a small number byte-swapped:
        // v3 written big-endian reads as 0x03000000 here. Saying so beats
        // reporting "unsupported version 50331648", which reads like corruption
        // and sends the reader looking at the wrong thing. llama.cpp makes the
        // same check for the same reason.
        if version != 0 && (version & 0x0000_FFFF) == 0 {
            return Err(Error::ByteOrderMismatch { found: version });
        }
        // **v2 and v3 have the same layout.** The u32-to-u64 change to string
        // and array lengths was v1 to v2, not v2 to v3 -- v3 added big-endian
        // support and left the field widths alone. llama.cpp reads both with one
        // code path and refuses v1 outright; so does this.
        if !(2..=3).contains(&version) {
            return Err(Error::UnsupportedVersion(version));
        }

        let tensor_count = cur.u64("tensor count")?;
        if tensor_count > MAX_COUNT {
            return Err(Error::ImplausibleCount {
                what: "tensor",
                value: tensor_count,
            });
        }
        let kv_count = cur.u64("metadata count")?;
        if kv_count > MAX_COUNT {
            return Err(Error::ImplausibleCount {
                what: "metadata",
                value: kv_count,
            });
        }

        let mut metadata: Metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = cur.string("metadata key")?;
            if key.is_empty() {
                return Err(Error::EmptyKey);
            }
            let ty = cur.u32("metadata value type")?;
            let value = cur.value(ty)?;
            // Refuse rather than overwrite. `BTreeMap::insert` silently kept the
            // last value, so a container with two `general.architecture` keys
            // loaded as whichever came second while another reader might take
            // the first -- the definition of a silent wrong value. llama.cpp
            // refuses these too.
            if metadata.contains_key(&key) {
                return Err(Error::DuplicateKey(key));
            }
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(65536) as usize);
        for _ in 0..tensor_count {
            let name = cur.string("tensor name")?;
            let n_dims = cur.u32("tensor rank")?;
            if n_dims > 8 {
                return Err(Error::ImplausibleCount {
                    what: "tensor rank",
                    value: n_dims as u64,
                });
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cur.u64("tensor dim")?);
            }
            let ty = GgmlType(cur.u32("tensor type")?);
            let offset = cur.u64("tensor offset")?;
            // Names are how every caller finds a tensor, so two with the same
            // name means one is unreachable and which one is arbitrary.
            if tensors.iter().any(|t: &TensorInfo| t.name == name) {
                return Err(Error::DuplicateTensor(name));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
            });
        }

        // Tensor data starts at the next `general.alignment` boundary.
        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .filter(|a| a.is_power_of_two())
            .unwrap_or(32);
        let pos = cur.pos as u64;
        let data_offset = pos.div_ceil(alignment) * alignment;

        Ok(Gguf {
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    /// Model architecture, e.g. `"deepseek2"` or `"kimi-k3"`.
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    /// Total bytes of all indexed tensors whose type we understand.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().filter_map(TensorInfo::size_bytes).sum()
    }

    /// Split total bytes into `(routed_expert, everything_else)`.
    ///
    /// This is *the* number for streaming inference: the second element is read
    /// every token and wants to be RAM-resident, while the first is read only
    /// as routing selects it.
    pub fn expert_vs_dense_bytes(&self) -> (u64, u64) {
        let mut expert = 0;
        let mut dense = 0;
        for t in &self.tensors {
            let Some(size) = t.size_bytes() else { continue };
            if t.is_routed_expert() {
                expert += size;
            } else {
                dense += size;
            }
        }
        (expert, dense)
    }
}
