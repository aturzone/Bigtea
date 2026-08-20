//! Read a `.safetensors` file.
//!
//! # Why this exists
//!
//! The FLUX.2 autoencoder — the part that turns a diffusion latent into pixels —
//! ships as `flux2-vae.safetensors`. Every other set of weights this project
//! touches is a GGUF, so nothing here could open it.
//!
//! # The format, and it is a kind one
//!
//! Eight bytes of little-endian length, that many bytes of JSON, then raw tensor
//! data. Each JSON entry gives a dtype, a shape, and a half-open byte range
//! **relative to the start of the data section**, not to the file. One key,
//! `__metadata__`, is a string map rather than a tensor and has to be skipped —
//! treating it as one is the first mistake a reader makes.
//!
//! Measured against the real file (`Comfy-Org/flux2-dev`, ungated, 251 tensors):
//! dtypes `F32` and `I64`, `__metadata__` present, and
//! `bn.num_batches_tracked` has `"shape": []` — a **scalar**, one element, which
//! a naive `shape.iter().product()` gets right only by accident and a
//! `shape[0]` gets wrong immediately.
//!
//! # What is validated, and what deliberately is not
//!
//! Every range is checked for **self-consistency**: its size must match the
//! element count implied by the dtype and shape. That is the check worth having,
//! because a range of the wrong size hands out a correctly-bounded slice of the
//! wrong numbers and nothing downstream can tell.
//!
//! **Whether the data is actually present is not checked at parse time.** It was
//! at first, and inspecting a header-only range fetch then failed with
//! `decoder.mid_block.attentions.0.to_out.0.weight: ends at 2713108 but the data
//! section is 2069088 bytes` — a correct complaint about a file nobody claimed
//! was complete. Reading a 28 KB table out of a 300 MB download before it
//! finishes is a normal thing to want, so presence is [`SafeTensors::bytes_of`]
//! returning `None`, and [`SafeTensors::is_complete`] when the caller wants to
//! ask directly.

use chaos_grammar::Json;

/// Element types safetensors defines, restricted to the ones that can appear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Dtype {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::Bf16,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            _ => return None,
        })
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::Bf16 | Dtype::I16 => 2,
            Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
        }
    }
}

/// One tensor's description and where its bytes are.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Byte range within the data section, half-open.
    pub start: usize,
    pub end: usize,
}

impl Entry {
    /// Elements this tensor holds.
    ///
    /// **An empty shape is one element, not zero.** `bn.num_batches_tracked` in
    /// the real VAE is exactly that, and a reader that returns zero here decides
    /// the tensor is empty and silently skips it.
    pub fn elements(&self) -> u64 {
        self.shape.iter().product::<u64>().max(1)
    }
}

/// A parsed safetensors file: its table, and where the data begins.
#[derive(Clone, Debug)]
pub struct SafeTensors {
    entries: Vec<Entry>,
    /// Offset in the whole file where the data section starts.
    data_at: usize,
    /// Key/value strings from `__metadata__`, in file order.
    metadata: Vec<(String, String)>,
}

/// What can go wrong, said specifically enough to act on.
#[derive(Debug, PartialEq)]
pub enum Error {
    /// Fewer than the eight bytes the length itself needs.
    TooShort,
    /// The declared header length does not fit in the file.
    HeaderOutOfRange { declared: u64, file: usize },
    /// The header is not JSON, or not an object.
    BadJson(String),
    /// A tensor entry is missing a field or has the wrong type.
    BadEntry(String),
    /// A dtype string safetensors does not define.
    UnknownDtype(String),
    /// A byte range outside the data section, or inconsistent with the shape.
    BadRange(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::TooShort => write!(f, "not a safetensors file: shorter than eight bytes"),
            Error::HeaderOutOfRange { declared, file } => write!(
                f,
                "header claims {declared} bytes but the file is {file}; truncated, or not \
                 safetensors at all"
            ),
            Error::BadJson(m) => write!(f, "the header is not a JSON object: {m}"),
            Error::BadEntry(m) => write!(f, "bad tensor entry: {m}"),
            Error::UnknownDtype(d) => write!(f, "unknown dtype {d:?}"),
            Error::BadRange(m) => write!(f, "bad byte range: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl SafeTensors {
    /// Parse the header of `file`, which must be the whole file or at least its
    /// head plus the data section that follows.
    ///
    /// The data is *not* copied: [`bytes_of`] slices the caller's buffer.
    pub fn parse(file: &[u8]) -> Result<Self, Error> {
        if file.len() < 8 {
            return Err(Error::TooShort);
        }
        let declared = u64::from_le_bytes(file[..8].try_into().expect("eight bytes"));
        // Checked before it is used as an index. A 401 error page parsed as this
        // gave a declared length of 8,367,815,047,113,827,137, which is what
        // happens when HTML is read as a u64 -- so the message says "or not
        // safetensors at all".
        let end = 8usize
            .checked_add(
                usize::try_from(declared).map_err(|_| Error::HeaderOutOfRange {
                    declared,
                    file: file.len(),
                })?,
            )
            .ok_or(Error::HeaderOutOfRange {
                declared,
                file: file.len(),
            })?;
        if end > file.len() {
            return Err(Error::HeaderOutOfRange {
                declared,
                file: file.len(),
            });
        }
        let text = std::str::from_utf8(&file[8..end])
            .map_err(|e| Error::BadJson(format!("not UTF-8: {e}")))?;
        let json = Json::parse(text).map_err(|e| Error::BadJson(format!("{e}")))?;
        let Json::Obj(fields) = json else {
            return Err(Error::BadJson("the top level is not an object".into()));
        };

        let mut entries = Vec::with_capacity(fields.len());
        let mut metadata = Vec::new();
        for (name, value) in fields {
            if name == "__metadata__" {
                // Strings only, and anything else is skipped rather than
                // refused: metadata is documentation, and a stray number in it
                // is no reason to fail to load the weights.
                if let Json::Obj(kv) = value {
                    for (k, v) in kv {
                        if let Json::Str(s) = v {
                            metadata.push((k, s));
                        }
                    }
                }
                continue;
            }
            entries.push(entry_from(&name, value)?);
        }
        Ok(SafeTensors {
            entries,
            data_at: end,
            metadata,
        })
    }

    /// Every tensor, in the order the file lists them.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// One tensor by exact name.
    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// `__metadata__`, in file order.
    pub fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    /// Bytes of data the table describes: the furthest `end` any entry names.
    ///
    /// A complete file is `data_offset() + data_len()` long. Useful for checking
    /// a download before opening it, which is what `chaos-model`'s
    /// `expected_file_bytes` does for GGUF.
    pub fn data_len(&self) -> usize {
        self.entries.iter().map(|e| e.end).max().unwrap_or(0)
    }

    /// Does `file` hold every byte the table describes?
    ///
    /// Separate from parsing on purpose: a header-only fetch parses fine and is
    /// a normal thing to have.
    pub fn is_complete(&self, file: &[u8]) -> bool {
        file.len() >= self.data_at + self.data_len()
    }

    /// Where the data section starts in the file.
    pub fn data_offset(&self) -> usize {
        self.data_at
    }

    /// The bytes of one tensor, sliced out of the same buffer that was parsed.
    ///
    /// `None` when the buffer is shorter than the entry needs, which is the case
    /// when only the header was fetched.
    pub fn bytes_of<'a>(&self, file: &'a [u8], entry: &Entry) -> Option<&'a [u8]> {
        let from = self.data_at.checked_add(entry.start)?;
        let to = self.data_at.checked_add(entry.end)?;
        file.get(from..to)
    }
}

/// Build one [`Entry`], validating everything the header claims.
fn entry_from(name: &str, value: Json) -> Result<Entry, Error> {
    let Json::Obj(fields) = value else {
        return Err(Error::BadEntry(format!("{name}: not an object")));
    };
    let find = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, v)| v);

    let Some(Json::Str(dtype)) = find("dtype") else {
        return Err(Error::BadEntry(format!("{name}: no dtype string")));
    };
    let dtype = Dtype::parse(dtype).ok_or_else(|| Error::UnknownDtype(dtype.clone()))?;

    let Some(Json::Arr(dims)) = find("shape") else {
        return Err(Error::BadEntry(format!("{name}: no shape array")));
    };
    let mut shape = Vec::with_capacity(dims.len());
    for d in dims {
        let Json::Num(n) = d else {
            return Err(Error::BadEntry(format!(
                "{name}: a shape entry is not a number"
            )));
        };
        if *n < 0.0 || n.fract() != 0.0 {
            return Err(Error::BadEntry(format!(
                "{name}: shape {n} is not a whole count"
            )));
        }
        shape.push(*n as u64);
    }

    let Some(Json::Arr(range)) = find("data_offsets") else {
        return Err(Error::BadEntry(format!("{name}: no data_offsets array")));
    };
    if range.len() != 2 {
        return Err(Error::BadEntry(format!(
            "{name}: data_offsets has {} values, not two",
            range.len()
        )));
    }
    let mut ends = [0usize; 2];
    for (i, v) in range.iter().enumerate() {
        let Json::Num(n) = v else {
            return Err(Error::BadEntry(format!(
                "{name}: a data_offset is not a number"
            )));
        };
        if *n < 0.0 || n.fract() != 0.0 {
            return Err(Error::BadEntry(format!(
                "{name}: data_offset {n} is not whole"
            )));
        }
        ends[i] = *n as usize;
    }
    let (start, end) = (ends[0], ends[1]);
    if end < start {
        return Err(Error::BadRange(format!(
            "{name}: end {end} before start {start}"
        )));
    }
    // The one check that catches a header disagreeing with itself, which is
    // worth more than all the bounds checks put together: a range that is the
    // wrong *size* for its shape hands out a correctly-bounded slice of the
    // wrong numbers, and nothing downstream can tell.
    let want = entry_bytes(dtype, &shape);
    if end - start != want {
        return Err(Error::BadRange(format!(
            "{name}: {} bytes for shape {shape:?} of {dtype:?}, which needs {want}",
            end - start
        )));
    }
    Ok(Entry {
        name: name.to_string(),
        dtype,
        shape,
        start,
        end,
    })
}

/// Bytes a tensor of this dtype and shape occupies. An empty shape is one
/// element.
fn entry_bytes(dtype: Dtype, shape: &[u64]) -> usize {
    let n: u64 = shape.iter().product::<u64>().max(1);
    n as usize * dtype.size()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal file so the reader can be tested without a 300 MB
    /// download.
    fn build(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn a_real_shaped_header_parses() {
        // The shape of the actual FLUX.2 VAE header, reduced: an I64 scalar with
        // an empty shape, an F32 vector, and metadata.
        let header = concat!(
            r#"{"__metadata__":{"format":"pt"},"#,
            r#""bn.num_batches_tracked":{"dtype":"I64","shape":[],"data_offsets":[0,8]},"#,
            r#""bn.running_mean":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#
        );
        let data = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x80, 0x3F, 0, 0, 0, 0x40];
        let file = build(header, &data);
        let st = SafeTensors::parse(&file).expect("parse");

        assert_eq!(st.entries().len(), 2, "__metadata__ is not a tensor");
        assert_eq!(st.metadata(), &[("format".to_string(), "pt".to_string())]);

        let scalar = st.get("bn.num_batches_tracked").expect("scalar");
        assert_eq!(scalar.dtype, Dtype::I64);
        assert!(scalar.shape.is_empty());
        assert_eq!(scalar.elements(), 1, "an empty shape is ONE element");

        let vec2 = st.get("bn.running_mean").expect("vector");
        assert_eq!(vec2.shape, vec![2]);
        assert_eq!(vec2.elements(), 2);

        // The bytes come back from the caller's buffer, at the right place.
        let b = st.bytes_of(&file, vec2).expect("bytes");
        assert_eq!(b.len(), 8);
        assert_eq!(f32::from_le_bytes(b[..4].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(b[4..].try_into().unwrap()), 2.0);
    }

    /// A header-only fetch parses, and asking for data says no rather than
    /// panicking.
    ///
    /// This is the normal case while a download is in flight, and it is why
    /// `parse` does not require the data: requiring it made inspecting a 28 KB
    /// table inside a 300 MB download impossible.
    #[test]
    fn a_header_without_its_data_is_not_a_panic() {
        let header = r#"{"a":{"dtype":"F32","shape":[1000],"data_offsets":[0,4000]}}"#;
        // Declared 4000 bytes of data, none present -- a header-only fetch.
        let mut head = Vec::new();
        head.extend_from_slice(&(header.len() as u64).to_le_bytes());
        head.extend_from_slice(header.as_bytes());

        // It **parses**, because the table is self-consistent and the caller
        // never claimed the file was complete.
        let st = SafeTensors::parse(&head).expect("a header alone must parse");
        assert_eq!(st.entries().len(), 1);
        assert_eq!(st.data_len(), 4000);
        assert!(!st.is_complete(&head), "the data is not here");
        let e = st.get("a").expect("a");
        assert!(st.bytes_of(&head, e).is_none(), "no data to hand out");

        // With the data present, both answers flip.
        let full = build(header, &vec![0u8; 4000]);
        let st = SafeTensors::parse(&full).expect("parse");
        assert!(st.is_complete(&full));
        let e = st.get("a").expect("a");
        assert_eq!(st.bytes_of(&full, e).map(<[u8]>::len), Some(4000));
        assert!(st.bytes_of(&full[..100], e).is_none(), "short buffer");
    }

    /// The size check catches a header that disagrees with itself.
    #[test]
    fn a_range_the_wrong_size_for_its_shape_is_refused() {
        // 3 F32s need 12 bytes; the header claims 16.
        let header = r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}}"#;
        let file = build(header, &[0u8; 16]);
        let err = SafeTensors::parse(&file).expect_err("must not accept");
        match err {
            Error::BadRange(m) => {
                assert!(m.contains("needs 12"), "{m}");
                assert!(m.contains("16 bytes"), "{m}");
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    /// An HTML error page is refused with a message that names the possibility.
    ///
    /// Not hypothetical: fetching the gated FLUX.2 repo returned a 401 page, and
    /// its first eight bytes read as a declared header length of
    /// 8,367,815,047,113,827,137.
    #[test]
    fn an_error_page_is_not_mistaken_for_a_file() {
        let html = b"{\"error\":\"Repo model black-forest-labs/FLUX.2-dev is gated.\"}";
        let err = SafeTensors::parse(html).expect_err("must not accept");
        match err {
            Error::HeaderOutOfRange { declared, file } => {
                assert!(declared > file as u64);
            }
            other => panic!("wrong error: {other:?}"),
        }
        assert!(format!("{err}").contains("not safetensors at all"));

        // And something genuinely too short. Compared as errors rather than
        // Results, because `SafeTensors` holds no `PartialEq` and does not need
        // one.
        assert_eq!(
            SafeTensors::parse(b"abc").expect_err("three bytes"),
            Error::TooShort
        );
    }

    /// Every dtype safetensors defines has the right width.
    #[test]
    fn dtype_widths_are_right() {
        for (s, bytes) in [
            ("F64", 8),
            ("F32", 4),
            ("F16", 2),
            ("BF16", 2),
            ("I64", 8),
            ("I32", 4),
            ("I16", 2),
            ("I8", 1),
            ("U8", 1),
            ("BOOL", 1),
        ] {
            let d = Dtype::parse(s).unwrap_or_else(|| panic!("{s} should parse"));
            assert_eq!(d.size(), bytes, "{s}");
        }
        assert!(Dtype::parse("F8_E4M3").is_none(), "not a safetensors dtype");
    }
}
