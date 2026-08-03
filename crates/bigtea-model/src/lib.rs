//! A model on disk: several GGUF shards, resolved into one addressable set of
//! tensors.
//!
//! # What this solves
//!
//! A large model ships as `name-00001-of-00005.gguf` .. `-00005-of-00005.gguf`.
//! Each shard is a self-contained GGUF with its own header and its own tensor
//! index covering only the tensors it holds, and every offset in that index is
//! relative to *that shard's* data section. Nothing in the format gives you a
//! single view of the model.
//!
//! [`Model`] builds that view: one name-to-location map across every shard, so
//! callers ask for `blk.7.ffn_gate_exps.weight` and get bytes, without knowing
//! which file it lives in or where the data section starts.
//!
//! # Working with an incomplete download
//!
//! GGUF puts the header and tensor index at the *start* of each shard, so a
//! partially-downloaded shard still has a complete, parseable index. That is
//! genuinely useful: the full layout of a 144 GiB model can be known — and
//! planned against — while it is still downloading. [`Model::open`] therefore
//! accepts shards it cannot fully read, and [`Model::availability`] reports
//! which tensors are actually resident on disk right now.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use bigtea_gguf::{GgmlType, Gguf};
use bigtea_io::{DirectFile, IoMode};

mod discover;
mod resident;

pub use discover::discover_shards;
pub use resident::{LoadReport, ResidentSet, SkipReason, Skipped};

#[derive(Debug)]
pub enum Error {
    Io { path: PathBuf, source: std::io::Error },
    Parse { path: PathBuf, source: bigtea_gguf::Error },
    NoShards,
    /// Two shards claimed the same tensor name.
    DuplicateTensor(String),
    UnknownTensor(String),
    /// A tensor's bytes are not (yet) on disk.
    NotDownloaded { name: String, need: u64, have: u64 },
    /// The tensor's ggml type is one we cannot size.
    Unsizable(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Error::Parse { path, source } => write!(f, "{}: {source}", path.display()),
            Error::NoShards => f.write_str("no shards given"),
            Error::DuplicateTensor(n) => write!(f, "tensor {n} appears in more than one shard"),
            Error::UnknownTensor(n) => write!(f, "no tensor named {n}"),
            Error::NotDownloaded { name, need, have } => write!(
                f,
                "{name} needs {need} bytes but its shard only has {have} on disk \
                 (download incomplete)"
            ),
            Error::Unsizable(n) => write!(f, "cannot size tensor {n}: unknown type"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Where a tensor's bytes live.
#[derive(Debug, Clone)]
pub struct Location {
    pub shard: usize,
    /// Absolute offset in the shard file — the shard's data section start plus
    /// the tensor's own relative offset.
    pub file_offset: u64,
    pub size: u64,
    pub ty: GgmlType,
    pub dims: Vec<u64>,
    /// Read only when routing selects it.
    pub routed_expert: bool,
}

impl Location {
    /// Number of values in this tensor — the product of its dimensions.
    ///
    /// Distinct from [`Self::size`], which is bytes: a quantized tensor packs
    /// many values into each stored byte, so the two differ by the type's
    /// compression ratio.
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

struct Shard {
    file: DirectFile,
    on_disk: u64,
}

/// A model spread across one or more GGUF shards.
pub struct Model {
    shards: Vec<Shard>,
    tensors: BTreeMap<String, Location>,
    architecture: String,
    metadata: bigtea_gguf::Metadata,
    /// Tensors the model declares in total, if the shards say so.
    declared_tensor_count: Option<u64>,
}

impl Model {
    /// Open a set of shards, in any order.
    ///
    /// Shards whose data is still downloading are accepted: their index is
    /// complete even when their weights are not.
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        if paths.is_empty() {
            return Err(Error::NoShards);
        }
        let mut shards = Vec::with_capacity(paths.len());
        let mut tensors: BTreeMap<String, Location> = BTreeMap::new();
        let mut architecture = String::new();
        let mut metadata = bigtea_gguf::Metadata::new();
        let mut declared_tensor_count = None;

        for (idx, path) in paths.iter().enumerate() {
            let file = DirectFile::open(path).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            let on_disk = file.len();

            // The header and index live at the start, so a modest prefix is
            // enough even for a huge shard.
            let head_len = on_disk.min(128 << 20) as usize;
            let head = file.read_at(0, head_len).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            let gguf = Gguf::parse(&head).map_err(|source| Error::Parse {
                path: path.clone(),
                source,
            })?;

            if architecture.is_empty() {
                if let Some(a) = gguf.architecture() {
                    architecture = a.to_string();
                }
            }
            if declared_tensor_count.is_none() {
                declared_tensor_count = gguf.get_u64("split.tensors.count");
            }
            // Keep the richest metadata seen; shards repeat the common keys.
            for (k, v) in &gguf.metadata {
                metadata.entry(k.clone()).or_insert_with(|| v.clone());
            }

            for t in &gguf.tensors {
                let size = t.size_bytes().ok_or_else(|| Error::Unsizable(t.name.clone()))?;
                let loc = Location {
                    shard: idx,
                    file_offset: gguf.data_offset + t.offset,
                    size,
                    ty: t.ty,
                    dims: t.dims.clone(),
                    routed_expert: t.is_routed_expert(),
                };
                if tensors.insert(t.name.clone(), loc).is_some() {
                    return Err(Error::DuplicateTensor(t.name.clone()));
                }
            }

            shards.push(Shard { file, on_disk });
        }

        Ok(Model {
            shards,
            tensors,
            architecture,
            metadata,
            declared_tensor_count,
        })
    }

    /// Open a split model from any one of its shards, discovering the rest.
    pub fn open_split(any_shard: impl AsRef<Path>) -> Result<Self> {
        let paths = discover::discover_shards(any_shard.as_ref());
        Self::open(&paths)
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn metadata(&self) -> &bigtea_gguf::Metadata {
        &self.metadata
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(bigtea_gguf::Value::as_u64)
    }

    /// Architecture-scoped metadata, e.g. `arch_u64("expert_count")`.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        self.get_u64(&format!("{}.{}", self.architecture, suffix))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// How many tensors the model declares overall, when shards say so.
    /// Comparing against [`Self::tensor_count`] reveals missing shards.
    pub fn declared_tensor_count(&self) -> Option<u64> {
        self.declared_tensor_count
    }

    pub fn location(&self, name: &str) -> Option<&Location> {
        self.tensors.get(name)
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Whether cache-bypassing reads engaged for every shard.
    pub fn io_mode(&self) -> IoMode {
        if self.shards.iter().all(|s| s.file.mode() == IoMode::Direct) {
            IoMode::Direct
        } else {
            IoMode::Buffered
        }
    }

    /// Total bytes of all tensors, split `(routed_expert, always_read)`.
    pub fn expert_vs_dense_bytes(&self) -> (u64, u64) {
        let mut expert = 0;
        let mut dense = 0;
        for loc in self.tensors.values() {
            if loc.routed_expert {
                expert += loc.size;
            } else {
                dense += loc.size;
            }
        }
        (expert, dense)
    }

    /// Is this tensor's data actually on disk yet?
    pub fn is_available(&self, name: &str) -> Result<bool> {
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::UnknownTensor(name.to_string()))?;
        Ok(loc.file_offset + loc.size <= self.shards[loc.shard].on_disk)
    }

    /// `(available, total)` tensor counts — useful while a download runs.
    pub fn availability(&self) -> (usize, usize) {
        let available = self
            .tensors
            .values()
            .filter(|loc| loc.file_offset + loc.size <= self.shards[loc.shard].on_disk)
            .count();
        (available, self.tensors.len())
    }

    /// Read a tensor's raw bytes, exactly as stored (still quantized).
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::UnknownTensor(name.to_string()))?;
        let shard = &self.shards[loc.shard];

        if loc.file_offset + loc.size > shard.on_disk {
            return Err(Error::NotDownloaded {
                name: name.to_string(),
                need: loc.file_offset + loc.size,
                have: shard.on_disk,
            });
        }
        shard
            .file
            .read_at(loc.file_offset, loc.size as usize)
            .map_err(|source| Error::Io {
                path: shard.file.path().to_path_buf(),
                source,
            })
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (expert, dense) = self.expert_vs_dense_bytes();
        let (avail, total) = self.availability();
        f.debug_struct("Model")
            .field("architecture", &self.architecture)
            .field("shards", &self.shards.len())
            .field("tensors", &total)
            .field("available", &avail)
            .field("expert_bytes", &expert)
            .field("dense_bytes", &dense)
            .field("io", &self.io_mode())
            .finish()
    }
}
