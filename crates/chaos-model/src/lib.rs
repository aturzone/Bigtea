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
use std::sync::Arc;

use chaos_gguf::{GgmlType, Gguf};
use chaos_io::{DirectFile, IoMode, SkewedBuf};

pub mod adapter;
pub mod catalogue;
mod discover;
pub mod download;
pub mod find;
mod resident;
pub mod validate;

pub use discover::discover_shards;
pub use resident::{measure_spill_rate, LoadReport, ResidentSet, SkipReason, Skipped};

#[derive(Debug)]
pub enum Error {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: PathBuf,
        source: chaos_gguf::Error,
    },
    NoShards,
    /// Two shards claimed the same tensor name.
    DuplicateTensor(String),
    UnknownTensor(String),
    /// A tensor's bytes are not (yet) on disk.
    NotDownloaded {
        name: String,
        need: u64,
        have: u64,
    },
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
    /// Extra handles onto the same file, one per concurrent reader.
    ///
    /// # Why a pool and not one handle
    ///
    /// Positioned reads carry their own offset, so several threads reading
    /// through one handle need no locking *in this code* — which is what the
    /// streaming path assumed. But a Windows handle opened without
    /// `FILE_FLAG_OVERLAPPED` is **synchronous**, and the I/O manager serialises
    /// operations on it: concurrent `ReadFile` calls queue behind one another
    /// however many threads issue them. The drive is then held at queue depth 1,
    /// where an NVMe delivers a fraction of its rated throughput.
    ///
    /// That is exactly what the old "no further gain past four readers" plateau
    /// was. Measured on this machine, 256 scattered 4 MiB reads:
    ///
    /// ```text
    /// threads      one shared handle      one handle each
    ///       1           1.54 GiB/s             1.61 GiB/s
    ///       4           2.01                   2.65
    ///       8           2.05                   2.69      <- +31%
    /// ```
    ///
    /// 2.69 GiB/s is also **above** the 2.37 GiB/s this project had recorded as
    /// the drive's sequential ceiling, so the ceiling was the handle too.
    readers: Vec<DirectFile>,
    on_disk: u64,
}

impl Shard {
    /// The handle for reader `slot`. Falls back to the primary handle when the
    /// pool could not be opened, so a file-descriptor limit degrades throughput
    /// rather than failing the run.
    fn reader(&self, slot: usize) -> &DirectFile {
        if self.readers.is_empty() {
            &self.file
        } else {
            &self.readers[slot % self.readers.len()]
        }
    }
}

/// How many extra handles each shard opens.
///
/// Eight is where the per-handle curve above flattens. Higher costs descriptors
/// and buys nothing; the drive is saturated by then.
pub(crate) const READER_HANDLES: usize = 8;

/// A model spread across one or more GGUF shards.
pub struct Model {
    shards: Vec<Shard>,
    tensors: BTreeMap<String, Location>,
    architecture: String,
    metadata: chaos_gguf::Metadata,
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
        let mut metadata = chaos_gguf::Metadata::new();
        let mut declared_tensor_count = None;

        for (idx, path) in paths.iter().enumerate() {
            // `--no-direct-io` sets `CHAOS_IO=buffered`. Direct I/O bypasses
            // the page cache, which is what makes streaming a 144 GB model
            // predictable -- but on a filesystem that refuses it, or when the
            // same model is read repeatedly and the page cache is *wanted*,
            // buffered is the better answer. Both are real modes here, which
            // is why the flag exists rather than being declined.
            let buffered = std::env::var("CHAOS_IO")
                .map(|v| v.eq_ignore_ascii_case("buffered"))
                .unwrap_or(false);
            let opened = if buffered {
                DirectFile::open_buffered(path)
            } else {
                DirectFile::open(path)
            };
            let file = opened.map_err(|source| Error::Io {
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
                let size = t
                    .size_bytes()
                    .ok_or_else(|| Error::Unsizable(t.name.clone()))?;
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

            // Open the reader pool now rather than on the first token: a handle
            // opened mid-stream would show up as a stall in exactly the phase
            // being optimised. A failure here is not fatal — `Shard::reader`
            // falls back to the primary handle.
            let readers: Vec<DirectFile> = (0..READER_HANDLES)
                .map_while(|_| DirectFile::open(path).ok())
                .collect();
            shards.push(Shard {
                file,
                readers,
                on_disk,
            });
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

    pub fn metadata(&self) -> &chaos_gguf::Metadata {
        &self.metadata
    }

    /// Replace one metadata entry -- llama.cpp's `--override-kv`.
    ///
    /// The escape hatch for a container whose metadata is wrong. A GGUF is
    /// often converted by a third party, and a mislabelled `rope.freq_base` or
    /// a missing `attention.head_count_kv` makes the model answer fluently and
    /// wrongly with nothing to point at. Overriding is safer than editing a
    /// multi-gigabyte file, and it is visible in the run that used it.
    ///
    /// The architecture is re-read afterwards because `general.architecture`
    /// is itself overridable, and it decides which config reader runs.
    pub fn override_metadata(&mut self, key: &str, value: chaos_gguf::Value) {
        self.metadata.insert(key.to_string(), value);
        if let Some(arch) = self
            .metadata
            .get("general.architecture")
            .and_then(chaos_gguf::Value::as_str)
        {
            self.architecture = arch.to_string();
        }
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(chaos_gguf::Value::as_u64)
    }

    /// An UNSCOPED string key, read exactly as given.
    ///
    /// Distinct from [`Self::arch_str`] on purpose: adapter metadata
    /// (`adapter.type`, `adapter.lora.alpha`) is not prefixed by the
    /// architecture, so the scoped accessor would look for
    /// `llama.adapter.type`, find nothing, and report a perfectly good adapter
    /// as not being one.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(chaos_gguf::Value::as_str)
    }

    /// An unscoped float key. See [`Self::meta_str`].
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(chaos_gguf::Value::as_f32)
    }

    /// Architecture-scoped metadata, e.g. `arch_u64("expert_count")`.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        self.get_u64(&format!("{}.{}", self.architecture, suffix))
    }

    /// An architecture-scoped string, e.g. `qwen3.rope.scaling.type`.
    pub fn arch_str(&self, suffix: &str) -> Option<&str> {
        self.metadata
            .get(&format!("{}.{}", self.architecture, suffix))
            .and_then(chaos_gguf::Value::as_str)
    }

    pub fn arch_f32(&self, suffix: &str) -> Option<f32> {
        self.metadata
            .get(&format!("{}.{}", self.architecture, suffix))
            .and_then(chaos_gguf::Value::as_f32)
    }

    /// An architecture-scoped array of floats, e.g. per-layer clamp limits.
    ///
    /// Several DeepSeek-V4 hyper-parameters are **per layer** rather than
    /// per model — `swiglu_clamp_exp`, `swiglu_clamp_shexp`,
    /// `attention.compress_ratios`. Reading only a scalar from those keys, or
    /// reading index 0 and applying it everywhere, gives a model that is
    /// correct on the first layer and quietly wrong on the rest.
    pub fn arch_f32_array(&self, suffix: &str) -> Option<Vec<f32>> {
        self.metadata
            .get(&format!("{}.{}", self.architecture, suffix))
            .and_then(chaos_gguf::Value::as_array)
            .map(|vs| vs.iter().filter_map(chaos_gguf::Value::as_f32).collect())
    }

    pub fn arch_i64_array(&self, suffix: &str) -> Option<Vec<i64>> {
        self.metadata
            .get(&format!("{}.{}", self.architecture, suffix))
            .and_then(chaos_gguf::Value::as_array)
            .map(|vs| {
                vs.iter()
                    .filter_map(|v| v.as_f32().map(|f| f as i64))
                    .collect()
            })
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

    /// Read part of a tensor — one expert's slice out of a stacked bank.
    ///
    /// This is what makes streaming cheap: a routed-expert tensor holds every
    /// expert for a layer, but a token needs only a few, and each is a
    /// contiguous run. Reading the run instead of the tensor is the difference
    /// between 1 GiB and 16 GiB per token.
    pub fn read_tensor_range(&self, name: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::UnknownTensor(name.to_string()))?;
        if offset + len > loc.size {
            return Err(Error::NotDownloaded {
                name: format!("{name} (range {offset}+{len})"),
                need: offset + len,
                have: loc.size,
            });
        }
        let shard = &self.shards[loc.shard];
        let at = loc.file_offset + offset;
        if at + len > shard.on_disk {
            return Err(Error::NotDownloaded {
                name: name.to_string(),
                need: at + len,
                have: shard.on_disk,
            });
        }
        shard
            .file
            .read_at(at, len as usize)
            .map_err(|source| Error::Io {
                path: shard.file.path().to_path_buf(),
                source,
            })
    }

    /// Read part of a tensor **into memory the caller already owns**.
    ///
    /// The streaming path stacks several expert slices into one buffer to bind
    /// as a single tensor. [`Self::read_tensor_range`] makes that cost two full
    /// copies of every byte — one out of the I/O scratch into a fresh `Vec`,
    /// another out of that `Vec` into the stack. This writes the slice straight
    /// into its final position instead.
    ///
    /// Returns **how many bytes were copied** through a scratch buffer on the
    /// way in; zero means the drive wrote every byte into `dst` itself. See
    /// [`chaos_io::SkewedBuf`] for how a caller arranges that.
    pub fn read_range_into(&self, name: &str, offset: u64, dst: &mut [u8]) -> Result<usize> {
        self.read_range_into_via(name, offset, dst, 0)
    }

    /// [`Self::read_range_into`], but through reader handle `slot`.
    ///
    /// Concurrent readers must pass **distinct** slots. Sharing one handle
    /// serialises them in the OS and holds the drive at queue depth 1 — see
    /// [`Shard::readers`] for the measurement. This is the only difference
    /// between the two, and it is worth 31% of expert-read throughput.
    pub fn read_range_into_via(
        &self,
        name: &str,
        offset: u64,
        dst: &mut [u8],
        slot: usize,
    ) -> Result<usize> {
        let len = dst.len() as u64;
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::UnknownTensor(name.to_string()))?;
        if offset + len > loc.size {
            return Err(Error::NotDownloaded {
                name: format!("{name} (range {offset}+{len})"),
                need: offset + len,
                have: loc.size,
            });
        }
        let shard = &self.shards[loc.shard];
        let at = loc.file_offset + offset;
        if at + len > shard.on_disk {
            return Err(Error::NotDownloaded {
                name: name.to_string(),
                need: at + len,
                have: shard.on_disk,
            });
        }
        shard
            .reader(slot)
            .read_at_into(at, dst)
            .map_err(|source| Error::Io {
                path: shard.file.path().to_path_buf(),
                source,
            })
    }

    /// Where a tensor's bytes begin in its shard file, and whether that offset
    /// is sector-aligned.
    ///
    /// Callers stacking slices need this to know whether they can build an
    /// aligned destination that the drive can write into directly — GGUF pads
    /// tensor data to `general.alignment`, which defaults to 32, not 4096.
    pub fn range_is_sector_aligned(&self, name: &str, offset: u64) -> bool {
        self.tensors
            .get(name)
            .is_some_and(|loc| (loc.file_offset + offset) % chaos_io::ALIGN as u64 == 0)
    }

    /// Read a whole tensor into a buffer shaped for the drive to fill directly,
    /// shared so it can be bound repeatedly without ever being copied.
    ///
    /// This is what residency wants: the bytes are read once, then bound into a
    /// `ggml` context on every block of every token. Returning `Vec<u8>` would
    /// mean copying them into an `Arc` to share them, and the skew is what lets
    /// the drive write into the allocation in the first place — see
    /// [`chaos_io::SkewedBuf`].
    pub fn read_tensor_shared(&self, name: &str) -> Result<Arc<SkewedBuf>> {
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::UnknownTensor(name.to_string()))?;
        let (size, file_offset) = (loc.size, loc.file_offset);
        let mut buf = SkewedBuf::new(size as usize, SkewedBuf::skew_for(file_offset));
        self.read_range_into(name, 0, &mut buf)?;
        Ok(Arc::new(buf))
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
