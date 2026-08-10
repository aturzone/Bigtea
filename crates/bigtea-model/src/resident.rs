//! Load the always-read weights into RAM and keep them there.
//!
//! This is the step that decides whether a model is usable on a small machine.
//! The always-read weights — attention, routers, shared experts, embeddings —
//! are touched on *every* token. Resident, they cost one read for the whole
//! session. Not resident, they are re-read from disk on every token forever.
//!
//! # The budget is a hard ceiling, not a target
//!
//! Loading past what RAM can hold does not degrade gracefully; it makes the OS
//! swap, and swapping is **slower than the streaming it was meant to replace**.
//! So the loader refuses. A model that will not fit is reported as not fitting,
//! with the number, rather than started and left to thrash.
//!
//! # Why it loads into our own memory rather than mapping the file
//!
//! Owning the allocation is the point: it cannot be evicted behind our back,
//! it does not double-count against the page cache, and its cost is exactly
//! what we accounted for. A mapped file offers none of those guarantees
//! precisely when memory is tight — which is always, here.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use bigtea_io::SkewedBuf;

use crate::{Error, Model, Result};

const GIB: f64 = (1u64 << 30) as f64;

/// Weights held in RAM for the whole session.
///
/// Buffers are shared rather than owned outright, because residency only pays
/// off if binding is free. Every block of every token binds these into a `ggml`
/// context; handing out `&[u8]` would force the binder to copy, which is the
/// cost residency exists to avoid. [`Self::get_shared`] hands out a refcount
/// bump instead, and the address never moves.
pub struct ResidentSet {
    tensors: HashMap<String, Arc<SkewedBuf>>,
    bytes: u64,
    skipped: Vec<Skipped>,
}

/// A tensor that was planned as resident but could not be loaded.
#[derive(Debug, Clone)]
pub struct Skipped {
    pub name: String,
    pub size: u64,
    pub reason: SkipReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Its bytes are not on disk yet.
    NotDownloaded,
    /// Loading it would have exceeded the budget.
    OverBudget,
}

impl fmt::Display for SkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SkipReason::NotDownloaded => f.write_str("not downloaded"),
            SkipReason::OverBudget => f.write_str("over budget"),
        }
    }
}

/// What a load attempt produced.
#[derive(Debug, Clone)]
pub struct LoadReport {
    pub loaded_tensors: usize,
    pub loaded_bytes: u64,
    pub budget_bytes: u64,
    pub skipped_over_budget: u64,
    pub skipped_not_downloaded: u64,
    pub seconds: f64,
}

impl LoadReport {
    /// Everything that should be resident, is.
    pub fn complete(&self) -> bool {
        self.skipped_over_budget == 0 && self.skipped_not_downloaded == 0
    }

    /// Achieved load throughput, useful as an independent check on the
    /// bandwidth the probe measured.
    pub fn bytes_per_sec(&self) -> f64 {
        if self.seconds <= 0.0 {
            return 0.0;
        }
        self.loaded_bytes as f64 / self.seconds
    }
}

impl fmt::Display for LoadReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "loaded {} tensors, {:.2} GiB of {:.2} GiB budget in {:.1}s ({:.2} GB/s)",
            self.loaded_tensors,
            self.loaded_bytes as f64 / GIB,
            self.budget_bytes as f64 / GIB,
            self.seconds,
            self.bytes_per_sec() / 1e9
        )?;
        if self.skipped_over_budget > 0 {
            write!(
                f,
                "; {:.2} GiB did not fit and will be re-read every token",
                self.skipped_over_budget as f64 / GIB
            )?;
        }
        if self.skipped_not_downloaded > 0 {
            write!(
                f,
                "; {:.2} GiB not downloaded yet",
                self.skipped_not_downloaded as f64 / GIB
            )?;
        }
        Ok(())
    }
}

impl ResidentSet {
    /// Load every always-read tensor that fits in `budget_bytes`.
    ///
    /// Largest-first: within the always-read class every byte saves the same
    /// amount per token, so ordering cannot improve the outcome — it only
    /// determines how many tensors end up stranded, and fewer is simpler.
    ///
    /// Tensors whose bytes have not downloaded are skipped and reported rather
    /// than faked. Never returns partially-filled buffers: a short read is an
    /// error, because silently wrong weights produce silently wrong output.
    pub fn load(model: &Model, budget_bytes: u64) -> Result<(Self, LoadReport)> {
        Self::load_with_progress(model, budget_bytes, |_, _| {})
    }

    /// Load using `threads` concurrent readers.
    ///
    /// A single synchronous read cannot saturate an NVMe device: the drive
    /// wants several requests in flight to reach its rated bandwidth, and one
    /// blocking read at a time leaves most of it idle. Issuing reads from
    /// several threads is the simplest way to get queue depth without an async
    /// runtime.
    ///
    /// Positioned reads (`seek_read` / `pread`) carry their own offset and do
    /// not share a file cursor, so concurrent readers on one handle are safe
    /// and need no locking.
    pub fn load_parallel(
        model: &Model,
        budget_bytes: u64,
        threads: usize,
    ) -> Result<(Self, LoadReport)> {
        let start = std::time::Instant::now();
        let plan = Plan::build(model, budget_bytes)?;
        let threads = threads.max(1).min(plan.to_load.len().max(1));

        let mut tensors = HashMap::with_capacity(plan.to_load.len());
        let mut bytes = 0u64;

        if threads == 1 {
            for (name, size) in &plan.to_load {
                let data = read_checked(model, name, *size)?;
                bytes += *size;
                tensors.insert(name.clone(), data);
            }
        } else {
            // Round-robin rather than contiguous chunks: tensor sizes vary by
            // orders of magnitude, so contiguous slices of a size-sorted list
            // would hand one thread all the large tensors.
            let results = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let slice = &plan.to_load;
                        scope.spawn(move || {
                            let mut out = Vec::new();
                            for (name, size) in slice.iter().skip(t).step_by(threads) {
                                out.push(
                                    read_checked(model, name, *size).map(|d| (name.clone(), d)),
                                );
                            }
                            out
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap_or_default())
                    .collect::<Vec<_>>()
            });
            for result in results {
                let (name, data) = result?;
                bytes += data.len() as u64;
                tensors.insert(name, data);
            }
        }

        let report = LoadReport {
            loaded_tensors: tensors.len(),
            loaded_bytes: bytes,
            budget_bytes,
            skipped_over_budget: plan.over_budget,
            skipped_not_downloaded: plan.not_downloaded,
            seconds: start.elapsed().as_secs_f64(),
        };
        Ok((
            ResidentSet {
                tensors,
                bytes,
                skipped: plan.skipped,
            },
            report,
        ))
    }

    /// As [`Self::load`], calling `progress(loaded_bytes, total_planned)` as it
    /// goes — loading several gigabytes takes long enough to warrant feedback.
    pub fn load_with_progress(
        model: &Model,
        budget_bytes: u64,
        mut progress: impl FnMut(u64, u64),
    ) -> Result<(Self, LoadReport)> {
        let start = std::time::Instant::now();

        // Always-read tensors only. Routed experts stream by design.
        let mut planned: Vec<(String, u64)> = model
            .tensor_names()
            .filter_map(|name| {
                let loc = model.location(name)?;
                (!loc.routed_expert).then(|| (name.to_string(), loc.size))
            })
            .collect();
        planned.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

        let total_planned: u64 = planned.iter().map(|(_, s)| *s).sum();

        let mut tensors = HashMap::with_capacity(planned.len());
        let mut skipped = Vec::new();
        let mut bytes = 0u64;
        let mut over_budget = 0u64;
        let mut not_downloaded = 0u64;

        for (name, size) in planned {
            if bytes.saturating_add(size) > budget_bytes {
                over_budget += size;
                skipped.push(Skipped {
                    name,
                    size,
                    reason: SkipReason::OverBudget,
                });
                continue;
            }
            if !model.is_available(&name)? {
                not_downloaded += size;
                skipped.push(Skipped {
                    name,
                    size,
                    reason: SkipReason::NotDownloaded,
                });
                continue;
            }

            let data = model.read_tensor_shared(&name)?;
            if data.len() as u64 != size {
                // A short read here would put wrong bytes into the model.
                return Err(Error::NotDownloaded {
                    name,
                    need: size,
                    have: data.len() as u64,
                });
            }
            bytes += size;
            tensors.insert(name, data);
            progress(bytes, total_planned);
        }

        let report = LoadReport {
            loaded_tensors: tensors.len(),
            loaded_bytes: bytes,
            budget_bytes,
            skipped_over_budget: over_budget,
            skipped_not_downloaded: not_downloaded,
            seconds: start.elapsed().as_secs_f64(),
        };
        Ok((
            ResidentSet {
                tensors,
                bytes,
                skipped,
            },
            report,
        ))
    }

    /// Bytes held in RAM.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// A resident tensor's bytes, or `None` if it is not resident — in which
    /// case the caller must stream it.
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|b| &b[..])
    }

    /// The same bytes as a shared handle, for binding without copying.
    pub fn get_shared(&self, name: &str) -> Option<Arc<SkewedBuf>> {
        self.tensors.get(name).cloned()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The names held, for a caller that wants to walk the set.
    ///
    /// Owned rather than borrowed because the one caller — weight repacking —
    /// mutates the set as it goes.
    pub fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Hand a resident tensor's bytes to a caller that will hold them itself,
    /// and stop accounting for them here.
    ///
    /// # Why a set that has just been filled would want to give bytes back
    ///
    /// Weight repacking rearranges a quantised tensor into the layout the CPU
    /// kernels want, and it allocates its own buffer to do it. Keeping the
    /// original alongside would **double** the memory for everything it
    /// touches, and this set exists precisely because that memory is the
    /// binding constraint — a 7.38 GiB always-read set on a 15.7 GiB machine
    /// has no second copy to give.
    ///
    /// The rearranged bytes are a complete replacement, so the caller takes
    /// this handle, repacks from it, and drops it. Peak cost is one tensor
    /// rather than the whole set, and the steady state is unchanged.
    ///
    /// The name is then absent: [`get`](Self::get) and
    /// [`get_shared`](Self::get_shared) report `None` and a caller that has not
    /// been told about the replacement falls back to streaming it from disk —
    /// slower, but correct. **It is the taker's job to serve it instead.**
    pub fn take(&mut self, name: &str) -> Option<Arc<SkewedBuf>> {
        let taken = self.tensors.remove(name)?;
        self.bytes = self.bytes.saturating_sub(taken.len() as u64);
        Some(taken)
    }

    /// Return bytes a caller took but could not use after all.
    ///
    /// Repacking asks `ggml` whether it has a kernel for a given type and
    /// shape, and the shape check is a heuristic — `ggml` has the final say and
    /// can still decline. Without this the tensor would be left absent and
    /// stream from disk on every token, which is much worse than the
    /// rearrangement simply not happening.
    pub fn put_back(&mut self, name: String, bytes: Arc<SkewedBuf>) {
        self.bytes += bytes.len() as u64;
        self.tensors.insert(name, bytes);
    }

    /// Tensors that were planned but not loaded, and why.
    pub fn skipped(&self) -> &[Skipped] {
        &self.skipped
    }
}

/// Which always-read tensors will be loaded, decided before any I/O.
///
/// Separating the decision from the work is what makes parallel loading
/// possible: the budget is a running total, which is inherently sequential,
/// so it is resolved first and the reads are then independent.
struct Plan {
    to_load: Vec<(String, u64)>,
    skipped: Vec<Skipped>,
    over_budget: u64,
    not_downloaded: u64,
}

impl Plan {
    fn build(model: &Model, budget_bytes: u64) -> Result<Self> {
        let mut planned: Vec<(String, u64)> = model
            .tensor_names()
            .filter_map(|name| {
                let loc = model.location(name)?;
                (!loc.routed_expert).then(|| (name.to_string(), loc.size))
            })
            .collect();
        planned.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

        let mut to_load = Vec::with_capacity(planned.len());
        let mut skipped = Vec::new();
        let (mut running, mut over_budget, mut not_downloaded) = (0u64, 0u64, 0u64);

        for (name, size) in planned {
            if running.saturating_add(size) > budget_bytes {
                over_budget += size;
                skipped.push(Skipped {
                    name,
                    size,
                    reason: SkipReason::OverBudget,
                });
                continue;
            }
            if !model.is_available(&name)? {
                not_downloaded += size;
                skipped.push(Skipped {
                    name,
                    size,
                    reason: SkipReason::NotDownloaded,
                });
                continue;
            }
            running += size;
            to_load.push((name, size));
        }
        Ok(Plan {
            to_load,
            skipped,
            over_budget,
            not_downloaded,
        })
    }
}

/// Read a tensor, refusing anything but its exact declared size.
fn read_checked(model: &Model, name: &str, size: u64) -> Result<Arc<SkewedBuf>> {
    let data = model.read_tensor_shared(name)?;
    if data.len() as u64 != size {
        return Err(Error::NotDownloaded {
            name: name.to_string(),
            need: size,
            have: data.len() as u64,
        });
    }
    Ok(data)
}

impl fmt::Debug for ResidentSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResidentSet")
            .field("tensors", &self.tensors.len())
            .field("bytes", &self.bytes)
            .field("skipped", &self.skipped.len())
            .finish()
    }
}
