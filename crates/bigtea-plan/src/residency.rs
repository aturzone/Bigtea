//! Decide what lives in RAM and what streams from disk.
//!
//! # The policy, and why it is this one
//!
//! Weights fall into two classes with completely different economics:
//!
//! * **Always-read** — attention, routers, shared experts, embeddings, norms.
//!   Every token reads all of them. A resident byte here saves exactly one byte
//!   of reading per token, forever, unconditionally. The payoff is linear and
//!   certain.
//! * **Routed experts** — a token reads only the handful routing selects, and
//!   which ones changes every token. Caching these pays *nothing* until the
//!   cache can hold a whole token's working set, because below that an entry is
//!   evicted before it is ever reused.
//!
//! So the policy is not a heuristic, it is forced: **fill RAM with always-read
//! weights first, and only consider an expert cache with what is left over —
//! and only if that leftover clears one token's working set.** On the machine
//! class this targets it essentially never does, which is why the planner will
//! usually tell you to spend nothing on expert cache.
//!
//! Within the always-read class, every byte is worth the same (all are read
//! every token), so when the budget cannot hold all of them there is no clever
//! ordering to find. The planner packs largest-first purely to reduce the count
//! of half-loaded tensors, and says plainly how much did not fit.

use crate::GIB;
use bigtea_gguf::{Gguf, TensorInfo};

/// Where a tensor lives at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Loaded once into RAM and kept for the whole session.
    ResidentRam,
    /// Read from disk when routing selects it.
    StreamFromDisk,
}

impl std::fmt::Display for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Placement::ResidentRam => f.write_str("ram"),
            Placement::StreamFromDisk => f.write_str("disk"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Placed {
    pub name: String,
    pub bytes: u64,
    pub placement: Placement,
    /// True for routed-expert tensors, which are read only when selected.
    pub routed: bool,
}

/// A concrete plan: which tensors are resident, what it costs, what streams.
#[derive(Debug, Clone)]
pub struct Layout {
    pub placed: Vec<Placed>,
    /// RAM the resident set consumes.
    pub ram_used_bytes: u64,
    /// Budget the planner was given.
    pub ram_budget_bytes: u64,
    /// Always-read bytes that did not fit, and so are re-read every token.
    pub always_read_shortfall_bytes: u64,
    /// Always-read bytes in total.
    pub always_read_bytes: u64,
    /// Routed-expert bytes in total.
    pub expert_pool_bytes: u64,
    pub notes: Vec<String>,
}

impl Layout {
    pub fn resident_count(&self) -> usize {
        self.placed
            .iter()
            .filter(|p| p.placement == Placement::ResidentRam)
            .count()
    }

    pub fn all_always_read_resident(&self) -> bool {
        self.always_read_shortfall_bytes == 0
    }

    /// RAM left after the resident set — the only budget an expert cache could
    /// ever use.
    pub fn spare_ram_bytes(&self) -> u64 {
        self.ram_budget_bytes.saturating_sub(self.ram_used_bytes)
    }

    /// Whether an expert cache is worth building at all.
    ///
    /// It is worth it only when the spare budget clears one token's expert
    /// working set; below that the hit rate is zero, not merely low.
    pub fn expert_cache_is_worthwhile(&self, expert_bytes_per_token: u64) -> bool {
        expert_bytes_per_token > 0 && self.spare_ram_bytes() >= expert_bytes_per_token
    }
}

/// Build a residency plan for a container's tensors under a RAM budget.
///
/// `budget_bytes` is RAM available **for weights** — the caller must already
/// have subtracted OS, KV cache and scratch.
pub fn plan_layout(tensors: &[TensorInfo], budget_bytes: u64) -> Layout {
    // Split by class, keeping only tensors whose size we actually know. A
    // tensor we cannot size is a correctness problem, surfaced in notes rather
    // than silently packed.
    let mut always_read: Vec<(&TensorInfo, u64)> = Vec::new();
    let mut experts: Vec<(&TensorInfo, u64)> = Vec::new();
    let mut unsized_count = 0usize;

    for t in tensors {
        match t.size_bytes() {
            Some(size) if t.is_routed_expert() => experts.push((t, size)),
            Some(size) => always_read.push((t, size)),
            None => unsized_count += 1,
        }
    }

    let always_read_bytes: u64 = always_read.iter().map(|(_, s)| *s).sum();
    let expert_pool_bytes: u64 = experts.iter().map(|(_, s)| *s).sum();

    // Largest-first: every always-read byte is worth the same, so this is only
    // about leaving fewer stranded tensors, not about picking better ones.
    always_read.sort_by_key(|(_, s)| std::cmp::Reverse(*s));

    let mut placed = Vec::with_capacity(tensors.len());
    let mut ram_used = 0u64;
    let mut shortfall = 0u64;

    for (t, size) in &always_read {
        let fits = ram_used.saturating_add(*size) <= budget_bytes;
        if fits {
            ram_used += size;
        } else {
            shortfall += size;
        }
        placed.push(Placed {
            name: t.name.clone(),
            bytes: *size,
            placement: if fits {
                Placement::ResidentRam
            } else {
                Placement::StreamFromDisk
            },
            routed: false,
        });
    }

    // Routed experts always stream at this machine class. Making them resident
    // would only be possible if the whole pool fit, which is precisely the case
    // this tool does not target.
    for (t, size) in &experts {
        placed.push(Placed {
            name: t.name.clone(),
            bytes: *size,
            placement: Placement::StreamFromDisk,
            routed: true,
        });
    }

    let mut notes = Vec::new();
    if shortfall == 0 && always_read_bytes > 0 {
        notes.push(format!(
            "all {:.2} GiB of always-read weights are resident: they are read \
             once and then cost nothing per token",
            always_read_bytes as f64 / GIB as f64
        ));
    } else if shortfall > 0 {
        notes.push(format!(
            "{:.2} GiB of always-read weights did not fit and will be re-read \
             every token. Freeing RAM, or a smaller quantization, removes this \
             cost directly -- it is the highest-value change available",
            shortfall as f64 / GIB as f64
        ));
    }
    if unsized_count > 0 {
        notes.push(format!(
            "{unsized_count} tensor(s) had an unknown type and were excluded \
             from the plan -- results are incomplete"
        ));
    }

    Layout {
        placed,
        ram_used_bytes: ram_used,
        ram_budget_bytes: budget_bytes,
        always_read_shortfall_bytes: shortfall,
        always_read_bytes,
        expert_pool_bytes,
        notes,
    }
}

/// Convenience: plan straight from a parsed container.
pub fn plan_from_gguf(gguf: &Gguf, budget_bytes: u64) -> Layout {
    plan_layout(&gguf.tensors, budget_bytes)
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let g = |b: u64| b as f64 / GIB as f64;
        writeln!(
            f,
            "budget           {:.2} GiB for weights",
            g(self.ram_budget_bytes)
        )?;
        writeln!(
            f,
            "resident         {:.2} GiB across {} tensors",
            g(self.ram_used_bytes),
            self.resident_count()
        )?;
        writeln!(
            f,
            "always-read      {:.2} GiB total, {:.2} GiB not resident",
            g(self.always_read_bytes),
            g(self.always_read_shortfall_bytes)
        )?;
        writeln!(f, "expert pool      {:.2} GiB (streams)", g(self.expert_pool_bytes))?;
        writeln!(f, "spare ram        {:.2} GiB", g(self.spare_ram_bytes()))?;
        for note in &self.notes {
            writeln!(f, "  ! {note}")?;
        }
        Ok(())
    }
}
