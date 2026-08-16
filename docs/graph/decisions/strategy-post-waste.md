---
decision: Chaos's position after WASTE/ds4 — what "be the best" actually means
status: proposed        # Atur must decide; extends (does not contradict) fork-vs-wrapper.md
links: [../research/waste-engine-verified.md, ../research/moe-landscape-2026-08.md, ../research/advisory-evaluation-deepseek-0731.md, fork-vs-wrapper.md]
---

## Context

- **The "big bang" already fired, twice.** Running a frontier MoE on consumer hardware is *done*:
  WASTE runs Kimi K3 (2.8T) on a 64 GB Mac at ~0.5 tok/s; ds4/DwarfStar (antirez) runs
  DeepSeek-V4-Flash on Metal/CUDA/ROCm at ~20k stars. Chaos cannot claim that headline — it was
  claimed in June–July 2026 (../research/moe-landscape-2026-08.md).
- **Every engine is model-family locked.** WASTE = Kimi only (converter hard-built for Moonshot's
  KDA+MLA architecture, *no DeepSeek support, none planned*). ds4 = DeepSeek-V4/GLM-5.2 only.
  ktransformers = DeepSeek-class (native V4-Flash since v0.6.2). llama.cpp = broad but weakest
  offload. (../research/waste-engine-verified.md, ../research/moe-landscape-2026-08.md)
- **They are also accelerator-locked.** WASTE has **zero CUDA/ROCm** and its only GPU path (Metal)
  is *22% slower than its own CPU path*. ds4 is Mac-primary. ktransformers is NVIDIA-first.
- **Therefore the user's actual question has no answer today**: *"I have this machine and I want
  that model — which engine, which quant, which budget, and how fast will it be?"* Every preflight
  that exists (`waste plan`, ds4's 80% resolver, llama.cpp `--fit`) is single-engine and requires
  you to have already downloaded the model.
- **The physics is settled and public**: cache floor = one token's working set (below it, hit rate
  is *exactly* zero); usable cache only in whole working-set multiples; past the sweet spot the OS
  paging cliff is catastrophic, not graceful. Three engines converged on this independently. It is
  a law to build on, not a secret to discover.
- **ktransformers observability is still unclaimed** — verified twice (2026-07-28, re-verified
  2026-08-02): nothing shipped, nothing on the roadmap.

## Options

### A. Stay as scoped — a wrapper over ktransformers whose differentiator is observability
- Wins: already planned, tickets exist, T4/T5 ground is verified-open.
- Costs: modest ambition; "nicer front-end for one engine" is not "the best" at anything, and a
  single upstream release (or leloch's llama.cpp cache merging) reprices the whole differentiator.

### B. Reposition to the cross-engine advisor: predict → choose → tune → verify
- Wins: answers the question nobody answers; **absorbs** option A (observability is *how* you
  verify a prediction, not a separate product); the CLI shape already built in wrapper-core
  (`probe`/`recommend`/`launch`/`bench`) is literally this; the engines' family-lock *is* the moat
  — the more engines fragment, the more a router is needed.
- Costs: requires being genuinely multi-engine (at least ktransformers + llama.cpp) before the
  pitch is true; prediction accuracy becomes the product, so measurement rigor is existential.

### C. Build an engine and chase the headline
- Wins: the headline.
- Costs: rejected by fork-vs-wrapper.md and reinforced here — WASTE is specialist systems
  engineering by a funded org, ds4 is antirez. Entering this lane solo, on a $20/mo budget, against
  a 20k-star incumbent, is not a plan.

## Recommendation

**Option B.** Chaos is not "a ktransformers wrapper" — it is **the answer to which-engine-for-my-
machine-and-model, with a predicted tok/s before you download 600 GB**. Three verified facts force
this: engines are model-family locked *and* accelerator-locked, so the choice is real and hard;
every existing preflight is single-engine and post-download; and the underlying physics is now
public and identical across engines, which means one predictor can score all of them.

This **extends rather than contradicts** the accepted fork-vs-wrapper ADR — still no fork, still
drive upstream through stable surfaces, still never vendor/pin. It re-aims the *value* from
"observability for one engine" to "a trustworthy recommendation across engines," and observability
becomes the verification loop that makes recommendations credible rather than a standalone bet.

**Strongest counterargument**: a router is only as good as the engines it routes to, so Chaos
inherits every upstream's install pain and can be flattened if one engine becomes universal. It
loses because fragmentation is *increasing*, not decreasing — WASTE (Kimi) and ds4 (DeepSeek/GLM)
both launched as *new* family-locked engines within 90 days, and neither can run the other's model
family. Convergence to one engine is the scenario that would kill this, and the market is moving
the other way.

## Consequences

- **wrapper-core** becomes the flagship, not plumbing: `recommend` must score ≥2 engines and emit a
  predicted tok/s + confidence, not just flags for one.
- **hardware-profiler** gains the `regime ∈ {gpu, disk, hybrid}` branch and a real SSD-bandwidth
  term — the disk regime is now mainstream (correction applied to hardware-profiling.md).
- **gap-closure T4/T5** keep their scope but change their *justification*: observability exists to
  validate predictions. The T0 recorder gate still governs them.
- **benchmark-harness** must key every report to **working-set multiples** and record `regime`,
  `working_set_bytes`, `budget_multiple`, model-volume bandwidth — a bare "46% hit rate" is
  uninterpretable (fine on the GPU lane, terrible on the disk lane).
- **Adopt the budget resolver rule**: refuse below floor; whole working-set multiples only; take
  the largest that fits available memory. (Adopt the rule, **not** the advisory's invented "7/8 of
  physical RAM" constant.)
- **Future backends**: ds4/DwarfStar is the warm candidate (MIT, DeepSeek+GLM, CUDA/Metal/ROCm).
  **WASTE is rejected as a wrap target** for v1 scope — Kimi-only and no CUDA serves neither of
  our axes.
- **Do not rebuild**: per-expert bit allocation (flat 1.01×), batching-for-I/O (1.63× ceiling),
  3-bit trunk (logit collapse). **But prefetch is NOT refuted** — WASTE revived it via next-layer
  router weights on the current hidden state (59.0% recall@16) and shipped it for 1.17×, one day
  after refuting the co-occurrence variant. Refutations bind the *mechanism tested*, not the idea.
- **Revisit if**: one engine becomes genuinely universal across model families and accelerators;
  or ktransformers ships observability; or ds4/WASTE add a cross-engine advisor themselves.

## The hardware reality this decision must state plainly

> **CORRECTED 2026-08-02 (later wins).** An earlier version of this section claimed K3 on a 15.7 GB
> laptop is "not possible" and needs a RAM upgrade. **That was wrong** — it quoted WASTE's *design*
> floor as if it were physics. See `../research/k3-on-16gb-feasibility.md`.

**K3's 29.06 GB floor is a policy choice, not a physical law.** It is dominated by WASTE's decision
to keep the 27.28 GB dense trunk RAM-resident. Three independent proofs from WASTE's own docs:
1. They already stream one dense component — the 1.11 GB embedding table — with **bit-identical
   logits and zero throughput cost** (LEARNED.md §13).
2. They tested streaming the LM head too. It worked **correctly**; they rejected it on *speed*
   grounds, not correctness.
3. `WASTE_E_RAM_BUDGET` is an explicit **refusal policy** — the engine chooses to refuse rather
   than run correct-but-slow.

Nothing in K3's mathematics requires any byte to be resident. **The true correctness-only floor is
~5–6 GB.** A 16 GB machine clears it with ~13 GB to spare.

**The real cost is throughput, and it is severe.** At 16 GB you must stream ~52% of the trunk
every token: **~31.5 GB of reads per token** (vs ~17 GB in WASTE's 29 GB build). Every spare byte
should go to *trunk* residency, never expert cache — expert cache is provably worthless below a
17.0–17.4 GB working set, while trunk residency pays back linearly and cliff-free at 1 GB/token
per resident GB. On Atur's PCIe-3-class NVMe (~3.5 GB/s) that lands near **~0.056 tok/s ≈ 18
seconds per token**. This is a demo, not a usable assistant — and it is a genuine world-first.

**The actual binding constraint is disk, not RAM.** Measured 2026-08-02: Atur's drive is a 953 GB
NVMe with **745.9 GB free**. WASTE's K3 container is **982 GB**; the native HF checkpoint is
**1.56 TB** (not the 594 GB previously cited). **He is ~236 GB short.** The gate is therefore a
**~$100–150 2 TB NVMe**, not a RAM upgrade — cheaper than the RAM, and it unblocks the milestone.

**Runs on the laptop today, unchanged:** **Kimi-Linear-48B** (19 GiB container, 1.28 GB floor,
~10.7 tok/s, CPU-only). Real proof point available now, zero purchases.

## Option D — Build the K3-on-16GB milestone (added 2026-08-02, Atur's direction)

Atur's stated goal is to be **first to run K3 on a laptop** using mathematical/physical/engineering
methods. Research confirms this is **unclaimed and feasible**: nobody has published K3 — or any
1T+ model — below WASTE's 29.06 GB floor; WASTE itself refuses to try.

- **This is not "write an engine from scratch"** (which fork-vs-wrapper rightly rejected). It is a
  bounded, additive extension to WASTE's Apache-2.0 C11 codebase, which already ships ~80% of the
  needed machinery: streamed `pread` I/O, a read-ahead thread pattern, an oracle-diff correctness
  harness, and a budget resolver. What's missing is applying all of it to the *trunk* instead of
  only to experts, and degrading instead of refusing below the floor.
- **Trunk streaming is easier than the expert case WASTE already solved** — trunk access order is
  100% deterministic every token (no routing to predict), and residency policy needs no eviction
  heuristic because every trunk byte is read every token (benefit is uniform per byte).
- **It is a legitimate fork rationale**: upstream refuses this mode *by policy*. Apache-2.0 permits
  it; obligations are light (state changes, keep the license, rename).
- **Prior art validates the technique, not the target**: AirLLM, FlexGen, DeepSpeed ZeRO-Inference
  all ship dense layer-wise streaming with double buffering — none applied to a 1T+ MoE trunk, none
  targeting a 16 GB *total-system* envelope.
- **Spillover value beyond the stunt**: proper async I/O overlap (io_uring / IOCP) would speed up
  *both* the new 16 GB regime and WASTE's existing 29 GB regime.
- **Honest risks**: ~0.02–0.4 tok/s depending on drive; every number above the WASTE-sourced ones
  is first-principles, not measured; upstream could ship a below-floor mode first and erase the
  claim; and this competes for the same solo, budget-capped hours as Option B.

**Relationship to Option B**: not mutually exclusive, but they are different games — B is a durable
tooling product, D is a milestone with a headline. D's measurements feed B's predictor with the
only real 1T-scale calibration data anyone would have. **Atur's call on sequencing.**
