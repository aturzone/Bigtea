---
decision: Bigtea's position after WASTE/ds4 — what "be the best" actually means
status: proposed        # Atur must decide; extends (does not contradict) fork-vs-wrapper.md
links: [../research/waste-engine-verified.md, ../research/moe-landscape-2026-08.md, ../research/advisory-evaluation-deepseek-0731.md, fork-vs-wrapper.md]
---

## Context

- **The "big bang" already fired, twice.** Running a frontier MoE on consumer hardware is *done*:
  WASTE runs Kimi K3 (2.8T) on a 64 GB Mac at ~0.5 tok/s; ds4/DwarfStar (antirez) runs
  DeepSeek-V4-Flash on Metal/CUDA/ROCm at ~20k stars. Bigtea cannot claim that headline — it was
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

**Option B.** Bigtea is not "a ktransformers wrapper" — it is **the answer to which-engine-for-my-
machine-and-model, with a predicted tok/s before you download 600 GB**. Three verified facts force
this: engines are model-family locked *and* accelerator-locked, so the choice is real and hard;
every existing preflight is single-engine and post-download; and the underlying physics is now
public and identical across engines, which means one predictor can score all of them.

This **extends rather than contradicts** the accepted fork-vs-wrapper ADR — still no fork, still
drive upstream through stable surfaces, still never vendor/pin. It re-aims the *value* from
"observability for one engine" to "a trustworthy recommendation across engines," and observability
becomes the verification loop that makes recommendations credible rather than a standalone bet.

**Strongest counterargument**: a router is only as good as the engines it routes to, so Bigtea
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

Running **Kimi K3 on Atur's own machine is not possible**: K3's floor is **29.06 GB RAM** and the
laptop has **15.7 GB** — short by ~2×, regardless of GPU or SSD, on any engine. Two honest paths:
- **Today, unchanged**: **Kimi-Linear-48B** (19 GiB container, **1.28 GB** RAM floor, ~10.7 tok/s,
  CPU-only — no VRAM needed) runs on this laptop now. That is a real, verifiable "frontier-ish MoE
  on my own machine" demo.
- **For K3 specifically**: the gate is a **RAM upgrade, not software** — the i7-13650HX platform
  takes 2× DDR5 SO-DIMM; a 64 GB kit (~$100–200) clears the 29.06 GB floor with headroom for the
  cache multiples that decide speed. No amount of Bigtea engineering substitutes for this.
