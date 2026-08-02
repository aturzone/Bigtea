---
epic: cross-engine-advisor
status: proposed (pending ADR acceptance)
links: [../decisions/strategy-post-waste.md, ../research/waste-engine-verified.md, ../research/moe-landscape-2026-08.md]
---

## Constraint

Every ticket in this epic must be buildable and testable on Atur's own laptop (Windows,
15.7 GB RAM, RTX 3050 6 GB, Python 3.11, stdlib-only preference). No ticket may require
the capable rig this project has been blocked on — that unblock is the point of this epic.

## Tickets
- [ ] T1: Engine capability registry — a declarative data file (engine → supported model
  families, supported accelerators [CUDA/ROCm/Metal/CPU-only], quant formats, offload
  regime) + a loader, with a `source` + `date` field per row so stale facts are visible —
  depends: ../decisions/strategy-post-waste.md — evidence: ../research/waste-engine-verified.md, ../research/moe-landscape-2026-08.md — acceptance: loader parses the registry file and returns a typed record per engine with all fields non-empty including source+date, verified by a unit test with no network/weights access.
- [ ] T2: Pre-download working-set + floor calculator — given only a HuggingFace-style
  `config.json` (layer count, expert count, experts-per-token, hidden dims, quant bits),
  compute one token's expert working-set size and the RAM floor, with zero weight
  downloads — depends: ../research/waste-engine-verified.md — evidence: ../decisions/strategy-post-waste.md (physics section) — acceptance: unit test feeds a synthetic config.json for Kimi-Linear-48B's published architecture and the calculator's floor is within 10% of the verified 1.28 GB figure, with no file I/O beyond the config.json.
- [ ] T3: Budget resolver — given available memory and the T2 floor/working-set size,
  refuse any budget below the floor, otherwise select the largest whole working-set
  multiple that fits; must use measured available memory, never a fixed fraction of
  physical RAM — depends: cross-engine-advisor.md#T2, ../decisions/strategy-post-waste.md — evidence: ../decisions/strategy-post-waste.md (rejects the "7/8 of physical RAM" fabrication) — acceptance: unit tests cover (a) budget below floor → explicit refusal, not a silent clamp, (b) budget selection returns an integer multiple of working-set size, (c) no constant resembling 7/8 or 0.875 appears anywhere in the resolver's source.
- [ ] T4: Regime classifier + per-regime prediction — classify a (machine, model, engine)
  triple as `gpu` / `disk` / `hybrid` from the T1 registry + T2/T3 outputs, then apply the
  matching throughput model (GPU lane: bandwidth-bound; disk lane: SSD-read-bound, where
  expert I/O can exceed 50% of a decode step) — depends: cross-engine-advisor.md#T1, cross-engine-advisor.md#T2, cross-engine-advisor.md#T3 — evidence: ../research/waste-engine-verified.md — acceptance: unit test on the laptop's own profile (RTX 3050 6 GB, 15.7 GB RAM) classifies Kimi-Linear-48B as `disk`/CPU-bound (no VRAM required) and Kimi K3 as refused-below-floor, matching the ADR's stated hardware reality.
- [ ] T5: Cross-engine recommendation output — score every engine in the T1 registry for a
  given (machine, model) pair using T2-T4, rank the results, emit predicted tok/s with an
  explicit confidence band, and explain refusals in plain text (e.g. "engine X cannot run
  this model family", "floor exceeds your RAM by 2x") — depends: cross-engine-advisor.md#T1, cross-engine-advisor.md#T4 — evidence: ../decisions/strategy-post-waste.md (Option B) — acceptance: running the recommender for (laptop profile, Kimi K3) produces a ranked list where every entry is either a tok/s+confidence prediction or a human-readable refusal string, and no entry is silently omitted.
- [ ] T6: Real-hardware calibration on Atur's own laptop — run Kimi-Linear-48B (19 GiB
  container, ~1.28 GB RAM floor, CPU-only, no VRAM) under WASTE on the 15.7 GB laptop and
  compare measured tok/s against T5's prediction — the project's first real measured MoE
  data point, requiring no rented rig — depends: cross-engine-advisor.md#T5 — evidence: ../research/waste-engine-verified.md (Minimum hardware reality) — acceptance: a recorded run produces one measured-tok/s number, one predicted-tok/s number, and their percent error, with the run log and hardware fingerprint saved; no specific accuracy threshold required at this stage.

## Issues
- T1 #28 · Engine capability registry
- T2 #29 · Pre-download working-set + floor calculator
- T3 #30 · Budget resolver
- T4 #31 · Regime classifier + per-regime prediction
- T5 #32 · Cross-engine recommendation output
- T6 #33 · Real-hardware calibration on Atur's laptop
