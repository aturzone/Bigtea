# 05 — Proposed actions for the Chaos graph (not applied)

Per Atur's instruction this folder stays outside `docs/graph/`. The items below are **proposals**,
each mapped to where it belongs and who owns the decision (all are Atur). Nothing in the graph
was modified while writing this folder.

## A1. Correct the SSD claim in `research/hardware-profiling.md`

- **What:** the node's line ~31 claim that SSD speed only affects load time is contradicted by
  measured evidence (WASTE streams a 982 GB model; 54.8% of a K3 token is expert I/O; enclosure
  bridge = 12.78 vs 0.94 GB/s ceiling).
- **Where:** amend the research node (note as a dated amendment, per graph convention), and add a
  pointer in `hardware-profiler` backlog T3.
- **Effort:** one node edit + one backlog line.

## A2. Working-set-multiple schema fields in `benchmark-harness`

- **What:** add to the run record: `regime ∈ {gpu, disk, hybrid}` (from `03-landscape` §3),
  `working_set_bytes`, `budget_multiple`, and `model_volume_bandwidth` (NVMe vs USB bridge).
- **Where:** `backlog/benchmark-harness.md` T2 (run protocol) and T6 (hit-rate reporter must plot
  hit rate vs budget multiples, not absolute RAM).
- **Effort:** backlog edits; the schema work is T2/T6 scope anyway — this just locks the field
  list before implementation.

## A3. Adopt the working-set resolver for `gap-closure` T2

- **What:** replace the buggy `--fit`-style heuristic with WASTE's step-down rule: refuse a
  budget below the floor; cache only in whole working-set multiples; largest multiple under 7/8
  of physical RAM.
- **Where:** `backlog/gap-closure.md` T2 (split/offload auto-recommender) and `wrapper-core.md`
  T3 (flag assembly maps the rule to each engine's dialect).
- **Effort:** backlog edit + a note that the rule is now externally validated (three independent
  engines converged on it).

## A4. Currency pass on the gaps research node

- **What:** `research/ktransformers-vs-llamacpp-moe-offload-gaps.md` is stale for the llama.cpp
  side: it predates `cached-experts-v2`, `--moe-hot-cache`, `/moe-layer-perf`, the 45.89%
  break-even, and FATE. Re-verify before finalizing T4/T5 scope (the ADR's dynamic-tracking
  amendment is the mechanism).
- **Where:** research node amendment + a scope note on `backlog/gap-closure.md` T4/T5.
- **Effort:** one re-research pass (hours), plus ticket-scope edit.

## A5. "Known-refuted" note in gap-closure so we don't regenerate dead ends

- **What:** record WASTE's measured refutations as a short pointer (per-expert bit alloc flat
  at 1.01×; cross-layer prefetch ≤ previous-token set; batching ceiling 1.63×; 3-bit trunk logit
  collapse) — enough that a future gap-research ticket doesn't rebuild them.
- **Where:** `backlog/gap-closure.md` or a research node footnote.
- **Effort:** trivial.

## A6. Future-backend research queue: WASTE + ds4 (v1.1+, warm)

- **What:** add WASTE and ds4 to the future-backend queue with triggers: revisit when (a) v1
  ships, or (b) either gains a Linux/Metal/Windows support matrix Chaos can target.
- **Where:** `backlog/mvp-v1.md` "future backends" note or a research queue node.
- **Effort:** trivial.

## A7. Decisions-convention: dated, append-only, refutations with numbers

- **What:** adopt LEARNED-style practice for the decisions/ directory — record refuted ideas with
  the number that killed them, keep them on file, date every entry.
- **Where:** a convention note in `decisions/` (or the graph's CONTRIBUTING/INDEX conventions).
- **Effort:** one short doc + habit.

## Suggested order of application (if approved)

1. A2 + A3 (schema + resolver) — they sharpen active backlog work first.
2. A4 (currency pass) — unblocks accurate T4/T5 scoping.
3. A5 + A6 + A7 (low-cost, prevent regressions).
4. A1 (correction) — whenever the graph is next touched.

## Explicit non-actions

- No INDEX.md edit, no backlog ticket edits, no code changes were made while writing this
  folder. This folder is standalone advisory.
- No recommendation to enter the engine-building lane, reposition to the Mac frontier, or build
  prefetch/batching/spec-decode features (see `04-strategy-options.md` §don't).
