# Deepseek V4 flash 0731 — advisory: WASTE and the MoE-on-consumer-hardware field

Author: Deepseek V4 flash (0731) — written as an advisory from me to Atur, for the Chaos project.
Date: 2026-08-01. Source of record: `github.com/sqliteai/waste` (README, `docs/LEARNED.md` §1–32,
`docs/EFFICIENCY.md`, `docs/FORMAT.md`, `docs/GATES.md`, `docs/BACKENDS.md`, `docs/K3.md`,
`docs/SERVE.md`, `CLAUDE.md`), plus web research on the surrounding field (ds4/DwarfStar,
llama.cpp MoE hot-cache, FATE/llama-moe-cache, ktransformers) and a full read of Chaos's
knowledge graph and current code (`chaos/*.py`, `tests/*.py`).

> This folder lives outside `docs/graph/` by Atur's instruction. Nothing in the graph or the
> backlog was modified. `05-actions.md` lists what *could* be folded into the graph later, for
> Atur to approve.

## One-paragraph verdict

WASTE is not a direct competitor to Chaos — it is an inference *engine* (it *is* the backend,
like llama.cpp or ktransformers), while Chaos is a *wrapper* over such engines. The real threat
is narrative and timing, not product overlap: in the space of one week WASTE became the
authoritative public source on **expert-cache economics** (cache floor = one token's working set;
above a whole multiple the OS pages and you go *8× slower*), which is exactly the territory
Chaos planned to make its own (gap-closure T4/T5 observability, benchmark-harness T6
cache-hit-rate-by-position). Meanwhile the field around the same idea is moving fast: ds4
(antirez) reached ~13.4k stars on SSD-streaming MoE a month before WASTE, and llama.cpp now has a
`--moe-hot-cache` / `/moe-layer-perf` feature (in a branch, not yet mainline) that looks like
Chaos's T4/T5/T6 for the llama.cpp side. The good news: **three independent engines converged on
the same physical model** (working-set floor, pressure-zone ceiling, ~80% RAM-cap), which
*validates* Chaos's measurement-methodology bets; and the **wrapper lane is still unclaimed** —
nobody does cross-engine probe → recommend → launch → bench with a pre-download speed prediction
and a benchmark schema keyed to working-set multiples. That is the defensible ground.

## What this folder is

| File | Scope |
|---|---|
| `01-waste-deep-dive.md` | The WASTE engine, format, memory model, quantization, portability, server — everything worth knowing, with the measured numbers. |
| `02-lessons-for-chaos.md` | Each transferable lesson mapped to a concrete Chaos ticket (gap-closure, benchmark-harness, hardware-profiler, wrapper-core). |
| `03-landscape-2026.md` | The competitive field: WASTE vs ds4 vs llama.cpp hot-cache vs FATE vs ktransformers; the convergence finding; the staleness warning on Chaos's gaps research. |
| `04-strategy-options.md` | Where Chaos stays differentiated; what not to chase; risk register. |
| `05-actions.md` | Proposed (not applied) graph/backlog actions, with acceptance hints. |

## The one-sentence takeaway

> Chaos should treat WASTE (and ds4) as *evidence that the cache-hit-rate and offload-tuning
> problem is real and worth solving*, as *input that sharpens Chaos's benchmark and split
> methodology*, and as *candidate future backends to wrap* — not as something to out-engine.
> The engine-writing lane is a different game, and Chaos's ADR already chose not to play it.

## Links into the graph (for cross-referencing)

- `docs/graph/research/benchmarking-methodology.md` — Chaos's planned metric + protocol; lesson 2–3 land here.
- `docs/graph/research/hardware-profiling.md` — the "SSD gates load-time only" claim that WASTE disproves (lesson 4).
- `docs/graph/research/ktransformers-vs-llamacpp-moe-offload-gaps.md` — now partially stale for llama.cpp (see `03-landscape-2026.md` §Currency).
- `docs/graph/decisions/fork-vs-wrapper.md` — the ADR; its observability bet is validated but being commoditized.
- `docs/graph/backlog/gap-closure.md` (T2 split tool, T4/T5 observability), `backlog/benchmark-harness.md` (T2 protocol, T6 hit-rate reporter), `backlog/hardware-profiler.md` (T3 SSD probe, T6 mixed-tier), `backlog/wrapper-core.md` (T2 version tracking), `backlog/mvp-v1.md` (pacing).
