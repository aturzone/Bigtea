---
milestone: mvp-v1
status: open
links: [../research/mvp-scope.md, ../decisions/fork-vs-wrapper.md, wrapper-core.md]
---
## v1 milestone

Scope: DeepSeek-class MoE (MLA) on Linux+NVIDIA only, per ../research/mvp-scope.md.
Sequencing/rationale below is a plan over existing tickets — no new tickets here.

Pacing (re-paced 2026-07-28, solo Atur + Claude Code on the $20/mo plan): **one track at
a time, ~2-3 Claude-driven tickets per calendar week** — no parallel work; the original
4-6-week plan assumed team velocity. Re-check the real rate at the week-4 midpoint.
Sync-audit checkpoint (per CLAUDE.md) runs at phase boundaries marked ⏹, not per commit.

Re-ordered 2026-07-28 (**local-first**): no capable rig exists (local machine ruled out
for T0 — see issue #21 comments: 15.7GB RAM / 6GB VRAM vs ~19GB Q4 weights). T0 (#21) is
deferred to the hardware-dependent tail; gap-closure stays frozen per the ADR's T0 gate.
**v1 cannot tag without at least one session on a capable machine** (≥48-64GB RAM,
≥16GB VRAM, Ubuntu) for T0 + the end-to-end validation run.

## Week-by-week

- **Week 1**: wrapper-core T1 (#22) CLI skeleton + T2 (#23) version detection — hardware-independent.
- **Week 2**: hardware-profiler T1 (#8) RAM probe + T2 (#9) VRAM/GPU probe — developed and tested on the local laptop (multi-vendor acceptance only partially verifiable here; reviewer marks CANNOT VERIFY on the AMD/DXGI legs).
- **Week 3**: hardware-profiler T4 (#11) decode-ceiling model + benchmark-harness T1 (#15) run-record schema.
- **Week 4**: ⏹ midpoint audit + rate check (re-pace or trim if behind ~2-3 tickets/week) + **hardware re-check: if a capable machine exists by now, T0 (#21) preempts everything below**.
- **Week 5**: wrapper-core T3 (#24) launch-flag assembly — pure flag generation, testable without a live model.
- **Week 6**: benchmark-harness T2 (#16) run protocol + wrapper-core T5 (#26) packaging/README skeleton.
- **Weeks 7-8 (hardware-dependent tail — needs the capable machine)**: T0 (#21) first; then gap-closure T1 (#1) + T2 (#2); then wrapper-core T4 (#25) end-to-end run (probe -> recommend -> flags -> `sglang.launch_server`) printing a real tok/s; finalize README quickstart; ⏹ audit; tag v1. If no machine has appeared, v1 pauses here — everything above still lands on main via PRs.
  - **If T0 passes** ("recorder populated"): tail proceeds as written.
  - **If T0 comes back "bypassed"**: ADR revisit trigger fires (../decisions/fork-vs-wrapper.md). Drop gap-closure T4/T5 (#4/#5, observability) entirely; v1 narrows to wrapper core + auto-tuning + basic benchmarking; observability becomes a v2+ upstream-engagement thread. Nothing before the tail contained T4/T5, so no earlier work is wasted.

## "Launch" bar
pip install -> wrapper-core T2 (#23) version detect -> hardware-profiler T1/T2/T4 (#8/#9/#11) probes -> gap-closure T2 (#2) split recommendation -> wrapper-core T3 (#24) flags -> wrapper-core T4 (#25) launch + report tok/s.

## Out of v1
gap-closure T3/T6/T7 (#3/#6/#7); hardware-profiler T3/T5/T6/T7 (#10/#12/#13/#14); benchmark-harness T3/T4/T5/T6 (#17/#18/#19/#20) — full rationale in ../research/mvp-scope.md.
