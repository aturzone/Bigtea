---
milestone: mvp-v1
status: open
links: [../research/mvp-scope.md, ../decisions/fork-vs-wrapper.md, wrapper-core.md]
---
## v1 milestone

Scope: DeepSeek-class MoE (MLA) on Linux+NVIDIA only, per ../research/mvp-scope.md.
Sequencing/rationale below is a plan over existing tickets — no new tickets here.

## Week-by-week

- **Week 1**: gap-closure T0 (#21) (GATE) run end-to-end on real rig — top priority, blocks gap-closure T1-T7. In parallel: wrapper-core T1 (#22) CLI skeleton, wrapper-core T2 (#23) version detection, wrapper-core T5 (#26) packaging scaffold, hardware-profiler T1 (#8) + T2 (#9) probes.
- **Week 2, if T0 (#21) passes** ("recorder populated"): gap-closure T1 (#1) preflight checks; start gap-closure T2 (#2) auto-split tool on week-1 probe outputs.
- **Week 2, if T0 (#21) comes back "bypassed"**: ADR revisit trigger fires (../decisions/fork-vs-wrapper.md). Drop gap-closure T4/T5 (#4/#5, observability) entirely — no externally-visible signal to build a metric on. v1 narrows to wrapper core + gap-closure T2 (#2) auto-tuning + basic benchmarking; observability becomes a v2+ upstream-engagement thread.
- **Week 3**: gap-closure T1 (#1) done; gap-closure T2 (#2) wired to week-1 probes; begin wrapper-core T3 (#24) launch-flag assembly.
- **Week 4**: finish wrapper-core T3 (#24); first true end-to-end run via wrapper-core T4 (#25) (probe -> recommend -> flags -> `sglang.launch_server`). Start benchmark-harness T1 (#15) + T2 (#16) for a real tok/s number.
- **Week 5**: polish CLI UX/error messages; write wrapper-core T5 (#26) README quickstart; fix bugs from real end-to-end runs.
- **Week 6 (buffer)**: absorb gap-closure T0 (#21)/T2 (#2) slippage and install-preflight edge cases; tag the v1 release.

## "Launch" bar
pip install -> wrapper-core T2 (#23) version detect -> hardware-profiler T1/T2/T4 (#8/#9/#11) probes -> gap-closure T2 (#2) split recommendation -> wrapper-core T3 (#24) flags -> wrapper-core T4 (#25) launch + report tok/s.

## Out of v1
gap-closure T3/T6/T7 (#3/#6/#7); hardware-profiler T3/T5/T6/T7 (#10/#12/#13/#14); benchmark-harness T3/T4/T5/T6 (#17/#18/#19/#20) — full rationale in ../research/mvp-scope.md.
