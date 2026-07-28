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

## Week-by-week

- **Week 1**: gap-closure T0 (#21) (GATE) only — the rented rig is billing, so T0 is the sole focus; nothing else runs this week. ⏹ audit after T0 resolves.
  - **If T0 passes** ("recorder populated"): plan proceeds as below.
  - **If T0 comes back "bypassed"**: ADR revisit trigger fires (../decisions/fork-vs-wrapper.md). Drop gap-closure T4/T5 (#4/#5, observability) entirely; v1 narrows to wrapper core + auto-tuning + basic benchmarking; observability becomes a v2+ upstream-engagement thread. Weeks below are unchanged (none of them contained T4/T5).
- **Week 2**: wrapper-core T1 (#22) CLI skeleton + T2 (#23) version detection.
- **Week 3**: hardware-profiler T1 (#8) RAM probe + T2 (#9) VRAM/GPU probe.
- **Week 4**: gap-closure T1 (#1) preflight check; start gap-closure T2 (#2) auto-split tool on the probes. ⏹ midpoint: audit + compare actual vs ~2-3 tickets/week; re-pace or trim (benchmark-harness T2 (#16) is the first cut) if behind.
- **Week 5**: finish gap-closure T2 (#2); hardware-profiler T4 (#11) decode-ceiling model.
- **Week 6**: wrapper-core T3 (#24) launch-flag assembly; first end-to-end run via wrapper-core T4 (#25) (probe -> recommend -> flags -> `sglang.launch_server`).
- **Week 7**: benchmark-harness T1 (#15) schema + T2 (#16) run protocol, minimal — the end-to-end run prints a real tok/s number.
- **Week 8**: polish CLI UX/errors; wrapper-core T5 (#26) packaging + README quickstart; tag v1. ⏹ audit before release.

## "Launch" bar
pip install -> wrapper-core T2 (#23) version detect -> hardware-profiler T1/T2/T4 (#8/#9/#11) probes -> gap-closure T2 (#2) split recommendation -> wrapper-core T3 (#24) flags -> wrapper-core T4 (#25) launch + report tok/s.

## Out of v1
gap-closure T3/T6/T7 (#3/#6/#7); hardware-profiler T3/T5/T6/T7 (#10/#12/#13/#14); benchmark-harness T3/T4/T5/T6 (#17/#18/#19/#20) — full rationale in ../research/mvp-scope.md.
