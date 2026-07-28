---
decision: fork ktransformers vs independent wrapper
status: proposed        # Atur must sanity-check before this becomes accepted
links: [../research/licensing-fork-vs-wrapper.md, ../research/ktransformers-vs-llamacpp-moe-offload-gaps.md]
---
## Context
- Nearly every Bigtea gap is externally reachable: auto-tuning and per-expert observability are (a) — documented SGLang/kt-kernel CLI flags and `optimize_rule` YAML, zero source changes; only dynamic-update cache-hit-rate is partial (b) (../research/licensing-fork-vs-wrapper.md).
- Two of three epics (auto-tuning, hardware-profiling/config-recommendation) collide with ktransformers' own Q2 2026 roadmap, assigned to core maintainers; observability appears on no roadmap — uncontested (../research/licensing-fork-vs-wrapper.md).
- No ktransformers fork or wrapper has ever gained traction (largest fork: 45 stars vs 19k); llama.cpp precedent shows wrappers hit staleness ceilings (ollama, ~56% perf gap from a pinned commit) and forks hit parity lag (koboldcpp MTP breakage) and bus-factor risk (ik_llama.cpp) (../research/licensing-fork-vs-wrapper.md).
- Small, well-scoped PRs merge in ~1 day median; maintainers explicitly solicit contributors, and themselves run the fork-a-dependency-and-sync pattern on sglang/flashinfer (../research/licensing-fork-vs-wrapper.md).
- Fork carries Apache-2.0 §4 ongoing obligations (mark modified files, re-check NOTICE each sync) plus a trademark rename; a pure wrapper triggers essentially none (../research/licensing-fork-vs-wrapper.md).

## Options
### A. Fork ktransformers
- Wins: full control over kernels, install scripts, and serving path; only route to Windows-native/AMD platform gaps (../research/ktransformers-vs-llamacpp-moe-offload-gaps.md).
- Costs: permanent parity-lag treadmill against an active upstream (koboldcpp precedent) and solo bus-factor (ik_llama.cpp precedent) (../research/licensing-fork-vs-wrapper.md).
- Costs: zero successful ktransformers-fork precedent; §4 change-marking on every sync; must rebrand (../research/licensing-fork-vs-wrapper.md).
- Costs: directly competes with the core team's own Q2 roadmap on two of three epics (../research/licensing-fork-vs-wrapper.md).

### B. Pure independent wrapper
- Wins: lightest licensing burden; covers auto-tuning, static-placement observability, install preflight, and benchmarking via documented flags alone (../research/licensing-fork-vs-wrapper.md).
- Wins: serving reliability now rides SGLang's mature stack, not archived balance_serve code (../research/licensing-fork-vs-wrapper.md).
- Costs: cannot close the dynamic-expert-update cache-hit-rate gap (no swap-event log exists) or root-fix upstream build fragility (../research/licensing-fork-vs-wrapper.md).
- Costs: ollama-style ceiling risk if upstream flags/behavior drift and the wrapper can only work around, never fix (../research/licensing-fork-vs-wrapper.md).

### C. Wrapper + small upstream patches (hybrid)
- Wins: everything B wins, plus upstream PRs for the two real (b)/(c) items — a dynamic-swap event log and CMake/install robustness fixes — the exact PR class that merges in <1 day (../research/licensing-fork-vs-wrapper.md).
- Wins: opt-in instrumentation hooks have in-tree precedent (`get_profile_stats` env-var-gated profiler), and maintainers actively onboard outsiders (../research/licensing-fork-vs-wrapper.md).
- Costs: outside-account PRs run 13–33 days until trust builds; requires engaging the WeChat-mediated inner ring (../research/licensing-fork-vs-wrapper.md).
- Costs: still exposed to roadmap collision on auto-tuning/profiling — must deconflict or concentrate on observability (../research/licensing-fork-vs-wrapper.md).

## Recommendation
**Option C: wrapper + small upstream patches.** The feasibility mapping makes forking unnecessary — every gap Bigtea targets is (a) via documented CLI flags except one (b) item and platform-native work Bigtea shouldn't attempt solo. The precedent record makes forking actively bad: no ktransformers fork has ever worked, and even successful llama.cpp forks pay parity lag and bus-factor costs a small team can't absorb. Pure wrapping (B) fails only at the margins — the dynamic-update cache-hit-rate log and install root-fixes — and those margins are precisely where the evidence is strongest for tiny upstream PRs: ~1-day median merges, in-tree precedent for opt-in instrumentation, maintainers who run this exact hybrid pattern on their own dependencies. Roadmap collision then dictates where to aim: concentrate Bigtea's differentiating effort on observability (on no upstream roadmap), treat auto-tuning/profiling as consumers of upstream flags rather than rivals to @ovowei's assigned work. Licensing is one-sided: hybrid keeps the wrapper's near-zero burden since patches are contributed upstream, not redistributed.

Strongest counterargument: the ollama staleness ceiling — wrappers eventually fall behind and get squeezed when upstream automates their value-add (llama.cpp `--fit` precedent). It loses because ollama's failure was self-inflicted (vendoring a pinned commit); a wrapper driving a user-installed upstream via documented flags tracks head by construction, and the squeeze risk is already answered by aiming at observability, which upstream has left off two consecutive roadmaps.

## Consequences
- **backlog/gap-closure.md**: proceeds as wrapper-side tooling plus 2 small upstream PRs (swap-event log, install fixes). **Re-scope T6**: it was written against balance_serve, which is now archived — serving-reliability work must target the SGLang integration path instead (../research/licensing-fork-vs-wrapper.md).
- **backlog/hardware-profiler.md**: proceeds standalone (preflight probe + tok/s prediction don't need ktransformers source), but its config-recommendation output overlaps a Q2 roadmap item — deconflict with upstream before building a competing recommender.
- **backlog/benchmark-harness.md**: proceeds unchanged; the cache-hit-rate reporting ticket ships static-placement-only until the dynamic swap-event upstream patch lands.
- Revisit this ADR if: small scoped PRs stop merging fast (the ~1-day dynamic breaks); an upstream roadmap adds observability/monitoring; runtime testing shows kt-kernel bypasses SGLang's ExpertDistributionRecorder (open question in the research node); or the `kt` CLI matures to subsume the wrapper's auto-tuning value entirely.
