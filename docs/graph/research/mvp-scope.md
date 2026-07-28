---
topic: realistic 4-6 week v1 scope for a small OSS team (post-ADR)
status: resolved
links: [../decisions/fork-vs-wrapper.md, hardware-profiling.md, benchmarking-methodology.md, ../backlog/wrapper-core.md, ../backlog/mvp-v1.md]
---
## Findings

### Assumptions (for Atur to correct)
- Team = 1-2 self-taught devs, part-time-to-full-time, i.e. roughly 80-240 total hours across the window — **unverified, flag for Atur**; every week-by-week estimate below scales linearly with this.
- "4-6 weeks" read as calendar weeks with slack, not 4-6 dedicated FTE-weeks; week 6 is treated as buffer, not new scope.
- T0 (gap-closure #21) and hardware-profiler T7 (#14, calibration) both require hands-on access to a real MoE-capable rig (multi-channel RAM, a GPU that runs kt-kernel/SGLang, a working ktransformers install) starting week 1 — if Atur doesn't have this yet, it is the actual week-1 blocker, ahead of any code task.
- Week-1 risk is entirely front-loaded on T0: it's a GATE (blocks gap-closure T1-T7 per gap-closure.md) and its outcome changes what v1 even is (see fail-path below) — nothing else in gap-closure should be scheduled ahead of it.
- Assumes DeepSeek-class MoE (MLA attention) is the reachable local test model in some quantized form — unverified against Atur's actual hardware budget; a smaller MoE (e.g. Qwen-MoE class) may be the practical dev-loop target even if DeepSeek-class is the eventual README headline (see open questions).

### v1 cut (in / out / new work no epic covers)
**NEW work — the wrapper core itself, not owned by any existing ticket:**
- CLI skeleton (subcommands, arg parsing) and pip packaging/entry point — no ticket in any epic builds "the wrapper" as a runnable program; all three epics assume a caller already exists.
- Upstream version detection (`pip show`/import-and-check ktransformers, kt-kernel, SGLang) implementing the ADR's version-tracking amendment ("drive the user's installed upstream dynamically, never vendor/pin") — this rule has no ticket; it must be built as v1 plumbing or the ADR's core mitigation for ollama-style staleness is unimplemented.
- Launch-flag assembly/orchestration: turning gap-closure T2's (#2) split recommendation plus hardware-profiler outputs into actual `--kt-expert-placement-strategy` / `--kt-num-gpu-experts` / `--kt-gpu-experts-ratio` flags and invoking `python -m sglang.launch_server` — T2's own acceptance criterion stops at "returns a split recommendation," not "launches anything." This glue is new.
- End-to-end one-command UX (probe -> recommend -> flags -> launch -> report tok/s) and the README quickstart itself.

**IN v1:**
- gap-closure T0 (#21) — GATE, must run and resolve first, week 1.
- gap-closure T1 (#1) — install/toolchain preflight — small, high-value, directly serves the quickstart ("why did my build fail" is the #1 new-user failure mode per the gaps research).
- gap-closure T2 (#2) — offload auto-split recommendation tool — the core value prop the wrapper exists to deliver.
- hardware-profiler T1 (#8) RAM bandwidth probe and T2 (#9) VRAM+GPU-string probe — both fast, standalone, and are T2's direct inputs.
- hardware-profiler T4 (#11) — decode tok/s ceiling model, VRAM-resident-only or single-tier case — gives the quickstart a "will this run at all" number without needing the harder mixed-tier math.
- benchmark-harness T1 (#15) run-record schema and T2 (#16) run protocol, minimal — needed so the quickstart can print one real tok/s figure, not just a recommendation.

**OUT of v1 (explicit, push to v1.1+):**
- gap-closure T3 (#3) batch/VRAM budget warning — safety net, not core path.
- gap-closure T4 (#4) per-expert activation metric and T5 (#5) cache-hit-rate metric — these are the strategic observability differentiator, but building full reporting on top of them is more work than fits alongside the wrapper core in 4-6 weeks, and their very existence in v1 is contingent on T0 (see fail path). Land observability as the v1.1 headline, not v1.
- gap-closure T6 (#6) concurrent-serving regression harness — large (N simulated users, threshold tuning), defer.
- gap-closure T7 (#7) NUMA wrapper — out of scope entirely, see guardrails.
- hardware-profiler T3 (#10) fio SSD probe, T5 (#12) prefill/TTFT estimator, T6 (#13) mixed-tier VRAM+RAM prediction model, T7 (#14) calibration harness — all defer. T6/T7 in particular are the highest-complexity math in the whole backlog and T7 needs logged benchmark runs that won't exist until after v1's benchmark-harness pieces have been run a few times (chicken-and-egg, naturally sequences to v1.1).
- benchmark-harness T3 (#17) seedable prompt generators, T4 (#18) hardware-fingerprint integration, T5 (#19) variance gate, T6 (#20) cache-hit-rate-by-position reporter — defer. T4 explicitly needs the hardware-profiler probes it fingerprints against; since v1 only ships RAM+VRAM probes (not SSD, T3), its fingerprint is necessarily partial until v1.1. T6 is doubly blocked: needs gap-closure T4/T5 (deferred, and possibly killed by a T0-fail) plus the ADR's note that cache-hit-rate reporting ships static-placement-only until an upstream dynamic-swap patch lands.

### Week-by-week sequencing (incl. T0-fails path)
- **Week 1**: Run T0 (gap-closure #21) end to end on the real rig — this is the single highest-priority task. In parallel (unblocked by T0): CLI skeleton, packaging scaffold, upstream version-detection module, hardware-profiler T1 (#8) and T2 (#9) probes — none of these touch the recorder question.
- **Week 2, if T0 passes** ("recorder populated"): proceed as planned — gap-closure T1 (#1) preflight checks; start T2 (#2) auto-split tool consuming week-1 probe outputs.
- **Week 2, if T0 comes back "bypassed"**: this is explicitly one of the ADR's revisit triggers ("runtime testing shows kt-kernel bypasses SGLang's ExpertDistributionRecorder" — decisions/fork-vs-wrapper.md Consequences). v1 narrows immediately: drop gap-closure T4/T5 (observability) from the roadmap entirely, not just defer — there is no externally-visible signal left to build a metric on for the dynamic path. v1 becomes wrapper core + auto-tuning (T2) + basic benchmarking only, with "observability" downgraded to a research/upstream-engagement thread (a dynamic-swap-log PR, per the licensing research's 13-33 day outside-account merge time) that is explicitly a v2+ bet, not part of this 4-6 week box.
- **Week 3**: gap-closure T1 (#1) done; gap-closure T2 (#2) built and wired to the week-1 probes; begin launch-flag assembly (new work, above).
- **Week 4**: finish launch-flag assembly + version detection integration; first true end-to-end run (probe -> recommend -> flags -> actual `sglang.launch_server` invocation) on the target model/platform. Start benchmark-harness T1 (#15) + T2 (#16) so the run produces a real tok/s number, not just "it started."
- **Week 5**: narrow to the one supported model class and one supported platform (see below); polish the CLI UX and error messages; write the README quickstart; fix bugs surfaced by real end-to-end runs.
- **Week 6 (buffer, if the range extends to 6)**: absorb T0/T2 slippage, install-preflight edge cases on real machines, tag the v1 release.
- **"Launch" = README quickstart that takes a user from `pip install` to a printed tok/s number** via: detect installed ktransformers/kt-kernel/SGLang version -> run hardware probes -> print a split recommendation -> generate and pass launch flags -> run one benchmark -> report the number. One supported model class and one supported platform, picked and justified below — not a matrix.
  - **Model class: DeepSeek-V3/R1-class MoE (MLA attention)** — this is the model class kt-kernel's offload machinery and existing tutorials (injection_tutorial, experts-sched-tutorial) are built and documented around, and it's the case hardware-profiler T4's architecture-parameterized KV term (MHA/GQA/MLA, not a flat heuristic) exists specifically to handle — picking anything else means validating the wrapper against a path upstream itself treats as secondary.
  - **Platform: Linux + NVIDIA** — this is the combination ktransformers' own Q2 2026 roadmap targets ("best MoE inference performance on consumer-grade x86 + NVIDIA GPU," per ../research/licensing-fork-vs-wrapper.md); Windows-native and AMD are (c)-class gaps in the wrapper-feasibility mapping (same research doc) that only a fork could close, so a wrapper-only v1 has no realistic route to them anyway.
  - Sanity check against comparable OSS v1s: ollama's original 2023 launch pitch was "Docker for LLMs" / "Run LLMs on your Mac" — a thin single-command wrapper around llama.cpp, single platform at launch, no multi-backend or tuning ambitions (HN "Show HN: Ollama," news.ycombinator.com/item?id=36802582). RamaLama's initial Nov-2024 scope (developers.redhat.com/articles/2024/11/22/how-ramalama-makes-working-ai-models-boring) was similarly narrow: detect GPU, pull one matching container, run — no cross-backend tuning or observability at launch either. Both support treating "one model class, one platform, one command" as a realistic v1 bar rather than under-scoping.

### Out-of-scope guardrails
- NUMA-aware launch wrapper (gap-closure T7, #7) — dual-socket tuning is a deep rabbit hole two people cannot absorb alongside the rest of v1; not even attempted.
- AMD support and Windows-native paths — structurally a fork-only gap per the wrapper-feasibility mapping (../research/licensing-fork-vs-wrapper.md, "(c), unavoidably"); a 2-person wrapper team has nothing to inject here and should not try.
- Any auto-tuning/config-recommendation work beyond deterministic split math (gap-closure T2, hardware-profiler T4) that starts competing with ktransformers' own Q2 2026 roadmap item ("establish recommended configurations for typical consumer hardware setups," @ovowei's offload-scheduling work) — the ADR flags this collision explicitly (decisions/fork-vs-wrapper.md, hardware-profiler.md epic note); v1 must stay simple/deterministic here or deconflict with upstream first, not build a rival smart recommender.
- Any upstream source patch or PR in v1 — the ADR's hybrid patches (dynamic-swap-event log, install robustness fixes) are real but outside-account PRs run 13-33 days to merge per the licensing research; that alone doesn't fit a 4-6 week box. v1 stays wrapper-only, driving upstream via its existing stable CLI/API surface.
- Full concurrent-serving regression harness (gap-closure T6, #6) with N simulated users — scope creep, defer.
- Mixed-tier (VRAM+RAM+SSD) offload prediction and its calibration harness (hardware-profiler T6/T7, #13/#14) — highest-complexity math in the backlog, and T7 has no calibration data to run against until v1's own benchmark runs accumulate.
- Vendoring or pinning a specific ktransformers/SGLang commit "to stabilize the demo" under schedule pressure — this directly contradicts the ADR's version-tracking amendment and reintroduces the exact ollama-staleness failure mode the ADR was written to avoid.

## Open questions
- Real hardware access timeline: does Atur already have a rig that can run T0 and the hardware probes in week 1, or does procuring/renting one push the whole schedule right? Not resolved here.
- Whether DeepSeek-V3/R1-class is actually runnable on Atur's own hardware for dev-loop iteration, or whether a smaller MoE model should be the working target with DeepSeek-class reserved for the README's headline number — not resolved, depends on the hardware answer above.
- Whether a T0 "bypassed" result should be read as a hard, permanent cut of the observability epic or just a deferral pending the dynamic-swap-log upstream PR — the ADR frames it as a revisit trigger, not an automatic kill, so this is a judgment call for Atur once/if T0 fails.
- Actual weekly hours Atur can commit — flagged above as an assumption, not verified; changes every week-by-week estimate proportionally.
