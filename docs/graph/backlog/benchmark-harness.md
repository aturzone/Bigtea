---
epic: benchmark-harness
status: open
note: cache-hit-rate reporting (T6) ships static-placement-only until the dynamic swap-event upstream patch lands (../decisions/fork-vs-wrapper.md)
links: [../research/benchmarking-methodology.md, ../decisions/fork-vs-wrapper.md]
---
## Tickets
- [ ] T1: Implement a shared run-record schema defining pp tok/s, tg tok/s, TTFT, TPOT/ITL (p50/p90/p99), e2el, and MoE cache-hit-rate as distinct typed fields, never blended into one throughput number — depends: ../decisions/fork-vs-wrapper.md — evidence: ../research/benchmarking-methodology.md — acceptance: a schema validator accepts a sample run record containing all required fields and rejects one that merges pp and tg into a single tok/s field.
- [ ] T2: Implement a configurable run protocol with warmup enabled by default (opt-out flag), N repeats (default 5) reporting mean/stddev/CV, and an explicit cold-vs-warm cache-state label recorded per run — depends: ../decisions/fork-vs-wrapper.md — evidence: ../research/benchmarking-methodology.md — acceptance: two consecutive invocations with default settings each produce 5 repeat measurements and a non-empty cold/warm label in their output.
- [ ] T3: Implement standardized, seedable prompt/response-length generators (fixed random in/out, realistic-conversation-length, fixed-length reference set) selectable by name so separate harness runs compare on identical input distributions — depends: ../decisions/fork-vs-wrapper.md — evidence: ../research/benchmarking-methodology.md — acceptance: invoking the same generator name and seed twice produces byte-identical prompt sets.
- [ ] T4: Implement machine-readable per-run result output (JSON/JSONL) whose hardware-fingerprint section calls the hardware-profiler epic's RAM/VRAM/SSD probes rather than re-implementing probing — depends: ../decisions/fork-vs-wrapper.md, ../backlog/hardware-profiler.md — evidence: ../research/benchmarking-methodology.md — acceptance: a generated result row's fingerprint fields match the values independently returned by the hardware-profiler probes when run on the same machine.
- [ ] T5: Implement a post-run variance gate that flags any metric exceeding 5% CV across repeats as "noisy — investigate" instead of publishing it silently — depends: ../decisions/fork-vs-wrapper.md — evidence: ../research/benchmarking-methodology.md — acceptance: feeding the gate a synthetic repeat set above 5% CV returns a flagged status, and a set below 5% CV returns a clean status.
- [ ] T6: Implement a cache-hit-rate-by-token-position reporter that consumes the offloaded-expert hit/miss counters exposed by the observability ticket rather than instrumenting the engine itself — depends: ../decisions/fork-vs-wrapper.md, ../backlog/gap-closure.md — evidence: ../research/benchmarking-methodology.md — acceptance: given a logged hit/miss counter sequence across token positions, the reporter outputs a hit-rate curve binned by position rather than one aggregate number.

## Issues
T1 #15 · T2 #16 · T3 #17 · T4 #18 · T5 #19 · T6 #20
