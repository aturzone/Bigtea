# Bigtea graph index

One line per node: path — what it contains — what it links to.
This is the ONLY file read in full at session start. Open only the nodes a task links to.
Rule: any node change updates its line here, in the same commit.

## research/ — one file per topic (frontmatter: topic, status: open|resolved, links)
- research/ktransformers-vs-llamacpp-moe-offload-gaps.md — gaps for solo/small-team MoE offload (install UX, cache auto-tuning, monitoring, batching) in ktransformers, llama.cpp, and both — links: backlog/gap-closure.md
- research/hardware-profiling.md — cross-platform RAM/VRAM/SSD probing (STREAM, fio, nvidia-smi/DXGI) + MoE decode-speed roofline math + prior-art tokens/sec calculators (none model VRAM+RAM+SSD offload split) — links: ktransformers-vs-llamacpp-moe-offload-gaps.md, backlog/hardware-profiler.md
- research/benchmarking-methodology.md — reproducible tok/s+TTFT+MoE-expert-cache-hit-rate methodology: llama-bench/vLLM/MLPerf/LocalScore prior art, metric definitions, warmup/repeat/thermal-steady-state protocol, hardware-fingerprint schema, 5% CV reproducibility threshold — links: hardware-profiling.md, ktransformers-vs-llamacpp-moe-offload-gaps.md, backlog/benchmark-harness.md
- research/licensing-fork-vs-wrapper.md — fork ktransformers (Apache-2.0) vs independent wrapper: PR/issue contribution dynamics, per-gap wrapper feasibility (auto-tuning + per-expert observability both externally achievable via documented SGLang/kt-kernel CLI flags; cache-hit-rate under dynamic updates partial), llama.cpp wrap/fork precedent (ollama/koboldcpp/ik_llama.cpp), Apache-2.0 vs MIT mechanics, hybrid-option viability — links: ktransformers-vs-llamacpp-moe-offload-gaps.md
- research/mvp-scope.md — v1 scope for **solo Atur + Claude Code ($20/mo budget-capped, ~2-3 tickets/wk → ~8 calendar weeks)**: T0 gates week 1 with a fail-path that drops observability entirely; v1 cut = wrapper core (new: CLI, version-detection, launch-flag glue) + gap-closure T0/T1/T2 + hardware-profiler T1/T2/T4 + benchmark-harness T1/T2, one model class (DeepSeek-class MoE) on one platform (Linux+NVIDIA); T0 itself runs the smallest supported MoE — links: ../decisions/fork-vs-wrapper.md, hardware-profiling.md, benchmarking-methodology.md, backlog/wrapper-core.md, backlog/mvp-v1.md

## decisions/ — one ADR per choice (context → options → choice → why; links to informing research)
- decisions/fork-vs-wrapper.md — **accepted 2026-07-28**: wrapper + small upstream patches (hybrid), never fork; amendments: T0 recorder-verification gate (#21) blocks gap-closure, wrapper tracks installed upstream dynamically (never vendor/pin) — links: ../research/licensing-fork-vs-wrapper.md, ../research/ktransformers-vs-llamacpp-moe-offload-gaps.md

## backlog/ — one file per epic; small tickets referencing the decision they depend on
- backlog/gap-closure.md — 8 tickets (T0 recorder-verification GATE blocks the rest; T6 re-scoped to SGLang path) closing solo/small-team MoE-offload gaps (installer UX, auto-split tuning, per-expert/cache-hit observability, serving reliability) — links: ../research/ktransformers-vs-llamacpp-moe-offload-gaps.md, ../decisions/fork-vs-wrapper.md
- backlog/hardware-profiler.md — 7 tickets for a standalone preflight hardware-probe (RAM/VRAM/SSD) + tokens/sec prediction model (decode/prefill/mixed-tier) + calibration harness — links: ../research/hardware-profiling.md, ../decisions/fork-vs-wrapper.md
- backlog/benchmark-harness.md — 6 tickets for a reproducible benchmark/reporting harness (metric schema, run protocol, prompt standardization, result format+hardware fingerprint, variance gate, cache-hit-rate reporting) — links: ../research/benchmarking-methodology.md, ../decisions/fork-vs-wrapper.md
- backlog/wrapper-core.md — 5 tickets for the wrapper product itself (CLI skeleton+config, upstream version detection, launch-flag generation for DeepSeek-class MoE on Linux+NVIDIA, end-to-end orchestration, pip packaging+quickstart README) — links: ../research/mvp-scope.md, ../decisions/fork-vs-wrapper.md
- backlog/mvp-v1.md — v1 milestone plan (no new tickets), re-paced for solo+Claude Code and re-ordered 2026-07-28 **local-first** (no capable rig: T0 + gap-closure + end-to-end deferred to a hardware-dependent weeks-7-8 tail; weeks 1-6 are laptop-buildable tickets; midpoint hardware re-check at week 4), incl. T0-fails fallback — links: ../research/mvp-scope.md, ../decisions/fork-vs-wrapper.md, wrapper-core.md

## Research queue — not yet nodes; researcher takes the top item, writes the node, moves the line up
(empty — initial queue complete)
