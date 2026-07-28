# Bigtea graph index

One line per node: path — what it contains — what it links to.
This is the ONLY file read in full at session start. Open only the nodes a task links to.
Rule: any node change updates its line here, in the same commit.

## research/ — one file per topic (frontmatter: topic, status: open|resolved, links)
- research/ktransformers-vs-llamacpp-moe-offload-gaps.md — gaps for solo/small-team MoE offload (install UX, cache auto-tuning, monitoring, batching) in ktransformers, llama.cpp, and both — links: backlog/gap-closure.md

## decisions/ — one ADR per choice (context → options → choice → why; links to informing research)
(none yet)

## backlog/ — one file per epic; small tickets referencing the decision they depend on
- backlog/gap-closure.md — 7 tickets closing solo/small-team MoE-offload gaps (installer UX, auto-split tuning, per-expert/cache-hit observability, batching/serving reliability) — links: ../research/ktransformers-vs-llamacpp-moe-offload-gaps.md, ../decisions/fork-vs-wrapper.md

## Research queue — not yet nodes; researcher takes the top item, writes the node, moves the line up
1. hardware-profiling — benchmark real RAM/VRAM/SSD random-read speed; predict tokens/sec before any download
2. benchmarking-methodology — reproducible tokens/sec + TTFT + cache-hit-rate reporting
3. licensing — fork ktransformers (Apache-2.0) vs independent wrapper calling it as a backend
4. mvp-scope — realistic 4–6 week MVP for a small open-source team
