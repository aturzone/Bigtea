# Bigtea graph index

One line per node: path — what it contains — what it links to.
This is the ONLY file read in full at session start. Open only the nodes a task links to.
Rule: any node change updates its line here, in the same commit.

## research/ — one file per topic (frontmatter: topic, status: open|resolved, links)
(none yet)

## decisions/ — one ADR per choice (context → options → choice → why; links to informing research)
(none yet)

## backlog/ — one file per epic; small tickets referencing the decision they depend on
(none yet)

## Research queue — not yet nodes; researcher takes the top item, writes the node, moves the line up
1. ktransformers-vs-llamacpp-moe-offload-gaps — current gaps for solo/small-team use: installer UX, cache auto-tuning, monitoring, batching
2. hardware-profiling — benchmark real RAM/VRAM/SSD random-read speed; predict tokens/sec before any download
3. benchmarking-methodology — reproducible tokens/sec + TTFT + cache-hit-rate reporting
4. licensing — fork ktransformers (Apache-2.0) vs independent wrapper calling it as a backend
5. mvp-scope — realistic 4–6 week MVP for a small open-source team
