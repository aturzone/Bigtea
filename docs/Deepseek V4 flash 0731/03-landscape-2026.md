# 03 — Landscape 2026: WASTE, ds4, llama.cpp hot-cache, FATE, ktransformers

Researched via web search on 2026-08-01. Numbers from primary sources (repos, PRs/discussions,
author blog posts); star counts are point-in-time and drift.

## 1. The field in one table

| Project | What it is | License | Traction (≈2026-07/08) | Relevance to Chaos |
|---|---|---|---|---|
| **WASTE** (sqliteai) | Embeddable C11 MoE streaming engine; Kimi K3 2.78T on 64 GB | Apache-2.0 | Show HN 2026-07-29, ~375 stars in week 1, ~1k installs, 148k+ model downloads | The empirical reference for expert-cache economics; candidate future backend to wrap |
| **ds4 / DwarfStar** (antirez) | SSD-streaming MoE engine; DeepSeek V4 Flash-class + others; on-disk KV cache that persists sessions | open source | ~13.4k stars, June 2026 | The same physical bet, one month earlier, much louder; disk-KV persistence is an innovation WASTE doesn't claim |
| **llama.cpp `--moe-hot-cache`** (leloch fork/branch) | MoE expert cache + per-layer hit reporting in llama.cpp | MIT | PR #24524 closed "too large"; RFC #24528; branch `cached-experts-v2` | The exact T4/T5/T6 observability Chaos planned for the llama.cpp side — but branch-only, not mainline |
| **FATE / llama-moe-cache** (ongunm) | ~500 LoC llama.cpp extension: GPU-resident expert cache + temporal predictive prefetch | dual AGPL-3.0/commercial | Apr 2026 | Predictive prefetch with 99.5%/99.94% hit on Mixtral Q4_K_M (12 GB GPU) — the one place prefetch is shown to work (GPU lane) |
| **ktransformers** (kvcache-ai) | Flexible MoE offload engine, kt-kernel with expert placement | MIT | — | Chaos's primary wrapped engine; T4/T5 observability target. *Nothing* public here on MoE cache observability — still open ground |
| **SGLang** | Production serving (triton kernels, `--tp`) | Apache-2.0 | — | Chaos's server-mode backend; no consumer-grade offload story to speak of |

## 2. The convergence finding (the encouraging part)

Three independent engines arrived at the same physical model, with no evidence of cross-copying:

- **WASTE**: floor = one token's working set; cache only in whole working-set multiples; cap
  budget at ~7/8 of RAM (refuse below floor). Measured paging cliff: 8× slowdown at one multiple
  over.
- **ds4**: SSD-streaming with an automatic cache budget; default = **~80% of the backend's
  recommended working set** — the same 7/8-ish cap, phrased as a working-set fraction.
- **llama.cpp hot-cache** (RFC discussion): a **break-even hit rate of 45.89%** was derived for
  their GPU lane (compute is ~free once the experts are resident, so cache is only worth it past
  a modest threshold) and a practical ceiling around ~70%.

The meaning for Chaos: **the methodology bets in the ADR and benchmarking node are validated by
the market, not outflanked.** Three engines independently chose the same rule — that's a physical
law of MoE on consumer hardware, not a feature choice. Chaos's benchmark schema (working-set
multiples, regime, machine state) is built on the same law and will read correctly against any of
these engines.

## 3. The two regimes (must go into the benchmark schema)

| | GPU lane (llama.cpp hot-cache, FATE) | Disk lane (WASTE, ds4) |
|---|---|---|
| Where experts live | VRAM | NVMe/SSD |
| Cache economics | Compute is free once resident; break-even hit rate ≈45.9%; ~70% ceiling | Cache is existential; below one working set = zero hit; above a multiple = OS page faults |
| Best measured hit | 99.5% (FATE, prefetched) | 13–37% on K3 (streamed); 78% Kimi-Linear |
| Cliff behavior | Graceful (slower) | Catastrophic (8×) |

A benchmark report that does not record its regime is meaningless: a "46% hit rate" is *fine* on
the GPU lane and *terrible* on the disk lane. Chaos's benchmark-harness T2 must carry
`regime ∈ {gpu, disk, hybrid}` as a mandatory field — this was already latent in the schema; the
field now has a name.

## 4. The direct threats to Chaos's planned work

**T4/T5 (ktransformers observability) is the plan at risk.** WASTE's release week made
expert-cache hit-rate-by-position the most talked-about MoE number on the internet; llama.cpp has
a working `--moe-hot-cache` + `/moe-layer-perf` (per-layer hit counts, `--moe-layer-perf-out`
heatmap, Web UI) — in a **branch**, explicitly too large to merge in its first PR attempt
(#24524), with an open RFC (#24528). ktransformers still has *nothing* public.

Consequences for Chaos:
- The *concept* of MoE cache observability is no longer novel. The value has shifted from "having
  the idea" to "**having it mainline, for ktransformers, cross-engine, with an actionable
  recommend step**."
- Chaos's `ktransformers-vs-llamacpp-moe-offload-gaps.md` research node is **stale for the
  llama.cpp side**: it does not know about `cached-experts-v2`, `--moe-hot-cache`, the 45.89%
  break-even, or FATE. Before T4/T5 scope is finalized, that node needs a currency pass.
- The ADR's "track upstream dynamically" amendment is the right mechanism — this is exactly the
  moment it exists for.

## 5. What each engine does that Chaos should record (notes, not actions)

- **ds4**: on-disk KV cache persisting a session across restarts (the "resume a conversation as a
  file" idea); Metal + CUDA + ROCm; 2-bit experts for its q2 variant. Its recommended-working-set
  auto-budget is the closest public sibling to WASTE's resolver.
- **FATE**: the *one* place predictive prefetch is shown to pay — but on the GPU lane, where
  prefetch displaces nothing critical. This is consistent with WASTE's "29.0% ≤ 29.5%" refutation
  on the disk lane; the two are not contradictory, they are regime-specific.
- **WASTE server**: stdlib-only OpenAI-compatible server is the model for "embedding-grade"
  serving; its markup-vs-content tokenizer split is a security pattern worth stealing regardless
  of engine.

## 6. Currency warning (explicit, per the ADR amendment)

Re-verify before relying on *any* of this in ticket scope: llama.cpp `cached-experts-v2` may
merge or die; FATE may add a disk lane; ds4 may add WASTE-style container compat; ktransformers
may release observability. The staleness half-life here is weeks. The mitigation is Chaos's
existing dynamic-tracking rule; this document is a point-in-time snapshot dated 2026-08-01 and
should be re-read with that in mind.
