---
topic: MoE landscape verification (Aug 2026) — fact-check advisory claims + currency pass on gaps node
status: resolved
links: [ktransformers-vs-llamacpp-moe-offload-gaps.md, waste-engine-verified.md]
---

Verifies claims in `docs/Deepseek V4 flash 0731/03-landscape-2026.md` (advisory by another model) +
currency pass on `research/ktransformers-vs-llamacpp-moe-offload-gaps.md`. All checks below done
2026-08-02 via GitHub REST API (`api.github.com`), repo pages, and HuggingFace model cards.

## Verification results (claim → verdict + URL)

**1. ds4 / DwarfStar (antirez)** — CONFIRMED, real name/URL: **DwarfStar (ds4)**,
https://github.com/antirez/ds4 by Salvatore Sanfilippo. Repo description: "DeepSeek 4 Flash and
PRO local inference engine for Metal, CUDA and ROCm." MIT license. Created 2026-05-06.
- Runs DeepSeek V4 Flash (primary target), DeepSeek V4 PRO (high-mem), GLM 5.2 — confirmed.
- On-disk KV cache persisting sessions across restarts — confirmed: `--kv-disk-dir` flag, README
  states it survives server restarts and re-uses cached prefixes for agent clients that resend
  full conversation history each request.
- Metal + CUDA + ROCm — confirmed (Metal primary for 96GB+ Macs, CUDA multi-GPU, ROCm for Strix
  Halo).
- 2-bit expert variant — confirmed: "very asymmetrical quantization," only routed MoE experts
  quantized to IQ2_XXS/Q2_K, rest of model untouched.
- Auto cache budget ~80% of recommended working set — confirmed verbatim: "takes 80% of the
  backend's recommended working set, subtracts non-routed weights, then applies" headroom.
- Star count: advisory says "~13.4k stars, June 2026." Live API check (2026-08-02):
  **19,881 stars**, repo `pushed_at` 2026-08-01. Not contradicted (plausible growth trajectory —
  a community issue on the repo cites "93 stars/day" current growth rate) but **stale**; current
  figure is ~20k, not 13.4k.

**2. llama.cpp `--moe-hot-cache` (leloch)** — PARTIALLY CONTRADICTED. Core facts check out, several
specific details do not.
- PR #24524 "cuda: MoE expert cache, adaptive VRAM caching of CPU-resident experts," author
  leloch — CONFIRMED closed (not merged), closed_at 2026-06-12.
  https://github.com/ggml-org/llama.cpp/pull/24524 — but "closed as too large" oversimplifies:
  GitHub/maintainer auto-closed it citing **three** reasons — AI-generated-content disclosure (PR
  body: "it is AI-generated/assisted"), mixed CPU+CUDA backend changes in one PR, and size (2,222
  lines/3 commits); maintainer suggested opening a discussion instead.
- #24528 — CONFIRMED open, but it is a **GitHub Discussion**, not an issue/formal RFC ticket
  (returns 404 on the issues API endpoint). Title: "RFC: MoE expert cache, VRAM caching of hot
  CPU-resident experts with hybrid hit/miss execution," author leloch.
  https://github.com/ggml-org/llama.cpp/discussions/24528
- Branch name **CONTRADICTED**: no `cached-experts-v2` branch exists. leloch's actual llama.cpp
  fork branches (checked via GitHub API): `master`, `moe-cache`, `moe-cache-pr`, `thp-alloc`,
  `v3-expert-cache`. https://github.com/leloch/llama.cpp
- `/moe-layer-perf` endpoint, `--moe-layer-perf-out` heatmap flag, Web UI — **NOT FOUND**. Absent
  from the PR body, the RFC discussion (searched in full), and the `v3-expert-cache` branch
  tree/README. No web search hit found either. Likely fabricated by the advisory's authoring
  model.
- "45.89% break-even hit rate" / "~70% practical ceiling" — **NOT FOUND**. Searched the RFC
  discussion thread in full; these numbers do not appear anywhere in it or in web search. The
  discussion's *real* cited numbers are different: "top 10% of experts take ~80% of hits"
  (measured on Qwen3.5-122B), "~99% simulated hit rate at 69% expert budget" (cited prior work),
  and throughput gains of +7%–+57% (PR) / +25% on GLM-5.1 754B (RFC). If Chaos needs a citable
  break-even/ceiling figure, use these real ones, not 45.89%/70%.
- Status: PR **closed, not merged**; Discussion/RFC **open** as of 2026-08-02.

**3. FATE / llama-moe-cache (ongunm)** — CONFIRMED, closely matches. Real repo:
https://github.com/ongunm/llama-moe-cache, description "Expert cache + predictive prefetch for
MoE inference in llama.cpp. A 12GB GPU can run a 120GB model at native speed." C++, 8 stars.
Created 2026-04-06, last pushed 2026-04-08 — matches "Apr 2026."
- Dual AGPL-3.0 / commercial license — confirmed in repo (AGPL-3.0 SPDX + README states
  "AGPL v3 for open source use, commercial license available").
- ~500 LoC — confirmed ("~500-line C++ extension," 5 files).
- GPU-resident expert cache + cross-layer/temporal predictive prefetch — confirmed.
- Hit rates — confirmed but with a **minor attribution mix-up** in the advisory: README reports
  99.50% hit rate on **Qwen3-30B-A3B** Q4_K_M (75,690 hits/384 misses) and 99.94% on
  **Mixtral-8x7B-Instruct** Q4_K_M (39,162 hits/24 misses), both on a 12GB RTX 4070 Ti. The
  advisory's "99.5%/99.94% hit rate on Mixtral Q4_K_M" implies both numbers are Mixtral's; only
  99.94% is. Both figures are real, just on two different models.

**4. Model landscape** — both real, see below for details.
- "DeepSeek V4 Flash" — CONFIRMED real released model.
- "Kimi K3" — CONFIRMED real open weights.

## Real model landscape (what actually exists as open weights, sizes, dates)

- **DeepSeek-V4-Flash** (DeepSeek AI) — MoE, sources vary slightly on exact param count: one
  summary states 284B total/13B activated, the model card itself (fetched directly) states "304
  billion parameters" total without giving an explicit activated-param figure — **not fully
  reconciled, flag as open question**. Hybrid attention (Compressed Sparse Attention + Heavily
  Compressed Attention per one source) for long-context efficiency; up to ~1M token context
  claimed. License: MIT (per model card, single-pass fetch, not independently cross-checked).
  Official release **DeepSeek-V4-Flash-0731** (2026-07-31) supersedes an earlier preview (DeepSeek
  V4 preview news dated 2026-04-24 at api-docs.deepseek.com). HF URLs:
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 ,
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash ,
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base ,
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark (speculative-decoding variant),
  https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4 , https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF ,
  collection: https://huggingface.co/collections/deepseek-ai/deepseek-v4 .
  ktransformers already ships **native support** for it (kt-kernel MXFP4 MoE operator) as of
  v0.6.2, 2026-05-03 — see currency corrections below.
- **Kimi K3** (Moonshot AI) — 2.8T total params, 104B activated per token (896 experts, 16
  selected/token), 1,048,576-token (1M) context, 93 layers (69 Kimi Delta Attention + 24 Gated
  MLA), MXFP4 weight / MXFP8 activation quantization (QAT), MoonViT-V2 vision encoder (401M
  params) for native multimodal input. Custom "Kimi K3 License" (not a standard permissive
  license, but allows download/deploy/fine-tune/build). Native MXFP4 download ~594GB. Released
  2026-07-26 (~1 day ahead of a planned 2026-07-27 target). Paper: "Kimi K3: Open Frontier
  Intelligence," https://huggingface.co/papers/2607.24653 . HF URL:
  https://huggingface.co/moonshotai/Kimi-K3 , GGUF: https://huggingface.co/unsloth/Kimi-K3-GGUF .

## Currency corrections to ktransformers-vs-llamacpp-moe-offload-gaps.md (what is now stale/wrong)

(Node not edited — corrections listed here per instructions.)

- **Wrong even at write time, not just stale** — ktransformers issue **#1074** (balance_serve
  loses/doesn't reuse KV cache between prompts) is cited as "unresolved at report time." Actual
  status: **closed 2025-04-09** (state_reason "completed"), i.e. it was closed **over a year**
  before the gaps node was written (2026-07-28). This citation was factually wrong when the node
  was written, not merely aged.
- Similarly stale/wrong: ktransformers issues **#1104** ("CMake exit code 1" install failure,
  closed 2025-04-09) and **#1022** (wheel build failure, closed 2025-04-03) are cited as current
  evidence that "source build frequently fails on consumer setups" — both were closed well over a
  year before the node's writing date. The broader claim (install UX still rough) is still
  separately supported by roadmap issue #1779 (confirmed still open, posted 2026-01-04, "improve
  installation experience for new users" still unchecked) — but these two specific issue
  citations are stale and should not be used as current evidence.
- Issue **#109** (Windows install.bat error) — closed 2025-12-11, ~7 months before the node was
  written; cited only as historical illustration in the node so lower-severity, but also stale.
- Issue **#1173** (Windows Vulkan binary request) — confirmed **still open**, no correction
  needed.
- All five llama.cpp `--fit` issues cited as "open correctness issues as of mid-2026" are actually
  **all closed**: #20308 (Windows overflow) closed 2026-03-10; #22592 (authoritative
  failure-handling) closed 2026-05-02; #20492 (Qwen3.5 fused-gate+up slowdown) closed 2026-03-24
  as "bug-unconfirmed"; #22442 (context-reduction-without-CPU-offload) closed 2026-04-27; #18390
  (per-device margin) closed 2026-01-08. All closed well before the gaps node's 2026-07-28 write
  date — this framing was inaccurate when written, not just aged. Caveat: "closed" on the tracker
  does not guarantee the underlying bug was fixed (e.g. #20492 closed as unconfirmed, not
  necessarily resolved) — would need PR-level verification if these specifics become
  ticket-relevant again.
- Issue **#20757** (no persistent GPU expert cache for CPU-offloaded experts) — the node calls it
  an "open feature request." Actual tracker state: **closed 2026-04-08**, state_reason
  "completed," closed by the original reporter. However: a commenter (Interpause) immediately
  questioned the closure as a possible glitch, and technical discussion continued actively for
  **three more months** (through 2026-07-06) with multiple people building independent
  prototypes/forks (JigSawPT, koren1712, kisasexypantera94) — strong evidence nothing actually
  shipped in mainline. The real locus of this work is now **PR #24524** (leloch, closed unmerged
  2026-06-12) and **Discussion #24528** (leloch, open) — see verification section above. Net: the
  underlying gap (no persistent GPU-resident expert cache in mainline llama.cpp) **still exists**
  as of 2026-08-02, but citing #20757 itself as "open" is wrong — it's closed, and the live
  artifact to track is #24528 + leloch's branches.
- **ktransformers MoE cache/expert observability**: no change — still nothing shipped. Releases
  through v0.6.4 (2026-07-23) are about fine-tuning throughput, not observability; roadmap issue
  #1779 (open) has no observability line item. Two adjacent-but-distinct open issues exist: **#2093**
  "Add co-activation-aware and online-EMA GPU expert placement" (placement heuristic, not
  observability/metrics) and **#2003** "Add MESH expert residency with io_uring direct I/O" (I/O
  throughput, not observability). Neither ships a hit-rate/metrics surface. The gaps node's "*
  ktransformers still has nothing public on MoE cache observability*" **remains accurate**.
- **New fact missing from the gaps node**: ktransformers **v0.6.2** (2026-05-03) added **native
  DeepSeek-V4-Flash support** via a kt-kernel MXFP4 MoE operator plus AVX2/AVX-VNNI RAWINT4
  backend expansion. The gaps node predates this and doesn't mention DeepSeek-V4-Flash
  compatibility at all — relevant given this whole advisory doc is about that model family.
  (Two open ktransformers issues, #2099 "OOM using ktransformers with Deepseek v4 flash" and
  #2035 "Deployment of deepseek_v4_flash with L20×8 failed" (closed), show early rough edges in
  that support.)
- llama.cpp Discussion #19197 (aggregated Prometheus metrics across router-mode backend
  instances) — confirmed still **open/unshipped**, no change from the gaps node's characterization;
  community has since built workarounds (service-discovery scraping, a public Grafana dashboard
  published 2026-07 by a community member) but nothing merged into llama.cpp itself.

## Competitive position for a wrapper

- The physical-law convergence the advisory flags (WASTE / ds4 / leloch's RFC all landing on
  similar cache-budget and break-even-hit-rate reasoning) is real in spirit but the llama.cpp-side
  specific numbers (45.89%/70%) the advisory cites don't exist in the primary source — Chaos's
  benchmark schema should still carry a `regime` field, just cite the RFC's real numbers (~80% of
  hits from top 10% of experts, +7–57%/+25% throughput gains) if referencing this work.
- **ktransformers observability (T4/T5) is still open ground** — confirmed nothing shipped, no
  roadmap commitment, and the two nearest-adjacent open issues (#2093, #2003) are about placement/
  I/O, not observability. This is the least contested part of Chaos's plan.
- **llama.cpp side is more contested than the original gaps node suggests, but still open**: a
  real, working (if unmerged) expert-cache implementation exists (leloch's PR/branches), with an
  active RFC discussion. It has not merged as of 2026-08-02 — mainline llama.cpp still has the
  #20757 gap in practice — but it is closer to shipping than a cold feature request. Chaos's
  ADR "track upstream dynamically" amendment is the correct posture: if `moe-cache`/`v3-expert-
  cache` merges before Chaos's relevant ticket lands, that ticket should become "expose upstream's
  new flag + hit-rate metric" rather than build one from scratch.
- **DwarfStar (ds4)** is the most credible adjacent project (~20k stars in 3 months) but occupies
  a different lane: a narrow, single-model-family engine (DeepSeek V4/GLM 5.2 only) on
  Mac-primary/CUDA/ROCm, not a general wrapper over ktransformers/llama.cpp. It doesn't directly
  compete with Chaos's wrapper-core scope, but its shipped UX (disk-persistent KV cache, 80%
  auto-budget resolver) is a concrete bar for what "good" cache-budget UX looks like, worth
  referencing when Chaos designs its own auto-tuning ticket.
- **FATE/llama-moe-cache (ongunm)** is a tiny (8-star), single-author, narrow (~500 LoC) extension
  — not a competitive threat, but a working proof that GPU-lane predictive prefetch can hit
  >99% in a favorable (small-model, ample-VRAM-headroom) case. Its AGPL/commercial dual license
  means Chaos should not vendor its code even for reference implementation without checking terms.
- **Net**: no project has shipped a general, cross-engine (ktransformers + llama.cpp),
  install+auto-tune+observability wrapper. Individual gaps are each being chipped at by narrow,
  single-purpose, mostly-unmerged/unshipped efforts (leloch's branches, ongunm's extension,
  ktransformers' placement/residency issues) rather than a unified product — which is still
  Chaos's opening. The gaps node's original open question ("no independent comparison of
  solo-user experience between the two projects") still stands.

## Open questions

- Whether leloch's Discussion #24528 (or a follow-up PR) merges into llama.cpp mainline, and on
  what timeline — directly determines whether Chaos's llama.cpp-side observability/cache ticket
  scope should shift to "wrap upstream's new flag" instead of building from scratch.
- Whether ktransformers' #2093 (EMA-based expert placement) or #2003 (MESH residency) ship, and
  whether either exposes any hit-rate/observability signal as a side effect.
- DeepSeek-V4-Flash's exact param count is unreconciled between sources (284B/13B activated vs.
  304B total, activated unstated) — not resolved here, low priority unless a ticket needs the
  exact figure.
- Whether ds4/DwarfStar broadens beyond Metal-primary + CUDA/ROCm (e.g. a Windows/Vulkan story) —
  no evidence found either way.
- The llama.cpp `--fit` issues are tracker-closed but not all confirmed fixed at the code level
  (e.g. #20492 closed as "bug-unconfirmed") — would need PR-level verification if `--fit`
  reliability becomes directly relevant to a ticket.
