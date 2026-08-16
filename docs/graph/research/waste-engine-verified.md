---
topic: Verify sqliteai/waste (WASTE engine) claims — fact-check advisory
status: resolved
links: [ktransformers-vs-llamacpp-moe-offload-gaps.md]
---

Verifies claims made in `docs/Deepseek V4 flash 0731/` (advisory by another model, 2026-08-01).
Primary sources: github.com/sqliteai/waste (repo created 2026-07-28, C, Apache-2.0, 796 stars /
77 forks / 2 open issues as of 2026-08-02 read), raw docs (README.md, CLAUDE.md, docs/LEARNED.md,
docs/EFFICIENCY.md, docs/K3.md, docs/TECHNICAL.md, docs/ENGINE.md, docs/FORMAT.md,
docs/BACKENDS.md, docs/KDA.md), GitHub issues, Hacker News item 49098966. NOTE: repo is under
very active multi-times-per-day development (LEARNED.md entries dated 2026-07-29 through
2026-08-01, shipping perf changes daily) — several specific benchmark numbers in the advisory
are already stale relative to the current docs, not because the advisory fabricated them but
because the project moved. This is itself a finding (see below).

## Verification results (claim → CONFIRMED/CONTRADICTED/NOT FOUND, with source)

1. **"Embeddable C11 MoE streaming engine, ~6000 LoC in src/, zero 3rd-party runtime deps,
   Apache-2.0"** — C11/embeddable/MoE: CONFIRMED (CLAUDE.md: "WASTE is an embeddable MoE
   inference engine in C11"). Zero runtime deps: CONFIRMED ("no BLAS, Python, CUDA, or other
   external dependency for the current CPU inference path" — README). Apache-2.0: CONFIRMED
   (GitHub API `license` field + README: "distributed under the permissive Apache 2.0 license").
   **~6000 LoC: NOT FOUND** — no line-count claim in README/CLAUDE.md; repo total size is 3,628 KB
   (includes docs/tests/tools, not a src/-only LoC figure); unverifiable without running `cloc`
   directly on the tree.

2. **Kimi K3 exists / 982 GiB container / 0.49–0.54 tok/s / 29.05 GiB min RAM** — **Kimi K3 IS
   REAL**, CONFIRMED independently: Moonshot AI released it 2026-07-26, 2.8T params, MoE with 896
   experts / top-16 routed (Fast Company, interconnects.ai, HuggingFace blog "kimi-k3-model-
   overview-mxfp4-quantization", qz.com). Container 982 GB: CONFIRMED (README table). Min RAM:
   CONFIRMED within rounding — current README/docs/TECHNICAL.md say **29.06 GB @ 4K**, advisory
   says 29.05 GiB (same figure, negligible rounding). Decode speed: **CONFIRMED but stale** —
   current live README states **0.45–0.62 tok/s** (wider band, centered higher) vs advisory's
   0.49–0.54; other repo docs show 0.32–0.34 (older writeup), 0.53–0.55 (mlock section, "quiet
   machine"), 0.56–0.63 (current TECHNICAL.md paging table) — the number moved upward across the
   week from shipped optimizations (see LEARNED.md §35: router-lookahead shipped 2026-08-01,
   1.17× throughput).

3. **Kimi-Linear 48B: 19 GiB container, 1.87 GiB min RAM, 10.7 tok/s** — Container 19 GB:
   CONFIRMED. Min RAM: **CONTRADICTED** — current README says **1.28 GB**, not 1.87 GiB (real
   number is ~32% lower than claimed; likely the advisory captured an earlier value before a
   later optimization reduced the floor, i.e. drift not fabrication). Decode: CONFIRMED within
   rounding (README: **10.65 tok/s** vs advisory's 10.7).

4. **Cache-floor finding (0% below floor, 2604/2704 evictions; K3 = 16 experts × 92 layers =
   17.0–17.4 GB/token)** — **CONFIRMED, exact.** docs/LEARNED.md §4 verbatim: "the hit rate is
   *exactly zero* — 2604 evictions in 2704 accesses... **The cache floor is one token's working
   set.** For K3 that is 16 x 92 x 11.8 MB = **17.4 GB**." docs/K3.md confirms 93 total layers, 92
   are MoE (layer 0 dense), 896 total/16-routed experts, per-token cold I/O "17.0 GB" — the
   advisory's 17.0–17.4 range spans both cited figures (cold-I/O vs working-set calc) correctly.

5. **Paging-cliff sweep table (32/46/52/58 GB) + "freeing 1.11 GB → 8× slower"** —
   **CONTRADICTED as currently published.** Advisory table: 32GB→0%→0.31; 46GB→13%→0.32;
   52GB→27%→0.11–0.14; 58GB→37%→0.04 tok/s. Current docs/TECHNICAL.md table (fetched 2026-08-02):
   `32 GB→3.32GB cache→29.1% hit→0.56–0.58 tok/s`; `46 GB→17.32GB→36.2%→0.63 tok/s`;
   `52 GB→23.32GB→38.4%→0.07–0.09 tok/s`; `58 GB→29.32GB→41.3%→0.07–0.08 tok/s`. Same *shape*
   (sharp cliff above the sweet spot: ~0.6 tok/s collapses to ~0.07 tok/s) but every absolute
   number differs — consistent with the project's daily-shipping pace, not with the advisory
   inventing figures. The "freeing memory → slower" mechanism is corroborated in text
   ("the freed gigabyte was spent by the budget resolver on more cache, which was the one place
   it made things worse" — EFFICIENCY.md/LEARNED.md), but the specific 0.32→0.04 (8×) pairing
   could not be reproduced against the current table (current 58GB row is 0.07–0.08, not 0.04) —
   **the 8× multiplier looks stale**, the phenomenon it describes is real.

6. **Budget resolver (refuse below floor; whole multiples; largest under 7/8 physical RAM)** —
   Refuse-below-floor: **CONFIRMED verbatim**, docs/ENGINE.md: "A budget under the floor fails at
   open with `WASTE_E_RAM_BUDGET` and a pointer to `waste plan`." Whole-multiple stepping:
   **CONFIRMED verbatim**: "the default steps down a whole working set at a time and takes the
   largest that fits: `floor + 3x`, else `2x`, else `1x`, else the floor." **"7/8 of physical
   RAM": NOT FOUND / likely wrong** — the same doc section states the cap is "available system
   memory, not a fixed fraction like 7/8" per the fetch; no "7/8" fraction located anywhere
   reachable in the docs.

7. **VQ: 3 stages × 256-entry codebooks / 8-dim vectors, 3.00 bits/weight; never-dequantize
   2.15→0.22 s/token** — 3-stage residual VQ: CONFIRMED ("error by stage count is 57.5% / 33.2% /
   19.5%" — decreasing error per added stage). **256-entry codebook / 8-dim vector shape: NOT
   FOUND** in any doc section reachable (EFFICIENCY.md, ENGINE.md, K3.md, TECHNICAL.md all
   searched). Bits/weight: CONFIRMED within rounding — K3.md gives **3.01** bits/weight for the
   shipped expert record (matches "3.00"). **2.15→0.22 s/token fused-matvec timing: NOT FOUND**
   verbatim anywhere searched — the underlying LUT mechanism it describes is real and shows up in
   the time-breakdown table (LUT apply 23.9%, LUT build 2.7%, see #8) but this specific before/
   after pair could not be located.

8. **Time breakdown: expert I/O 54.8%, expert matmul 27.2%, KDA 9.3%** — **CONFIRMED EXACTLY.**
   docs/EFFICIENCY.md §4E table verbatim: expert I/O 9.95s/**54.8%**; expert matmul 4.94s/
   **27.2%** (of which LUT apply 4.34s/23.9%); KDA 1.69s/**9.3%**; LUT build 0.48s/2.7%. Perfect
   match, only claim verified to the decimal with zero drift.

9. **Absorbed MLA KV latent: 53× less cache, logits identical to 1.2e-05** — Mechanism
   (kv_b_proj absorbed into q_absorb/out_absorb) CONFIRMED to exist (HN/GitHub discussion text;
   K3.md architecture lists `kv_lora 512`). **53× multiplier and 1.2e-05 logit figure: NOT
   independently confirmed** — a nearby logit-diff figure of **1.14e-05** turned up in one search
   (same order of magnitude, possibly the same number reported with different rounding, but not
   pinned to this specific absorbed-MLA claim with certainty) and cache-size figures found
   (17.4/15.6/22.0 GB) did not clearly match the advisory's cited 4K/128K pair (11.25→0.21 GB,
   360→6.75 GB). Mark PARTIAL: real feature, precise numbers unverified.

10. **Refuted-levers table** — item by item:
    - 2-bit experts "34% vs 19.4% at 3-bit": 19.4% at 3-bit **CONFIRMED exactly** ("vq3 | 3.01 |
      19.4%"). 34% at 2-bit **NOT CONFIRMED** — the 2-bit number actually found is much worse:
      "rtn2-g64 | 2.25 | **71.8%**" (a naive RTN baseline). Possibly a different, better 2-bit VQ
      variant scoring 34% exists and wasn't surfaced by search — unclear, flag as unresolved
      rather than contradicted outright.
    - 3-bit trunk "1.4× slower, 36% logit error, generation collapse": **CONFIRMED** — "Q3G trunk
      | 21.13 GB | 23.48 GB | 29% | 0.16 | `+` and spaces" and "logits land 36% off... generation
      collapses."
    - GEMQ per-expert allocation "flat 1.01× across layers": **CONFIRMED verbatim** ("between
      layers — 1, 5, 23, 46, 69, 92 | 1.01x").
    - Cross-layer prefetch "29.0% vs 29.5%": **CONFIRMED as of 2026-07-31 (§29) but SUPERSEDED
      2026-08-01.** This is the single most important correction: LEARNED.md §34 ("§29 refuted
      the wrong predictor", 2026-08-01) shows a *different* method — running next-layer's router
      weights on the current layer's hidden state — hit **59.0% recall@16**, was judged viable,
      and §35 (same day) says it **shipped on the decode path for a 1.17× throughput gain**. The
      advisory's refuted-levers table presents cross-layer prefetch as a dead end; as of the
      doc's own most recent entries it is not dead, it was revived via a different mechanism one
      day after the co-occurrence version was refuted, and it shipped.
    - Batching ceiling "1.63×": **CONFIRMED** ("ceiling 1.63x"; "grouping tokens removes ~70–76%
      of I/O and 0% of compute").
    - Purgeable cache "1.6× slower at working budget": **CONFIRMED within rounding** (EFFICIENCY.md
      §24: purgeable-on 0.29–0.33 tok/s vs off 0.49–0.52 tok/s ≈ 1.6–1.7×).
    - LFRU "29.4% vs LRU 5.1%": **CONFIRMED EXACTLY** ("LRU collapses to 5.1% where LFRU still
      gets 29.4%").
    - mlock, stage-major, int8/SDOT, index-layout: not independently re-verified this pass
      (lower priority; general direction plausible given the accuracy hit rate above).

11. **Traction: Show HN 2026-07-29, ~375 stars week 1** — Show HN thread **CONFIRMED to exist**:
    "Show HN: A new engine to run Kimi K3 on a laptop", news.ycombinator.com/item?id=49098966.
    Exact date not independently pinned to 07-29 (repo created 2026-07-28; search engine described
    the post as "3 days" old relative to an early-August query, consistent with ~07-30, i.e.
    within a day of the claim, not exact). Star count: **stale, not wrong** — current actual count
    read via GitHub API on 2026-08-02 (repo is ~5 days old) is **796 stars / 77 forks / 2 open
    issues**, roughly double the "~375 week 1" figure — consistent with continued fast growth
    after the advisory's snapshot, not a contradiction of the week-1 number itself (unverifiable
    retroactively without a stars-over-time API).

12. **CLI/API surface for wrapping** — **CONFIRMED, real and wrapper-usable, three depths:**
    - **CLI subprocess surface**: `./waste plan <model.waste>` (preflight budget/report),
      `./waste run <model.waste> "<prompt>" -n <N> [--image <path>]`, `./waste chat <model.waste>`.
    - **HTTP surface**: `serve/` is an OpenAI-compatible server, stdlib-Python + ctypes into
      `libwaste.{so,dylib,dll}` — `python3 -m serve <model.waste> --port 8000`, then standard
      `POST /v1/chat/completions` with `{"model":..., "messages":[...]}`. Same shape as
      llama.cpp/vLLM servers Chaos already wraps conceptually.
    - **Native C API**: `src/waste.h`, ~26 functions, no global state (per CLAUDE.md) — an FFI
      integration point if a wrapper ever needs deeper control than CLI/HTTP.
    - **Env/config toggles**: `WASTE_VERIFY`, `WASTE_PURGEABLE`, `WASTE_MLOCK`, `WASTE_Q8`, and
      `ram_budget_bytes` in `waste_cfg` (the budget the resolver in #6 operates on).

## What WASTE actually is

An embeddable C11 MoE inference engine (Apache-2.0, sqliteai org, repo created 2026-07-28) whose
entire reason to exist is running the two open-weight Moonshot AI "Kimi" MoE models — **Kimi K3
(2.78T)** and **Kimi-Linear-48B** — on hardware too small to hold them in RAM, by keeping a dense
"trunk" resident and streaming per-token-selected experts off NVMe with cache-bypass I/O
(`O_DIRECT`/`F_NOCACHE`/`FILE_FLAG_NO_BUFFERING`) into an engine-owned bounded LFRU cache. Its
core engineering culture (append-only dated `LEARNED.md`, "run a cheap real test before an
expensive one," refuted ideas kept with their numbers) is verified as genuinely practiced — the
doc's own most-recent entries (§29→§34–36, 2026-07-31→08-01) show exactly that pattern in action:
an idea refuted, then revived via a better mechanism one day later and shipped. CPU-only by
default (portable C baseline + NEON/AVX2/AVX-512 dispatch); Metal exists but is 22% *slower* than
CPU for this workload; CUDA/ROCm/BLAS are explicitly unimplemented ("the flag refuses to build").

## Limitations + wrapper-relevant gaps

- **No GPU acceleration path at all on Chaos's target platform.** Chaos's v1 scope is
  Linux+NVIDIA (per `../research/mvp-scope.md`); WASTE has zero CUDA/ROCm support (issue #11,
  open, "CUDA backend: what would have to be true for it to pay" — the maintainers themselves are
  unconvinced it's worth building), and the one GPU backend that exists (Metal) is *slower* than
  CPU for this architecture ("several hundred small dependent matvecs per token, the worst
  possible shape for an accelerator" — docs/BACKENDS.md). A Chaos wrapper around WASTE would get
  no VRAM-offload story whatsoever — the opposite of ktransformers/llama.cpp's whole value
  proposition that Chaos's other research nodes are built around.
- **Model-family lock-in: only Kimi K3 and Kimi-Linear-48B, ever.** The `.waste` converter
  (`tools/convert.py`) and the residual-VQ quantization scheme are hand-built for Moonshot's
  specific KDA+MLA-hybrid latent-MoE architecture (docs/FORMAT.md, docs/ENGINE.md: docs are
  explicitly Kimi/Moonshot-specific, no generic-architecture converter or plan). **No DeepSeek
  support of any kind** — this is the single biggest gap relative to Chaos's own DeepSeek-class
  MoE focus (`../research/mvp-scope.md`): wrapping WASTE today would not serve Chaos's target
  model family at all without WASTE (or Chaos) building an entirely new converter + kernel path.
- **Inherently sub-1-tok/s at the flagship model regardless of tuning** (0.3–0.6 tok/s range
  across the week's docs for K3) — this is a physics/I/O-bandwidth ceiling, not something a
  wrapper can fix; only relevant for curiosity/correctness runs, not serving.
- **No concurrent/multi-user serving.** `waste_ctx` is explicitly not thread-safe; the server
  "serializes on one lock" (advisory §8, corroborated by the docs' own stated design). Measured
  batching ceiling is only 1.63× and "doesn't compose with read-ahead," no MTP head in the open
  release. This is *worse* than the batching/concurrency gaps Chaos already found in
  llama.cpp/ktransformers (`ktransformers-vs-llamacpp-moe-offload-gaps.md`) — WASTE doesn't even
  attempt concurrent serving yet.
- **Windows/portability still unclaimed in places**: only MinGW-w64 cross-compile is CI-verified;
  MSVC, ARM64 Windows, and cache-bypass I/O under load are explicitly not claimed. The AVX-512
  backend is "compiled and dispatched, never executed" on real hardware — CI runs on a Zen 3
  (AMD EPYC 7763) runner that predates AVX-512, so an entire SIMD path is untested on real silicon.
- **Genuinely brand-new, bugs still surfacing in the first days**: closed issues from the first
  week include O_DIRECT silently falling back to plain buffered I/O on Linux (#4 — meaning the
  cache-bypass design goal was broken in an early release), a SIGFPE crash on x86 with vision
  tensors (#10), an oracle-fixture/conversion mismatch (#7), and a broken Q8=0/Q4G trunk
  combination (#6). Two open issues remain: Linux auto-budget over-estimating usable memory under
  cgroup pressure (#14) and the CUDA-backend feasibility debate (#11).
- **No runtime observability/metrics surface found** (no Prometheus-style `/metrics`, unlike
  llama.cpp) — the closest thing is `waste plan`'s pre-flight budget report, which overlaps
  conceptually with Chaos's own `hardware-profiler.md` backlog rather than replacing the
  per-expert/cache-hit-rate observability gap Chaos already identified as open in both
  llama.cpp and ktransformers.
- **Conversion is a separate, Python-dependent offline stage** — inference itself needs zero
  third-party deps, but getting a model *into* `.waste` format needs Python + torch + safetensors
  (docs/FORMAT.md: "converter is Python... it needs torch and safetensors, which the inference
  path never does"), same "extra manual stage" pattern already flagged as a ktransformers gap in
  `ktransformers-vs-llamacpp-moe-offload-gaps.md`. Peak conversion RAM is modest ("a few hundred
  MB regardless of model size," shards streamed one at a time) but K3 conversion still implies
  having ~1.4 TB of source weights reachable somewhere.
- **Numbers are a genuinely moving target.** This verification's own paging-cliff table pull
  (2026-08-02) disagreed substantially with the advisory's (sourced ~2026-07-31) despite both
  presumably being accurate snapshots of the same repo four days apart — a caution for Chaos's
  own `benchmarking-methodology.md` concerns about pinning commit SHAs when citing any external
  project's numbers, WASTE very much included.

## Minimum hardware reality

- Smallest supported model: **Kimi-Linear-48B**, 19 GiB `.waste` container, current published
  minimum RAM **1.28 GB** (advisory's 1.87 GiB figure appears stale/superseded), ~10.65–10.7 tok/s
  measured on a 64 GB Mac at an 8 GB cache budget with 78% hit rate.
- **A 15.7 GB RAM / 6 GB VRAM / consumer-NVMe machine could run Kimi-Linear-48B** — the RAM floor
  (1.28–1.87 GB) is trivially met, and **no VRAM is required at all**: WASTE's default path is
  pure CPU (portable baseline + AVX2/NEON dispatch); the GPU (Metal) path is not needed and is
  slower anyway; CUDA doesn't exist as an option. Consumer NVMe only needs to hold the 19 GiB
  container, well within reach of any modern laptop/desktop SSD.
- **Open gap**: no published number for Kimi-Linear throughput at a realistic ~10–13 GB cache
  budget (i.e. what a 15.7 GB total-RAM machine could actually spare after OS/other overhead) —
  the only published data point (10.65–10.7 tok/s, 78% hit) is at an 8 GB *cache* budget on a 64
  GB machine with presumably more headroom elsewhere; a same-ballpark but not confirmed-identical
  result should be expected on the smaller box, not assumed identical.
- Kimi K3 is **not** runnable on that hardware profile at all — its floor alone (29.06 GB) exceeds
  15.7 GB RAM by nearly 2×, regardless of VRAM or NVMe.

## Open questions

- Exact `src/`-only LoC count (need to run `cloc`/`wc -l` directly on the tree; not in docs).
- Whether the 34%-at-2-bit and 53×/1.2e-05-absorbed-MLA figures exist verbatim somewhere in the
  repo not reached by this pass (candidates: sections of EFFICIENCY.md/K3.md beyond what the
  fetch tool surfaced, given these are large, actively-edited, section-numbered documents).
  This task had no local shell/`git clone` access, only WebFetch (which summarizes via a small
  model rather than returning raw bytes) — a real methodological caveat on this whole
  verification pass. Recommend a direct `git clone` + `grep` pass if these specific numbers
  matter for a decision.
- Precise Show HN posting date/points/comment count (HN item 49098966 rate-limited/blocked direct
  fetch during this pass).
- Whether the paging-cliff and other benchmark tables should be treated as re-measured on every
  Chaos decision that cites WASTE, given how fast they moved between 07-31 and 08-02 in this
  pass alone.
