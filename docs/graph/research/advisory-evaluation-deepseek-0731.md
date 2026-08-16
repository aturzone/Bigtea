---
topic: Evaluation of the Deepseek V4 flash 0731 advisory — what to adopt, what to reject
status: resolved
links: [waste-engine-verified.md, moe-landscape-2026-08.md, ../decisions/strategy-post-waste.md]
---

Evaluates `docs/Deepseek V4 flash 0731/` (6 files, advisory by another model, dated 2026-08-01),
against the two verification nodes. Atur's instruction: "they are just advice — check what is good
what is not." Verdict per item, with the verification source.

## Overall grade

**Directionally excellent, factually ~75%.** The strategic reasoning is the strongest part and
mostly survives verification. The failure mode is characteristic: **specific numbers and
identifiers invented with the same confident tone as the real ones** — a reader who trusted it
line-by-line would have cited three fabricated facts in public. It flagged its own staleness risk
honestly (03 §6) and did not touch the graph, which was correct discipline.

## GOOD — verified, adopt

- **The core physical model**: cache floor = one token's working set; below it hit rate is
  *exactly* zero. CONFIRMED verbatim (2604 evictions / 2704 accesses; K3 = 16×92×11.8MB = 17.4GB).
  This is the most valuable transfer in the whole folder — see `waste-engine-verified.md` §4.
- **Time breakdown** expert I/O 54.8% / matmul 27.2% / KDA 9.3% — CONFIRMED to the decimal, zero
  drift. Reads are ~2× the arithmetic. Grounds the "I/O is a first-class term" argument.
- **Budget resolver shape**: refuse below floor; step down in *whole* working-set multiples
  (3×→2×→1×→floor). CONFIRMED verbatim. Genuinely copy-worthy.
- **Refuted levers, mostly CONFIRMED**: GEMQ per-expert bit allocation flat at 1.01× across layers;
  batching ceiling 1.63×; purgeable cache ~1.6× slower at the working budget; LFRU 29.4% vs LRU
  5.1%; 3-bit trunk logit collapse (36% off, generation degenerates to `+` and spaces); 19.4%
  error at 3 bits. Adopting these saves real tickets.
- **L4 — "SSD only gates load time" is false**: correct and important. 54.8% of a K3 decode step
  is expert I/O; enclosure bandwidth is a hard ceiling. Our `hardware-profiling.md` claim needs the
  correction the advisory proposed.
- **L8 — "a test on the small model does not test the big one"**: correct, and independently a
  vindication of the T0-gate instinct.
- **L7 / A7 — dated, append-only, keep-refutations-with-numbers**: adopted; this correction block
  convention is now in use (see the gaps node).
- **Strategic core (04)**: "don't out-engine; the wrapper lane is unclaimed" — survives
  verification and is *strengthened* by it (no project ships cross-engine probe→recommend→launch
  →bench). The "don't chase" list (no engine-building, no Mac frontier, no prefetch/batching
  rebuilds, no KV-offload headline) is sound.

## BAD — fabricated or wrong, do NOT propagate

- **`/moe-layer-perf` endpoint, `--moe-layer-perf-out` heatmap flag, Web UI** — **do not exist.**
  Absent from PR #24524, Discussion #24528, and every leloch branch. Invented.
- **"45.89% break-even hit rate" and "~70% practical ceiling"** — **do not exist** in the RFC or
  anywhere. Invented. The *real* numbers from that discussion: top 10% of experts take ~80% of
  hits (Qwen3.5-122B); ~99% simulated hit rate at 69% expert budget; +7–57% (PR) / +25% on
  GLM-5.1 754B. Cite these instead.
- **Branch `cached-experts-v2`** — does not exist. Real branches: `moe-cache`, `moe-cache-pr`,
  `v3-expert-cache`.
- **"7/8 of physical RAM" cap** — NOT FOUND in WASTE's docs; the docs say the cap is available
  system memory, "not a fixed fraction like 7/8". The advisory built L2/A3 on a constant it seems
  to have invented. Adopt the *rule* (largest whole multiple that fits), not the 7/8 figure.
- **Cross-layer prefetch presented as a dead end** — the single most consequential error. WASTE
  refuted the co-occurrence predictor (§29, 29.0% vs 29.5%) and then **revived it via a different
  mechanism one day later** — running the *next* layer's router weights on the *current* layer's
  hidden state hit **59.0% recall@16** (§34) and **shipped for a 1.17× throughput gain** (§35,
  2026-08-01). The advisory's "do not rebuild this" list would have steered us away from a lever
  that demonstrably works. **Lesson: a refutation is scoped to the mechanism tested, not the idea.**

## STALE — right when written, wrong now (the project ships daily)

- Kimi-Linear minimum RAM: advisory 1.87 GiB → actual **1.28 GB**.
- K3 decode: advisory 0.49–0.54 tok/s → current README **0.45–0.62**, other sections up to 0.63.
- The whole paging-cliff sweep table: same *shape* (sharp cliff past the sweet spot), every
  absolute number different four days later. **Never cite WASTE's tables without a commit SHA.**
- Stars: WASTE ~375 → **796** (2026-08-02); ds4 13.4k → **~19.9k**.
- "8× slower from freeing 1.11 GB" — mechanism corroborated, the 8× multiplier no longer
  reproduces (current 58GB row is 0.07–0.08, not 0.04).

## The advisory's biggest strategic miss

It recommended (04 §C, 05 §A6) keeping **WASTE warm as a future wrapped backend**. Verification
kills this for our scope: WASTE has **zero CUDA/ROCm** (issue #11 open; maintainers themselves
unconvinced), its only GPU path (Metal) is **22% slower than CPU**, and its converter is
**hard-locked to Moonshot's Kimi architecture — no DeepSeek support exists or is planned**.
Chaos's v1 is DeepSeek-class on Linux+NVIDIA. Wrapping WASTE would serve *neither* axis.
**ds4/DwarfStar is the far better future-backend candidate** (DeepSeek-V4-Flash + GLM-5.2,
Metal+CUDA+ROCm, MIT, ~20k stars) and the advisory under-weighted it.

## Net actions taken

- A1 (SSD correction) → adopted, see `hardware-profiling.md` amendment.
- A2 (working-set schema fields) → adopted into the backlog, constants corrected.
- A3 (budget resolver) → adopted as a *rule*, minus the invented 7/8 constant.
- A4 (currency pass) → **done**, found worse than the advisory predicted (see gaps-node
  correction block).
- A5 (known-refuted note) → adopted **with the prefetch-revival correction**, which the advisory
  got backwards.
- A6 (WASTE as warm backend) → **rejected for WASTE**, redirected to ds4.
- A7 (dated append-only convention) → adopted, in use.

## Open questions

- Whether the 34%-at-2-bit and 53×/1.2e-05 absorbed-MLA figures exist verbatim in WASTE's repo —
  the verification pass had WebFetch only (summarizing), not raw bytes. A `git clone` + `grep`
  would settle both.
- Whether WASTE's §34 router-lookahead prefetch generalizes off the disk lane (it shipped for a
  streaming engine; the GPU lane already has FATE's 99%+ result). If it generalizes, the
  "prefetch is refuted" line dies completely.
