---
topic: R0.1 — a cache warmed on the prompt predicts what generation needs. R1 is worth building.
status: resolved
links: [routing-skew-is-per-prompt-2026-08-08.md, ../backlog/next-session-handoff.md, ../backlog/the-big-bang.md]
---

R0 left R1 undecided. It showed the hot expert set does **not** transfer between
prompts — pinned from one it covers 53.7% of another, and 37.5% across subjects
against 25.0% for caching at random. But R1 does not pin. It warms on the prompt
it is actually running, and then spends that warmth on the tokens the model goes
on to generate. **Nothing had measured whether the prompt predicts its own
continuation**, and the two bounds R0 left were 1.60 and 7.76 tok/s.

## Answer

**It predicts it well.** A top-64 set taken from the prompt alone covers **86.3%**
of the routing of the tokens generated after it, against an in-prompt oracle of
90.8% and a cross-prompt figure of 53.7%.

Two prompts, a 166-token coding one and a 164-token prose one, 15 generated
tokens each, learned-gating layers only:

| top-K | cache | frozen `code_a` | frozen `prose_a` | warming | in-prompt oracle | random |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 4.28 GiB | 43.5% | 36.0% | 44.5 / 37.7% | 46.3 / 36.7% | 3.1% |
| 16 | 8.57 GiB | 58.6% | 50.8% | 59.4 / 53.4% | 60.1 / 53.1% | 6.2% |
| 32 | 17.13 GiB | 72.9% | 69.2% | 74.0 / 72.6% | 75.7 / 72.0% | 12.5% |
| **64** | **34.27 GiB** | **86.3%** | **85.9%** | 87.0 / 87.8% | 90.8 / 89.0% | 25.0% |

**Generation routing sits within ~4 points of the oracle and ~32 points above the
cross-prompt figure, on both.** The regime a cache actually operates in is the
favourable one, and it does not depend on the subject.

**Over 15 tokens, continuing to warm adds almost nothing** — about a point at
top-64 — and there is no decay: `code_a` drifts 82.5% → 86.8% across thirds,
`prose_a` 87.6% → 85.4%.

### ⚠ That does not survive a longer horizon

Re-measured over **46** generated tokens (17-token prompt, same method):

| top-64 | frozen at the prompt | kept warming |
|---|---:|---:|
| 15 tokens, 166-token prompt | 86.3% | 87.0% |
| **46 tokens, 17-token prompt** | **59.4%** | **81.7%** |

A frozen set decays steadily — 85% on the first generated token to ~50% by the
46th, first third 63.9% against last third 55.3% — while one that keeps warming
holds at 81.7%. **Continuing to warm is worth 22 points, not 0.7.**

So the earlier "fill it during prefill and leave it" was an artefact of a short
horizon, and is withdrawn. **Keep warming.** The implementation already does —
frequency-gated admission runs on every miss — so nothing changes in the code,
only the claim.

Two things confound the comparison and neither is resolved here: the longer run
also has a *shorter prompt* (17 vs 166 tokens), so its frozen set is estimated
from 102 selections per layer rather than 996 and starts weaker. Its `in-prompt`
column reads 100.0% at top-64 for the same reason — a 17-token prompt touches
only ~40 distinct experts per layer, so that column is saturated and means
nothing. **What is solid is the direction: the frozen set decays and the warmed
one does not.**

```
per generated token, top-64 frozen
code_a   78 74 85 89 87 90 88 93 90 88 82 85 89 90 87
prose_a  90 88 85 88 86 87 86 82 85 83 86 86 87 86 82
```

## Method, and why the deltas are trustworthy

`bigtea-run` regenerates statelessly: pass *k* re-runs prefill over prompt plus
*k* generated tokens. The model is causal, so token *i*'s routing is identical in
every pass containing it — which makes `pass[k] - pass[k-1]` **exactly** the
routing of the one token generated in between.

That property is asserted, not assumed. The analysis checks every delta is
non-negative and sums to exactly `n_expert_used` per layer; a violation would
mean a token's routing changed between passes and the subtraction is invalid. On
both runs: **0 negative cells, 6.0 selections per layer per token.**

```
BIGTEA_ROUTING=1 BIGTEA_ROUTING_DUMP=gen/code_a.csv \
  bigtea-run DeepSeek-V4-Flash-...-00001-of-00005.gguf "<166-token prompt>" -n 16
python tools/routing/analyse_gen.py gen/code_a.csv
```

The per-pass dump is new in this change; before it, all passes accumulated into
one histogram, which is what inflated v0.0.2's chi-square. **The artefact became
the instrument.**

## What it is worth

Disk floor on the measured 2.37 GiB/s drive against 3.21 GiB of routed experts
per token:

| cache | warmed hit | GiB/token | disk floor |
|---|---:|---:|---:|
| 4.28 GiB | 44.5 / 37.7% | 1.78 / 2.00 | **1.33 / 1.18 tok/s** |
| 8.57 GiB | 59.4 / 53.4% | 1.31 / 1.50 | 1.82 / 1.58 tok/s |
| 17.13 GiB | 74.0 / 72.6% | 0.83 / 0.88 | 2.84 / 2.70 tok/s |
| 34.27 GiB | 87.0 / 87.8% | 0.42 / 0.39 | 5.67 / 6.05 tok/s |

`code_a / prose_a`. llama.cpp generates this model at **0.21–0.31 tok/s** on the
same machine.

**⚠ These floors describe a KV-cached step, which does not exist yet.** They
assume a token needs 6 experts per layer. Today a generated token re-runs prefill
over the whole sequence, and expert reads are **deduplicated per block across the
batch** — so a pass reads the *distinct* experts its tokens select, which for a
166-token sequence is **122.8 distinct experts per layer, ~66 GiB in one pass**
(measured: 6 at one token, 39.7 at 17 tokens, 122.8 at 166).
**A few GiB of cache cannot touch that.** The numbers above are what R1 is worth
*after* R3, not before it. See `../backlog/next-session-handoff.md`.

**On this 15.7 GiB laptop**, after the 7.38 GiB always-read set there is room for
roughly the 4.28 GiB tier, so the floor is **~1.3 tok/s — about 4–6x llama.cpp.**

Worth stating plainly: v0.0.2 predicted ~1.3 tok/s for this laptop too. It got
there by assuming a *global* hot set at ~45% hits; the truth is a *warmed
per-prompt* set at 44.5% hits at the same size. **The laptop number survived its
own reasoning being wrong** — which is a reason to keep measuring, not a reason
to trust the old arithmetic.

## What this settles, and what it does not

- **R1 is worth building, but only after R3.** The hit rates justify the cache;
  the dedup note above says it cannot pay until a step's working set shrinks to
  6 experts per layer. **R3 → R1 → R2**, in that order.
- **Keep warming during generation.** Over 15 tokens it buys a point; over 46 it
  buys 22, because a frozen set decays and a warmed one does not. The cache
  already behaves this way.
- **Routing is not bitwise stable across sequence lengths.** At exactly one
  transition — 63 → 64 tokens — the net stayed at +6 selections per layer, so
  one token really was added, but **477 selections (~3%) of tokens already in
  the sequence moved**. Every other transition had zero. That is the router's
  top-6-of-256 flipping on near ties when ggml changes matmul blocking at a size
  boundary: arithmetic, not causality. **Anything that assumes routing is
  reproducible across batch shapes — a prefetcher, a replay, or R3's
  equivalence test — must tolerate it.**
- **R2.3's speculative prefetch is now sized**: prefetching block L+1 on the
  previous token's routing should hit at roughly this rate, because it is the
  same quantity.
- **The cache's frequency counter must be weighted by selections, not requests.**
  Reads are deduplicated, so an expert chosen by ninety tokens and one chosen by
  a single token arrive as one request each. Unweighted, every count ties on a
  long prompt and the cache freezes on whatever loaded first.

Not settled:

- **46 generated tokens is still not 500.** The decay measured over 46 has not
  flattened, so a long answer may fall further. The warmed figure is the one to
  plan with.
- **The long-horizon run changes two variables at once** (46 tokens *and* a
  17-token prompt), so the frozen/warmed gap cannot be attributed to horizon
  alone. Re-running 46 tokens from a 166-token prompt would separate them; it
  costs about 40 minutes.
- **Three prompts, one model.**
- **Disk floor only.** Compute floors at ~27 tok/s, so it does not bind yet, but
  the arena and memcpy costs this project has already hit twice are not in this
  arithmetic.
- **The cache must own its memory.** Past ~6 GiB on Qwen3 a 71%-hit cache was the
  *slowest* configuration measured, because cached bytes got paged out and a
  "hit" became a page fault in disguise. **A hit rate is not a tok/s.** None of
  the floors above are measured throughput and none should be quoted as if they
  were.
