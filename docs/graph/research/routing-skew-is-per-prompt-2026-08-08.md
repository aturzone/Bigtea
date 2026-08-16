---
topic: R0 — the routing skew is real, but the hot set is per-prompt, so it cannot be pinned. Corrects two numbers published in v0.0.2.
status: resolved
links: [routing-skew-changes-everything.md, ../backlog/next-session-handoff.md, verify-before-citing.md, v4flash-vs-llamacpp-2026-08-07.md]
---

R0 from `next-session-handoff.md`: re-measure the skew on several prompts before
building anything on it, because v0.0.2 measured **one** and the whole hot-set
cache plan rests on it. The answer changes the plan.

## The question, stated so it can be answered wrong

v0.0.2 published: 64 experts of 256 absorb **97.8%** of selections, chi-square
**7805** against uniform's ~255. From that came a 34.27 GiB cache, a **33.6
tok/s disk floor**, and "20 tok/s needs a ~48 GiB desktop, not 150".

Every one of those depends on a number measured **in-sample on one prompt**. A
cache is pinned *before* the prompt arrives, so the figure that matters is
coverage of a prompt the hot set was **not** chosen from. That was never
measured.

## Method

Eight prompts, four subjects, two each: English code, English prose, English
maths, and Persian — one Persian prose, one Persian *coding* question, so subject
and language are crossed rather than confounded. 145–198 tokens each, giving
870–1188 routing decisions per layer.

```
CHAOS_ROUTING=1 CHAOS_ROUTING_DUMP=csv/<name>.csv \
  chaos-run DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf "<prompt>" -n 1
```

`-n 1` is load-bearing: **one prefill, no generation** (see the last section for
why that matters). `CHAOS_ROUTING_DUMP` is new — it writes raw
`layer,expert,count` rows, which is what makes two runs comparable at all.

**Layers 0-2 are excluded from every number here.** They select out of
`ffn_gate_tid2eid` by token id, so their "skew" is the token distribution, not a
router. Excluding them *raises* the measured skew slightly (88.2% → 90.8% at
top-64 on `code_a`), so that caveat pointed the safe way.

Three controls v0.0.2 did not have:

- **uniform null** — the same number of draws from a uniform router. With 996
  draws over 256 experts you cannot spread them over more than 996 experts, so
  top-64 covers a large share *by construction*.
- **noise ceiling** — a fresh sample of the same size from a prompt's *own*
  distribution, scored against another such sample. Identical true router by
  construction, so whatever coverage this loses is estimation noise.
- **out-of-sample scoring** — the hot set from prompt A scored on prompt B.

## The skew is real

Consistent across all eight, and nowhere near the null:

| prompt | sel/layer | top-8 obs / null | top-64 obs / null |
|---|---:|---:|---:|
| code_a | 996 | 46.3 / 7.1 | 90.8 / 41.6 |
| code_b | 870 | 43.1 / 7.4 | 90.3 / 43.2 |
| farsi_a | 1188 | 51.4 / 6.8 | 92.2 / 40.2 |
| farsi_b | 1158 | 52.0 / 6.8 | 93.0 / 40.4 |
| math_a | 924 | 45.1 / 7.3 | 89.9 / 42.4 |
| math_b | 882 | 43.0 / 7.4 | 90.1 / 43.0 |
| prose_a | 984 | 36.7 / 7.1 | 89.0 / 41.7 |
| prose_b | 882 | 34.6 / 7.4 | 88.5 / 43.0 |

Top-8 runs **5-7x the null**. The router is genuinely skewed and that finding
survives.

**But top-64 is 90.5% on average, not 97.8%** — and against a null of ~41%, so a
large part of the published headline was sample-size saturation rather than
routing.

## The hot set does not transfer

Top-64, averaged over all ordered pairs:

| | random cache | different subject | same subject | noise ceiling | in-sample |
|---|---:|---:|---:|---:|---:|
| coverage | 25.0% | **37.5%** | **61.3%** | 88.7% | 90.5% |

The noise ceiling is 88.7% against an in-sample 90.5%, so **estimation noise
costs under two points.** The collapse to 61.3% within a subject and 37.5%
across subjects is therefore real divergence, not a small sample.

**Across subjects a pinned global hot set is barely better than a random one**
(37.5% against 25.0%).

Subject matters more than language — `farsi_b` is a coding question in Persian,
and it sits with the code prompts, not with Persian prose:

| top-64 | same language | different language |
|---|---:|---:|
| **same subject** | 61.1% | 48.5% |
| **different subject** | 39.4% | 32.2% |

## What a pinned cache would actually deliver

Leave-one-out: build the hot set from the other seven prompts, score it on the
held-out one. This is the number a shipped static cache achieves.

| cache | size | pinned hit rate | disk floor | *in-prompt* hit | *ceiling* |
|---|---:|---:|---:|---:|---:|
| top-8 | 4.28 GiB | 14.5% | 0.86 tok/s | 44.0% | 1.32 tok/s |
| top-16 | 8.57 GiB | 21.8% | 0.94 tok/s | 58.6% | 1.79 tok/s |
| top-32 | 17.13 GiB | 34.2% | 1.12 tok/s | 75.0% | 2.95 tok/s |
| **top-64** | **34.27 GiB** | **53.7%** | **1.60 tok/s** | 90.5% | 7.76 tok/s |

Disk floor only, on the measured 2.37 GiB/s drive against 3.21 GiB of routed
experts per token. The *in-prompt* column uses each prompt's own top-k — no cache
can reach it, since it would have to know the routing before running the prompt.
It is an upper bound on any adaptive policy, not a target.

**The published 33.6 tok/s becomes 1.60 tok/s at the same 34.27 GiB.** A 21x
correction, and it comes entirely from scoring out-of-sample instead of in.

### 20 tok/s

20 tok/s on this drive needs a **96.3%** hit rate. A pinned cache does not reach
it at any size measured — top-128 (68.5 GiB) gives 76.7%. The in-prompt ceiling
reaches it only at 68.5 GiB, which would need an ~80 GiB machine *and* a policy
that performs like an oracle.

**"20 tok/s on a ~48 GiB desktop" is not supported by this measurement.**

## Why v0.0.2's two numbers cannot both be right

97.8% coverage with chi-square 7805 does not occur in any single-pass run here.
`code_a` gives chi **7402** — essentially the published value — at **88.2%**
coverage. The 17-token smoke prompt gives 98.8% coverage at chi **1282**. High
chi and high coverage are opposite ends of prompt length.

They co-occur when the same short prompt is counted repeatedly. `chaos-run`
regenerates **statelessly** — every generated token re-runs prefill over the
whole sequence — so `routing_report` counts one prompt once per pass. Same
17-token prompt, three depths:

| | prefill passes | pooled chi-square | top-64 coverage |
|---|---:|---:|---:|
| `-n 1` | 1 | 1282 | 100.0% |
| `-n 4` | 4 | 5464 | 100.0% |
| `-n 8` | 8 | 11469 | 100.0% |

**Chi-square scales linearly with the number of passes; coverage does not move
at all.** (Slightly super-linear because the sequence grows by one token per
pass.) A chi-square built from re-counted passes is not a test statistic — the
draws are not independent, they are the same draws counted again.

The exact v0.0.2 command was never recorded, so this is the mechanism rather
than a reconstruction. That omission is the finding: this project's own rule
that a claim needs its command line and output pasted was applied to
*competitors'* numbers and not to its own.

## Corrections

- **top-64 = 97.8%** → **90.5% in-sample, 53.7% out-of-sample.** The deployable
  figure is the second one.
- **chi-square 7805** → not a valid statistic as computed. `routing_report` now
  prints a per-layer chi-square alongside the pooled one, and pooling across
  layers is itself questionable (expert 7 of layer 3 and expert 7 of layer 30 are
  unrelated weights).
- **33.6 tok/s disk floor at 34.27 GiB** → **1.60 tok/s** pinned, 7.76 tok/s at
  the unreachable in-prompt ceiling.
- **"20 tok/s needs ~48 GiB"** → unsupported; ≥68.5 GiB of cache on an oracle
  policy, so an ~80 GiB machine at best.

## What this does to the plan

- **R1 must not pin.** The frequency-gated adaptive policy already in `stream.rs`
  is the right shape — it warms per prompt, which is exactly the regime that
  transfers. What dies is the *sizing story* built on a global hot set.
- **"Prune the model to its hot set" is dead as written.** The handoff called it
  the most promising item with no research risk: keep 64 of 256 experts per
  layer, 34 GiB instead of 144, "loses 2.2% of routing". Measured out-of-sample
  it loses **46%**. A pruned container would route unseen prompts to experts that
  are not in it.
- **A VRAM tier has the same constraint.** Pinning the top-10 experts into 6 GB
  of VRAM inherits the 37.5% cross-subject figure. It must be warmed, not pinned.
- **The next measurement, and it now gates R1's value:** does a set warmed on the
  *prompt* predict the routing of *generated* tokens? Everything here is prefill.
  If prefill routing predicts generation routing, an adaptive cache lands near
  the in-prompt column and is worth building. If it does not, it lands near the
  cross-prompt column and is worth much less. Nothing measured so far
  distinguishes these.

## Caveats

- **Eight prompts, one model, prefill only.** Leave-one-out pools seven; a larger
  calibration corpus would raise 53.7% somewhat. It will not reach 97.8% — the
  noise ceiling shows the gap is divergence, not sampling.
- **top-128 is saturated and nothing should be built on it.** Observed 99.5%
  against a null of 70%; at ~1000 selections per layer the metric is mostly
  measuring sample size. Trust top-8 through top-64.
- **Disk floors ignore compute**, which the earlier node put at a 27 tok/s
  ceiling, and ignore this project's own warning that a cached byte which gets
  paged out is a page fault in disguise. Only tok/s at a stated footprint counts.
- Prompts, captures, and the analysis script are reproducible from
  `CHAOS_ROUTING_DUMP` plus `analyse.py`; the raw CSVs are not committed.
