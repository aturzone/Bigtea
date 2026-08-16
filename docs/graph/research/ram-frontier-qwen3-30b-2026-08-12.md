# The tok/s-versus-RAM frontier for Qwen3-30B-A3B

**2026-08-12.** Branch `ticket/r14-architectures`. The first published curve of
generation speed against *owned* cache size for a model of this class — and the
reason it can be swept at all is that this engine is told how much RAM to use.
`mmap` cannot be asked for exactly N GiB.

Links: [the-knee-moves-with-n-2026-08-14.md](the-knee-moves-with-n-2026-08-14.md) ·
[gpu-tier-smallest-honest-slice-2026-08-11.md](gpu-tier-smallest-honest-slice-2026-08-11.md) ·
[expert-cache-is-early-not-wrong-2026-08-08.md](expert-cache-is-early-not-wrong-2026-08-08.md) ·
[v4flash-has-no-slack-2026-08-10.md](v4flash-has-no-slack-2026-08-10.md)

> **2026-08-14 — this curve's headline holds only at `-n 16`.** Sweeping
> generation length as a second axis found the working set growing with it:
> 5.53 GiB at 16 tokens, 7.05 at 64, 10.14 at 256, so the knee moves 6 → 8 → 12
> GiB and at 256 tokens the curve has not flattened by the largest budget
> swept. Every number below is correct *for sixteen tokens*, and this node's
> own closing section is what asked for that measurement — "where the curve
> flattens is a property of the workload". It was, and the workload includes
> how much you generate. See
> [the-knee-moves-with-n-2026-08-14.md](the-knee-moves-with-n-2026-08-14.md).

## Read this before the numbers

**Qwen3-30B-A3B is NOT in `VERIFIED_ARCHITECTURES`.** It was removed the same
day, in this branch: its first eight-prompt diff against llama.cpp returned
1 FAIL + 6 unstable. One cause was found and fixed (an MoE container has no
`ffn_gate`, so it was classified ungated and ran GELU where the reference runs
SiLU); a smaller stable-reference divergence remains, four countries into the
factual prompt. It is the only container on this machine in the size class where
the curve is interesting, so the sweep is published with that stated rather than
withheld or quietly labelled.

**2026-08-14, re-run with `-b 1` in the harness's re-check set — it did not
clear.** The remaining FAIL became `unstable`, exactly as predicted, because the
reference does disagree with itself on that prompt once batching is probed. But
the *count* held: 0 FAIL and **6 of 8 prompts unstable**, which
`parity-check.sh` reads as a cluster rather than chance and exits non-zero on.
Every prompt tokenizes identically in both engines, so it is not the input. The
standing of this curve is therefore unchanged and slightly worse-founded than it
read on the 12th: it is measured on a model whose divergence from llama.cpp is
now **unexplained rather than excused**. The activation was one bug, it is
fixed, and something else is still there.

This matters more than a footnote, because **the activation fix changed the
workload, not just the arithmetic.** The same sweep on the pre-fix build read:

| | streamed at ≥6 GiB | hit rate | 1 GiB | 6 GiB |
|---|---|---|---|---|
| GELU (wrong) | 7.00 GiB | 80% | 1.26 tok/s | 2.92 |
| SiLU (correct) | **5.53 GiB** | 70% | 0.78 | 2.63 |

Different activations produce different FFN outputs, which become different
router inputs at the next layer, which select **different experts**. A wrong
activation is therefore a wrong *residency benchmark* too. Anyone measuring a
cache on an unverified model is measuring the bug.

## Method

```bash
# One run, repeated. `--cache GIB` is the whole point: the engine owns that
# many GiB of expert cache and nothing else decides.
./target/release/chaos-run.exe \
    -m /c/Projects/models/qwen3moe/Qwen3-30B-A3B-Q4_K_M.gguf \
    -p "The capital of France is" -n 16 --temp 0 --cache <1|2|4|6|8|10|12>
```

Five rounds, **interleaved** — every budget once per round, in order — so
drift spreads across the curve instead of loading onto whichever budget ran
last. Medians of five. Free physical RAM sampled immediately before and after
every run:

```bash
powershell.exe -NoProfile -Command \
  '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1048576,2)'
```

That column is not decoration. In the first attempt at this sweep, round 5's
low-budget runs came out flat (1.55/1.50/1.52 against 1.26/1.75/2.54) while free
RAM *rose* 8.7 → 10.4 GiB — this session's own git and doc work releasing
memory mid-round. Without the column that round would have been folded into the
medians and reported a flatter curve.

## The curve

Qwen3-30B-A3B-Q4_K_M, 17.28 GiB container, 0.93 GiB always-read, 16.35 GiB of
routed experts. Machine: 15.7 GiB total, **9.3–10.2 GiB free throughout**.

| `--cache` | tok/s (median of 5) | range | vs 1 GiB | streamed | hits | evictions | disk |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 GiB | 0.78 | 0.61–1.19 | 1.00x | 12.13 GiB | 35% | 1758 | 19.3 s |
| 2 | 1.62 | 1.33–2.03 | 2.08x | 9.34 | 50% | 1957 | 9.0 |
| 4 | 1.85 | 1.63–2.18 | 2.37x | 6.69 | 64% | 1286 | 7.3 |
| **6** | **2.63** | 2.10–2.74 | **3.37x** | **5.53** | 70% | **0** | 5.6 |
| 8 | 2.56 | 2.17–3.12 | 3.28x | 5.53 | 70% | 0 | 5.5 |
| 10 | 2.13 | 1.56–3.05 | 2.73x | 5.53 | 70% | 0 | 8.6 |
| 12 | 2.56 | 2.36–2.92 | 3.28x | 5.53 | 70% | 0 | 6.0 |

**The frontier rises to 6 GiB and is flat after it. 3.37x for 6 GiB of owned
residency.**

## Why it flattens exactly there, and how much of the tail is noise

The saturation is not a speed observation, it is a **capacity** one, and the
engine reports it directly: at 6 GiB and above, `streamed` is 5.53 GiB and
`evictions` is **0**. The distinct expert bytes this prompt touches over 16
generated tokens are 5.53 GiB. Once the budget covers them, more budget has
nothing to hold. Below 6 GiB the same run streams 6.69, 9.34, 12.13 GiB — it is
re-reading experts it already had.

**The 8/10/12 rows are a free null.** They are provably one configuration —
identical streamed bytes, identical hit rate, zero evictions — so their spread
is this machine's noise floor with no work required:

- medians 2.56 / 2.13 / 2.56 → **16.8% spread between medians of identical runs**
- single runs across all fifteen: **1.56 – 3.12 tok/s**

So the 10 GiB dip is noise, and **nothing above 6 GiB is distinguishable.** The
1 → 6 GiB climb (3.37x) is far outside that band; the 6 → 8 step (−2.7%) is far
inside it and is not a result.

## A confound named rather than hidden

Round medians declined monotonically: **2.72, 2.56, 2.16, 2.10, 2.03** across
rounds 1–5, while free RAM held at 9.3–10.2 GiB. So it is not memory pressure.
Thermal or drive state is the likely cause and neither was measured.

Because budgets run in a fixed order within each round, this drift penalises the
*late* budgets in every round — the large ones. The measured climb is therefore
**conservative**, not inflated. It also means the absolute tok/s here are
session-local: compare rows within this table, never against another day's.

## What this decides

The GPU node argued the VRAM tier's value is a point on this curve, and that if
the curve had already flattened the tier was dead for the same reason the
byte-reduction roadmap closed. **The curve is flat above 6 GiB on this model, on
this machine, at this generation length** — and the machine has 9–10 GiB free,
so the flat region is already reachable without a GPU.

That is not "the VRAM tier is dead", and the difference is the whole finding:

- **Where the curve flattens is a property of the workload, not the hardware.**
  It flattens at 5.53 GiB because that is what 16 generated tokens of this
  prompt touch. Distinct expert bytes grow with generation length — measured on
  the earlier build as roughly `n^0.18`, so a longer generation saturates later
  and higher.
- **On this model the flat region fits in host RAM, so VRAM adds nothing.** 5 GiB
  of VRAM against a 16.35 GiB expert bank is 31%; against V4-Flash's 137 GiB it
  is 3.6%. Neither is the case where a second tier changes the shape.
- **The case where it would** is a model whose working set exceeds free host RAM
  but not host + VRAM. That is a genuine window and this machine cannot reach it:
  Qwen3-30B saturates below free RAM, V4-Flash needs orders of magnitude more.

So the recommendation from the GPU node stands and is now measured rather than
argued: **no GPU ticket.** The next thing worth measuring is the same sweep at
several generation lengths, because the frontier is a *surface* in (cache size,
tokens generated) and only one slice of it exists.

## Honest limits

- **One prompt, one model, one machine, one session.** The hot set is per-prompt
  (`routing-skew-is-per-prompt-2026-08-08.md`), so the saturation point is too.
- **The model is unverified**, with a known remaining divergence from llama.cpp.
- **`-n 16`.** Short. The saturation point moves with it, and that dependence is
  the more interesting axis.
- Round-over-round drift of ~25% top to bottom, cause unidentified.
- No comparison against llama.cpp, which cannot be given a cache budget — that
  is why this curve is ours to publish, and also why it has no baseline.
