# Generation on Qwen3-30B is no longer 2x behind — it is 0.90x

**2026-08-16.** Qwen3-30B-A3B-Q4_K_M (17.28 GiB container) on a 15.7 GiB
machine. Both engines run **alternately in one session**, five pairs, medians.
`llama-completion` from `llamacpp-unsloth/build/bin`, CPU build.

```
chaos-run  <model> "The capital of France is" -n 16 --temp 0 --force
llama-completion -m <model> -p "The capital of France is" -n 16 --temp 0 --no-warmup -no-cnv
```

| | Chaos | llama.cpp | ratio |
|---|---:|---:|---:|
| generation tok/s | **3.03** `[2.67 2.92 3.03 3.38 3.59]` | **3.35** `[2.89 3.00 3.35 3.60 3.69]` | 0.90x |
| prefill tok/s | **1.22** `[1.21 1.22 1.22 1.24 1.25]` | **1.17** `[1.04 1.10 1.17 1.19 1.56]` | 1.04x |

## What this retracts

`CLAUDE.md` has said, for months and in bold:

> **Generation is still ~2x behind** (1.07 vs 2.16) — do not claim otherwise.

**That no longer reproduces.** Measured today, alternating in one session, it is
0.90x. The instruction was right when it was written and is wrong now, which is
exactly the failure mode this project keeps finding: a number written down once
and never re-run.

## What it does not claim

**Not parity, and certainly not a lead.** 0.90x is *behind*, and the generation
spread is wide on both sides — Chaos 2.67–3.59, llama.cpp 2.89–3.69, ranges
overlapping almost completely. What can be said is that **the gap is now inside
the measurement noise of this setup**, and that the specific "2x" figure is
dead.

Prefill at 1.04x is the tighter number: Chaos's five runs span 1.21–1.25 (3%)
against llama.cpp's 1.04–1.56 (50%), so the medians are comparable but the
reference is the noisy one there.

**A short prompt with a warm page cache.** Six prompt tokens, 16 generated, and
by the fifth pair the OS holds a great deal of the container. Both engines get
that equally because the runs alternate, but the absolute numbers are not what a
cold machine would show. `CLAUDE.md` already records that V4-Flash figures drift
with page-cache state and must only be compared within a session; the same
applies here.

## Why it probably moved

Nothing here was aimed at this number. Everything landed between the old
measurement and this one:

- the `-t` / `-tb` split, after the old sweep turned out to have a disconnected
  knob (1.66x/1.69x)
- `compute()` called once per phase instead of per intermediate (1.9x at one
  token)
- a file handle per reader instead of one shared (2.01 → 2.65 GiB/s)
- frequency-gated cache admission (17% → 70% hit rate)
- R2 read/compute overlap, and R3's KV cache

## The rule

**Re-run the headline before quoting it.** This is the third retraction in this
project's short history and the first one that moved a number *in our favour* —
which is not better. A stale claim that flatters us is exactly as wrong as one
that does not, and it is worse for planning, because the whole "generation is
2x behind" framing has been steering which work gets picked.

## Addendum: re-measured after parallel experts landed

Same protocol, five alternating pairs, with `parallel-experts-2026-08-16.md`
merged.

| pair | Chaos | llama.cpp | ahead |
|---:|---:|---:|---|
| 1 | 3.53 | 3.62 | llama.cpp |
| 2 | 3.81 | 3.87 | llama.cpp |
| 3 | 3.60 | 3.25 | Chaos |
| 4 | 3.31 | 2.96 | Chaos |
| 5 | 3.03 | 2.53 | Chaos |

Medians 3.53 against 3.25 — 1.09x — and prefill 1.22 against 1.25, 0.98x.

**This is parity and nothing more.** The paired count is 3–2, the ranges overlap
almost entirely (3.03–3.81 against 2.53–3.87), and **both series decline across
the session** (Chaos 3.53 → 3.03, llama.cpp 3.62 → 2.53), which after hours of
back-to-back 17 GiB runs looks like thermal drift rather than anything about
either engine. Alternating controls for that *between* the two, which is why the
paired count is the number to read and the medians are not.

So the honest sequence for generation on this model is **0.90x → parity**, with
the 1.10x from parallel experts accounting for the move. A lead would need a
cold machine, more pairs, and a reason to believe the decline is not systematic.
