---
topic: the tok/s-versus-resident-RAM frontier for V4-Flash, and what 20 tok/s actually costs
status: resolved
links:
  - v4flash-has-no-slack-2026-08-10.md
  - parallel-experts-do-not-transfer-2026-08-16.md
  - ram-frontier-qwen3-30b-2026-08-12.md
  - where-we-stand-vs-llamacpp-2026-08-16.md
  - ../backlog/bigger-machine-prompt.md
---

# The V4-Flash frontier, and the price of 20 tok/s

**Question**: can this project reach 20 tok/s on DeepSeek-V4-Flash (144 GB) on
this machine, and if not, on what machine?

**Answer, measured**: no, and not on any CPU. **With every expert resident —
infinite RAM, zero disk — this engine on this CPU tops out at 1.19 tok/s**,
because 0.84 s of every token is work that does not touch the disk at all.
Buying enough RAM to hold the whole model is worth **2.9x** (0.42 → 1.19), not
48x. 20 tok/s needs a token in 50 ms, which is **17x below the fixed cost alone**,
and separately needs **67.7 GB/s** of sustained bandwidth to the expert weights.
That is a GPU-memory specification, not a code change and not a RAM purchase.

Everything below was measured on 2026-08-16 in one session: i7-13650HX,
15.71 GiB RAM, RTX 3050 6 GB, `DeepSeek-V4-Flash-UD-Q4_K_XL` across five shards.

## How the frontier was swept without more RAM

This machine cannot be given more memory, but it can be given **less**. A balloon
process commits and touches N GiB — writing every page, because .NET commits
lazily and an untouched allocation is an imaginary balloon — and Chaos reads the
reduced free RAM at start and sizes its resident block accordingly. That turns an
unmeasurable axis into a measurable one, and the curve's *shape* is what
extrapolates.

Interleaved, not blocked: a whole pass over every balloon size, then another.
Three runs at one point followed by three at the next would have returned a slope
that was really a clock.

| balloon | free GiB | resident GiB | spill GiB | n | median tok/s | spread | s/token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.48 | 7.38 | 0.00 | 3 | **0.411** | 1.9% | 2.40 |
| 2 | 8.44 | 6.16 | 1.22 | 3 | 0.352 | 4.8% | 2.80 |
| 4 | 6.35 | 4.09 | 3.28 | 3 | 0.278 | 3.2% | 3.60 |
| 6 | 4.32 | 2.04 | 5.33 | 3 | 0.221 | 2.7% | 4.50 |

Least squares on seconds per token against spilled GiB:

```
t = 0.395 s/GiB * spill + 2.353 s        R^2 = 0.997
```

Two things fall out. The spilled always-read weights are re-read at
**2.53 GiB/s**, close to the drive's 2.74 GiB/s ceiling — they are read as a
per-block prefetch, which is a friendlier access pattern than the scattered
six-slice expert gather. And the intercept, 2.353 s, is a token with the
always-read weights fully resident and the experts still streaming.

Spreads of 1.9-4.8% are unusually tight for this machine, because the balloon
controls the variable that normally moves everything else.

## Where a token actually goes

Same session, `CHAOS_BLOCK_TIMING=1`, always-read weights fully resident
(7.38 GiB), median of four generated tokens:

| | s | what it is |
|---|---:|---|
| expert slice read | **1.56** | 3.15 GiB of distinct experts at 2.02 GiB/s |
| block work, not disk | 0.71 | attention, dense, the routed matmuls, graph eval |
| outside blocks | 0.13 | head, sampling, embedding |
| **token** | **2.40** | 0.416 tok/s |

So **F, the disk-independent cost, is 0.84 s per token.**

Note the expert read is *faster* here (2.02 GiB/s) than in a run with a 1.53 GiB
shortfall (1.65 GiB/s). Memory pressure was slowing the read itself, which is a
second, indirect cost of not fitting — the spill penalty in the table above is
not only the re-read.

### F is at its floor, not a bad thread choice

The whole conclusion rests on F, so it was attacked directly. At full residency,
one run each:

| `-t` | tok/s | s/token | expert read s | block work, not disk | compute s |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.383 | 2.6 | 1.54 | 0.92 | 0.54 |
| **4** | **0.411** | **2.4** | 1.54 | **0.78** | **0.43** |
| 8 | 0.381 | 2.6 | 1.55 | 0.95 | 0.51 |
| 16 | 0.346 | 2.9 | 1.55 | 1.27 | 0.69 |

`-t 4` is the optimum, matching what `CLAUDE.md` already recorded, and the fixed
work is worse on both sides of it. The expert read is flat at 1.54-1.55 s
regardless — it is the drive, as expected. **F is a floor on this CPU, not an
artefact of a knob left in the wrong place.**

## The frontier, extrapolated

With `f` the fraction of the expert bank resident, the model that the data
supports is the simplest one that could be true:

```
t(f) = (1 - f) * 1.56 s  +  0.84 s
```

The expert matmul's own DRAM traffic is already inside the 0.84 s — at `f = 1`
the weights are read from RAM by the matmul rather than from disk by the reader,
and that cost is what the fully-resident case already pays.

Expert bank ≈ 144 GB container − 7.9 GB of always-read weights ≈ **136 GB**.

| machine RAM | expert residency | tok/s |
|---:|---:|---:|
| 16 GB (this one) | ~3% | **0.42 measured** |
| 32 GB | ~15% | 0.46 |
| 64 GB | ~38% | 0.55 |
| 128 GB | ~85% | 0.93 |
| 160 GB | 100% | **1.19 (the ceiling)** |

**The curve is worth reading as a purchasing decision.** Going from this laptop
to a 64 GB machine buys 1.3x. Going all the way to holding the entire 144 GB
model in RAM buys 2.9x. The frontier is real, it rises, and it asymptotes
somewhere nobody wants to be.

This is the first time this curve has been measured for a model of this size, and
it is only measurable by an engine that owns residency — `mmap` cannot be told to
use exactly N GiB.

## What 20 tok/s actually requires

20 tok/s is a **50 ms** token. Against that budget:

1. **F = 0.84 s is 17x over, by itself.** No amount of memory touches it. This
   is the finding that closes the question for CPU inference: even with the whole
   model resident, this class of CPU is 17x short.
2. **The expert weights need 67.7 GB/s sustained.** 3.15 GiB of distinct experts
   per token × 20 tokens per second = 63.1 GiB/s. Dual-channel DDR5 is at roughly
   this figure *in theory* and below it in practice, so even a fully-resident CPU
   machine is at or past its memory-bandwidth limit before any arithmetic
   happens. HBM is three orders of magnitude clear of it.
3. Therefore **20 tok/s means the model resident in GPU memory** — about 144 GB
   of it, with the per-token graph evaluated in under 50 ms.

**Flagged as arithmetic, not measurement**: point 3 rests on a premise this
project has not tested. What *is* measured is that the GPU is **4.3x slower** on
streaming MoE (2.61 → 0.61), because the experts cross PCIe every token.
Resident-in-VRAM is a different regime and the number for it does not exist yet.
Do not quote a GPU V4-Flash figure until someone runs one — that is precisely the
mistake this project has made three times.

## What this closes, and what it opens

**Closed**: "20 tok/s on V4-Flash" as an engineering target for this codebase on
this hardware. It was already closed on the byte-reduction side
(`v4flash-has-no-slack-2026-08-10.md`: 79 MB/token needed, 3288 read). This
closes it on the other side too — the fixed cost alone forbids it — and replaces
"it needs the active weights to stop coming from disk" with a number: **1.19
tok/s is what stopping the disk entirely is worth.**

**Open**: the same decomposition on a machine with real memory and a real GPU.
The two quantities to bring back are `F` on that CPU and `F` with the model
resident on that GPU, because the whole question reduces to them.
`../backlog/bigger-machine-prompt.md` is the session prompt for it.

## Method notes, each bought with a failure in this session

- **A filter that truncates is indistinguishable from a regression.** A test
  count read 374 instead of 570 because the command piped `cargo test` through
  `tail -40` and there were 50 result lines. The same mistake then truncated a
  block-timing log to 29 blocks and produced a decomposition that did not add up.
  Both were caught only by a number disagreeing with one already measured.
- **Windows PowerShell 5.1 wraps a native executable's stderr in an ErrorRecord**
  — even with plain `2> file` — and with `$ErrorActionPreference = 'Stop'` that
  aborts the script on the first run. Chaos writes its status lines to stderr, so
  two attempts at this sweep died before producing a row. The sweep is Bash.
- **`git checkout main` fails when main is checked out in another worktree**, and
  `2>/dev/null` hid it; the fast-forward then landed on the *ticket branch*. The
  "did the merge bring the new files?" check passed for the wrong reason, because
  the branch had them. Check `git branch --show-current`, not the files.
- Free RAM drifted 8.05 → 10.45 GiB between the first baseline and the sweep,
  with no action taken — which is exactly why every arm here was re-measured
  inside the sweep instead of compared against the earlier figure.
- **An inference from one counter was wrong and the sweep corrected it.** The
  `dense` phase reads 0.01 s per token against a warning that predicts ~1.0 s for
  the shortfall, which looked like the warning being 100x pessimistic. The sweep
  says the spill really does cost, 0.395 s/GiB — so the cost is not in that
  counter, and a single phase timer was not evidence about a whole-token effect.
