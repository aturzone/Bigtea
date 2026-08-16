---
topic: what a residency shortfall actually costs per token, and how the runner should estimate it
status: resolved
links:
  - v4flash-ram-frontier-2026-08-16.md
  - the-plateau-was-ours-2026-08-10.md
  - r2-overlap-2026-08-11.md
  - threads-were-never-plumbed-2026-08-10.md
---

# The shortfall warning was 1.6x pessimistic, and the obvious fix was worse

**Question**: `chaos-run` tells a user that N GiB of always-read weights will be
re-read on every token and what that costs, then names the processes to close.
The cost came from `missing / report.bytes_per_sec()` — the rate the *initial
load* achieved. Is that the right denominator, and if not, what is?

**Answer, measured**: it is the wrong denominator and it **overstated the cost by
about 1.5x**. The load is essentially one stream at 1.6-2.0 GB/s; the spill comes
back across the eight-handle reader pool. Two fresh balloon sweeps put the true
marginal cost at **0.410-0.418 s/GiB**, confirming
[v4flash-ram-frontier-2026-08-16.md](v4flash-ram-frontier-2026-08-16.md)'s 0.395
from a different session. What ships is neither that constant nor the first
instrument tried: the runner now **re-reads a 256 MiB sample of the spilled
tensors themselves, through the same pool, and times it**, which lands within 2%
of the swept rate on the mean.

Machine: i7-13650HX, 15.71 GiB RAM, `DeepSeek-V4-Flash-UD-Q4_K_XL` across five
shards. Everything below was measured 2026-08-16/17 in one session.

## The slope reproduces across three sessions

Two sweeps, each 4 balloon sizes x 3 interleaved passes, `-n 5 --temp 0`, driven
by `scripts/spill-sweep.sh` (new — the earlier sweep was done by hand and could
not be re-run):

| sweep | fit | R² | implied re-read rate |
|---|---|---:|---:|
| 2026-08-16, by hand | `t = 0.395*spill + 2.353` | 0.997 | 2.53 GiB/s |
| this session, build A | `t = 0.418*spill + 2.394` | 0.982 | 2.39 GiB/s |
| **this session, build B (shipped)** | **`t = 0.410*spill + 2.204`** | **0.997** | **2.44 GiB/s** |

**The slope is 0.41 ± 0.01 s/GiB across three independent sweeps.** The
intercepts differ (2.20 vs 2.35) because free RAM differed between sessions; only
the slope is being claimed. Build B's twelve rows, medians of three:

| spill GiB | tok/s | s/token | spread |
|---:|---:|---:|---:|
| 1.54 | 0.350 | 2.9 | 1.1% |
| 3.05 | 0.295 | 3.4 | 0.3% |
| 4.57 | 0.242 | 4.1 | 0.8% |
| 6.07 | 0.212 | 4.7 | 0.0% |

Spreads under 1.1% — tighter than anything else measured on this machine,
because the balloon pins the variable that normally moves everything.

## The obvious fix does not work: prefetch wall time is occupancy, not cost

The first instrument was the one that suggests itself — accumulate bytes and
elapsed time inside `prefetch_dense_via`, the single funnel every spilled read
passes through, and report the rate. **Built, measured, reverted.** It read
**0.80 GiB/s**, i.e. 1.25 s/GiB, which is not a 1.6x overestimate corrected but a
**3x overestimate introduced** — worse than the bug.

The cause is R2 overlap, which is on by default. The background prefetch for
block N+1 runs on **2 of the 8 handles while block N computes**, so its wall
clock is stretched to cover the block, and dividing bytes by that measures how
long the thread was *occupied*, not what the bytes cost. Disabling the overlap
confirms it — same binary, same prompt:

| | measured prefetch rate | implied s/GiB |
|---|---:|---:|
| `CHAOS_PREFETCH_OVERLAP=1` (default) | 0.80 GiB/s | 1.25 |
| `CHAOS_PREFETCH_OVERLAP=0` | 1.99 GiB/s | 0.50 |

**A counter placed inside an overlapped path measures the overlap, not the
work.** The same trap as the `dense` phase timer reading 0.01 s per token in the
previous session's notes: both are correct readings of a quantity that is not
the cost.

## What shipped, and why it is not a proxy

`chaos_model::measure_spill_rate` reads a 256 MiB sample **of the spilled tensors
themselves** — the exact bytes that will be re-read — round-robin across the same
eight reader handles, at the same alignment and skew, and times it. It runs only
when there is a shortfall, costs ~0.1 s, and returns `None` rather than a guess
when the spill is too small to sample.

It is not a model of the operation; it is the operation with the concurrency
removed. That distinction is what separates it from the thread tuner this project
deleted, whose 150 ms DRAM microbenchmark had to be corrected until it agreed
with the objective.

**One sizing decision was wrong and the sweep caught it.** Build A capped every
read at 16 MiB to bound the transient footprint. Whether a spilled tensor
happened to exceed that cap changed the read size and therefore the throughput,
and the rate swung **1.54-2.65 GiB/s** across twelve runs — non-monotonic in
spill, mean 2.06 against a true 2.39, 14% low. Build B reads **whole tensors**,
as the prefetch does, bounding the sample by its total instead:

| | rate range over 12 runs | mean | swept truth | error on the mean |
|---|---|---:|---:|---:|
| build A, 16 MiB chunks | 1.54-2.65 GiB/s | 2.06 | 2.39 | 14% low |
| **build B, whole tensors** | **2.10-2.80 GiB/s** | **2.40** | **2.44** | **2% low** |

**The buffer allocation costs nothing measurable.** Build A timed the reads with
and without it and the two collapsed — 1.87 against 1.86, 2.52 against 2.52 — so
the two-ended range that was going to be printed said nothing, and it was
removed. That is a negative worth keeping: the transient allocation is not part
of the spill cost.

## What the user sees

Same twelve runs, medians per point. `truth` is `0.410 * spill` from the fit:

| spill GiB | old, from load rate | **new, measured** | truth | old error | new error |
|---:|---:|---:|---:|---:|---:|
| 1.54 | 0.97 s | **0.7 s** | 0.63 s | 1.54x | 1.11x |
| 3.05 | 2.00 s | **1.4 s** | 1.25 s | 1.60x | 1.12x |
| 4.57 | 2.78 s | **1.7 s** | 1.87 s | 1.49x | 0.91x |
| 6.07 | 3.43 s | **2.3 s** | 2.49 s | 1.38x | 0.92x |

**A consistent 1.5x overestimate becomes a mean of 1.02x with ±10% scatter**, and
the scatter now falls on both sides instead of always the same one. The case in
the original report — 1.53 GiB spilled, printed as `~1.1s` — now prints `~0.7s`
against a swept 0.63.

This matters because the next line is *"closing these would free up to N GiB"*.
The old number oversold closing an editor by half.

## Honest limits

- **It is an estimate with a measured denominator, not the marginal cost.** The
  calibration cannot reproduce two things a real pass does: it does not share the
  drive with the expert slice reads, and it does not get part of itself hidden by
  the R2 overlap. Those pull in opposite directions and roughly cancel here; on a
  machine where they do not, this drifts and the sweep is the only arbiter.
- **One model, one drive, one machine.** The point of the change is that the
  number is now taken on whatever machine is running, but the *agreement* between
  the calibration and the sweep is only established here.
- **Only the slope is claimed.** Absolute tok/s and the intercept moved between
  sweeps with free RAM; compare rows within one table.
- The sweep is `-n 5` on a 5-token prompt. The distinct-expert working set grows
  with generation length, so the intercept would rise with `-n`; the slope is a
  property of the spill and should not.
- **`chaos-serve` still prints nothing about a shortfall.** It loads a resident
  set the same way and never calls the warning at all.

## Method notes, each bought with a failure here

- **A `.ps1` that will not run looks exactly like a balloon that did not
  inflate.** This box refuses unsigned scripts, and the refusal is a security
  error on stderr with no non-zero exit — the first sweep attempt would have
  produced a full set of rows against a machine that was never made short.
  `-ExecutionPolicy Bypass`, and the balloon writes a marker file that the sweep
  waits for rather than sleeping a fixed time.
- **An untouched allocation is an imaginary balloon.** .NET commits lazily, so
  the balloon writes one byte per 4 KiB page; without that, free physical memory
  does not move.
- **The research node cited in the ticket appeared not to exist.** It had merged
  to `main` (PR #92) between two `git` calls in this session, and the first check
  ran against a stale `origin/main`. `git merge-base --is-ancestor` is only as
  current as the last fetch.
- Two `cargo test` counts disagreed (570 vs 571) because a test was added and
  then reverted with the approach it tested. The count is the cheapest check that
  a refactor did not quietly drop coverage.
