---
topic: `-t` reached exactly one architecture, and once it reached the rest the default turned out to be the worst setting available — 1.66x and 1.69x thrown away. Generation and prefill want opposite thread counts
status: measured
links: [lts-parity-criteria.md, qwen3-4b-vs-llamacpp-2026-08-10.md, the-plateau-was-ours-2026-08-10.md]
---

Two findings, and the first is what made the second visible.

## 1. The flag did nothing

`chaos-run -t N` set `CHAOS_THREADS`, and **only `deepseek4_forward.rs` ever
read it**. Every other architecture — qwen3, qwen3moe, llama, phi3, gemma2 —
computed its own count from `available_parallelism()` and ignored the flag. A
second, differently spelled variable (`CHAOS_EXPERT_THREADS`) was read in
`StreamingRunner::new` into a binding named `_threads` and discarded, so that
override had never worked either.

The comment above the flag said:

> Set once, read by every graph evaluation. A flag that only reached some of
> them would make `-t` look ineffective on exactly the paths that matter.

The code did the thing the comment warned about. **This is the second
comment-versus-code mismatch found in two days** — the other claimed one
`compute` materialised Q, K and V while the code called it three times, and
fixing that was worth 1.30x.

### How it surfaced

Not by reading the code. By running:

```
$ chaos-run Qwen3-4B-Q4_K_M.gguf "..." -n 16 -t 1
    time: 0.0s disk, 1.0s qkv, 0.7s attention, 1.7s ffn, ...
$ chaos-run Qwen3-4B-Q4_K_M.gguf "..." -n 16 -t 20
    time: 0.0s disk, 1.0s qkv, 0.7s attention, 1.7s ffn, ...
```

**Bit-identical phase timings at 1 and 20 threads.** No real matmul does that.
A thread sweep run earlier in the same session had produced 4.07/4.00/4.31/4.67
tok/s at 2/4/8/20 threads and been read as "threads are not the lever" — it was
six measurements of the same configuration, and the noise was mistaken for a
result. **A sweep whose knob is disconnected looks exactly like a flat
response.** The check that would have caught it costs nothing: confirm the knob
moves *something* before concluding it moves nothing.

## 2. Once connected, "all cores" was the worst setting available

Machine: i7-13650HX, **14 physical cores (6 P + 8 E), 20 logical**. Three
repetitions, generation only, `-n 32`.

| threads | Qwen3-4B | Llama-3.2-1B |
|---:|---:|---:|
| 1 | 5.26 | — |
| **2** | **7.64** | **21.95** |
| 4 | 7.51 | 21.45 |
| 6 | 7.09 | 19.79 |
| 8 | 6.24 | 16.78 |
| 14 | 4.84 | 14.50 |
| 20 — *the old default* | 4.49 | 12.22 |

**The default cost 1.70x on one model and 1.80x on the other.**

Generation streams every weight once per token and does almost no arithmetic per
byte, so it saturates DRAM long before it runs out of cores; past that point
threads only contend, and the E-cores make it worse. This is a property of the
hardware, not of Chaos — llama.cpp shows the same curve on the same machine:

```
$ llama-bench -m Qwen3-4B-Q4_K_M.gguf -n 128 -p 0 -r 3 -t 1,4,8,20
| threads |  test | t/s          |
|       1 | tg128 | 5.45 ± 0.07  |
|       4 | tg128 | 9.16 ± 0.43  |
|       8 | tg128 | 8.30 ± 0.06  |
|      20 | tg128 | 5.85 ± 0.03  |

$ llama-bench -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 128 -p 0 -r 3 -t 2,4,6,8,20
|       2 | tg128 | 24.38 ± 0.19 |
|       4 | tg128 | 27.85 ± 1.98 |
|       8 | tg128 | 20.46 ± 0.09 |
|      20 | tg128 | 13.65 ± 0.13 |

build: daef2b3 (1)
```

## 3. Prefill wants the opposite, which is why llama.cpp has two flags

Same model, same binary, 519-token prompt, `-n 1`:

| threads | Qwen3-4B prefill |
|---:|---:|
| 4 | 47.4 |
| 8 | 70.9 |
| 14 | 78.2 |
| 20 | **81.5** |

Prefill multiplies a whole block at once, so it is compute-bound and scales with
cores. **The best generation count is close to the worst prefill count.** One
`-t` cannot serve both; llama.cpp spells the second one `-tb` /
`--threads-batch` and now so does Chaos. The count follows the *token count* of
the step rather than the call site, because `forward_cached` serves both phases.

## 4. A calibration that failed, and why it was deleted rather than tuned

First attempt at a default: a ~150 ms microbenchmark at load — stream a buffer
larger than L3 across `t` threads, take the smallest `t` that saturates DRAM.

It measures a real quantity and it is **not predictive of this one.** Six
consecutive runs of the same binary on the same model chose:

```
6, 8, 12, 12, 4, 6
```

while the true optimum was 2-4, and the spread it introduced was worse than the
bad default it replaced: **5.51-8.20 tok/s against 7.53-7.74 pinned.** A pure
read has no per-node barrier; a ggml graph has one per node, and that
synchronisation is most of what extra threads actually cost.

It was deleted rather than corrected. **A proxy that has to be adjusted until it
agrees with the objective is just the objective, measured badly.**

## 5. What shipped: tune on real tokens

`ThreadTuner` walks a ladder of thread counts over the **first few generated
tokens**, timing each where the cost actually occurs, and keeps the fastest. It
stops on two consecutive regressions greater than 5% — the curve has one peak
and falls monotonically after it — so it usually costs four tokens rather than
eight. Ties go to the smaller count, because over-threading costs 1.8x and
under-threading about 10%; the two errors are not worth the same.

`-t` still overrules it. An explicit flag is an instruction, not a hint.

### Interleaved A/B, same session, `-n 64`, three repetitions

Runs alternated so both configurations see the same thermal and background state.

| | tuned (default) | `-t 20` (old default) | |
|---|---:|---:|---|
| Qwen3-4B | 8.30 / 7.54 / 8.18 → **8.01** | 4.84 / 5.08 / 4.58 → 4.83 | **1.66x** |
| Llama-3.2-1B | 20.88 / 20.65 / 18.62 → **20.05** | 11.91 / 12.05 / 11.70 → 11.89 | **1.69x** |

## The scoreboard, stated two ways because they say different things

llama.cpp's default on this machine is 10 threads, which is also off its own
peak:

```
$ llama-bench -m Qwen3-4B-Q4_K_M.gguf -n 64 -p 0 -r 3
| CPU | 10 | tg64 |  6.52 ± 0.33 |
$ llama-bench -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 64 -p 0 -r 3
| CPU | 10 | tg64 | 20.91 ± 0.65 |
```

| generation | Chaos | llama.cpp | verdict |
|---|---:|---:|---|
| Qwen3-4B, **both at default** | **8.01** | 6.52 ± 0.33 | **1.23x ahead** |
| Llama-3.2-1B, **both at default** | 20.05 | 20.91 ± 0.65 | 1.04x behind — parity |
| Qwen3-4B, **both hand-tuned** | 7.64 (t=2) | 9.16 ± 0.43 (t=4) | **1.20x behind** |
| Llama-3.2-1B, **both hand-tuned** | 21.95 (t=2) | 27.85 ± 1.98 (t=4) | **1.27x behind** |

**Both rows are true and neither may be quoted alone.** Out of the box Chaos is
ahead on Qwen3-4B because it measures the machine and llama.cpp uses a fixed
default. Given the same care on both sides, **llama.cpp is still faster**, by
1.20x and 1.27x. The auto-tuning is a real advantage for a user who types no
flags; it is not evidence that the engine is faster, and the underlying deficit
is unchanged from what was recorded before any of this — 1.23x — which is what
gives confidence the ratio is real rather than an artefact of the operating
point.

## MoE wanted ONE thread, and it is worth 2.4x

The prediction below — "MoE generation is disk-dominated, so threads should
matter far less" — was **wrong**, and the tuner is what caught it.

On Qwen3-30B-A3B the tuner chose 1 thread. That looked like a failure: on a
streaming model most of a token is disk, and how much varies per token with
cache hits and warming, so the signal could easily be swamped. The tuner now
subtracts read time and times only what the thread count can affect. **It still
chose 1, three runs in a row**, and a direct sweep says it is right:

| threads | Qwen3-30B gen | expert compute |
|---:|---:|---:|
| **1** | **2.88 tok/s** | 2.2 s |
| 2 | 2.54 | 2.5 s |
| 4 | 2.23 | 2.9 s |
| 8 | 1.80 | 3.6 s |
| 20 — *the old default* | 1.21 | 5.2 s |

Disk was flat at ~3.0 s throughout, so this is entirely compute. Each expert
matmul at one token is a 768x2048 matrix-vector, a layer's graph holds 24 of
them, and splitting each across 20 threads leaves ~38 rows per thread per
barrier. **The threads cost more than the work they do.**

End to end this took Qwen3-30B generation from **1.07 to 2.63 tok/s (2.46x)**.

### And it re-opened a competitive number in our disfavour

```
$ llama-bench -m Qwen3-30B-A3B-Q4_K_M.gguf -n 32 -p 0 -r 2 -t 1,4,10
|       1 | tg32 | 1.95 ± 0.64 |
|       4 | tg32 | 4.21 ± 0.28 |
|      10 | tg32 | 3.64 ± 0.22 |
```

The recorded reference was **2.16**; llama.cpp's own best is **4.21**. So
Qwen3-30B generation is **1.60x behind**, not the ~2x on record and nowhere near
the win that 2.63-against-2.16 would have been. Third stale competitor number
found this week by re-running the opposing command.

**llama.cpp peaks at 4 threads where we peak at 1**, which says its expert path
parallelises and ours does not. That is the concrete lead for the remaining
1.60x: give each barrier real work — batch the expert matmuls — instead of 24
tiny nodes per layer.

## V4-Flash has the same curve, and still has its old default

`deepseek4_forward.rs` reads `CHAOS_THREADS` directly and does **not** go
through the tuner, so the flagship model still defaults to every core. Swept
with `-t`, `"The capital of France is"`, `-n 4`:

| threads | V4-Flash generation |
|---:|---:|
| 1 | 0.331 tok/s |
| 2 | 0.378 |
| **4** | **0.380** |
| 8 | 0.346 |
| 20 — its current default | 0.296 |

**Fixed once r9 was merged in** — and it was *not* the one-line cap it looked
like. Capping `threads()` at 4 was the first attempt, and V4-Flash prefill
promptly lost 1.29x:

| V4-Flash, back to back | 4 threads | all cores |
|---|---:|---:|
| generation | **0.196** | 0.177 |
| prefill, 180 tokens | 2.24 | **2.89** |

So this file needed the same split as the dense path, not a cap. `threads()`
reads a batch size that `forward` sets — the single funnel both `prefill` and
`step` pass through — and resolves the two counts through `CHAOS_THREADS` and
`CHAOS_THREADS_BATCH`.

**This retires a line that was in `CLAUDE.md`**: "4/12/20 threads all cost the
same on a V4-Flash prefill". That was measured at **5 tokens**, where the pass
is almost entirely disk. At 180 tokens it is 2.24 against 2.89. *A measurement
taken at one prompt length is not a fact about the engine* — the same error as
scoring a hot set on the prompt it was chosen from.

### The version that made it slower

The first split called `std::env::var` inside `threads()`. That function runs at
every `ctx.compute` — thousands of times per token — and each call locks the
process environment and allocates a `String`. Generation fell to **0.267**,
*below* the 0.296 the change was meant to fix. Both counts are resolved once
now; the per-call cost is an atomic load and a branch.

### Absolute V4-Flash numbers drift, badly

The same `-t 4` vs `-t 20` comparison read **0.380 / 0.296** earlier in the day
and **0.196 / 0.177** after a dozen heavy runs, as the page cache for a 144 GB
container filled with the wrong things. The direction held both times; the
magnitude did not. **Only compare V4-Flash numbers within one session**, and
never against the published 0.374 tok/s, which was 47 tokens with a warm cache.

## What this does not measure

- **Any other machine.** Every number here is one laptop with one hybrid CPU and
  two memory channels. The *mechanism* (generation saturates DRAM, prefill
  scales with cores, and tiny per-node work makes threads a cost) is general;
  the optimum is not, which is precisely why the shipped answer is measured at
  run time rather than hardcoded.
- **Long context**, where attention grows and the balance may shift.

## Correctness

Thread count must not change output. Verified byte-identical at 2 and 20 threads
on all five verified dense architectures — qwen3, llama, phi3, gemma2 and
tinyllama (`"The capital of France is"`, `-n 12`, greedy). 229 tests pass.
