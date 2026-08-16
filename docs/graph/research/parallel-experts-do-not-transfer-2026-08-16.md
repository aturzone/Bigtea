# Parallel experts do not transfer to V4-Flash — the arithmetic is under 5% of a token

**2026-08-16.** DeepSeek-V4-Flash-UD-Q4_K_XL, 144.4 GiB across five shards,
i7-13650HX (20 logical cores), 15.7 GiB RAM, RTX 3050 6 GB, NVMe. All numbers
below were taken in one session.

`parallel-experts-2026-08-16.md` gained **1.29x on expert compute and 1.10x end
to end** on the Qwen3 streaming path by running N whole experts side by side
with one ggml thread each. V4-Flash runs a different forward pass
(`deepseek4_forward.rs`) and never received that change, which made it the
obvious next port.

**It cannot pay here, and the ceiling is a measurement rather than an argument:
the entire routed expert arithmetic is under 5% of a V4-Flash token.**

## What a V4-Flash token is actually made of

`BIGTEA_BLOCK_TIMING=1`, summed over all 43 blocks of one pass. The block builds
one graph and evaluates it in a single `compute` at the end, so every other
phase timer measures graph *construction* — plus, inside `ffn`, the expert slice
read. `compute` is the only line where arithmetic happens, and it had been
folded into the residual, which is why this split had never been written down.

| phase | generation, per token | share |
|---|---:|---:|
| expert slice read (disk) | **1.70 s** | **67%** |
| `compute` — attention, both FFNs, the head | 0.44 s | 17% |
| `tail` — routing, which forces an early compute | 0.40 s | 16% |
| dense binds (resident + prefetched) | 0.01 s | <1% |
| **total** | **2.52 s** | |

Prefill at 5 tokens is the same shape: read 5.51 s, compute 1.39 s, tail 0.46 s.

## Costing the expert matmul by subtraction

A throwaway build kept the disk read and returned only the shared expert,
dropping the three routed `mul_mat_id` calls. Output is wrong — that is the
point; it isolates their cost. Both arms alternated in one session:

| | generation tok/s | block `compute` |
|---|---:|---:|
| with the routed experts | 0.370 | 0.44 s |
| routed matmuls dropped | 0.388 | 0.43 s |

**0.01 s of 0.44 s**, and 4.9% end to end — and that 4.9% is an *upper* bound,
because the shorter arm also builds a smaller graph. Perfect parallelisation of
the expert matmuls, at zero cost and zero overhead, would therefore be worth at
most **1.05x**, and realistically nothing measurable on a machine that drifts by
3%.

The toggle was removed rather than shipped. A flag that silently produces wrong
output is the failure mode this project is most expensive at.

## And there is nothing to gather, so the mechanism is absent too

The port had a second premise: that this path suffers the same problem the
`mul_mat_id` route hit on Qwen3, where making the selected `Arc<[u8]>` experts
contiguous cost ~1.02 GB/token. It does not.

`read_expert_slices` packs the selected slices **contiguously as it reads them**
into one `SkewedBuf`, which is then bound as a `[ne0, ne1, n_unique]` stack and
run through three `mul_mat_id` calls. So this path already has the batched form,
and it got it for free: the copy that killed the Qwen3 version is the read that
had to happen anyway. There are no N subgraphs here to run side by side.

Two independent reasons, then — no headroom and no mechanism.

## The measured baseline this replaces guesswork with

Three alternating rounds, same session, `--temp 0`, greedy:

| | run 1 | run 2 | run 3 | median |
|---|---:|---:|---:|---:|
| generation, 7 tokens | 0.387 | 0.396 | 0.400 | **0.396 tok/s** |
| prefill, 51 tokens | 1.53 | 1.54 | 1.52 | **1.53 tok/s** |
| prefill, 5 tokens | 0.62 | 0.63 | 0.63 | **0.63 tok/s** |

Spread is 3.3% on generation and 1.3% on prefill, which is tighter than this
machine usually manages and is worth stating so the next session knows what
counts as a real move. Machine state at the time: 8.6 GiB available, the resident
budget took 6.53 GiB, and **0.85 GiB of the always-read set did not fit** and was
re-read every token by the background prefetch.

Command lines, verbatim:

```
bigtea-run DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf "The capital of France is" -n 8
bigtea-run DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf -f prompt51.txt -n 1
```

## Where the remaining headroom is, and why it is not reachable in code

67% of a token is the expert slice read: 3.19 GiB moved per token at **1.88
GiB/s**. `bigtea-iobench` on a shard of this same model, 256 scattered 4 MiB
slices per pass:

```
 THREADS    SHARED GiB/s      PER-HANDLE        GAIN
       1            1.63            1.67       1.03x
       2            2.12            2.44       1.15x
       4            2.08            2.74       1.32x
       8            2.10            2.73       1.30x
      16            2.17            2.64       1.22x
      32            2.07            2.67       1.29x
```

The drive tops out at **2.74 GiB/s and stops climbing at four handles**, so the
existing 8-handle pool is not the limit and raising it buys nothing — that
question is now closed with a number.

The gap between 1.88 and 2.74 GiB/s is the **per-block barrier**: a block reads
76 MiB, computes, and only then can the *next* block's routing be evaluated,
which is what decides the next block's addresses. The drive therefore alternates
between a short burst at queue depth 6 and an idle stretch, and nothing can be
queued during the idle stretch because nothing knows what to ask for yet.

That is the same wall `v4flash-has-no-slack-2026-08-10.md` reached from the byte
side, arrived at from the latency side. Within a block the only thing that could
overlap the read is the expert arithmetic, and this note has just measured that
at under 5%.

## Related

- [[parallel-experts-2026-08-16]] — the change this failed to port, and the path
  where it does pay.
- [[expert-read-overlap-does-not-pay-2026-08-16]] — the same overlap idea on the
  Qwen3 path, 1.03x, reverted. It failed there because the cache absorbed 64–70%
  of expert reads; it fails here for the opposite reason, that there is almost no
  compute to hide.
- [[v4flash-has-no-slack-2026-08-10]] — the byte budget, closed.
- [[the-plateau-was-ours-2026-08-10]] — where the 8-handle pool came from.
