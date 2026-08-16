# Overlapping expert reads with compute is worth ~1.03x — built, measured, reverted

**2026-08-16.** Qwen3-30B-A3B-Q4_K_M, i7-13650HX, 15.7 GiB. Generation.
Alternating runs in one session.

After parallel experts landed, the phase breakdown of a 16-token run looked like
an obvious next win:

```
time: 3.2s disk, 0.4s qkv, 0.4s attention, 0.1s ffn, 1.7s expert compute,
      0.0s slice copies, 0.0s kv build, 0.2s other
```

Disk and expert compute are 3.2 s and 1.7 s of an 8.3 s run and appear
**additive**. Overlap them and the larger should absorb the smaller — arithmetic
says ~1.25x.

**It is worth 1.03x.**

| | generation tok/s |
|---|---|
| read everything, then compute | 3.19 `[3.12 3.19 3.20 3.39]` |
| read chunk k+1 while 0..k compute | 3.28 `[3.19 3.28 3.38 3.44]` |

Pipelined ahead in **3 of 4** pairs. Output byte-identical. That is inside the
noise, and it was reverted.

## Why the arithmetic was wrong

**The cache absorbs 64–70% of expert reads.** "3.2 s disk" is the total time in
the read path, not time spent waiting on the drive — most of it is cache lookups
and `Arc` handling that no amount of overlap removes. The genuinely
disk-blocked fraction is perhaps a third of that, so the ceiling was never 1.25x.

And the trade is not free: chunking the fetch into four gives up read
concurrency to buy overlap. `read_slices_parallel` issues all of a block's
slices at once across eight pooled file handles; four sequential chunks of two
cannot use the queue depth the single call does. The overlap gained and the
concurrency lost very nearly cancel.

## The measurement that nearly did not happen

The first comparison was **pipelined-now against a remembered number from an
earlier session** — 3.41 against "3.86 before". That reads as a large
regression, and it is meaningless: by then the machine had been running 17 GiB
models back to back for hours, and *both* arms of an unrelated head-to-head were
declining across their five pairs (3.53 → 3.03 and 3.62 → 2.53).

So a `CHAOS_EXPERT_PIPELINE=0` toggle was added purely to get both paths into
one alternating session, and only then did the honest number appear. **This
machine drifts by more than the effect being measured.** Any change worth less
than ~10% here needs both arms in one session, alternating, or it cannot be
assessed at all.

## What stays

Nothing. The pipeline, the toggle, and the chunked fetch are all removed; the
expert path reads a block's slices in one call and computes them across four
workers, which is what `parallel-experts-2026-08-16.md` shipped.

**An unmeasurable win is not worth a concurrency structure.** This project has
deleted five dead forward paths and the rule applies to read paths too.

## What this says about the remaining disk cost

Disk is still the largest single phase, but it is not 3.2 s of *waiting*. Making
it smaller now means reading less or caching better, not scheduling differently
— and the cache budget sweep run the same day says caching better is also done:
2/4/6/8 GiB gives 2.22 / 2.66 / 3.45 / 3.43 tok/s, so it plateaus at 6 GiB and
the default already sits on the plateau.
