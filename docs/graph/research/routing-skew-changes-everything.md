---
topic: V4-Flash's router is violently skewed — 25% of experts serve 97.8% of tokens, and that makes 20 tok/s reachable
status: open
links: [v4flash-vs-llamacpp-2026-08-07.md, v4flash-28-tokens-per-second.md, roadmap-after-v0.0.1.md]
---

## The measurement

`BIGTEA_ROUTING=1`, DeepSeek-V4-Flash, a real coding prompt, all 43 layers,
every routing decision counted:

```
top-N experts per layer   share of selections   resident cost
    1   (  0.4% of model)      12.1%              0.54 GiB
    4   (  1.6%)               36.2%              2.14 GiB
    8   (  3.1%)               52.9%              4.28 GiB
   16   (  6.2%)               70.4%              8.57 GiB
   32   ( 12.5%)               86.1%             17.13 GiB
   64   ( 25.0%)               97.8%             34.27 GiB
  128   ( 50.0%)              100.0%             68.53 GiB

uniform routing would give top-16 = 6.2%.  Measured: 70.4%.
chi-square against uniform = 7805   (uniform on 255 d.o.f. ≈ 255)
```

**One expert in 256 — half a gigabyte — absorbs 12% of all selections. Sixty-four
of them absorb 97.8%.**

## What this destroys

Every speed estimate in this project assumed the router spreads evenly, so all
137 GiB of experts were equally cold and none was worth holding. On that
assumption, 20 tok/s needed 64 GiB/s of disk against a 2.37 GiB/s drive — **27x
short, declared impossible** in `roadmap-after-v0.0.1.md` hours before this was
measured.

**The assumption was wrong, and so was the conclusion.** With a cache sized to
the hot set, the bytes that must come off disk collapse:

| cache | hit rate | bytes/token | disk floor |
|---|---:|---:|---:|
| 4.28 GiB (top-8) | 52.9% | 1548 MiB | 1.6 tok/s |
| 8.57 GiB (top-16) | 70.4% | 973 MiB | 2.5 tok/s |
| 17.13 GiB (top-32) | 86.1% | 457 MiB | 5.3 tok/s |
| **34.27 GiB (top-64)** | **97.8%** | **72 MiB** | **33.6 tok/s** |

And compute is not the wall either:

```
6 experts x 43 layers x 2 matmuls x 4096x2048 x 2 flops = 8.7 GFLOP per token
at 239 GFLOPS — already measured on this project's own expert matmuls —
  = 36 ms/token = 27 tok/s
```

**Both floors sit above 20 tok/s.** 20 tok/s for a 144 GB model is not a physics
violation. It is a cache-sizing problem.

## The correction that matters commercially

The old claim was "20 tok/s needs a machine that holds the model — ~150 GB of
RAM." That is **wrong by 3x**. It needs the *hot set*, not the model:

```
34.27 GiB  hot experts (97.8% of selections)
 7.38 GiB  always-read weights
~6 GiB     KV cache, arenas, OS
--------
~48 GiB    a normal desktop, not a server
```

**Running a 144 GB model at 20+ tok/s on a 48 GB desktop** is a real claim, and
no engine does it — llama.cpp mmaps the container and lets the kernel's LRU
decide, which is exactly the policy this project has already measured as the
worst available for a cyclic expert scan.

## What it means on *this* laptop, 15.7 GiB

Not 20 tok/s. With 10.5 GiB free, 7.38 GiB goes to always-read weights and about
**3.1 GiB is left for experts** — between top-4 and top-8, so roughly a 45% hit
rate:

```
bytes/token   3.21 GiB x 0.55 = 1.77 GiB
disk floor    1.77 / 2.37 = 0.75 s/token = 1.34 tok/s
```

**~1.3 tok/s against llama.cpp's measured 0.31 — about 4x ahead.** That is the
win on this hardware, and it is worth having: the current implementation manages
0.064 tok/s, so the cache is worth ~20x here before any other change.

## Caveats, stated so this is not over-read

- **One prompt.** A coding prompt, 43 layers, a few hundred routing decisions per
  layer. The skew is far too large to be noise (chi-square 7805 against ~255) but
  the *shape* may shift with domain. **Before building the cache, re-measure on
  several prompts across different subject matter.** If the hot set is
  prompt-dependent rather than global, the cache must be warmed adaptively rather
  than pinned — which is a different and harder design.
- **Layers 0-2 route by token id**, not by learned gating, so their skew reflects
  the token distribution rather than the router. They are 3 of 43 and cannot
  explain a chi-square this size, but they should be excluded from the next
  measurement.
- The hit-rate-to-bytes conversion assumes a miss costs a full slice read, which
  it does.
- **This project has a written warning that cache hit rate is not a success
  metric** — past ~6 GiB on Qwen3 the expert cache reached 71% hits and was the
  *slowest* configuration measured, because cached bytes got paged out and a
  "hit" became a page fault in disguise. **That is a real risk here and the
  reason the cache must own its memory rather than rely on the page cache.**
  Bigtea already owns its allocations, which is precisely why it can do this and
  an mmap-based engine cannot.

## Next, in order

1. **Re-measure skew across several prompts and domains**, excluding layers 0-2.
   One day. It decides whether the hot set is global (pin it) or prompt-dependent
   (warm it adaptively).
2. **Frequency-gated expert cache sized from the probe.** The policy already
   exists in `stream.rs` for Qwen3 and took hit rate 17% → 70% there; it has
   never been wired into the V4-Flash path.
3. Then the I/O/compute overlap from `roadmap-after-v0.0.1.md` T1, which is worth
   the remaining serial gap once the cache has removed most of the reads.

## The honest headline

**On this laptop: ~1.3 tok/s, roughly 4x llama.cpp — not 20.**
**On a 48 GiB desktop: 20-27 tok/s, and nothing else does that.**

Neither number is measured yet. Both floors are arithmetic on measurements that
are.
