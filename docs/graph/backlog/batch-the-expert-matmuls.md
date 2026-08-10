---
topic: The scoped ticket for the remaining 1.60x on Qwen3-30B — batch the expert matmuls so a barrier has real work. Includes the arithmetic that says it pays, and the measurement that would say it does not
status: scoped, not started
links: [../research/threads-were-never-plumbed-2026-08-10.md, lts-parity-criteria.md]
---

## Where this came from

The thread tuner found that Qwen3-30B-A3B generation wants **exactly one
thread** — 2.88 tok/s against 1.21 at twenty, with expert compute going
2.2 s → 5.2 s as threads are added. **llama.cpp peaks at four threads on the
same model.** Its expert path parallelises and ours does not.

That is the whole of the remaining 1.60x deficit, and it is not a mystery: it
is what the graph looks like.

## What the graph looks like now

`expert_ffn_single` builds one graph per layer, and for each of the 8 selected
experts it does three `ws.bind` calls and three `mul_mat`s:

```
per layer:  8 experts x (bind gate, bind up, bind down)   = 24 binds
            8 experts x (mul_mat, silu, mul, mul_mat, scale, add) ~ 48 nodes
per token:  48 layers x that                              = 1,152 binds
```

Each `mul_mat` is `2048 x 768` against a **single column**. Split across 20
threads that is ~38 rows per thread, then a barrier. The threads cost more than
the work, which is exactly what the sweep shows.

## The arithmetic that says batching pays

Qwen3-30B-A3B: `n_embd 2048`, `n_ff_expert 768`, 8 experts used, 48 layers,
Q4_K at ~0.56 bytes/weight.

```
per layer per token   8 x 3 x (2048 x 768) x 0.56 B  =  21.1 MB
per token             x 48 layers                    =   1.01 GB
```

Measured expert compute at 1 thread is **2.2 s for 8 tokens = 275 ms/token**,
so the expert path currently runs at **3.7 GB/s**. The dense FFN on this machine
runs at **~13 GB/s** — at DRAM speed. **There is ~3.5x of headroom, and it is
not bandwidth.** It is per-node overhead: 1,152 binds and ~2,300 graph nodes per
token, each one tiny.

If batching got the expert path to DRAM speed:

```
copy 8 experts' slices contiguous   1.01 GB / 13 GB/s  =  78 ms
batched matmul at DRAM speed        1.01 GB / 13 GB/s  =  78 ms
                                                   total 156 ms
against today                                            275 ms
```

**~119 ms/token saved of a ~380 ms token — about 1.45x, or ~3.8 tok/s against
llama.cpp's 4.21.** That closes most of the gap and would make this the fastest
configuration measured on the model.

## Why the copy is unavoidable, and the one way to avoid most of it

Expert slices arrive as separate `Arc<[u8]>` — `ExpertSlices` is
`HashMap<(String, u32), Arc<[u8]>>` — so the 8 experts selected for a layer are
8 unrelated allocations. `mul_mat` over all of them at once needs one tensor,
which needs one buffer.

**But the copy is only needed for cache hits.** Misses are read from disk into a
fresh buffer, and `read_expert_slices` could read *directly into* one contiguous
per-layer buffer at no extra cost. At the measured 69% hit rate that still
leaves ~700 MB/token of copying (~54 ms), so the estimate above is the
conservative one.

## The measurement that would have killed it — DONE, and it says build

`bigtea-kernelbench` already does exactly this experiment: it binds the
**stacked** expert tensor and runs `mul_mat_id`, which is the batched form.

```
$ bigtea-kernelbench Qwen3-30B-A3B-Q4_K_M.gguf --reps 3 --threads
layer 20, 6 experts, 2048 x 768 per matrix, 17.5 MiB resident

TOKENS    MS/PASS    MS/TOKEN    GFLOP/s      GiB/s
     1      1.53       1.53         37.0      11.17

 THREADS    MS/PASS   SPEEDUP
       1      1.69      1.00x
       2      0.69      2.44x
       4      0.59      2.86x
       8      1.15      1.47x
      20      1.39      1.22x

reference  single-threaded memcpy: 16.8 GiB/s (read+write)
```

**The batched form reaches 11.17 GiB/s against a 16.8 GiB/s memcpy ceiling, and
it parallelises to 4 threads (2.86x).** Our streaming path runs the same
arithmetic at **3.7 GB/s on one thread**. So node count *is* the bottleneck, the
kernel is not, and the ticket is worth building.

| per layer, one token | |
|---|---:|
| streaming path today (8 experts, 24 nodes) | ~5.7 ms |
| batched `mul_mat_id`, 4 threads | **0.59 ms** |

Across 48 layers that is **~275 ms/token → ~28 ms**, and even paying the full
78 ms copy the expert phase lands near **106 ms** — a saving of ~170 ms on a
~380 ms token.

**Do not read this as 3.6x on generation.** Disk is ~228 ms/token of that same
token and is untouched by any of it; see the note at the bottom.

## What must not be assumed

- **That llama.cpp's 4-thread peak means our path can reach one.** It uses
  `ggml_mul_mat_id` over the *whole stacked* expert tensor with an index tensor,
  which it can do because it mmaps the entire model. Bigtea streams a subset and
  cannot bind the stacked tensor — **it has to build a stack of the 8 selected
  experts and pass ids `0..8`.** That is the copy above, and it is why our
  version is worth ~1.45x where llama.cpp's costs nothing.

## How to build it

`bigtea-kernelbench` lines ~265-275 are working reference code for the batched
form, including `Context::mul_mat_id`, which already exists in the wrapper. The
change in `expert_ffn_single` is:

1. one contiguous buffer per matrix kind, 8 experts wide, filled from the
   `Arc<[u8]>` slices (misses can be read straight into it — see above);
2. bind each as a 3D `[n_embd, n_ff, 8]` tensor instead of 8 separate 2D ones;
3. `mul_mat_id(gate, x, ids)`, `silu`, `mul`, `mul_mat_id(down, act, ids)` with
   `ids = 0..8`;
4. the routing weights still have to be applied per expert and summed — that
   part does not change, and getting it wrong is fluent nonsense rather than an
   error, so check the output text before the timing.
- **That the 1.60x is all here.** Disk is 7.3 s of a 12.2 s 32-token run — 60%.
  Even a perfect expert path leaves that untouched, and R2 (overlap) is the
  ticket that addresses it.
