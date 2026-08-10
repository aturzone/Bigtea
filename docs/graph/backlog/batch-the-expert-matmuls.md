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

## The measurement that would kill this ticket

**Do this first; it is one afternoon and it decides the rest.** Take one layer's
8 experts, copy them into a contiguous buffer by hand, and time a single
`mul_mat` of `2048 x 6144` against a column versus the 8 separate `2048 x 768`
matmuls it replaces, at 1, 4 and 8 threads.

- If the batched form reaches ~13 GB/s, the ticket is worth ~1.45x — build it.
- If it stalls at 3-4 GB/s too, the bottleneck is not node count and **this
  whole ticket is void**. Say so and close it.

`bigtea-kernelbench` already times the expert FFN with weights resident and is
the right place to put this.

## What must not be assumed

- **That llama.cpp's 4-thread peak means our path can reach one.** It uses
  `ggml_mul_mat_id` over the *whole stacked* expert tensor with an index tensor,
  which it can do because it mmaps the entire model. Bigtea streams a subset and
  cannot bind the stacked tensor. The comparison motivates the ticket; it does
  not size it.
- **That the 1.60x is all here.** Disk is 7.3 s of a 12.2 s 32-token run — 60%.
  Even a perfect expert path leaves that untouched, and R2 (overlap) is the
  ticket that addresses it.
