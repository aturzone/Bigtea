---
topic: Bigtea vs llama.cpp on DeepSeek-V4-Flash — every number, and the gap that remains
status: open
links: [v4flash-generation-first-numbers.md, zero-copy-expert-reads.md, head-to-head-llamacpp-2026-08-05.md, ../backlog/lts-0-0-0.md]
---

One machine, one model, both engines. **We are faster on prefill and slower on
generation**, and this node says exactly by how much and exactly why.

## The machine and the model

```
CPU    20 threads          RAM   15.7 GiB total, ~6.7 GiB free
Disk   NVMe, 2.55 GB/s sequential (2.37 GiB/s), measured by bigtea-probe
Model  DeepSeek-V4-Flash-UD-Q4_K_XL, 144 GB across 5 shards
       43 blocks, 4096 embd, 256 experts (6 used + 1 shared)
       7.38 GiB always-read, 137 GiB routed experts
```

Neither engine can hold this model. The always-read set alone is 7.38 GiB
against 6.7 GiB free, so **even the resident part does not fit today**.

## llama.cpp — the command and its output

```
llama-cli -m DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf \
          --no-repack -c 512 -t 12 -n 32 -p "..."

load 12.3s   prefill 0.41 tok/s   eval 0.45 tok/s   correct output
```

`--no-repack` is required: without it llama.cpp builds a `CPU_REPACK` buffer
outside the mmap and dies. With it, **llama.cpp runs this model fine.** The claim
that it cannot is retracted — see `head-to-head-llamacpp-2026-08-05.md`.

## Bigtea — the command and its output

```
bigtea-run DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf \
           "The capital of France is" -n 5

resident   251 tensors, 6.21 GiB of 6.21 GiB budget in 4.1s (1.65 GB/s)
           1.17 GiB did not fit and will be re-read every token
prefill    5 tokens in 10.1s (0.49 tok/s)
output      Paris.",
generate   4 tokens in 51.9s (0.077 tok/s, 13.0s per token)
```

## The comparison

| | Bigtea | llama.cpp | |
|---|---:|---:|:--|
| load | 4.1s | 12.3s | **3.0x faster** |
| prefill | **0.49 tok/s** | 0.41 tok/s | **1.20x faster** |
| generation | 0.077 tok/s | **0.45 tok/s** | **5.8x slower** |

**Prefill is now genuinely ahead. Generation is not, and no claim is made that
it is.**

## Today moved generation 1.83x and prefill 2.2x

| change | prefill (5 tok) | 1-token pass |
|---|---:|---:|
| starting point | 32.4s | — |
| zero-copy expert reads | 23.7s | — |
| residency | 22.0s | 7.9s |
| **one graph per block, not 24** | **11.5s** | **4.6s** |
| batched parallel expert reads | **10.1s** | **4.0s** |

The single largest win was not I/O at all. `Context::compute` evaluates a
tensor's *entire ancestor graph*, so calling it on every intermediate **re-does**
the work once per call. The block had 24 such calls and needed 6. Removing the
other 18 was worth **1.9x**, and it is invisible on a long prefill because the
matmuls there are large enough to bury it.

## The number that decides the future: a single-token pass is 4.0s

Generation has no KV cache yet, so each token re-runs the whole sequence. That
makes the 0.077 figure pessimistic and not the one to plan against.

**A single-token forward pass costs 4.0s**, and that is exactly what one step of
a KV-cached loop will cost, because a single token routes to 6 experts per layer
rather than the ~26 a 5-token pass touches.

```
with a KV cache:   4.0 s/token  =  0.25 tok/s
llama.cpp:         2.2 s/token  =  0.45 tok/s
                                   still 1.8x short
```

## Where the 4.0s goes, and what is left to take

Measured per block (`BIGTEA_BLOCK_TIMING=1`), scaled to 43 blocks:

| | per token | share | lever |
|---|---:|---:|---|
| expert reads | 2.3s | 58% | 1.4 GiB/s achieved against 2.37 available |
| dense reads | 0.7s | 18% | **0 if the always-read set fits** — 1.17 GiB short today |
| compute (ffn, attn, tail) | 1.0s | 24% | flat from 4 to 20 threads |

Three things remain, in order of measured size:

1. **Fit the always-read set.** 1.17 GiB short. Closing an editor does it; so
   would a smaller quant. Worth 0.7s/token, and it is the user's RAM, not a code
   change. Bigtea already prints which processes to close and what it costs.
2. **Overlap reads with compute.** llama.cpp gets this free from `mmap` and
   kernel readahead; Bigtea reads then computes, serially, per layer. Perfectly
   overlapped this is `max(2.3, 1.0)` instead of `3.3` — worth ~1.0s/token.
   **Layers 0-2 route by token id, so their expert set is knowable before any
   compute runs** and they are trivially prefetchable.
3. **An expert cache across generated tokens.** Adjacent tokens route to
   overlapping experts. Unmeasured on this model, and bounded by RAM there is
   little of.

Optimistically: `4.0 − 0.7 − 1.0 = 2.3 s/token = 0.43 tok/s`. That is **parity
with llama.cpp, not victory.** Beating them needs the cache, and its size on
this machine is the thing nobody has measured.

## The honest position

- **Prefill: ahead, 1.20x, measured.**
- **Load: ahead, 3.0x, measured.**
- **Generation: behind, 5.8x today, ~1.8x once a KV cache exists.** Parity is a
  credible target from measured levers. A win is not yet.
- **Long context is the untested claim with the most upside.** llama.cpp's dense
  weights get evicted by cold expert traffic; Bigtea's are owned allocations that
  cannot be. That should widen with context and has never been measured on this
  model.

Nothing here should be quoted as "Bigtea beats llama.cpp" without the word
*prefill* attached to it.
