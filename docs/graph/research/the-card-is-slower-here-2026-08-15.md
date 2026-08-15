# Phase A: the card runs the model, and it is 0.42x the CPU

**2026-08-15.** The GPU bar moves — Bigtea's own binary runs a full Qwen3-4B
prefill on an RTX 3050 — and the number is a **negative**. The card is slower
than the CPU path it replaces, by a factor that does not improve with batch.

Links: [gpu-the-card-works-vulkan-not-cuda-2026-08-15.md](gpu-the-card-works-vulkan-not-cuda-2026-08-15.md) ·
[the-igpu-is-not-a-tier-2026-08-15.md](the-igpu-is-not-a-tier-2026-08-15.md) ·
[the-knee-moves-with-n-2026-08-14.md](the-knee-moves-with-n-2026-08-14.md)

## The numbers

Qwen3-4B-Q4_K_M, every weight resident on the device, same process, same
prompt, back to back. The CPU side gets all 20 threads, which is what prefill
wants — a mistuned baseline would flatter the card, and here it would have been
flattering a loss.

```bash
bigtea-gpubench C:/Projects/models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf
```

| tokens | target | prefill tok/s | load | load-to-first-token |
|---|---|---:|---:|---:|
| 512 | cpu | **73.23** | 2.69s | 9.69s |
| 512 | device | **30.79** | 3.30s | 19.92s |
| 1 | cpu | 3.56 | 2.72s | 3.00s |
| 1 | device | 1.56 | 3.31s | 3.95s |

**0.42x at 512 tokens, 0.44x at one.** Logit checksums agree (625.01 vs 621.17;
540.45 vs 539.52), so this is the right answer computed slowly, not a broken
path computed quickly.

The upload itself is not the problem. It is 2.32 GiB in 0.83–1.66s across runs,
against a 2.0–3.2s disk read the CPU path also pays; the tier's marginal load
cost is under a second.

## Why, and it is our design rather than the card

llama.cpp gets **2042 pp512** on this same card and model. We get 30.79. The
difference is not the kernels — it is the same ggml underneath — it is that
**this engine round-trips every activation through host memory between stages.**

A layer, on the device path, currently does:

```
upload x -> compute q,k,v -> download q,k,v
upload q, upload K cache, upload V cache -> compute attention -> download
upload x -> compute dense ffn -> download x
```

That is by design and it is *why the streaming engine exists*: activations live
in host memory because the expert path streams from disk into host memory, and
`forward_cached` hands host `Vec<f32>` between stages so the KV cache, the
router and the expert loop can all read them. On a CPU that costs nothing. On a
device every one of those arrows is a PCIe transfer, roughly 5 MB each at 512
tokens, times three or four per layer, times 36 layers.

**The ratio being flat between 1 and 512 tokens is the evidence.** A fixed
per-call overhead would be crushed by a 512x larger batch. A cost that scales
with activation *volume* would not — and activations scale with tokens exactly
as compute does, so the ratio holds. That is what a transfer-bound path looks
like.

## What this does and does not retire

**It does not retire the GPU tier.** It says the tier cannot be reached by
swapping the executor underneath a host-resident forward pass. The fix is
structural: keep activations on the device across the whole layer, and ideally
across all layers, so the graph is uploaded once and downloaded once per pass
rather than per stage. That is the shape llama.cpp already has.

**It does sharpen Phase C.** The differentiator was to be dense weights resident
on the card with routed experts streaming from disk — and the expert path is
precisely the part that *must* return to host memory, because that is where the
streamed bytes land. Phase C therefore inherits this round trip by construction
at every MoE layer, on top of needing `ggml_backend_sched`. The honest ceiling
for Phase C is lower than the 1.3x that was estimated from the 24%-of-a-token
compute share, because the estimate assumed the compute moved for free.

**It confirms the estimate was right about the disk.** 76% of a token on the MoE
path is disk and no GPU fixes disk. What Phase A adds is that the remaining 24%
does not move for free either.

## Method notes

- `bigtea-gpubench` loads the model fresh for each target, so `load` is honest
  rather than warm.
- The CPU baseline is `-t 20`, not the default and not `-t 4`. Prefill is
  compute-bound and wants every thread; quoting a 4-thread CPU prefill would
  have turned 0.42x into roughly 0.8x and hidden most of the loss.
- Logit checksums are compared because **a wrong device path returns plausible
  numbers, never an error** — the standing failure mode in this project.
- `BIGTEA_PREFILL_TOKENS` overrides the batch, which is how the 1-token row was
  taken; it exists because "works at 1, dies at 512" and "dies at 1" are
  different bugs.

## Three segfaults, one cause, and the API that was supposed to prevent it

Every crash on the way to this number was the same mistake: **writing a tensor
before the context was realized.** On a device a tensor has no memory until
`ggml_backend_alloc_ctx_tensors_from_buft` runs, and that cannot run until the
graph is complete — so any `set_*` before that point writes through a null
pointer and the process dies with no Rust backtrace.

`Compute::realize` exists precisely to make that ordering explicit, and it still
happened three times:

1. `pos.set_i32` in the QKV builder — died at layer 0, immediately after the
   embedding.
2. `mask.set_bytes` **inside** `attention_flash` — the function built part of
   the graph and wrote to it, so the caller could not have ordered it correctly.
   It now returns the mask tensor unwritten.
3. The original mixed host/device experiment, which is the same fault seen from
   the other side.

The lesson is not "remember to call realize". It is that **a function which both
builds graph nodes and writes into them cannot be used on a device**, because
those two operations must be separated by an allocation the function does not
control. Any future builder has to return its input tensors rather than fill
them.
