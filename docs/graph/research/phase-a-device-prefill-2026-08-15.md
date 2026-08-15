# Phase A: the card runs the model, and the number was wrong twice

**2026-08-15.** The GPU bar moves: Bigtea's own binary runs a full Qwen3-4B
prefill on an RTX 3050. **The current figure is 1.33-1.52x**, from the repeat
harness. Two earlier figures from this same day are retracted below — 0.42x and
1.62-1.78x — and the second one reached a merge-commit headline before it was
corrected.

**This file no longer carries a number in its name.** It used to be
`phase-a-the-card-at-1.7x-2026-08-15.md`, and that number was wrong within
hours. A filename cannot be corrected in place the way a body can, and it gets
quoted by people who never open the file.

Links: [gpu-the-card-works-vulkan-not-cuda-2026-08-15.md](gpu-the-card-works-vulkan-not-cuda-2026-08-15.md) ·
[the-igpu-is-not-a-tier-2026-08-15.md](the-igpu-is-not-a-tier-2026-08-15.md) ·
[the-knee-moves-with-n-2026-08-14.md](the-knee-moves-with-n-2026-08-14.md)

## RETRACTED TWICE, and the second retraction is the more useful one

**1.62-1.78x is also retracted.** It was measured one prefill per process, and
`#68`'s merge commit headline says 1.73x on the strength of it. Run through the
repeat harness — one load per target, a discarded warm-up, three timed prefills
— the same code reads:

```
run 1    cpu   59.93  device   80.27 tok/s   1.34x
run 2    cpu   52.68  device   80.02 tok/s   1.52x
run 3    cpu   48.84  device   73.09 tok/s   1.50x

cpu     median    52.68 tok/s   (48.84-59.93)
device  median    80.02 tok/s   (73.09-80.27, warm-up discarded)
```

Three prefills in one process is a different measurement from one prefill per
process, and the harness that repeats is the one that counts. The cause of the
gap was found by building that harness: **the first version loaded the model
inside the timed loop.** Each load reads 2.32 GiB; eight back to back thrashed
the page cache and the drive and swung the CPU baseline 26.48-67.35 tok/s — a
2.5x spread that buried the effect being measured.

So the rule this node produced needs a second clause: **repeats, AND nothing
expensive inside the timed region.** The warm-up discard fixes the shader cache
and does nothing about a 2.32 GiB read sitting in the loop.

## RETRACTED: the 0.42x in the first version of this node

The first measurement of this path read **0.42x — the card slower than the
CPU** — and it was published with a confident causal story about PCIe round
trips. Both were wrong.

**The same binary, unchanged, measured 1.49x an hour later and 1.62-1.78x on
every run after.** The cause is the **Vulkan pipeline cache**: ggml's Vulkan
backend compiles a large set of compute shaders on first use, the driver
persists them to disk, and every later process starts warm. The opening runs
paid that compilation inside the timed region.

Three things went wrong at once, and only one of them was the number:

1. **A cold-cache run was reported as steady state.** One measurement, no
   repeats, of a path whose first execution does work no later execution does.
2. **A mechanism was asserted rather than measured.** The PCIe explanation
   sounded right and was never checked against arithmetic — the activations
   moved per prefill are ~1.4 GB, under a second at the measured rate, against
   a gap of nearly ten seconds. The number did not fit the story and that was
   visible without any new experiment.
3. **The retraction was only caught by accident**, because a build failed and
   the previous binary ran again. Nothing in the process would otherwise have
   re-measured a result that had already been written down.

This project has a standing rule that a competitive claim needs the command
line and the output in a doc. **The missing half of that rule is repeats**: a
single run of a path with warm-up behaviour is not a measurement, and the first
run of any GPU path is a different program from the second.

## 2026-08-16: planned allocation takes it to 2.5x

Wiring `ggml_gallocr` into the device path — every graph gets *planned* storage
instead of every tensor getting its own bytes — moved it again:

| | cpu median | device median | ratio |
|---|---:|---:|---:|
| context allocation | 52.68 | 80.02 | 1.33–1.52x |
| **planned allocation** | **73.91** | **183.86** | **2.49–2.65x** |

Both columns moved because the machine was quieter, which is exactly why the
ratio is the claim and the medians are printed beside it.

**Logit checksums did not move at all** — cpu 625.0074, device 621.1722, the same
values as before the change. The device computes the same answer faster and the
CPU path is untouched. That is the gate: a wrong device path returns plausible
numbers, never an error.

Weights were not involved. They already carry device pointers from
`load_resident_on_device`, and a graph allocator only assigns tensors that still
need storage — the split llama.cpp uses.

What is left is transfers: **upload 2.01s, download 1.85s** across three runs.
Removing them needs `x` to stay a tensor across the layer boundary and `q` to
stop round-tripping, both of which require the QKV and attention graphs to share
a context. See `backlog/activations-resident-across-layers.md`.

## The numbers

Qwen3-4B-Q4_K_M, every weight resident on the device, same process, same
prompt, back to back, three consecutive runs. The CPU side gets all 20 threads,
which is what prefill wants — a mistuned baseline would flatter the card.

```bash
bigtea-gpubench C:/Projects/models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf
```

| run | cpu tok/s | device tok/s | ratio |
|---|---:|---:|---:|
| 1 | 59.93 | 80.27 | **1.34x** |
| 2 | 52.68 | 80.02 | **1.52x** |
| 3 | 48.84 | 73.09 | **1.50x** |

Medians: cpu 52.68, device 80.02, **ratio 1.33-1.52x across invocations**. The
1.62-1.78x rows this table used to hold were single-shot per process and are
retracted above.

Logit checksums agree (625.01 cpu vs 621.17 device), so this is the same answer,
faster. **Load-to-first-token is better on the device too** — about 3.5s better
— because the 1.0-1.7s upload is repaid by a prefill that finishes 4s sooner.

## Where the device time goes

Measured per operation rather than attributed, which is the correction this node
exists to make. Of roughly 4.7s of device prefill:

| | seconds | calls |
|---|---:|---:|
| graph compute | 1.80 | 110 submissions |
| upload | 1.04 | |
| download | 0.66 | |
| realize (device allocation) | 0.64 | 110 allocations |

**Transfers are 36% of device time and allocation another 14%** — so the round
trips are real and worth removing, but they were never the 10s the retracted
version needed them to be. Compute is only 38% of the device's own time, which
is why keeping activations resident is the obvious next move rather than a
kernel question.

## Still far behind llama.cpp, and that part stands

llama.cpp gets **2042 pp512** on this same card and model. We get ~108. The
difference is not the kernels — it is the same ggml underneath — it is that
**this engine round-trips every activation through host memory between stages**,
and submits 110 separate graphs per prefill instead of one.

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

The per-operation table above is the evidence, and it replaces the argument the
retracted version made from the flat 1-vs-512 ratio. That ratio was measured on
the cold-cache run and says nothing.

## What this does and does not retire

**The tier is real but half-built.** 1.4x is worth having and it is not what
this hardware can do: half the device's time goes to transfers and allocations
that exist only because the forward pass hands host `Vec<f32>` between stages.
Keeping activations resident across a layer — and then across layers — is the
next move, and it is the shape llama.cpp already has.

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
  roughly double the apparent speedup.
- **Three consecutive runs, and the first ever run of a GPU path is discarded.**
  That is the rule this node was written to establish.
- Logit checksums are compared because **a wrong device path returns plausible
  numbers, never an error** — the standing failure mode in this project.
- `BIGTEA_PREFILL_TOKENS` overrides the batch, which is how the 1-token row was
  taken; it exists because "works at 1, dies at 512" and "dies at 1" are
  different bugs.

## Three segfaults, one cause — split into its own node

Every crash on the way to this number was the same fault: a tensor written
before its context was realized, which on a device writes through a null
pointer. The full account, and the rule it produced — *a function that both
builds graph nodes and writes into them cannot be used on a device* — is in
[mixed-residency-segfaults-2026-08-15.md](mixed-residency-segfaults-2026-08-15.md),
because it is the obvious thing to try and it costs a day.
