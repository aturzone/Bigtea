# Phase A: the card runs the model at 1.7x — and the first number was wrong

**2026-08-15.** The GPU bar moves: Bigtea's own binary runs a full Qwen3-4B
prefill on an RTX 3050 at **1.62-1.78x the CPU path**.

Links: [gpu-the-card-works-vulkan-not-cuda-2026-08-15.md](gpu-the-card-works-vulkan-not-cuda-2026-08-15.md) ·
[the-igpu-is-not-a-tier-2026-08-15.md](the-igpu-is-not-a-tier-2026-08-15.md) ·
[the-knee-moves-with-n-2026-08-14.md](the-knee-moves-with-n-2026-08-14.md)

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

## The numbers

Qwen3-4B-Q4_K_M, every weight resident on the device, same process, same
prompt, back to back, three consecutive runs. The CPU side gets all 20 threads,
which is what prefill wants — a mistuned baseline would flatter the card.

```bash
bigtea-gpubench C:/Projects/models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf
```

| run | cpu tok/s | device tok/s | ratio | load-to-first-token |
|---|---:|---:|---:|---|
| 1 | 60.92 | 108.45 | **1.78x** | 12.48s cpu vs 8.50s device |
| 2 | 60.34 | 104.36 | **1.73x** | 12.17s cpu vs 8.56s device |
| 3 | 67.16 | 108.85 | **1.62x** | 11.77s cpu vs 8.56s device |

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

**The tier is real but half-built.** 1.7x is worth having and it is not what
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
