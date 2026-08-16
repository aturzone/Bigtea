# `--op-offload` works, and it cannot pay while we submit a graph per block

**2026-08-16.** Qwen3-4B-Q4_K_M, RTX 3050 6 GB via Vulkan, i7-13650HX.
Three runs per point, **median** reported, `-n 4` so prefill dominates.

The scheduled forward pass runs. `--op-offload` is implemented, produces the
same completion as every other path, and is **slower than not using it** — by
19% on the case it was supposed to win.

Links: [scheduled-forward-pass-2026-08-16.md](scheduled-forward-pass-2026-08-16.md) ·
[ngl-frontier-2026-08-16.md](ngl-frontier-2026-08-16.md)

## The bug that blocked it was one missing call

`ggml_set_input`. The scheduled path aborted on the attention graph with

```
ggml-alloc.c:623: GGML_ASSERT(buffer_id >= 0) failed
```

`ggml_backend_sched_backend_id_from_cur` has an explicit branch for it:

```c
// graph input
if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
    cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
```

Without the flag, a leaf with **no buffer, no data and no op** is something the
scheduler cannot place, and the unplaced node reaches
`ggml_gallocr_allocate_node` as backend `-1`, which **aborts the process**.
llama.cpp marks every graph input this way; that is not a style choice.

It also explains why the CPU must be passed **last** — the branch hardcodes
`n_backends - 1` as where an input lives.

**Found by bisection in a 60-line test, not in the model.** Two hypotheses were
wrong first: the caller-owned scratch buffer (swapped for a fresh arena, still
aborted) and the views (a `mul_mat → reshape → permute → cont` chain schedules
fine). Only when a minimal flash-attention graph reproduced it — and then
reproduced it *without* the flash attention — did the shared property become
visible: every leaf in that graph was bare.

## The measurement

| prompt | plain CPU (repacked) | `--op-offload` (host weights) | `-ngl 99` |
|---|---:|---:|---:|
| 11 tokens | 34.23 | 35.04 | 56.93 |
| ~900 tokens | **79.24** | **64.39** | 205.37 |

prefill tok/s. `--op-offload` is a wash on the short prompt and a **19% loss**
on the long one.

## Why, and it is structural rather than a tuning problem

The prediction written down before this ran was "`--op-offload` is a
long-prefill flag or it is nothing": the weight copy is fixed, the compute
scales with tokens, so a long enough prefill amortises the copy. **That
reasoning assumes the copy happens once per pass.**

It does not. This engine submits roughly **five graphs per block** — embedding,
QKV, attention, FFN, output head — so a 36-layer model is ~180 graph
submissions, and the scheduler copies a graph's weights in **per submission**.
The copy is amortised over one block, not over the model. Prefill length does
not help, because the same copies happen at every block whatever the batch is.

llama.cpp submits **one** graph for the whole pass. Its copies amortise across
all 36 blocks at once, which is the entire difference.

On top of that, scheduling costs the **1.39x repack**: a repacked tensor is in
the layout the CPU kernels want, and a scheduler that may hand it to a Vulkan
kernel makes that layout wrong rather than merely unhelpful. So the flag starts
19% behind before it moves a single operation.

**So `--op-offload` is not slow because the scheduler is slow.** It is slow
because of the same graph-per-block structure that
`backlog/activations-resident-across-layers.md` exists to fix, and it will not
pay until that lands. That makes it a second, independent argument for the same
piece of work — the first was 110 graph submissions costing 0.64 s of allocation
on a single prefill.

## What ships

The flag, off by default, with the measurement printed when it is switched on.
It is implemented and correct; it is not a good idea today, and telling a user
that at the moment they enable it is better than burying it in a node they will
not read.

`ggml_set_input` is applied on **every** path, not just the scheduled one.
Marking an input is what it is regardless of who runs the graph, and the 561
tests plus the placement matrix (plain, `-ngl 8`, `-ngl 99`, `-ot "*=CPU"`) all
produce the same completion with it.

## Open

- **`--split-mode` / `--tensor-split`** are now the only structural blockers
  left in the GPU set, and they need a second usable device. This machine's
  other GPU is the iGPU, which `the-igpu-is-not-a-tier` already excludes, so
  they cannot be verified here and stay declined.
- **Re-measure `--op-offload` after the graph count drops.** It is the cheapest
  test of whether fusing graphs did what it claims.
