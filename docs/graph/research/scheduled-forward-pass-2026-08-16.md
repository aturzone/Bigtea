# The scheduled forward pass: built, and it stops at one assert

**2026-08-16.** Branch `ticket/r25-scheduled-forward-pass`, **not merged**.
Llama-3.2-1B-Instruct-Q4_K_M, RTX 3050 6 GB via Vulkan.

Running every graph through `ggml_backend_sched` is what `--op-offload`,
`--split-mode`, `--tensor-split` and a block-splitting `--override-tensor` all
wait on. The path is built end to end and it does not work. Recorded now because
the failure is specific and the next session should not re-derive the design.

Links: [ngl-partial-offload-2026-08-16.md](ngl-partial-offload-2026-08-16.md) ·
[mixed-residency-segfaults-2026-08-15.md](mixed-residency-segfaults-2026-08-15.md)

## What is built, and works at the unit level

- **`OwnedHostBuffer`** — a CPU buffer that owns its bytes, so a `WeightSet` can
  hold the allocation and the ggml buffer together without being
  self-referential.
- **`WeightSet::use_host_buffers`** — host weights get a CPU buffer, which is
  what makes them copyable across a split.
- **`Compute::Sched`** — the third variant, threaded through all seven
  realize/run sites.
- **`StreamingRunner::set_op_offload`** and a `Scheduler` built **per pass**,
  because it borrows the backends it was made from and holding one beside them
  inside the runner would be self-referential.

## The scheduler demonstrably does its job

With every weight on the host (`-ot "*=CPU"`), `GGML_SCHED_DEBUG=2`:

```
## SPLIT #0: CPU # 0 inputs
node #  0 (  GET_ROWS)  [CPU]     leaf_0 (205M) [CPU]
## SPLIT #0: Vulkan1 # 5 inputs: [leaf_2] [leaf_0] [leaf_4] [leaf_5] [leaf_6]
node #  0 (  RMS_NORM)  [Vulkan]
node #  2 (   MUL_MAT)  [Vulkan]  Vulkan1#leaf_0#0 (2M) [NULL]
node #  5 (   MUL_MAT)  [Vulkan]  Vulkan1#leaf_5#0 (576K)
```

It puts the embedding lookup on the CPU, moves the QKV matmuls onto Vulkan, and
copies the host weights across the boundary. That is exactly the behaviour
`--op-offload` names, and it is the first time a mixed graph has run in this
engine at all.

## Where it stops

The **third** graph — attention — splits and then dies in allocation:

```
ggml-alloc.c:623: GGML_ASSERT(buffer_id >= 0) failed
```

That is `ggml_gallocr_allocate_node` receiving `-1` from the scheduler's
`hv_tensor_backend_ids`: a node the split left unassigned. Two properties make
the attention graph the odd one out and are the first things to check:

1. it is the only graph built in a **caller-owned scratch buffer**
   (`Context::in_buffer`) rather than a fresh arena, and
2. it is full of **views** — `view_2d`/`view_3d`/`permute`/`cont` over the KV
   cache — and a view is assigned through its `view_src`.

**A view whose `view_src` is itself unassigned** is the hypothesis, untested.

## What was tried and did not fix it

**Reserving before allocating.** llama.cpp always calls
`ggml_backend_sched_reserve`, and ggml's multi-buffer allocator will not
auto-reserve from `alloc_graph` once there is more than one backend, so this
looked right. It did not help, **and it broke two scheduler tests that had been
passing** — which is its own answer, and the reason it is not in the branch.

## The cost this exposed, which survives regardless

A scheduled graph cannot use **repacked** weights. Repacking rearranges a tensor
into the layout the CPU kernels want, and a scheduler that may hand the same
tensor to a Vulkan kernel makes that layout wrong rather than merely unhelpful.
That is **1.39x of CPU prefill** given up as the entry price.

So `--op-offload` is not obviously a win even once it runs, and its first
measurement should be against that 1.39x rather than against nothing. The
arithmetic that frames it: Qwen3-4B holds 2.33 GiB of weights and the bus runs
at ~2.6 GiB/s, so a pass that copies them costs ~0.9 s. Against a 22-token
prefill (0.5 s on CPU) that is a clear loss; against a 4096-token prefill (~95 s)
it is noise. **`--op-offload` is a long-prefill flag or it is nothing**, and that
should be stated before anyone measures it on a short prompt and concludes the
scheduler is slow.

## Why the flag stays declined

A flag that kills the process is worse than one that refuses. `--op-offload`
remains in `REFUSED`, with a reason that names the assert rather than saying
something vague — a reader who hits it learns where the work stopped.
