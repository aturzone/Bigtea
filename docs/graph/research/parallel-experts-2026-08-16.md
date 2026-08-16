# Parallelise across experts, not inside the matmul — 1.29x on expert compute

**2026-08-16.** Qwen3-30B-A3B-Q4_K_M, i7-13650HX (20 logical cores), 15.7 GiB.
Generation only. Runs interleaved so a warming page cache cannot masquerade as a
speedup.

| workers | 1 | 2 | **4** | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| generation tok/s | 3.52 | 3.74 | **3.86** | 3.87 | 3.82 |

Four alternating pairs of 1 against 4, spread under 1% on both sides:

| | generation tok/s | expert compute |
|---|---:|---:|
| 1 worker | 3.52 `[3.50 3.52 3.52 3.54]` | 2.2 s `[2.1 2.2 2.2 2.2]` |
| 4 workers | **3.86** `[3.84 3.86 3.89 3.89]` | **1.7 s** `[1.7 1.7 1.7 1.7]` |

**1.29x on expert compute, 1.10x end to end.** Output is byte-identical across
1, 2, 4 and 8 workers on three prompts.

## Why the previous attempt failed and this one does not

`CLAUDE.md` names the lead:

> **The MoE expert path wants ONE thread — 2.4x on Qwen3-30B** … llama.cpp peaks
> at 4 threads where we peak at 1, so its expert path parallelises and ours does
> not — batching the expert matmuls is the lead for the remaining 1.60x.

**ggml parallelises within a node.** Each expert matmul is a 2048×768
matrix-vector product; split twenty ways that is ~38 rows per thread per
barrier, and the threads cost more than the work. That is why the tuner settles
on one and why `-t 20` is 2.4x *slower* here.

The batching route was built and reverted: `mul_mat_id` measured 11.17 GiB/s in
`bigtea-kernelbench`, but on the streaming path the selected experts are
unrelated `Arc<[u8]>` slices and **making them contiguous costs ~1.02 GB/token**
— exactly what the kernel saves. Byte-identical output, 1.34 → 1.27 tok/s,
reverted.

This approach never gathers anything. Each expert keeps its own subgraph and its
own weights exactly where they already are; **N whole experts run side by side,
one ggml thread each**, and the partial sums are added in Rust — 2048 floats per
worker, which is nothing. The parallelism is across nodes, which is the axis
ggml does not offer and Rust does.

## Why four

The plateau is 4–6 and it falls by 8. This is **not** a core count: the win is
running whole experts concurrently and this model selects eight per token, so
past that there is nothing left to split, and the per-worker arena and context
setup start to show. Four is the safe end of the plateau.

`BIGTEA_EXPERT_WORKERS` overrides it for hardware this was not measured on, and
`1` restores the single-graph path exactly.

## Scope, stated so nobody over-reads it

- **Generation only.** `expert_ffn_single` is the one-token path. The batched
  prefill path (`expert_ffn_block`) is untouched, and prefill measured 1.31,
  1.31, 1.30, 1.32 tok/s across 1/2/4/8 workers — flat, as it should be.
- **1.10x end to end, not 1.29x.** Expert compute is 33% of a token here; disk
  is 39%. Amdahl does the rest.
- **V4-Flash is a different forward pass** (`deepseek4_forward.rs`) and does not
  go through this.

## The safety argument

ggml contexts are independent, but its context allocator is not documented as
thread-safe, so creation is serialised behind a mutex and only the compute runs
concurrently — microseconds against the thing being parallelised. Each worker
owns its own arena from a pool held on the runner, so buffers are not
reallocated per position, and **no `Context` ever crosses a thread boundary**:
`std::thread::scope` lets the workers borrow the shared `Arc` weights without a
`Send` wrapper around a raw `ggml_context`.
