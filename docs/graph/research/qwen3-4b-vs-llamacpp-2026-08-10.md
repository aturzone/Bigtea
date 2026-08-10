---
topic: The first dense head-to-head against llama.cpp — Qwen3-4B, both command lines recorded. Bigtea is 2.9x behind on prefill and 8.8x behind on generation, and the cause of each is now known
status: measured
links: [lts-parity-criteria.md, the-plateau-was-ours-2026-08-10.md, head-to-head-llamacpp-2026-08-05.md]
---

Every previous comparison in this project was on a model that streams from disk,
where the result is dominated by I/O and says little about the compute path.
**Qwen3-4B fits entirely in RAM (2.32 GiB), so this is the first measurement of
the arithmetic on its own.** It is also the cheapest comparison available and it
had never been run.

Both sides on the same machine, same session, 20 threads.

## llama.cpp

```
$ llama-bench.exe -m models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf -p 512 -n 128 -r 2 -t 20

| model                  |     size |   params | backend | threads |  test |            t/s |
| ---------------------- | -------: | -------: | ------- | ------: | ----: | -------------: |
| qwen3 4B Q4_K - Medium  | 2.32 GiB |   4.02 B | CPU     |      20 | pp512 | 111.20 ± 1.64 |
| qwen3 4B Q4_K - Medium  | 2.32 GiB |   4.02 B | CPU     |      20 | tg128 |   5.90 ± 0.12 |

build: daef2b3 (1)
```

## Bigtea

```
$ bigtea-run.exe models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf -f target/prompt512.txt -n 1
prompt ... -> 651 tokens
generated  1 tokens in 16.9s

$ bigtea-run.exe models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf "Write a short paragraph about the sea." -n 128
generated  128 tokens in 191.7s (0.67 tok/s)
```

## The scoreboard

| Qwen3-4B dense, CPU | Bigtea | llama.cpp | verdict |
|---|---:|---:|---|
| prefill | **38.5 tok/s** (651 tok in 16.9 s) | **111.2** (pp512) | **2.9x behind** |
| generation | **0.67 tok/s** (128 tok) | **5.90** (tg128) | **8.8x behind** |

Not a like-for-like prefill length — 651 against 512 — and longer prompts are
*harder*, so 2.9x is if anything generous to Bigtea. Neither figure is quotable
as anything but a deficit.

## Why generation is 8.8x behind, and it is not the kernel

**The dense path has no KV cache.** Every generated token rebuilds the graph
over the entire sequence from position zero. Generating 128 tokens from a
9-token prompt means 128 forward passes averaging ~73 tokens each — roughly
**9,300 token-positions of work to produce 128 tokens**.

That is the same defect R3 fixed on the V4-Flash path, where it was worth 2.3x
immediately and more once residency was satisfied. On V4-Flash the waste was
partly hidden because a pass is dominated by reading 3.21 GiB of experts, which
is paid per pass rather than per token. **Here there is no such excuse: the
model is resident, so the wasted arithmetic is the whole cost.**

`KvCache` already exists and is already used by the streaming path. Wiring it
into the dense path is the single largest performance item on the LTS list, and
it is not research — the cache is written and the equivalence harness that
verified it on V4-Flash (`prefill(0..n)` then `step(n)` must match
`prefill(0..=n)`) applies unchanged.

## Two bugs found while measuring, both fixed here

### 1. A 651-token prompt aborted the process

```
ggml_new_object: not enough space in the context's memory pool
  (needed 2149124256, available 2147483648)
GGML_ASSERT(obj_new) failed
```

The dense arena was a hardcoded `2 << 30`. `ggml` does not return an error when
its arena is exhausted — it calls `GGML_ASSERT` and the process dies — so this
was an abort with no diagnosable message, at a length llama.cpp handles without
comment.

The arena is now computed from the shape, and the important term was the one
that was missing: **it is per layer, not per pass.** One graph spans all 36
blocks in a single context and `ggml` frees nothing inside a context, so every
layer's intermediates are alive simultaneously. A further term was still needed
after that: `ggml_graph_compute_with_ctx` allocates the graph struct *and its
per-thread work buffer* out of the same arena, so sizing for tensor data alone
left it 0.1% short — which is still an abort.

`bigtea-run` now also **refuses** a prompt that will not fit, with the arena it
needs, the memory that is free, and the longest prompt that would work.

### 2. The output projection ran on every position

`build_graph` projected the whole sequence through the output matrix and then
used one row. On a 651-token prompt that is `651 x 2560 x 151936` = **253
GFLOP** and 395 MB of logits, for one row of it.

Now only the final position is projected. This is a large part of why the arena
was so big, and it is pure waste removed rather than a trade.

## What this does not measure

- **Quality.** Both engines were run greedy where possible; no perplexity or
  eval was collected. `--llamacpp-defaults` exists on the Bigtea side so a
  sampled comparison measures engines rather than sampler settings.
- **Prefill at matched length.** 651 against 512. Worth redoing once the dense
  path can be given an exact token count.
- **Memory footprint**, which is where Bigtea's design argument actually lives
  and which llama-bench does not report.
