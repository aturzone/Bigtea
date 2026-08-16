---
topic: The first dense head-to-head against llama.cpp — Qwen3-4B, both command lines recorded. Chaos is 2.9x behind on prefill and 8.8x behind on generation, and the cause of each is now known
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

## Chaos

```
$ chaos-run.exe models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf -f target/prompt512.txt -n 1
prompt ... -> 651 tokens
generated  1 tokens in 16.9s

$ chaos-run.exe models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf "Write a short paragraph about the sea." -n 128
generated  128 tokens in 191.7s (0.67 tok/s)
```

## The scoreboard

| Qwen3-4B dense, CPU | Chaos | llama.cpp | verdict |
|---|---:|---:|---|
| prefill | **38.5 tok/s** (651 tok in 16.9 s) | **111.2** (pp512) | **2.9x behind** |
| generation | **0.67 tok/s** (128 tok) | **5.90** (tg128) | **8.8x behind** |

Not a like-for-like prefill length — 651 against 512 — and longer prompts are
*harder*, so 2.9x is if anything generous to Chaos. Neither figure is quotable
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

`chaos-run` now also **refuses** a prompt that will not fit, with the arena it
needs, the memory that is free, and the longest prompt that would work.

### 2. The output projection ran on every position

`build_graph` projected the whole sequence through the output matrix and then
used one row. On a 651-token prompt that is `651 x 2560 x 151936` = **253
GFLOP** and 395 MB of logits, for one row of it.

Now only the final position is projected. This is a large part of why the arena
was so big, and it is pure waste removed rather than a trade.

## FIXED, same day: the dense path now uses the KV cache

`StreamingRunner::forward_cached` already existed and already had a working KV
cache — it was only ever reached for MoE models, because the branch was
`if config.is_moe()`. Dense models fell through to the stateless path. Routing
them through the same code needed two guards that the streaming path was
missing for non-Qwen architectures:

- **QK norm**, which only Qwen3 has (the same fix already made in `qwen3.rs`).
- **RoPE type**, which was hardcoded to NeoX and must be NORM for llama/mistral.
  Both run without error on either layout, so this one would have been fluent
  nonsense rather than the clean "missing tensor" the QK-norm one gave.

Correctness first: cached and uncached produce **byte-identical text** on
Qwen3-4B (`CHAOS_UNCACHED=1` keeps the old path reachable so this stays
checkable, rather than being asserted once and then trusted).

| generation, 128 tokens | before | after | llama.cpp | verdict |
|---|---:|---:|---:|---|
| **Qwen3-4B** | 0.67 tok/s | **4.27** | 5.90 | 8.8x behind → **1.38x** |
| **Llama-3.2-1B** | — | **10.12** | 12.91 | **1.28x behind** |

**6.4x on Qwen3-4B generation from one branch condition.** The remaining ~1.3x
is real and is now the honest gap on dense generation.

## The prefill gap is weight repacking, and nothing else

Measured after the KV cache and the arena fix, at matched length, with
`llama-completion` on the **same file and the same prompt file** so the two
sides are doing identical work:

```
$ llama-completion -m Qwen3-4B-Q4_K_M.gguf -f target/p512.txt -n 1 -t 20 --no-warmup
prompt eval time = 5970.89 ms / 527 tokens (88.26 tokens per second)

$ llama-completion ... --no-repack
prompt eval time = 8276.33 ms / 527 tokens (63.68 tokens per second)

$ chaos-run Qwen3-4B-Q4_K_M.gguf -f target/p512.txt -n 1 -t 8
prefill 519 tokens in 8.6s (60.29 tok/s)
```

| Qwen3-4B prefill, 20 threads | tok/s | vs Chaos |
|---|---:|---:|
| llama.cpp, repacking **on** (its default) | **88.26** | 1.46x ahead |
| llama.cpp, repacking **off** | 63.68 | **1.06x ahead** |
| Chaos | 60.29 | — |

**Repacking is worth 1.39x to llama.cpp. Without it the two engines are 6%
apart.** Since both link the *same* ggml, that is the expected result once the
call pattern is equivalent — and it says the remaining gap is one named,
buildable thing rather than a diffuse deficit.

### What was ruled out on the way

Each of these was the obvious suspect and each is measured, not assumed:

| suspect | measurement | verdict |
|---|---|---|
| thread count | 8/10/12/16/20 give 60.3/57.9/54.6/56.8/57.4 tok/s | **not it** |
| graph-build and threadpool overhead | 108 `compute()` calls over 9.3 s ≈ 0.2% | **not it** |
| the matmul kernel itself | FFN runs at 472 GFLOP/s; `chaos-kernelbench` peaks at 420 for Q4_K | **already at the ceiling** |
| our arena sizing | fixed; was aborting, not slowing | **not it** |

Where the time goes, from the runner's own breakdown at 519 tokens:

```
1.5s qkv    1.2s attention    5.6s ffn    (of 9.3s)
```

The feed-forward is 60% of prefill, which is where repacking would land.

### Why we do not repack, and what it would take

Chaos binds weights **zero-copy**: `ggml` is handed a pointer into the mapped
container. That is what makes a 144 GB model run on a 15.7 GiB machine, and it
is not negotiable on the streaming path.

llama.cpp repacks through `ggml-backend`'s *extra buffer types*, which rearrange
a quantised tensor into a vectorisation-friendly layout when it is allocated.
Chaos uses the raw graph API and never sees that path.

**For a dense model that fits in RAM the trade is different**: the weights are
already copied into memory, so repacking them once at load costs a rearrange
and no extra residency. That is a contained change — it applies only where the
model is resident, and the streaming path keeps zero-copy binding untouched.

## What this does not measure

- **Quality.** Both engines were run greedy where possible; no perplexity or
  eval was collected. `--llamacpp-defaults` exists on the Chaos side so a
  sampled comparison measures engines rather than sampler settings.
- **Prefill at matched length.** 651 against 512. Worth redoing once the dense
  path can be given an exact token count.
- **Memory footprint**, which is where Chaos's design argument actually lives
  and which llama-bench does not report.
