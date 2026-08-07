---
topic: Bigtea vs llama.cpp on DeepSeek-V4-Flash — measured back to back, and we lose on all three
status: resolved
links: [v4flash-generation-first-numbers.md, zero-copy-expert-reads.md, head-to-head-llamacpp-2026-08-05.md, verify-before-citing.md]
---

## ⚠ CORRECTION, same day: the first version of this node was wrong

Its first version claimed Bigtea was **3.0x faster on load** and **1.20x faster
on prefill**. Both are **FALSE** and were published in the README, the CHANGELOG
and the v0.0.1 release notes before being caught.

The error: Bigtea's numbers were measured today, and llama.cpp's were copied from
`head-to-head-llamacpp-2026-08-05.md`, taken two days earlier under different
free-RAM conditions. **The two engines were never run back to back.** This project
has a written rule against exactly this — *a competitive claim is not citable
until the competitor's exact command line and its output are in a doc* — and the
rule was followed for the *format* of the claim while its *substance* was stale.

Corrected below. Run back to back, same machine, same minute, twice.

## The machine

```
CPU    20 threads          RAM   15.7 GiB total, 10.5 GiB free at measurement
Disk   NVMe, 2.55 GB/s sequential (2.37 GiB/s)
Model  DeepSeek-V4-Flash-UD-Q4_K_XL, 144 GB across 5 shards
       7.38 GiB always-read, 137 GiB routed experts, 6 of 256 per token
```

**10.5 GiB free is the first time the whole 7.38 GiB always-read set fitted.**
Every earlier Bigtea measurement was taken with 1-3 GiB of it streaming, so this
is also the first fair reading of the design as intended.

## The commands, and their output

**Bigtea**

```
bigtea-run DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf \
           "The capital of France is" -n 4

resident   loaded 1199 tensors, 7.38 GiB of 8.15 GiB budget in 7.6s (1.04 GB/s)
loaded     10.0s
prefill    5 tokens in 12.2s (0.41 tok/s)
output      Paris.",
generate   3 tokens in 46.7s (0.064 tok/s, 15.6s per token)
```

**llama.cpp**

```
llama-completion -m DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf \
                 --no-repack -c 512 -t 12 -n 4 \
                 -p "The capital of France is" --no-warmup

The capital of France isParis. The capital
load time        = 10532.14 ms
prompt eval time = 10524.22 ms / 7 tokens (1503.46 ms per token, 0.67 tok/s)
eval time        =  6432.39 ms / 2 runs   (3216.19 ms per token, 0.31 tok/s)
total time       = 16980.19 ms / 9 tokens
```

Second run of each, minutes later: Bigtea prefill 0.43 tok/s, generation 0.081;
llama.cpp prompt eval 0.69 tok/s, eval 0.21.

## The comparison

The prompt tokenizes to 5 tokens for us and 7 for llama.cpp, so **per prompt
token** is the only fair prefill comparison.

| | Bigtea | llama.cpp | |
|---|---:|---:|:--|
| load | 10.0s | 10.5s | parity |
| prefill, per prompt token | 2440 ms | **1503 ms** | **llama.cpp 1.62x faster** |
| generation | 0.064-0.081 tok/s | **0.21-0.31 tok/s** | **llama.cpp 3-4x faster** |

**Bigtea loses on prefill and on generation. It does not lead on anything here.**

Note also that llama.cpp's reported "load time" and "prompt eval time" are nearly
identical (10532 vs 10524 ms) and `load + eval ≈ total`, so its load is
substantially *overlapped with* the first evaluation via mmap — which is itself
the finding below.

## Why: they overlap I/O with compute and we do not

Both engines are reading the same ~3.2 GiB of routed experts per token from the
same drive. Neither can cache 137 GiB. So the disk work is equal, and the
difference is what happens *around* it.

llama.cpp `mmap`s the container and the kernel reads ahead **while the CPU is
computing the previous layer**. Bigtea reads a layer's experts, waits, computes,
then reads the next layer's — strictly serial. Measured on Bigtea, per token:
**2.3s of I/O and 1.0s of compute, run one after the other**. Overlapped, that
same work is `max(2.3, 1.0) = 2.3s` rather than `3.3s`.

That is the whole gap, and it is an architectural difference rather than a
constant factor. It is also the single most valuable thing left to build.

## What this does and does not change

**Unchanged and still true**: on Qwen3-30B-A3B, prefill beats llama.cpp at 565
(27.64 vs 23.55) and 2206 tokens (36.60 vs 33.59), matching at 8775. Those were
measured back to back in `head-to-head-llamacpp-2026-08-05.md` and stand.

**Retracted**: every claim of leading llama.cpp on V4-Flash, on any metric.

**The lesson, for the third time in this project**: a competitor's number has a
shelf life. Re-run it in the same session as the number you compare it against,
or do not make the comparison. See [[verify-before-citing]].
