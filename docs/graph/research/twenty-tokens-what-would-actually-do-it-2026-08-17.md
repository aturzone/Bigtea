---
topic: every remaining route to 20 tok/s on V4-Flash, costed rather than argued
status: resolved
links:
  - v4flash-ram-frontier-2026-08-16.md
  - v4flash-has-no-slack-2026-08-10.md
  - parallel-experts-do-not-transfer-2026-08-16.md
  - ../reference/hard-won-facts.md
---

# 20 tok/s: what would actually do it

Atur asked for this again after the frontier node closed it, so it was worked
properly rather than restated. **Two levers had never been costed** — batch
amortisation and quantisation — and one of them is the largest number this
project has found. Neither reaches 20 tok/s single-stream, and the reason is the
same in both cases and worth stating once:

```
t(token) = expert bytes / bandwidth  +  F
```

`F = 0.84 s` is measured, thread-swept, and **independent of every byte lever**.
20 tok/s is a 50 ms token. Any route that only reduces bytes is bounded by
`1/F = 1.19 tok/s` before it starts.

## Lever 1 — batching, and it is worth 8x

**Not previously costed, and it is the biggest thing here.** Expert reads are
deduplicated per block across a whole pass, so a pass over `n` tokens does *not*
read `n` times one token's bytes. Measured distinct experts per layer:

| tokens in the pass | distinct experts/layer | GiB read | **GiB per token** |
|---:|---:|---:|---:|
| 1 | 6 | 3.2 | **3.20** |
| 17 | 39.7 | 21 | **1.24** |
| 166 | 122.8 | 66 | **0.40** |

**Per-token expert bytes fall 8x between one token and 166.** Routing selects
6 of 256 per token, and 166 tokens select 122.8 distinct — the overlap is what a
single-token pass throws away.

With `F(n) = 0.84·n^0.49` (compute scales as `n^0.49`, measured) and 2.02 GiB/s:

| tokens in flight | seconds/pass | aggregate tok/s |
|---:|---:|---:|
| 1 | 2.42 | **0.41** (measured) |
| 17 | 13.8 | 1.24 |
| 166 | 42.9 | 3.87 |
| ~2000 | ~103 | **~19.5** |

At ~2000 tokens in flight the pass reads essentially the whole expert bank once
and divides it by 2000, and **20 tok/s of aggregate throughput is reachable on
this laptop**. It is also useless for what Atur wants: a pass takes ~100 s, so
every individual stream sees a token every 100 seconds. **Throughput and latency
are different products and 20 tok/s means the second one.**

Worth knowing anyway, because it is what a *serving* deployment would exploit,
and it is 8x rather than the few percent that everything else on the list is
worth. The engine cannot do it today — one sequence at a time — and the KV cache
for 2000 positions is its own problem.

## Lever 2 — a smaller quant, and it is worth 1.75x at best

Also never costed. V4-Flash is Q4_K_XL at 144 GB. Bytes per token scale with it:

| quant | container | GiB/token | t = bytes/2.02 + 0.84 | tok/s |
|---|---:|---:|---:|---:|
| Q4_K_XL (today) | 144 GB | 3.19 | 2.42 | 0.41 |
| IQ2-class | ~50 GB | ~1.11 | 1.39 | 0.72 |
| IQ1-class | ~35 GB | ~0.78 | 1.23 | 0.81 |

**Even at one bit per weight it does not double**, because `F` does not move and
already dominates at that point. This is the clearest possible demonstration of
what the frontier node found: past a certain point the disk stops being the
problem. It would also cost accuracy on a model whose expert bank is already
measured to be full-rank with no redundancy to give up.

## Lever 3 — put the *dense* weights on the GPU, not the experts

The one genuinely untried code direction, and the only one that attacks `F`.

The measured "GPU is 4.3x slower on streaming MoE" was the experts crossing PCIe
every token, which is the worst possible split. **The opposite split has never
been tried**: always-read weights resident in VRAM, experts streaming to the CPU.
Those weights are 7.38 GiB against a 6 GB card here, so it needs `-ot` to place a
subset — and `-ot` is already bound.

`F` is 0.43 s of graph evaluation plus 0.41 s of everything else. If a GPU took
the attention and dense arithmetic and `F` fell to ~0.25 s:

- alone: `1.58 + 0.25 = 1.83 s` → **0.55 tok/s** (1.34x)
- with an IQ2 quant: `0.55 + 0.25 = 0.80 s` → **1.25 tok/s** (3.0x)

**3x is the honest ceiling for this laptop with everything applied**, and it is
worth having — but it is not 20, and no combination of these three reaches 20 on
this hardware.

## What actually gives Atur 20 tok/s tonight

A smaller model, and he already has one. Measured on this machine, this week:

| model | tok/s | note |
|---|---:|---|
| Llama-3.2-1B-Q4_K_M | **18.94** | measured through the installed binary |
| Qwen3-4B-Q4_K_M, `-ngl 99` | 8.85 | |
| Qwen3-30B-A3B | 3.03–3.86 | needs `--force` |
| V4-Flash | 0.41 | 144 GB on 15.7 GiB |

**20 tok/s exists on this laptop. It just does not exist on a 144 GB model**, and
the frontier node says why in one line: 3.15 GiB of distinct expert weights per
token needs 67.7 GB/s to move at 20 Hz, which is a memory-bandwidth
specification no consumer CPU platform meets.

## Verdict

- **Do not** re-propose quantisation as a route to 20 tok/s. Costed: 1.75x.
- **Do not** re-propose batching as a route to 20 tok/s *interactive*. Costed:
  8x on throughput, ~100 s latency.
- **Do** try dense-on-GPU with experts on the CPU, worth ~1.34x on its own and
  the only untried thing that attacks `F`. It needs `-ot` and a subset that fits
  6 GB.
- The product answer for "20 tok/s with a coding agent" is **a model that fits**,
  and Chaos's job is to say which one that is before the download — which is what
  `chaos-model-info --budget` already does.
