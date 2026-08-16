---
topic: The first quality measurement in this project — perplexity, matched to llama.cpp's windowing, agreeing to 0.5% and 1.1% on two architectures
status: measured
links: [lts-parity-criteria.md, qwen3-4b-vs-llamacpp-2026-08-10.md]
---

Every correctness check in this project until now has been *"does it say
Paris"*. That catches a broken forward pass and nothing subtler. A slightly
wrong RoPE base, an off-by-one in the causal mask, an F16 rounding difference in
the KV cache, or a repacked kernel that is *almost* right — all of them answer
Paris.

**Perplexity is a number over thousands of tokens, so it moves when any of those
are wrong**, and llama.cpp reports the same quantity, so the two can be compared
directly. The LTS checklist has said "no perplexity or eval was collected" since
it was written.

## The result

Same corpus, same 128-token chunks, same models, `-t 4` on both sides.

| perplexity | Chaos | llama.cpp | difference |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct Q4_K_M | **29.0909** | 29.2456 ± 6.49 | **0.53%** |
| Qwen3-4B Q4_K_M | **33.6434** | 34.0293 ± 9.64 | **1.13%** |

```
$ chaos-run <model> -f ppl_natural.txt --ppl-chunk 128 -t 4
perplexity 29.0909 over 189 tokens in 3 chunks of 128
           mean NLL 3.3704 nats/token

$ llama-perplexity -m <model> -f ppl_natural.txt -c 128 -t 4 --no-warmup
perplexity: calculating perplexity over 3 chunks, n_ctx=128
Final estimate: PPL = 29.2456 +/- 6.49413
```

**Two architectures, two tokenizers (SPM and BPE), and both agree within ~1%.**
That is the strongest statement this project has been able to make about
correctness: it exercises the tokenizer, RoPE, the causal mask, the KV cache,
the fused attention kernel, the FFN, weight repacking and the output projection
against an independent implementation, on a number that would move if any of
them were wrong.

Both figures are slightly *below* llama.cpp's, consistently. The corpus is only
189 scored tokens and llama.cpp's own error bar is ±6.49, so **this is not a
claim that Chaos is more accurate** — it is well inside the noise. A larger
corpus would be needed to say anything about the sign.

## The windowing is the measurement

Two details decide the number, and both were wrong in the first version:

**1. Whole chunks only.** A trailing fragment gives its scored tokens far less
context than a full chunk. Including one 98-token remainder alongside three full
chunks took the answer from 29.25 to **33.65 — a 15% error from one short chunk
out of four.** llama.cpp drops the remainder; so do we.

**2. Score the second half, and count it exactly.** llama.cpp scores
`n_ctx - 1 - n_ctx/2` tokens per chunk — 63 at a context of 128, not 64 — so
every scored token has at least half a chunk of history. Scoring from position 1
instead measures mostly how short the context was: on the same file that gave
**1.9232**, which looks like a spectacular result and means nothing.

An off-by-one here is invisible in the output and simply shifts the number.
**Anyone comparing these figures to a published perplexity must match the chunk
size and the corpus**, or they are comparing windowings.

## How it runs, and why it is slow

Tokens are fed **one at a time**. The forward pass projects only the final
position through the output matrix — that was a 253 GFLOP saving on prefill and
it is worth keeping — so per-position logits are only available a step at a
time. A 128-token chunk therefore costs 128 forward passes.

Llama-3.2-1B: ~6.3 s per chunk. Qwen3-4B: ~20 s. That is fine for a quality
check and would need the all-positions logits path back if it ever had to run
over a real corpus like wikitext-2.

## What is not covered

- **Corpus size.** 189 scored tokens is enough to catch a broken model and not
  enough to rank two working ones. The ±6.49 on the reference says so directly.
- **The MoE and V4-Flash paths**, untested here — they were left alone to keep
  memory free. `--ppl-chunk` works on them and the run would simply be slow.
- **Quantisation comparisons**, which is what perplexity is usually *for*. This
  measures the engine against another engine on the same file, not one quant
  against another.
