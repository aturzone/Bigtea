---
topic: AirLLM — "70B on a 4GB GPU" — what it actually claims, and how Bigtea compares
status: resolved
links: [head-to-head-llamacpp-2026-08-05.md, verify-before-citing]
---

Checked 2026-08-05 after it was raised as a competitive threat.
Repo: https://github.com/lyogavin/airllm

## What it actually claims

- **"4GB" is GPU VRAM, not system RAM.** The headline is "70B large language models run on a
  single 4GB GPU card."
- Technique is **layer-by-layer streaming**: "AirLLM only ever keeps one layer on the GPU at a
  time." Optional 4-bit/8-bit block-wise compression. GPU-first; CPU support was added later
  (v2.10.1, 2024-08-18).
- **It publishes no tokens-per-second figure anywhere.** The only performance number in the
  README is a relative "3x inference speed up" with compression enabled — no absolute throughput.

That absence is the same pattern as the Bigtea claim retracted earlier the same day: a
memory/capability headline with no speed number behind it. See [[verify-before-citing]].

## What it measures at, per third parties

0.5–2 tok/s typically, with one report of **292 seconds per token** (0.003 tok/s) on an
RTX 6000 Ada. Community writeups are consistent that it converts a memory bottleneck into a disk
bottleneck. One summarises it exactly right: *"AirLLM does not make 70B fast on a 4GB GPU. It
makes 70B possible on a 4GB GPU."*

- https://starlog.is/articles/llm-engineering/lyogavin-airllm/
- https://ai505.com/airllm-run-70b-models-on-your-4gb-gpu-but-pack-a-lunch/
- https://abrarqasim.com/blog/airllm-the-hype-vs-the-reality/
- https://news.ycombinator.com/item?id=49154228

## Bigtea at a minimum footprint, measured

```
bigtea-run.exe Qwen3-30B-A3B-Q4_K_M.gguf "The capital of France is" -n 16 --cache 0.5

cache      0.50 GiB for experts
generated  16 tokens in 16.9s (0.94 tok/s)
streaming  resident 0.93 GiB, streamed 14.63 GiB over 16453 expert reads, 22% cache hits
PEAK RSS: 1.58 GiB
```

**1.58 GiB peak resident, 0.94 tok/s, CPU only, no GPU.** Against AirLLM's 4 GB of VRAM that is
2.5x less memory, and the speed sits inside their measured 0.5–2 tok/s band.

## The architectural difference, which is the real point

AirLLM streams **every layer** per token, so it reads the *entire model* from disk for each
token generated. That 100% re-read is the whole reason it is slow.

Bigtea keeps the always-read weights resident and streams **only the routed experts**. At the
1.58 GiB footprint above it read 0.91 GiB per token — **5% of the 17.28 GiB model** — and that
is with the cache deliberately starved to 0.5 GiB. Frequency-gated admission has no equivalent
in AirLLM.

This is a difference in access pattern, not in tuning, and it is the same reason Bigtea's design
should hold up on V4-Flash where llama.cpp's LRU lets cold expert traffic evict dense weights.

## Honest caveats — do not drop these when repeating the comparison

- **AirLLM runs dense models** (Llama 70B). Bigtea's streaming path is MoE-only so far, so the
  two do not currently cover the same models.
- **AirLLM is GPU-first**, Bigtea is CPU. Different hardware targets; this is not like-for-like.
- The 0.94 tok/s above is *inside* AirLLM's reported range, not clear of it. The win to claim is
  **footprint at comparable speed**, not raw speed.
- No AirLLM run was performed on this machine. Its numbers here are third-party reports, not
  measurements taken alongside Bigtea's — which is weaker evidence than the llama.cpp ladder,
  where both commands were run back to back. Flagged deliberately.
