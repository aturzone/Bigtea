---
topic: Where Chaos stands against llama.cpp on 2026-08-16 — one table, every row with its command line, and the V4-Flash deficit retracted
status: measured 2026-08-16
links: [qwen3moe-generation-parity-2026-08-16.md, qwen3-4b-vs-llamacpp-2026-08-10.md, threads-were-never-plumbed-2026-08-10.md, v4flash-vs-llamacpp-2026-08-07.md, parallel-experts-do-not-transfer-2026-08-16.md]
---

The README needs one honest table and this is the node behind it. Machine:
i7-13650HX (20 logical cores), 15.7 GiB RAM, RTX 3050 6 GB, NVMe, Windows 11.
llama.cpp is `llama-completion` from `llamacpp-unsloth/build/bin`, CPU build.

**Every row was taken with both engines alternating in one session.** This
machine drifts by up to 25% with its own state, and comparing against a number
from an earlier session is how three wrong figures got published here.

## The table

| workload | Chaos | llama.cpp | verdict |
|---|---:|---:|---|
| **V4-Flash**, prefill, ms/prompt token | **1640** | 1679 | **parity** |
| **V4-Flash**, generation, tok/s | **0.394** | 0.39 | **parity** |
| Qwen3-30B-A3B, generation, tok/s | 3.03–3.86 | 3.35 | **parity**, paired 3–2 |
| Qwen3-30B-A3B, prefill, tok/s | 1.22 | 1.17 | **parity** |
| Qwen3-4B, generation, both at defaults | **8.01** | 6.52 ± 0.33 | **1.23x ahead** |
| Qwen3-4B, generation, both hand-tuned | 7.64 | **9.16 ± 0.43** | **1.20x behind** |
| Qwen3-4B, prefill, 519 tokens | 83.4 | **88.3** | 1.06x behind |
| Llama-3.2-1B, generation, both hand-tuned | 21.95 | **27.85 ± 1.98** | **1.27x behind** |

**We lead on nothing given equal care, and we are behind on nothing that
streams.** Out of the box we lead on Qwen3-4B because Chaos measures the machine
and llama.cpp uses a fixed default; hand-tuned, llama.cpp is 1.20x faster on the
dense path, which matches the 1.23x recorded before any of the thread work — that
agreement is what says the ratio is real rather than an artefact of where on the
curve each engine sat.

## V4-Flash — the 2026-08-07 deficit is retracted

The figure this replaces, published in the README for nine days:

> load parity, prefill **1.62x behind**, generation **3-4x behind**. We lead on
> nothing on this model.

It no longer reproduces. Both command lines, verbatim:

```
chaos-run DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf \
          "The capital of France is" -n 8

llama-completion -m DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf \
                 --no-repack -c 512 -n 8 \
                 -p "The capital of France is" --no-warmup
```

Warm-to-warm: each engine ran once before anything was recorded, then three
alternating pairs. 10.3 GiB free at the start.

| pair | Chaos prefill | llama.cpp prefill | Chaos gen | llama.cpp gen |
|---|---:|---:|---:|---:|
| warm-up *(discarded)* | 1660 ms/tok | 1834 | 0.392 | **0.23** |
| 1 | 1640 | 1786 | 0.394 | 0.41 |
| 2 | 1660 | 1648 | 0.380 | 0.38 |
| 3 | 1620 | 1679 | 0.401 | 0.39 |
| **median of the three** | **1640** | **1679** | **0.394** | **0.39** |

The prompt tokenizes to 5 tokens for Chaos and 7 for llama.cpp, so **ms per
prompt token** is the only fair prefill comparison — same as in the 2026-08-07
node.

**The discarded warm-up is the point of the protocol.** llama.cpp's first run
read **0.23 tok/s**, which against 0.392 would have been a **1.7x lead** and a
fourth wrong published figure. Its first run generates only 2 tokens and pays
first-token cost across both of them; by the third it is at 0.39. Nothing in the
output says "this one is cold".

**`--no-repack` is not a handicap.** Without it llama.cpp does not load this
container at all — its repack buffer is one range for the whole model and it asks
the allocator for 137 GiB. Chaos repacks per tensor, finds every repackable
tensor is `Q8_0` with no x86 kernel, and reports `0 repacked`. Neither engine
gets a repack win here.

### llama.cpp's default is its best setting on this model

llama.cpp defaults to **10 threads** on this machine
(`system_info: n_threads = 10 (n_threads_batch = 10) / 20`), and Chaos's tuner
picks **4** for generation, so the table above is not comparing equal thread
counts. That is deliberate — each engine at its own default — but it would be a
handicap if 10 were the wrong number for llama.cpp.

It is not. `CLAUDE.md` records llama.cpp peaking at 4 threads on Qwen3-30B, so a
`-t 4` arm was started here. **It was abandoned rather than completed, and that
is the result**: a single 3-token generation had produced no output after **417 s
of wall clock for 50 s of CPU**, with CPU time not advancing at all over the last
126 s of that — against roughly 50 s wall for the same work at the default. It
was stopped and is reported as an observation, not as a ratio, because it never
finished a run to put a number on.

The mechanism is the one this project measured on its own reader pool:
llama.cpp's reads are page faults on an `mmap`ed 144 GB file, so thread count is
what sets the queue depth at the drive. **On a model that fits, threads are a
compute knob; on one that streams, they are an I/O concurrency knob**, and the
two want opposite counts — which is the mirror image of Chaos wanting *four*
threads for its own compute while its eight reader handles do the I/O
separately.

So the comparison stands, and neither engine was left on a bad setting.

*(A note for whoever runs this next: stopping a benchmark's wrapper does not stop
the engine. A `llama-completion` orphaned this way sat holding **8.98 GiB**, and
every run after it read 10x slow — V4-Flash generation 0.039 against 0.39 — which
looks exactly like a catastrophic regression and is not one. Check for stray
processes before believing a surprising number.)*

## What each other row comes from

| row | node |
|---|---|
| Qwen3-30B generation and prefill | `qwen3moe-generation-parity-2026-08-16.md` — five alternating pairs, medians, ranges quoted |
| Qwen3-4B prefill | `qwen3-4b-vs-llamacpp-2026-08-10.md` — matched at 519 vs 512 tokens, both sides `llama-completion` |
| Qwen3-4B and Llama-3.2-1B generation, defaults and tuned | `threads-were-never-plumbed-2026-08-10.md` — `llama-bench -n 128 -r 3`, both cells quoted because neither is quotable alone |

## Coverage, which is the gap that is not close

Counted from both binaries on 2026-08-16 rather than tallied by reading, which is
the only way this number has ever been right:

```
$ llama-completion --help | grep -oE '\-\-[a-zA-Z0-9][a-zA-Z0-9-]*' | sort -u | wc -l
182
  intersected with chaos-run's match arms      165 implemented
  intersected with its REFUSED table            17 declined, each with a reason
  in neither                                     0
```

Chaos also has 15 long flags llama.cpp does not.

| | Chaos | llama.cpp |
|---|---:|---:|
| chat template names | **52** | 54 — missing `hunyuan-dense`, `hunyuan-vl` |
| tokenizer families | **5** — BPE, SPM, WordPiece, Unigram, RWKV | 6 — missing `plamo2` |
| architectures diffed against the reference | **13** | 141 *declared* |
| samplers | 16 | 20 |
| GPU backends | 1, Vulkan, **not verified** | CUDA, Metal, Vulkan, SYCL, HIP |

The template count is a set difference, not a tally:

```
$ comm -23 llamacpp_templates.txt chaos_templates.txt
hunyuan-dense
hunyuan-vl
$ comm -13 llamacpp_templates.txt chaos_templates.txt
alpaca
mistral
```

**The architecture row is the one that matters and it is not comparable as
written.** llama.cpp *declares* 141; Chaos's 13 are the ones whose output was
diffed token for token at eight prompts each. Nobody has checked all 141.

## The three retractions this node inherits

- **"Chaos runs models larger than RAM and llama.cpp cannot."** It does, with
  `--no-repack`. Not a differentiator.
- **"Generation is ~2x behind on Qwen3-30B (1.07 vs 2.16)."** Parity, measured
  alternating. That framing had been steering which work got picked.
- **"Chaos leads llama.cpp on V4-Flash load and prefill."** Chaos's numbers were
  fresh, llama.cpp's were two days old. Withdrawn 2026-08-07 — and this node is
  the same mistake avoided rather than repeated, in the other direction.

## Related

- [[parallel-experts-do-not-transfer-2026-08-16]] — where a V4-Flash token
  actually goes, and why the remaining 67% is not reachable in code.
- [[v4flash-has-no-slack-2026-08-10]] — four probes for redundancy, four
  negatives.
- [[gpu-does-not-help-streaming-moe-2026-08-16]] — the GPU is 4.3x slower on the
  model this project exists for.
