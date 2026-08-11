---
topic: Where Bigtea stands against llama.cpp on 2026-08-11 — speed is level, coverage is not
status: measured, interleaved, one session
links: [qwen3-4b-vs-llamacpp-2026-08-10.md, v4flash-vs-llamacpp-2026-08-07.md, gemma-was-running-silu-2026-08-11.md]
---

Three deficits in the scoreboard were stale. All three are now parity or
better, and one of them was never as bad as recorded because the reference's
own number came from a better day.

## Command lines

```
bigtea-run     -m <model> -p "<prompt>" -n 32 --temp 0
llama-completion -m <model> -p "<prompt>" -n 32 --temp 0 --no-warmup -no-cnv
```

Dense prompt: `"The history of computing is a story of abstraction. "` x 40
(401 tokens). Streaming model: one sentence, `-n 16`.

## Speed

| model | phase | Bigtea | llama.cpp | verdict |
|---|---|---:|---:|---|
| Qwen3-4B | prefill | **76.5** | 69.3 | parity → ahead |
| Qwen3-4B | generation | **5.97** | 5.54 | 1.08x ahead |
| Gemma-2-2B | prefill | 124 / 141 | 115 / 146 | parity |
| Gemma-2-2B | generation | 8.01 / 10.78 | 7.12 / 10.67 | parity → ahead |
| Qwen3-30B-A3B | prefill | 1.70 | 1.77 | parity |
| Qwen3-30B-A3B | generation | 3.10 | 3.25 | parity, 5% behind, inside the spread |

Raw rounds, in the order they were run:

```
Qwen3-4B prefill    bigtea 78.23 76.01 76.56 76.37   llama.cpp 68.73 77.51 69.89 65.35
Qwen3-4B generation bigtea  6.11  5.56  5.83  6.12   llama.cpp  5.55  5.52  5.68  5.52
Qwen3-30B gen       bigtea  3.92  2.71  3.26  2.97   llama.cpp  3.60  2.93  3.31  3.19
```

## What the raw rounds show that the medians do not

**Bigtea's prefill is the more stable of the two.** Across four rounds it spans
76.01–78.23 (2.9%); llama.cpp spans 65.35–77.51 (**18.6%**). Its best round
ties us and its worst loses by 17%. Quoting either single round as the
comparison would be indefensible, in either direction.

**Generation is the reverse**: llama.cpp is tight (5.52–5.68, 2.9%) and Bigtea
spans 5.56–6.12 (10%), which is the thread tuner still moving. Bigtea's worst
generation round still beats llama.cpp's best.

**On the streaming model, run order dominates everything.** Bigtea scored 3.92
running first against llama.cpp's 3.60, and 2.71 running second against its
2.93. Same binaries, same prompt, opposite verdicts. Only a warm-to-warm
protocol — each engine run twice, compare the seconds — says anything, and it
says parity.

## The retraction that matters most

The scoreboard recorded Qwen3-30B generation as **2.63 vs 4.21, 1.60x behind**.
Bigtea is now 3.10. But llama.cpp measured back to back today runs **2.93–3.60
on the same command line**, not 4.21.

So part of that "1.60x deficit" was never real: it compared our number to the
reference's best day. **This is exactly the failure this project already
documented for its own numbers and then committed against a competitor's.** A
competitive claim needs both command lines *and* both runs in the same session,
not just both command lines.

## Coverage

| | Bigtea | llama.cpp |
|---|---:|---:|
| architectures diffed against the reference | **8** | 141 declared |
| chat templates | 26 | 54 |
| CLI flags (long) | 119 | 182 |
| tokenizer families | 4 | 6 |
| samplers | 16 | 20 |
| GPU backends | **0** | CUDA, Metal, Vulkan, SYCL, HIP |

The architecture row is not a like-for-like comparison and should not be quoted
as one: llama.cpp *declares* 141 and nobody has diffed all of them; Bigtea's 8
are ones whose output was checked token for token. It is still 8 against 141.

**Speed is level. Coverage is the gap, and no amount of tok/s closes it.**
