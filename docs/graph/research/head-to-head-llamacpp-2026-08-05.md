---
topic: Bigtea vs llama.cpp head-to-head on the target machine — the claim we were publishing is false
status: resolved
links: [moe-landscape-2026-08.md, benchmarking-methodology.md, verify-before-citing]
---

Measured 2026-08-05 on the target machine (i7-13650HX, 15.71 GiB RAM, Windows 11).
llama.cpp build 2026-08-03 (`llamacpp-unsloth`). Bigtea `ticket/rust-core` @ 59875d0.
Same prompt (`"The capital of France is"`), greedy (`--temp 0`), 12 threads.

## CORRECTION BLOCK (2026-08-05)

Three claims in `CLAUDE.md`, `memory/bigtea-runner-state.md`, and every prior session summary
were **wrong**. All three are retracted here.

**Claim 1 — "llama.cpp refuses this class of model here."** FALSE.
llama.cpp runs Qwen3-30B-A3B (17.28 GiB container) on this 15.71 GiB machine without
special flags, and produces correct text. It mmaps the file and lets the OS page it.

**Claim 2 — the error `failed to allocate buffer of size 147169738752` proves it.** MISATTRIBUTED.
147,169,738,752 bytes = 137.06 GiB. That is **DeepSeek-V4-Flash**, not Qwen3-30B. The error was
never produced by the Qwen3 run it was cited against. It was never written into any doc — it
lived only in session summaries, which is why it went unchecked for days.

**Claim 3 — implied: llama.cpp cannot run V4-Flash on this box.** FALSE.
The V4-Flash failure is caused by **one default flag**. `--repack` (default on) tries to
allocate a single 137 GiB `CPU_REPACK` buffer outside the mmap. With `--no-repack` llama.cpp
loads the 144 GB model in 12.3s and generates correct text at 0.45 tok/s.

This is the second time a Bigtea claim survived because nobody ran the opposing command.
See [[verify-before-citing]]. Rule going forward: **a competitive claim is not citable until the
competitor's exact failing command line is in a doc, with its flags.**

## Qwen3-30B-A3B Q4_K_M (17.28 GiB), 16 tokens

|                  | llama.cpp | Bigtea |
|------------------|-----------|--------|
| runs             | yes       | yes    |
| eval speed       | **2.83 tok/s** | 0.85 tok/s |
| prompt eval      | 1.44 tok/s | — (not separated) |
| load time        | 3.5s      | 0.9s   |
| peak working set | 8.87 GiB  | **0.93 GiB** |
| peak private     | 17.38 GiB | (streamed, not committed) |
| output           | identical | identical |

Both produced: `Paris. The capital of Italy is Rome. The capital of Spain is Madrid.`

**llama.cpp is 3.3x faster. Bigtea holds 9.5x less resident memory.** Those are the only two
honest deltas. Bigtea streamed 16.97 GiB over 19,185 expert reads in 12.3s of the 18.8s run —
i.e. **65% of our wall time is disk**, which is exactly what llama.cpp avoids by letting the
page cache keep hot experts.

Commands:
```
llama-completion.exe -m Qwen3-30B-A3B-Q4_K_M.gguf -p "The capital of France is" \
  -n 16 -no-cnv --temp 0 -t 12 --no-warmup
bigtea-run.exe Qwen3-30B-A3B-Q4_K_M.gguf "The capital of France is" -n 16
```

## DeepSeek-V4-Flash UD-Q4_K_XL (144 GB, 5 shards), 8 tokens

| flags | result |
|---|---|
| default | **fails**: `ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 147169738752` → `alloc_tensor_range: failed to allocate CPU_REPACK buffer` → `unable to create context` |
| `--no-repack -c 512` | **runs**: load 12.3s, prompt eval 0.41 tok/s, eval **0.45 tok/s**, correct output |

Output: `The capital of France is Paris.",\n    "The capital of France` (correct, then drifts into
what looks like JSON training data — expected for a base-ish completion at temp 0 with no
chat template).

Bigtea cannot run this model at all — the architecture is not implemented.

## What this means for positioning

"Runs models larger than RAM on a small machine" is **not a differentiator**. mmap has done this
for years; llama.cpp does it for a 144 GB model on a 15.7 GiB laptop today, faster than Bigtea
does it for a 17 GiB one.

What remains genuinely different, and is *not yet proven to be better*:

- **Explicit residency vs. OS page cache.** We pin the always-read weights and stream the rest.
  llama.cpp lets the kernel decide, so a cold expert read can evict a hot dense weight. Our
  0.93 GiB vs their 8.87 GiB is real — but a low working set is only a *win* if it buys
  something the user feels (machine stays responsive, or speed holds at long context).
- **Untested hypotheses that could become the real claim:**
  1. Long context — does llama.cpp's advantage survive at 8k/32k, where the page cache thrashes
     and our KV cache + bounded residency should degrade more gracefully? Not measured.
  2. Machine responsiveness under load — 8.87 GiB of page cache pressure vs 0.93 GiB. Not measured.
  3. Concurrent use — can the user keep working while a model runs? Not measured.

None of these is established. Until one is, Bigtea is **slower software that solves an
already-solved problem**, and should be described that way internally.

## Open questions

- Does the Bigtea/llama.cpp gap invert at long context? This is the single highest-value
  measurement available and it has not been taken.
- How much of our 3.3x deficit is the expert cache being useless (<4% of slices) vs. our
  kernels being slower than llama.cpp's repacked AVX2 paths? Separable: run Bigtea on the
  dense 4B, where no streaming happens, and compare tok/s there.
- ~~`--no-repack` costs llama.cpp speed on Qwen3-30B too — if repack is what makes them faster,
  and repack cannot work above RAM, the honest claim gets sharper.~~ **REFUTED, measured
  2026-08-05.** llama.cpp with `--no-repack` on Qwen3-30B: eval **3.89 tok/s** (vs 2.83 with
  repack on). Disabling repack did not hurt them at all. Caveat: this run had the file warm in
  the page cache from the previous run, which flatters it — but the direction is unambiguous,
  repack is *not* load-bearing for their advantage. That escape route is closed; our 3.3x
  deficit is our own kernels and our own I/O, not a flag they get to use and we don't.
