---
topic: Bigtea vs llama.cpp head-to-head on the target machine — measured, retracted claims, and where we actually stand
status: resolved
links: [moe-landscape-2026-08.md, benchmarking-methodology.md, verify-before-citing]
---

Measured 2026-08-05 on the target machine (i7-13650HX, 15.71 GiB RAM, Windows 11).
llama.cpp build 2026-08-03 (`llamacpp-unsloth`). Bigtea `ticket/rust-core`.
Same prompt files for both, greedy (`--temp 0`), 12 threads.

## CORRECTION BLOCK (2026-08-05)

Three claims in `CLAUDE.md`, `memory/bigtea-runner-state.md`, and every prior session summary
were **wrong**. All three are retracted.

**Claim 1 — "llama.cpp refuses this class of model here."** FALSE.
llama.cpp runs Qwen3-30B-A3B (17.28 GiB) on this 15.71 GiB machine without special flags and
produces correct text. It mmaps the file and lets the OS page it.

**Claim 2 — the error `failed to allocate buffer of size 147169738752` proves it.** MISATTRIBUTED.
147,169,738,752 bytes = 137.06 GiB. That is **DeepSeek-V4-Flash**, not Qwen3-30B. The error was
never produced by the run it was cited against. It lived only in session summaries, never in a
doc, which is why it went unchecked for days.

**Claim 3 — implied: llama.cpp cannot run V4-Flash on this box.** FALSE.
The failure is **one default flag**. `--repack` (on by default) allocates a single 137 GiB
`CPU_REPACK` buffer outside the mmap. With `--no-repack` llama.cpp loads the 144 GB model in
12.3s and generates correct text at 0.45 tok/s.

Rule adopted: **a competitive claim is not citable until the competitor's exact command line and
its output are in a doc.** See [[verify-before-citing]].

## Qwen3-30B-A3B Q4_K_M (17.28 GiB) — the full ladder

Bigtea with fused attention and a 2048-token prefill block; llama.cpp with a fully warm page
cache. Both produce identical, correct output at every length.

| prompt tokens | prefill: Bigtea / llama.cpp | eval: Bigtea / llama.cpp |
|---|---|---|
| 565  | **27.64** / 23.55 | 1.40 / 3.19 |
| 2206 | **36.60** / 33.59 | 1.11 / 2.46 |
| 4395 | 38.40 / 40.25 | 1.07 / 2.16 |
| 8775 | 34.88 / 35.01 | 0.78 / 1.62 |

**Prefill now beats llama.cpp at 565 and 2206 tokens and matches it at 4395 (95%) and 8775
(99.6%).** Raising the block to 4096 at 4395 tokens gives 43.61 tok/s against their 40.25 — so
the crossover is a memory budget, not a wall.

**Generation is still ~2x behind** and that gap has not moved. Where the time goes at 4395
tokens with a 4096 block: 41.1s expert compute, 25.8s attention, 12.3s disk, 11.8s other. The
expert matmuls are genuine arithmetic at roughly 239 GFLOPS; the remaining lever is that
llama.cpp repacks weights for its Q4_K kernels and we do not.

Earlier in the session this table read 19.92/23.68/19.79/14.46 on prefill, and before the
session's optimisation work the 565-token figure was **1.20 tok/s**. Prefill is 23x faster than
where it started.

Memory: Bigtea holds 0.93 GiB resident + a 6.26 GiB expert cache ≈ 7.2 GiB. llama.cpp's peak
working set was 8.87 GiB, and it additionally benefits from the OS page cache holding most of
the remaining model — effectively ~11 GiB working for it. **Bigtea also bypasses the page cache
deliberately** (direct I/O), so on a model that nearly fits, it is competing with one hand tied:
the kernel can use all free RAM elastically, Bigtea reserves a fixed budget.

### The long-context hypothesis is dead

The prediction was that llama.cpp's page cache would thrash at long context while Bigtea's
bounded residency degraded gracefully. It does not happen. From 565 to 8775 tokens llama.cpp's
eval falls 49% (3.19 → 1.62) and Bigtea's falls 56% (1.48 → 0.65). Bigtea degrades *slightly
faster*, and the ratio between them stays roughly constant at 2.2–2.5x.

An earlier reading of this as "the gap is narrowing" was an artefact: llama.cpp's first run was
cold, later ones warm.

### The memory-pressure experiment FAILED — do not cite it

To test the design premise (a model far larger than available RAM) without waiting on the
V4-Flash port, 7 GiB was held resident by a ballast process, leaving 4.28 GiB free against a
17.28 GiB model. Results looked unremarkable — llama.cpp 25.33 prefill / 2.26 eval, Bigtea 19.58
/ 1.47, both close to their unpressured numbers.

**That is because the experiment did not work.** The ballast touched its pages once and then
slept, so Windows paged it straight back out to satisfy llama.cpp's demand. Bigtea, running
second, then probed 6.66 GiB as "available" — more than the ballast was supposedly holding,
which is the tell. No pressure was ever applied and these numbers measure nothing. A valid
version needs `VirtualLock`ed pages, and continuously re-touching them would burn the memory
bandwidth being measured.

## DeepSeek-V4-Flash UD-Q4_K_XL (144 GB, 5 shards)

| flags | result |
|---|---|
| default | **fails**: `failed to allocate buffer of size 147169738752` → `failed to allocate CPU_REPACK buffer` → `unable to create context` |
| `--no-repack -c 512` | **runs**: load 12.3s, prefill 0.41 tok/s, eval **0.45 tok/s**, correct output |

Bigtea cannot run this model — the architecture is not implemented. **This is the only regime
where Bigtea's design should win, and it is exactly the one we cannot yet measure.** At 144 GB
against 15.7 GiB the page cache holds under 8% of the model, so llama.cpp's 4 KiB demand-paged
faults are competing against Bigtea's ~0.9 MiB sequential direct reads with an explicit
frequency-based policy. That comparison is the whole thesis and it remains untested.

Also worth noting: 0.45 tok/s means a 500-token answer takes 18 minutes. llama.cpp *loads* the
model; nobody can *work* with it. "Can this machine run it" is answered. "Can this machine run
it fast enough to code with" is not, by anyone.

## What was fixed on 2026-08-05, and what each was worth

Every number measured on this machine, before and after.

1. **Expert matmuls ran on one thread.** `compute()` floors its thread count at 1 and the expert
   path passed 0, so the bulk of the model's arithmetic used one core of twelve.
   Prefill at 2206 tokens: 11.21 → 18.62 tok/s.
2. **Expert slices were copied twice per use** — once out of the cache, once into the binder,
   which took a `Vec` and boxed it. ~1 GiB of memcpy per token for bytes that never change.
   `WeightSet` now holds `Arc<[u8]>`. Generation: 0.98 → **1.66 tok/s**, and 6s of a 16s run
   disappeared. This was the single largest win and it was invisible until profiled.
3. **Experts were re-read per token during prefill.** A 565-token prompt cost 609,665 expert
   reads and 537 GiB of disk for a model whose experts total 16.35 GiB. Grouping a block's
   tokens by expert and reading each once: prefill 1.20 → 9.08 tok/s, reads → 42,848,
   disk → 37.89 GiB.
4. **The expert cache was hardcoded at 1 GiB** while ~10 GiB of RAM sat unused. Now sized from
   measured free RAM.
5. **Cache policy was recency-based, which is the worst possible choice here.** The access
   pattern is a cyclic scan over 16.35 GiB of experts; when the cycle exceeds the cache, layer 0
   is always the oldest thing present when layer 47 needs room, so it is evicted just before the
   next block asks for it. A 6.26 GiB LRU-ish cache returned a **17% hit rate with 20,975
   evictions** — worse than pinning an arbitrary third would have been for free. Frequency-gated
   admission (a newcomer must be wanted strictly more often than what it displaces): **70% hits,
   0 evictions**.
6. **Single-token generation built 1,152 ggml graphs per token**, one per expert matmul, each a
   single column wrapped in a 12-thread barrier. With one position all expert outputs are
   `n_embd x 1`, so they are scaled by routing weights and summed inside one graph — one compute
   per layer instead of 24.

Supporting fixes: arenas computed from actual shapes rather than fixed constants (attention holds
`n_total * n_new * n_head` floats twice, which blows a 512 MiB arena past ~1.5k tokens and ggml
*aborts* rather than erroring); `arena_for` reserves 16 MiB not 1 MiB because
`ggml_graph_compute_with_ctx` allocates the graph object from the same arena and `ggml_new_graph`
always builds a 2048-node graph (3,060,816 bytes measured); the vocabulary projection runs for
the last position only (151,936 rows, previously computed for every prompt token to produce
logits nothing reads).

Net effect on the same 565-token prompt: prefill **1.20 → 19.92 tok/s (16.6x)**, generation
**0.88 → 1.48 tok/s**.

## Expert cache size sweep — more cache is not better

2206-token prompt, 8 generated, `--cache` forced. Same machine, ~11.5 GiB free.

| cache | hit rate | disk read | prefill tok/s | eval tok/s |
|---|---|---|---|---|
| 1 GiB  | 8%  | 56.27 GiB | 21.85 | 0.92 |
| 3 GiB  | 23% | 47.53 GiB | 22.13 | 1.02 |
| **6 GiB** | 41% | 36.21 GiB | **23.17** | **1.08** |
| 9 GiB  | 61% | 24.54 GiB | 17.59 | 0.95 |
| 11 GiB | 71% | 17.84 GiB | 13.95 | 0.76 |

Hit rate rises monotonically and disk traffic falls by 3.2x across the range — and past 6 GiB
the runner gets *slower*, ending 40% down on prefill at its best-ever hit rate. **The cache wins
the metric it optimises and loses the one that matters.**

The cause is that beyond ~6 GiB we are bidding against the OS for the same pages. Our cached
bytes get paged out, so a "hit" returns memory the kernel has to fault back in from disk — a
disk read with extra bookkeeping, counted as a hit. Hit rate stops being a proxy for speed the
moment the cache does not fit in physical RAM.

This validates the 4 GiB headroom default, which produces a 6.26 GiB cache on this machine —
within noise of the measured optimum. It also means **hit rate must never be reported as a
success metric on its own**; only tok/s at a given footprint says anything.

## Fused attention

Explicit attention materialises an `n_kv * n_batch * n_head` score matrix, reads it back for the
softmax and again for the value product. At 4395 tokens with a 512-token block that is 288 MiB
written and read twice, per layer — measured at about **4 GFLOPS**, an order of magnitude under
what the arithmetic alone costs. It is memory-bound, not compute-bound.

`ggml_flash_attn_ext` keeps the running softmax in registers and never builds the matrix.
Attention at 4395 tokens: **38.8s → 25.8s**. The larger effect is indirect: the arena falls from
~1.3 GiB to ~100 MiB, which is what makes a 4096-token prefill block affordable, and the block
size is worth more than the kernel (30.5 → 43.6 tok/s going from 512 to 4096).

Two traps, both of which produce silent nonsense rather than an error:
- **V is not transposed** for `flash_attn_ext`, unlike the `mul_mat` path it replaces.
- **The mask must be F16 and contiguous** (ggml asserts). Since the only values are 0 and -inf,
  the bit patterns `0x0000` and `0xFC00` are written directly — no conversion, and -inf stays
  exact.

## Honest position

- "Runs models larger than RAM" is **not a differentiator**. mmap has done it for years, and
  llama.cpp does it for a 144 GB model on this laptop today.
- **Prefill is now competitive and sometimes faster.** That is real and reproducible, but it is
  prompt processing — it does not make generation usable.
- Generation remains ~2x behind on a model that nearly fits, and the kernel's page cache is
  elastic and free where ours is fixed and hand-managed. Expect to keep losing that one here.
- Bigtea's measured advantage is memory: 7.2 GiB against llama.cpp's 8.87 GiB working set plus
  page cache. That only matters to a user if it buys responsiveness, which is unmeasured.
- The one place the design should win — model ≫ RAM, where cache policy dominates and we have
  direct evidence that frequency beats recency 70% to 17% — cannot be tested until Bigtea runs
  a model that large.

## Where generation time goes, and one lever ruled out

At 32 generated tokens: **12.6s expert compute, 5.8s disk, 3.9s other, 1.5s attention.** Expert
compute is 60% of it.

Those matmuls are single-column against Q4_K weights. Sweeping the thread count they run on
(`BIGTEA_EXPERT_THREADS`) changes almost nothing:

| threads | eval tok/s | expert compute |
|---|---|---|
| 12 | 1.54 | 9.8s |
| 8  | 1.55 | 9.8s |
| 6  | 1.57 | 9.6s |
| 4  | 1.57 | 9.6s |

Dropping from 12 threads to 4 costs nothing, so generation is **not** limited by the thread
barrier. Nor is it limited by memory bandwidth: 967 MiB of expert weights per token in ~0.4s is
about 2.4 GB/s, far under what this DDR5 delivers. What is left is **Q4_K dequantisation** —
the work of unpacking 4-bit weights before the dot product.

That is exactly what llama.cpp's weight repacking addresses (its `REPACK = 1`), interleaving
rows so several dequantise under one SIMD operation. **Repacking expert slices once on cache
admission, then binding the repacked layout, is the single remaining lever on generation** — and
it fits this design unusually well, because a cached slice is repacked once and reused across
every token that routes to it.

### The obstacle, checked 2026-08-05

**ggml does not expose repacking publicly.** There are no `_R4`/`_R8` entries in the `ggml_type`
enum (`ggml.h`: Q4_K is 12, the enum ends at 43 with no repacked variants), and nothing matching
`repack` appears anywhere in `ggml/include`. Upstream applies it through *extra buffer types*
inside `ggml-cpu`, selected during `ggml-backend` buffer allocation — a path Bigtea bypasses
entirely, because binding a weight means pointing a tensor's `data` at memory we already hold.

So this is not a matter of calling one more function. Two possible routes, neither cheap:

1. Route cached expert slices through `ggml-backend` buffer allocation so the extra buffer type
   applies. Costs a copy per slice — acceptable here, since a cached slice is copied once and
   reused for every token that routes to it — but the extra-buffer-type selection API is not
   public either, so it may mean linking against ggml-cpu internals.
2. Write the interleaved Q4_K dequant/dot kernel ourselves in AVX2. Full control, no reliance on
   ggml internals, but it is a real kernel with a real correctness risk — and a wrong one gives
   fluent nonsense rather than a crash, like every other numerical mistake in this project.

Route 1 first, and only fall back to 2 if the buffer-type API proves unreachable. Either way this
is a multi-session piece of work, not an afternoon's.

## Open questions

- **The V4-Flash port is now the critical path**, not a nice-to-have. It is the only way to test
  the thesis. llama.cpp's 0.45 tok/s there is the bar.
- Does bypassing the page cache still make sense when a model nearly fits? A runner that picked
  its I/O mode from the model-size-to-RAM ratio would use the kernel's cache when it helps and
  direct I/O when it would only double-buffer.
- Our 4 GiB headroom is conservative: Bigtea uses 7.2 GiB where llama.cpp effectively uses ~11.
  How much of the remaining gap is just that? Untested.
- Where does generation time go now? At 565 tokens: 3.2s disk, 5.4s expert compute, 1.0s
  attention, ~3.3s unattributed. The expert compute is single-column matmuls against Q4_K
  weights — llama.cpp repacks for this and we do not.
