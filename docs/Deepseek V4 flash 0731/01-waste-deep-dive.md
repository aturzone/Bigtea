# 01 — WASTE, deep dive

Source of record: sqliteai/waste. Every number below was measured by the WASTE authors on the
commit it ships with (MacBook Pro M5 Pro, 64 GB, internal NVMe), unless marked as an estimate.
Sizes are powers of two (GiB) the way the engine reports them.

## 1. What it is

WASTE ("Weight-Aware Streaming Tensor Engine") is an **embeddable MoE inference engine in C11**,
~6,000 lines in `src/`, zero third-party runtime dependencies (just libc + pthreads), Apache-2.0.
The proof point: the complete open-weights **Kimi K3 (2.78T parameters)** runs as a 982 GiB
container on a 64 GB MacBook at **0.49–0.54 tok/s**. "This is not a distilled, pruned, or
reduced variant."

The core bet: **a MoE model activates ~4% of itself per token, so almost all weight can live on
disk** — only what a token routes to needs to be *reachable in time*. WASTE keeps the dense,
repeatedly-used part (the "trunk") resident in RAM, streams selected experts off NVMe, and spends
every remaining byte of RAM on a bounded expert cache.

| Model | Container | Minimum RAM | Tested speed |
|---|---|---|---|
| Kimi K3 2.78T | 982 GiB | 29.05 GiB | 0.49–0.54 tok/s |
| Kimi-Linear 48B | 19 GiB | 1.87 GiB | 10.7 tok/s |

Three commitments shape the design (from their CLAUDE.md):

1. **Disk I/O is the budget, not arithmetic.** ~55% of a K3 decode step is expert reads.
   Optimizations are judged on bytes read per token and cache hit rate, not FLOPs.
2. **RAM is a hard ceiling, not a hint.** `ram_budget_bytes` bounds everything the engine
   allocates. Exceeding it means the OS pages, and a paged "cache hit" is slower than the disk
   read it replaced.
3. **Correctness is measured against an oracle** — every layer diffed against a PyTorch
   reference, not asserted.

## 2. The design that makes it fast

- **Placement decides speed.** One expert's gate/up/down matrices are adjacent in a single
  4 KiB-aligned record, so routing to an expert costs exactly **one `pread`** — not three, not a
  seek per matrix. K3's record is 12,406,784 bytes = exactly 3029 pages, which is what makes
  direct I/O possible.
- **Cache-bypass reads.** `F_NOCACHE` (macOS), `O_DIRECT` (Linux), `FILE_FLAG_NO_BUFFERING`
  (Windows). Deliberate: with a container smaller than RAM the kernel caches everything and
  every measured hit rate is the *kernel's*, a fiction that does not survive contact with a
  982 GB model.
- **The expert cache is the engine's, not the OS's.** Bounded LFRU (frequency-first — LRU
  collapses to 5.1% where LFRU keeps 29.4%). Read-ahead on two threads hides expert I/O behind
  the matmuls (~1.6× measured).
- **Never dequantize an expert.** Experts are residual-vector-quantized (3 stages × 256-entry
  codebooks over 8-dim vectors, 3.00 bits/weight, one f16 scale per output row). `sum_s C_s[i]·x_v`
  depends only on (stage, code, vector position), never on the output row — so the engine builds
  a per-token table of partial dot products and every expert row is **3 table reads + 2 adds**.
  Dequantization went from 87.5% of the time to zero.
- **The trunk stays at 4/8 bits.** K3 was QAT-trained on the *experts only*, so it has no
  trained tolerance for a squeezed trunk (see refuted levers).

## 3. The container format (v0, `docs/FORMAT.md`)

A `.waste` model is a **directory**, not one file (shard-friendly, resumable conversion):

```
model.waste/
  manifest.json       # format_version, arch, config (verbatim), expert_quant, layers, trunk index
  trunk.bin           # resident dense part (KDA/MLA attn, routers, shared experts, norms, head)
  experts-L{n}.bin    # one expert bank per layer
  codebooks.bin       # VQ codebooks, resident
  tokenizer.model / specials.json / vision.json / chat.json
  usage.waste         # runtime-appended routing stats / learned hotlist
```

Key properties worth internalizing:

- `format_version` is enforced — a container from another version is refused.
- **Containers are untrusted input**: the manifest is a hardened parser; every record's header is
  validated O(1) on the read path (magic, "the expert the index asked for", offsets that fit).
  The payload `crc32` is `--verify` / `WASTE_VERIFY=1`, **off by default** (≈5% on Kimi-Linear,
  ≈1% on K3). A damaged record stops generation and names the record instead of answering from
  wrong bytes.
- The two format items that are **specified and not implemented**: a shared low-rank basis
  (measured, doesn't pay: 0.12 bits for 0.3 pp) and a 1-bit SUB1 substitute bank (HOBBIT-style).
- `usage.waste` is a runtime-learned hotlist (LFRU preload; cold 1602→warm 1175 misses on
  Kimi-Linear, 61%→72% hit).

## 4. The memory model — the most important part for Bigtea

### The floor is one token's working set

The single most predictive number in the project. K3 touches 16 experts × 92 layers per token =
**17.0–17.4 GB**. Below that, an expert cached for one token is evicted before the next asks for
it, and the hit rate is **exactly zero — not "low"** (measured: 0% at a cache below one working
set; 2604 evictions in 2704 accesses).

### The ceiling is the OS paging cliff

Above a whole multiple of the working set, the machine pages and a cache "hit" becomes a page
fault — slower than the `pread` the engine was managing. Measured sweep (46 GB is the top):

| budget | expert cache | hit rate | decode |
|---|---|---|---|
| 32 GB | 3.32 GB | 0% | 0.31 tok/s |
| 46 GB | 17.32 GB | 13% | **0.32 tok/s** |
| 52 GB | 23.32 GB | 27% | 0.11–0.14 tok/s |
| 58 GB | 29.32 GB | 37% | **0.04 tok/s** |

An optimization that freed 1.11 GB (embedding table off the resident set) fed straight into the
cache at a fixed 58 GB budget and turned 0.32 tok/s into **0.04** — *freeing memory made it
eight times slower*, because the freed memory went somewhere the OS could take it back.

### The budget resolver (the copy-worthy bit)

- A budget under the floor is **refused** (`WASTE_E_RAM_BUDGET`), never swapped into.
- Cache is only worth anything in **whole multiples of one token's working set**. The default
  starts from the container's recommendation (`floor + 3×`) and steps down a multiple at a time
  — 3×, 2×, 1×, floor — taking the largest that fits under **7/8 of physical RAM**.
- The default therefore does *not* fill the machine. K3 asks for floor+3× (80.63 GB) and gets
  floor+1× = **46.24 GB** on this laptop, a 17.56 GB cache — the top of the measured curve, no
  flag needed. A 128 GB machine still gets the full 3×.
- The earlier default that filled the machine put the out-of-the-box run *inside the cliff*.

### The RAM-budget accounting that was wrong first

- The trunk was loaded twice at first (57 GB peak before one token); now streamed via `pread`,
  load 34s → 20s.
- Scratch was a guess; K3's real floor is 30.38 GB, not the 29.69 the first planner claimed.
- Method note: **a test on the small model does not test the big one** — the budget check ran
  green for weeks on Kimi-Linear because its scratch is measured in megabytes.

### KV cache: absorbed MLA latent

K3's attention is 3:1 KDA : MLA. The MLA layers cache the 512-wide latent instead of expanded
per-head K/V, with `kv_b_proj` absorbed into query and output. **Identical logits to 1.2e-05,
53× less cache**: 11.25 GB → 0.21 GB at 4K context; 360 GB → 6.75 GB at 128K. This is what makes
long context possible at all.

## 5. The numbers table (K3, end to end)

- Floor: 29.05 GB @4K context; 30.54 @32K; 35.63 @128K; 83.21 @1M.
- Resident trunk 27.28 GB; read per token 17.0 GB (read-ahead, 2 threads).
- Model load 20s; prefill 0.47 tok/s chunked / 0.29 sequential (pre-read-ahead).
- Decode 0.49–0.54 tok/s at the default budget.
- Where the time goes (read-ahead on): expert I/O **54.8%**, expert matmul 27.2% (LUT apply
  23.9%), KDA 9.3%, LUT build 2.7%. The reads are still **twice** the arithmetic.
- Kimi-Linear: 10.7 tok/s at an 8 GB budget, 78% cache hit.

## 6. Refuted levers — "do not rebuild these" (LEARNED.md + EFFICIENCY.md)

Each was measured, not argued:

| Idea | Verdict (number) |
|---|---|
| **2-bit experts** | 34% weight error vs 19.4% at 3 bits — 2-bit unsafe |
| **3-bit trunk** | Gets the better hit rate (29% vs 12%) yet 1.4× slower, and logits land 36% off; generation collapses (`+` and spaces). K3's QAT covered experts only — the trunk has no trained tolerance. Quality wall sits in front of the speed wall |
| **Per-expert bit allocation** (GEMQ) | Delta between experts is flat (1.06–1.15× in a layer, **1.01× across layers**, both models). An optimal allocator and a coin flip write the same container. Routing-frequency split saves disk and ~0% of the reads (cold experts are not read) |
| **Shared low-rank basis** (KBVQ) | 0.12 bits for 0.3 pp; loses badly at equal budget (28.9% vs 15.2% at 4.01 bits); Kimi experts are nearly mutually orthogonal (overlap 0.046 vs random 0.031) |
| **Purgeable cache** (`WASTE_PURGEABLE`) | 6× faster at an over-large budget, **1.6× slower at the budget that works** — macOS reclaims volatile objects eagerly. "Volatile memory is memory you have given away." Kept off by default as a bad-budget escape hatch |
| **mlock** (`WASTE_MLOCK`) | Does not raise the ceiling; removes the variance (38%→12% spread). Wiring the *trunk* (read in full every token), not the cache, is what helps. Off by default (Linux `RLIMIT_MEMLOCK` = 8 MB) |
| **Stage-major records** (2-of-3 stages per read) | The router has no tail to demote: ranks 9–16 carry 33.3% of routing mass, so demoting them costs 19.5%→24.9% error for 16.7% of reads — the same bad straight line |
| **Cross-layer predictive prefetch** | Recall@16 of L+1 from layer L = 29.0%, which does **not beat** the previous token's set (29.5%) the cache already gets free; even the overfit ceiling (49.7%) loses. Wrong prefetch displaces a needed read — would make K3 *slower* |
| **Batching / speculative decoding** | Grouping tokens removes 76% of I/O and 0% of compute; ceiling 1.63×; doesn't compose with read-ahead. No MTP head in the open release anyway |
| **Index-layout blocking** | 1.44× in a microbenchmark, nothing in the real engine (microbenchmarks lie about systems) |
| **int8/SDOT** | Only fits the trunk (dense dots); trunk is ~16% of a token so Amdahl caps it at ~13%, which f32-math matches without quantizing activations. Quantizing activations costs 4 orders of magnitude of accuracy |

**What shipped and moved the needle:** never-dequantize (fused VQ matvec, 2.15→0.22 s/token),
hoisted gate/up tables, read-ahead (0.32→0.49 tok/s, 1.5–1.6×; variance 39%→6%), absorbed MLA
KV (53× less cache), int8 trunk *storage* with f32 arithmetic (5.6 GB RAM freed).

## 7. Portability — the three-shim lesson

The whole engine is `src/platform.h`: **six calls** are not POSIX (positional read, aligned
allocation, CPU count, file size, cache-bypass open, threads). Windows cost, honestly reported:
10 of 13 TUs compiled unchanged; the real work was two things that compile cleanly — **`long` is
32-bit on Windows** (every file offset silently truncated to 2 GB on a 17 GB format), and the
archiver following `CC`. CI cross-compiles with MinGW-w64 and runs on `windows-latest`; what is
still not claimed is MSVC, ARM64 Windows, and the cache-bypass under load.

Backends: one dispatch table (struct of function pointers) filled with an always-compiled CPU
baseline; SIMD per ISA in its own TU (NEON/AVX2/AVX-512), chosen at runtime from CPUID+XGETBV;
accelerators are build-time options (Metal exists, correct, 22% slower — "this engine issues
several hundred small dependent matvecs per token, the worst possible shape for an accelerator").

## 8. The server and the security line

`serve/` is an OpenAI-compatible HTTP server (stdlib-only, ctypes into `libwaste`). Two things
are worth stealing:

- **Markup vs content is a security boundary, not a convenience.** `waste_tokenize_markup`
  resolves `<|open|>` to a control token; `waste_tokenize` treats the same bytes as text.
  Structure goes through the first, anything a user/document/tool wrote through the second.
  Concatenating and encoding once lets content forge a system message with real control-token
  ids. The prompt renderer is a *port* of the release's `encoding_k3.py` (K3 ships no Jinja
  template), checked segment-for-segment against it.
- **Honest defaults**: thinking on (what the model was trained for), requests serialize on one
  lock (`waste_ctx` is not thread-safe), a disconnect stops tokens immediately.

## 9. The docs culture — as important as the engineering

`docs/LEARNED.md` is **append-only and dated. Later wins.** Refuted ideas stay in, with the
numbers that killed them. The working rule: *before every long/expensive operation, run a cheap
real test that could kill it* (their "Gates" — Gate H saved a 1.4 TB download onto a disk that
cannot stream it). Wrong numbers are recorded as wrong, not quietly corrected. The README
carries no comparison table and invites counter-examples. This honesty is a competitive asset —
it is the reason the project is being taken seriously within a week of release, and it is a
standard Bigtea's own graph (T0 recorder-gate, 5% CV gate) is already positioned to match.
