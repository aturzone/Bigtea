# STATUS — where Bigtea is, and what is left

**Read this first, in any session.** It is the single place that says what is
true today. Update it in the same commit as any change that moves a number or
closes a task; if it disagrees with a doc, this file is wrong and the doc is
right, so fix this file.

**Last updated**: 2026-08-09 · **Version**: v0.0.2 · **Branch**: `main` ·
**Open PR**: R3 step 1 on `ticket/r3-kv-cache` — the KV cache's raw path.
**Unmerged, and its equivalence test does not pass yet** (see R3 below).
PR #43 (R0/R0.1/R1) is **merged**.

---

## In one paragraph

Bigtea is a Rust inference runner for models that do **not** fit in memory. It
keeps the always-read weights resident and streams routed experts from disk per
token, borrowing `ggml` for arithmetic while owning memory, residency, streaming
and the token loop. It runs DeepSeek-V4-Flash (144 GB) and Qwen3-30B-A3B on a
15.7 GiB laptop and produces correct text. **It is not yet faster than
llama.cpp on V4-Flash — on that model it leads on nothing.**

## The honest scoreboard

Never quote a comparison without the model name and the phase.

| model | phase | Bigtea | llama.cpp | verdict |
|---|---|---:|---:|---|
| **V4-Flash** | load | ~10s | ~10.5s | parity |
| **V4-Flash** | prefill | 2440 ms/tok | **1503 ms/tok** | **1.62x behind** |
| **V4-Flash** | generation | 0.064 tok/s | **0.21–0.31** | **3–4x behind** |
| Qwen3-30B-A3B | prefill @565 | **27.6** | 23.6 | ahead |
| Qwen3-30B-A3B | prefill @2206 | **36.6** | 33.6 | ahead |
| Qwen3-30B-A3B | generation | 1.07 | **2.16** | ~2x behind |

Sources, with both command lines and outputs:
`docs/graph/research/v4flash-vs-llamacpp-2026-08-07.md` and
`head-to-head-llamacpp-2026-08-05.md`.

**Two claims are retracted and must never be repeated**: that Bigtea leads
llama.cpp on V4-Flash load/prefill, and that llama.cpp cannot run models larger
than RAM. It runs the 144 GB model with `--no-repack`. "Larger than RAM" is not
the differentiator; **tok/s at a stated footprint under an owned residency
policy** is.

## What is done

- **v0.0.2 public**, Apache-2.0, CI green on Linux/macOS/Windows. 168 unit +
  16 container-backed tests. `clippy -D warnings` and `fmt` enforced.
- **V4-Flash port complete and verified** against llama.cpp element-sums: all 43
  blocks, all three attention builders, both routing schemes.
- **Prefill 2.2x faster** than Bigtea's own previous version (32.4s → 10.1s at 5
  tokens), via skewed direct reads, batched expert reads and 24→6 graph
  evaluations per block.
- **R0 answered** (2026-08-08): the router is genuinely skewed, but **the hot
  expert set is per-prompt and cannot be pinned**. It corrected four v0.0.2
  numbers and killed the model-pruning plan. PR #43.
- **R0.1 answered** (2026-08-08): **a set warmed on the prompt covers ~86% of
  what generation goes on to need** (86.3% on a code prompt, 85.9% on a prose
  one) — within ~4 points of an oracle and ~32 above the cross-prompt figure.
  This is what makes R1 worth building. **Over a longer horizon the cache must
  keep warming**: with the same prompt, frozen coverage falls 86.3% → 68.8% as
  generation goes 15 → 46 tokens, and warming recovers it to 75.8%. R0.1's
  "fill it and leave it" is withdrawn — it held only for the first ~20 tokens.
- **R1 built** (2026-08-08): frequency-gated expert cache wired into the
  deepseek4 path, sized from the probe, hit rate reported with footprint and
  tok/s. **But it cannot pay until R3 exists** — see the ordering note below.

## What is left, in the order the measurements justify

| id | work | state | why it is next |
|---|---|---|---|
| **R3** | KV cache | **step 1 built, NOT yet verified** — `ticket/r3-kv-cache`, fully scoped in `backlog/r3-kv-cache.md` | the unlock for everything else, not just a speed win. ~24 MB of state across **three** structures (the compressor ring is the one that is easy to miss). Verified without a new oracle: `prefill(0..n) then step(n)` must match `prefill(0..=n)` — argmax and a tolerance, **not** bit-identical, since routing already flips ~3% on near ties at a ggml blocking boundary. Test at 2, 5 and 165 tokens because each runs a different attention builder. Worth **~0.33 tok/s** from the measured 3.0s single-token pass alone, against llama.cpp's 0.21–0.31, and it is what makes R1 pay |
| **R1** | frequency-gated expert cache on the deepseek4 path | **built 2026-08-08, inert until R3** | implemented, tested against the oracle, sized from the probe, `--cache <GiB>` now works on this path. Warms on the prompt, never pinned. Cannot pay while a pass still reads ~123 distinct experts per layer |
| **R2** | overlap I/O with compute | ready, but smaller than it looks | per block it is ~53 ms read against ~23 ms compute, so the ceiling is ~1.4x — and all three expert tensors already read in one batched call, with everything after depending on them. Scoped against the code in the handoff |
| **R4** | fit the always-read set | user-side | 7.38 GiB; needs ~10.5 GiB free. Worth 0.7s/token. The runner already names the processes to close |
| **R5** | the product | not started | `bigtea pull`, quant selection from the probe, self-configuration, **OpenAI-compatible `/v1/chat/completions`**, prebuilt binaries |
| **R6** | run well on any machine | not started | one binary that reads the probe, configures itself, and says what tok/s to expect *before* doing anything |

**The order is not a preference, it is a dependency.** Expert reads are
deduplicated per block across the batch, so a pass reads the *distinct* experts
its tokens select. Measured on real prompts:

| tokens in the pass | distinct experts/layer | read per pass |
|---:|---:|---:|
| 1 (needs a KV cache) | 6 | **3.2 GiB** |
| 17 | 39.7 | 21 GiB |
| 166 | 122.8 | 66 GiB |

A cache of a few GiB cannot touch 66 GiB. Only once a step needs **6 experts per
layer** is the working set cacheable, and that is exactly what the KV cache buys.
So **R3 → R1 → R2**.

Detail for each: `docs/graph/backlog/next-session-handoff.md`.
Strategy and the bets beyond R6: `docs/graph/backlog/the-big-bang.md`.

## R3 in progress — what is built and what is open

**Built** (`ticket/r3-kv-cache`): `Deepseek4Cache` (raw latents + compressed
summaries, slot = absolute position), absolute positions threaded through all
four hardcoded sites, and `forward`/`step` as **one** code path — a prefill is a
step against an empty cache, so every existing test exercises the new machinery
rather than leaving the `pos0 != 0` branch unrun.

**All 14 llama.cpp oracle tests still pass through it**, so prefill is still
element-exact and the cache machinery itself is sound.

**One real bug caught before it shipped**: at `nt = 1` a compressed layer's
`fired` is `1 / ratio == 0`, so an incremental step fell back to Raw and
**silently dropped the compressed half of attention**. The guard now keys on the
*sequence* having completed a block, not the batch, and there is a test asserting
the refusal.

**Open, and the reason this is unmerged**: the equivalence test
(`prefill(0..n)` + `step(n)` vs `prefill(0..=n)`) fails.

```
n = 1 (2 tokens)   step 446595.72   full 445449.16   +0.257%
n = 2 (3 tokens)   step 399234.41   full 398126.00   +0.278%
```

argmax agrees at both, and the error is **flat, not accumulating** — systematic
from the first step rather than drift. That is the shape the routing-flip effect
would have, which `r3-kv-cache.md` predicted, but predicted is not demonstrated.
**The tolerance has deliberately not been widened**; the ticket names that as how
a real cache bug ships. Next step is to log the selected expert ids for the last
token in both paths and compare: differ → the flip is the cause and the assertion
becomes "argmax equal, at most N routing differences" with N stated; identical →
the cache is wrong and the sum is telling the truth.

Still to do after that: the compressor input ring (HCA then CSA), then the ring
wraparound that lifts the 256-token ceiling.

## Known limitations

- **V4-Flash is capped at 256 tokens of context. Confirmed 2026-08-08.**
  `attention()` builds one F16 cache of `kv_lora_rank * N_KV` = 512 × 256 and
  indexes it by absolute position. A 388-token prompt used to read weights for
  eight seconds and then panic with `range end index 198656 out of range for
  slice of length 131072` — 512 × 388 against 512 × 256. It now **refuses before
  reading anything**, with the limit and the reason. Every V4-Flash measurement
  this project has published is 5–198 tokens, which is why nothing caught it.
  The long-context prefill figures in the docs are Qwen3, a different path.
  **Lifting this is part of R3.**
- **No KV cache on the V4-Flash path**, so every generated token re-runs prefill
  over the whole sequence. The 0.015–0.064 tok/s generation figures are an
  artefact of that, not a measure of the engine. **A single-token pass costs
  3.0s** (re-measured 2026-08-08 with the whole always-read set resident), so a
  cached step is worth **~0.33 tok/s against llama.cpp's 0.21–0.31** — R3 alone
  turns a 3–4x deficit into a slight lead.
- **No GPU support** anywhere in the compute path.
- **No installer.** Building needs the GNU Rust toolchain, MSYS2 and a
  hand-built ggml. There are no prebuilt binaries and no model downloader.
  **Windows binaries are now redistributable** (2026-08-08) — the GNU C++ and
  OpenMP runtimes link statically, so the `.exe` needs only system DLLs. Before
  that it died with `0xC0000135` before `main` on any machine without MSYS2,
  silently. The CI release job is still to write.

## Things that are true and cost time to rediscover

The full list is in `CLAUDE.md` under *Facts that cost time to rediscover*. The
three that have burned the most time:

- **A wrong tokenizer or forward pass produces fluent nonsense, never a crash.**
  Test pieces separately, against an oracle.
- **ggml aborts on arena exhaustion** — no error to catch. Size arenas up front,
  and scale every one of them with the prefill block.
- **Cache hit rate is not a success metric.** Past ~6 GiB on Qwen3 a 71%-hit
  cache was the *slowest* configuration measured, because cached bytes got paged
  out and a "hit" became a page fault in disguise. Only tok/s at a stated
  footprint counts.

And the process rule this project has paid for twice: **a competitive claim is
not citable until the competitor's exact command line and its output are in a
doc, run in the same session as the number it is compared against.**

## How to resume

```bash
# ggml must be built first
export GGML_LIB_DIR=C:/Projects/llamacpp-unsloth/build/ggml/src
cargo test --release          # 168 tests (+16 container-backed, --ignored)
cargo build --release
./target/release/bigtea-probe --quick        # RAM/disk/GPU + what to close
```

Windows needs the **GNU** Rust toolchain and `C:\msys64\mingw64\bin` on PATH —
Git Bash's own `/mingw64` is not MSYS2's and has no `gcc`, which shows up as
`cannot find -lgomp` at link time.

Models are at `C:\Projects\models\` (v4flash 144 GB / 5 shards, qwen3moe 17.28
GiB, qwen3-4b 2.33 GB). **Do not download more without asking** — limited home
internet.

## Hardware this is measured on

15.7 GiB RAM (typically 3–10 GiB free), NVMe at **2.37 GiB/s** measured,
RTX 3050 6 GB laptop. **No GPU code exists** — `bigtea-probe` detects the card,
nothing in the compute path touches it.

## Working rules

- Implementation goes on `ticket/<name>` branches + PR; **Atur merges.** Docs may
  go to `main`.
- Push with the token from `C:\Projects\.env` inline in the URL, output redacted.
  Never in git config, never echoed. Model files stay gitignored.
- Graph docs live in `docs/graph/`; read `INDEX.md`, then only the 2–3 nodes a
  task links to. Any node change updates its INDEX line in the same commit.
