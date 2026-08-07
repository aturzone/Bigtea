# Handoff — everything found, everything left, and the 20 tok/s problem

Written 2026-08-07 at the end of a long session. Start here next time, then read
only the two or three nodes a task links to.

---

## Where the project actually stands

**Released**: `v0.0.1` public, Apache-2.0, CI green on Linux/macOS/Windows, 157
unit + 16 container-backed tests, `clippy -D warnings` and `fmt` enforced. A
fresh clone builds in 22s and runs V4-Flash.

**Measured back to back against llama.cpp on DeepSeek-V4-Flash** (this is the
honest scoreboard; do not quote anything else):

| | Bigtea | llama.cpp |
|---|---:|---:|
| load | 10.0s | 10.5s |
| prefill, per prompt token | 2440 ms | **1503 ms** |
| generation | 0.064 tok/s | **0.21–0.31 tok/s** |

**Bigtea leads on nothing on V4-Flash.** It leads on Qwen3-30B-A3B prefill at 565
and 2206 tokens. A claim of leading V4-Flash was published in v0.0.1 and
retracted the same day — see `v4flash-vs-llamacpp-2026-08-07.md`.

**Today's gains** (all against Bigtea's own previous version, all verified
against the llama.cpp oracle): prefill 32.4s → 10.1s for 5 tokens (**2.2x**),
generation 0.042 → 0.077 tok/s, single-token pass 7.9s → 4.0s.

---

## The finding that matters most

`routing-skew-changes-everything.md`. **The router is violently skewed.**

```
top-1  of 256 experts (0.54 GiB)  →  12.1% of all selections
top-8         (4.28 GiB)          →  52.9%
top-16        (8.57 GiB)          →  70.4%
top-64        (34.27 GiB)         →  97.8%

uniform would give top-16 = 6.2%.  chi-square vs uniform = 7805 (uniform ≈ 255)
```

Every speed estimate this project ever made assumed uniform routing. **It is not
uniform, and that assumption was the basis of declaring 20 tok/s impossible.**

With a hot-set cache, bytes/token fall from 3.21 GiB to 72 MiB → **33.6 tok/s
disk floor**; compute floors at 27 tok/s (8.7 GFLOP/token at the 239 GFLOPS this
project has already measured on its own expert matmuls).

**20 tok/s is a cache-sizing problem, not physics.** It needs ~48 GiB (34.3 hot
experts + 7.4 dense + working), which is a desktop, not a server — a 3x
correction on the "you need 150 GB" claim.

**On a 15.7 GiB laptop it implies ~1.3 tok/s — about 4x llama.cpp**, and ~20x
what we manage today.

---

## Remaining work, in the order the measurements justify

### R0 — Re-measure the skew before building anything on it  *(one day, blocks R1)*

The skew is from **one coding prompt**. Far too large to be noise, but the
*shape* may be prompt-dependent.

- Several prompts across different domains (code, prose, maths, another language)
- **Exclude layers 0-2**: they route by token id via `ffn_gate_tid2eid`, not by
  learned gating, so their skew reflects the token distribution
- The question to answer: **is the hot set global or per-prompt?** Global → pin
  it. Per-prompt → warm it adaptively, which is a harder design.
- `BIGTEA_ROUTING=1` already prints the histogram.

### R1 — Frequency-gated expert cache on the V4-Flash path  *(the big one)*

The policy already exists in `stream.rs` for Qwen3, where it took hit rate 17% →
70%. It has **never been wired into the deepseek4 path**.

- Size it from `bigtea-probe`, not a constant
- **The cache must own its memory.** This project has already measured that past
  ~6 GiB on Qwen3 a 71%-hit cache was the *slowest* configuration, because cached
  bytes got paged out and a "hit" became a page fault in disguise. That is
  exactly why an mmap-based engine cannot do this and Bigtea can.
- Report hit rate **next to** tok/s and footprint, never instead of them

### R2 — Overlap I/O with compute

Both engines read the same bytes from the same drive; llama.cpp's mmap lets the
kernel read ahead **while the CPU computes the previous layer**. Bigtea reads,
waits, computes, reads: measured **2.3s I/O + 1.0s compute, strictly serial**.

- **R2.1** Within a block: `gate` and `up` are needed before `down`'s matmul, so
  `down` can stream while the first two compute. **No routing prediction needed.**
- **R2.2** Layers 0-2 route by token id → **knowable before any compute runs**.
  Three layers, zero speculation risk, proves the machinery.
- **R2.3** Layers 3-42: prefetch on the previous token's routing. A miss costs a
  wasted read, not a wrong answer.

### R3 — KV cache

Generation re-runs the whole sequence per token, so 0.064 tok/s is an artefact.
**A single-token pass costs 4.0s** — that is what a cached step will cost.

- Raw window bounded at `sliding_window` = 128; compressed summaries at 256
  blocks; ~33 MB for all 43 layers
- **A wrong cache here yields fluent nonsense, never an error.** Needs its own
  oracle capture at two consecutive positions before it is trusted.

### R4 — Fit the always-read set

7.38 GiB; fits only above ~10.5 GiB free. Worth 0.7s/token when it does not.
Bigtea already names the processes to close and the cost per token.

### R5 — The product (`lts-0-0-0.md` T1–T5, unchanged)

`bigtea pull` from Hugging Face with resume and checksums · quant selection from
the probe with the tok/s prediction stated *before* a 144 GB download ·
self-configuration · **OpenAI-compatible `/v1/chat/completions`, the single item
that makes it usable from a coding agent** · prebuilt binaries.

### R6 — Sync to any machine  *(Atur's explicit requirement)*

One binary that reads the probe and configures itself: on 8 GiB, 16, 48 or 128,
pick the quant, the cache size, the prefill block, the I/O mode — and **say what
tok/s to expect before doing anything.** `bigtea-model-info` already predicts;
it needs the skew model folded in, because a hot-set cache changes the
prediction completely.

---

## Brainstorm: 20 tok/s on **8 GiB** — where to attack next time

Atur's next target, recorded honestly with the arithmetic so the next session
starts ahead rather than re-deriving it.

**The wall**: 20 tok/s needs ~97.8% hit rate ⇒ ~34 GiB of hot experts at Q4. In
8 GiB total, the always-read set alone is 7.38 GiB. **Both have to shrink**, and
that is now the whole problem — a much better problem than "read 64 GiB/s".

Ideas, roughly by expected value:

1. **Prune the model to its hot set.** The skew says 64 of 256 experts per layer
   carry 97.8% of decisions. A container keeping only those is **34 GiB instead
   of 144** and loses 2.2% of routing. That is an offline repack with tooling this
   project already has, and it is a *shippable artefact* others could use. At Q2
   it is ~17 GiB. **This is the single most promising item and it has no research
   risk — only a quality measurement.**
2. **Two-tier precision.** Hot experts resident at 2-bit as a *predictor*, full
   precision fetched only when the router's weight for that expert is high. The
   top-1 expert carries most of the weight mass; the 5th and 6th contribute
   little and may tolerate low precision. Needs a quality measurement, not new
   theory.
3. **Domain-specialised hot sets.** If R0 finds the hot set is prompt-dependent,
   a *coding* hot set may be far smaller than a general one — and a coding agent
   is the target use. A 4 GiB coding-specific cache could plausibly beat a 34 GiB
   general one for that workload.
4. **Use the 6 GB of VRAM as a second cache tier.** RTX 3050 at ~200 GB/s is 80x
   this NVMe. 6 GB holds ~top-11 experts per layer ≈ 60% of selections. Combined
   with RAM this is a genuine three-tier residency problem, and nothing in the
   codebase touches the GPU yet.
5. **Speculative decoding**, ~2.2x, proven, independent of all the above, needs a
   draft model sharing V4-Flash's tokenizer.
6. **Not this**: contextual sparsity. Measured dead — V4-Flash's experts are 9.1%
   negligible, not 80%, because the router's 6-of-256 *is* this architecture's
   contextual sparsity.

**The honest framing to keep**: 8 GiB and 20 tok/s and *the full 144 GB model*
may not all three be satisfiable. 8 GiB with a **pruned 34 GiB container** at
20 tok/s is a different and much more plausible claim, and it is still "run
DeepSeek-V4-Flash fast on a small machine".

---

## Rules this session re-learned the hard way

- **A competitor's number has a shelf life.** Re-run it in the same session as
  the number you compare it against. Publishing a lead built on a two-day-old
  llama.cpp figure cost a public retraction today.
- **Measure the regime you care about.** The 1.9x graph-batching win was
  dismissed twice from *prefill* profiles and only appeared in a *single-token*
  measurement.
- **Check the assumption under the arithmetic.** "20 tok/s is impossible" was
  correct arithmetic on an unmeasured premise.
- Reasoning ahead of measurement is now **0 for 6** on this project.
