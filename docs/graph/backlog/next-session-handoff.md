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

| | Chaos | llama.cpp |
|---|---:|---:|
| load | 10.0s | 10.5s |
| prefill, per prompt token | 2440 ms | **1503 ms** |
| generation | 0.064 tok/s | **0.21–0.31 tok/s** |

**Chaos leads on nothing on V4-Flash.** It leads on Qwen3-30B-A3B prefill at 565
and 2206 tokens. A claim of leading V4-Flash was published in v0.0.1 and
retracted the same day — see `v4flash-vs-llamacpp-2026-08-07.md`.

**Today's gains** (all against Chaos's own previous version, all verified
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

### R0 — Re-measure the skew before building anything on it  ✅ **DONE 2026-08-08**

Answered on eight prompts across four subjects:
`../research/routing-skew-is-per-prompt-2026-08-08.md`.

**The hot set is per-prompt. Warm it adaptively; do not pin it.**

- The skew is real — top-8 takes 5-7x what a uniform router would at the same
  sample size, on every prompt.
- But out-of-sample it collapses: a top-64 set pinned from one prompt covers
  **61.3%** of another on the same subject and **37.5%** across subjects, against
  **25.0%** for a *random* cache. Sampling noise costs under 2 points, so this is
  real divergence.
- Subject matters more than language: a Persian *coding* prompt routes like the
  English coding prompts, not like Persian prose.
- **Three v0.0.2 numbers are corrected**, including the 33.6 tok/s disk floor
  (actually 1.60) and the "~48 GiB desktop" claim (unsupported). See the
  correction block on `routing-skew-changes-everything.md`.
- Tooling is committed: `tools/routing/capture.sh` + `analyse.py`,
  and `CHAOS_ROUTING_DUMP=<path>` writes raw per-layer counts.

### R0.1 — Does prompt routing predict *generated*-token routing?  ✅ **DONE 2026-08-08 — yes**

`../research/routing-prefill-predicts-generation-2026-08-08.md`.

A top-64 set taken from the prompt alone covers **86.3%** of the routing of the
tokens generated after it — against a 90.8% in-prompt oracle and R0's 53.7%
cross-prompt figure. **The regime a cache actually operates in is the favourable
one, and R1 lands near the top of its range, not the bottom.**

- **Fill the cache during prefill and leave it.** Continuing to warm during
  generation is worth 0.7 points (87.0% vs 86.3%) — not worth the complexity.
- **No decay** over 15 generated tokens; it drifts slightly *up*.
- Projected disk floor on this laptop's reachable 4.28 GiB tier: **~1.3 tok/s
  against llama.cpp's measured 0.21–0.31.**
- Method: passes are kept apart in the dump, and because the model is causal
  `pass[k] - pass[k-1]` is exactly one generated token's routing. The analysis
  asserts every delta is non-negative and sums to 6 per layer.

### R1 — Frequency-gated expert cache  ✅ **BUILT 2026-08-08, and it is early, not wrong**

`../research/expert-cache-is-early-not-wrong-2026-08-08.md`.

Built, verified against the llama.cpp oracle, **off by default** — because
measured on today's engine it is a regression:

| run | cache | hit rate | prefill | generation |
|---|---|---:|---:|---:|
| 17 tokens | none | — | 18.2s | 0.049 tok/s |
| 17 tokens | 1.51 GiB | 4.1% | 19.3s | 0.050 tok/s |
| 166 tokens | none | — | **64.5s** | 0.015 tok/s |
| 166 tokens | 1.75 GiB | 1.9% | **75.3s** | 0.015 tok/s |

Both pairs produced **identical text**, so the cache is correct. It simply has
nothing to cache: a pass reads the *distinct* experts its tokens select, which is
**122.8 per layer, ~66 GiB** at 166 tokens, and 1.75 GiB is 2% of that. The 17%
slower prefill is the admission copy.

**It inverts the moment R3 lands** — a KV-cached step needs 6 experts per layer,
3.21 GiB, and R0.1 measured a prompt-warmed set covering 86% of it. Nothing about
the cache needs changing; the thing it feeds on does not exist yet.

Already handled inside it: sized from `--cache`/the probe, never pre-loaded,
owns its memory, frequency weighted **by selections rather than requests**
(reads are deduplicated, so unweighted counts tie and the cache freezes), and hit
rate reported beside footprint and tok/s.

### R2 — Overlap I/O with compute

Both engines read the same bytes from the same drive; llama.cpp's mmap lets the
kernel read ahead **while the CPU computes the previous layer**. Chaos reads,
waits, computes, reads: measured **2.3s I/O + 1.0s compute, strictly serial**.

- **R2.1** Within a block: `gate` and `up` are needed before `down`'s matmul, so
  `down` can stream while the first two compute. **No routing prediction needed.**
- **R2.2** Layers 0-2 route by token id → **knowable before any compute runs**.
  Three layers, zero speculation risk, proves the machinery.
- **R2.3** Layers 3-42: prefetch on the previous token's routing. A miss costs a
  wasted read, not a wrong answer.

**Scoped against the code, 2026-08-08** — read before estimating this.

Per block the split is roughly **53 ms of read against 23 ms of compute**
(2.3s and 1.0s over 43 blocks), so reads dominate and the ceiling on perfect
overlap within a block is ~1.4x, not 2x.

The awkward part is that **there is almost nothing to overlap with.** All three
expert tensors are already read in *one* batched parallel call —
`read_expert_slices` at `deepseek4_forward.rs:1262`, four readers, and batching
them together was itself a win (`d242f1c`). Everything after that read depends on
the weights it returns, and the next block's attention depends on this block's
FFN output, so there is no independent work sitting idle.

That leaves exactly three seams, in order of confidence:

- **R2.1** is real but smaller than it sounds. `down` is not consumed until
  `deepseek4_forward.rs:1333`, after `gate` (1298), `up` (1300) and `swiglu`
  (1302). So `down`'s third of the bytes can stream behind roughly half the
  block's compute — perhaps 10 ms of 53. **Caution: splitting one 3-tensor batch
  into two smaller ones may cost more read throughput than the overlap gains.**
  `0fd2036` already recorded parallel expert reads winning 1.25x in isolation and
  *losing* in the runner. Measure the split before building on it.
- **R2.2** needs `ffn_gate_tid2eid` looked up for the whole sequence up front,
  which is a table read — cheap, exact, and it proves the prefetch machinery on
  3 of 43 blocks.
- **R2.3** is where the real bytes are (40 of 43 blocks) and it is speculative.
  **Its hit rate is exactly what R0.1 measures**, so do not size it before R0.1
  reports.

**Do R3 first.** The KV cache removes whole passes rather than overlapping one,
and 0.064 tok/s is an artefact of re-running the sequence per token.

### R3 — KV cache  *(now the critical path)*

**Scoped against the code, 2026-08-08.**

State to keep, from the config rather than guessed: `kv_lora_rank` 512,
`N_KV` 256, `sliding_window` 128, `CSA_RATIO` 4, 43 layers. The KV is a low-rank
latent, so a position costs 512 F16 values:

```
43 layers x (256 raw + 256 compressed) x 512 x 2 B = 21.5 MiB
```

Today `attention()` rebuilds that F16 cache **from scratch every pass**
(`deepseek4_forward.rs:898`), converting the whole sequence's `kv_full` each
time. The cached version appends one position instead.

Three traps, all of which give fluent nonsense rather than an error:

- **Slot index is currently the absolute position.** The mask loop compares
  `key > query` and `query - key >= window` with `key` indexing the raw cache
  directly. Any ring buffer or window slide breaks that identity, and the mask
  must be rewritten in the same change, not after.
- **RoPE is applied at the absolute position** before the value enters the
  cache, so cached entries must never be re-rotated — but a wrap changes which
  slot holds which position.
- **The compressor works on blocks of `CSA_RATIO` = 4.** A generated token
  usually does *not* complete a block, so the compressed half updates on one
  step in four. Getting that cadence wrong is invisible at short lengths.

**⚠ 256-token ceiling — CONFIRMED 2026-08-08.** A 388-token prompt read weights
for eight seconds and then panicked: `range end index 198656 out of range for
slice of length 131072`, which is 512 x 388 against 512 x 256. **V4-Flash's
usable context today is 256 tokens**, and the long-context prefill figures in the
docs are Qwen3-only. `prefill` now refuses over-long prompts before reading
anything (`ArchError::ContextTooLong`), with a container test. **Lifting the
ceiling is part of this ticket** — it is the same allocation.

### R3 — KV cache, original notes

Generation re-runs the whole sequence per token, so 0.064 tok/s is an artefact.
**A single-token pass costs 4.0s** — that is what a cached step will cost.

- Raw window bounded at `sliding_window` = 128; compressed summaries at 256
  blocks; ~33 MB for all 43 layers
- **A wrong cache here yields fluent nonsense, never an error.** Needs its own
  oracle capture at two consecutive positions before it is trusted.

### R4 — Fit the always-read set

7.38 GiB; fits only above ~10.5 GiB free. Worth 0.7s/token when it does not.
Chaos already names the processes to close and the cost per token.

### R5 — The product (`lts-0-0-0.md` T1–T5, unchanged)

`chaos pull` from Hugging Face with resume and checksums · quant selection from
the probe with the tok/s prediction stated *before* a 144 GB download ·
self-configuration · **OpenAI-compatible `/v1/chat/completions`, the single item
that makes it usable from a coding agent** · prebuilt binaries.

### R6 — Sync to any machine  *(Atur's explicit requirement)*

One binary that reads the probe and configures itself: on 8 GiB, 16, 48 or 128,
pick the quant, the cache size, the prefill block, the I/O mode — and **say what
tok/s to expect before doing anything.** `chaos-model-info` already predicts;
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

1. ~~**Prune the model to its hot set.**~~ **DEAD — R0 killed it, 2026-08-08.**
   This was called the most promising item with no research risk. The premise was
   that 64 of 256 experts per layer carry 97.8% of decisions, so a 34 GiB
   container would lose 2.2% of routing. Measured **out of sample it loses 46%**:
   a pinned global top-64 covers only 53.7% of an unseen prompt's selections. A
   pruned container would route unseen prompts to experts it does not contain.
   The 97.8% was in-sample on the one prompt the set was chosen from.
2. **Two-tier precision.** Hot experts resident at 2-bit as a *predictor*, full
   precision fetched only when the router's weight for that expert is high. The
   top-1 expert carries most of the weight mass; the 5th and 6th contribute
   little and may tolerate low precision. Needs a quality measurement, not new
   theory.
3. **Domain-specialised hot sets.** **R0 promoted this.** The hot set *is*
   prompt-dependent, and subject drives it more than language — so a coding-only
   hot set is the right shape, and a coding agent is the target use. Measured:
   within-subject transfer is 61.3% at top-64 against 37.5% across subjects. Two
   coding prompts is far too thin a base to size one from; that needs a proper
   coding corpus.
4. **Use the 6 GB of VRAM as a second cache tier.** RTX 3050, 5682 MiB free
   measured. One expert index across all 43 layers is 0.535 GiB, so ~5.1 GiB
   usable holds **~9-10 indices**. **R0 caveat: it must be warmed, not pinned** —
   pinned it inherits the 37.5% cross-subject figure, barely above a random 25%.
   Three blockers before any of this is testable: no CUDA toolkit on this machine,
   the linked ggml has no CUDA backend built, and Chaos's zero-copy weight
   binding hands ggml a host pointer, which a device tier cannot do. Nothing in
   the codebase touches the GPU today — `chaos-probe` only detects it.
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
