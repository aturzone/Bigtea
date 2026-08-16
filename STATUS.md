# STATUS — where Bigtea is, and what is left

**Read this first, in any session.** It is the single place that says what is
true today. Update it in the same commit as any change that moves a number or
closes a task; if it disagrees with a doc, this file is wrong and the doc is
right, so fix this file.

**Last updated**: 2026-08-15 · **Version**: v0.0.2 · **Branch**:
`ticket/r15-parity-discriminator` · **Open PRs**: #65.

## The parity scoreboard, re-scored under the discriminator (2026-08-15)

**All thirteen dense architectures re-swept**, eight prompts each, against
`llama-completion` at `--temp 0`, with the harness that separates *"the
reference wobbles"* from *"our answer is a third one"*:

| | prompts |
|---|---:|
| **exact** — byte-identical to llama.cpp's default | **102** |
| **near-tie** — byte-identical to one of llama.cpp's *own* no-op outputs | **2** |
| **outside the band** — a third answer | **0** |
| **FAIL** | **0** |

**13 of 13 models exit 0.** The two near-ties are both Phi-3, and one of them
reproduces llama.cpp's `-b 1 -fa off` output — a **composed** configuration that
neither flag alone accounts for, which is exactly the class the r14 session
identified and the single-flag probe could not see.

`qwen3moe` is the fourteenth and is **not** in that table: 2 exact, 4 near-tie,
2 outside. It stays off `VERIFIED_ARCHITECTURES`. What changed is the size of the
question — the evidence for a defect there is **2 of 8, not the 6 of 8** that two
sessions independently reported, because four of those six reproduce llama.cpp's
own output byte for byte. See the discriminator section below.

**What this is not.** It is evidence about *these eight prompts* on *these
thirteen models*. `starcoder2` once passed 3/3 while running the wrong
pre-tokenizer, and V4-Flash is not swept here at all.

**Current**: **526 tests**, clippy `--workspace --all-targets -D warnings` 0, fmt
clean. `#60`, `#63` and `#64` are merged; `#65` is open and green. The counts in
dated sections further down are what was true on their date and are left alone —
this line is the one to read.

**`VERIFIED_ARCHITECTURES` is thirteen** — `baichuan`, `internlm2` and `olmo`
added on `ticket/r14-architectures`, each diffed at **eight** prompts. Widening
the harness from three prompts to eight **found three bugs in code that had been
on `main` for weeks**, two of them inside entries already listed as verified:
Llama-3.1/3.2/3.3 rotated with the wrong RoPE (`rope_freqs.weight` was never
read), Falcon3 was prefilled one token short (no BOS, no `add_bos_token`), and a
USER_DEFINED token's raw `\n` was silently dropped by the byte decoder. Twelve
models re-run, **eleven at 8/8** (Phi-3 6 ok + 2 reference-unstable), 426 tests.
The rule those bugs cost: **"the reference disagrees with itself" is not a safe
verdict** — it compares the reference to itself and cannot see that *our input*
differed, and nine of eleven `unstable` verdicts this session were bugs. The
harness acted on it in `b2ad35f`: a mismatch now compares the **tokenized
prompt** first and reports FAIL on a different count, which catches all three of
those bugs in one test, and three near-ties in eight exits non-zero as a cluster.
**All twelve models were re-swept under the stricter script with every result
unchanged**, every model exiting 0. Details:
`research/eight-prompts-found-three-bugs-2026-08-11.md`.

**Everything is merged.** PR #55 brought R3, R7, R8 and R9 into `main` in one
merge — the KV cache, six architectures, four tokenizer families, 106 CLI flags,
weight repacking, the thread work and the first quality measurement this project
has had. #44, #56 and #57 closed with it; their branches are deleted.

In flight: `ticket/r10-grammar-and-overlap` — GBNF/JSON-schema constrained
decoding as a new crate, then **R2** (overlap disk reads with compute).

---

## In one paragraph

Bigtea is a Rust inference runner for models that do **not** fit in memory. It
keeps the always-read weights resident and streams routed experts from disk per
token, borrowing `ggml` for arithmetic while owning memory, residency, streaming
and the token loop. It runs DeepSeek-V4-Flash (144 GB) and Qwen3-30B-A3B on a
15.7 GiB laptop and produces correct text. **It is not yet faster than
llama.cpp on V4-Flash — on that model it leads on nothing.**

## Where we actually stand, measured 2026-08-11 in one session

**Speed is level. Coverage is not.** Both halves matter and they have different
answers, so they are stated separately.

### Speed — parity on every model measured today

Interleaved runs, same session, `--temp 0`, 401-token prompt for the dense
models. **Absolute tok/s drifts up to 25% with machine state, so only
within-session comparisons are quoted**, and each number below is a median of
the rounds actually run.

| model | phase | Bigtea | llama.cpp | verdict |
|---|---|---:|---:|---|
| Qwen3-4B (dense, fits RAM) | prefill | **76.5** | 69.3 | parity → ahead |
| Qwen3-4B | generation | **5.97** | 5.54 | **1.08x ahead** |
| Gemma-2-2B | prefill | 124 / 141 | 115 / 146 | parity |
| Gemma-2-2B | generation | 8.01 / 10.78 | 7.12 / 10.67 | parity → ahead |
| Qwen3-30B-A3B (streams from disk) | prefill | 1.70 | 1.77 | parity |
| Qwen3-30B-A3B | generation | 3.10 | 3.25 | parity (5% behind, inside the spread) |

**Three rows in the old scoreboard are now retracted as stale**, and all three
were deficits:

- Qwen3-4B "prefill 38.5 vs 111.2, **2.9x behind**" — now 76.5 vs 69.3.
- Qwen3-4B "generation 0.67 vs 5.90, **8.8x behind**" — now 5.97 vs 5.54.
- Qwen3-30B "generation 2.63 vs 4.21, **1.60x behind**" — now 3.10 vs 3.25,
  i.e. inside the noise. **The 4.21 was llama.cpp on a better day**: measured
  back to back today it runs 2.93–3.60 on the same command line, which is the
  clearest possible demonstration of why cross-session numbers are worthless.

**On the streaming model, run order dominates the result.** Whichever engine
runs second is slower — Bigtea 3.92 running first against llama.cpp 3.60, and
2.71 running second against 2.93. A warm-to-warm protocol (each engine twice,
compare the seconds) is the only one that says anything, and it says parity.

**Nothing here is a claim about V4-Flash**, which was last measured 2026-08-10
and is unchanged: prefill 1.25x behind, generation at parity. See below.

### Coverage — this is the real gap

| | Bigtea | llama.cpp | gap |
|---|---:|---:|---|
| architectures **diffed against the reference** | **13** | 141 declared | the big one |
| chat templates | 26 | 54 | half |
| CLI flags (long) | 119 | 182 | 63 |
| tokenizer families | 4 | 6 | rwkv, plamo2 |
| samplers | 16 | 20 | adaptive-p, infill, 2 lazy-grammar |
| GPU backends | **1** (Vulkan, *not verified*) | CUDA, Metal, Vulkan, SYCL, HIP | 4 |

The architecture number is not comparable as written: llama.cpp *declares* 141
and Bigtea's 8 are ones whose output was diffed token for token against it.
Nobody has checked all 141. But 8 is still 8.

**The honest one-line answer: on this machine, for CPU inference on the models
we support, Bigtea is as fast as llama.cpp. It supports far less.**

## The honest scoreboard

Never quote a comparison without the model name and the phase.
**All V4-Flash rows below were measured back to back on 2026-08-10** with 9.3 GiB
free, which is the first time the whole 7.38 GiB always-read set fitted.

| model | phase | Bigtea | llama.cpp | verdict |
|---|---|---:|---:|---|
| **V4-Flash** | prefill | 2060 ms/tok | **1644 ms/tok** | **1.25x behind** |
| **V4-Flash** | generation, 9 tok | 0.344 tok/s | **0.39** | 1.13x behind |
| **V4-Flash** | generation, 23 tok | 0.363 tok/s | — | cache still warming |
| **V4-Flash** | generation, 47 tok | **0.374 tok/s** | **0.39** | **96% — parity** |
| Qwen3-30B-A3B | prefill @565 | **27.6** | 23.6 | ahead |
| Qwen3-30B-A3B | prefill @2206 | **36.6** | 33.6 | ahead |
| Qwen3-30B-A3B | generation | **2.63** | **4.21 ± 0.28** | **1.60x behind** |

**The Qwen3-30B generation row moved twice on 2026-08-10 and both corrections
matter.** Bigtea went 1.07 → **2.63** (2.46x) purely from the thread tuner
choosing **one** thread for the expert matmuls. And the llama.cpp reference is
**4.21 ± 0.28** at its own best (`-t 4`), not the 2.16 previously recorded — so
this is still a **deficit**, and a re-measured one, not a win.

**V4-Flash generation went 0.064 → 0.374 tok/s in one day — 5.8x** — and the
deficit against llama.cpp went from 3-4x to **4%**. It is parity, not a win, and
must not be quoted as one.

The trend is the interesting part: 0.344 at 9 tokens, 0.363 at 23, 0.374 at 47,
with the expert cache's hit rate climbing 9.7% → 20.2% → 23.5% as it warms.
llama.cpp is flat because it has nothing that warms. **Longer answers should
favour Bigtea**, and that is measurable but not yet measured past 47 tokens.

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
| **R3** | KV cache | **working, verified** — `ticket/r3-kv-cache`, fully scoped in `backlog/r3-kv-cache.md` | the unlock for everything else, not just a speed win. ~24 MB of state across **three** structures (the compressor ring is the one that is easy to miss). Verified without a new oracle: `prefill(0..n) then step(n)` must match `prefill(0..=n)` — argmax and a tolerance, **not** bit-identical, since routing already flips ~3% on near ties at a ggml blocking boundary. Test at 2, 5 and 165 tokens because each runs a different attention builder. Worth **~0.33 tok/s** from the measured 3.0s single-token pass alone, against llama.cpp's 0.21–0.31, and it is what makes R1 pay |
| **R1** | frequency-gated expert cache on the deepseek4 path | **built 2026-08-08, inert until R3** | implemented, tested against the oracle, sized from the probe, `--cache <GiB>` now works on this path. Warms on the prompt, never pinned. Cannot pay while a pass still reads ~123 distinct experts per layer |
| **R2** | overlap I/O with compute | ready, but smaller than it looks | per block it is ~53 ms read against ~23 ms compute, so the ceiling is ~1.4x — and all three expert tensors already read in one batched call, with everything after depending on them. Scoped against the code in the handoff |
| **R4** | fit the always-read set | user-side | 7.38 GiB; needs ~10.5 GiB free. Worth 0.7s/token. The runner already names the processes to close |
| **R5** | the product | **started** | `bigtea pull`, quant selection from the probe, self-configuration, **OpenAI-compatible `/v1/chat/completions`**, prebuilt binaries |
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

## R3 — the KV cache works

**Generation no longer re-runs the sequence.** `bigtea-run` keeps one cache for
the session: the prompt fills it, each token appends a single row.

```
generate 5 tokens   0.145 tok/s   (6.9 s/token)     was 0.064
```

**2.3x, and measured under memory pressure** — 5.7 GiB free at the time, so only
3.42 of the 7.38 GiB always-read set was resident and 3.95 GiB was re-read on
every token. A single-token pass with the whole set resident measured **3.0s**
(2026-08-08), which is ~0.33 tok/s; that figure has **not** been re-measured
since the cache landed and should not be quoted as achieved.

llama.cpp on the same model is 0.21–0.31 tok/s. **We are not yet past it on a
measurement taken under equal conditions.** The next honest comparison needs
~10.5 GiB free on both sides.

### What it took, and two bugs that would have shipped silently

Both were caught by the equivalence harness (`prefill(0..n)` + `step(n)` must
match `prefill(0..=n)`), not by reading the code:

1. **The compressor ring.** `compressor` front-padded `state_rows` zeros where
   llama.cpp keeps a ring — exact on a prefill, where the previous window is in
   the batch, and a lie on a step. It now slides on *every* pass through a
   compressed layer, including the three in four that complete no block.
2. **`fired` was relative.** It asked `nt / ratio > 0`, which is zero for any
   single-token step, so a step built no summary *and* told `attention` there was
   no compressed half at all — discarding everything the sequence had compressed.
   Now absolute: `(pos0 + nt) / r > pos0 / r`. **This one measured 15.05% wrong
   with the argmax still agreeing**, which is exactly the failure mode that reads
   as fluent nonsense. After the fix: 0.090%.

Equivalence now holds on both paths — raw 0.278% apart, compressed 0.090%, argmax
equal on both, with the residual proven to be a near-tie re-route rather than a
cache fault (hash-routed layers, which cannot depend on batch shape, agree
exactly).

**Still open**: the 256-token ceiling (#46) needs the ring wraparound.

## R1 re-measured (#47): the cache pays, once residency is satisfied first

With the always-read set fully resident, the expert cache stops competing and
starts helping:

| run | cache | hits | generation |
|---|---:|---:|---:|
| 9 tokens | off | — | 0.310 tok/s |
| 9 tokens | 1.0 GiB | 9.7% | **0.344** |
| 23 tokens | 1.5 GiB | 20.2% | **0.363** |
| 47 tokens | 1.5 GiB | 23.5% | **0.374** |

Earlier, under memory pressure, the same cache *hurt*: a byte given to it came
out of residency, where it would have been read on every token. `bigtea-run`
refuses a cache while the always-read set is still streaming, and that rule is
now confirmed from both sides — it hurt at 2.43 GiB resident, it helps at 7.38.

**R0.1's ~86% is not reached, and 23.5% is not evidence against it**: that
figure is coverage of a prompt-warmed *set*, this is hit rate against a 1.5 GiB
budget holding ~1% of the model's experts. The measurement that tests R0.1
needs a much larger cache than this machine has spare.

## The byte budget, and why 20 tok/s is not a code problem (2026-08-10)

Generation reads **3.21 GiB per token**. 20 tok/s at the measured 1.58 GiB/s
direct-read rate allows **79 MB**. The gap is **42x**, and this session went
looking for it in the two places nobody had measured.

**Both were negative.** Full detail and controls:
`docs/graph/research/v4flash-has-no-slack-2026-08-10.md`.

| lever | worth | status |
|---|---:|---|
| expert-bank factorisation | 1.0x | **dead — measured, 1.2x from random noise** |
| drop the router's tail | ~1.2x | **costs 8.8% of routing mass — measured** |
| contextual sparsity | 1.1x | dead — experts are 9.1% negligible |
| pinned hot set | 1.0x | dead — R0, 37.5% vs 25.0% random |
| speculative decoding | 1.4x | real, but the docs' **2.2x does not transfer** |
| 4.25 → 2.5-bit experts | 1.7x | unproven on an MoE this size, quality-risky |
| warmed expert cache | 1.3x | measured at 23.5% hits with ~6 GiB spare |

Everything still alive, multiplied, is **3.1x**.

Three findings, all first measurements:

1. **The expert bank is full-rank.** `bigtea-spectrum` (new) asked whether all
   256 experts in a layer share a subspace — if they did, one resident basis plus
   small per-expert coefficients would cut bytes by `4096/r` *and* cut flops. A
   rank-512 basis holds **20.4%** of the bank's energy against **16.6%** for
   matched random noise. 1.23x from nothing, confirmed on two layers and two
   projections, and converged (10 power iterations move rank-256 from 11.4% to
   11.5%).
2. **The router's tail is not small.** Renormalised weights, sorted, mean over 43
   layers: **33.5 / 20.6 / 15.0 / 12.1 / 10.1 / 8.8%**. Uniform would be 16.7%.
   The standing assumption that "the 6th expert contributes little" is false —
   reading three instead of six buys 2x and discards **31%** of the routing mass.
3. **Speculative decoding is ~1.4x here, not 2.2x.** The literature's figure
   assumes the verify pass costs what a single-token pass costs. Here it costs
   more, because more tokens select more distinct experts (`U(n) ≈ 6·n^0.667`,
   from this project's own dedup measurements). Below α≈0.75 it is a net *loss*.

Together with the earlier 9.1%-negligible result that is four independent probes
and four negatives, which says something about the model rather than the runner:
**V4-Flash has no redundancy left to harvest.** Its experts are mutually
distinct, internally dense, and its router spreads real weight across all six.
The 6-of-256 is the whole of this architecture's sparsity and Bigtea already
exploits it.

**So 3.21 GiB/token is what this model costs, not an artefact.** 20 tok/s does
not need a better runner; it needs the active weights to stop coming from disk.
That makes the next question a measurable one nobody has published: **what is
the tok/s-versus-RAM frontier for a 144 GB model?** Bigtea can sweep it because
it owns residency; an `mmap` engine cannot be told to use exactly N GiB.

## The plateau was ours, not the drive's (2026-08-10) — 1.32x on expert reads

Two written-down "facts" were ceilings we had built. Full detail and both new
tools: `docs/graph/research/the-plateau-was-ours-2026-08-10.md`.

**Where a token actually goes**, measured with `BIGTEA_BLOCK_TIMING=1`:

| phase | before | share |
|---|---:|---:|
| dense always-read re-reads (disk) | 2.15 s | 39% |
| expert slice reads (disk) | 2.03 s | 37% |
| tail + graph overhead | 1.10 s | 20% |
| **expert matmul** | **0.18 s** | **3%** |

**76% of a token is disk; the arithmetic is 3%.** `bigtea-kernelbench` (new)
times the expert FFN with weights already in RAM: 3.02 ms per block at **24.7
GiB/s**, which is *above* single-threaded memcpy on this machine. The kernel is
at DRAM speed and there is nothing to win in it.

**All four readers shared one file handle.** A Windows handle without
`FILE_FLAG_OVERLAPPED` is synchronous and the OS serialises reads on it, so the
drive never left queue depth 1. `bigtea-iobench` (new), identical reads, one
variable:

| threads | shared handle | one handle each |
|---:|---:|---:|
| 4 | 2.01 GiB/s | **2.65** |
| 8 | 2.05 | **2.69** |

2.69 GiB/s is also above the 2.37 recorded as the drive's sequential ceiling.
Implemented: an 8-handle pool per shard, `READERS` 4 → 8, and `prefetch_dense`
reading a block's non-resident always-read tensors across the pool.

| | before | after | gain |
|---|---:|---:|---:|
| **expert slice reads** | 2.03 s | **1.54 s** | **1.32x** |
| dense re-reads, per GiB missing | 0.691 s | **0.496 s** | **1.39x** |

**1.32x on expert reads is the clean number** — independent of residency, and it
matches the bench's 1.31x prediction. The end-to-end rows (5.46 → 4.33 s/step,
0.182 → 0.227 tok/s) are **not** a clean A/B: the runs had 3.11 and 2.66 GiB
missing respectively. Normalised, the step gain is **1.19x**, and that is the
figure to quote. A clean end-to-end A/B needs stable free RAM and is not done.

This also corrects the speculative-decoding pessimism above: measured compute
scales as ~`n^0.49` in the batch, not linearly, so the byte table is a fair
estimate of total speedup rather than an optimistic one.

**Revised ceiling on this machine**: with residency satisfied and reads overlapped
with compute (R2, not done), a token is about `max(1.54, 0.6)` s ≈ **0.65 tok/s**
against llama.cpp's 0.39 — a real 1.7x lead rather than parity. Not 20 tok/s.
The remaining gap is entirely disk bandwidth against 3.21 GiB per token.

## Coverage: the Llama family now opens (2026-08-10)

Atur reset the goal: **standards-compliant, opens any model, matches or beats
llama.cpp on the criteria, all its options — then tag v0.0.X LTS, then 20 tok/s.**
The checklist that decides when LTS ships is
`docs/graph/backlog/lts-parity-criteria.md`; every row is done / gap / won't.

Coverage was the larger gap and had never been written down:

| | was | now | llama.cpp |
|---|---:|---:|---:|
| architectures | 3 | **5 families** | ~100 |
| tokenizers | 1 (`gpt2`) | **2 (`gpt2`, `llama`)** | 6 |
| chat templates | 0 | 0 | ~40 |
| samplers | greedy only | greedy only | ~10 |

**Verified on real containers, not fixtures:**

| model | architecture | tokenizer | output |
|---|---|---|---|
| TinyLlama-1.1B | `llama` | SPM | "The capital of France is **Paris.**" |
| Llama-3.2-1B-Instruct | `llama` | BPE | "**Paris.** The capital of Germany is Berlin." |
| Qwen3-4B | `qwen3` | BPE | unchanged — no regression |

Three things were refusing the Llama family, and two would have shipped silently:

1. **QK norm was mandatory.** `required_tensors` listed `attn_q_norm`/
   `attn_k_norm` on every block; llama, mistral, qwen2, gemma and phi do not
   have them, so the up-front check was a false negative on all of them. Now
   detected from the container.
2. **RoPE type was hardcoded to NeoX.** llama.cpp uses NORM for llama/mistral
   and NeoX for qwen/phi/gemma. Both run without error on either layout — the
   wrong one is fluent nonsense. Now chosen by architecture, and an
   architecture *not* on the list is **flagged as a guess** in the runner's
   output rather than silently defaulted.
3. **SentencePiece did not exist.** It merges by vocabulary *score*, not by
   merge rank; space is `▁`; unknown text falls back to `<0xXX>` byte tokens.

**One real bug the round-trip test caught**: decoding tokens one at a time is
unsound for any multi-byte character — an emoji is four byte-fallback tokens,
and Persian or Chinese characters are two or three, so each fragment became `�`
permanently. `decode_bytes` returns bytes and generation now buffers to a valid
UTF-8 boundary. **This affected the BPE path too**, so it was breaking non-ASCII
output on every model, not just the new ones.

## C2 chat templates — instruct models now actually answer (2026-08-10)

The single largest quality gap, and it was invisible because nothing errored.

**Same model, same prompt, greedy decoding, Llama-3.2-1B:**

| | answer |
|---|---|
| raw prompt (before) | *"The sentence should be concise and evocative, using sensory details…"* |
| `--chat` (after) | *"The vast expanse of the ocean stretches out before us, a seemingly endless blue canvas of waves, tides, and mysteries…"* |

An instruct model handed raw text does not fail — it **completes the
instruction instead of following it**. Every quality impression of this runner
so far was formed against that.

**Detection, not Jinja evaluation.** GGUF stores the template as Jinja2;
Llama-3's alone uses `set`, `if defined`, loops and tool-call branches. llama.cpp
does not evaluate them either — it matches known families by substring and
applies a hardcoded formatter, and so does this. Nine families: chatml, llama3,
llama2, mistral, zephyr, phi3, gemma, vicuna, alpaca. An unrecognised template
reports itself **not recognised** rather than borrowing someone else's framing.

Verified against the real templates in the containers on this machine:

| model | template detected |
|---|---|
| TinyLlama-1.1B | zephyr |
| Llama-3.2-1B | llama3 |
| Qwen3-4B | chatml |

**The invisible half — control tokens.** Applying the template changed nothing
at first. `<|start_header_id|>` was being run through BPE and split into `<`,
`|`, `start`, … — pieces the model has never seen in that position — so the
framed prompt was just characters and the model answered as if given raw text.
There is no error anywhere in that path. `encode` now partitions on the
container's CONTROL and USER_DEFINED tokens and maps each to its own id;
the framed prompt above is **17 tokens**, not 40-odd.

`bigtea-serve` now parses `messages[]` with roles in order and applies the
template, instead of concatenating the contents.

## C3 the server streams, samples and stops (2026-08-10)

`bigtea-serve` answered one way: greedy, no sampling controls, and
`finish_reason` was **always** `"length"` because nothing checked for
end-of-sequence. It also buffered the whole answer before sending a byte.

Now:

| | |
|---|---|
| `stream: true` | server-sent events, one per token, flushed each time |
| sampling | `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `repetition_penalty` |
| stopping | EOS **and** `stop` sequences → `finish_reason: "stop"` |
| `stop` | accepted as a string *or* an array, both spellings clients send |

Two details that would have been wrong quietly:

- **The default temperature is 1.0, not 0.0.** OpenAI's default is sampling;
  a client that sends no `temperature` does not expect greedy. `bigtea-run`
  keeps greedy as its default for the opposite reason — it keeps a wrong
  forward pass diagnosable.
- **Stop sequences are matched against the accumulated text, not the token**,
  because a stop string can straddle a token boundary.
- Streaming re-uses the UTF-8 buffering rule: a chunk is emitted only at a
  character boundary, so a multi-byte character never becomes `�` mid-stream.

### Chat framing against llama.cpp, both paths — one bug, four open (2026-08-15)

`scripts/jinja-vs-llamacpp.py` claims in its docstring to compare "Bigtea's Jinja
rendering against `llama.cpp --jinja`" and **never runs Bigtea** — it runs
llama.cpp twice, `--jinja` against `--no-jinja`. That measurement is real and
worth keeping (the reference disagrees with **itself** on 5 of 18 containers) but
it was being cited for a claim it does not test. Same failure as the `REFUSED`
row: a description that outlived the code.

`scripts/jinja-bigtea-vs-llamacpp.py` runs the four-way that does, on **token IDs
rather than rendered text**. It found a real bug on its first execution: **BOS was
being emitted twice** under `--jinja`, because the template contains the literal
`<bos>` *and* `encode` prepended one. gemma-3, Llama-3.2, internlm2, Phi-3 were
all prefilled a token **long** — the exact mirror of Falcon3, which was a token
short. Fixed; agreement went **4 → 6** of 14 loadable containers.

**A second silent bug, in the tokenizer.** A Phi-3 chat turn was **14 tokens
where llama.cpp makes 8** — identical input. llama.cpp drops whitespace
*following* a special token (`LLAMA_TOKEN_ATTR_RSTRIP`), and SPM's dummy prefix
then re-tokenizes the next word. **The attribute is not in the container**:
`llama-vocab.cpp` sets it from `_contains_any(model_name, {"phi-3", "phi3"})` —
the tokenizer's behaviour depends on `general.name`. Matched, with the same three
exemptions (`<unk>`, `<s>`, `<|endoftext|>`) and a test that any *other* model
keeps its whitespace. Agreement **6 → 7**; Phi-3 now matches on both paths.

**Neither bug was reachable from the parity sweep**, which uses plain prompts
with no special tokens — so none of the 104 prompts behind "102 exact" could have
found either, and both affect every chat-framed request the server handles. Two
different checks, two different bug classes.

Of the seven that still differ, three are models with **no chat template**
(`OLMo`, `starcoder2`, `all-MiniLM`), where llama.cpp passes the text through
untouched and we impose a `System:/User:/Assistant:` framing. Deliberate and
announced, but a divergence, and feeding a base model invented structure is the
mirror of the bug that made instruct models continue rather than answer. **Not
changed** — a product decision, recorded so it gets made rather than inherited.
One (`tinyllama`, family path) is us matching the model's template where
llama.cpp's hardcoded renderer does not. Three are genuine rendering differences
(`Falcon3`, `gemma-2`, `internlm2`). Full table:
`research/chat-framing-vs-llamacpp-2026-08-15.md`.

### `/v1/embeddings` — the fifth endpoint, implemented 2026-08-15

It answered **501** with a reason that was half true: *"this runner's graph
returns logits, not hidden states."* True of what the graph **returned**, false
about what it **computed** — the pre-projection hidden state is the input to the
vocabulary matmul and was being discarded one line later. A refusal that cites a
missing capability should be checked against the code, not against the last
person who wrote it down.

Taken **after `output_norm` and before the vocabulary projection**, which is
where llama.cpp takes it. Earlier, the vector carries a per-model scale that
makes similarity between two models meaningless; later, it is a distribution over
tokens rather than an embedding. Opt-in per pass (`set_want_embedding`), so
generation does not pay for a `compute` the sampler never reads.

**Verified semantically, not just structurally** — the old message warned that a
vector derived from the wrong place "would look like an embedding and behave like
noise", so returning 2048 plausible floats proves nothing:

```
cos(cat, dog) = 0.5867
cos(cat, SQL) = 0.2063
cos(dog, SQL) = 0.1585      L2 norms: 1.0, 1.0, 1.0
```

`input` is accepted as a **string or an array of strings**, both of which are in
real client code, and each input gets a **fresh KV cache** — sharing one would
make every vector after the first depend on the texts before it, and they would
still look plausible while silently encoding the batch order.

Still refused, by name: the **V4-Flash** path, whose forward pass does not expose
a hidden state. That is a different engine, not a missing line.

**The server now serves any supported architecture.** It refused everything
except V4-Flash, which made the one component an agent actually talks to
useless for the models people actually run. Verified end to end:

| model | template | result |
|---|---|---|
| Llama-3.2-1B | llama3 | `"Pacific Ocean"`, `finish_reason: "stop"` |
| TinyLlama-1.1B | zephyr | answers, SPM tokenizer, same binary |

`/v1/models` reports the container's own name (`Llama-3.2-1B-Instruct`), not a
constant. A stop sequence truncates correctly: asking it to repeat
"alpha beta gamma delta" with `stop: ["gamma"]` returns `"alpha beta "` and
`finish_reason: "stop"`.

One wire-format bug caught by looking at the raw bytes rather than trusting the
code: the SSE headers were being emitted with **leading whitespace**, because a
multi-line string literal in the source kept its indentation. `curl` tolerated
it; a stricter client would not.

## The first dense head-to-head (2026-08-10) — and it is a deficit

Every previous comparison was on a model that streams from disk, where I/O
dominates. **Qwen3-4B fits in RAM, so this is the first measurement of the
compute path on its own.** Both command lines and outputs:
`docs/graph/research/qwen3-4b-vs-llamacpp-2026-08-10.md`.

| Qwen3-4B dense, CPU, 20 threads | Bigtea | llama.cpp | verdict |
|---|---:|---:|---|
| prefill (matched, 519 vs 512) | **83.4 tok/s** | **88.3** | **1.06x behind** |
| generation (128 tok, 3 reps) | **4.3 tok/s** | **5.28 ± 0.33** (tg128) | **1.23x behind** |

*(The original 38.5 / 0.67 figures were taken on the uncached path with a
broken arena; both are superseded.)*

**The prefill gap is weight repacking, and nothing else.** Same file, same
prompt, `llama-completion` both sides:

| Qwen3-4B prefill | tok/s |
|---|---:|
| llama.cpp, repacking on (default) | **88.26** |
| llama.cpp, `--no-repack` | 63.68 |
| Bigtea | 60.29 |

**Without repacking the two engines are 6% apart** — expected, since both link
the same ggml. Ruled out by measurement on the way: thread count (8–20 all
within 10%), graph/threadpool overhead (~0.2% of the pass), and the matmul
kernel itself (our FFN runs at 472 GFLOP/s against a measured Q4_K ceiling of
420). Detail and the command lines:
`docs/graph/research/qwen3-4b-vs-llamacpp-2026-08-10.md`.

**Built, and now ON by default:**

| Qwen3-4B prefill, 519 tokens | tok/s |
|---|---:|
| llama.cpp | 88.3 |
| **Bigtea** | **83.4** |
| Bigtea, `--no-repack` | 58.6 |

**1.42x, and the prefill deficit goes 1.46x → 1.06x.** 216 tensors, 1.64 GiB
rearranged.

It reaches `ggml`'s repacked kernels without adopting `ggml-backend`: a tensor
allocated in the repack buffer type gets `tensor_traits` hung off its `extra`,
and `ggml_compute_forward` consults that **on the plain graph path too**.

**It defaults ON because it is the side that AGREES WITH llama.cpp** — which is
the opposite of how it first looked. Enabling it changed Llama-3.2's
continuation, which read as a regression until the reference was actually
consulted. Raw greedy completion, same container:

| prompt | llama.cpp | Bigtea repacked | Bigtea unpacked |
|---|---|---|---|
| "The largest ocean on Earth is the" | "Pacific Ocean, covering an area of approximately" | **same** | "which covers an area of" |
| "Water boils at" | "100 degrees Celsius at standard atmospheric pressure" | same | same |

**The repacked path matches; the unpacked one is the outlier.** Whatever the
residual difference in the plain Q4_K path is, repacking is the side that
reproduces the reference implementation — so it is the better default on
correctness grounds *before* the 1.42x is counted. `--no-repack` turns it off.

Three uses break on a repacked tensor and none fail loudly, so all three are
excluded:

- **`get_rows`** — `token_embd` is indexed by token id and repacked rows are
  interleaved. Llama-3.2 ties it to the output projection, so repacking it
  corrupted both at once.
- **`view_2d` by byte offset** — Phi-3's fused `attn_qkv` and `ffn_up` are split
  into q/k/v and gate/up that way. Repacking them made Phi-3 emit
  `[PAD32063]rit[PAD32063]…`.
All five architectures answer correctly, and the 19 container-backed V4-Flash
tests still pass — that path binds through `ResidentSet` rather than
`load_resident`, so it is untouched and could take the same win later.

### The V4-Flash path cannot take that win, and trying found a crash (2026-08-10)

**"The same 1.42x is sitting there for V4-Flash" is false on x86, and the number
is 0 tensors, not 1.42x.** Detail and both engines' output:
`docs/graph/research/v4flash-repacking-2026-08-10.md`.

`ggml_repack_get_optimal_repack_type` branches on the **CPU** as well as the
tensor, and `Q8_0` has no x86 branch at all — its repacked kernels are NEON and
RISC-V only. Every always-read tensor in `V4-Flash-UD-Q4_K_XL` with a repackable
shape is `Q8_0`; the rest are F32 or BF16. Measured: **42 offered, 42 declined,
0 repacked.** The container upcasts exactly the tensors repacking would help.

**llama.cpp is worse off on the same file, not better**: with repacking on (its
default) it does not load at all, because its repack buffer is one range for the
whole model —

```
E alloc_tensor_range: failed to allocate CPU_REPACK buffer of size 147169738752
E llama_model_load: error loading model: unable to allocate CPU_REPACK buffer
```

137 GiB. That is why every V4-Flash figure here passes `--no-repack`, a quirk
that had been recorded without its cause. Bigtea repacks per tensor, so the same
container loads, reports `0 repacked`, and runs. **No tok/s is won by this** —
it is a difference in kind, and `--no-repack` gets llama.cpp running too.

**A crash was already shipping.** ggml's repack `init_tensor` sets
`tensor->extra` to `nullptr` when there is no kernel and returns
`GGML_STATUS_SUCCESS`; `set_tensor` then dereferences it. No assert, no error
code — `STATUS_ACCESS_VIOLATION` and the process is gone. `is_repackable`
accepts `Q8_0` and `Q2_K`, so **any `*.Q8_0.gguf` would have killed `bigtea-run`
on x86 before printing a token.** None of the Q4_K_M containers here hold a
`Q8_0` 2-D weight, which is the only reason it had never been seen. `repack` now
reads what ggml actually decided instead of trusting the shape check.

The machinery was kept and is verified: `RepackedDense` rearranges once at load
(V4-Flash rebuilds its `WeightSet` per block, so rearranging in the bind loop
would redo the whole set 43 times per token), hands the bytes over out of the
resident set rather than duplicating them, and re-attaches per block. Checked
numerically on x86 with `Q4_K` against ggml's own ordinary kernel, bound into two
contexts from one rearrangement. An ARM build gets the win for free.

**FIXED the same day.** The cause was one branch condition: `forward_cached`
already had a working KV cache but was only reached `if config.is_moe()`, so
dense models fell through to a stateless path that rebuilt the whole sequence
per token. Routing them through it needed two guards the streaming path lacked
— QK norm (Qwen3-only) and the RoPE type (NORM for llama, NeoX for qwen).

| generation, 128 tokens | before | after | llama.cpp | verdict |
|---|---:|---:|---:|---|
| **Qwen3-4B** | 0.67 tok/s | **4.27** | 5.90 | 8.8x behind → **1.38x** |
| **Llama-3.2-1B** | — | **10.12** | 12.91 | **1.28x behind** |

Cached and uncached produce **byte-identical** text on Qwen3-4B;
`BIGTEA_UNCACHED=1` keeps the old path reachable so that stays checkable.

Two bugs found while measuring, both fixed:

- **A 651-token prompt aborted the process.** The dense arena was a hardcoded
  2 GiB, and `ggml` answers exhaustion with `GGML_ASSERT`, not an error. The
  arena is now computed from the shape — and the term that was missing is that
  it is **per layer**: one graph spans all 36 blocks in one context and `ggml`
  frees nothing inside a context. `bigtea-run` now refuses a prompt that will
  not fit, naming the arena needed and the longest prompt that would work.
- **The output projection ran on every position.** `build_graph` projected the
  whole sequence through the 151936-wide output matrix and used one row — 253
  GFLOP wasted on a 651-token prompt. Now only the final position is projected.

## A8: unverified architectures are now refused, not answered wrongly

Downloaded Gemma-2-2b and Phi-3-mini to verify A4/A5 rather than guess. They
failed in the two opposite ways, and only one of them is safe:

| model | outcome |
|---|---|
| **Phi-3-mini** | fails cleanly — `container has no tensor "blk.0.attn_q.weight"` (fused QKV) |
| **Gemma-2-2b** | **loads, runs, and answers "The capital of France is" with `himſelf`** |

Gemma-2 needs post-norms after attention and the FFN, logit soft-capping,
attention soft-capping, embedding scaling by `sqrt(n_embd)` and sliding-window
attention on alternate layers. **None of those announce themselves as a missing
tensor**, so the generic dense path ran it and produced confident nonsense.

That is the failure mode this project is most expensive at, and it is the one
thing a runner whose pitch is *"it tells you the truth about your machine"*
cannot do. So `VERIFIED_ARCHITECTURES` is now a list of what has actually been
run and read — `deepseek4, llama, qwen3, qwen3moe` — and anything else is
**refused with the reason**. `bigtea-run --force` runs it anyway; **the server
does not offer that escape hatch at all**, because an API client has no way to
see that an answer is unsound.

**Phi-3 is now supported and verified** (same day): it fuses *both* Q/K/V into
one `attn_qkv` and the FFN gate/up into one `ffn_up`, and both split into views
along whole quantisation blocks, so the fix is free at runtime. It answers "The
capital of France is" with "Paris." and "2 + 2 =" with "4", matching llama.cpp's
own output on the same container. `VERIFIED_ARCHITECTURES` is now
**deepseek4, llama, phi3, qwen3, qwen3moe**.

A silent bug found alongside it: **the RoPE frequency base defaulted to 1e6**,
which was Qwen3's *declared* value generalised into a fallback. Phi-3 declares
none, so it was being rotated at 100x the right frequency. llama.cpp's default
is 10000 and that is now ours. Qwen3 (1e6) and Llama-3.2 (5e5) declare theirs,
so nothing regressed — checked on all four.

**Gemma-2 is now supported too.** It needed four things, none of which announce
themselves: post-norms after attention *and* the FFN, attention-logit
soft-capping at 50 (which has to go **into** the fused kernel — those logits do
not exist outside it), final-logit soft-capping at 30, and embedding scaling by
`sqrt(n_embd)`. Output now matches llama.cpp exactly, markdown and all:

```
llama.cpp   The capital of France is **Paris**. 🇫
Bigtea      The capital of France is **Paris**.
```

**Its 4096-token sliding window is not implemented, so anything past 4096 is
refused** — below the window every layer is effectively full attention, so short
sequences are exactly right and long ones would silently let the local layers
see too far. That is a limit of this implementation, not of the architecture,
and it says so.

`VERIFIED_ARCHITECTURES` is now **deepseek4, gemma2, llama, phi3, qwen3,
qwen3moe** — six families, from two at the start of the day.

## V4-Flash is re-verified after today's changes

Today touched code V4-Flash shares with the dense path — `flash_attn_ext` gained
a `logit_softcap` argument, `threads()` stopped defaulting to a hardcoded 12,
and the RoPE frequency default changed. **All 19 container-backed V4-Flash tests
pass**, including the ones comparing element sums against llama.cpp captures:

```
cargo test --release --test deepseek4_forward -- --ignored
test result: ok. 19 passed; 0 failed  (272s)
```

**And they can now actually be run.** They aborted the whole test binary when
run in parallel: 19 tests each allocating GB-sized arenas exhausted memory, and
`ggml` answers that with `GGML_ASSERT(ctx->mem_buffer != NULL)`, which kills the
process. It surfaced as `error: test failed ... process didn't exit
successfully` rather than as a failing test, and every result after the abort
was lost — so in practice they had stopped being run. They now share a `heavy()`
lock, and the plain command above works without `--test-threads=1`.

## Generation: q, k and v now share one graph — 1.30x

`compute()` re-evaluates the **whole ancestor graph** of its output. The Q/K/V
phase called it three times, once per tensor, so the normalisation they share
ran three times and it paid three graph builds and three threadpool cycles per
layer per token. At one token those fixed costs dominate: the matmuls are
matrix-*vector* products and tiny.

The comment above the code already said *"one compute materialises all three;
they share a graph"*. The code did not.

`Context::compute_many` expands one graph with several roots. Measured on
Qwen3-4B, 96 tokens:

| Qwen3-4B, 96 tokens | before | after |
|---|---:|---:|
| generation | 3.94 tok/s | **5.13** |
| Q/K/V phase | 8.3 s | **5.3 s** |

**1.30x**, and output is unchanged on all five architectures.

**The deficit that follows from it is 1.23x, not 1.15x**, and the difference is
a lesson rather than a rounding error. 5.13 was measured at 96 tokens against a
llama.cpp run that happened to report 5.90; re-measured at the *same* 128 tokens
`llama-bench` uses, with 3 repetitions, llama.cpp is **5.28 ± 0.33** and Bigtea
is **4.3**. Generation slows as context grows, so a shorter run flatters us —
and a single un-repeated reference run has a ±0.33 spread that is a third of the
gap being claimed. Both sides now get matched length and repetitions.

Llama-3.2-1B, same treatment: Bigtea **13.5**, llama.cpp **16.21 ± 0.29** —
**1.20x behind**. An earlier single llama.cpp run read 12.91, which would have
made this a *win*. It is not one.

This is the third time this exact fact has cost time — it is already in
`CLAUDE.md` as *"24 calls per block became 6 — 1.9x"*. Worth grepping for
`compute(` in any hot loop before assuming the arithmetic is the cost.

## `-t` was never plumbed, and the default was the worst setting (2026-08-10)

Full write-up, every command line both sides:
`docs/graph/research/threads-were-never-plumbed-2026-08-10.md`.

`-t N` set `BIGTEA_THREADS` and **only `deepseek4_forward.rs` read it.** Every
other architecture computed its own count from `available_parallelism()`. What
exposed it: `-t 1` and `-t 20` produced *bit-identical* phase timings. An
earlier sweep reading 4.07/4.00/4.31/4.67 tok/s had been recorded as "threads
are not the lever" — it was six measurements of one configuration.

**A sweep whose knob is disconnected is indistinguishable from a flat response.**
Confirm the knob moves something before concluding it moves nothing.

Once connected, generation and prefill turned out to want opposite counts, so
there are now two — `-t` and llama.cpp's `-tb` / `--threads-batch` — chosen by
the token count of the step, not the call site:

| threads | Qwen3-4B gen | Llama-3.2-1B gen | Qwen3-4B prefill |
|---:|---:|---:|---:|
| 2 | **7.64** | **21.95** | — |
| 4 | 7.51 | 21.45 | 47.4 |
| 8 | 6.24 | 16.78 | 70.9 |
| 20 (the old default) | 4.49 | 12.22 | **81.5** |

Generation streams every weight once per token and saturates DRAM long before it
runs out of cores; prefill multiplies a whole block and scales with cores.
llama.cpp shows the same curve on this machine, so it is the hardware, not us.

**A calibration that failed and was deleted**: a 150 ms DRAM-saturation
microbenchmark at load chose 6, 8, 12, 12, 4, 6 on six consecutive runs while
the optimum was 2-4, and its spread (5.51-8.20) was worse than the bad default
it replaced. A pure read has no per-node barrier; a ggml graph does. *A proxy
that must be corrected until it agrees with the objective is the objective,
measured badly.* What shipped instead tunes on **real generated tokens** and
stops after ~4 of them.

Interleaved A/B, same session, `-n 64`, 3 reps:

| | tuned (new default) | `-t 20` (old default) | |
|---|---:|---:|---|
| Qwen3-4B | **8.01** | 4.83 | **1.66x** |
| Llama-3.2-1B | **20.05** | 11.89 | **1.69x** |

### Against llama.cpp — both cells, neither quotable alone

| generation | Bigtea | llama.cpp | verdict |
|---|---:|---:|---|
| Qwen3-4B, **both at default** | **8.01** | 6.52 ± 0.33 (t=10) | **1.23x ahead** |
| Llama-3.2-1B, **both at default** | 20.05 | 20.91 ± 0.65 (t=10) | 1.04x — parity |
| Qwen3-4B, **both hand-tuned** | 7.64 (t=2) | 9.16 ± 0.43 (t=4) | 1.20x behind |
| Llama-3.2-1B, **both hand-tuned** | 21.95 (t=2) | 27.85 ± 1.98 (t=4) | 1.27x behind |

Out of the box we lead on Qwen3-4B because we measure the machine and llama.cpp
uses a fixed default. **Given equal care on both sides llama.cpp is still
faster.** The hand-tuned deficit (1.20x) matches what was recorded before any of
this work (1.23x), which is what says the ratio is real rather than an artefact
of where on the curve each engine was sitting.

Output is byte-identical at 2 and 20 threads on all five verified dense
architectures. 235 tests pass.

### The MoE path wanted ONE thread, and nobody had checked

The tuner picked 1 thread for Qwen3-30B-A3B. That looked like a bug — its signal
is disk-dominated on a streaming model — so the tuner now subtracts read time and
measures only what the knob affects. It still picked 1, three runs in a row, and
a direct sweep says it is right:

| threads | Qwen3-30B gen | expert compute |
|---:|---:|---:|
| **1** | **2.88 tok/s** | 2.2 s |
| 4 | 2.23 | 2.9 s |
| 8 | 1.80 | 3.6 s |
| 20 — *the old default* | 1.21 | 5.2 s |

**2.4x, and expert compute more than doubles as threads are added.** Each expert
matmul at one token is a 768x2048 matrix-vector; a layer's graph holds 24 of
them, and splitting each across 20 threads leaves ~38 rows per thread per
barrier. The threads cost more than the work.

llama.cpp peaks at **4 threads** on the same model where we peak at 1, which
says its expert path parallelises and ours does not. **That is the lead for the
remaining 1.60x**, now scoped with its arithmetic in
`docs/graph/backlog/batch-the-expert-matmuls.md`: the expert path runs at
**3.7 GB/s** where the dense FFN runs at ~13, so the headroom is per-node
overhead (1,152 tensor binds and ~2,300 graph nodes per token), not bandwidth.
**Built, measured, reverted — it does not pay on the streaming path.** The
batched `mul_mat_id` form is genuinely faster (expert compute 7.0 s → 4.2 s over
24 tokens, output byte-identical), but the selected experts arrive as unrelated
`Arc<[u8]>` and making them contiguous costs ~1.02 GB of copying per token —
about what the kernel saves. Generation went **1.34 → 1.27 tok/s**.

`bigtea-kernelbench`'s 11.17 GiB/s for the batched form is real and was
misleading: **it binds the model's already-stacked expert tensor zero-copy.** A
kernel benchmark measures the kernel, not the data movement needed to feed it.

**The version that would pay is a different ticket**: bind the whole stacked
expert tensor with the real ids and copy nothing — which needs the experts
*resident*. Qwen3-30B-A3B is 17.28 GiB and fits on a 32 GB machine, so a
residency-dependent expert path is worth having, and it belongs with the
tok/s-versus-RAM frontier work. Full numbers:
`docs/graph/backlog/batch-the-expert-matmuls.md`.

`llama-bench -m Qwen3-30B-A3B-Q4_K_M.gguf -n 32 -p 0 -r 2 -t 1,4,10`:
1.95 ± 0.64 / **4.21 ± 0.28** / 3.64 ± 0.22.

### V4-Flash has the same curve and still has its old default — 1.28x unclaimed

`deepseek4_forward.rs` reads `BIGTEA_THREADS` directly and does not go through
the tuner, so the flagship model still defaults to every core:

| threads | 1 | 2 | **4** | 8 | 20 *(its default)* |
|---|---:|---:|---:|---:|---:|
| V4-Flash generation | 0.331 | 0.378 | **0.380** | 0.346 | 0.296 |

**Done once r9 was merged in.** `deepseek4_forward.rs` now splits the count the
same way the dense path does, and the split had to be measured in *both*
directions because a blanket cap was tried first and would have traded one
regression for the other:

| V4-Flash, back to back | 4 threads | all cores |
|---|---:|---:|
| generation | **0.196** | 0.177 |
| prefill, 180 tokens | 2.24 | **2.89** |

**Prefill loses 1.29x at four threads; generation loses 1.11x at twenty.** So
`threads()` reads the batch size set by `forward`, the single funnel both
`prefill` and `step` pass through.

**This retires a note that was in `CLAUDE.md`** — "4/12/20 threads all cost the
same on a V4-Flash prefill". True at 5 tokens, where the pass is almost entirely
disk; false at 180.

**V4-Flash absolute numbers drift hard with page-cache state.** The same
`-t 4` vs `-t 20` comparison read 0.380/0.296 earlier in the day and 0.196/0.177
after a dozen heavy runs. Only compare within one session.

One trap on the way: the first version of the split called `std::env::var`
inside `threads()`, which is called at every `ctx.compute` — thousands of times
per token. Locking the environment and allocating a `String` that often cost
more than the split saved, taking generation to 0.267, *below* the 0.296 it was
meant to fix. Both counts are resolved once now.

## Gemma-2 sliding-window attention (2026-08-10) — the 4096 refusal is gone

Detail and command lines: `docs/graph/research/gemma2-sliding-window-2026-08-10.md`.

Gemma-2 alternates a sliding-window layer with a full-attention one. Neither the
window nor a way to live without it existed, so anything past 4096 tokens was
refused. Now the even layers get a second mask with the old keys closed off.

Verified three ways, because two of them prove nothing alone:

1. **Below the window** output is unchanged (`**Paris**.`) — a regression check.
2. **Above the window** (5201 tokens, greedy, `-no-cnv` on both sides) Bigtea and
   llama.cpp produce the same continuation.
3. **The layer parity is load-bearing** — flipping it to odd-slide changes the
   output on the same prompt. Without this, check 2 is also consistent with the
   window never being applied, because a repetitive prompt continues itself.

`-no-cnv` matters: without it `llama-completion` applies Gemma's chat template
and answers as an assistant, and the two engines are not doing the same work.

### Three arenas were short; reading ggml's error correctly found the one that mattered

**`available` in `not enough space in the context's memory pool` is the pool's
total size, not the remainder.** Reading it as the remainder points at whichever
arena was nearly full instead of the one that was too small, and cost two wrong
fixes. `56,624,208 ≈ 3 × 18,874,368` identified it exactly: `post_norm` budgeted
one `n_embd × n_new` tensor and allocated three. Gemma-only, which is why nothing
else ever hit it. The dense-FFN and attention arenas were under-counted too and
are fixed here; they would have aborted at a larger block.

**`arena_for` doubles its total, and that doubling is what hides an undercount
until the block grows enough to eat it.**

### Prefill: not a win, and it nearly got quoted as one

| Gemma-2-2b prefill, 5200 tokens | best of each | verdict |
|---|---:|---|
| llama.cpp | **127.35** (t=20) | — |
| Bigtea | 114.99 (t=4) | **1.11x behind** |

At `-t 4` on both sides it reads 114.99 against 76.76 — 1.50x ahead — because
prefill wants every core and llama.cpp was being handicapped. Run the opposing
command at the setting its own author would choose.

## Quality is measured now — perplexity, and it agrees with llama.cpp (2026-08-10)

Every correctness check in this project had been *"does it say Paris"*, which
catches a broken forward pass and nothing subtler. `bigtea-run --ppl-chunk N`
reports perplexity with llama.cpp's exact windowing:

| perplexity, 128-token chunks | Bigtea | llama.cpp | difference |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct Q4_K_M | **29.0909** | 29.2456 ± 6.49 | **0.53%** |
| Qwen3-4B Q4_K_M | **33.6434** | 34.0293 ± 9.64 | **1.13%** |

Two architectures, two tokenizer families. It exercises the tokenizer, RoPE, the
causal mask, the KV cache, fused attention, repacking and the output projection
against an independent implementation, on a number that would move if any were
wrong. **Both sit inside llama.cpp's own error bar — this is agreement, not a
claim to be more accurate.**

**The windowing is the measurement**, and both details were wrong first time:
including one 98-token remainder alongside three full chunks took the answer
from 29.25 to **33.65**, and scoring from position 1 instead of the second half
gave **1.9232**, which looks spectacular and means nothing. Match the chunk size
and the corpus or you are comparing windowings.
`docs/graph/research/perplexity-2026-08-10.md`.

## CLI parity with llama.cpp (2026-08-11) — 21 flags to 106, counted properly

Full table and every refusal with its reason:
`docs/graph/backlog/llamacpp-flag-audit.md`.

llama.cpp has **182** long flags, counted from `llama-completion --help`. The
parity doc had said "~100", which was a guess. Bigtea now accepts **106**.

| bucket | | state |
|---|---:|---|
| samplers | 22 | **21 done** — only `--backend-sampling` (a GPU concept) left |
| interaction | 22 | **done**, including a REPL and `--interactive-first` |
| logging | 13 | **11 done**; status moved to **stderr** |
| RoPE / YaRN | 15 | **9 done**, 6 refused |
| KV type + prompt cache | 7 | **done** |
| runtime / memory | 31 | I/O mode, `--override-kv`, `--mlock`; **most refused with reasons** |
| GPU | 15 | **10 done** — `--device`/`--main-gpu`, `--list-devices`, `-ngl`/`--gpu-layers`/`--n-gpu-layers`, `-ot`/`--override-tensor`, `--op-offload`; 5 refused, and `--split-mode`/`--tensor-split` need a second usable device this machine does not have |
| grammar / JSON schema | 4 | the r10 worktree session owns this |

**Nothing is accepted that does nothing.** ~20 flags are refused outright with a
written reason — `--keep` (no context shift), `--numa`, `--parallel`,
`--cpu-mask`, `--defrag-thold`, `--swa-full`, `--jinja`, and the GPU set. That
standard exists because `-t` was accepted, echoed and ignored for weeks.

### What the flag work found, which is the point of doing it by hand

Six flags were **accepted and silently did nothing** before being fixed:

- `-t` reached one architecture of six. `-t 1` and `-t 20` gave *bit-identical*
  phase timings. Connecting it was **1.66x**, and led to the MoE expert path
  wanting **one** thread (**2.46x** on Qwen3-30B) and V4-Flash wanting four.
- `--logit-bias` and `--ignore-eos` were skipped by the greedy short-circuit at
  temperature 0, which is the default.
- `--mirostat 2` produced **byte-identical output to greedy** — twice, through
  two different early returns.
- `--chat-template` landed on the deepseek4 path only, so it did nothing on
  every model anyone would test it with.

Each was invisible to a test that checks the process exits zero. They were found
by running the flag and reading the *output* — or the token ids, when the header
would have lied.

### Two numbers of my own that were wrong

- **The flag count** was measured from the help text for eight commits, which
  lists each flag under one spelling. 81 was an undercount of 25. *Measure the
  thing, not a description of the thing.*
- **Batching the expert matmuls** was scoped at ~1.45x from a kernel benchmark,
  built, and reverted: making the streamed experts contiguous costs what the
  batched kernel saves. A kernel benchmark measures the kernel, not the data
  movement needed to feed it.

### Quality is measured now

`--ppl-chunk N` reports perplexity with llama.cpp's windowing. Llama-3.2-1B
**29.0909 vs 29.2456**; Qwen3-4B **33.6434 vs 34.0293** — 0.53% and 1.13% on two
architectures and two tokenizer families. That same tool then measured the
quantised KV cache: **q8_0 costs 0.64% of perplexity for roughly half the
memory**.

## Gemma was running the wrong activation (2026-08-11) — and `VERIFIED` was wrong

**`gemma2` was in `VERIFIED_ARCHITECTURES` and had never been diffed against
llama.cpp.** Its output is now identical; it was not before.

```
bigtea (before)  **Paris**.
llama.cpp        :  a) Paris  b) Lyon  c) Marseille  d)
```

Two bugs, both silent by construction:

1. **SiLU where the whole Gemma family uses GELU.** `grep -rn "gelu" crates/`
   returned nothing — every gated FFN in the crate was SwiGLU. Nothing in a
   container records the activation: a GELU model and a SiLU model hold
   **byte-identical tensor sets**, so this is not a missing tensor, not a shape
   error and not a crash. It is a model that keeps answering in English and
   disagrees with the reference from the first token. Now `FfnAct`, chosen by
   architecture, applied in one place. **This alone fixed Gemma-3.**
2. **The scale went to the kernel instead of into Q.** llama.cpp pre-scales Q
   by `1/sqrt(head_dim)` and passes `scale = 1.0`; ggml folds the soft cap into
   the scale (`scale /= cap`), so the two are the same algebra and
   `0.0625f/50f` vs `0.0625f*(1f/50f)` differ by **one ULP**. Through the cap's
   `tanh` that flipped Gemma-2's first token between `:` and ` Paris`, and with
   it the whole completion. **A soft cap turns a scale into a non-linearity's
   argument** — match the reference's order, not its algebra.

Also fixed: the Gemma **27B-only** attention scale (`n_embd/n_head`, not
`head_dim`), which coincides at every other size — a check that passed here
would still have been wrong at 27B.

**Verified**: 3 prompts x 32 tokens x both engines, `--temp 0`, back to back.
`gemma-2-2b-it` and `gemma-3-1b-it` identical token for token; llama, qwen2 and
qwen3-4b re-checked and unchanged. Architectures **7 -> 8**, tests **409 ->
411**, clippy 0, fmt clean.

New: **`print_hparams` at `-v`** — llama.cpp has printed its hyper-parameters at
load since the beginning, and the hours spent guessing which scale Gemma-2 used
were hours nobody with that output would have spent. It prints *derived* values
(`attn_scale`, per-layer RoPE bases, the windowed-layer list), because a key
read under the wrong name looks exactly like a key that was absent.

Full account: `docs/graph/research/gemma-was-running-silu-2026-08-11.md`.

### Every architecture re-checked, and greedy decoding is not always reproducible

`scripts/parity-check.sh` diffs both engines on three prompts at `--temp 0`.
Seven containers, six architectures: **19 of 21 exact, 0 failures.**

The two exceptions are the finding. **llama.cpp disagrees with itself** on
them — `def fibonacci(n):` on Llama-3.2-1B answers "up to the nth term" with
`-fa on` and "the first n Fibonacci numbers" with `-fa off`; `The capital of
France is` on Phi-3 changes under `--no-repack`. Both flags only reorder a sum.
Those prompts sit on a near-tie, and any engine that accumulates differently
lands on the other side and writes a different paragraph.

So token-for-token identity is not always an achievable target. The script
re-runs the reference under a second configuration before calling anything a
failure and reports `unstable` instead — **a test whose expected value is not
reproducible in the reference must say so rather than fail.** Gemma was not
this: its reference was stable and we were wrong.


## Six more flags, and a list that could not be trusted (2026-08-11)

`--binary-file`, `--chat-template-file`, `--log-colors`/`--no-log-colors`,
`--prio`/`--prio-batch`, `--warmup`/`--no-warmup`, `--completion-bash`. Each
was checked to change something observable before being accepted, which is the
standard `-t` failed for weeks. Two came off the **refused** list:

- **`--prio` was refused for "no thread-affinity or scheduler layer".** Wrong
  premise — process priority needs one syscall, not an affinity layer. It is
  real now, applied before the model opens so the load benefits. **`--prio 3`
  maps to HIGH, not REALTIME, and says so**: realtime outranks the kernel's
  own input and disk threads and can leave a desktop with no way to click
  anything.
- **`--warmup` was refused for "nothing is warmed".** Also wrong: the page
  cache, the repacked tensors, the arenas and the thread ladder all are. It
  runs one throwaway pass on a discarded cache. **Off by default, unlike
  llama.cpp** — warming a disk-streaming runner reads gigabytes, and the cold
  cost is the number this project exists to report honestly.

### The completion list drifted in both directions inside an hour

Hand-written from the help text, it claimed **four flags that do not exist**
and was **missing 23 that do**. A phantom flag is worse than a missing one:
the shell suggests it and the binary rejects it.

Same failure as the flag count this project carried for eight commits.
**Anything that enumerates the flags is a second copy of the parser and will
drift**, so `build.rs` now scans `bigtea-run.rs` for the string literals its
`match` arms are made of and generates the list: **119 long flags**, 0 phantom,
0 missing.

## Chat templates 25 -> 54, and 11 of the old ones were wrong (2026-08-11)

llama.cpp knows 54 template names. Bigtea knew 25 — **and eleven of those
rendered differently from the reference**, which nothing had ever checked.

The oracle is `scripts/capture-chat-templates.py`: it runs llama.cpp with
`--verbose-prompt` and reconstructs, token by token, the exact prompt it builds
for every template it knows. That capture is a fixture in the repo and a test
replays all of it. "Bigtea supports `gpt-oss`" now means **byte-identical to
llama.cpp on a recorded command line**, not "it looked right".

**52 of 54 match exactly.** The two skipped are Hunyuan variants whose bytes the
capture model's tokenizer cannot round-trip; baking a corrupted expectation in
would be worse than not comparing.

### The eleven that were already wrong

| family | what it did | what llama.cpp does |
|---|---|---|
| `llama2` | emitted the `<<SYS>>` block | plain — that block is `llama2-sys` |
| `llama2-sys` | `<<SYS>>` *before* `[INST]` | `[INST] ` first, `<<SYS>>` inside it |
| `falcon3` | shared RWKV-World's `System:` framing | `<\|system\|>`-shaped, nothing alike |
| `zephyr` | the container's EOS | hardcodes `<\|endoftext\|>` |
| `granite` (x3) | a newline after `<\|end_of_role\|>` | no newline |
| `chatglm3` | no preamble, no space | `[gMASK]sop` and a space after the role |
| `chatglm4` | no trailing newline | `<\|assistant\|>
` |
| `deepseek` | blank lines between turns | single newlines |
| `minicpm` | labelled the system turn `<AI>` | emits it raw |
| `monarch` | Bailing's `<role>HUMAN</role>` | `<s>role
content</s>` — a different family |
| `orion` | dropped the system turn's `Human: ` | opens `Human: ` on the system turn |

`glmedge` was aliased to `chatglm4` and `bailing` to `monarch`; both are
separate families, so those containers were fed two tokens at position 0 they
were never trained to see.

**A wrong template does not fail.** The model answers, fluently, having been
handed a framing it has never seen — it comments on the question instead of
answering it, or answers the system prompt. No test that checks "did it produce
a string" can see that, which is why the expectation had to come from llama.cpp
rather than from me.

One place we deliberately differ, and it is recorded in the code: llama.cpp's
Zephyr renderer hardcodes `<|endoftext|>` because its renderers have no
vocabulary to read. **TinyLlama uses the Zephyr framing with `</s>`**, and its
own Jinja template says `eos_token`, so the reference frames it with a token it
has never seen. `eos_or` prefers the container's EOS when there is one and
falls back to llama.cpp's literal when there is not — the fixture test passes
`""` and so reproduces llama.cpp exactly.

## Samplers 16 -> 20: parity (2026-08-11)

| sampler | what it is |
|---|---|
| `--adaptive-target` / `--adaptive-decay` | aim for a token of a given *probability*, with the target moving as it observes what it actually picked — a feedback controller like mirostat, not a filter |
| `--infill` | suppress fill-in-the-middle control tokens |
| `--grammar-lazy` | hold a grammar back until the model writes a trigger, then constrain everything after it |

Bigtea now implements **20 of llama.cpp's 20** sampler entry points.

**Adaptive-p was written in the wrong slot first**, and the mistake is worth
recording because it looked plausible: it went next to mirostat, *before* the
truncations, since both replace the temperature tail. The transform hands every
token whose probability is near the target the same peak logit — so on an
untruncated 150k vocabulary it spread the mass across the whole dictionary and
produced `LOGGER冲突ユー ihm definit🏤谋划`. It is llama.cpp's **terminal**
sampler, in `dist`'s slot, and needs a candidate set top-k and top-p have
already cut down. Moved, it produces `in a magical world called Aylum, a
mysterious dragon slayer`.

`is_greedy()` gained both new knobs in the same commit. **That method has now
been the bug twice** — a knob that changes the output but is not listed there is
accepted, echoed in the header, and silently ignored at temperature 0.
`--mirostat 2` produced byte-identical output to greedy for a whole release
that way.

`--grammar-lazy` takes **substrings, not regexes**, and the help says so.
llama.cpp's `--grammar-lazy-patterns` takes regexes; a half-implemented regex
engine that silently mismatches would arm the grammar at the wrong moment,
which is worse than not having the flag. Verified three ways: a trigger that
fires (`grammar armed after 1 tokens`, then JSON), one that never appears
(prose throughout), and no trigger at all (armed from token 1).

`--infill` resolves the FIM tokens **from the vocabulary's own text** rather
than from metadata keys, because containers disagree about which keys they set
while the token text is stable. Qwen3-4B: 4 tokens found. Qwen2-0.5B: 0, and it
says `0` rather than pretending.

## Every llama.cpp flag is now recognised — 158 implemented, 24 declined

**Updated 2026-08-14, and the previous headline was false.** It read "every
llama.cpp flag is now recognised" while `--flash-attn`/`-fa` was in neither the
implemented set nor the declined one — and an unrecognised flag was not an
error, it became the *prompt*. `bigtea-run -m m.gguf -fa off "hello"` ran with
`prompt = "-fa"`, discarded `"hello"`, and exited **0**. The claim was checked by
reading a table; the gap was in the code the table does not describe.

The counts are now **computed from both sources** rather than tallied:

```
llama-completion --help | grep -oE '\-\-[a-zA-Z0-9][a-zA-Z0-9-]*' | sort -u   # 182
```

intersected with `bigtea-run`'s match arms and with its `REFUSED` table:

| | count |
|---|---:|
| implemented — the flag changes something observable | **158** |
| declined with a reason — recognised, exits 2, names what is missing | **24** |
| in neither — silently swallowed | **0** |

**That is still not flag parity and must not be quoted as one.** 24 flags do
nothing here, and 15 of them are GPU.

An unknown `-` token is now an error, with `--` as the escape hatch for a prompt
that genuinely starts with a dash. `declined_flags_actually_decline` extracts the
`REFUSED` table from source at test time and runs the binary once per row, so the
table cannot drift from the binary again — it had, silently: `--jinja` sat in the
table claiming "no Jinja engine" while `bigtea-jinja` evaluated templates one
match arm above it, and because `REFUSED` is consulted from the *fallback* arm,
the explicit arm shadowed the row. Dead code that lies.

A command line copied from llama.cpp now runs or explains itself, instead of
dying on an unknown flag. What it never does is quietly do less than it says:

```
$ bigtea-run -m m.gguf --n-gpu-layers 32
bigtea-run: --n-gpu-layers is not supported: no GPU backend exists
  Declined rather than ignored: a run never quietly does less
  than its command line says. Drop the flag to continue.
$ echo $?
2
```

**`-t` was accepted and ignored here for weeks**, and a disconnected knob is
indistinguishable from a flat response — the sweep that "proved threads are not
the lever" was measuring a flag that reached nothing. Refusing out loud is the
cheap defence against repeating that.

What is declined, and the honest reason:

All 24, by group — the counts add up to 24 because they are the table's rows,
not a summary of it:

| n | flags | why |
|---:|---|---|
| 10 | `--device`, `--list-devices`, `--gpu-layers`, `--n-gpu-layers`, `--main-gpu`, `--split-mode`, `--tensor-split`, `--kv-offload`, `--op-offload`, `--override-tensor` | **no GPU backend exists.** `bigtea-probe` detects the card and nothing uses it; a VRAM tier needs a CUDA-enabled ggml *and* a non-zero-copy binding path, since weights are bound by handing ggml a host pointer (`weights.rs:286`). Scoped 2026-08-11 in `research/gpu-tier-smallest-honest-slice-2026-08-11.md`: this machine has **no CUDA toolkit at all**, and dense-layers-in-VRAM is a 1.10x ceiling on the model where it fits and doesn't fit on the model where it would matter |
| 4 | `--cache-type-{k,v}-draft`, `--spec-draft-type-{k,v}` | speculative decoding measured ~1.4x here, not the literature's 2.2x, and is a net loss below ~0.75 acceptance |
| 2 | `--grp-attn-n`, `--grp-attn-w` | self-extend, which needs a change to `stream.rs` |
| 2 | `--parallel`, `--defrag-thold` | one sequence by design; an append-only KV cache that cannot fragment |
| 2 | `--poll`, `--poll-batch` | spin-vs-yield inside ggml's threadpool, which ggml owns. Affinity, NUMA-isolate and `--prio` all moved *out* of this row and are implemented — they were one syscall each, and "no affinity layer" described the code rather than the difficulty |
| 2 | `--no-host`, `--no-mmproj` | a host buffer type and a multimodal projector, neither of which exists here |
| 1 | `--backend-sampling` | a GPU concept |
| 1 | `--docker-repo` | a different protocol, not a URL. `-hf`, `--hf-repo` and `--model-url` are implemented |

**`-fa off` is refused too but is not in that table**, because it is a refused
*value* of an implemented flag: one attention path exists and it is the flash
one. It is declined by name rather than accepted, since `-fa off` is a control
`parity-check.sh` passes to the *reference* — ignoring it would silently turn a
parity check into a comparison of a run with itself.

Jinja, reasoning-format, the download flags, affinity and the adapters **left
this table**. The 57 → 24 move is mostly those, not a change of standard: the
adapter flags now load and shape-check a LoRA, though nothing applies it yet, and
that gap is stated where the flag is documented rather than by declining it.

Three more implemented in the same batch: `--mmap` (the default, spelled out),
`--ubatch-size` (takes the smaller of it and `-b`, and says which), and
`--swa-full`, which **is already the behaviour** — Bigtea's KV cache is always
full and the window lives in the attention mask, so it reports that rather than
accepting the flag silently.

## `-hf` works: the runner fetches its own models (2026-08-11)

Seven flags moved from **declined** to **implemented**: `-hf`, `--hf-repo`,
`--hf-file`, `--hf-token`, `--model-url`, `--offline`, `--cache-list`. One
command now downloads and runs:

```
$ bigtea-run -hf Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf \
             -p "The capital of France is" -n 8 --temp 0
model      fetched .../bigtea/models/Qwen--Qwen2-0.5B-Instruct-GGUF--qwen2-0_5b-...gguf
 Paris. It is the most populous city
```

Second run reports `model cached`. `--cache-list` shows it; `--offline` runs
from it and refuses to reach the network.

### Two things that are not "shell out to curl"

**Every download is checked for GGUF's magic number, and a file that fails it
is deleted.** A half-succeeded download is the worst outcome available here: a
truncated container parses far enough to report a plausible architecture and
then fails deep in a forward pass, and a gated repo returns an *HTML error
page* which lands under a `.gguf` name. Leaving that on disk means the next run
re-reads it and misdiagnoses a corrupt model. Four bytes settle it.

**A repo without a filename is refused, not guessed.** `-hf owner/name` and
`owner/name:Q4_K_M` both name a repo holding several quants, and resolving
either needs the Hugging Face listing API, which this build does not call. It
says so, and names both ways out:

```
--hf-repo unsloth/gemma-3-1b-it-GGUF names a repo but not a file. Pass
--hf-file <name.gguf>, or use -hf unsloth/gemma-3-1b-it-GGUF/<name.gguf>.
```

Guessing `<name>-Q4_K_M.gguf` is right for some repos and a 404 for others, and
a 404 saved under a `.gguf` name is exactly the failure above. **This project
has already paid for guessing once**: the pre-tokenizer fallback that guessed
`llama-bpe` where llama.cpp defaults to GPT-2, found today, wrong on every
`gpt2` container that omits the key.

The token is read and **never echoed, including on the failure path** — a
failed download is exactly when output gets pasted into an issue.

Flags recognised: **187** — 137 implemented, 50 declined. Tests **413 -> 420**.

## Both branches merged, and one process rule tightened (2026-08-11)

`main` carries the whole day. Two sessions, no collisions, three branches
deleted after `git merge-base --is-ancestor` confirmed containment.

### The rule that changed, and why it should have been obvious

`starcoder2` was added to `VERIFIED_ARCHITECTURES` on a **3/3 parity pass while
running the wrong pre-tokenizer**. It agreed on those three prompts only
because its merge table differed from the model that failed. Three prompts were
enough to certify an architecture and not enough to notice that its *input* was
being split wrongly.

So `parity-check.sh` runs **eight** prompts now, and its header states what a
pass means: **evidence about these prompts, not about the architecture.** The
five added are a numeric run, a list continuation, arithmetic, SQL and formal
register — each stresses a different part of the vocabulary and a different
part of the graph.

It earned itself immediately: Gemma-3 has an eighth-prompt near-tie
(`Q: What is 17 plus 25? A:`) that three prompts never reached, and Phi-3 has
two. All are reported `unstable` — llama.cpp disagrees with itself on them —
rather than passed or failed.

**A single factual prompt is the weakest test available.** "The capital of
France is Paris" survives a surprising amount of wrong arithmetic, because the
answer is overdetermined by the training data. Both bugs found today —
Gemma's activation and the pre-tokenizer — were caught by the *code* prompt.

### What the merge brought in from the other session

LayerNorm bound beside RMSNorm; the full bias set; **partial RoPE**, where
`rope.dimension_count` had been ignored entirely and `head_dim` went in as
`n_rot` unconditionally, over-rotating every container that declares the key;
ungated FFN; and the pre-tokenizer default. Two traps worth carrying forward:

- **A bias not in `required_tensors` is never loaded, and the graph silently
  skips it.** `output_norm.bias` is the worst case — applied once, so it shifts
  every logit equally and the text stays fluent.
- **A missing `ffn_gate` means two different things.** Phi-3 *fuses* it into a
  tensor twice `n_ff` wide; StarCoder2 has none. Testing for the tensor alone
  made Phi-3 ungated and broke a verified architecture. `ne1 == 2*n_ff`
  separates them.

## `--check-tensors` and `--fit` (2026-08-11) — four more off the declined list

### `--check-tensors`, and the two bugs it found in itself first

Container parsing validates **structure**. All of it can be perfect while the
numbers are ruined, and the symptom is not a crash: the first NaN reaching a
softmax makes every probability NaN, `argmax` returns index 0, and the model
emits one token forever. That reads as a broken *model*, so the search starts in
the forward pass instead of in the file.

Verified by corrupting 4 KiB of a known-good container at a known offset:

```
check      blk.12.ffn_up.weight: non-finite block scale at block 72335
bigtea-run: 1 tensor(s) hold non-finite values. This container is damaged
```

The refusal this retracts claimed a values-level scan "would have to dequantise
every tensor". Wrong: the **f16 block scales** are floats, need no
dequantisation, and are exactly where a ruined quantise shows up.

Two bugs, both caught only by running it against a container **known to be
healthy**:

1. **Q4_K and Q5_K carry their scales at the start of the block, not the tail.**
   Packed 4-bit quants at offset 140 read as `inf`, so the validator called a
   healthy Qwen2 container damaged. Worse: **the unit test asserted the tail
   too** — written from the same assumption as the code, so it proved only that
   the two agreed. Both now cite `ggml-common.h`.
2. **The 8 MiB chunk was not a multiple of 144 or 210 bytes**, so every chunk
   after the first began mid-block. It failed at `token_embd.weight` "block
   246754" — exactly where chunk one ended.

An unknown quant type is **counted as uninspectable, never guessed at**: reading
the wrong two bytes as a scale invents failures, and a validator that cries wolf
is worse than none.

### `--fit`, `--fit-target`, `--fit-ctx`

The one flag group where Bigtea should be *ahead* rather than level: llama.cpp
asks "will this fit in device memory" from outside the engine, and owning
residency is this project's whole design.

| | effect, measured |
|---|---|
| default (`--fit on`, target 1024 MiB) | 7.46 GiB expert cache |
| `--fit-target 6144` | **2.44 GiB** — the headroom moved and the cache gave way |
| `--fit off` | fixed 1.00 GiB, machine-independent |
| `--cache 3` + `--fit-target 6144` | **3.00 GiB** — an explicit argument still wins |

`--fit` only ever adjusts arguments the user did **not** set, which is what
makes llama.cpp's default-on safe to match. `--fit off` gives a fixed 1 GiB
rather than everything free, because the point of turning fitting off is
reproducibility and "all of RAM" is the least reproducible number available.

The 2 GiB headroom this file hardcoded is now `--fit-target`, and the header
prints which value it used — **a headroom you cannot see is a headroom you
cannot argue with.**

`--fit-ctx` reports the question this project exists to answer: *given this
machine, how much context is there room for?* Its first version answered "0
tokens" on a machine with 8 GiB free, because it subtracted the expert cache —
which is by construction everything left after headroom. **The cache is elastic
and the KV cache is not**, so the honest answer is what fits once the cache has
shrunk to its floor: 568,519 tokens for Qwen2-0.5B.

Flags: **140 implemented, 47 declined**, of 187 recognised.

## CPU affinity: six more off the declined list, and the mask reaches the metal

`--cpu-mask`, `--cpu-range`, `--cpu-strict` and their three `-batch` variants.
The proof they work is not that they parse:

```
--cpu-mask 0xf      prefill 151 tokens in 1.2s (122.85 tok/s)
--cpu-mask 0xfffff  prefill 151 tokens in 0.5s (303.19 tok/s)
```

**2.5x from the mask alone** — the flag reaches the hardware, which is exactly
what `-t` failed to do for weeks while being accepted and echoed.

I refused these earlier for "no thread-affinity layer". **That premise was
wrong in the same way `--prio`'s and `--warmup`'s were**: process affinity is
one syscall, and every thread ggml spawns inherits it. Bigtea does not need to
own a threadpool to pin one. Three refusals in a row have now turned out to
rest on a wrong premise rather than a real limit — the pattern is refusing on
*architecture* ("we have no X layer") when the flag only needs a *syscall*.

What it genuinely cannot do is a different mask for prefill and generation,
since ggml owns the pool. The `-batch` variants share the mask and the runner
says so, rather than taking a second one and dropping it.

### Two things the tests caught before the hardware did

**`5` means different CPUs to the two flags.** It is CPUs 0 and 2 as a hex
mask and CPU 5 as a range — which is *why* llama.cpp carries two flags. My one
heuristic parser guessed hex and would have pinned `--cpu-range 5` to two cores
instead of one, silently. Split into `parse_cpu_mask` and `parse_cpu_range`.

**`--cpu-strict` capped generation threads and not prefill**, so a 4-CPU mask
still ran 20 prefill threads. Oversubscription is the thing strict mode exists
to prevent, and half-applying it is worse than not offering it — the header
then reads as though it worked. Both counts now follow the mask, and an
explicit `-t`/`-tb` still wins over both.

Flags: **147 implemented, 44 declined**, of 191 recognised. Tests **435**.

## Context shift: generation past the context limit (2026-08-11)

`--context-shift` (default on), `--no-context-shift`, `--keep N`. 40 tokens
generated under a 24-token limit:

```
$ bigtea-run -m m.gguf -p "Once upon a time" -n 40 -c 24 --keep 4
shift      context full: kept 4, dropped 9. ...
generated  40 tokens in 1.6s (25.10 tok/s)
```

**The shift was unreachable when first written.** The `-c` check refused the
run before generation started — the exact case the shift exists to handle — so
the flag fired zero times while being accepted and echoed. That check is now
gated on `--no-context-shift`, and its message names the way forward instead of
just the wall.

### The limitation is stated at runtime, not buried

```
The shifted keys still carry the rotation of their ORIGINAL positions --
llama.cpp re-ropes them and this build does not, so history past the first
shift is approximate. --no-context-shift stops instead.
```

A key is computed with RoPE applied at its absolute position. After the slide it
sits at a lower one, so every shifted key carries a rotation for a position it
no longer occupies. llama.cpp corrects this (`llama_kv_cache_seq_add`); this
does not. The output degrades visibly after a shift, and **saying so once, in
the run itself, is the difference between a documented approximation and a
silent one.** It is still better than refusing to generate, and it is the trade
llama.cpp made before it added re-roping.

`KvCache::shift_out` carries three unit tests, including one that checks a
slid position holds what the *later* position held rather than what used to be
in that slot — the failure mode that would look like plausible text.

Flags: **150 implemented, 44 declined**, of 194 recognised. Tests **438**.

## `unstable` was a verdict; it is a suspicion now (2026-08-11)

The parity harness re-ran the reference under `-fa off` and `--no-repack` and,
when llama.cpp disagreed with itself, called the prompt a near-tie and moved on.
**Nine of eleven `unstable` verdicts in one session turned out to be bugs** —
Llama-3.2 rotating with the wrong RoPE, Falcon3 prefilled a token short.

The flaw is structural, not a threshold: **that re-check compares the reference
to itself, and cannot see that OUR INPUT differed.** When the input differs, a
near-tie is exactly the symptom — the model is answering a slightly different
question and lands on the other side of whatever was close.

Two changes:

1. **On a mismatch, the tokenized prompt is compared.** Different token counts
   mean the two engines are not answering the same question, and it is reported
   as a **FAILURE** rather than a tie. One check catches the whole class: a
   missing BOS, a wrong pre-tokenizer, a byte-fallback that drops characters.
2. **Near-ties are counted, and three is a cluster.** One in eight is ordinary;
   three is a bug nobody has found yet, and the script exits non-zero saying so
   rather than printing eight reassuring lines.

Phi-3's two survive both checks — identical tokenization, below the cluster
threshold — which is the answer the harness should have been giving all along.

## Reasoning blocks: six more off the declined list (2026-08-11)

`--reasoning-format`, `--reasoning`, `--reasoning-budget`,
`--reasoning-budget-message`, `--reasoning-preserve`,
`--no-reasoning-preserve`. On Qwen3-4B, which thinks:

```
default                     <think>Okay, the user is asking...</think> 2 + 2 = 4
--reasoning-format auto     2 + 2 = 4
--reasoning-budget 20       reasoning  budget of 20 tokens reached while
                                       still inside <think>; stopping
```

**Refused earlier as "downstream of Jinja".** That was wrong for the fourth
time in the same shape: the block is delimited by ordinary text in the output,
and finding it needs no template engine at all. The pattern in every one of
these — `--prio`, `--warmup`, the affinity group, and now this — is refusing on
*architecture* ("we have no X layer") when the feature only needs to read what
is already there.

Two decisions worth recording:

**The tags are matched as text, not as token ids.** Qwen3 emits `<`, `think`,
`>` as three tokens, and the tags are ordinary vocabulary in most models.
Matching ids would have worked on one model and failed silently on the next —
which is this project's signature failure.

**Hitting the budget stops rather than forcing `</think>`.** Injecting a close
tag means guessing a token id that differs per vocabulary, and a model still
thinking at its budget has not produced an answer — cutting mid-thought and
continuing would read as one. `--reasoning-budget-message` prints in its place
so the truncation is visible as truncation.

Flags: **156 implemented, 38 declined**, of 194 recognised.

## `--load-mode` and `--numa isolate` (2026-08-11) — the fifth and sixth wrong premise

```
--load-mode dio          model qwen2 (direct (cache bypassed))
--load-mode mmap         model qwen2 (buffered (page cache in use))
--load-mode mmap+mlock   ... mlock 0.34 GiB pinned in physical memory
--numa distribute        refused BY NAME, with what it would need
```

**`--load-mode` was refused for "`--direct-io`/`--no-direct-io` are the two
modes that exist".** llama.cpp now marks `--mlock`, `--mmap` and `--direct-io`
all *deprecated in favour of* `--load-mode`, and every one of its five modes
maps onto a switch this build already had. The modes existed; the spelling did
not. `mmap+mlock` is one mode, not two flags — that is the part a naive alias
would have got wrong.

**`--numa` was refused for "no NUMA-aware allocation to select between".** Half
right, and the half that matters was wrong: `isolate` is a mask and a syscall,
exactly like the affinity group. `distribute` and `numactl` place *individual
threads* on chosen nodes and ggml owns the pool, so those two are refused **by
name** with what they would need, rather than the whole flag being declined.

On a single-node machine `isolate` reports that there is nothing to isolate.
Silently pinning to "the whole machine" would have looked like it worked.

**Six refusals in a row have now turned out to rest on a wrong premise** —
`--prio`, `--warmup`, the affinity group, the reasoning group, `--load-mode`,
`--numa`. The question that produced all six was "do we have a subsystem named
after this?". The right one is "what does this actually require?".

Flags: **158 implemented, 36 declined**, of 194 recognised.

## `rope_freqs.weight` is ignored — every Llama-3.x model is wrong (2026-08-11)

The eight-prompt sweep found it. Llama-3.2-1B:

```
FAIL  SELECT name, COUNT(*) FROM users WHERE
  bigtea   :  age > 18 AND gender = 'male' GROUP BY name;
  llama.cpp:  age > 18 GROUP BY name HAVING COUNT(*) > 1;
```

llama.cpp is **stable** on that prompt across `-fa on`, `-fa off`,
`--no-repack` and `-t 4`, so it is not a near-tie.

Llama-3.x containers ship a `rope_freqs.weight` tensor and llama.cpp passes it
to `ggml_rope_ext` as `freq_factors`. **This build passes `None` at all four
call sites** — and the parameter is already there as an `Option`, so nothing
was missing except the value.

**`llama` has been in `VERIFIED_ARCHITECTURES` since the beginning**, and
TinyLlama passes 8/8 — because TinyLlama is Llama-2 and has no such tensor. One
container in a family exercising a feature and another not is exactly the gap a
three-prompt set leaves. Read `llama` as "verified on Llama-2-shaped
containers" until this lands.

Ticket: `docs/graph/backlog/rope-freqs-ignored.md`. The fix is three lines in
`qwen3.rs`/`stream.rs`, which the other session owns.

### The harness also cried wolf once, and that is worth as much

TinyLlama reported a FAIL on `Q: What is 17 plus 25? A:` where both engines
answered ` 42`. llama.cpp prints `[end of text]` on EOS and Bigtea stops
silently — **the generated tokens were identical.** Stripped now.

A harness that cries wolf is worse than no harness: the first thing anyone does
with a FAIL is go looking in the forward pass. Two FAILs appeared in this sweep
and exactly one was real; without checking both, the real one would have been
dismissed along with the false one.

## The eight-prompt sweep, re-run after the harness fix (2026-08-11)

| container | ok | unstable | FAIL |
|---|---:|---:|---:|
| tinyllama-1.1b-chat | 8 | 0 | 0 |
| Qwen2-0.5B-Instruct | 8 | 0 | 0 |
| gemma-2-2b-it | 8 | 0 | 0 |
| gemma-3-1b-it | 8 | 0 | 0 |
| Qwen3-4B | 8 | 0 | 0 |
| Phi-3-mini-4k-instruct | 6 | 2 | 0 |
| **Llama-3.2-1B-Instruct** | 3 | 4 | **1** |

**Gemma-3's arithmetic prompt and Gemma-2's are no longer unstable.** Both were
the `[end of text]` artefact, not near-ties — the harness had been comparing
llama.cpp's EOS marker against our silence. Five containers are now clean at
eight prompts where three prompts had certified them.

**Phi-3's two unstable prompts survive the harness fix**, which settles what
they are: llama.cpp genuinely disagrees with itself on them under `--no-repack`.
Gemma's did not survive it, so the two cases are different and only one was ever
about the models.

Llama-3.2 is the outlier twice over: the only FAIL (`rope_freqs.weight`,
ticketed) and the only container with four genuine near-ties in eight.

## The Jinja gap, scoped rather than guessed at (2026-08-11)

`--jinja` is the last CLI capability that is not GPU, not a draft model and not
an adapter. It has stayed unbuilt because of the rule in `chat.rs`: **a
half-implemented Jinja silently produces the wrong framing.**

Censusing all 12 `tokenizer.chat_template`s on disk makes the subset bounded:

```
if/endif 123 · set 98 · else 40 · for/endfor 31 · elif 21
loop.index0 20 · loop.last 12 · loop.first 10
namespace() 10 · raise_exception() 6 · strftime_now() 1
filters: tojson 15, trim 6, length 5
operators: in, not, is defined, is string, is not none
```

**No macros, no imports, no inheritance, three filters.** That is a
self-contained crate with no dependencies, the same shape as `bigtea-grammar`
— a weekend, not a quarter.

The acceptance test already exists: `chat-templates.txt` is llama.cpp's own
rendering of all 54 templates, and 52 of the family renderers are verified
against it. A Jinja engine agreeing with them is a **cross-check between two
independent implementations**, not a self-check.

Ticket: `docs/graph/backlog/jinja-chat-templates.md`.

## `-b 1` joins the no-op probe, and why that is a cost as well as a fix

The other session asked for it and the principle holds: batching changes how
many tokens a forward pass covers, which for a correct engine only reorders
sums. llama.cpp disagrees with **itself** under it, verified here on
Qwen3-30B-A3B:

```
default : ...Spain is Madrid. The capital of Germany is Berlin.
-b 1    : ...Spain is Madrid. The capital of Portugal is Lisbon.
```

**The set of no-op configurations tested decides what counts as a bug**, and
that cuts both ways. Every configuration added makes `unstable` easier to reach,
and `unstable` is exactly where a real bug hides — Llama-3.2 reported **four**
unstable prompts for a day and all four turned out to be `rope_freqs.weight`
being ignored. The cluster was the signal, not the noise.

So the harness now **names which configuration moved it**:

```
unstable  Phi-3-mini-4k-instruct  The capital of France is
  the reference disagrees with itself under: -fa-off --no-repack -b-1
```

"`-b 1` only" is a weaker claim than "every no-op moves it", and collapsing the
two into one word is how a cluster stops looking like a cluster. The rule that
three or more unstable in eight exits non-zero is what keeps the addition
honest.

One correction back to that session: their report says `-b 1` reproduces **both**
Phi-3 near-ties byte-identically against Bigtea. Re-run here, only the
arithmetic prompt does; `The capital of France is` gives `Paris. Paris is known
for its rich history` under `-b 1` against Bigtea's `Paris. <|assistant|> That's
correct!`. The classification is unchanged — the reference is unstable there
under all three configurations — but the stated reason was not reproducible.

## `--jinja` is wired, and the fallback is the feature (2026-08-13)

The container's own template is evaluated when asked, and **declines loudly** on
anything the engine does not fully understand:

```
$ bigtea-run -m Qwen2-0.5B --jinja -sys SYS -p HI
chat       template evaluated (--jinja)
prompt     "<|im_start|>system
SYS<|im_end|>
<|im_start|>user
HI<|im_end|>
..."

$ bigtea-run -m Llama-3.2-1B --jinja -sys SYS -p HI
prompt     "<|begin_of_text|><|start_header_id|>system<|end_header_id|>


            Cutting Knowledge Date: December 2023
Today Date: 13 Aug 2026

SYS..."

$ bigtea-run -m Phi-3-mini --jinja -sys SYS -p HI
chat       template has no system branch; merging it into the first user turn
chat       template evaluated (--jinja)

$ bigtea-run -m gemma-2-2b-it --jinja -sys SYS -p HI
chat       --jinja declined: template rejected this conversation: System role not supported
           falling back to the family matcher.
```

**Off by default, unlike llama.cpp.** The family renderers are verified
byte-identical to llama.cpp's for 52 of its 54 names; making evaluation the
default would change the prompt on models that are currently verified. That is
a thing to opt into.

Every decline names the construct. A fallback nobody can see is
indistinguishable from a flag that does nothing — which is the failure `-t`
already cost this project once.

Gemma-2's decline is worth its own note: its template **raises** on a system
turn, and falling back means the family matcher then accepts a conversation the
model's own template forbids. The fallback is still the safe move; the family
matcher's permissiveness is the open question.

Flags: **165 implemented, 30 declined**, of 195 recognised. Tests **481**.

## Jinja: every template on disk renders (2026-08-13)

**15 containers: 6 agree with the family matcher, 8 differ, 1 refuses** — and
the refusal is Gemma-2's template *correctly* raising on a system turn.

Our rendering is **byte-identical to `llama-completion --jinja`** on Llama-3.2,
date included. Two fixes got the last four tokens:

- **`strftime_now`, and treating a built-in as `is defined`.** Llama-3 guards
  with `if strftime_now is defined` and falls back to a hardcoded
  `26 Jul 2024` — so answering `false` put a two-year-stale date in every
  Llama-3 prompt.
- **Jinja strips one trailing newline** (`keep_trailing_newline=False`), which
  Llama-3's template depends on.

The 8 "differ" rows are **not failures**: llama.cpp behaves identically, its
`--no-jinja` matching our family matcher and its `--jinja` matching our engine.
Hardcoded renderers drop content the templates specify — a property of the
approach, not a bug in either engine.

One judgement reversed: `'' + true` was refused on the principle that silent
coercion is how a template prints `None`. llama.cpp evaluates with **minja,
which coerces**, and DeepSeek writes exactly that. The line is now **a defined
scalar coerces, `none` still refuses** — the dangerous case was never `true`,
it was a missing variable becoming the literal text `None`.

## Adapters: loaded and checked, applied nowhere (2026-08-13)

`--lora`, `--lora-scaled`, `--control-vector`, `--control-vector-scaled`,
`--control-vector-layer-range`. `bigtea-model/src/adapter.rs`, 8 unit tests.

**The loader is deliberately separate from the application.** Applying either is
a change to the forward pass; deciding whether an adapter *belongs to this
model* is arithmetic on shapes — and that is where the silent failures are:

- A LoRA whose `lora_a` is stored untransposed **still multiplies**, against the
  wrong axis, and gives a model that answers fluently and is not the fine-tune.
  llama.cpp calls this one out by name and so does the error here.
- **The scale is `alpha / rank`, not `alpha`.** A rank-64 adapter with alpha 16
  scales by 0.25; using alpha alone applies it 4x too strongly — which does not
  error, and produces a model that *is* recognisably the fine-tune and wrong in
  degree. The hardest kind of wrong to notice.
- A control vector for a 32-layer model applied to a 26-layer one shifts the
  wrong residuals. `--control-vector-layer-range` **clears** out-of-range
  layers rather than clamping, because clamping would apply a direction to a
  layer the user excluded.

**The run is refused, not warned.** A run that loaded an adapter and did not
apply it would produce base-model output under a command line asking for a
fine-tune, and nothing downstream could tell:

```
$ bigtea-run -m model.gguf --lora adapter.gguf
bigtea-run: adapters are checked but NOT YET APPLIED -- the forward-pass half is
unimplemented, so this run would give you base-model output. Drop the adapter
flags to continue.
```

Flags: **170 implemented, 25 declined**, of 195 recognised. Tests **492**.

## RWKV: the fifth tokenizer family (2026-08-13)

llama.cpp has five real vocabulary types — SPM, BPE, WPM, UGM, RWKV. This had
four. `crates/bigtea-tokenizer/src/rwkv.rs`, 8 unit tests plus 6 through the
public `from_metadata` path.

It is **greedy longest match over a trie of raw byte strings**: no merge table,
no scores, no pre-tokenizer. Two details are easy to get subtly wrong and
neither raises:

- **The vocabulary is stored escaped.** `\n`, `\t` and `\xNN` appear as literal
  backslash sequences, so a loader that keeps the stored text builds a trie
  keyed on the *text of the escape*. A real newline then never matches, and
  every line break becomes an unknown token. Decoding has the inverse problem —
  emitting the stored text puts a literal backslash-n where the model produced
  a newline. Both directions are tested.
- **Longest match is the last node *with a value*, not the deepest reached.**
  With `ab` and `abcd` present and `abc` absent, the walk descends past the
  answer; taking the deepest node would emit nothing at all.

`\xNN` can denote a byte that is not valid UTF-8 alone, which is why the
unescape works in bytes rather than `char`s. An empty vocabulary entry is
skipped at build time — it matches at every position with length zero, so the
loop would hang on real input rather than merely answer wrongly.

**Implemented is not verified**, and the parity row says so. There is no RWKV
container on this machine, so the family is exercised against a hand-built
vocabulary through the real loading path — not against llama.cpp. This project
has already shipped `gemma2` as "verified" while it ran the wrong activation.
Loading is not evidence, and neither is a test I wrote myself.

Tests **492 → 507**.

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
  silently. **The release workflow is written** (2026-08-09, `release.yml`): it
  builds on a tag for all three platforms, **asserts every binary actually
  starts** — a missing runtime kills the process before `main`, so silence is the
  symptom — reports what each links against, and attaches the archives. Not yet
  fired against a real tag.

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
./target/release/bigtea-probe        # RAM/disk/GPU + what to close
```

Windows needs the **GNU** Rust toolchain and `C:\msys64\mingw64\bin` on PATH —
Git Bash's own `/mingw64` is not MSYS2's and has no `gcc`, which shows up as
`cannot find -lgomp` at link time.

**Toolchain fix, 2026-08-10**: MSYS2 updated to gcc 16.1.0 and its `libmingwex`
dropped `_gnu_exception_handler`, `__mingw_oldexcpt_handler` and the
`__mingw_initlts*` symbols that rustup's bundled `crt2.o` still references. Every
link began failing with "undefined reference" on code that compiles cleanly.
`.cargo/config.toml` now sets `link-self-contained=no` for
`x86_64-pc-windows-gnu`, so rustc uses MSYS2's startup files, which match MSYS2's
libraries. Scoped to that target; MSVC, Linux and macOS are untouched.

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

## R10.1 — constrained decoding: GBNF and JSON schema (2026-08-11)

`crates/bigtea-grammar` (new, **no dependencies at all** — not ggml, not even
the tokenizer) parses GBNF, compiles it to a stack matcher, and turns the bytes
generated so far into the token ids that may legally come next. Detail:
`docs/graph/research/gbnf-grammars-2026-08-11.md`.

Unlocks 4 of the 182 flags: `--grammar`, `--grammar-file`, `--json-schema`,
`--json-schema-file`. **The library is done; the CLI wiring is not** —
`sample.rs` and `bigtea-run.rs` belong to another session, so the hook stops at
one function:

```rust
constraint.allowed(generated_so_far).apply(&mut logits);
```

**Verified against llama.cpp, not against expectations.** A grammar that accepts
everything passes any test that only checks acceptance, so the accepted text is
llama.cpp's own output under the same grammar at `--temp 0`, and every case
also checks a rejection that is a one-character edit of it.

| grammar | llama.cpp's output | ours |
|---|---|---|
| `json.gbnf` | `{"name":"John","age":30,...}` | accepted, complete |
| `--json-schema` person | `{"name":"John","age":30}` | accepted, complete |
| `--json-schema` array | `{"city": "New York", "scores": [1, 2, 3, 4, 5] }` | accepted, complete |

Two bugs, one found by a unit test and one only findable this way:

1. **Only the first alternative of the root rule was explored.** A rule is
   entered through a `RuleRef`, which fans out over alternatives; the root has
   none pointing at it. `root ::= "cat" | "car"` took `cat` and refused `car`.
2. **Three of the eight grammars llama.cpp ships did not parse.** `json.gbnf`,
   `json_arr.gbnf` and `c.gbnf` put the rule body on the line after `::=`.
   A test that walks the whole `grammars/` directory found it on its first run.

Everything unimplemented is **refused by name, never ignored** — token literals
(`<think>`), `allOf`, `pattern`, `minimum`, `additionalProperties: true` and the
rest. Ignoring a schema keyword yields a grammar *looser* than asked for, so the
model emits output that satisfies the grammar and violates the schema, and
nothing downstream can tell.

66 tests here; 255 pass in the ggml-free CI job, which now includes this crate.

## R2 — reads now overlap compute: 1.13x on generation (2026-08-11)

**Supersedes the `R2 | overlap I/O with compute | ready, but smaller than it
looks` row in the table above.** Built, measured, and **on by default**. Detail:
`docs/graph/research/r2-overlap-2026-08-11.md`.

Block N+1's always-read weights are read **while block N computes**. Exact, not
speculative: routing is data-dependent so N+1's *experts* cannot be known before
N runs, but its **dense** tensors do not depend on routing at all.

Four runs, one session, **free RAM matched to within 0.03 GiB** — the axis these
figures drift along — with 3.10 GiB of the always-read set still streaming:

| | free | prefill | generation | dense read | expert read |
|---|---:|---:|---:|---:|---:|
| overlap off | 7.10 GiB | 0.56 tok/s | 0.280 tok/s | **2.15 s** | 7.01 s |
| overlap on | 7.13 | **0.60** | **0.316** | **0.02 s** | 8.13 s |
| on, repeat | 7.11 | **0.60** | **0.317** | 0.02 s | 8.21 s |

**1.07x prefill, 1.13x generation**, reproducible to the third decimal.

**The dense read is now free — 2.15 s to 0.02 s across 86 block-passes — and the
expert reads gave 1.16 s of it back.** That is why this is a third of the ~1.4x
ceiling rather than all of it, and the reason is measured rather than guessed:

| prefetch readers | dense | expert |
|---:|---:|---:|
| 0 (off) | 2.56 s | 7.02 s |
| 2 | 0.02 s | 8.39 s |
| 4 | 0.04 s | 8.43 s |

Two handles hide the dense read as completely as four, and four cost the experts
no more than two — so **the toll is the drive, not the pool split.** Both sets of
reads compete for the same bandwidth, and moving bytes off the critical path
does not make them free. This is `the-plateau-was-ours` read from the other side:
there the ceiling was ours, here the drive is genuinely the limit.

Two things that had to be right first:

- **`read_range_into_via` requires distinct slots**, and `read_expert_slices`
  already used all eight handles. A prefetch started naively would have
  reintroduced by hand the queue-depth-1 bug whose fix was worth 1.32x — and it
  would have shown up as "overlap does not help", not as an error. The pool is
  partitioned: foreground `0..6`, prefetch `6..8`.
- **With residency satisfied the overlap is off, not merely idle.** Shrinking the
  foreground pool to feed a thread that reads nothing is a pure loss, so the
  decision is made once per pass from whether block 1 has a non-resident tensor.

All 21 container-backed V4-Flash tests pass with it active, including the
element-sum comparisons against llama.cpp — the overlap changes *when* bytes are
read, never which. `BIGTEA_PREFETCH_OVERLAP=0` disables it;
`BIGTEA_PREFETCH_READERS` tunes the split.

## The GPU does not help a streaming MoE model — 4.3x slower (2026-08-16)

`-ngl` is a smooth win on a dense model. On the model this project exists for it
is a large loss. Qwen3-30B-A3B, medians of three, spread under 2%:

| | prefill tok/s | generation tok/s |
|---|---:|---:|
| CPU only | 1.30 | **2.61** |
| `-ngl 12` (of 48) | 1.30 | 1.44 |
| `-ngl 48` | 1.09 | **0.61** |

**Not a bug.** 76% of a token is disk, and **the experts run on the host
whatever `-ngl` says** — they stream per block into host memory and their FFN
builds its own CPU context. `-ngl` places only the resident set: 0.93 GiB, about
5% of what a token actually reads. So offloading moves the small part, leaves
the large part, and adds a host round trip for the activation at every one of 48
blocks. Putting the experts on the card is not available either — ~16 GiB
against 5.11 GiB of VRAM, the same wall that made the model stream.

**The rule: a speedup measured on a model that fits does not transfer to one
that does not.** Every GPU number published here — 25.6x on a kernel,
1.33–1.52x on a Qwen3-4B prefill, 1.79x on the `-ngl` frontier — was measured on
a model that fits, and none of them predicted this one.

`bigtea-run` warns, with the measurement in the message, when a device is opened
on a model that streams experts. Full node:
`research/gpu-does-not-help-streaming-moe-2026-08-16.md`.

## `--op-offload` works, and it cannot pay yet (2026-08-16)

The scheduled forward pass runs. `--op-offload` is implemented, produces the
same completion as every other path, and is **slower than not using it**.

**The blocking bug was one missing call: `ggml_set_input`.** The scheduler has
an explicit branch — `if (tensor->flags & GGML_TENSOR_FLAG_INPUT) cur_backend_id
= sched->n_backends - 1` — and without the flag a leaf with no buffer, no data
and no op is unplaceable, reaching `ggml_gallocr_allocate_node` as `-1`, which
aborts. It also explains why the CPU must be passed **last**. Found by bisection
in a 60-line test after two wrong guesses (the scratch buffer; the views).

| prompt | plain CPU | `--op-offload` | `-ngl 99` |
|---|---:|---:|---:|
| 11 tokens | 34.23 | 35.04 | 56.93 |
| ~900 tokens | **79.24** | **64.39** | 205.37 |

**The prediction written down first was wrong.** "A long-prefill flag or
nothing" assumed the weight copy happens once per pass. It does not: this engine
submits ~5 graphs per block — ~180 per pass — and the scheduler copies weights
**per submission**, so the copy amortises over a *block*, and prefill length
never helps. llama.cpp submits **one** graph and its copies amortise across all
36 blocks. That is the entire difference. Scheduling also gives up the 1.39x
repack, so the flag starts 19% behind before moving an operation.

**So this is a second, independent argument for
`activations-resident-across-layers`** — the first was 110 graph submissions
costing 0.64 s of allocation on a single prefill. `--op-offload` is the cheapest
test of whether fusing graphs did what it claims.

Ships off by default, printing the measurement when enabled. `ggml_set_input` is
applied on **every** path: marking an input is what it is regardless of who runs
the graph. Full node: `research/op-offload-cannot-pay-2026-08-16.md`.

## The offload frontier is a smooth dial (2026-08-16)

`-ngl` shipped with no performance number, which is a gap: a placement flag
whose effect on speed is unmeasured cannot inform a decision. Qwen3-4B-Q4_K_M,
RTX 3050, **three runs per point, medians**:

| `-ngl` | prefill tok/s | generation tok/s |
|---:|---:|---:|
| 0 | 43.29 | 6.34 |
| 9 | 48.38 | 6.41 |
| 18 | 54.57 | 6.99 |
| 27 | 63.78 | 7.06 |
| 36 | 66.49 | 7.78 |
| 99 | **77.34** | **8.85** |

Both monotonic, no knee: **1.79x prefill and 1.40x generation end to end**, with
every intermediate point on the line. That is the useful result — `-ngl` is a
dial a user sets from the VRAM they have, not an all-or-nothing switch.

**The single-run version of this table said something false.** One run per point
gave `36: 72.41` against `99: 65.80`, which reads as "offloading the output head
costs something". The three runs at 36 were 63.41 / 66.49 / **81.04** — a 28%
spread, wider than the entire difference being explained. Third time this
project has caught a causal story built on one GPU run, and the first two both
reached a published number.

**This is not the interesting frontier.** The model fits (2.33 of 5.11 GiB), so
every point was a free choice rather than a constraint. The larger-than-VRAM
curve is the one CLAUDE.md names as unpublished by anyone, and `-ngl` is what
makes it sweepable. Full node: `research/ngl-frontier-2026-08-16.md`.

## `--override-tensor`, and the same bug three times (2026-08-16)

`-ot <pattern>=<CPU|GPU>` places named tensors regardless of `-ngl`, which is
how llama.cpp users keep MoE experts off a card that cannot hold them. It reuses
the per-tensor residency `-ngl` introduced, so the flag is mostly a pattern and
two refusals.

**The pattern is a substring with `*`, not a regex, and a regex is refused.**
This workspace has no external dependencies, so a regex engine would be a new
one for a single flag. The refusal is the part that matters: `blk\.(1[0-9])\..*_exps`
treated as a literal matches nothing, the flag appears to work, and the model
loads exactly where it would have anyway — a flag accepted and ignored, which is
what the declined-flag table exists to prevent.

**A rule that splits a single block is refused too, by name.** llama.cpp can put
attention on the card and the FFN on the host inside one layer because its one
graph goes through `ggml_backend_sched`. Here a block's graph runs in exactly one
place, so `*ffn_down.weight=CPU` would build a mixed graph — and that segfaults
rather than failing.

**THE SAME BUG APPEARED THREE TIMES IN ONE DAY, and the third one is the
lesson.** Every instance was a *graph* placed by one rule while its *weights*
were placed by another:

1. `rope_freqs.weight` bound host-side while block 0 ran on the card (`-ngl`).
2. The device duplicate of it keyed on `gpu_layers > 0 && <= n_layer` — true for
   a partial `-ngl`, **false for `-ot`**, which implies a full offload the rules
   then carve into. Seven blocks read a device pointer from the host.
3. `edge_device` never consulted the overrides, so `-ot "*=CPU"` ran the
   embedding on the card over a host tensor.

Each was exit 139 with no error. The fix is structural rather than three
patches: **residency is resolved once at load into `block_placement` and
`edge_placement`, and everything that decides where a graph runs reads those.**
A second derivation of the same fact is what kept being wrong.

`-ot "*=CPU"` now reproduces the pure-CPU completion exactly, which is the
strongest check available for the flag: force everything home and the device
path must vanish.

## `-ngl` runs, and it says the device path was never checked (2026-08-16)

`ggml_backend_sched` is bound and tested. Partial offload works. And the thing
worth carrying forward is neither: **the device path fails 1 of 8 parity prompts
where the CPU path fails none, and nobody had run that comparison.**

`scripts/parity-check.sh` takes `NGL=n` now and passes `-ngl n` to **both**
engines, which is the only honest way to diff a partial offload — the
reference's own answer moves with the split. Llama-3.2-1B, RTX 3050, Vulkan:

| offload | ok | FAIL |
|---|---:|---:|
| `-ngl 0` — both on CPU | 6 (+1 unstable, +1 near-tie) | **0** |
| `-ngl 8` — 8 of 16 blocks on the card | 7 | **1** |
| `-ngl 99` — all of it | 7 | **1** |

So **`-ngl` costs nothing over `--device`** — same score, and a *different*
failing prompt each time, which is a near-tie landing differently rather than a
broken split. The 1-in-8 belongs to `--device`, and it has been there since
Phase A, which was accepted on "it runs and it is 1.73x" with **no completion
diff at all**. The GPU tier is not verified and must not be called finished.

**The first reading of this was wrong.** One prompt swept over `-ngl 0..17` had
us changing at 5 values and llama.cpp at none, which looks exactly like our bug.
Eight prompts reversed it: llama.cpp answers `A triangle has a base of 5 units`
at `-ngl 0` and `a base of 10 cm` at `-ngl 99`, and Bigtea flips the *opposite*
way. A CPU kernel and a Vulkan kernel do not produce bit-identical sums and
greedy decoding turns the last bit into a different word — in both engines.

**The scheduler is not what makes `-ngl` work**, and the two changes should not
borrow each other's credit. A mixed *graph* is undefined behaviour; a mixed
*model* is not, because this engine materialises the activation as a host
`Vec<f32>` at every block boundary. The per-block round trip that costs
everywhere else is what makes the split free here. The scheduler becomes
load-bearing when `backlog/activations-resident-across-layers.md` lands.

**One tensor broke the rule and segfaulted.** `rope_freqs.weight` carries no
`blk.` prefix but every block reads it, so hosting it while block 0 ran on the
card was a mixed graph: exit 139, no error, every `-ngl` from 1 to 16 dead while
0 and 17+ passed. It is bound on both sides now. **A tensor every block reads
must exist on both sides of a split** — it is the only one today, and a new
architecture that adds another will fail identically.

**Two near-misses, both invisible to the harness.** `CLAUDE.md`'s `GGML_LIB_DIR`
points at a ggml build with **no Vulkan archive**, and the GPU tests *skip*
rather than fail without a card — so `6 passed` was reported for a file whose
two GPU tests had never run, and the scheduler commit's first draft claimed a
mixed graph had computed when it had not. And `splits() >= 2` was asserted on a
**single-node** graph, which cannot split however its operands are placed: an
unfalsifiable assertion, only revealed when a real card started evaluating it.

**The 1-in-8 turned out to be arithmetic, measured the same day.** `bigtea-gpubench`
grew `--prompt <text>` and a real comparison — the old one was
`sum(|logits[0..64]|)` to four decimals, which is what Phase A's "logit
checksums agree" rested on and cannot see the top token move. On all eight
parity prompts the device picks the **same first token**; the kernels disagree
by **0.37–0.71** (mean 0.06–0.09) and the model's own top-2 margin falls to
**0.399**, so on `Dear Sir or Madam` the difference is 94% of the margin. Within
a 32-token continuation some position has a margin under 0.4 and the token
flips. **A wiring bug does not agree on 8 of 8** — and this is why a text diff
is not a valid acceptance test for a GPU path in any engine, which llama.cpp's
own 2-in-8 flip rate was already saying. Still unproven either way: whether our
spread is larger than llama.cpp's, which needs its logits rather than its text.

Full node: `research/ngl-partial-offload-2026-08-16.md`.

## R12 — the 256-token V4-Flash context cap is gone (2026-08-11)

**Supersedes the "V4-Flash is capped at 256 tokens of context" entry under Known
limitations.** Issue #46. Detail:
`docs/graph/research/ring-wraparound-2026-08-11.md`.

The raw KV latents were `kv_lora_rank * 256` per layer **indexed by absolute
position**, so position 256 wrote past the end. They now live in a 1024-slot
ring; the compressed half grows. The container declares
`context_length = 1048576` — the cap was ours.

**The only limit left is on one pass: 897 tokens**, which chunking satisfies
(`-b` defaults to 256). The error reports the batch limit rather than a sequence
limit, because chunking is what a caller can act on.

Why a ring is exact, and where it would not be:

| structure | indexed by | fix |
|---|---|---|
| `raw` | absolute position | **ring**, `position % 1024` — sound only because raw attention is *sliding* (`attention.sliding_window = 128`), so a position older than the window can never be read again |
| `comp` | **block** index | **grows; cannot be a ring** — the compressed half is visibility-limited, not windowed, so every complete block behind a token stays reachable |
| compressor input ring | `pos0`-relative | already correct, untouched |

`sliding_window = 0` would mean full causal attention, where a ring would
silently drop keys still in scope; that case is refused rather than served.

The ring size is the **window plus the batch**, not the window: a pass's
*earliest* query still reaches `window - 1` behind `pos0`. Measuring from the
last query instead would drop exactly the keys the first rows of a prefill need.
45 MB across 43 layers, against 11 MB before.

The mask was rewritten with it, as the cache's own comment said it would have to
be — the key axis is no longer the slot index but a gathered run of absolute
positions, and handing the mask slot indices would attend to whatever `p % 1024`
held.

Verified with the R3 equivalence harness past the old cap — `prefill(0..=257)`
against `prefill(0..257)` + `step(257)`:

```
past 256: argmax 91 agrees; sums 350740.59 vs 352047.19 (0.373% apart)
```

Not bit-identical, deliberately: routing flips on near ties when the batch shape
changes. 22 container-backed tests pass at 2, 5, 165 and 258 tokens — which is
Raw, CSA and HCA, since prompt length decides which builder runs. `raw_span` is
a pure function with unit tests covering wraparound, the batch limit and the
property the whole design rests on: no two positions in one span share a slot.

~~**Still stale, and not mine to change**: `bigtea-serve.rs` reports
`context_limit() = 256` for deepseek4, so the server refuses sequences the engine
now handles. One line, and it belongs to whoever owns that file.~~

**CLOSED 2026-08-11 in `9f024e7`, merged at `7a81502`** — it reports 897, the
per-pass cap. Recorded because of *how* this nearly went wrong: the note above
outlived the fix, and a later session repeated "still reports 256" from the note
instead of reading the file. **A stale note reads exactly like a current fact.**

## StableLM and StarCoder2: one shared blocker (2026-08-11)

Both downloaded, run and diffed. **Neither is verified.** They fail for the same
reason, which is why the work is scoped as a feature and not as two models:
`docs/graph/backlog/layernorm-and-biases.md`.

```
stablelm -> ??地なutorsemie路emieemieا起
```

The qwen2 CJK-noise signature. What is missing, after #60's Q/K/V bias support:

1. **`bigtea-ggml` has no LayerNorm.** It binds `ggml_rms_norm` and not
   `ggml_norm`. LayerNorm subtracts the mean and carries a **bias**; RMSNorm
   does neither. The tell is `attn_norm.bias` in the container, and the metadata
   key being `attention.layer_norm_epsilon` rather than `..._rms_epsilon`.
2. **Biases beyond Q/K/V.** #60 added `attn_bias` (Q/K/V, detected from
   `blk.0.attn_q.bias`). StarCoder2 also needs `attn_output.bias`,
   `ffn_up.bias` and `ffn_down.bias`.
3. **Partial RoPE.** `rope.dimension_count` is ignored — `head_dim` is passed as
   `n_rot` unconditionally. StableLM declares **16 of its 64** dimensions, so its
   rotation is wrong today. **This is a real bug beyond StableLM**: any container
   declaring the key is currently over-rotated.
4. **Ungated FFN.** StarCoder2 has no `ffn_gate` — plain MLP with GELU rather
   than SwiGLU. `FfnAct` (added by #60) is where an ungated variant belongs, and
   `ctx.gelu()` now exists.

LayerNorm plus biases is also the shape of falcon, gpt2, gptneox, bloom, phi2
and starcoder, so building it once moves the count by more than these two.

Ruled out: the tokenizer (both declare `gpt2`, supported) and the RoPE
convention (both NeoX, already mapped). The failure is entirely in the block.

## StarCoder2 verified; StableLM is one tokenizer line away (2026-08-11)

**`VERIFIED_ARCHITECTURES` is nine** — `starcoder2` added, 3/3 exact on
`parity-check.sh`. StableLM is **not** added; its block is right and the
remaining difference is in the tokenizer.

What the dense path gained, all detected from the container rather than by name:

- **LayerNorm.** `bigtea-ggml` now binds `ggml_norm` beside `ggml_rms_norm`.
  A norm carrying a bias *is* a LayerNorm — RMSNorm never centres and has no
  shift — and substituting one was the fluent CJK noise both models produced.
- **The full bias set.** `attn_output`, `ffn_up`, `ffn_down` and the norms,
  on top of the Q/K/V biases.
- **Partial RoPE.** `rope.dimension_count` was **ignored entirely**;
  `head_dim` went in as `n_rot` unconditionally. StableLM rotates 16 of its 64.
  This was a real bug beyond StableLM — any container declaring the key was
  over-rotated.
- **Ungated FFN.** `down(gelu(up(x)))` when there is no `ffn_gate`.

**Two traps, both caught by the reference and not by an error:**

1. **A bias that is not in `required_tensors` is never loaded**, and the graph
   then silently skips it — `weights.get` returns `None` and the shift is simply
   not applied. StableLM read *almost* right for exactly this reason. The
   easiest to miss is `output_norm.bias`: applied once, so a wrong final norm
   shifts every logit by the same vector and the text stays fluent.
2. **A missing `ffn_gate` means two different things.** Phi-3 fuses gate and up
   into one tensor twice `n_ff` wide; StarCoder2 has no gate at all. Testing for
   the tensor alone made Phi-3 ungated and **broke a verified architecture** —
   caught by the regression sweep, which is why it runs. The shape separates
   them.

**StableLM: the block is correct, the tokenizer is not.** Two of three prompts
match exactly; `def fibonacci(n):` tokenizes to **4 tokens where llama.cpp makes
5**, so the prompt differs before a single weight is read. The cause is ours and
recent: `tokenizer.ggml.pre` is **absent** in that container, and
`Tokenizer::from_metadata` falls back to `"llama-bpe"` where llama.cpp's default
is the plain GPT-2 rule. A6c refused every unknown `pre` **by name** and then
guessed the absent case, which is the same mistake one layer down.

The fix is a `default` GPT-2 variant in `pretok.rs` plus one line in
`crates/bigtea-tokenizer/src/lib.rs` — a file another session owns, so it is
reported rather than taken.

Regression sweep after these changes, `parity-check.sh` at 32 tokens: gemma2,
gemma3, qwen3-4b, qwen2, tinyllama, starcoder2 all 3/3; llama32-1b and phi3 2/3
plus one `unstable`, which is llama.cpp disagreeing with itself on a near-tie
and is documented. 411 workspace tests, clippy and fmt clean.

## StableLM verified — the absent pre-tokenizer was guessed (2026-08-11)

**`VERIFIED_ARCHITECTURES` is ten.** `stablelm` added, 3/3 exact.

The block had been right since LayerNorm landed; the last difference was the
**tokenizer**, and the bug was ours and recent. When `tokenizer.ggml.pre` is
**absent**, `Tokenizer::from_metadata` fell back to `"llama-bpe"`. llama.cpp
falls back to its `LLAMA_VOCAB_PRE_TYPE_DEFAULT` GPT-2 rule.

```
llama-tokenize  "def fibonacci(n):"  ->  def / ' fibonacci' / ( / n / '):'   5
bigtea, before                       ->                                      4
bigtea, after                        ->                                      5
```

A6c refused every *unknown* `pre` **by name** and then quietly guessed the
**absent** case — the same mistake one layer down from the one it fixed.

**The default is structurally unlike the other variants**: four regexes applied
in **sequence**, each splitting what the last produced, rather than one ordered
alternation. The first pass cuts a run of punctuation out *whole and first*, so
`(n):` becomes `(` `n` `):` before anything else runs. That single pass is the
entire disagreement.

**It also narrows a claim made an hour earlier.** `starcoder2` was verified 3/3
while running this same wrong fallback — it declares no `pre` either, and only
agreed because its merge table differs from StableLM's. It was re-run after the
fix and is still 3/3, so the entry stands; but "verified" meant less than it
looked at the time, and the re-run is what makes it mean what it says.

Containers affected: any `gpt2`-BPE container omitting the key. Of those on
disk, `stablelm` and `starcoder2`. Everything that declares its `pre` explicitly
— qwen2, qwen3, llama32-1b, v4flash — is untouched and re-checked unchanged.

Regression sweep after the fix: stablelm 3/3, starcoder2 3/3, qwen2 3/3,
qwen3-4b 3/3, gemma2 3/3, llama32-1b 2/3 + one documented `unstable`.
414 workspace tests, clippy and fmt clean.

## Eight prompts instead of three: three bugs, two of them in "verified" code

`ticket/r14-architectures`, 2026-08-11. Four architectures were on the list —
olmo, falcon3, internlm2, baichuan. Three of them needed almost nothing. The
harness change that preceded them is what earned the session.

**`VERIFIED_ARCHITECTURES` is thirteen**: baichuan, deepseek4, gemma2, gemma3,
internlm2, llama, olmo, phi3, qwen2, qwen3, qwen3moe, stablelm, starcoder2.

### The three bugs, all pre-existing on `main`, all confirmed by stashing

| bug | before | after |
|---|---|---|
| `rope_freqs.weight` never read (Llama-3.1/3.2/3.3) | 3 ok / 4 unstable / 1 FAIL | **8 ok** |
| no BOS for a BPE container that declares none (Falcon3) | 1 ok / 5 unstable / 2 FAIL | **8 ok** |
| USER_DEFINED token byte-decoded instead of copied | newlines vanished | **byte-exact** |

The first is the serious one. Llama-3.1 onwards carry `rope_scaling = "llama3"`
as a **tensor** — `rope_freqs.weight`, `n_rot/2` per-frequency divisors, handed
to `ggml_rope_ext` as `freq_factors`. We passed `None`. The metadata reports
`rope scaling = linear, freq_scale_train = 1` whether or not the tensor exists,
so nothing announces it; llama.cpp's only sign is one debug line. `llama` has
been in `VERIFIED_ARCHITECTURES` the whole time.

It needed two changes, and the second is the trap: the tensor had to be added to
`required_tensors()`, or it is **never loaded**, `weights.get` returns `None`,
and the rotation is quietly the un-extended one. Same shape as StableLM's
missing biases.

### The rule those bugs cost

**"The reference disagrees with itself" is not a safe verdict.** The harness
re-runs a mismatch under `-fa off` and `--no-repack` and calls the prompt a
near-tie if llama.cpp's answer moves. That compares the reference *to itself*. It
cannot see that **our input differed** — and when it does, a near-tie is exactly
the symptom, because the model is answering a slightly different question.

**Nine of the eleven `unstable` verdicts in this session were bugs.** One near-tie
in eight is ordinary; five is a bug not yet found.

Also fixed in the harness: `llama-completion` prints ` [end of text]` on EOS and
Bigtea prints no equivalent, so any model terminating early read as a FAIL whose
two sides were identical (`bigtea: 42` vs `llama.cpp: 42 [end of text]`).

### What the four architectures actually needed

- **olmo** — one real feature: **non-parametric norms.** llama.cpp builds every
  one as `build_norm(x, NULL, NULL, LLM_NORM)`, and the container holds no
  `attn_norm.weight`, `ffn_norm.weight` or `output_norm.weight`. `layer_norm` and
  `norm_bias` had to split into two booleans — they were one because every
  LayerNorm so far had a bias, and OLMo made the loader demand an
  `output_norm.bias` that cannot exist. Also: **`olmo` was listed as NeoX RoPE
  with `known = true`** while llama.cpp lists it in the NORM branch. A guess
  wearing the label of a checked fact.
- **internlm2** — 8/8 first run; only needed the NORM RoPE entry.
- **baichuan** — 8/8 on the 7B. **The 13B is now refused**: llama.cpp gives it
  ALiBi by *layer count* (`n_layer == 40`), the two share a tensor set and an
  architecture name, and the 13B would load, rotate keys it should not, and
  answer fluently.
- **falcon3** — **not a new architecture.** It converts to `llama`, and `falcon3`
  is one more alias in llama.cpp's `llama-bpe` arm. Everything it exposed was in
  shared code. Its container is also the reason `gpt-2` and `default` are now
  separate pre-tokenizers here: they are separate entries in llama.cpp
  (`PRE_TYPE_GPT2` is one regex, the `default:` arm wraps it in three more
  passes) and `from_name` had mapped `gpt2` onto `default`.

### Scoreboard, one session, one build, `parity-check.sh <model> 32`

```
OLMo-1B.Q4_K_M                    8 ok  NEW    Qwen2-0.5B-Instruct        8 ok
internlm2-math-plus-1_8b.Q4_K     8 ok  NEW    Qwen3-4B                   8 ok
baichuan2-7b-chat.Q4_K_M          8 ok  NEW    gemma-2-2b-it              8 ok
Falcon3-1B-Instruct (arch llama)  8 ok         gemma-3-1b-it              8 ok
stablelm-2-1_6b-chat              8 ok         Llama-3.2-1B-Instruct      8 ok  fixed
starcoder2-3b                     8 ok         tinyllama-1.1b-chat        8 ok  fixed
                                               Phi-3-mini-4k    6 ok, 2 unstable
```

426 workspace tests, clippy `--workspace --all-targets -D warnings` clean, fmt
clean.

**Not done**: the `clamp_kqv` path (MPT/DBRX/OLMo) is written against
llama.cpp's code, not a run — OLMo-1B declares `0.0`. Phi-3's two unstable
prompts are unexamined, and after nine `unstable` verdicts turned out to be bugs,
"it was already like that" is a weak defence. Containers live at
`C:/Projects/models/{olmo,internlm2,falcon3,baichuan}/` and are the only copies
on this machine.

## The GPU tier, scoped before any code — and the guessed slice does not survive

`research/gpu-tier-smallest-honest-slice-2026-08-11.md`, 2026-08-11. Written as
a scoping node on instruction, with no GPU code attached.

**The hypothesis was "N dense layers resident in VRAM, experts still streamed to
host". Measured, it fails twice.**

| model | always-read (dense) | routed experts | verdict |
|---|---:|---:|---|
| DeepSeek-V4-Flash-UD-Q4_K_XL | **7.38 GiB** | 137.06 GiB | **does not fit** 6.0 GiB of VRAM |
| Qwen3-30B-A3B-Q4_K_M | 0.93 GiB | 16.35 GiB | fits, with nothing worth moving |

For V4-Flash the dense half is larger than the card, so that variant needs a
mixed-device graph and a `ggml_backend_sched` — the *largest* possible first
slice. For Qwen3-30B it fits with 5 GiB spare, but of 5.4 s accounted in a
measured run the entire dense path is **9%** (disk 52%, expert compute 37%). A
**1.10x ceiling**, below the 1.4x already unclaimed in R2's overlap.

Moving the *expert* matmuls instead addresses 37% but pushes ~1.15 GiB/token
over PCIe — the same shape this project already built and reverted (contiguous
experts, ~1.02 GB/token, byte-identical output, **1.34 → 1.27 tok/s**), with a
bus added.

**Blocker (a) is worse than this file recorded.** It is not "needs a
CUDA-enabled ggml": there is **no CUDA toolkit on this machine at all** —
`nvcc` absent, no `ggml-cuda.a` — only a CUDA-capable driver (610.74).

**Blocker (b) is one line.** `crates/bigtea-ggml/src/weights.rs:286` writes a
host pointer into `tensor->data`. `ggml-cuda` cannot be handed one; a device
tensor is filled by a copy. So a GPU path is a second `bind_shared` plus a
scheduler, not a flag.

**The slice that does survive: VRAM as a read cache in front of the disk,
computing nothing.** It never binds a device tensor, so blocker (b) is
sidestepped rather than solved; it needs the CUDA runtime rather than a second
ggml build; and its failure mode is *slower*, not *wrong*, which is the only GPU
change with that property. It pays where VRAM is a meaningful fraction of the
expert bank — 31% of Qwen3-30B's 16.35 GiB, **3.6% of V4-Flash's 137 GiB** — so
the 20–70 GiB class, not the model this project talks about most.

**Recommended next action is not a GPU ticket.** Sweep tok/s against host cache
size first: the VRAM tier's value is a point on a curve that does not exist yet,
and if the curve has already flattened, the tier is dead for the same reason the
byte-reduction roadmap closed. That sweep needs no toolkit and no new code.

PCIe bandwidth in that node is labelled arithmetic, not measurement — it cannot
be measured until the toolkit is installed.

## The tok/s-versus-RAM frontier, measured — and it says no GPU ticket

`research/ram-frontier-qwen3-30b-2026-08-12.md`, 2026-08-12. The first
published curve of generation speed against **owned** cache size for a model of
this class. It can be swept at all only because this engine is told how much RAM
to use; `mmap` cannot be asked for exactly N GiB.

Qwen3-30B-A3B, `--cache` 1→12 GiB, `-n 16`, five interleaved rounds, medians,
free RAM sampled on every row.

| `--cache` | tok/s | vs 1 GiB | streamed | evictions |
|---:|---:|---:|---:|---:|
| 1 GiB | 0.78 | 1.00x | 12.13 GiB | 1758 |
| 2 | 1.62 | 2.08x | 9.34 | 1957 |
| 4 | 1.85 | 2.37x | 6.69 | 1286 |
| **6** | **2.63** | **3.37x** | **5.53** | **0** |
| 8 | 2.56 | 3.28x | 5.53 | 0 |
| 10 | 2.13 | 2.73x | 5.53 | 0 |
| 12 | 2.56 | 3.28x | 5.53 | 0 |

**Rises to 6 GiB, flat after: 3.37x for 6 GiB of owned residency.** It flattens
for a *capacity* reason the engine reports directly — at ≥6 GiB `evictions` is
**0** and `streamed` is 5.53 GiB, which is what 16 generated tokens of this
prompt distinctly touch. Below it the same run re-reads what it already had.

**The 8/10/12 rows are a free null** — provably one configuration, so their
16.8% median spread is the noise floor. Nothing above 6 GiB is distinguishable;
the 1→6 climb is far outside it.

### Two methodological findings that outlive the numbers

**A wrong activation is a wrong residency benchmark.** Fixing GELU-for-SiLU on
this model moved streamed bytes **7.00 → 5.53 GiB** and hits 80% → 70%, because
different FFN outputs become different router inputs and select **different
experts**. The pre-fix sweep measured a different workload. Do not benchmark a
cache on an unverified model.

**The free-RAM column is not decoration.** A first attempt had an entire round
flattened by this session's own git work releasing memory — visible only as free
RAM *rising* 8.7 → 10.4 GiB mid-round. Without the column it would have been
folded into the medians.

### Verdict for the VRAM tier

**No GPU ticket, now measured rather than argued.** The flat region already fits
in this machine's 9–10 GiB of free RAM, so VRAM adds nothing on this model. 5 GiB
of VRAM is 31% of Qwen3-30B's expert bank and **3.6% of V4-Flash's** — neither is
the window where a second tier changes the shape.

But **where the curve flattens is a property of the workload, not the hardware**:
it saturates at 5.53 GiB because that is what 16 tokens touch, and distinct
expert bytes grow with generation length. The frontier is a *surface* in (cache
size, tokens generated) and only one slice of it exists. That slice is the next
measurement, and it is not a GPU ticket either.

**Caveats, stated in the node:** one prompt, one machine, one session; `-n 16`;
round-over-round drift of ~25% with free RAM stable and the cause unidentified;
and **Qwen3-30B-A3B is not in `VERIFIED_ARCHITECTURES`** — it was delisted the
same day for a remaining stable-reference divergence, and it is the only
container here in the size class where the curve is interesting.

## The knee moves with `-n` — the slice above was the flattering one

`research/the-knee-moves-with-n-2026-08-14.md`, 2026-08-14. The measurement the
section above asked for: the second axis, 3 rounds × `-n` {16, 64, 256} ×
`--cache` {1, 2, 4, 6, 8, 12}, interleaved on both, free RAM every row.

**The working set grows with what you generate.** Read off the `evictions = 0`
rows, where `streamed` is the whole distinct working set:

| `-n` | working set | first budget with 0 evictions | best tok/s |
|---|---|---|---|
| 16 | 5.53 GiB | 6 | 3.13 |
| 64 | 7.05 GiB | 8 | 4.38 |
| 256 | 10.14 GiB | 12 | 4.70 |

So **"the frontier is flat after 6 GiB" was a statement about sixteen tokens.**
At 256 the knee is 12 GiB and tok/s is still climbing there — it had not
flattened by the largest budget swept. Growth is strongly sublinear (16× the
tokens, 1.83× the set, ≈`n^0.22`), but it extrapolates a 2048-token generation to
**~14–18 GiB of expert cache on a 15.7 GiB machine**. The honest product claim
is *the largest model at the speed you want, **for the length you generate***.

**More cache made it slower at identical work.** At `-n 16`, budgets 6/8/12 read
the same 5.53 GiB, hit the same 70% and evict nothing — byte-identical work — and
run 3.13/3.02/2.91. That is 7% lost to memory the OS could have used: the
page-fault-wearing-a-hit's-disguise effect, measured under control rather than
inferred, and invisible to the hit counter. It appears only where the budget
*exceeds* the working set; at `-n 256` more is monotonically better.

**Two methodological results.** `streamed`, `hit%` and `evictions` were
**bit-identical across all three rounds** — the workload is deterministic and
only wall-clock moves, which is what makes 18 cells from 2 clean rounds
trustworthy. And **contamination is a property of the period, so discard the
round, not the row**: round 1 ran at 0.25 tok/s where the clean rounds agree on
2.48, and a naive "free ≥ 4 GiB" row filter still admitted a row showing 7.45
GiB free that ran 5× slow.

Same caveats as the slice it extends, plus one more: the sweep needs `--force`,
because **`qwen3moe` refuses to run without it** — 0 FAIL but 6 of 8 prompts
unstable under the widened harness.

## `unstable` was answering the wrong question — 6 of 8 is really 2 of 8

**2026-08-15, and it corrects the line directly above.** The harness classified a
disagreement by asking *"does llama.cpp disagree with itself here?"* and the
report read as though that settled *"is Bigtea's output one of the things it
disagrees between?"* Those come apart precisely where it matters, and **the nine
of eleven `unstable` verdicts that turned out to be real bugs were all the second
kind**. Same model, same prompts, same build, with the two separated:

| | |
|---|---|
| `ok` — matches the default | **2** |
| `near-tie` — reproduces one of llama.cpp's *own* no-op outputs **byte for byte** | **4** |
| `unstable` — a **third** answer it never gives | **2** |

**Four of the six were never unexplained.** So the evidence for a defect in the
`qwen3moe` path is **2 of 8, not 6 of 8** — below the cluster threshold rather
than absent, and the harness now exits 0.

**The variation is the evidence, more than the count.** Which configuration we
land on is not constant: `-b 1` twice, `-fa off` once, `-b 1 -fa off` once. A
systematic defect would be systematic — quietly running batch-1 semantics would
reproduce `-b 1` on *every* such prompt. Three different configurations across
four prompts is what a real near-tie looks like. **So the discriminator is a
diagnostic and not only a verdict:** a *constant* answer would name the behaviour
we share, and would be the lead.

Two prompts are still outside the band. `Q: What is 17 plus 25? A:` was examined
first because arithmetic has a right answer, and **it came back the opposite way
to the guess**: Bigtea emits `42`, exactly as every reference configuration does.
The earlier "it skips the answer" reading was an artefact of capturing the two
sides with different tail-truncation. It was flagged as not citable before anyone
acted on it, which is the only reason it cost nothing.

The reference spans **three distinct outputs across five configurations** on that
prompt — `42`, `A: 42` on its own line, and `17 + 25 = 42` — so the continuation
after the answer is barely determined at all. Bigtea is a fourth, agreeing with
`-fa off` at the token where the reference splits. **That is weak evidence of a
defect, not strong**: the bugs this harness has caught (Llama-3.2's RoPE,
Falcon3's short prefill) broke prompts that had a determined answer, and this one
gets the determined part right. `research/parity-band-discriminator-2026-08-15.md`
carries the full table.

The threshold moved without moving: three-in-eight still fails, but on the
sharper class, which is *stricter* — everything excusable has been taken out of
it. And a bound was added the other way, because every configuration added widens
the band and "in band" gets cheaper as the probe grows: six ties in eight now
fails too.

## The GPU tier, step 1: the card works — 25.6x, and it is llama.cpp's number

`research/gpu-the-card-works-vulkan-not-cuda-2026-08-15.md`, 2026-08-15.

**GPU is still 0%.** Nothing here is Bigtea. This is the precondition the ticket
set — *if llama.cpp cannot use the card, we cannot either* — answered in an hour
rather than three days, which is what step 1 was for.

**CUDA is not installable here without a toolchain migration.** `nvcc` on Windows
supports only MSVC as its host compiler, and this machine has **no MSVC at all**,
while everything the project builds with is MSYS2 mingw64 (`cc.exe`, `c++.exe`,
`cmake.exe`, gcc 16.1.0). The CUDA route is: install Visual Studio Build Tools,
build `ggml-cuda` with MSVC, then link an MSVC static library into a **GNU-target**
Rust binary — against the `.cargo/config.toml` workaround `CLAUDE.md` says not to
delete. That is a decision, not a step 1.

**ggml's Vulkan backend compiles with the compiler already in use.** Eight MSYS2
packages, verified first not to touch gcc/binutils/CRT; the driver already shipped
the loader; built into a separate `build-vulkan` so the 507 tests keep pointing at
the CPU ggml. `-D_WIN32_WINNT=0x0A00` is required — vendored `cpp-httplib` calls
`::CreateFile2` from `common`, so `-DLLAMA_BUILD_SERVER=OFF` does not avoid it.

Qwen3-4B-Q4_K_M (2.32 GiB, fits VRAM), llama.cpp `daef2b3`, one session, `-r 2`:

| config | pp512 | tg128 |
|---|---:|---:|
| CPU, 20 threads | 79.65 ± 5.93 | 3.65 ± 0.10 |
| CPU, 4 threads | 40.25 ± 0.95 | **6.39 ± 0.08** |
| **RTX 3050, `-ngl 99`** | **2042.60 ± 5.52** | **56.53 ± 0.04** |
| Intel iGPU, `-ngl 99` | 38.13 ± 2.09 | 3.26 ± 0.03 |
| RTX 3050, `-ngl 0` | 497.82 ± 243.16 | 3.42 ± 0.08 |

**Against the best CPU configuration of each: prefill 25.6x, generation 8.8x.**

**Two rules, not footnotes** — this project has retracted a competitive claim
before. **(1) The baseline must come from the baseline's build.** `-ngl 0` on a
GPU build is the GPU backend with nothing offloaded, not the CPU path: it reads
3.42 tg128, *below* the real CPU 6.39, with a ±49% error bar on prefill, and
quoting it buys a fake 16x. A disabled accelerator is not a control.
**(2) Tune the baseline before you beat it.** `llama-bench` defaulted to 10
threads, which is wrong for both phases; against that default this would have
read 30.1x instead of 25.6x.

**Our `-t`/`-tb` finding reproduces on the reference.** Prefill 40.25 → 79.65
going 4 → 20 threads, generation 6.39 → 3.65 going the other way — the same two
levers pulling opposite ways, at the same crossover, on llama.cpp's own binary.
That is independent confirmation of the threading work, not a quirk of our
scheduler.

**The Intel iGPU is not a second tier, and it is the attractive idea.** It has
more free memory than the discrete card (7387 vs 5233 MiB) and `uma: 1`, so the
upload problem would not exist there — and it runs 0.48x the CPU on prefill and
0.51x on generation. It has no matrix cores and it shares the DRAM the CPU path
already saturates: **a UMA device removes the copy, not the bottleneck.** Full
node: `research/the-igpu-is-not-a-tier-2026-08-15.md`.

**Blocker (b) is untouched and is still the whole ticket.** A Vulkan tensor is
filled by `ggml_backend_tensor_set`, which copies, exactly as CUDA would. Vulkan
removes an MSVC migration from *in front of* the work; it does not touch the work.
And 76% of a token on the MoE path is disk, which no GPU fixes — the 25.6x above
is prefill on a model that fits in VRAM, the one slice this card can plausibly win.

## The GPU tier, Phase A: the card runs a full prefill — at 1.33–1.52x

`research/phase-a-device-prefill-2026-08-15.md`, 2026-08-15. Bigtea's own binary
runs a complete Qwen3-4B prefill on an RTX 3050 through Vulkan, every weight
resident on the card.

```bash
bigtea-gpubench C:/Projects/models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf --repeat 3
```

| | cpu (`-t 20`) | device | ratio |
|---|---:|---:|---:|
| median | 52.68 tok/s | 80.02 tok/s | **1.33–1.52x** across invocations |
| range | 48.84–59.93 | 73.09–80.27 | warm-up discarded |

Logit checksums agree (625.01 vs 621.17), so it is the same answer, faster.

**Two figures from the same day are retracted, and the second one reached a
merge-commit headline.** `#68` merged as "…at 1.73x"; that came from one prefill
per process. **The repeat harness says 1.33–1.52x and that is the number.** An
earlier 0.42x was a cold Vulkan pipeline cache — the driver persists compiled
shaders to disk, so run 1 of any GPU path is a different program from run 2.

**Two rules came out of it, both now enforced by `bigtea-gpubench` itself.**
A GPU measurement needs **repeats** — `--repeat 1` is refused without `--force`.
And **nothing expensive belongs inside the timed region**: the first harness
reloaded 2.32 GiB per run and swung the CPU baseline 26.48–67.35 tok/s, a 2.5x
spread that buried the effect being measured.

**Where the device time goes**, measured per operation rather than attributed:
compute 1.80s over 110 graph submissions, upload 1.04s, download 0.66s, device
allocation 0.64s over 110 allocations. Transfers are 36% and allocation 14%, so
half the device's time is structural overhead rather than arithmetic.

**This is not a differentiator and STATUS must not claim it is.** llama.cpp does
2042 pp512 on the same card and model with the same ggml underneath, because it
runs one graph for the whole pass with no host round trips. We submit 110. The
gap is our design, not the kernels — `backlog/activations-resident-across-layers.md`
sizes closing it at 2.5–3x, still far short of llama.cpp.

**`ggml_backend_sched` is mandatory for Phase C, proven not assumed**:
`research/mixed-residency-segfaults-2026-08-15.md`. And Phase C's ceiling is
revised **down** from 1.3x — that estimate assumed the compute moved for free,
and Phase A shows it does not.

## The architecture count overstates the work, and here is the evidence

**2026-08-16.** Three containers downloaded to verify three "new" architectures.
Two of them declare `llama`:

| container | `general.architecture` | verdict |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | **`llama`** | 8/8 exact — but verifies `llama`, already on the list |
| Yi-1.5-6B-Chat | **`llama`** | same family, same path |
| gemma-1.1-2b-it | `gemma` | **8/8 twice — genuinely new, now verified** |

**A GGUF names an architecture, not a model family.** Mistral, Yi, Vicuna,
Zephyr, TinyLlama, WizardLM and most fine-tunes all ship as `llama`, so they run
today and always did. "12 of 141" counts llama.cpp's *dispatch arms*, and a
large share of the models people actually run funnel through a handful of them.

That does not make the 141 wrong — those arms are real and some are genuinely
different models. It makes **the bar a poor proxy for coverage**, and it means
the honest question is "does the model you have run?", not "how many arms are
implemented".

**Mistral's first run failed with `梦梦梦梦…` and llama.cpp emitted nothing.**
That was a corrupt download — two fetch processes resuming into one file, 4973 MB
against an expected ~4370 MB — not a forward-pass bug. **Two engines failing the
same container is a file problem**, and the size said so before any debugging
did. A clean re-download passed 8/8.
