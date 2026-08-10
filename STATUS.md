# STATUS — where Bigtea is, and what is left

**Read this first, in any session.** It is the single place that says what is
true today. Update it in the same commit as any change that moves a number or
closes a task; if it disagrees with a doc, this file is wrong and the doc is
right, so fix this file.

**Last updated**: 2026-08-10 · **Version**: v0.0.2 · **Branch**: `main` ·
**Open PRs**: [#44](https://github.com/aturzone/Bigtea/pull/44) — R3, the KV
cache. `ticket/r5-product` (release workflow, `bigtea-pull`, `bigtea-serve`) and
`ticket/r7-factored-experts` (this session's measurements). **All unmerged, Atur
merges.** PR #43 (R0/R0.1/R1) is **merged**.

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
| Qwen3-30B-A3B | generation | 1.07 | **2.16** | ~2x behind |

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
architectures. 229 tests pass.

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
