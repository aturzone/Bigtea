---
decision: The best real path to run Kimi K3 on Atur's exact machine
status: proposed        # needs Atur's go/no-go on the download only
links: [../research/k3-on-16gb-feasibility.md, ../research/fixed-hardware-design-space.md, ../research/waste-engine-verified.md]
---

Written 2026-08-03 while Atur slept, with the standing instruction: find the **best real** path
to run K3 on **this exact system**, no hardware upgrades. Fixed target:
**15.7 GiB RAM · 745.9 GiB free disk · 3.09 GB/s NVMe · RTX 3050 6 GB · Windows 11.**

## Verdict

**Download unsloth's `UD-IQ1_S` K3 GGUF (594 GB / 553.2 GiB) and run it on unsloth's llama.cpp
fork (PR #48).** It fits the disk with ~190 GiB to spare, needs **zero conversion**, and is an
officially supported configuration — slow, but supported. Everything except the download is
already built and verified on the machine.

## Why this beats the WASTE path I was pursuing

| Path | Download | Conversion | Container | Fits 745.9 GiB? |
|---|---|---|---|---|
| A — WASTE, 3-stage VQ (default) | 1.56 TB | heavy | 982 GiB | **No** |
| B — WASTE, `--stages 2` (~2-bit experts) | 1.56 TB | heavy | ~665 GiB (D) | Yes |
| **C — unsloth UD-IQ1_S GGUF** | **553.2 GiB** | **none** | **553.2 GiB** | **Yes, +190 GiB spare** |

Path C downloads **~1 TB less** than A/B and skips conversion entirely. Paths A/B require pulling
the full 1.56 TB source — which **does not even fit on this disk**, forcing a streaming
download→convert→delete pipeline that nobody has built — and then re-quantizing 2.78T parameters
on a laptop CPU, which is its own multi-day-to-infeasible compute problem. Path C's quantization
was already done by unsloth on their hardware.

The `--stages 2` discovery (a real, shipped flag that would put K3 at ~665 GiB) is still
interesting and is *the* fallback if C's quality disappoints — but it is strictly more expensive to
reach. **Do not start there.**

## The full menu of K3 quants that fit (checked 2026-08-03, HF API, sizes in GiB)

| Repo | Size | Pruned? | ~Download @10 MB/s |
|---|---|---|---|
| prometheusAIR/Kimi-K3-REAP55-GGUF (IQ1_M) | **319.0** | 55% of experts removed | ~9 h |
| hellohazime/Kimi-K3-REAP640-IQ1_S-GGUF | **411.1** | 28.6% removed (896→640) | ~12 h |
| GrEarl/Kimi-K3-GGUF-IQ1_S | 528.0 | no | ~15 h |
| **unsloth/Kimi-K3-GGUF UD-IQ1_S** | **553.2** | **no** | **~16 h** |
| unsloth IQ2_XXS | 662.2 | no | ~19 h |

**Critical correction to an intuition worth stating, because it inverts the obvious conclusion:
REAP pruning saves disk and download, but NOT speed.** Pruning shrinks the expert *pool*
(896→640), yet each token still routes to **16** experts — so expert bytes read per token are
**unchanged**. A pruned model is not a faster model here; it is only a cheaper one to obtain.

**Recommendation stands on UD-IQ1_S (553.2 GiB)** despite being the largest full-model option:
it is the only one with published quality benchmarks (unsloth's KLD/PPL table), and running the
*unpruned* model keeps the milestone claim clean — "we ran K3" rather than "we ran a 71%-expert
derivative of K3." The extra ~4 hours of download buys an unambiguous result.
**If download time proves to be the real pain**, `REAP640-IQ1_S` (411.1 GiB) is the fallback: its
28.6% prune sits inside the 25–50% band REAP's paper reports as near-baseline, and it saves
142 GiB. `REAP55` at 319 GiB is past that band — treat its quality as unknown.

## The evidence that this configuration works at all

- unsloth's own guide, verbatim: *"Best rule of thumb: RAM+VRAM ≈ the quant size; otherwise **it'll
  still work, just much slower due to disk offloading**."* (unsloth.ai/docs/models/kimi-k3)
  Their *recommended* RAM for UD-IQ1_S is 610 GB; we have 15.7. We are firmly in the
  "works, much slower" regime — which is exactly the milestone being attempted.
- UD-IQ1_S is a **dynamic** quant: experts pushed to ~1-bit while attention/router/shared weights
  are kept at higher precision. That is the same mixed-precision principle WASTE arrived at
  independently, already applied and already benchmarked by unsloth (their published KLD/PPL table
  shows UD-IQ1_S beating a naive IQ1_M that is *larger*).
- Architecture confirmed from the K3 model card: 2.8T total, **104B activated/token**, 93 layers
  (69 KDA + 24 Gated MLA), 896 experts, **16 routed + 2 shared**, attention hidden 7168, MoE hidden
  3072, latent MoE 3584, vocab 160K, 1M context.

## Honest speed expectation

Every path lands in the same order of magnitude, because all of them are bound by the same wall:
the model cannot fit in RAM, so each token requires reading its activated slice from NVMe.

- Activated per token ≈ 104B params. Under a dynamic ~1.7 bits/param average, that is roughly
  **~18–31 GB read per token** (D — the range reflects how much of the repeatedly-read attention
  trunk the OS page cache manages to hold in ~13 GiB).
- At the **measured** 3.09 GB/s: **~6–10 s/token ceiling**.
- Realistic, after mmap's 4 KiB random page-fault overhead (llama.cpp streams via the OS page
  cache, not via WASTE's deliberate large contiguous reads): **~15–30 s/token** (D).
- Practical meaning: **a 100-token answer takes 25–50 minutes.** This is a demonstration and a
  world-first, not an assistant. That must be stated plainly whenever the result is published.

**The one free speedup available:** put the *attention/dense* layers on the RTX 3050 and keep all
MoE experts on CPU/disk (`-ngl 99 --n-cpu-moe 999` style). Attention is re-read **every single
token**, so ~6 GB of it living in VRAM is ~6 GB/token that never touches the disk — potentially
20–30% off the per-token cost. Costs nothing to try.

## What is already done on the machine (no download needed)

- **WASTE 0.6.3 built and running**: MSYS2 + MinGW-w64 gcc 16.1 installed, `C:\Projects\waste\waste.exe`
  (0.75 MB), AVX2 + AVX-512 paths compiled, full CLI verified. Kept as the fallback engine for
  Path B and as the better-engineered streaming reference.
- **unsloth llama.cpp fork cloned and K3 support confirmed**: `C:\Projects\llamacpp-unsloth`,
  branch `pr48`, containing `LLM_ARCH_KIMI_K3` and `src/models/kimi-k3.cpp`. Mainline llama.cpp
  and the fork's own `master` do **not** have K3 — PR #48 specifically is required.
- **Conversion toolchain ready**: Python 3.11 + torch 2.13.0+cpu + safetensors + numpy.
- **3.04 GB of Kimi-Linear-48B** already downloaded (resumable) — the small-model validation target.

## The only remaining cost: the download

**553.2 GiB, resumable.** Observed throughput during the aborted Kimi-Linear fetch was roughly
**8–13 MB/s** (D, measured over a few minutes, not a careful benchmark) ⇒ **~16–31 hours** of
wall-clock downloading.

**This is Atur's call, not mine.** He stopped the previous download because it destroyed his
gaming ping, and a 553 GiB pull is 180× larger than the 3 GB that caused the complaint. It must be
scheduled deliberately (overnight, rate-limited, or both) and **must not be started without his
explicit go-ahead.** A `curl --limit-rate` cap is the obvious courtesy — halving the rate roughly
doubles the wall-clock but leaves the line usable.

## Recommended order of operations

1. **Confirm the toolchain end-to-end on the small model first.** Finish the Kimi-Linear-48B
   download (91.5 GiB, already 3 GB in) and run it under WASTE. This validates streaming inference
   on this exact hardware for ~1/6th the bytes of the K3 pull, and produces the project's first
   real measured tok/s. *A test on the small model does not test the big one — but it does catch
   every toolchain failure before a 553 GiB commitment.*
2. **Measure the real link speed** deliberately (a timed 1 GiB pull) so the K3 download window can
   be planned rather than guessed.
3. **Pull UD-IQ1_S rate-limited**, overnight, resumable.
4. **Run with attention on GPU, experts on CPU/disk**; measure honestly; publish with the caveats
   in full view.

## What would make this a real contribution rather than a stunt

Running it is the headline. The *durable* results are:
- The first published measurement of a 2.8T model on a **16 GB consumer laptop**, with a full
  method and honest tok/s — nobody has done this (see `k3-on-16gb-feasibility.md`).
- A measured comparison of **mmap page-fault streaming (llama.cpp) vs deliberate large-block
  streaming (WASTE)** on identical hardware and model class. This is the single most useful number
  the project could produce for the field, and having both engines built on one machine makes it
  cheap to get.
- The bytes/token prediction model validated (or corrected) against reality — which is exactly the
  `cross-engine-advisor` product thesis.

## Open questions

- Whether llama.cpp's mmap path degrades gracefully or thrashes pathologically at a 35:1
  model:RAM ratio. **Unknown, and it is the single biggest risk to this plan.** The Kimi-Linear
  step in (1) partially de-risks it; nothing fully de-risks it but the attempt.
- Whether UD-IQ1_S's ~1-bit experts preserve enough quality for the output to be worth showing.
  unsloth's own KLD table is encouraging but is not a substitute for running it.
- Whether Windows' page cache handles a 553 GiB mapping sanely; a WSL2/Linux comparison may be
  needed, and WSL2 is not currently installed.
- Whether the 6 GB VRAM offload of attention actually pays on this fork — untested.
