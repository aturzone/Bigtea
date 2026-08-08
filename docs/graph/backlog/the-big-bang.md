---
topic: The big bang — what "the best model runner for anyone" has to mean, which bets are backed by measurement, and the order to take them
status: proposed, awaiting Atur
links: [next-session-handoff.md, ../research/routing-skew-is-per-prompt-2026-08-08.md, ../research/v4flash-vs-llamacpp-2026-08-07.md, lts-0-0-0.md]
---

A brainstorm, written to be argued with. Every claim is tagged **measured**,
**arithmetic**, or **unproven**, because this project has been wrong six times
by reasoning ahead of measurement and right every time it measured first.

## First, what we cannot claim

Three appealing pitches are already dead, and pretending otherwise costs
credibility we have spent twice.

- ~~"Runs models larger than RAM."~~ **measured** — llama.cpp runs the 144 GB
  V4-Flash with `--no-repack`. Not a differentiator.
- ~~"Faster than llama.cpp."~~ **measured** — on V4-Flash we lose prefill 1.62x
  and generation 3-4x. We lead on Qwen3 prefill only.
- ~~"20 tok/s for a 144 GB model on a laptop."~~ **arithmetic** — needs a 96.3%
  cache hit rate; the best measured static figure is 76.7% at 68.5 GiB.

Feature parity with llama.cpp is also not the path. It is years of work across
every architecture and every backend, contributed by hundreds of people. We
would arrive late at someone else's finish line.

## The claim that is actually ours

> **Tell it your machine, and it tells you what you can run and how fast —
> before you download 144 GB — then runs it that way.**

Two halves, and both are things no other runner does.

**Owned residency.** Bigtea decides what stays in memory. llama.cpp `mmap`s the
container and hands the policy to the kernel's LRU. That is not a small
difference on this workload:

- Expert access is a **cyclic scan**, so recency is the worst possible policy —
  layer 0 is always the oldest entry when layer 47 needs room. Frequency-gated
  admission took hit rate **17% → 70%** at the same budget. **measured**
- A cache that does not own its memory gets paged out, and then a "hit" is a
  page fault in disguise. Past ~6 GiB on Qwen3 a 71%-hit cache was the *slowest*
  configuration measured. **measured**

An mmap-based engine cannot fix either without ceasing to be mmap-based. **This
is architectural, and it is the whole reason this project exists.**

**Honest prediction.** `bigtea-model-info` already predicts fit and tok/s from
the probe. Nothing else tells you *before* a 144 GB download that the answer
will take ten minutes a token on your machine. That is worth more to a user than
another 10% of throughput, and it is the half nobody is competing for.

## What "for anyone" requires

Speed work is worth nothing if the thing cannot be installed. All of this is
`lts-0-0-0.md` T1-T5 and none of it blocks on performance:

1. **Prebuilt binaries.** Today it needs the GNU Rust toolchain, MSYS2, and a
   hand-built ggml. That is a wall, not an install.
   **The Windows half of this is now unblocked** (2026-08-08): the GNU C++ and
   OpenMP runtimes are linked statically, so the binary depends only on system
   DLLs. Before that a downloaded `.exe` died with `0xC0000135` before `main`,
   printing nothing — a release workflow would have shipped binaries that simply
   did not start. What remains is the CI release job itself.
2. **`bigtea pull`** with resume, checksums, and a disk-space check *before*
   starting 144 GB.
3. **Quant selection from the probe**, with the tok/s prediction stated first.
4. **Self-configuration** — cache size, prefill block, threads, I/O mode, all
   from the probe rather than from flags nobody knows to pass.
5. **OpenAI-compatible `/v1/chat/completions`.** The single item that makes it
   usable from a coding agent, which is the actual use.

**This is the "for anyone" half and it is currently 0% done.** A runner that is
2x faster and unusable loses to one that is 2x slower and works.

## The performance bets, ranked by evidence

### Tier 1 — measured, bounded, do these

| bet | worth | status |
|---|---|---|
| **KV cache** (R3) | generation currently re-runs the whole sequence per token, so 0.064 tok/s is an artefact. A single-token pass costs **4.0s** — that is what a cached step costs. **measured** | ready; needs an oracle at two consecutive positions, since a wrong cache gives fluent nonsense |
| **Overlap I/O with compute** (R2) | 2.3s I/O + 1.0s compute, strictly serial. Overlapped: `max(2.3, 1.0)`. **measured** | ready; start with layers 0-2, which route by token id and are knowable before any compute runs |
| **Adaptive expert cache** (R1) | a set warmed on the prompt covers **86%** of what generation needs. **measured, R0.1** | **built 2026-08-08** and verified against the oracle — but inert until R3 shrinks a step's working set |

**These three are the whole gap to llama.cpp on V4-Flash.** Nothing below matters
until they are done.

**And they are ordered, not parallel.** Expert reads are deduplicated per block
over the whole batch, so a pass reads the *distinct* experts its tokens select —
a 166-token prefill touches **122.8 distinct experts per layer, ~66 GiB in one
pass** (measured, not estimated). A few GiB of cache covers ~2% of that and can
do nothing. Only once a step needs **6 experts per layer — 3.2 GiB** — is the
working set small enough to cache, and that is exactly what the KV cache buys.

```
KV cache (R3)  ->  a step touches 6 experts/layer (3.2 GiB), not 122.8 (66 GiB)
   then R1     ->  a few GiB now covers a real share of that
   then R2     ->  overlap what reads remain
```

R0.1's 86.3% is coverage of *selections*, which is the right metric for a
single-token step and the wrong one for today's stateless re-prefill. **Building
R1 before R3 measures nothing**, which is why the R1 benchmark reports short and
long prompts separately rather than one number.

### Tier 2 — real, unproven here

- **Heterogeneous execution.** Not "VRAM as a cache" — that only trades NVMe for
  PCIe. The prize is running the *hot* experts' matmuls on the GPU where their
  weights already live, and streaming the cold ones to the CPU. One expert index
  across all 43 layers is 0.535 GiB, so ~5.1 GiB of VRAM holds 9-10 of them.
  **Three blockers**: no CUDA toolkit here, the linked ggml is CPU-only, and
  Bigtea binds weights by handing ggml a *host* pointer — a device tier needs a
  new binding path. **unproven, weeks of work.**
- **Speculative decoding.** ~2.2x, a proven technique, independent of everything
  above. Needs a draft model sharing V4-Flash's tokenizer. **unproven here.**

### Tier 3 — needs a quality measurement, not new theory

- **Sub-4-bit routed experts** (~1.7x on bytes). The one item with no fallback if
  quality does not hold.
- **Two-tier precision** — hot experts resident at 2-bit as a predictor, full
  precision fetched only when the router's weight is high. The top-1 expert
  carries most of the weight mass; the 6th contributes little.

### Dead — do not revive without new evidence

- **Contextual sparsity.** V4-Flash's experts are 9.1% negligible, not the 80-95%
  the literature reports for dense FFNs. The router's 6-of-256 *is* this
  architecture's contextual sparsity; harvesting it twice was the mistake.
  **measured.**
- **Pruning the model to a global hot set.** Loses 46% of routing on an unseen
  prompt, not the 2.2% claimed. **measured, R0.**
- **Any pinned hot set**, in RAM or VRAM. Across subjects it scores 37.5% against
  25.0% for caching at random. **measured, R0.**

## Where this actually lands

Honest arithmetic, not a target:

- **On this 15.7 GiB laptop**: Tier 1 reaches roughly parity with llama.cpp on
  V4-Flash. That is worth having — it makes the residency argument on equal
  footing — but it is not a headline.
- **On a 48-80 GiB desktop**: the hot set fits, and an adaptive cache that owns
  its memory is a policy an mmap engine cannot copy. **This is where the claim
  lives.** It is unmeasured, because nobody has run it on such a machine.
- **On a small machine**: the win is not speed, it is *knowing*. "This model will
  give you 0.3 tok/s here; this quant gives you 4; here is what to close."

**The strategic implication is uncomfortable and worth saying: the performance
claim needs hardware this project does not have.** Either R6 lands on a borrowed
desktop, or the headline stays "honest prediction + owned residency" rather than
"fastest".

## The demo that would land

One command, on a normal machine, that:

1. probes the hardware and says *"V4-Flash Q4 will fit and give you ~N tok/s;
   here is what to close, and here is the quant I recommend instead"* —
   **before** downloading anything,
2. pulls it with resume and checksums,
3. serves `/v1/chat/completions`,
4. and answers a real coding question from inside an editor.

No other runner does step 1 at all. Steps 2-4 are table stakes we do not yet
meet. **That gap — not tok/s — is what stands between this and being usable by
anyone**, and it is the cheapest work on this page.

## Suggested order

1. **R0.1 → R1 → R3 → R2.** Finish the engine argument on V4-Flash.
2. **T5 product work in parallel** — it blocks on nothing and it is what makes
   the engine reachable. Prebuilt binaries first; they cost the least and remove
   the largest wall.
3. **Then** heterogeneous execution or a borrowed desktop, whichever is available.

The open question for Atur: **is the goal to win a benchmark, or to be the thing
someone installs?** They are different projects, and the second one is closer,
cheaper, and unclaimed.
