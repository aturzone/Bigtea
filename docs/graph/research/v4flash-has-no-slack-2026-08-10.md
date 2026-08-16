---
topic: Three independent attempts to find redundancy in V4-Flash, all measured, all negative — the expert bank is full-rank, the router's tail is not small, and speculative decoding is worth far less here than the literature says
status: measured, closed
links: [v4flash-28-tokens-per-second.md, routing-skew-is-per-prompt-2026-08-08.md, ../backlog/the-big-bang.md]
---

Generation on V4-Flash reads **3.21 GiB per token**, and that single number is
the reason it is slow. This node asked, three different ways, whether those bytes
carry 3.21 GiB of information or whether some of it is slack that a cleverer
runner could refuse to read.

**All three said the bytes are real.** Two were measured here for the first time;
the third is a correction to a figure this project had been carrying from the
literature without checking whether it transfers.

Taken together with the earlier finding that the experts are only 9.1%
internally negligible, that is four independent probes and four negatives, which
is enough to state something about the model rather than about the runner:

> **DeepSeek trained V4-Flash with no redundancy left to harvest. Its experts are
> mutually distinct, internally dense, and its router spreads real weight across
> all six selections. The 6-of-256 routing is the whole of this architecture's
> sparsity, and Chaos already exploits it.**

That is a genuinely useful result — negative results of this shape save the next
person months — and it closes the two remaining ideas that could have delivered a
step change.

## 1. The expert bank does not factor — 1.2x from noise

### The idea

Every previous byte-cutting attempt attacked *which* experts to read, or *which
rows inside* an expert. This attacked neither: it asked whether all 256 experts
in a layer are **built from the same few hundred directions**.

If they were, the bank factors as `W_i ≈ C_i Bᵀ` — one shared basis `B`
(`4096 × r`) resident for the whole layer, only the small `C_i` streamed. Worth
`4096/r` on bytes, and unusually the arithmetic improves too: `W_i x = C_i(Bᵀx)`,
so `Bᵀx` is computed **once per layer** and reused by all six selected experts,
while the factored apply is cheaper than the dense one. Fewer bytes *and* fewer
flops, at every batch size, cache state and RAM budget. No other lever on the
list has that property.

It was also unexplored. The published MoE decomposition work (D2-MoE, MC-SMoE)
targets *memory capacity* on a GPU. Nobody had asked it of a disk-streaming
runner, where the same structure would buy bandwidth instead.

### The measurement

`chaos-spectrum`, new — no forward pass, no tokenizer, no GPU, only the weights.
It accumulates `G = Σ_i W_iᵀW_i` one expert at a time (the full bank dequantised
is 8.6 GB and would not fit; one expert is 33 MB, so memory is flat in the sample
size), recovers the top eigenspace by randomised subspace iteration, and reports
the share of `‖W‖_F²` each rank holds — beside matched-shape noise run through
the identical pipeline.

```
blk.20.ffn_up_exps.weight, 32 experts        blk.3 up        blk.20 down
RANK    ENERGY   CONTROL   BYTES/x       ENERGY  CONTROL    ENERGY  CONTROL
  64      3.4%      2.3%     64.0x         2.8%     2.1%      4.2%     3.5%
 128      6.3%      4.5%     32.0x         5.2%     4.2%      7.9%     7.0%
 256     11.4%      8.7%     16.0x         9.4%     7.9%     14.6%    13.4%
 512     20.4%     16.6%      8.0x
1024     35.1%     30.4%      4.0x
```

**The real bank is 1.23x more structured than random noise on `up`, 1.19x on a
different layer, and 1.09x on `down`.** A rank-512 basis — 8x fewer bytes — holds
20.4% where noise holds 16.6%. Keeping 90% needs a rank far past 1024, where the
"compressed" form is larger than what it replaces.

### Why the negative is trustworthy

- **The control is the result.** 512 of 4096 dimensions holds ~12.5% of *any*
  matrix by construction. Without the null, "rank 1024 holds 35%" reads as a
  finding; against 30.4% it is nothing. This project published an in-sample hot
  set as a headline once; the control now runs automatically and prints in the
  adjacent column.
- **Converged.** 10 power iterations gives 11.5% at rank 256 where 3 gave 11.4%.
  The flat spectrum is the data, not slow convergence.
- **Conservative by construction.** Subspace iteration returns a *lower* bound on
  the optimal rank-`r` subspace, so the method can only understate
  compressibility. A negative cannot be an artefact of it.
- Unit tests pin both directions: a synthetic rank-8 bank is reported as rank 8
  (`>99.9%` at `r=8`), Gaussian noise as full-rank.

## 2. The router's tail is not small — dropping half the experts costs 31%

The standing assumption, written into `the-big-bang.md` as Tier 3: *"The top-1
expert carries most of the weight mass; the 6th contributes little."* If true,
reading three experts instead of six is a free 2x.

Measured (`CHAOS_ROUTING_WEIGHTS=1`, renormalised weights, sorted descending,
mean over 43 layers × 15 tokens, captured with `-n 1` so regeneration cannot
double-count):

| keep | weight | cumulative | GiB/token | bytes |
|---:|---:|---:|---:|---:|
| 1 | 33.5% | 33.5% | 0.54 | 6.00x |
| 2 | 20.6% | 54.1% | 1.07 | 3.00x |
| 3 | 15.0% | 69.1% | 1.60 | 2.00x |
| 4 | 12.1% | 81.1% | 2.14 | 1.50x |
| 5 | 10.1% | 91.2% | 2.68 | 1.20x |
| 6 | 8.8% | 100.0% | 3.21 | 1.00x |

**The assumption is false.** A uniform router would give 16.7% each; this one
gives 33.5% to the top and still **8.8% to the sixth**. The profile is skewed by
only 3.8x end to end, and the six are close enough to equal that the cheapest
possible saving — dropping the single weakest expert, worth 1.2x on bytes —
already discards 8.8% of the router's weight. The 2x costs 31%.

Caveat, stated because it cuts the honest way: weight is not contribution.
A dropped expert's effect on the output is its weight times its output, and the
outputs are not equal-magnitude. This **bounds** the idea rather than deciding
it; a perplexity run could still find the tail cheap. But an idea that starts by
discarding 31% of the routing mass to buy 2x is not the free win it was filed as,
and it is now ranked accordingly.

## 3. Speculative decoding is ~1.4x here, not 2.2x

Not a new measurement — a correction, from numbers this project already had and
had not connected.

Two docs carry speculative decoding at **~2.2x**, cited from the literature and
tagged "proven, independent, needs no research". That figure does not transfer,
and the reason is the dedup curve already measured here:

```
distinct experts per layer per pass:  1 token 6.0   |  17 tokens 39.7  |  166 tokens 122.8
fitted over the range a draft lives in:  U(n) ≈ 6·n^0.667
```

In a GPU runner the verify pass costs the **same** as a single-token pass — the
weights are already resident — so throughput is simply the expected accept
length and any acceptance rate wins. **Here the verify pass costs more bytes than
a single-token pass, because more tokens select more distinct experts.**
Byte-speedup is `E[accepted] / (k+1)^0.667`:

| draft `k` | α=0.9 | α=0.8 | α=0.7 | α=0.6 |
|---:|---:|---:|---:|---:|
| 1 | 1.20x | 1.13x | 1.07x | 1.01x |
| 3 | 1.37x | 1.17x | 1.01x | 0.94x |
| 7 | **1.42x** | 1.04x | 0.83x | 0.71x |
| 15 | 1.28x | 0.79x | 0.60x | 0.49x |

**Below roughly α = 0.75 speculative decoding is a net loss on bytes**, and the
optimum draft is short — the opposite of the usual advice, where longer drafts
are strictly better until acceptance collapses. The best cell is 1.42x, and
wall-clock is worse because verify compute scales with the batch while nothing in
the I/O saving compensates.

Still worth doing eventually. **Not** the best lever on the list, not free, and
it needs a draft model sharing V4-Flash's tokenizer, which does not exist.

## Where the byte budget stands

20 tok/s at the measured 1.58 GiB/s direct-read rate allows **79 MB per token**.
V4-Flash reads 3288 MB. That is **42x**.

| lever | worth | status |
|---|---:|---|
| expert-bank factorisation | 1.0x | **dead — §1, this node** |
| drop the router's tail | ~1.2x | **costs 8.8% of routing mass — §2** |
| contextual sparsity | 1.1x | dead — 9.1% negligible |
| pinned hot set | 1.0x | dead — R0, 37.5% vs 25.0% random |
| speculative decoding | 1.4x | real, **overstated 1.6x by the docs — §3** |
| 4.25 → 2.5-bit experts | 1.7x | unproven on an MoE this size, quality-risky |
| warmed expert cache | 1.3x | measured at 23.5% hits with ~6 GiB spare |

Everything alive, multiplied, is **3.1x** against a 42x gap. The two ideas that
could have closed it are the two now measured and failed.

**3.21 GiB per token is not an implementation artefact. It is what this model
costs.** Reaching 20 tok/s does not need a better runner; it needs the active
weights to stop coming from disk, which is a statement about the model-to-RAM
ratio and not about the code.

## The question this makes answerable, and worth answering

If bytes-per-token is fixed, then tok/s is a function of **how many of those
bytes are already in memory** — and that is a curve nobody has published for a
model this size:

> **What is the tok/s-versus-RAM frontier for a 144 GB model?**

Chaos is the only engine that can produce it. `llama.cpp` `mmap`s the container
and hands residency to the kernel's LRU; it cannot be told to use exactly N GiB,
so it cannot be swept. Chaos owns residency by construction, and the sweep is
`--budget` in a loop.

The same curve answers the product question directly and honestly: *given your
machine, here is the largest model that runs at the speed you want* — which is
`chaos-model-info`'s job, and currently it predicts from a model rather than
from measurement.

## What was NOT tested

- The `gate` projection, and layers other than 3 and 20. The spectrum was flat on
  every combination tried, and a layer that factored while its neighbours did not
  would not help: the win must hold per layer to reduce a token's bytes.
- **Per-expert** low rank (each `W_i` alone, no shared basis). It saves nothing —
  an expert is read only when selected, so a private basis is read on exactly the
  same schedule as the weights it replaces, and `r(4096+2048) < 4096·2048` needs
  `r < 1365`, where this spectrum has already lost the quality.
- Whether energy loss tracks quality loss. Moot: no rank leaves meaningful energy.
- Whether the router's tail is cheap in **perplexity** rather than in weight.
  That is the one live thread out of §2 and it needs an evaluation harness this
  project does not have.
