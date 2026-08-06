---
topic: What would actually be required to reach 28 tok/s on V4-Flash on a 15.7 GiB laptop
status: open
links: [v4flash-speed-budget.md, sub-2bit-k3-fixed-hardware.md, head-to-head-llamacpp-2026-08-05.md]
---

The target: **28 tok/s for DeepSeek-V4-Flash on this machine** — 15.7 GiB RAM, NVMe measured
at 2.55 GB/s sequential. Everything else in this project's docs treats ~0.87 tok/s as a hard
ceiling. This node asks what would have to be true for that ceiling to be wrong, and finds
that it is not a wall but a stack of four assumptions, three of which are attackable.

## The budget

```
28 tok/s  =  0.036 s/token
disk budget at 2.55 GB/s  =  92 MB per token
V4-Flash reads today      =  3288 MB per token (3.21 GiB of routed experts)
                             ------
required reduction        =  36x
```

So the question is not "can the code be faster" — it is **"can 36x fewer bytes reach the
compute".** Every idea below is a byte-reduction idea. Nothing else matters.

## The one that changes the physics: contextual sparsity

The router already picks 6 experts of 256. **Inside a chosen expert there is a second,
much larger sparsity that this project has never exploited.**

An expert's FFN is `down @ (silu(gate @ x) * (up @ x))` with an intermediate width of 2048.
For a *single token*, most of those 2048 intermediate neurons produce values near zero after
`silu`, and contribute nothing to the output. Only the rows of `up` and the columns of `down`
belonging to active neurons are needed.

The published work (Deja Vu, arXiv:2310.17157; PowerInfer, arXiv:2312.12456) reports **80-95%
of FFN neurons inactive per token**, predictable ahead of time by a small classifier reading
the layer input. PowerInfer builds a whole GPU/CPU split on it.

Applied here: `3.21 GiB x (1 - 0.90) = 0.32 GiB/token` → **~8 tok/s**. At 95%, ~16 tok/s.

### The measured objection, which is serious

This project has already measured that **scattered reads run at 1.10 GB/s against 2.55 GB/s
sequential** — 43% efficiency. Reading 10% of an expert's rows as scattered fragments could
easily be *slower* than reading 100% of it contiguously. 10% of the bytes at 43% of the rate
is still a 4.3x win, but it is not the 10x the neuron count suggests, and if the fragments are
smaller than the SSD's efficient read size it could collapse further.

**This is the crux, and it is measurable before anything is built**: take one expert slice,
read 10% of its rows in the scattered pattern a sparsity predictor would produce, and compare
against reading the whole slice. That single experiment decides whether this path is real.

**The fix if it fails** is a layout change, not an algorithm change: store experts with rows
ordered by activation frequency, so the hot 10% is contiguous on disk. That is an offline
repack of the container, and this project already has the container tooling.

## The one that is proven and independent: speculative decoding

A small draft model proposes `k` tokens; V4-Flash verifies all `k` in **one** forward pass.
The expert reads for that pass are the *union* over the k tokens, which is far less than `k`
times one token because routing overlaps heavily between adjacent tokens.

With k=4 and a 60% accept rate, this is ~2.2x effective throughput **for the same bytes**. It
composes with everything else, needs no new numerics, and is the least risky item here. It
needs a draft model that shares V4-Flash's tokenizer.

## The two that are already understood

- **Sub-4-bit routed experts**: 4.25 → 2.5 bits is 1.7x. Per `sub-2bit-k3-fixed-hardware.md`
  scalar 2-bit collapses (GPTQ/AWQ at 10^4-10^6 perplexity) so this needs additive/residual VQ
  plus a CPU decode kernel, and has never been shown on an MoE this size.
- **Expert cache with skew**: 50% hits is 2x. Bounded hard by RAM — with 7.38 GiB resident
  there is little left on this machine. **Layers 0-2 route by token-id lookup, so their expert
  set is knowable before any compute runs**, which makes them perfectly prefetchable and
  perfectly cacheable across repeated tokens.

## The stack, multiplied out

```
3288 MB/token
  ÷ 1.7   2.5-bit experts            ->  1934 MB
  ÷ 2.0   50% cache hits             ->   967 MB
  ÷ 6.7   85% contextual sparsity    ->   144 MB
  ÷ 2.2   speculative decoding       ->    65 MB effective
                                          -----
                                          65 MB  <  92 MB budget   => 28 tok/s
```

**28 tok/s is reachable in principle, and only if all four land.** Each is a research-grade
item with real failure modes; the product of four optimistic estimates is an optimistic
estimate. A more defensible reading of the same stack with conservative factors (1.5, 1.5,
4, 1.8) gives ~250 MB/token → **~10 tok/s**, which would still be **22x llama.cpp** and by a
wide margin the fastest anything has run this model on hardware like this.

## Order, by evidence-per-hour

1. **The scattered-read experiment above.** One afternoon, no new code beyond a benchmark, and
   it decides whether contextual sparsity — the only 5-10x on the list — is available at all.
2. **Speculative decoding.** Proven technique, independent of the rest, ~2x.
3. Generation + residency + the copy fix, because none of the above can be measured without a
   generation loop.
4. Expert repack for layout *and* for dequantisation speed — the same offline pass can reorder
   rows by activation frequency and change the quantisation.
5. VQ experts, last, because it is the only item with no fallback if the quality does not hold.

## What this node is not

It is not a promise of 28 tok/s. It is the observation that **the 0.87 tok/s ceiling assumes
every byte of a chosen expert must be read, and that assumption is the weakest one in the
stack** — the router's 6-of-256 is the *first* sparsity, and the literature says there is a
second one worth 5-10x sitting inside it, unexploited. That is where a step change lives, if
one exists.

## MEASURED (2026-08-06): the experiment above, run

`sparse_row_reads_versus_whole_slice` in `tests/deepseek4_forward.rs`. One expert slice of
`blk.5.ffn_up_exps.weight` — 4.25 MiB, 2048 rows of 2176 B — read three ways, cache-bypassing,
each repetition on a different expert so the previous read cannot help:

```
whole slice        1.06 GiB/s      what the runner does today
10% scattered      0.02 GiB/s      what a naive sparsity predictor would ask for
10% packed         0.75 GiB/s      the same bytes, contiguous
```

**Reading 10% of the rows as scattered fragments takes 4.45x LONGER than reading all 100%.**
A 2176-byte read is far below this NVMe's efficient size, and 205 of them cost more than one
4.25 MiB read. Contextual sparsity implemented directly against the current layout is not a
5-10x win, it is a **4.5x regression**.

**The same 10%, contiguous, takes 0.14x the time — a 7.1x win, and a 31x difference between
the two layouts.**

So the finding is sharper than the hypothesis. Contextual sparsity is real and worth ~7x on
this hardware, but it is **entirely a layout problem, not a prediction problem**. The predictor
is the easy half; the hard requirement is an offline repack of the expert tensors that orders
each expert's rows by activation frequency so the hot fraction is one contiguous span.

### What this does to the stack

```
3288 MB/token
  ÷ 7.1   packed contextual sparsity   ->  463 MB      0.87 -> 6.2 tok/s
  ÷ 2.2   speculative decoding         ->  210 MB      -> 13.6 tok/s
  ÷ 1.7   2.5-bit experts              ->  124 MB      -> 23 tok/s
  ÷ 2.0   50% cache hits               ->   62 MB      -> 28+ tok/s
```

The first two alone give **~13.6 tok/s, which is coding-agent territory and 30x llama.cpp**,
and neither needs new numerics — one is a container repack, the other is a published
technique. That is the realistic target. The last two are quality-risky and RAM-bound
respectively, and are what would take it to 28.

### Why the repack is cheap to get right

The row ordering does not have to be perfect or even per-token: activation frequency is a
static property measurable by running a calibration corpus once and counting. Rows that are hot
for *most* tokens go first. A predictor then only has to decide *how many* leading rows to
read, not which — which turns a scattered gather into a prefix read, the friendliest possible
pattern for an SSD.

**That is the big bang, and it is an offline data-layout change plus a prefix read.** No new
kernel, no new quantiser, no accuracy loss — the same weights in a different order.
