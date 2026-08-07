# Everything left, in the order the measurements justify

Written 2026-08-07, immediately after the V4-Flash retraction. Nothing here is a
plan built on a claim; every number is measured and cited.

## First: is 20 tok/s for V4-Flash possible on this laptop?

**No. Not by streaming, and not by any factor of engineering.** The arithmetic is
short enough to check in one line.

```
20 tok/s                        = 0.050 s per token
V4-Flash reads per token        = 3.21 GiB of routed experts (6 of 256 experts)
required disk bandwidth         = 3.21 / 0.050 = 64 GiB/s
this machine's NVMe, measured   = 2.37 GiB/s
                                  ------
                                  27x short
```

A faster drive does not rescue it. A top-end Gen5 NVMe is ~14 GB/s, still **5x
short**, and four of them striped is still short. **The bytes have to not be
read at all**, which means they have to be in RAM — and they are 137 GiB against
this machine's 15.7 GiB.

Stacking every byte-reduction idea this project has considered, optimistically
and all at once:

| lever | factor | status |
|---|---:|---|
| 2.5-bit routed experts | 1.7x | never shown on an MoE this size; scalar 2-bit collapses |
| expert cache across tokens | ~2x | bounded by RAM: ~8 GiB of 137 GiB is 6% of the model |
| speculative decoding | ~2.2x | proven technique, needs a draft model sharing the tokenizer |
| contextual sparsity | ~1.1x | **measured dead**: V4-Flash experts are 9.1% negligible, not 80% |
| | **~8x** | |

8x against a 27x requirement. **20 tok/s on this hardware is not a hard target,
it is an impossible one**, and every hour spent on it is an hour not spent on the
gap that is real.

Where 20 tok/s *is* reachable: a machine whose RAM holds the model. Same binary,
different hardware. That is not a consolation prize — "one engine that runs the
same model well on a workstation and acceptably on a laptop, and tells you which
you have" is a product. "20 tok/s on a laptop for a 144 GB model" is not a
product, it is a violation of the machine's memory bandwidth.

## What *is* achievable: beating llama.cpp

Measured back to back today, we lose: prefill 1.62x, generation 3-4x. But the
reason is **one architectural difference, not a hundred small ones**.

Both engines read the same ~3.2 GiB per token from the same drive. llama.cpp
`mmap`s the container and the kernel reads ahead **while the CPU computes the
previous layer**. Bigtea reads a layer's experts, waits, computes, reads the
next. Per token, measured on our side:

```
I/O      2.3s        run strictly one after the other
compute  1.0s        total 3.3s

overlapped:  max(2.3, 1.0) = 2.3s        -> 30% faster, and it is the whole gap
```

**That is the work. It is bounded, it is not research, and it is the only thing
standing between Bigtea and parity.**

---

## T1 — Overlap I/O with compute  *(the whole gap)*

- **T1.1** Prefetch layer L+1's expert slices on a background thread while layer
  L computes. Routing for L+1 is not known until L finishes — **except on layers
  0-2, which route by token id via `ffn_gate_tid2eid` and are knowable before any
  compute runs.** Start there: it is 3 of 43 layers with zero speculation risk,
  and it proves the machinery.
- **T1.2** Within a block, overlap the three expert tensors: `gate` and `up` are
  needed before `down`'s matmul, so `down` can stream while the first two compute.
  No routing prediction needed at all. Worth ~1/3 of the read.
- **T1.3** For layers 3-42, prefetch *speculatively* on the previous token's
  routing. Adjacent tokens route to overlapping experts; a miss costs a wasted
  read, not a wrong answer. **Measure the overlap rate first** — it is also the
  input to any expert-cache decision, and nobody has measured it on this model.

## T2 — The KV cache

Generation currently re-runs the whole sequence per token, so 0.064 tok/s is an
artefact rather than a measure of the engine. **A single-token pass costs 4.0s**;
that is what a cached step will cost.

- **T2.1** Raw window: bounded at `sliding_window` = 128 positions.
- **T2.2** Compressed summaries: bounded at 256 blocks per layer.
- **T2.3** All 43 layers together are ~33 MB of state — small. The risk is
  correctness, not memory: **a wrong cache on this architecture yields fluent
  nonsense, never an error.** It needs its own oracle capture at two consecutive
  positions before it is trusted.

## T3 — Fit the always-read set

7.38 GiB, and it fits only when ~10.5 GiB is free. Worth 0.7s/token when it does
not.

- **T3.1** Bigtea already reports which processes to close and what the shortfall
  costs per token. Verify that advice on Linux and macOS.
- **T3.2** Offer a smaller quant when it cannot fit, rather than running slowly
  and silently.

## T4 — Reduce bytes per token  *(the only lever with real headroom)*

Ordered by evidence, not by appeal.

- **T4.1 Measure routing overlap between adjacent generated tokens.** One
  afternoon, no new machinery. It decides whether an expert cache is worth
  anything on this model *and* whether T1.3's speculative prefetch will hit.
  **Nothing else in this section should start before it.**
- **T4.2** Speculative decoding. Proven, independent, ~2.2x, needs a draft model
  sharing V4-Flash's tokenizer.
- **T4.3** Sub-4-bit routed experts. 1.7x, and the one item with no fallback if
  quality does not hold.
- **Not doing: contextual sparsity.** Measured: only 9.1% of V4-Flash's expert
  neurons are negligible, against the 80-95% the literature reports for dense
  FFNs. **The router's 6-of-256 is already this architecture's contextual
  sparsity**; harvesting it twice was the mistake.

## T5 — The product

Unchanged from `lts-0-0-0.md`, and none of it blocks on performance.

- `bigtea pull <model>` from Hugging Face — resume, checksums, disk-space check
  before starting a 144 GB download
- Quant selection from the probe, with the tok/s prediction stated *before*
  downloading
- Self-configuration: prefill block, cache budget, threads, I/O mode
- OpenAI-compatible `/v1/chat/completions` — the single item that makes it usable
  from a coding agent
- Prebuilt binaries so nobody needs Rust or ggml

## What to do first

**T4.1**, because it is a day's work and it gates two other items. Then **T1.2**,
because it needs no prediction at all. Then **T1.1**, then **T2**.

Do not start T4.3 or anything in T5 until T1 has been measured.
