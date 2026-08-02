---
topic: Design space for full K3 on FIXED hardware (745.9 GiB disk, 15.7 GiB RAM, 3.09 GB/s) — engineering synthesis, not literature survey
status: open
links: [k3-on-16gb-feasibility.md, waste-engine-verified.md, sub-2bit-k3-fixed-hardware.md]
---

Written 2026-08-03 as design reasoning by Claude at Atur's direction ("you can't reach the goal
with research alone"). **This node is derivation and hypothesis, not measurement.** Every number
is labelled M (measured, sourced), P (published elsewhere), or D (derived here). Nothing here is
validated. It exists to define what to test, in what order.

## The constraint inversion — the central insight

WASTE optimized against **abundant disk, scarce RAM** (64 GB Mac, big SSD). Our target inverts it:
**disk is the binding wall** (745.9 GiB, M) while RAM, though small, is not the thing that fails
first. Several of WASTE's *measured, correct* refutations were correct **only under their cost
model** and must be re-opened under ours:

| WASTE refutation | Why they killed it | Does it transfer to us? |
|---|---|---|
| Per-expert bit allocation (GEMQ) | Error-sensitivity flat (1.01× across layers, P) **and** saves ~0% of reads | **NO — re-open.** They judged it on *reads*. It saves **disk**, our binding constraint. |
| Routing-frequency expert split | "saves disk and ~0% of the reads" — dismissed because disk was free | **NO — re-open.** That exact sentence describes a *win* for us. |
| Shared low-rank basis (KBVQ) | 0.12 bits for 0.3pp; lost at equal budget ~4 bits | **Partially.** Cross-expert redundancy matters more when bit-starved at ~2.1 bits than at 4. Re-test at our operating point, not theirs. |
| 1-bit SUB1 substitute bank | Specified in their format, **never implemented** | **Untested, not refuted.** A 1-bit fallback for cold experts is a pure disk win. |
| 2-bit experts | 34%/71.8% error — but tested with **naive RTN only** | **NO — re-open.** RTN is the weakest possible 2-bit method. Refutation binds the mechanism tested. |
| Batching / spec-decode | 1.63× ceiling, doesn't compose with read-ahead | **Yes, transfers.** Still a dead end for single-stream latency. |
| Cross-layer prefetch | Refuted then **revived and shipped** (59.0% recall, 1.17×) | Already known live — see waste-engine-verified.md. |

**This is the single most important framing in the project**: we are not trying to out-engineer
WASTE at their problem. We are solving a *different* problem whose cost model makes their discards
valuable.

## The two thresholds that define the design

**Threshold 1 — disk fit.** Container ≤ ~700 GiB (leave ~45 GiB of 745.9 for OS/temp).
K3 = 2.78T params. 700 GiB ⇒ **2.16 bits/param average** (D). WASTE ships 3.04 (D, from 982 GiB).
Experts are ~955 GiB of the 982 (trunk is only 27.28, P) ⇒ **experts must reach ~2.12 bits** (D).

**Threshold 2 — trunk residency, and this is where the leverage is.**
Usable RAM for weights ≈ 15.7 − ~2.85 (OS/KV/scratch/pipeline buffers, P) = **~12.8 GiB** (D).
Trunk is 27.28 GiB at int8 (P). Trunk fits entirely in RAM at **≤3.75 bits/param** (D).

Why this matters more than anything else — bytes/token at 3.09 GB/s (M):

| Regime | bytes/token | s/token (ceiling) | tok/s realistic (~50% eff.) |
|---|---|---|---|
| Partial trunk residency (current plan) | ~31.5 GB (D) | 10.2 | **~0.05** |
| **Trunk fully resident, experts @3.01b** | ~17.4 GB (P) | 5.6 | **~0.09** |
| **Trunk fully resident, experts @2.12b** | ~12.3 GB (D) | 4.0 | **~0.13** |

**Full trunk residency is worth ~2.6×.** It is the difference between 20 s/token and ~8 s/token.
Everything below is in service of crossing that 3.75-bit trunk threshold without losing quality.

## Hypothesis A — the 3-bit trunk collapse was the routers, not the attention

WASTE measured a 3-bit trunk destroying the model: logits 36% off, generation degenerating to `+`
and spaces (P). Their stated cause: K3's QAT covered **experts only**, so the trunk has no trained
tolerance for compression.

**But their experiment quantized the trunk as one homogeneous blob.** The trunk is not homogeneous
— it contains at least: attention (KDA ×69, MLA ×24), **routers**, shared experts, **norms**, LM
head, vision tower.

Routers and norms are *tiny in bytes but categorically different in failure mode*. A quantization
error in an attention weight perturbs a value. A quantization error in a **router** changes *which
16 of 896 experts are selected* — a discrete, catastrophic failure that no downstream computation
can recover from. Degenerate output (`+` and spaces) is far more consistent with systematic
misrouting than with mildly noisy attention.

**Hypothesis (D, untested):** trunk components have wildly different bit-sensitivity, and a
mixed-precision trunk — **routers + norms kept at int8/fp16, attention + shared experts pushed to
3.5–4 bits** — clears the 3.75-bit average threshold while avoiding the collapse. Router bytes are
small enough that keeping them at high precision costs little: 92 layers × hidden × 896 experts is
order ~0.6 GiB at int8 (D, needs the real hidden dim to firm up).

**This is the cheapest high-value experiment in the whole program**, and it is testable on
Kimi-Linear-48B (which runs on this laptop today) before K3 is ever downloaded.

## Hypothesis B — AQLM-family methods keep the never-dequantize trick

WASTE's speed rests entirely on a fused VQ matvec: because `sum_s C_s[i]·x_v` depends only on
(stage, code, vector position) and never on the output row, they build a per-token table and each
expert row costs 3 lookups + 2 adds. Dequantization went 87.5% → 0% of runtime (P). **Any 2-bit
method requiring explicit dequantization is disqualified regardless of its compression ratio.**

The key structural observation (D): **additive/residual quantization is exactly the family that
admits this trick.** A weight approximated as a *sum of codebook entries* — which is literally what
AQLM (Additive Quantization) is, and what WASTE's 3-stage residual VQ already is — decomposes into
the same per-(stage, code, position) table. So AQLM-class methods are **structurally compatible by
construction**, not by luck.

The difference is that WASTE's codebooks appear analytically/K-means constructed (their per-stage
error 57.5 / 33.2 / 19.5%, P), whereas AQLM-class methods **learn** codebooks against calibration
data with joint optimization. That is precisely why they beat RTN so decisively at 2 bits — and
why WASTE's 2-bit refutation (tested on RTN) says nothing about them.

**Risk, and it is serious (D):** learned quantization requires a calibration/optimization pass over
the model. For 2.78T params, on this laptop, with a 6 GB GPU, that may be computationally
infeasible — potentially the real blocker of the entire program, ahead of any quality question.
Mitigations to evaluate: requantize *from* the existing MXFP4 checkpoint rather than full
precision; per-expert independent calibration (embarrassingly parallel, resumable, streamable);
or accept analytic (calibration-free) codebooks at some quality cost. **This must be scoped before
committing to the AQLM path.**

## Hypothesis C — frequency-tiered expert precision (disk-only win)

llama.cpp's RFC measured **top 10% of experts take ~80% of routing hits** (P, Qwen3.5-122B).
WASTE separately measured expert *error-sensitivity* is flat across layers (1.01×, P). These are
not contradictory — one is about **access frequency**, the other about **quantization sensitivity**
— and WASTE dismissed frequency-splitting because it saves no *reads*.

For us it saves **disk**, which is the wall:
- Hot 10% @ 2.5 bits + cold 90% @ 2.0 bits = **2.05 bits average** (D) — clears the 2.12 target.
- The 80% of actual computation runs at the *higher* precision, so quality cost concentrates in
  the 20% of routing events that touch cold experts.
- Composes with the SUB1 1-bit substitute bank WASTE specified but never built: coldest experts
  could drop to ~1 bit for further disk savings at bounded quality cost.

Requires per-expert routing statistics, which are obtainable from the recorder machinery already
at issue in T0 (#21) — the two threads connect.

## Candidate target configuration (all D — nothing validated)

| Component | Choice | Size | Rationale |
|---|---|---|---|
| Experts | Additive/learned VQ, frequency-tiered 2.5b hot / 2.0b cold | ~655 GiB | Hypotheses B + C |
| Trunk — attention + shared | 4 bits | ~11.5 GiB | Hypothesis A |
| Trunk — routers + norms | int8 / fp16 | ~1.0 GiB | Hypothesis A — protect the discrete failure mode |
| Vision tower (MoonViT-V2, 401M) | **dropped** (text-only) | −0.4 GiB | Honest scope cut; must be disclosed |
| **Container total** | | **~668 GiB** | fits 745.9 with ~78 GiB spare |
| **RAM: trunk resident** | | **~12.5 GiB** | + ~2.85 fixed = **~15.35 of 15.7 GiB** |
| **bytes/token** | experts only | **~12.3 GB** | trunk streaming eliminated |
| **Projected tok/s** | | **~0.13** (≈8 s/token) | 3.09 GB/s, 50% efficiency discount |

**RAM headroom is ~0.35 GiB. That is not a margin, it is a coin flip.** Any of the fixed-cost
estimates being 10% optimistic erases it. This configuration needs a real measured RAM budget
before it can be trusted, and a graceful fallback to partial trunk residency if it doesn't hold.

## Conversion under the disk wall — a real blocker with a real answer

The source checkpoint is **1.56 TB** (P) on a **745.9 GiB** disk (M). **The source cannot be
downloaded.** Conversion must be a **streaming download→convert→delete pipeline**:
peak disk = output (~668 GiB) + one shard (~16 GiB, from 96 shards) + slack ≈ **~690 GiB** (D) —
fits, with roughly 55 GiB to spare.

WASTE's converter already streams shards at "a few hundred MB regardless of model size" (P), but
almost certainly assumes the full checkpoint is present locally. **Building the fetch-convert-evict
pipeline is a required, buildable ticket** — and it is *independent* of every quantization question
above, so it can proceed in parallel and de-risks the whole program early.

Also unresolved: whether requantizing to a new scheme can proceed from the MXFP4 shards directly,
or requires a higher-precision source that does not exist publicly.

## Ordered experiment program — cheapest kill-shot first

1. **Router-sensitivity probe on Kimi-Linear-48B** (runs on this laptop today, no purchase, no K3
   download). Quantize trunk components independently; find the bit-width at which each collapses.
   **Kills or confirms Hypothesis A for ~0 cost.** Highest information per unit effort in the
   entire program.
2. **LUT-compatibility audit** of AQLM/QTIP/QuIP#/VPTQ — confirm the fused-table matvec survives.
   Pure desk work. Kills Hypothesis B before any compute is spent.
3. **Calibration-cost estimate** for the chosen 2-bit method at 2.78T params. This is the most
   likely program-killer; scope it *before* committing.
4. **Streaming conversion pipeline** — independent of 1–3, de-risks the disk wall.
5. Frequency-tiered expert quantization (needs routing stats — connects to T0/#21).
6. Only then: the full K3 conversion and run attempt.

## What would make this a genuine contribution (beyond the headline)

Even if K3-on-this-laptop fails, three results here are publishable and useful to everyone:
- **Component-wise bit-sensitivity of a frontier MoE trunk** (routers vs attention vs norms) — as
  far as this project can tell, unpublished, and it would explain *why* naive trunk quantization
  collapses.
- **The constraint-inversion table above** — a reusable statement of which offload optimizations
  are disk-bound vs RAM-bound, which the field currently conflates.
- **A streaming requantization pipeline** that converts a model larger than the host's disk.

## Open questions

- K3's exact hidden dimension and per-component trunk byte breakdown — needed to firm up every
  size estimate above; not found published (see k3-on-16gb-feasibility.md open questions).
- Whether K3's QAT genuinely covered experts only, and what that implies for attention tolerance —
  Moonshot's paper (arXiv 2607.24653) not yet read closely.
- Whether the ~2.85 GiB fixed RAM overhead holds on Windows (likely worse) vs Linux.
- Whether "K3 minus the vision tower, expert-quantized to ~2 bits" is still legitimately "running
  K3" — an honesty question for publication, to be answered *before* claiming the milestone, not
  after. Current position: yes if fully disclosed, no if the caveats are buried.
