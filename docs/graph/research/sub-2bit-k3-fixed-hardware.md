---
topic: Can full Kimi K3 (2.78T) fit in 700 GiB / 12.8 GiB-usable-RAM / 3.09 GB/s-NVMe via sub-3-bit quantization, trunk-bitwidth reduction, and/or REAP pruning — and does any method survive WASTE's never-dequantize LUT matvec?
status: resolved
links: [k3-on-16gb-feasibility.md, waste-engine-verified.md]
---

## Verdict

**No — the full, unpruned 982 GiB WASTE-VQ container does not fit in 700 GiB (or even the hard
745.9 GiB disk-free ceiling) with any known, real technique.** Shortfall: 236–282 GiB. No survey
of state-of-the-art quantization (item 1) found a method that both (a) reaches ≤2.16 bits/param
average with acceptable quality on a model this size, AND (b) ships or is even architecturally
suited to WASTE's zero-dequantize LUT matvec (item 2) — see filter results below.

**Best fitting configuration found: REAP-~30% pruned K3 + WASTE's own existing, already-shipped,
already-measured 3.01-bit expert VQ + WASTE's own existing trunk scheme (~27.3 GiB) ≈ 695.6 GiB.**
This requires **zero new quantization research** — REAP is a published, code-available pruning
method (arXiv:2510.13999) that leaves WASTE's kernel and codebook format completely untouched; only
new *pruning-integration* engineering is needed (unbuilt for K3+WASTE specifically, but each half
is independently proven: REAP works on K3's architecture per PipeNetwork's shipped REAP73/80 MLX
ports, and WASTE's 3.01-bit VQ+ADC kernel is what's already running today).

Estimated realistic throughput on Atur's measured hardware (15.7 GiB RAM, 3.09 GB/s NVMe):
**~0.049 tok/s (~20.5 s/token)** — about 10× slower than WASTE's own native 29 GiB-RAM regime
(0.45–0.62 tok/s), because 12.8 GiB usable RAM cannot hold the ~27.3 GiB trunk resident, forcing
~14.5 GiB/token of trunk re-streaming on top of ~17.2 GiB/token of (unavoidable, REAP-invariant)
expert reads. This is a genuine, previously-unbuilt milestone, not a citation.

**Honesty check on "is it still K3":** REAP-30% removes ~30% of the least-salient routed experts
per layer (router-weighted saliency scoring against a calibration corpus), keeps 100% of the
trunk/attention/routing/architecture, and keeps WASTE's exact existing expert-quantization scheme.
REAP's own published numbers (0.2% mean accuracy loss at 25%, 1.4–1.9% at 50%, on comparable-scale
MoE models) suggest this stays close to baseline behavior, but it is **not bit-identical full K3**
— it is "K3 with 30% of its lowest-salience experts surgically removed," a well-defined, honestly-
labeled derivative. No K3-specific REAP variant at this moderate ratio has been published or
independently validated by anyone (only the aggressive 73%/80%-pruned, admittedly-degraded variants
exist) — treat the quality-preservation claim as a cross-model extrapolation, not a K3 measurement.

**Absolute speed ceiling on this hardware, regardless of expert compression:** even in the
impossible limit of zero-cost expert reads, trunk-streaming alone costs ~14.5 GiB/token (below),
capping ceiling throughput at 3.09/14.5 = 0.213 tok/s (≈0.107 tok/s realistic). Pushing expert bits
below ~2.5 bits/weight buys diminishing returns once expert-bytes/token drops near this trunk floor
— see the ranked table.

## Sub-3-bit quantization state of the art (+ LUT-compatibility filter)

**Core distinction driving the whole filter — Type A vs Type B kernels:**
- **Type A ("cheap-dequant-then-GEMM")**: a lookup/formula converts a code → an actual float
  weight value (or a small block of them), which is then multiplied normally in a standard GEMM/
  tensor-core matmul. This is what **every currently-shipped GPU kernel for every method below
  does** (confirmed via FLUTE, arXiv:2407.10960 — "custom kernels that fuse the dequantization and
  matmul operations," explicitly GPU-tensor-core/shared-memory-bandwidth oriented; also confirmed
  for AQLM's own CUDA/Triton/Numba kernels, and VPTQ's public roadmap: "kernel fusion by combining
  dequantization (lookup) and Linear (GEMM)" — Microsoft's own words, meaning even VPTQ has not yet
  shipped this fusion). Materializing real weight values means dequantization cost scales with the
  **output row count**, exactly the 87.5%-of-runtime problem WASTE eliminated.
- **Type B ("asymmetric distance computation" / ADC, WASTE's actual trick)**: because the weight
  row is a sum of codebook vectors and the matvec is a dot product, `sum_s dot(codebook_s[code],
  x_chunk)` can be precomputed **once per token, for all 256 possible codes, for all chunks** —
  independent of which row/expert reads that code. Every row then costs table-reads + adds, never
  a multiply. This requires (i) genuinely **vector** (not scalar/per-element) quantization, (ii) a
  **small, shared** codebook reused across many rows, (iii) a table built once and amortized. This
  is the classical product-quantization/ADC trick from ANN search (Babenko & Lempitsky), applied to
  a matmul instead of a nearest-neighbor distance — sourced: OpenSearch's own ADC writeup
  (opensearch.org/blog/asymmetric-distance-computation-for-binary-quantization).

| Method | arXiv | Bits achieved | Quality (2-bit-ish) | Validated on | Code | Kernel type (shipped) | LUT/ADC-decomposable in principle? |
|---|---|---|---|---|---|---|---|
| **AQLM** (additive quantization) | 2401.06118 | 2–3 bpw, multi-codebook additive/residual VQ | Llama-2 WikiText2 PPL: 7B 6.93, 13B 5.70, 70B 3.94 (@2-bit) | LLaMA-1/2 7B–70B (dense, **not MoE, not 100B+**) | Yes, Vahe1994/AQLM, HF/PEFT-integrated | **Type A** (CUDA/Triton/Numba, "dequantization" explicit in docs) | **YES — same mathematical family as WASTE's own 3-stage residual VQ.** Additive quantization = residual VQ with jointly-optimized codebooks; the ADC trick generalizes directly. No new algorithm needed, just a new CPU kernel (WASTE already has one for this exact structure). |
| **QuIP#** | 2402.04396 | 2 bit, E8P lattice VQ, 8-dim, Hadamard incoherence rotation | Llama-2-70B WikiText2 PPL 3.91–4.16 @2-bit (fp16 baseline ~3.32) | LLaMA-1/2 7B–70B (dense) | Yes, Cornell-RelaxML/quip-sharp | **Type A** via FLUTE | **YES, in principle** — 8-dim vector codebook, same shape as WASTE's stages, but codebook is effectively 2^16 entries (vs WASTE's 256) — a per-token ADC table would be ~256× larger to build, a real (quantifiable, not yet measured) cost. |
| **QTIP** (trellis-coded) | 2406.11235 | 2 bit+, "current 2-bit PTQ frontier," beats QuIP#/AQLM at all tested bitrates | Llama-2/3 7B–70B (dense); exact 2-bit PPL not extracted from abstract-only fetch | LLaMA family, dense | Yes | **Type A**, GPU, parallel bitshift-trellis decode | **NO.** Explicitly designed to avoid a stored codebook — codewords are synthesized on the fly via a pseudorandom-Gaussian hash of a bit window (stateful, sequential-in-spirit like a convolutional code). There is no small fixed table to precompute per token. **Structurally incompatible** with WASTE's core trick without abandoning it. |
| **VPTQ** (Microsoft) | 2409.17066 | <2 bit claimed for 70B/405B | PPL reduction 0.01–0.34 (Llama-2), 4.41–7.34 (Llama-3) vs prior SOTA @2-bit | LLaMA-2/3, Mistral, up to 405B (dense) | Yes, microsoft/VPTQ | **Type A**, and even the dequant+GEMM *fusion* is still on the roadmap (not yet shipped) | **Plausibly yes** (vector quantization w/ 2nd-order-optimized codebooks) but has outlier/bias-correction terms whose row-dependence was not independently confirmed to be ADC-safe. |
| **HIGGS** | 2411.17525 (NAACL 2025) | zero-shot, Hadamard + Gaussian-MSE-optimal grid VQ | outperforms NF4; exact 2-bit PPL not pinned | Llama-3.1/3.2 (8B–405B), Gemma-2 (8B/27B) — **no MoE** | Yes, HF transformers PR #34997 | **Type A** via FLUTE | Plausibly yes (grid/lattice VQ, same lineage as QuIP#), same "larger codebook → bigger per-token table" caveat. |
| **SqueezeLLM** | 2306.07629 (ICML 2024) | validated only to **3-bit** ("lossless up to 3-bit") | — | LLaMA/OPT/Vicuna 7B–65B | Yes | Type A (scalar value-LUT + normal multiply) | **NO** — scalar (element-wise) k-means quantization, not vector; no shared multi-dim codebook to exploit. Also doesn't claim 2-bit at all. |
| **GPTQ / AWQ (2-bit)** | — | 2-bit group quantization | **Collapses**: 10⁴–10⁶ perplexity, or 0.00% accuracy in one cited ablation (arXiv:2604.19884) | many dense models | Yes | Type A, scalar | **NO** (scalar) and **quality-disqualified anyway** — confirms WASTE's own naive-RTN 71.8%-error 2-bit datapoint is not a strawman; independent sources reach the same "scalar 2-bit PTQ fails" conclusion. |
| **QMoE** (Alistarh/IST-DASLab, MoE-specific) | 2310.16795 (MLSys 2024) | **0.8 bits/param**, sub-1-bit | "minor accuracy loss" (self-reported) | SwitchTransformer-c2048, **1.6T params, MoE** — the closest prior-art scale match to K3 | Yes, IST-DASLab/qmoe | **Bespoke GPU decoding kernels**, custom lossless **entropy coding** (Huffman/arithmetic-style), not vector quantization | **NO — the worst case found.** Entropy coding requires full *sequential* decode of a compressed bitstream from its start; you cannot randomly `pread` into the middle of it. This is structurally the opposite of WASTE's per-token random-access architecture. Also GPU-only (4×A6000/8×3090 target). |
| BitsMoE (2026, MoE-specific) | 2606.00079 | 2-bit, SVD + spectrum-wise mixed precision | +27.8pp accuracy vs GPTQ | Qwen3-30B-A3B (30B MoE) | Yes, zjiayu064/BitsMoE | Not confirmed; SVD+mixed-precision-scalar pattern suggests **Type A at best**, likely not vector/ADC-decomposable | Unclear/unlikely — flagged, not confirmed. |
| TileQ (2026, MoE-specific) | 2605.09281 | low-bit, low-rank + 2D tiling | MMLU/HellaSwag/Winogrande | Mixtral, Qwen1.5-MoE, DeepSeek-MoE | Anonymous repo during review | Not extractable from available tools | Unresolved — flagged as a lead, not characterized. |
| KBVQ-MoE (2026, MoE-specific) | 2602.11184 | KLT/SVD-whitening + "bias-corrected **vector quantization**" | WikiText2/ARC/HellaSwag/etc; exact numbers locked in PDF figures, not extractable | Qwen1.5/3/3-Next, Mixtral, DeepSeek-v2 | Not stated | Not extractable | **Most structurally promising 2026 sibling of WASTE's own approach** (explicit VQ + a Hadamard-like whitening transform, same lineage as QuIP#) — genuinely unresolved due to tool limits, worth a manual PDF read if pursued further. |

**Net conclusion on item 1+2 combined:** the academic 2-bit-class frontier (QuIP#/QTIP/AQLM/VPTQ/
HIGGS) is real and *does* beat WASTE's naive-RTN strawman decisively on quality — but every one of
them is (a) validated only on **dense 7–405B LLaMA-family models, never on a 1T+ MoE**, and (b) has
**zero shipped Type-B (never-dequantize) kernel**; all current implementations are GPU-tensor-core
oriented. The one MoE-native, trillion-scale precedent (QMoE) achieves the best raw bit-rate
(0.8 bpw) but uses a scheme structurally *incompatible* with WASTE's random-access streaming
architecture. **The lowest-risk path to a better sub-3-bit expert scheme is not to adopt any of
these methods' kernels — it is to borrow AQLM's Hessian-aware sequential codebook-training
methodology and apply it to train WASTE's own existing 3-stage residual-VQ codebooks better,
while keeping WASTE's shipped CPU ADC kernel 100% unchanged.** This is a plausible, well-reasoned,
**unbuilt** engineering path, not a measured result.

## Trunk bit-width: how low can K3 go, and does the trunk become RAM-resident

**Correction to the prompt's framing: 4-bit trunk is NOT untested — it appears to already be
WASTE's shipped production default, and it works.** This is the single most important, and most
surprising, finding of this research pass.

- Direct quotes recovered from `docs/LEARNED.md` (raw GitHub fetch, this session):
  - `"Q4G trunk | 27.50 GB | 17.10 GB | 12% | 0.23 | coherent"` — 4-bit-bulk trunk produces
    **coherent** generation.
  - `"Q3G trunk | 21.13 GB | 23.48 GB | 29% | 0.16 | `+` and spaces"` — 3-bit trunk **collapses**
    (matches the prior research node's independently-sourced finding, exact wording corroborated
    twice now: "the logits land 36% off the 4-bit ones and generation collapses").
  - `"The trunk's bulk has been 4-bit by default since before §13 refuted 3."` — states plainly
    that **4-bit-for-the-bulk is already the standing default**, and 3-bit was tried and rejected.
  - `"make_test_container.py now mirrors convert.py's widths, 4 bits for the bulk and 8 at both
    ends"` — confirms the **production converter** (`convert.py`, not just a test fixture) ships
    this mixed 4-bit/8-bit scheme.
  - README.md (raw fetch, this session): **"Experts use 3-bit residual vector quantization, while
    the more sensitive shared weights remain at 4 or 8 bits."** — current top-level headline
    description, explicitly **not** "int8."
  - `"QAT covered the expert weights at MXFP4 and left every non-expert component in higher
    precision"` — independently corroborates the Kimi K3 technical report's own account
    (arXiv:2607.24653, Moonshot, 2026-07-27: QAT applied through SFT+RL using MXFP4
    weights/MXFP8 activations) that **only routed-expert weights received quantization-aware
    training** — the trunk has zero learned tolerance for quantization noise, which is exactly why
    it breaks first (at 3-bit) while experts tolerate far more aggressive compression (3.01-bit,
    "only" 19.4% weight error, no reported collapse).
- **Unresolved internal inconsistency, flagged rather than papered over:** the prior research node
  (`k3-on-16gb-feasibility.md`, same-day earlier pull) cites "27.28 GB, stored at int8 with f32
  arithmetic ('5.6 GB RAM freed')" as the current trunk figure. This session's fresher pull shows
  "Q4G trunk | 27.50 GB" and the README's "4 or 8 bits" language. The **total byte count has
  stayed essentially stable (~27.3–27.5 GiB) across both pulls**, even though the *description* of
  how those bytes are allocated (flat int8 vs. mixed 4-bit-bulk/8-bit-ends) differs. Two readings
  are consistent with this: either (a) "int8" described an earlier, now-superseded scheme that
  happened to total a similar size, or (b) both descriptions are describing the same current
  scheme loosely/inconsistently. **This is exactly the doc-drift pattern already flagged twice in
  this project's other WASTE-sourced nodes** ("numbers are a genuinely moving target," multiple
  ships per day). No local shell/`git clone` access was available this session (WebFetch's
  small-model summarizer repeatedly garbled table structure across three separate attempts) — a
  direct clone + `grep` pass is needed to pin the exact current byte count, not just its rough
  size. Treat ~27.3 GiB as reliable to ±1 GiB; do not treat the "int8 vs mixed-4/8" distinction as
  resolved.
- **No 5-bit or 6-bit trunk test was found anywhere** (LEARNED.md, TECHNICAL.md, K3.md, FORMAT.md,
  README.md all searched) — this part of the prompt's framing holds. Only Q3G (collapse), Q4G
  ("coherent," apparent default), and Q8G ("higher precision," ~0.64%-off-source quoted at low
  confidence) were found as tested tiers.

**Payoff computation — does the trunk become RAM-resident at 12.8 GiB usable RAM?**

| Trunk scheme | Size | Resident (min with 12.8 GiB budget) | Streamed/token | Status |
|---|---|---|---|---|
| int8 flat (as literally described in the prior node) | 27.28 GiB | 12.8 GiB | 14.48 GiB | tested-quality unclear if literally this scheme still ships |
| **Q4G (current apparent default, "coherent")** | **~27.3–27.5 GiB** | **12.8 GiB** | **~14.5–14.7 GiB** | **the real, current number to use** |
| Q3G (demonstrated collapse) | ~21.13 GiB (one reading) | 12.8 GiB (would fit better) | ~8.3 GiB | **quality-disqualified — do not use despite better fit** |
| hypothetical pure 4-bit uniform (27.28×4/8) | 13.64 GiB | 12.8 GiB | 0.84 GiB | **not what's shipped**; would need a from-scratch uniform-4-bit trunk requant, untested end to end |
| hypothetical 3.5-bit uniform | 11.94 GiB | 11.94 (fully resident!) | 0 GiB | **matches the prompt's own cited number**; UNTESTED at this precise point, and dangerously close to the demonstrated Q3G (3-bit) collapse |

**Honest answer: no, the trunk does not become fully RAM-resident at any bit-width that has been
shown to work.** Even WASTE's own already-aggressive current default (~27.3 GiB, apparently
already 4-bit-for-the-bulk) is **2.1× too big** for the 12.8 GiB usable-RAM budget. The prompt's
hope that 4/5/6-bit trunk represents fresh, unclaimed headroom does not pan out — **the WASTE team
has already pushed this exact lever about as far as it safely goes** (tried 3-bit, it collapsed;
landed on 4-bit-mixed as the floor). The genuinely open question is narrower than the prompt
assumed: not "does 4-bit work" (yes, apparently already shipped) but "does something *between*
Q3G's demonstrated collapse and Q4G's working point — e.g. an asymmetric ~3.5–3.75-bit scheme with
a different ends/bulk split than either tested tier — survive?" This is **untested** and is the
single highest-value remaining unknown in this whole report: if it works, full trunk residency
(0 GiB/token trunk cost, expert-reads-only decoding) becomes reachable; if it doesn't, ~14.5
GiB/token of trunk streaming is a fixed cost baked into every configuration on this hardware.

## Expert pruning (REAP) as a complementary lever

- REAP = Router-weighted Expert Activation Pruning, Cerebras, arXiv:2510.13999, accepted ICLR 2026.
  Scores each expert by `gate_value × ||expert_output||` over a calibration corpus, removes
  lowest-salience experts entirely. Code: github.com/CerebrasResearch/reap.
- **Top-k routing width is unchanged by pruning** (confirmed via the README: "in expert pruned
  SMoEs the router maintains independent control over the remaining experts" — pruning shrinks the
  *pool*, not the number selected per token). **This has a critical, non-obvious consequence for
  WASTE's I/O-bound decode: REAP reduces total container size but does NOT reduce per-token bytes
  read** — the engine still reads `top_k × num_MoE_layers` expert records every token, unaffected
  by how many total experts exist. REAP is therefore a **pure disk-footprint lever with zero direct
  tok/s effect** — orthogonal to (and stackable with) any bits/param change, which is the only
  lever that touches per-token I/O and hence speed.
- Published quality numbers (arXiv:2510.13999 + cerebras.ai/blog/reap, both fetched this session):
  **25% pruning → 0.2% mean accuracy decrease** (near-lossless) on non-agentic coding for
  Qwen3-Coder-480B-FP8-class / Kimi-K2-Instruct-W4A16-class models; **50% pruning → 1.4–1.9% mean
  decrease**, 97.6% retained on non-agentic coding, 96.7% on SWE-Bench (Qwen3-480B-Coder-FP8),
  "near-lossless" on Kimi-K2 (1T) at 50%. Models tested: ERNIE-4.5-21B-A3B, Qwen3-30B-A3B,
  Qwen3-Coder (30B & 480B), Mixtral-8x7B, GLM-4.5-Air, Llama-4-Scout-17B-16E, Kimi-K2. **No K3 was
  tested by the paper itself.**
- **K3-specific published REAP variants exist only at aggressive ratios**: `pipenetwork/Kimi-K3-
  REAP73-MLX-mxfp4-q8` (73% pruned, 451 GiB) and `Kimi-K3-REAP80-MLX-mxfp4-q8` (80% pruned,
  350 GiB) — both use MLX + native-ish mxfp4/q8 precision (**not** WASTE's tighter 3.01-bit VQ),
  both self-described as showing "noticeable degradation" (repetitive/list-like output, degraded
  Chinese-language performance). **No moderate (25–40%) K3 REAP variant has been published by
  anyone**, in any format. This is the honest gap: the near-lossless 25–50% evidence transfers
  from Kimi-K2 (same lineage, same MoE-with-shared-experts pattern, though K3 also adds
  KDA/hybrid-attention K2 may lack) — a reasonable but unconfirmed cross-model extrapolation, not a
  K3 measurement.
- **REAP mechanically works on K3's architecture** — proven, not hypothetical: PipeNetwork's own
  `kimi-k3-mlx` GitHub project explicitly runs REAP's saliency scoring (`gate × ||expert_output||`
  over a 12.6 MB mixed-language/code calibration corpus) against all 896 experts/layer of real K3
  weights and ships the resulting pruned models. What's missing is only a *moderate-ratio*,
  *WASTE-format* K3 REAP variant — an integration gap, not a feasibility question.
- **Container-size math (derived, using WASTE's own 982 GiB / 27.28 GiB(trunk) / 954.72 GiB
  (experts) split):** `container(p) = 27.28 + 954.72×(1−p)` GiB. Solving for the disk targets:
  - **p ≈ 24.7% pruned** hits the hard 745.9 GiB disk-free ceiling exactly (no OS headroom margin
    — not recommended as a target, shown for completeness).
  - **p ≈ 29.5% pruned** (round to **REAP-30**) hits the 700 GiB design-target ceiling exactly,
    leaving the intended OS headroom. → **695.6 GiB** at p=30% precisely.
  - Both fall inside REAP's own well-evidenced near-lossless-to-mild-degradation zone (25–50%),
    reinforcing REAP-30% as the recommended operating point.

## Ranked configuration table

All rows use: trunk streamed bytes = `max(0, trunk_GiB − 12.8)`; expert bytes/token =
`17.2 × (bits/3.01)` GiB (17.2 = midpoint of WASTE's measured 17.0–17.4 GiB/token range); ceiling
tok/s = `3.09 / bytes_per_token`; realistic tok/s = ceiling × 0.5 (WASTE's own measured
achieved-vs-ceiling ratio, established in `k3-on-16gb-feasibility.md`). Container = trunk_GiB +
`954.72 × (bits/3.01) × (1−p)`.

| # | Config | Expert bits | REAP p | Trunk | Container (GiB) | Fits 700 GiB? | Trunk resident? | Bytes/token | Ceiling tok/s | Realistic tok/s (s/token) | Quality cost | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | WASTE as-shipped, unmodified | 3.01 | 0% | ~27.3 GiB | **982** | **No** (−236 GiB) | n/a (needs 29 GiB total RAM, don't have it) | — | — | — | 19.4% weight err (baked in) | measured |
| 1 | **Full trunk streamed, no REAP** (prior node's own scenario) | 3.01 | 0% | ~27.3 GiB | 982 | **No** (−236 GiB) | 47% (12.8/27.3) | 31.68 GiB | 0.0975 | **0.049 (~20.5 s)** | 19.4% err | mostly measured + derived residency |
| 2 | **Recommended: REAP-30 + WASTE VQ unchanged** | 3.01 | 30% | ~27.3 GiB | **695.6** | **Yes** (margin 4.4 GiB) | 47% (12.8/27.3) | 31.68 GiB | 0.0975 | **0.049 (~20.5 s)** | 19.4% err + ~0.3–0.8% REAP (extrapolated) | container: derived; REAP quality: published on other models |
| 3 | REAP-25 + WASTE VQ unchanged | 3.01 | 25% | ~27.3 GiB | 743.3 | Barely (hard-ceiling only, no OS headroom) | same | same | same | same | 19.4% + ~0.2% REAP | derived + published |
| 4 | REAP-30 + expert 2.0-bit (WASTE's own measured 2-stage VQ) | 2.0 | 30% | ~27.3 GiB | 503.2 | Yes (huge margin, 197 GiB unused) | same | 14.5+11.43=25.93 GiB | 0.1192 | 0.060 (~16.8 s) | **33.2% weight err (measured, much worse)** + REAP | container derived; VQ-error measured at exactly this point |
| 5 | REAP-15 + expert 2.5-bit (interpolated) | ~2.5 | 15% | ~27.3 GiB | 687.6 | Yes (tight) | same | 14.5+14.29=28.79 GiB | 0.1073 | 0.054 (~18.6 s) | **~25–28% weight err — interpolated, NOT measured** | low confidence on quality number |
| 6 | (upside scenario, unconfirmed) REAP-30 + expert 3.01-bit + **hypothetical safe 3.5-bit trunk** | 3.01 | 30% | 11.94 GiB (fully resident) | 683.5 | Yes | **100%** | 0 + 17.2 = 17.2 GiB | 0.1797 | 0.090 (~11.1 s) | 19.4% + REAP + **unvalidated trunk risk** (adjacent to demonstrated 3-bit collapse) | **speculative — do not plan around this row** |
| 7 | Theoretical absolute floor (experts free, impossible) | 0 (n/a) | n/a | ~27.3 GiB | n/a | n/a | 47% | 14.5 GiB | 0.213 | 0.107 (~9.4 s) | n/a | derived, illustrates the trunk-streaming speed ceiling |

**Recommendation: row 2.** It uses only measured, already-shipped WASTE numbers plus a published,
code-available, architecture-proven pruning method, at REAP's own well-evidenced low-degradation
ratio, and comfortably clears the 700 GiB target. Rows 4–5 buy modest speed (+22–10%) at
disproportionate, partly-unmeasured quality cost the disk budget does not actually require paying
(row 2 already fits with margin). Row 6 is the highest-upside, highest-risk item for future work —
worth a cheap validation experiment (WASTE already has the oracle-diff harness to test it) before
being ruled in or out.

## Conversion under a 745.9 GiB disk constraint

**Not a blocker — WASTE's own converter already solves the core problem, and a full streaming +
pruning pipeline has already been built and shipped for K3 specifically (by a third party).**

- WASTE's own `docs/FORMAT.md` (raw fetch, this session): the converter is Python (needs
  torch/safetensors; the *inference* path needs neither), and explicitly **"stream[s] release
  shards one at a time — never needs the full 1.42 TB locally beyond the shard in flight plus the
  output."** Peak conversion RAM: "a few hundred MB regardless of model size" (WASTE's own figure)
  to "a few tens of GB" (PipeNetwork's independent MLX-target converter, which does the equivalent
  streaming job for a different backend). This is **already how WASTE produced its own existing
  982 GiB K3 container** from the 1.56 TB HF source — it is not hypothetical.
- **Concrete existing proof of the full combined pipeline (stream + prune + quantize + discard),
  built for K3 specifically**: PipeNetwork's `kimi-k3-mlx` (github.com/PipeNetwork/kimi-k3-mlx).
  Their own description: *"We wrote a streaming converter that walks one layer at a time, so that
  mlx_lm doesn't need to materialize the whole model [K3 is 5.6 TB in bf16]. REAP pruning sits on
  top and scores all 896 experts against a calibration corpus to keep only ones your workload
  needs."* For lossy tiers they explicitly "dequantize and requantize one expert at a time rather
  than stacking 896 of them in bf16." This is a **directly transferable proof of concept** — same
  target model, same combined stream+prune+quantize technique, different backend (MLX/Apple
  Silicon instead of WASTE/CPU).
- **The real practical constraint is network bandwidth/time, not disk capacity.** A correct
  REAP-then-quantize pipeline plausibly needs **two passes** over the source data in the worst
  case: pass 1 streams every expert once to compute REAP saliency scores (a forward-pass signal,
  `gate × ||expert_output||`, over a small ~12.6 MB calibration corpus — cheap compute, but still
  requires touching every one of the 1.56 TB of source bytes at least once to score them); pass 2
  re-streams only to quantize+write the *surviving* experts, discarding pruned ones and all
  non-selected data. Whether pass 2 can *skip downloading* pruned experts' shards entirely depends
  on whether HF's shard boundaries align with per-expert boundaries (not confirmed either way in
  this pass) — if not, pass 2 still downloads everything but only writes/quantizes a subset. At
  2×1.56 TB of network traffic, on a typical home connection (~100–500 Mbit/s), this is a
  **multi-hour-to-day-scale bandwidth/time cost**, not a disk-capacity blocker: peak simultaneous
  disk usage is only `output-so-far (up to ~695–745 GiB) + 1–2 shards in flight (a few hundred MB
  to a few tens of GB)`, comfortably inside 745.9 GiB free at every point in the pipeline.
- **Open engineering gap, not a physical blocker**: nobody has built this exact combined pipeline
  for WASTE's own format (only WASTE's plain stream-convert exists, and only PipeNetwork's
  stream-convert-prune exists, for a different backend). Porting PipeNetwork's REAP-scoring logic
  into WASTE's `tools/convert.py` (both are Python, both already do streaming) is plausible,
  bounded, unbuilt engineering work — not a research question.

## What is measured vs published vs derived

- **Measured on Atur's hardware** (from `k3-on-16gb-feasibility.md`, re-used here unchanged): RAM
  15.7 GiB, usable-for-weights ~12.8 GiB, NVMe seq. read 3.09 GB/s, disk free 745.9 GiB.
- **Measured/published by WASTE** (raw docs fetched 2026-08-02, this session, flagged where two
  pulls disagree): container 982 GiB, expert 3.01-bit/19.4%-error VQ, stage-error curve
  (57.5%/33.2%/19.4% at 1/2/3 stages), expert reads 17.0–17.4 GiB/token, RTN-2bit 71.8% error
  (weak strawman, confirmed real), Q3G-trunk collapse (exact quotes reproduced above), Q4G-trunk
  "coherent" and apparently-default (exact quotes reproduced above; **exact current byte count has
  an unresolved ±small discrepancy between two same-day pulls, ~27.28 vs ~27.50 GiB**), QAT
  covered experts-only (independently corroborated by both WASTE's own docs and secondary coverage
  of arXiv:2607.24653).
- **Published, independently verified this session**: AQLM/QuIP#/QTIP/VPTQ/HIGGS/SqueezeLLM/
  GPTQ-AWQ-2bit-collapse/QMoE bit-rates, quality numbers, kernel types, code availability (each
  cited with its own arXiv ID above); REAP's 25%/50% quality numbers and top-k-invariance
  (arXiv:2510.13999, cerebras.ai/blog/reap, github.com/CerebrasResearch/reap); Kimi-K3-GGUF
  community quantization numbers (Unsloth's UD-IQ1_S…UD-Q8_K_XL table with PPL/KLD/top-1, and
  GrEarl's IQ1_S tensor-precision breakdown — both real, measured, K3-specific data points,
  independent of WASTE); PipeNetwork's K3 REAP73/80 sizes and their streaming-converter design.
- **Derived (this report's own arithmetic, not independently measured)**: all container-size and
  bytes/token formulas in the ranked table above; the REAP-ratio-to-disk-target solve (p≈24.7%/
  29.5%); the expert-bits-vs-trunk-streaming crossover point (~2.54 bits); the 2.5-bit and
  3.5-bit-trunk interpolated quality figures (explicitly flagged low-confidence); the LUT/ADC-
  compatibility tier assignments for QuIP#/HIGGS/VPTQ/KBVQ-MoE (structurally reasoned, not tested).

## Open questions

- **Highest-value unresolved question**: does any trunk bit-width *between* the demonstrated Q3G
  collapse (~3-bit) and the working Q4G default (~4-bit-bulk) survive — e.g. an asymmetric
  3.5–3.75-bit scheme with a different bulk/ends split? If yes, full trunk residency (0 GiB/token)
  becomes reachable within 12.8 GiB and the realistic-throughput floor roughly doubles (row 6 in
  the table). WASTE's own oracle-diff harness could answer this cheaply if the codebase is cloned.
- **Exact current trunk byte count and precision scheme is unresolved** — two same-day WebFetch
  pulls disagree on "flat int8" vs "4-bit-bulk/8-bit-ends," while agreeing the total is ~27.3–27.5
  GiB. Needs a direct `git clone` + `grep`/`cloc`-style pass (not available to this research task;
  flagged as a limitation, same as the prior WASTE-sourced nodes).
- **No K3-specific, moderate-ratio (25–40%) REAP variant exists anywhere** — the near-lossless
  quality claim used to justify the recommended configuration is a cross-model extrapolation from
  Kimi-K2/Qwen3-Coder-480B/GLM-4.5-Air, not a K3 measurement. Building and testing one is the
  single most valuable next validation step.
- **KBVQ-MoE (arXiv:2602.11184) inference-kernel details are locked in PDF figures** the available
  tools could not extract — this is the most structurally-promising 2026 MoE-native VQ method
  found (explicit vector quantization + whitening transform) and deserves a manual read if this
  work continues.
- **Whether HF shard boundaries align with per-expert boundaries** (needed to know if a
  REAP-then-convert pipeline can skip downloading pruned experts' shards on a second pass, or must
  always re-download everything) — not confirmed either way.
- **AQLM/QuIP#-style codebook-training methodology has never been applied to WASTE's own residual-
  VQ format** — the report's own recommended lowest-risk quality-improvement path (borrow the
  calibration algorithm, keep WASTE's kernel) is a reasoned proposal, entirely unbuilt and
  unmeasured.
- Everything already flagged as open in `k3-on-16gb-feasibility.md` (per-layer trunk byte
  non-uniformity, large-block-random-read bandwidth vs. sequential, whether Bigtea should pursue
  this at all) still applies and is not re-litigated here.
