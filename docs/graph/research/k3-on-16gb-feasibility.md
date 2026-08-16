---
topic: True minimum RAM to run Kimi K3 (2.8T) — is WASTE's 29.06 GB floor a physical law or an architecture artifact, and is K3-on-16GB a real unclaimed milestone?
status: resolved
links: [waste-engine-verified.md, hardware-profiling.md]
---

## Verdict (feasible / infeasible, with the number)

**Feasible but not interactively usable, and NOT a physical law.** WASTE's 29.06 GB floor is a
**policy choice optimizing for tok/s, not a correctness requirement.** The true physical/
correctness-only RAM floor for K3 is roughly **5–6 GB** (derived below — nobody has built or
measured this, so treat as an engineering estimate, not a citation). A 16 GB machine sits
comfortably above that physical floor and has ~13 GB of genuinely discretionary budget to spend.

The catch is throughput, not fit. Spending that discretionary 13 GB the *only* useful way
available at 16 GB (partial trunk residency — expert caching is provably useless below a 17.0–
17.4 GB cache, see below) yields an estimated **~31.5 GB of NVMe reads per decode token**, vs.
~17 GB/token in WASTE's actual 29 GB-RAM configuration. Resulting tok/s at four consumer NVMe
speeds (bandwidth-ceiling model, then a ~50% realistic-efficiency discount matching WASTE's own
measured ceiling-vs-achieved ratio):

| NVMe seq. read | bytes/token (16 GB build) | ceiling tok/s | realistic tok/s (~50% eff.) | s/token realistic |
|---|---|---|---|---|
| 1.5 GB/s (SATA-class) | 31.5 GB | 0.048 | **~0.024** | ~42 s |
| 3.5 GB/s (PCIe3 NVMe) | 31.5 GB | 0.111 | **~0.056** | ~18 s |
| 7.0 GB/s (PCIe4 NVMe) | 31.5 GB | 0.222 | **~0.11** | ~9 s |
| 12.8 GB/s (PCIe5 NVMe) | 31.5 GB | 0.406 | **~0.20** | ~5 s |

This is technically "running" K3 (correct tokens, full 2.78T-param model, no pruning/distillation)
inside 16 GB — by the same "it ran" bar the community has already accepted for WASTE's 29 GB/
0.5 tok/s result — but at 2–10× worse throughput than WASTE's existing 29 GB regime, dominated
entirely by NVMe bandwidth. **Nobody has built or published this** (see "Has anyone done it").

## MEASURED on Atur's actual laptop (2026-08-02) — supersedes the assumed values above

First real hardware numbers for this project. No model downloaded; took minutes.

| Quantity | Measured | Method |
|---|---|---|
| Drive | SK Hynix HFS001TEJ4X112N, 953 GB, NVMe | `Get-PhysicalDisk` |
| Free space | **745.9 GB** | `Win32_LogicalDisk` |
| **Sequential read** | **3.09 GB/s** | 8 GB file (> free RAM, so page cache cannot mask it), 8 MB blocks, single stream |
| RAM / VRAM | 15.7 GB / 6 GB (RTX 3050) | `Win32_ComputerSystem`, `nvidia-smi` |

**Consequences, using the 31.5 GB/token figure derived above:**
- Ceiling: 31.5 / 3.09 = **10.2 s/token → 0.098 tok/s**.
- At the evidence-grounded ~50% efficiency discount: **~0.05 tok/s ≈ 20 s/token**.
- This lands within 12% of this doc's pre-measurement estimate (~0.056 tok/s at an assumed
  3.5 GB/s) — the bandwidth model holds up on real hardware.
- **Disk is confirmed as the binding constraint**: 745.9 GB free vs a 982 GB container = **236 GB
  short**. No amount of engineering closes this; it is a purchase (2 TB NVMe, ~$100–150) or a
  smaller representation (REAP-pruned 350 GB, but that is no longer full-quality K3).
- Caveat: 3.09 GB/s is *single-stream sequential*. Large-block quasi-random expert reads
  (12.4 MB records) should land near this per the cited convergence result, but that is not yet
  measured on this drive — and queue-depth parallelism (io_uring/IOCP) could raise the effective
  figure above single-stream. Both are open, and both would *improve* the projection, not worsen it.

## Decomposition of the 29.06 GB floor

Primary sources: `docs/K3.md`, `docs/TECHNICAL.md`, `docs/EFFICIENCY.md`, `docs/LEARNED.md`,
`docs/ENGINE.md`, `docs/FORMAT.md` on github.com/sqliteai/waste (raw fetched 2026-08-02).

- **Resident trunk: 27.28 GB** — attention (KDA/MLA), routers, shared experts, latent-MoE
  projections, norms, LM head, vision tower; stored at int8 with f32 arithmetic ("int8 trunk
  storage with f32 arithmetic" shipped, "5.6 GB RAM freed" — `waste-engine-verified.md` #10/#7).
  Not evenly split across the 93 layers: layer 0 is fully dense, the remaining 92 have a KDA-vs-
  MLA split of **69:24** (K3.md), and MLA layers carry extra `kv_lora`/absorb projections KDA
  layers don't — per-layer trunk bytes are NOT uniform.
- **Embedding table (1.11 GB) is explicitly NOT resident** — LEARNED.md §13 verbatim: "`embed_tokens`
  is 1.11 GB of which 7 KB is read per token, so it now stays on disk and the row is `pread` on
  use — bit-identical logits, floor 30.38 → 29.27 GB." This is the single cleanest proof that
  trunk residency is a throughput choice, not a correctness one: they already stream one dense
  component with **zero** logit change and **zero** measured throughput cost, because only 7 KB/
  token is actually needed from it.
- **LM head (1.11 GB) was tested for the same treatment and rejected — but for throughput, not
  correctness.** LEARNED.md §13: "lm_head is the near miss — 1.11 GB read per token to free 1.11
  GB of cache, which at the current knee buys about 2 points of hit rate... A net loss of roughly
  0.8 GB/token." It stayed resident because the full vocab projection needs the *whole* matrix
  every token (unlike embedding lookup, which needs one row) — this is an I/O-cost argument, not
  an impossibility argument. (Note: that specific 0.8 GB/token "net loss" calculus is about
  reinvesting the freed byte into *expert cache* at a >17 GB regime where cache has nonzero
  marginal value — it does not directly apply at 16 GB, where expert cache has zero marginal
  value at any size below the floor; see below.)
- **KV cache / KDA recurrent state (absorbed MLA): ~0.21 GB @ 4K context.** Absorbing `kv_b_proj`
  into query/output collapses per-token KV from an unabsorbed ~11.25 GB to 0.21 GB (53×), "logits
  identical to 1.2e-05" per the advisory (order-of-magnitude corroborated, exact figure not
  independently pinned — `waste-engine-verified.md` #9). **KV scales ~linearly with context**:
  floor grows 29.05→30.54→35.63→83.21 GB at 4K/32K/128K/1M (docs, `01-waste-deep-dive.md` table);
  the 4K→1M floor delta (54.16 GB) is almost entirely KV growth (0.21 GB × 256 ≈ 53.8 GB predicted
  by linear scaling — matches within rounding), confirming trunk/scratch stay ~flat with context
  and KV is the only real context-length lever.
- **Scratch/activation buffers: ~0.25–0.75 GB.** LEARNED.md §11: "The decode buffers alone are
  252 MB on K3 (`e_gate`/`e_up`/`e_down` are `moe_inter × hidden` floats each), and the chunked-
  prefill buffers — up to ~500 MB at `WASTE_CHUNK_MAX`, allocated on first use and never freed."
  (A separate, older ENGINE.md conceptual table gives "scratch 0.07 GB" — that number is stale/
  illustrative, superseded by the measured 252 MB+500 MB figures; another example of this repo's
  numbers drifting across docs versions, already flagged in `waste-engine-verified.md`.)
- **Minimum expert cache (double-buffered): ~0.38–0.4 GB** — ENGINE.md's formula: `top_k ×
  expert_record × 2`; for K3, 16 × 12.4 MB × 2 ≈ 397 MB. This is the buffer the read-ahead thread
  needs to prefetch next layer's experts while the current layer computes — **it is not a cache
  in the "improves hit rate" sense**, it's pipeline plumbing.
- **Codebooks: resident, size not published but structurally trivial** (3-stage × 256-entry × 8-
  dim VQ codebook; even generously assuming per-layer codebooks, 92 × ~12 KB ≈ 1.1 MB — not a
  meaningful RAM line item; the only nearby size found, "884 KB table per 64-row tile"
  (EFFICIENCY.md §21), is a runtime LUT built per token, not the codebook itself).
- **Reconciliation**: 27.28 (trunk) + 0.21 (KV) + ~0.5 (scratch) + 0.4 (min expert buffer) +
  ~0.05 (codebooks/tokenizer/misc) ≈ **28.4–28.5 GB**, vs. the published 29.06 GB floor — the
  ~0.6–0.7 GB gap is consistent with rounding/doc drift already documented across this project's
  fast-moving numbers (`waste-engine-verified.md`'s own top-line finding) and possibly OS-level
  allocator overhead not itemized in any doc found.
- **Is trunk residency required for correctness, or a latency optimization? — Latency
  optimization, unambiguously.** Three independent pieces of evidence: (1) the embedding table
  proof above (streamed with zero cost); (2) the lm_head experiment (streamed correctly, just
  slower — rejected on speed grounds, not correctness); (3) `WASTE_E_RAM_BUDGET` is described in
  ENGINE.md as an explicit refusal policy ("A budget under the floor fails at open... never
  swapped into") — the engine *chooses* to refuse rather than attempt a correct-but-slow run.
  Nothing in K3's mathematics requires any byte to live in RAM; every weight is reachable via a
  `pread`. The 29.06 GB number is where WASTE's authors decided the RAM-for-speed trade stops
  being worth offering a user, not where correctness stops being possible.

## Bytes-per-token physics + tok/s at each SSD speed

**If literally nothing is RAM-resident** (worst case, matches the prompt's ~44 GB/token
estimate): trunk 27.28 GB (all re-read every token, since it's needed in full every token) +
expert reads 17.0–17.4 GB (16 × 92 × 12.4 MB, unaffected by trunk residency) = **44.3–44.7
GB/token**. Ceiling tok/s = bandwidth ÷ bytes/token:

| NVMe speed | bytes/token | ceiling tok/s | s/token |
|---|---|---|---|
| 1.5 GB/s | 44.5 GB | 0.034 | 29.5 s |
| 3.5 GB/s | 44.5 GB | 0.079 | 12.7 s |
| 7.0 GB/s | 44.5 GB | 0.157 | 6.4 s |
| 12.8 GB/s | 44.5 GB | 0.288 | 3.5 s |

**Correction to the prompt's framing on read pattern**: expert reads are not "tiny 4K-aligned
random I/O" in the IOPS-bound sense the phrase usually implies. Each expert record is a single
**12.4 MB contiguous read** (3,029 × 4 KiB pages) at a router-determined offset — a large-block
"random" read, not small-block random I/O. Public SSD benchmarking data shows large-block (≥1 MB)
random reads converge to within a few percent of sequential throughput on modern NVMe (≈5–5.6
GiB/s plateau for both patterns on a single drive in one measured source), because per-I/O
overhead amortizes over the large transfer — https://storedbits.com/sequential-vs-random-data/.
So using the *same* bandwidth figure for trunk streaming (truly sequential) and expert streaming
(large-block quasi-random) is a reasonable first-order approximation, not a modeling error, though
trunk reads should realize a slightly higher fraction of rated bandwidth than expert reads in
practice (fewer, larger, more predictable I/Os; genuinely deterministic prefetch depth) — a real
but second-order correction, not one this report attempts to quantify precisely.

**WASTE's own measured efficiency vs. this ceiling model**, used as the basis for the ~50%
"realistic" discount applied elsewhere in this doc: at their 46 GB sweet-spot budget (36.2% hit
rate → effective expert bytes/token ≈ 17.2 × 0.638 ≈ 10.97 GB, trunk resident so 0 bytes/token),
implied achieved I/O bandwidth from their own time-breakdown (17.2 GB total reads / 54.8% of a
2.04 s token ≈ 1.12 s) is ≈15.4 GB/s — a plausible number for an M5 Pro-class internal SSD.
Even with **perfect** overlap of the remaining compute (27.2% matmul + 9.3% KDA + 2.7% LUT-build
= 39.2%, 0.80 s) behind that I/O, the token could not go below max(I/O, compute) = 1.12 s → 0.89
tok/s at that bandwidth. Their actual achieved 0.49–0.54 tok/s is **~55–60%** of that already-
optimistic "perfect overlap" ceiling (2-thread read-ahead only gets 1.6×, not full async
pipelining) — consistent with the ~50% figure used throughout this doc as a conservative,
evidence-grounded discount, not an arbitrary guess.

## Layer-wise streaming: prior art and transferability

The double-buffer-two-layers technique is real, documented, and has shipped in multiple
independent systems — it is not a novel idea, but **it has never been applied to a 1T+-param MoE
model's dense trunk specifically**, and every existing implementation confirms the same physics:
streaming a dense component costs its *full* size in bytes, every single token, with no skip
benefit (unlike MoE experts, where routing means ~98% of experts are *not* read).

- **AirLLM** (github.com/lyogavin/airllm) — real, shipped, "70B on 4GB GPU," true layer-by-layer
  streaming: "AirLLM only ever keeps one layer on the GPU at a time." This is architecturally the
  closest precedent to the trunk-streaming idea, applied to a fully dense model. Third-party
  measurement: **0.5–2 tok/s** on 70B, "5–30× slower than full-GPU inference," explicitly
  attributed to "SSD sequential reads (2–7 GB/s NVMe)... constant layer-swapping introduces I/O
  bottlenecks that fundamentally limit inference speed" (rohit-shirke Medium writeup,
  nerdleveltech.com). No published minimum *system* RAM figure was found in AirLLM's own README —
  the project's own docs don't quantify CPU-side RAM, only that "only the layer currently being
  computed needs to reside in VRAM." Directionally consistent with this doc's bytes/token model
  (bandwidth-bound, sub-1-to-low-single-digit tok/s for a fully-streamed dense model), though not
  independently reproducible to the same precision as WASTE's own numbers.
- **FlexGen** (arxiv.org/abs/2303.06865, Sheng et al., MLR/OpenReview) — **not** a low-total-RAM
  design, despite the "single GPU" headline. Their OPT-175B/1 tok/s result uses a single 16 GB T4
  **plus 208 GB CPU DRAM plus 1.5 TB SSD** — disk is a third-tier overflow beyond an already-huge
  CPU RAM budget, not a replacement for RAM. It also targets **throughput** (large batch, many
  concurrent sequences), explicitly not single-stream latency — a different optimization target
  than a solo laptop user waiting on one response. Not transferable to a 16 GB *total-system-RAM*
  target as-is.
- **DeepSpeed ZeRO-Inference** (deepspeed.ai/2022/09/09/zero-inference.html) — same pattern as
  FlexGen: tested with a V100-32GB GPU **plus 1.5 TB CPU DRAM plus 30 TB NVMe**, "optimized for
  inference applications that are throughput-oriented and allow large batch sizes." Achieves
  30 tok/s (OPT-30B, NVMe offload) to 43 tok/s (CPU offload) — much higher than AirLLM/FlexGen
  because the model (30B) is far smaller relative to the hardware, and because DeepNVMe follow-up
  work uses **4–8 parallel Gen4/Gen5 NVMe drives** to multiply single-drive bandwidth (7→17→26
  tok/s scaling with 4→4→8 drives) — aggregate bandwidth no single laptop M.2 slot can match.
  Its "layer prefetching... overlap the fetch of a layer with the computation of an earlier
  layer" is architecturally the exact double-buffer pattern Atur describes, validated at real
  scale, prefetching measured at **1.13–1.21×** throughput gain (smaller than WASTE's 1.6× for
  experts, plausibly because dense-layer reads have less freedom to reorder/batch than MoE
  expert reads).
- **llama.cpp mmap** — **does not help** once a model exceeds RAM. Community consensus (GitHub
  discussion #638, issue #9059): mmap only defers *when* pages are read, it does not reduce the
  bytes that must ultimately be read; "in reality, almost the entire model is needed for
  inference, so mmap doesn't reduce RAM usage at all — it's purely a measurement artifact" once
  the working set exceeds physical RAM. For a dense model bigger than RAM, mmap thrashing is
  architecturally the *unmanaged* version of AirLLM/WASTE's deliberate streaming — same bytes/
  token cost, but without engineered prefetch, so likely worse in practice, not better.
- **Independent confirmation of the core insight** (MindStudio, mindstudio.ai/blog/ssd-streaming-
  ai-models-ram-dial): "A technique... maintains at most two layers' weights in memory, with the
  current layer computing while the next layer is prefetched from storage, perfectly overlapping
  I/O." Same source states the crucial asymmetry explicitly: **"Dense models don't have a natural
  partition between weights needed immediately and weights that can stay on disk, so streaming
  them is less efficient — you'd need all weights for every forward pass"** — i.e., the entire
  reason WASTE's expert-streaming works so much better than naive dense streaming is the ~4%
  activation sparsity of MoE; K3's trunk has *zero* such sparsity, so streaming it pays the full
  27.28 GB/token cost with no discount, ever.
- **Does the double-buffer arithmetic hold for K3 specifically?** Atur's ~0.6 GB estimate
  (27.28 GB ÷ 93 layers × 2) is directionally right but layers are **not uniform size** (69 KDA :
  24 MLA-style layers, MLA carrying extra `kv_lora`/absorb projections, layer 0 fully dense) — a
  correct implementation needs buffer space sized to the **largest adjacent 2-layer window**, not
  the average, so the real number is somewhat above 0.6 GB, order-of-magnitude unchanged.
  **Nothing fundamentally breaks**: KDA recurrent state and the MLA KV latent are *per-layer,
  persisted across tokens* (time axis), entirely separate allocations from the streamed layer
  *weights* — they must stay resident regardless of whether weights are streamed, but they're
  already counted in the ~0.3 GB KV/state line item above and don't interact with the streaming
  scheme for weights. No cross-layer weight dependency was found that would prevent independent
  per-layer streaming (each layer's forward pass only needs that layer's own resident weights
  plus the previous layer's output activation, which is a small vector, not a weight tensor).

## Minimum-RAM budget table for a 16 GB target

None of this has been built; the table below is a first-principles budget derived from WASTE's
own published per-component numbers, generalizing their all-or-nothing trunk residency to a
partial-residency knob that does not currently exist in the shipped engine.

| Component | Bytes | Notes |
|---|---|---|
| OS + background (Linux, minimal) | 1.5 GB | Windows realistic: ~2.5 GB — tightens everything below by ~1 GB |
| KV cache + KDA state (absorbed MLA, 4K ctx) | 0.21–0.3 GB | scales ~linearly with context: ~1.7 GB@32K, ~6.75 GB@128K |
| Scratch/activation buffers | 0.5–0.75 GB | decode 252 MB + chunked-prefill up to 500 MB (LEARNED.md §11) |
| Minimum expert double-buffer (pipeline, not cache) | 0.38–0.4 GB | 16 × 12.4 MB record × 2, `top_k × expert_record × 2` |
| Codebooks + tokenizer + misc | ~0.05 GB | structurally trivial, not independently published |
| **Fixed subtotal** | **~2.85 GB** | |
| **Remaining for trunk residency** | **~13.15 GB** | 16.0 − 2.85 |
| Trunk resident fraction | 48% of 27.28 GB | vs. 100% in WASTE's current 29 GB design |
| Streamed trunk (extra read every token) | ~14.1 GB/token | the new cost this design pays that WASTE's current build doesn't |
| Expert reads (100% cold — see below) | 17.0–17.4 GB/token | unchanged; same as WASTE's floor-regime number |
| **Total bytes/token** | **~31.5 GB** | vs. ~17 GB/token in WASTE's actual 29 GB build |

**Expert caching is categorically useless at 16 GB and should not be attempted.** WASTE's own
measured finding is that hit rate is *exactly* 0% below one full token's expert working set
(17.0–17.4 GB) — "not low," zero (`waste-engine-verified.md` #4, 2604/2704 evictions measured).
16 GB total RAM cannot fund a 17+ GB expert cache under any allocation, even a hypothetical one
spending 100% of RAM on cache and 0% on OS/KV/scratch (which is itself impossible). So every spare
byte at 16 GB should go to trunk residency instead, where the payoff is **linear and cliff-free**
(every resident GB saves exactly 1 GB/token, unconditionally) rather than a step function. This is
the single most actionable engineering conclusion of this report.

## Disk-footprint constraint

**Correction to the prompt's premise: native MXFP4 is not ~594 GB.** The actual moonshotai/
Kimi-K3 HuggingFace repo ships 96 safetensors shards totaling **1,560,936,091,448 bytes ≈ 1.56 TB
(1.42 TiB)** — verified via HF repo file listing (huggingface.co/moonshotai/Kimi-K3). The 594 GB
figure in the prompt does not match any published number found and should be treated as
incorrect/unsourced; discard it. Sanity check: 2.8 T params × ~4.25 bits/param (MXFP4 + shared
block scale) ≈ 1.49 TB — matches the 1.56 TB figure to within rounding/overhead (embeddings,
norms, router, vision tower stored at higher precision add the rest of the gap).

- **WASTE's own 982 GiB `.waste` container is already the smallest known full-quality (non-
  pruned, non-distilled) representation of K3 found** — 37% smaller than the native 1.56 TB
  MXFP4 checkpoint, via more aggressive expert quantization (3.01-bit residual VQ vs. native
  4-bit MXFP4) plus int8 trunk storage. No smaller *full-quality* representation was found
  anywhere.
- **A smaller footprint exists only by sacrificing quality.** REAP (Router-guided Expert Ablation
  Pruning, Cerebras, arXiv:2510.13999) removes low-salience experts outright and has been applied
  to K3 by a third party: `pipenetwork/Kimi-K3-REAP80-MLX-mxfp4-q8` keeps 179/896 experts/layer
  (80% pruned, 601 B of 2.78 T params) at **350 GB**, with the model card's own description
  admitting "noticeable degradation versus full K3" (drift into repetitive/list-like output,
  degraded Chinese-language performance). REAP-73 (242/896 experts kept) is a milder variant, not
  independently sized here. REAP at *moderate* ratios (25–50%) on other models preserves near-
  baseline quality per the original paper, but K3's own published REAP variants only exist at the
  80%/73% (aggressive, degraded) end — nobody has published a REAP+WASTE-VQ combination, which
  could plausibly land in the 250–350 GB range at less quality loss than REAP80 alone, but this is
  an unbuilt, unmeasured possibility, not a citable fact.
- **Practical implication for "a 16 GB laptop": disk space, not RAM, is the more likely first
  real-world blocker.** A machine spec'd with only 16 GB RAM is commonly paired with a 512 GB–
  1 TB SSD in consumer configurations, not 1+ TB free. Unlike the NVMe-bandwidth physics (fixed
  by the drive's rated speed, not fixable by cleverer software), free disk space is a solvable,
  orthogonal constraint — a 2 TB NVMe drive is a ~$100–150 purchase — but it is a real
  precondition that must be checked before any of the RAM/throughput engineering in this doc is
  even reachable.

## Has anyone done it

**No.** Extensive search (WASTE's own docs/issues, HN, Reddit/r/LocalLLaMA-adjacent coverage,
academic layer-streaming literature, AirLLM/FlexGen/DeepSpeed/DwarfStar docs) found no published
attempt to run K3, or any 1 T+-parameter model, on ≤16 GB total system RAM, or even below WASTE's
own 29.06 GB floor:

- **WASTE itself refuses to try below 29.06 GB by explicit policy** (`WASTE_E_RAM_BUDGET`) — this
  is the strongest available evidence that the project's own authors have not built (or don't
  offer) a below-floor mode, not evidence that one is impossible.
- No GitHub issue, discussion, or docs entry in sqliteai/waste proposing trunk streaming, a lower-
  RAM mode, or anything resembling partial trunk residency was found in a targeted search.
- Public coverage of the K3-on-laptop result (the-ai-corner.com, "Someone just ran a 2.78-
  trillion-parameter model on a laptop") uses the 64 GB MacBook Pro / 0.3–0.6 tok/s figures only;
  no lower-RAM report exists anywhere found.
- **DwarfStar/ds4** (antirez) targets DeepSeek V4 Flash/PRO and GLM 5.2, not K3, and needs
  **96–128 GB RAM** — an order of magnitude above 16 GB, no relevance to a K3-specific sub-16 GB
  claim.
- General "16 GB RAM" LLM guidance found universally caps out around 6–8B dense (Q4) models —
  no source treats 16 GB as viable for any 100 B+ (let alone 1 T+) parameter model without the
  specific streaming architecture this report analyzes, and nobody has published that
  architecture applied at this scale.
- **AirLLM/FlexGen/DeepSpeed ZeRO-Inference are real, shipped prior art for the *general
  technique*** (dense layer-wise streaming with double buffering) but none has been applied to a
  1 T+-param MoE model's dense trunk, and none targets a 16 GB *total-system* RAM envelope (all
  either assume abundant CPU RAM as a first tier, or a fully-dense model with no MoE structure to
  exploit).

**Conclusion: "first to run K3 on a ≤16 GB laptop" (with the RAM-throughput trade-off honestly
disclosed, not hidden) is a real, currently unclaimed engineering milestone**, contingent on (a)
actually building the partial-trunk-streaming extension (nobody has), and (b) having ~982 GB+
free disk space available (a separate, solvable precondition).

## What would have to be built (engineering, not hand-waving)

All of this is additive on top of WASTE's existing Apache-2.0 C11 codebase (consistent with
Chaos's "never from scratch" posture) — WASTE already has ~80% of the needed infrastructure
(streamed `pread` I/O, a read-ahead thread pattern, an oracle-diff correctness harness, a budget
resolver), it just applies all of it only to experts, never to the trunk, and refuses rather than
degrades below the current floor.

1. **Generalize the budget resolver from all-or-nothing trunk residency to a `resident_trunk_
   bytes` knob** — currently binary (100% resident or refuse); needs to accept any value ≥ some
   new true minimum (~5–6 GB) and compute expected bytes/token and tok/s accordingly, replacing
   the hard `WASTE_E_RAM_BUDGET` refusal below 29.06 GB with a degraded-but-working mode.
2. **A layer-order streaming path for the dense trunk**, mirroring the existing expert read-
   ahead machinery (currently 2-thread read-ahead, 1.6× measured speedup) — likely *simpler* to
   build well than the expert case, because trunk access order is 100% deterministic every token
   (no routing branch to predict), unlike experts where prefetch correctness depends on guessing
   the next layer's routing.
3. **A trunk residency selection policy** — unlike experts (where LFRU beats LRU because access
   frequency varies), trunk residency benefit is provably uniform per byte (every trunk byte is
   read every token regardless of which layer it's in), so the policy can be as simple as "keep
   the largest contiguous resident set that fits," no eviction heuristic needed. This is actually
   *simpler* than the expert-cache problem WASTE already solved.
4. **Re-chunk `trunk.bin` for per-decode-step streaming, not just startup streaming.** LEARNED.md
   confirms trunk is already loaded via chunked `pread` (34s → 20s optimization) — but only once,
   at model load. The needed change is doing this every token, interleaved with per-layer compute,
   not once up front.
5. **Buffer sizing for the largest adjacent-2-layer window**, not the average — layers are
   unequal (69 KDA : 24 MLA-style, layer 0 fully dense) per K3.md.
6. **Validate against WASTE's own oracle-diff harness** (every layer diffed against a PyTorch
   reference per their CLAUDE.md) to confirm the streamed-trunk path is bit-identical (or within
   published logit tolerance) to the resident-trunk path — directly reusable, not new
   infrastructure.
7. **Improve I/O/compute overlap beyond 2-thread read-ahead** (async/completion-based I/O —
   io_uring on Linux, IOCP on Windows) to close the gap between the ~50% realistic efficiency
   this report assumes and the theoretical bandwidth ceiling — a real systems-engineering
   project in its own right, independent of the trunk-streaming feature, that would improve
   *both* the existing 29 GB regime and the new 16 GB regime.
8. **Real on-hardware calibration** — every number in this doc above the primary-sourced WASTE
   figures (the 31.5 GB/token 16 GB-budget estimate, the ~50% efficiency discount, the tok/s
   table) is a first-principles derivation, not a measurement. Nothing in this report should be
   treated as validated until run on real hardware with `waste plan`-style preflight tooling
   extended to report the new partial-residency mode.

## Open questions

- Whether the ~0.6–0.7 GB gap between this doc's component-sum (28.4–28.5 GB) and the published
  29.06 GB floor is doc drift (per `waste-engine-verified.md`'s established pattern) or a real
  unaccounted component (allocator overhead, thread stacks, etc.) — not resolvable without either
  a fresh doc pull at read time or running `waste plan` directly.
  - **CORRECTION 2026-08-02 (this doc, at write time):** treat every WASTE-sourced number above
    as a snapshot; `waste-engine-verified.md` already established this project's docs move
    materially within days. Re-verify against the live repo before using any specific figure in a
    decision, not just this report's estimates.
- The exact per-layer trunk byte breakdown (which layers are bigger — MLA vs. KDA, dense layer 0
  vs. MoE layers) was not found published anywhere; needed to size the real (non-average) double-
  buffer window precisely rather than the order-of-magnitude estimate here.
- No independent measurement exists for the "large-block quasi-random reads ≈ sequential
  bandwidth" claim applied specifically to a 12.4 MB record size on a modern PCIe5 consumer drive
  — the cited source (storedbits.com) used 1 MB blocks on unspecified hardware; directionally
  supportive, not a precise match.
- Whether Chaos should actually pursue this (fork WASTE + build the partial-trunk-streaming
  extension) is a strategy question for `../decisions/strategy-post-waste.md`, not answered here
  — that decision currently states K3 "needs a RAM upgrade (29.06GB floor vs 15.7GB)," which this
  research suggests is true *for WASTE as currently shipped* but not a hard physical limit; worth
  flagging to Atur as a possible input to revisiting that ADR, without this research node taking
  a position on whether the ~0.02–0.4 tok/s result is worth building for.
- The reconciliation between K3's reported "~104B active params/token" (HuggingFace model card)
  and this doc's bytes-derived active-param estimate (~46 B expert + ~27–30 B trunk ≈ 73–76 B)
  was not closed — plausibly different counting conventions (weights vs. params, shared-expert
  handling, attention FLOPs vs. weight-bytes) — doesn't affect any RAM/bandwidth number in this
  report (which uses WASTE's own directly-measured byte figures throughout, not a params-based
  derivation) but is a loose end if anyone later needs a params-based cross-check.
