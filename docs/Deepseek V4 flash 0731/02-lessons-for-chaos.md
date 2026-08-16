# 02 — Lessons for Chaos (map to tickets)

Every lesson below is drawn from `01-waste-deep-dive.md` and ends at a concrete Chaos ticket or
graph node. Where the target is a ticket, I name the epic + ticket id from the backlog nodes so
this stays a pointer, not a rewrite. **Nothing here was applied to the graph** — that is
`05-actions.md`'s job, pending Atur's approval.

## L1. The cache floor is one token's working set — make it a first-class metric

WASTE's most predictive finding: below a cache of one token's working set, MoE hit rate is
exactly zero; above a whole multiple of it, the OS paging cliff turns hits into page faults
(46→58 GB budget = 0.32→0.04 tok/s on K3). This is *the* physical model for MoE offload on any
engine, and it is precisely the effect Chaos's benchmark harness was designed to observe.

Targets:
- `backlog/benchmark-harness.md` **T6** (static-placement cache-hit-rate reporter): the report
  must plot hit rate **against budget expressed as working-set multiples** (floor, 1×, 2×, 3×),
  not against absolute RAM or "percent of model." A single number (e.g. "46% hit") is
  uninterpretable without the multiple.
- `backlog/benchmark-harness.md` **T2** (run protocol): record `working_set_bytes` (computed by
  probing the engine) and `budget_multiple` on every run record, so rows are comparable across
  machines and across engine versions.

## L2. Budget is a step-down resolver with a refusal floor, not a fill-the-machine default

WASTE's budget resolver: refuse a budget below the floor; cache is only worth whole multiples of
the working set; pick the largest multiple that fits under 7/8 of physical RAM. Their original
"fill the machine" default quietly sat *inside* the paging cliff out of the box.

Targets:
- `backlog/gap-closure.md` **T2** (split/offload auto-recommender): adopt the deterministic
  step-down-from-multiples rule. This is also a **correction to the `--fit` / auto-split
  heuristic that Chaos's research node flags as buggy** — a working-set-multiple rule is the
  fix we were looking for, and now it has a published measurement backing it.
- `backlog/wrapper-core.md` **T3** (launch-flag assembly): the resolver's output maps cleanly to
  flags per engine (llama.cpp `--fit`-style, ktransformers expert placement) — same rule, engine
  dialect.

## L3. Measurement protocol: sweep upward, refuse rows after paging, record machine state

WASTE's discipline: ascending sweeps only (a row taken after a paging row is void), machine
state recorded with every run, the budget validated as within real RAM before trusting numbers.
This validates and sharpens the protocol already planned in `benchmarking-methodology.md` and
`backlog/benchmark-harness.md` **T2**.

Targets:
- `backlog/benchmark-harness.md` **T2**: require ascending order, mark post-paging rows invalid,
  and store (RAM total, free-at-start, disk free, whether the model volume is NVMe or USB) in the
  run record. WASTE's "18%↔60% system noise" floor is a good bar for our 5% CV gate.

## L4. SSD is a real tier — the "SSD only gates load time" claim is now false

WASTE streams a 982 GB model at 0.5 tok/s; 55% of a K3 token is expert I/O; the *enclosure* is
the ceiling (12.78 GB/s internal NVMe vs 0.94 GB/s external bridge). This directly contradicts a
claim in `research/hardware-profiling.md` (line ~31, that SSD speed only affects load time) that
was already suspect. Streaming MoE turns the model volume into a live I/O budget.

Targets:
- `research/hardware-profiling.md`: flag the line-31 claim for correction (put on the graph's
  task list; I did not edit it directly).
- `backlog/hardware-profiler.md` **T3** (device probing): run a fio-style read-bandwidth probe on
  the *actual model volume* (not the system drive) and surface enclosure/bridge bandwidth as a
  first-class "is streaming viable" signal.
- `backlog/hardware-profiler.md` **T6** (mixed-tier placement): split recommendations must branch
  on SSD tier — streaming experts is only viable above ~X GB/s and at MoE working sets; below
  that the recommendation is RAM-only or refuse. This is the "SSD-condition" branch.

## L5. Never dequantize on the streaming path; never rebuild the refuted levers

WASTE's measured refutations are a buy-list for Chaos's *research* node on offload
methodologies — we should adopt the numbers, not rediscover them:
- Per-expert bit allocation: flat (1.01× across layers) — do not build an optimizer that
  allocates bits by routing frequency.
- Cross-layer predictive prefetch: recall 29.0% does not beat the previous-token set (29.5%) —
  do not propose it as a split/placement feature.
- Batching/spec-decode to dodge expert reads: ceiling 1.63×, doesn't compose with read-ahead —
  not a roadmap item for gap-closure.
- 3-bit trunk: logit collapse; K3's QAT covered experts only — trunk quantization quality is a
  *model* property, not an engine trick.

Targets: `backlog/gap-closure.md` — add a "known-refuted (WASTE, 2026)" note so our gap research
doesn't regenerate these ideas; `research/` node on MoE quantization gets the 3-bit-trunk
warning.

## L6. The decode-ceiling model must include an I/O term

WASTE: expert I/O is 54.8% of a K3 decode step — reads are still *twice* the arithmetic even
with read-ahead. Chaos's `gap-closure` T4/T5 observability work models "decode ceiling" as a
bottleneck decomposition; it must include disk/SSD read bandwidth (and the O_DIRECT-vs-paged
distinction), not just RAM and VRAM bandwidth terms.

Targets: `backlog/gap-closure.md` **T4/T5** — the breakdown visualizer/report should include an
I/O line for streaming backends; the "I/O-bound vs compute-bound" axis belongs in the benchmark
schema (see `03-landscape-2026.md` §regime).

## L7. Measurement culture as a project asset

WASTE's append-only, dated, refutation-keeping LEARNED.md is a *competitive* asset — it's why the
project is trusted within a week. Chaos's graph already has the right instincts (recorder gate,
5% CV gate, decisions with amendments). The delta: record refuted ideas **with the number that
killed them**, and date every entry (the graph already commits dates on decisions; LEARNED-style
notes in research nodes should follow).

Targets: propose a "decisions/ convention: dated, append-only, refutation numbers kept" in
`05-actions.md`.

## L8. "A test on the small model does not test the big one"

WASTE's budget checker ran green for weeks on Kimi-Linear because its scratch is megabytes; the
real floor was 30.38 GB, not the planned 29.69. This is a direct endorsement of Chaos's
**T0 gate** (refuse to run v1 without the capable rig) and of `mvp-v1.md`'s "hardware-dependent
tail" — and a warning that wrapper smoke-tests on small models are necessary but never sufficient
for split/budget logic.

Targets: `backlog/mvp-v1.md` — keep T0-gate wording; add "budget/split logic must be validated on
a real MoE container, not the smoke fixture."

## L9. The KV story: latent-MLA compression is where MoE cache wins get eaten

K3's MLA absorbed KV (53× less cache, identical logits to 1.2e-05) is why long context works at
all on small RAM. For a DeepSeek-class mvp-v1 (MLA architecture), this means KV cache is *not*
the bottleneck WASTE had to solve for non-MLA models — and it should **not** be the centerpiece
of Chaos's offload pitch to users. The MoE expert cache is the centerpiece; KV is already
solved by the architecture.

Targets: `backlog/gap-closure.md` **T2** — the split/offload explanation text should lead with
expert-cache economics, not KV offload.
