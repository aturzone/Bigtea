# 04 — Strategy options for Bigtea (post-WASTE)

Context in one line: Bigtea wraps engines (llama.cpp, ktransformers, SGLang) and never forks
(ADR fork-vs-wrapper); WASTE and ds4 *are* engines. So the strategic question is not "beat
WASTE," it is "where is the wrapper lane still defensible, and what is now noise?"

## Option A — Lean into ktransformers observability (T4/T5) as the defensible core

- **Why now:** WASTE made MoE cache hit-rate *the* number of the month, but nobody ships it
  **mainline for ktransformers**, and ktransformers is exactly the engine whose expert placement
  is a black box. llama.cpp's version is branch-only, already deemed "too large" once, and lives
  in a different engine dialect.
- **The move:** make T4 (placement introspection) + T5 (hit-rate-by-position breakdown) the
  flagship pair, shipped mainline against ktransformers, with the working-set-multiple lens from
  `02` (L1) so the output is comparable with WASTE's published curve — not a mystery number.
- **Effort vs value:** highest value/effort of anything Bigtea can do this quarter; it turns a
  fork-verification blocker (T0) into a product feature.

## Option B — Claim the cross-engine wrapper lane (the unclaimed ground)

- **Why now:** WASTE, ds4, llama.cpp, FATE each solve *one* engine. **Nobody** offers a
  cross-engine probe → recommend → launch → bench with a **pre-download speed prediction** and a
  benchmark schema keyed to working-set multiples. Bigtea's planned CLI (probe/recommend/launch/
  bench, wrapper-core) is already this shape.
- **The move:** benchmark-harness T2 + T6 and gap-closure T2 become the "why choose Bigtea"
  demo: one command, any engine, same working-set-multiple report. The recommend step (T2) is the
  differentiator — prediction *before* you download, not diagnosis after.
- **Risk:** this is a tooling value-prop that only lands if the measurements are trusted. That
  pushes protocol rigor (L3) to the top of the priority stack.

## Option C — Record WASTE/ds4 as future wrapped backends (v1.1+), don't act now

- **Why:** both are Apache-2.0/MIT-adjacent, CLI/API-friendly, and already do the disk-streaming
  lane Bigtea would never build. Wrapping them later costs nothing now; the ADR's dynamic-tracking
  rule keeps the option warm.
- **The move:** add a line to the future-backend research queue (in `05-actions.md`), with a
  trigger: revisit when (a) v1 ships, or (b) either engine gains a Linux/Metal/Windows support
  matrix Bigtea can actually target.

## What NOT to chase (the "don't" list)

- **The frontier-Mac lane.** ds4 owns the "128 GB Mac streams the biggest model" story and has
  13k stars; WASTE owns the K3-on-64GB story. Bigtea's mvp is Linux+NVIDIA. Do not reposition to
  compete for this audience.
- **From-scratch engine work.** ADR already decided; WASTE's existence reinforces it (an
  embedding-grade engine is 6k LoC of systems engineering — that's a company, not a milestone).
- **Batching/spec-decode or predictive-prefetch features.** Refuted or regime-bound in `01`/`03`;
  rebuilding them would burn tickets against measured dead ends.
- **KV-offload as the headline.** MLA already solves KV cache (L9); the honest headline for a
  DeepSeek-class mvp is expert-cache economics.
- **Chasing star-count.** Traction here is narrative; Bigtea's winnable game is *expertise* in
  the wrapper layer, which is invisible to HN and durable.

## Risk register

| Risk | Severity | Trigger to revisit | Mitigation now |
|---|---|---|---|
| llama.cpp `cached-experts-v2` merges to mainline | Med | Merge of #24524/#24528 | Benchmark schema already regime-aware; T4/T5 targets ktransformers, not llama.cpp |
| ds4/WASTE wrap-each-other or add wrapper features | Low | ds4 adds cross-engine CLI | Wrap decision is v1.1+, cheap to defer either way |
| ktransformers releases its own observability | Med | ktransformers repo shows placement metrics | T4/T5 velocity; first-to-mainline matters more than first-to-idea |
| FATE-style prefetch arrives for disk lane | Low | Prefetch shown paying on disk | Keep WASTE's refutation on file; only re-open with a measured counter-example |
| "Wrapper" narrative commoditized by CLI-tool frontends | Low | A one-command cross-engine tool gains traction | Depth of measurement protocol is the moat, not the CLI |

## The strategy in one paragraph

Ship T4/T5 as the ktransformers-first observability product with a working-set-multiple lens
(Option A); let the same lens power the benchmark + recommend story that is the wrapper lane's
reason to exist (Option B); keep WASTE/ds4 as warm v1.1+ wrapped backends and record the refuted
levers so Bigtea never rebuilds them (Option C). Treat WASTE not as a rival to out-engine but as
a calibrated, public oracle for the exact problem Bigtea is best positioned to *operationalize*.
