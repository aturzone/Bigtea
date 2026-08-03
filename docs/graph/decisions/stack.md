---
decision: implementation stack for Bigtea
status: accepted        # Atur, 2026-08-03
links: [k3-run-path.md, ../research/fixed-hardware-design-space.md, fork-vs-wrapper.md]
---

## Context

Bigtea's goal is **to be the runner** that makes frontier MoE models execute on
low-spec machines — not a front-end that shells out to someone else's runner.
That reframing (Atur, 2026-08-03) supersedes the narrower "wrapper/advisor"
framing in `strategy-post-waste.md`.

A runner for models larger than RAM is a **memory and I/O problem**, not a math
problem:
- page-aligned buffers for cache-bypassing reads (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`)
- explicit residency: decide what stays in RAM, forever, per byte
- a bounded expert cache with an eviction policy we own
- prefetch threads overlapping I/O with compute
- a hard memory budget that is *enforced*, never exceeded

## Decision

**Rust for everything we own. C (`ggml`, via FFI) for the compute we borrow.**

The division is the whole bet: **borrow the math, own the memory.**

- **Rust** — the streaming loader, residency policy, expert cache, budget
  resolver, prefetch pipeline, GGUF parsing, CLI. Gives byte-level control of
  layout and alignment (a hard requirement for `O_DIRECT`, which demands
  aligned buffers and sector-multiple lengths), no GC pauses in the token loop,
  a single static binary, trivial cross-compilation, and it is Atur's language.
- **`ggml` via FFI** — matmuls, quantized dot products, SIMD kernels. Writing
  these is years of specialist work, already done well, and re-doing it is not
  where the contribution is.

## Options rejected

- **Python** (initially chosen, ~400 lines written and thrown away): correct
  only under the false premise that Bigtea merely orchestrates. It cannot own a
  token loop, cannot control buffer alignment, and forces users of a
  *low-resource* tool to install a runtime first. The validated architecture
  math it produced ports directly — that was the real asset, not the code.
- **Go**: fine for orchestration, wrong for this. GC and weaker control over
  memory layout/alignment fight exactly the part we must own. Both comparable
  engines (WASTE = C11, llama.cpp = C++) are unmanaged for this reason.
- **C/C++**: the incumbent choice and a legitimate one — but Rust gets the same
  control with memory safety in code that is, by construction, full of raw
  offsets, alignment arithmetic and threads. Not Atur's primary language either.
- **Writing our own kernels**: rejected. Not the contribution, and a multi-year
  detour.

## Where the contribution actually is

Verified in `../research/fixed-hardware-design-space.md` and
`k3-on-16gb-feasibility.md`: every existing engine either assumes RAM ≈ model
size, or streams with a policy tuned for a *different* constraint. Nobody has
built the memory/streaming layer for the **≤16 GiB** tier, where:

- the **dense part must be resident** (it is re-read every token; residency pays
  back linearly and cliff-free), and
- **expert cache is worthless below one token's working set** (hit rate collapses
  to zero — measured, not theorised), so spare RAM goes to residency instead, and
- **bits/param is the only lever that improves fit *and* speed at once.**

That policy layer is what Bigtea owns.

## Consequences

- Workspace of small crates, each independently testable:
  `probe` (hardware), `gguf` (container parsing), `plan` (architecture math +
  prediction), `io` (aligned/bypassing reads + prefetch), `cache` (residency +
  eviction), `engine` (token loop), `cli`.
- **Milestone order is chosen so each step is provable before the next**:
  probe → parse a real GGUF → predict → stream → run. The GGUF metadata shard is
  5 MB, so container parsing is testable against a real DeepSeek-V4-Flash file
  long before the full 144 GiB has landed.
- `ggml` FFI is deferred until the streaming layer works; until then the engine
  is validated on byte-level correctness (does it read the right tensor bytes),
  not on generated tokens.
- Python remains for **model conversion only** (`convert.py`, torch,
  safetensors) — inescapable, and we invoke it rather than reimplement it.
