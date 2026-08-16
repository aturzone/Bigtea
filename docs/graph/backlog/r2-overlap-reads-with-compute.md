---
topic: R2 — overlapping disk reads with compute on V4-Flash, re-scoped against the code on 2026-08-11
status: DONE 2026-08-11 -- built, measured 1.07x prefill / 1.13x generation, on by default. Result: ../research/r2-overlap-2026-08-11.md
links: [next-session-handoff.md, ../research/the-plateau-was-ours-2026-08-10.md]
---

**Implemented 2026-08-11.** The result, and the reason it reached a third of
the ceiling rather than all of it, are in
`../research/r2-overlap-2026-08-11.md`. This node is the scoping that preceded
it; the handle-pool constraint below is the thing that had to be handled first,
and it was.

Ceiling ~1.4x: per block roughly **53 ms of read against 23 ms of compute**.

## The constraint nobody wrote down: the handle pool is already fully used

`Model::read_range_into_via(name, offset, dst, slot)` says it plainly:

> Concurrent readers must pass **distinct** slots. Sharing one handle serialises
> them in the OS and holds the drive at queue depth 1.

That is the finding that was worth 1.32x on expert reads
(`the-plateau-was-ours-2026-08-10.md`). But:

```
crates/chaos-model/src/lib.rs:158   const READER_HANDLES: usize = 8;
crates/chaos-arch/src/deepseek4_forward.rs:542   const READERS: usize = 8;
```

**The expert read already uses all eight.** `read_expert_slices` fans out over
`slots[j % READERS]` and `prefetch_dense` over `(0..READERS)`. So a background
prefetch thread started naively would pass slots the foreground read is already
using, and the two would **serialise on the same handles** — reintroducing by
hand the exact bug that was fixed, and it would show up as "overlap does not
help" rather than as an error.

Any implementation therefore has to *partition* the pool, which means the
overlap A/B is confounded with an expert-read A/B and both must be measured:

| configuration | expert readers | prefetch readers |
|---|---:|---:|
| today | 8 | — (serial, same 8) |
| split | 4 | 4 |
| widened pool | 8 | 8 (needs `READER_HANDLES` 16) |

The earlier bench makes the split look affordable — **4 handles measured 2.65
GiB/s against 2.69 at 8**, so the expert side gives up ~1.5% — but that was a
pure-read benchmark, and this project has already been burned once by a kernel
benchmark that did not survive contact with the real path
(`batch-the-expert-matmuls.md`). Measure it, do not assume it.

## What can be overlapped exactly, and what cannot

Routing is data-dependent: block N+1's experts are chosen from block N's output,
so they **cannot** be known before block N computes. Three seams survive that:

1. **Block N+1's dense (always-read) tensors.** These do not depend on routing
   at all, so this is exact, needs no prediction, and is the largest safe win.
   It pays **only when residency is short** — `prefetch_dense` skips anything
   already resident and returns an empty map when fewer than two tensors are
   missing. Measured worth when 3.1 GiB was missing: 2.15 s/token of dense
   re-reads, 39% of a token. With the always-read set fully resident it is
   worth exactly nothing, so **state the free RAM with any number produced
   here** — this is the axis V4-Flash figures drift along.
2. **Layers 0-2 route by token id** (`ffn_gate_tid2eid`, a `get_rows` on a
   table), so their experts *are* knowable before any compute runs. Zero
   speculation, 3 of 43 blocks — the right place to prove the machinery.
3. **Layers 3-42 on the previous token's routing.** Speculative; a miss costs a
   wasted read, not a wrong answer. This is where the bytes are, and R0.1's
   coverage figure is what sizes it.

## Shape

`block()` has no callers outside `forward()` and is not exported, so its
signature is free. The natural form is `std::thread::scope` per iteration:
spawn block N+1's prefetch, run block N on the main thread, join. A prefetch
that fails must fall back to reading inside `block()` — it is an optimisation,
and it must not be able to end a run.

Cost to hold: one block's dense tensors, ~147 MiB, alive while the previous
block computes. On a machine that is already short of RAM that byte competes
with residency, which is the same trade the expert cache lost
(`--cache` is refused while the always-read set is still streaming). **It should
default off and be measured before it defaults on.**

## Verification

`cargo test --release --test deepseek4_forward -- --ignored` (21 tests) is the
correctness gate — the overlap must not change a single element sum. Then a
back-to-back A/B **in one session with the free RAM recorded on both sides**;
absolute V4-Flash numbers drift a lot with page-cache state and only same-session
comparisons count.
