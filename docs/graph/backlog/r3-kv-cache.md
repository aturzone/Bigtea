---
topic: R3 — the KV cache for DeepSeek-V4-Flash, scoped against the code: exactly what state must persist, the incremental rules, the traps, and how to verify it without a new oracle
status: ready to implement
links: [next-session-handoff.md, ../research/expert-cache-is-early-not-wrong-2026-08-08.md, ../research/v4flash-compressed-attention.md]
---

R3 is the critical path. It is not one speed win, it is the unlock:

- **Generation stops being an artefact.** Today every token re-runs prefill over
  the whole sequence, so 0.015-0.064 tok/s measures the wrong thing.
- **It makes R1 pay.** A step's working set falls from **122.8 distinct experts
  per layer (~66 GiB)** to **6 (3.21 GiB)** — measured. At that size the expert
  cache goes from 2% hits to ~47% on the RAM this machine has.
- **It lifts the 256-token ceiling**, which is the same allocation.

It is also the most dangerous change in the project: **a wrong cache returns
fluent nonsense, never an error.**

## What must persist, per layer

Three things, not two. The third is the one that is easy to miss.

| state | shape | bytes (43 layers) |
|---|---|---:|
| **raw KV latents** | `kv_lora_rank × N_KV` = 512 × 256, F16 | 11.3 MB |
| **compressed summaries** | 512 × 256, F16, compressed layers only | ~10.8 MB |
| **compressor input ring** | `wide × state_rows` = 1024 × 8 (CSA) or 512 × 4 (HCA), F32 | ~1.4 MB |

**~24 MB total.** Memory is not the problem here; correctness is.

### The compressor ring is the trap

`deepseek4_forward.rs:698` says it outright:

> *The persistent ring llama.cpp maintains is not needed on a prefill:
> `state_source_idx` resolves to an appended zero row for `pos < 0` and to the
> current batch otherwise, so the ring is never read.*

On a prefill the previous window's rows are in the batch being processed. **In
incremental decode they are not.** `compressor()` builds `kv_buf` and `sc_buf`
with `state_rows` zeros at the front (line 755-768) precisely because the ring is
assumed empty. Decoding one token at a time means those rows must come from
persisted state instead, and `state_rows` is **8 for the overlapping (CSA) form
and `ratio` = 4 for the plain (HCA) one** — `wide` is `2 * head` when overlapping.

Getting this wrong does not fail. It quietly summarises the wrong span.

## The incremental rules

For a step at absolute position `p`, with `nt = 1`:

1. **Raw KV.** Compute `kv_full` for the one token, convert to F16, write into
   slot `p`. Attention reads slots `0..=p`.
2. **Mask.** Currently built for `nt` queries over `n_kv` keys with `key`
   indexing the raw cache *by absolute position* (`key > query`, and
   `query - key >= window` for `sliding_window` = 128). With one query the mask
   is a single row — but the slot/position identity must be preserved or
   rewritten in the same change.
3. **Compressor.** Blocks are `CSA_RATIO` = 4 positions. `n_blocks = nt / ratio`
   is **0** for a single token, so the compressed half is untouched on three
   steps in four and appends one summary on the fourth. Push the token's
   `kv`/`score` rows into the ring every step regardless.
4. **RoPE.** Applied at the absolute position *before* the value enters the
   cache, and the compressed entries are rotated at their **block-start**
   position with the compressed base (line 832). Cached entries must never be
   re-rotated.

## Positions are hardcoded, in three places

`pos.set_i32(&(0..nt as i32).collect())` at **line 615** (q/kv RoPE) and **line
957** (the de-rope before the output projection), plus
`(p % ratio)` at **line 740** (the compressor's within-block index) and
`comp_pos` at **846** (block-start rotation). A step at absolute position `p`
needs all four offset.

**Do not land that offset as a separate refactor.** With `pos0 = 0` everywhere
today, every existing test would still pass while the `pos0 != 0` path went
unexercised — a change that looks finished and is wrong, which is the exact
failure this codebase keeps warning about. Thread it *together with* the first
incremental step and its equivalence test, so the new path is used the moment it
exists.

## Traps, each of which is silent

- **Slot index is currently the absolute position.** Any ring or window slide
  breaks the mask's arithmetic. Rewrite both together.
- **`N_KV` = 256 is the hard ceiling** (`ArchError::ContextTooLong` now refuses
  above it). Lifting it means a real ring with wraparound, which changes the
  mask again. **Do it as a second step, after equivalence holds at ≤256.**
- **Prompt length decides which builders run.** At 2 tokens all 43 blocks fall
  back to Raw; at 5 CSA fires; at 165 HCA fires. **An incremental test at a short
  length exercises less than it appears to.** See
  `v4flash-compressed-attention.md`.
- **The compressed cache is guarded on being non-empty**, so a single-token step
  early in a sequence takes a different path from one later on.

## How to verify it, without capturing a new oracle

The handoff assumed R3 needs a fresh llama.cpp capture at two consecutive
positions. **It does not.** `prefill` is already verified against llama.cpp
element sums for all 43 blocks, so it is a trustworthy reference for itself:

```
full      = prefill(tokens[0..=n])                  # already oracle-verified
stepwise  = prefill(tokens[0..n]) then step(tokens[n])
assert stepwise == full                             # logits, exactly
```

If an incremental step disagrees with the full pass, the cache is wrong. This is
the same shape as `the_expert_cache_does_not_change_the_answer`, which caught
nothing only because there was nothing to catch.

Run it at **three lengths on purpose** — 2, 5 and 165 — because each exercises a
different attention builder, and at 165 both compressed kinds are live. Assert
argmax equality and a tight tolerance on the logit sum, and assert that the step
actually used the cache (a step that silently re-prefilled would pass).

## Suggested order

1. **State struct + equivalence test first, Raw path only** (layers 0-1, and any
   layer at ≤3 tokens). Small, and it proves the harness.
2. **HCA**, the plain compressor: one ring, no overlap.
3. **CSA**, the overlapping compressor: `wide = 2 * head`, `state_rows = 8`.
4. **Only then** the ring wraparound that lifts the 256-token ceiling.
5. Re-run the R1 benchmark. The expert cache should go from 1.9-4.1% hits to
   something near R0.1's 86% at top-64, and *that* is when R1's numbers mean
   anything.

## What it is worth

A single-token pass costs **4.0s** today, which is what a cached step will cost
before any other change — about **0.25 tok/s against llama.cpp's 0.21-0.31**, so
roughly parity on generation from this alone. Then R1 removes most of the 3.21
GiB that step reads, and R2 overlaps what is left.

**Do not quote a generation number for V4-Flash again until this exists.** Every
one measured so far is the cost of re-running the whole sequence.
