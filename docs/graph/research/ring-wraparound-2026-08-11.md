---
topic: R12 — the 256-token V4-Flash cap is gone; the raw latents are a ring and the compressed half grows
status: resolved
links: [../backlog/r3-kv-cache.md, v4flash-compressed-attention.md]
---

V4-Flash held its raw KV latents in `kv_lora_rank * 256` per layer, **indexed by
absolute position**. Position 256 wrote past the end. Issue #46.

Now: raw latents live in a 1024-slot **ring**, the compressed half **grows**, and
the only remaining limit is on a single pass — 897 tokens — which chunking
satisfies. The container declares `context_length = 1048576`; the cap was
entirely ours.

## Why a ring is exact here, and where it would not be

The raw half is not merely causal, it is **sliding**. The container declares
`deepseek4.attention.sliding_window = 128`, and the mask drops any key more than
128 positions behind its query. A position older than the window can never be
read again, so overwriting its slot loses nothing.

That is a property of *this* model, checked against the container rather than
assumed — and the code refuses `sliding_window = 0`, where raw attention would
be full causal and a ring would quietly drop keys that are still visible.

**The compressed half cannot be a ring at all.** It is visibility-limited rather
than windowed: a token sees *every* block that is complete and behind it, so
nothing in it ever becomes unreachable. At ratio 4 that is one slot per four
tokens, which is what a long context costs and why it is grown on demand instead
of capped.

That asymmetry is the whole design. Two structures, two different reasons, and
treating them alike either caps the context or wastes memory.

## Three position-indexed structures, not one

| structure | indexed by | what it needed |
|---|---|---|
| `raw` | absolute position | **ring**, `position % 1024` |
| `comp` | **block** index, not position | **growth** — never a ring |
| `ring` (compressor input) | `pos0`-relative | already correct, untouched |

The compressor's input ring is the one the scoping warned was easy to miss. It
turned out to need nothing: it is addressed as `state_rows + q - pos0`, relative
to the batch, so it never indexed by absolute position in the first place.

## The size is the window plus the batch, not the window

A pass covers queries `pos0 ..= pos0 + nt - 1`, and its **earliest** query still
reaches `window - 1` positions further back. So `window + nt - 1` positions must
be live at once — measured from `pos0`, not from `hi`.

Getting that from `hi` instead would drop exactly the keys the first rows of a
prefill need, and short attention is fluent nonsense rather than an error. It
has its own unit test for that reason.

1024 slots leave room for an 897-token batch, past any prefill block the runner
uses (`-b` defaults to 256). Beyond it `forward` refuses, reporting the **batch**
limit rather than the internal span, because chunking is what a caller can do
about it. Memory: 512 latents x 1024 x 2 bytes x 43 layers = **45 MB**, against
11 MB before.

## The mask had to be rewritten with the ring

The old key axis *was* the slot index, because slot and position were the same
number. They are not any more. The key axis is now a gathered run of absolute
positions `lo..=hi`, read out of the ring in position order, and the mask indexes
into that run.

Handing the mask slot indices instead would attend to whatever `p % 1024`
happened to hold. The cache's own doc comment predicted this — *"what a ring with
wraparound would break, so the mask and any future ring must be rewritten
together"* — and it was right.

One incidental win: the compressed half now contributes `comp_len` keys rather
than a fixed 256, so short sequences build a smaller mask and a smaller K tensor
than before.

## Verification

**End to end, past the cap.** `prefill(0..=257)` against `prefill(0..257)` then
`step(257)` — the same equivalence harness as the rest of R3, which needs no new
capture because `prefill` is already verified against llama.cpp's element sums:

```
past 256: argmax 91 agrees; sums 350740.59 vs 352047.19 (0.373% apart)
test past_the_old_256_cap_a_cached_step_agrees_with_a_full_prefill ... ok
```

Deliberately not bit-identical: routing flips on near ties when the batch shape
changes, so 0.373% is the expected residual and an equality assertion would fail
on correct code.

**The arithmetic, exhaustively and for free.** `raw_span` is a pure function with
its own unit tests: the reach-back measured from `pos0` and not `hi`, a step
needing exactly one window, spans that wrap the ring, the batch limit and one
past it, and — the safety argument itself — that no two positions in one span
ever share a slot.

All 22 container-backed tests pass, at 2, 5, 165 and 258 tokens, which covers
Raw, CSA and HCA: prompt length decides which attention builder runs, so a test
at one length verifies a path it did not change.

## Left stale, deliberately — and closed since

`bigtea-serve.rs` reported `context_limit() = 256` for deepseek4 when this was
written, so the **server refused sequences the engine had started handling**.
That file belonged to another session and was not touched here.

**Closed 2026-08-11 in `9f024e7`, merged at `7a81502`.** It reports 897 — the
per-pass cap, which is what the ring left as the only limit.

Worth recording *how* that was nearly got wrong twice: this note outlived the
fix, and a later session repeated "still reports 256" from the note rather than
from the file. **A stale note reads exactly like a current fact.** Check the
file.
