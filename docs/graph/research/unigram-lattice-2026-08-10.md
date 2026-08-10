---
topic: A6b — Unigram for the T5 family, why it is not the SentencePiece already implemented, and the one normalization step deliberately left out
status: resolved
links: [../backlog/lts-parity-criteria.md, wordpiece-spelling-2026-08-10.md]
---

`tokenizer.ggml.model = "t5"` now loads and tokenizes identically to llama.cpp
on `flan-t5-small`, five cases, token for token. With WPM that completes A6's
implementation half.

## Unigram is not SPM with different scores

Both come from SentencePiece, both spell a boundary `▁`, and both read
`tokenizer.ggml.scores`. It is very easy to route `"t5"` at the existing SPM
path, watch it produce valid ids, and ship it. They pick tokens by opposite
methods:

- **SPM/BPE is constructive and greedy.** Start from characters, repeatedly
  merge the best-scoring adjacent pair. Every decision is local and final.
- **Unigram is selective and global.** The vocabulary is a probability model;
  every possible cutting of the string is a candidate and the answer is the one
  whose scores sum highest — a shortest path over a lattice, solved with
  Viterbi.

The difference is not cosmetic. A locally worse split routinely wins because of
what it enables later, which greedy merging can never recover:

```
vocab   ab -1    a -2    bc -2    c -9
greedy  ab then forced onto c   -> -10
Viterbi a  then bc              ->  -4   <- what T5 was trained on
```

Both produce in-range ids and fluent-looking output. That case is a unit test.

## Three details that decide exact agreement

1. **`USER_DEFINED` tokens score 0**, not their stored score. Real scores are
   negative log probabilities, so zero beats any ordinary segmentation covering
   the same span. This is how T5's `<extra_id_*>` sentinels survive intact
   instead of being cut into `<`, `extra`, `_`, `id`.
2. **Path sums accumulate in `f64`.** llama.cpp says outright that this is what
   makes its output identical to the HuggingFace tokenizer; `f32` drifts and
   flips near-ties on long inputs.
3. **The unknown penalty is `min_score - 10.0`**, charged per *codepoint*, not
   per byte. A 4-byte emoji is one `<unk>`.

## What the oracle pinned

`llama-tokenize -m flan-t5-small.Q8_0.gguf -p "<text>"`:

| text | ids | what it pins |
|---|---|---|
| `The capital of France is Paris.` | `37 1784 13 1410 19 1919 5` | ordinary lattice path; `.` its own token |
| `translate English to German: How old are you?` | `13959 1566 12 2968 10 571 625 33 25 58` | T5's actual task prefix |
| `tokenization` | `14145 1707` | `▁token` + `ization`, a split greedy merging gets right by luck and Viterbi gets right by construction |
| `Hello, World! 42` | `8774 6 1150 55 6426` | punctuation, and `▁42` as one token |
| `  spaced   out  ` | `628 26 91` | whitespace runs collapse; **trailing whitespace adds no token** |

**`llama-tokenize` prints no trailing `</s>`** although the container declares
`add_eos_token = true`. Bigtea honours the container — T5's encoder input ends
with `</s>`, which is how it was trained — so the expectations are the oracle's
ids plus id 1, and the difference is written into the test rather than absorbed
by loosening it.

## A detokenizer bug this surfaced

SPM's decoder strips the dummy prefix only when the id list *starts with BOS*,
which is its evidence that it is looking at a whole sequence rather than one
streamed token. **T5 has no BOS at all** (`add_bos_token = false`), so the test
never fired and every decoded T5 sequence kept a leading space — `" The capital
of France is Paris."`.

A terminating EOS is the same evidence for a family that brackets the other way,
so the check now accepts either. A single generated token is neither, so
streaming still gets `" The"` with its space intact — the property that made the
original rule conditional.

## What is deliberately not claimed

**The precompiled charsmap is not applied.** T5 containers ship
`tokenizer.ggml.precompiled_charsmap`, a serialised NFKC automaton — 237,539
bytes in `flan-t5-small` — and llama.cpp walks it before normalization. Here,
normalization is the rest of the rule: whitespace to `▁`, runs collapsed, a
boundary prepended.

For text already in normal form, which is all ASCII and nearly all ordinary
prose, the charsmap is the identity, and every oracle case above passes without
it. Input that is *not* in normal form will diverge:

```
Ｈｅｌｌｏ   fullwidth forms      -> llama.cpp folds to "Hello", we do not
ﬁ          ligature             -> llama.cpp folds to "fi"
②          compatibility digit  -> llama.cpp folds to "2"
```

That is a genuine gap, not a rounding difference: those inputs would tokenize
differently and the model would see a stream it was not trained on. Implementing
it means decoding the DARTS automaton in the container, which is a self-contained
piece of work and the obvious next step if a T5 model is ever put in front of
user-supplied text. It is recorded here rather than discovered later.

## Cost

49 unit tests and 5 oracle cases; 252 workspace tests pass, `clippy -D warnings`
and `fmt` clean. One real bug found in existing code (the BOS-only whole-sequence
test), caught by a round-trip assertion rather than by reading output.
