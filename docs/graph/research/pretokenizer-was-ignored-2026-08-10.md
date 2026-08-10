---
topic: A6c — `tokenizer.ggml.pre` was read by nobody, so every BPE container was split with DeepSeek's rule
status: resolved
links: [../backlog/lts-parity-criteria.md, wordpiece-spelling-2026-08-10.md]
---

`tokenizer.ggml.pre` names the rule that splits text before BPE runs. BPE never
merges across a split boundary, so it decides which merges are even *possible*.
It was **ignored entirely**: one hand-written splitter, DeepSeek-V4-Flash's
`joyai-llm`, was applied to every byte-level BPE container regardless of what the
container asked for.

## What that cost, measured

`llama-tokenize` on the two containers in this repository that declare different
variants:

| input | `qwen2` (Qwen3-4B) | `llama-bpe` (Llama-3.2-1B) | ours, before |
|---|---|---|---|
| `4567` | `4` `5` `6` `7` | `456` `7` | `456` `7` |
| `12345678` | eight tokens | `123` `456` `78` | `123` `456` `78` |
| `don't` | `don` `'t` | `don` `'t` | `don` `'` `t` |

Two independent errors:

- **Digits.** Qwen takes one digit at a time; Llama groups up to three. Every
  number in a prompt tokenized wrongly on Qwen, and because the pieces after it
  shift, so did everything following the number.
- **Contractions.** Both reference implementations match `'s 't 're 've 'm 'll
  'd` as one piece, case-insensitively. The old splitter had no contraction rule
  at all, so `don't` became three pieces where llama.cpp makes two — on *both*
  families.

Neither produced an error. Both produce valid ids. The model answers fluently
from a stream it was not trained on, which is the failure this crate's header
warns about and the same shape as the WordPiece spelling bug found earlier today.

## What is implemented, and what is refused

Three variants, each checked against a real container:

| `pre` | container | rule |
|---|---|---|
| `llama-bpe`, `llama3` | Llama-3.2-1B | contractions, `\p{N}{1,3}` |
| `qwen2` | Qwen3-4B | contractions, **one digit** |
| `joyai-llm` | DeepSeek-V4-Flash | CJK rule, no contractions, `\p{N}{1,3}` |

Everything else is **refused by name** — `deepseek-llm`, `falcon`, `default`,
`smaug-bpe` and the rest of llama.cpp's list. There is no container here to check
them against, and the table above is exactly why guessing is not acceptable: a
plausible-looking rule shifts every boundary silently. The error says which
variant, what is implemented, and what adding one needs.

This follows A8's rule for architectures — refuse rather than answer wrongly —
and it is a deliberate behaviour change: a `default` gpt2 container that loaded
before will now be refused. It was being split with DeepSeek's rule, so it was
already producing wrong ids; failing is strictly better than that.

The variant is only consulted for byte-level BPE. SentencePiece, WordPiece and
Unigram do their own splitting, so an unfamiliar `pre` on one of those is not a
reason to refuse the model — Phi-3 and Gemma-2 both declare `pre = "default"` and
are unaffected.

## Why it is hand-written

The patterns need negative lookahead (`\s+(?!\S)`) and case-insensitive
alternation, and the workspace has no external dependencies. Each variant is an
ordered list of rules tried at each position, which is what an alternation *is* —
so the code mirrors the regex rather than reinterpreting it. Rule order carries
meaning: the contraction rule has to be tried before the punctuation rule, or
`'t` is taken as `'` and then `t`, which is precisely the old bug.

## Verification

- 6 oracle cases per variant on Qwen3-4B and Llama-3.2-1B, token for token,
  including digits, contractions, punctuation, newlines and a line of code.
- A test asserting the two variants **disagree** on `4567`, so a future change
  that collapses them back into one rule fails loudly.
- V4-Flash round-trips unchanged — it is the container the original splitter was
  written against and the regression that mattered most.
- Losslessness across all three variants: splitting never alters or drops input.

## Cost

52 unit tests + 5 real-container tests; 283 workspace tests pass, `clippy
-D warnings` and `fmt` clean.
