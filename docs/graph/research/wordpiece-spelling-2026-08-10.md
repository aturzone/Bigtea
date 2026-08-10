---
topic: A6a — WordPiece for the BERT family, and the vocabulary spelling that makes a textbook implementation produce fluent nonsense
status: resolved
links: [../backlog/lts-parity-criteria.md]
---

`tokenizer.ggml.model = "bert"` now loads and tokenizes identically to
llama.cpp, verified token-for-token on `all-MiniLM-L6-v2`. That unlocks the
BERT/embedding family, which is A6's first half.

## The finding worth keeping

**GGUF does not store WordPiece the way WordPiece is written down.**

HuggingFace's `vocab.txt` spells a continuation piece `##ization`. Every
description of the algorithm, including the ticket that asked for this one, says
"`##` continuations". The GGUF converter rewrites the whole vocabulary into
SentencePiece spelling instead:

```
HuggingFace        GGUF (all-MiniLM-L6-v2)
capital            ▁capital        <- word-initial gets a marker
##ization          ization         <- continuation is bare
```

In that container the strings `capital` and `##ization` **do not exist**. A
correct-looking `##` implementation matches nothing, and because WordPiece has
no byte fallback, every ordinary word collapses to one `[UNK]`:

```
want:  the capital of france is paris .
got:   the [UNK]   of [UNK]  is [UNK] .
```

No error. Every id valid and in range. The model would have produced a fluent
continuation of that, and the obvious suspect would have been the forward pass.
This is the same class of bug as Gemma-2 answering `"himſelf"` — and it was
caught only because the test asserted llama.cpp's exact ids rather than "looks
plausible".

Both spellings are supported and the one in use is **detected from the
vocabulary** (`does any token start with ▁`), because a container built directly
from a `vocab.txt` genuinely does use `##`.

## The rest of the rules, all pinned by an oracle case

`llama-tokenize -m all-MiniLM-L6-v2.Q8_0.gguf -p "<text>"`, thirteen cases in
`tests/wpm_real.rs`. Each exists because it can only pass if one rule is right:

| text | ids | what it pins |
|---|---|---|
| `The capital of France is Paris.` | `101 1996 3007 1997 2605 2003 3000 1012 102` | lowercasing; `.` split off; `[CLS]`/`[SEP]` added although the container declares neither flag |
| `tokenization` | `101 19204 3989 102` | `▁token` + `ization` |
| `café naïve` | `101 7668 15743 102` | accents **dropped**, not turned into `[UNK]` |
| `北京大学` | `101 1781 1755 1810 1817 102` | one word per CJK character |
| `hello 🦄 world` | `101 7592 100 2088 102` | an uncoverable word is **one** `[UNK]`, and its neighbours are unaffected |
| `Ω≈ç√` | `101 1179 30133 2278 30127 102` | non-ASCII symbols do **not** split |
| `3.14159` | `101 1017 1012 15471 28154 102` | digits split on punctuation, not per digit |

`Ω≈ç√` is the one that pins the splitting rule. A codepoint starts a new word if
it is punctuation, **or an ASCII symbol**, or CJK — a non-ASCII symbol does not.
`≈` and `√` stay inside the word and come out as continuation pieces. Splitting
on every symbol yields four standalone tokens, looks entirely reasonable, and is
wrong.

## Two defaults that are not in the container

`all-MiniLM-L6-v2` declares no `add_bos_token` and no `add_eos_token`, and
llama.cpp still wraps every sequence in `[CLS] … [SEP]`. Both default to on for
WordPiece. A missing `[CLS]` shifts every position by one.

## What is deliberately not claimed

- **Accent stripping is Latin-only.** NFD needs a Unicode table the workspace's
  no-dependency rule forbids, so decomposition is a table over Latin-1
  Supplement and Latin Extended-A, plus dropping combining marks so
  already-decomposed input behaves the same. Arabic harakat and Devanagari
  matras are left alone where llama.cpp would strip them. Those codepoints are
  essentially never in a WordPiece vocabulary and become `[UNK]` either way, but
  it is a real difference and is written down rather than assumed away.
- **Punctuation classification is range-based**, not a full Unicode category
  table, for the same reason. It covers ASCII, Latin-1, General Punctuation, CJK
  punctuation and fullwidth forms.
- **The round trip is lossy on purpose.** Case and accents are destroyed at
  encode time; no detokenizer restores them. The test asserts the *normalised*
  text, because asserting equality would be asserting the algorithm is something
  it is not.

## Cost

One wrong assumption (`##`), caught by the oracle in the first run of the
integration test rather than by reading the output of a model. 41 unit tests and
13 oracle cases; `clippy -D warnings` and `fmt` clean; 241 workspace tests pass.
