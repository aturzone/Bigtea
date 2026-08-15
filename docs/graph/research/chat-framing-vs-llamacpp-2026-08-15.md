---
topic: "Bigtea's chat framing against llama.cpp's, both paths, on token IDs — one bug found and fixed, three models where we invent a template llama.cpp does not"
status: measured 2026-08-15, four rendering differences open
links: [llamacpp-flag-audit.md, parity-band-discriminator-2026-08-15.md]
---

## The script that was cited for a claim it does not test

`scripts/jinja-vs-llamacpp.py` says, in its own docstring:

> Compare Bigtea's Jinja rendering against `llama.cpp --jinja`.

It does not. It runs `llama-completion` **twice** — once with `--jinja`, once
with `--no-jinja` — and never invokes Bigtea at all. What it measures is real and
worth keeping: **llama.cpp's hardcoded renderer disagrees with llama.cpp's own
Jinja on 5 of 18 containers**, which is the disagreement our family matcher has
to pick a side of. But it is not evidence about our engine, and it was being read
as though it were.

Same failure as the `REFUSED` table's `--jinja` row: a description that outlived
the code, still true-sounding, still being quoted.

`scripts/jinja-bigtea-vs-llamacpp.py` runs the four-way that settles it:

```
bigtea --jinja   vs  llama.cpp --jinja      does OUR Jinja match THEIRS?
bigtea           vs  llama.cpp --no-jinja   does our family matcher match?
```

on **token IDs rather than rendered text**, because the tokens are what the model
sees — two renderings differing only by a trailing newline the tokenizer drops
are not worth failing on, and two that tokenize apart are, however alike they
look.

## What it found immediately: BOS twice

```
bigtea --jinja : [2, 2, 105, 2364, 107, 86404, ...]
llama.cpp      : [2,    105, 2364, 107, 86404, ...]
```

`encode` prepended BOS whenever the container declared `add_bos_token`, while the
control-token splitter separately mapped the literal `<bos>` **inside the
template** to its own id. Gemma's template opens with `<bos>`, Llama-3's with
`<|begin_of_text|>`, so the model was prefilled **a token long** on gemma-3,
Llama-3.2, internlm2 and Phi-3.

The exact mirror of the Falcon3 bug, which was a token *short*, and just as
quiet — nothing raises, and the model answers fluently from a position nobody
trained it on.

**The feature and the bug arrived together.** The hardcoded family renderers
never emit the BOS text, so this could not exist until a real Jinja engine began
evaluating the container's own template. Adding `--jinja` added this.

Fixed and tested (`bos_is_not_doubled.rs`), guarded narrowly — only the literal
spelling counts, an empty or absent BOS answers `false` rather than matching
everywhere, and dropping a BOS the text did *not* supply is the Falcon3 bug and
has its own test. Agreement went **4 → 6** of 14 loadable containers.

## Three models where we invent a template and llama.cpp does not

`all-MiniLM-L6-v2`, `OLMo-1B` and `starcoder2-3b` have **no chat template**.
llama.cpp emits the user text and nothing else:

```
starcoder2   llama.cpp: [14776]                      <- "HI", and that is all
             bigtea   : [2964, 63, 18273, 222, 514, 63, 37211, 222, 17595, 63]
                                                     <- "System: SYS\nUser: HI\nAssistant:"
MiniLM       llama.cpp: [101, 7632, 102]             <- [CLS] hi [SEP]
             bigtea   : [101, 2291, 1024, ... 102]
```

Ours is a deliberate fallback and it announces itself (*"chat template not
recognised — using a plain framing; the model may not respond as an
assistant"*), so it is not an accident. It is still a divergence, and the
direction is worth stating plainly: **inventing `System:/User:/Assistant:` for a
base model feeds it text it was never trained on**, which is the mirror of the
bug that made instruct models continue a conversation instead of answering it.
llama.cpp's choice — pass the text through untouched — is the one that matches
what a base model expects.

**Not changed here**, because it is a product decision rather than a defect, and
the two readings genuinely conflict: a chat CLI that prints nothing useful for
`starcoder2` is also a bad answer. Recorded so the choice is made deliberately
the next time rather than inherited.

## One where our family renderer is closer to the template than llama.cpp's

`tinyllama`, family path only — our Jinja and llama.cpp's Jinja already agree:

```
bigtea   : ... 14816, 29903,  2, 29871, 13, 29966 ...   <- `</s>` between turns
llama.cpp: ... 14816, 29903,     29966 ...              <- no separator
```

The model's own template emits `</s>` between turns; llama.cpp's *hardcoded*
renderer drops it, and ours does not. This is one of the 5 containers where
llama.cpp disagrees with itself, and on this one **we match its template rather
than its shortcut.** Left alone.

## Still open: four genuine rendering differences

| model | what differs |
|---|---|
| `Falcon3-1B` | llama.cpp's Jinja emits a trailing newline after every turn; ours does not — a block-trimming difference, not a content one |
| `gemma-2-2b` | the system text is merged into the first user turn (Gemma has no system role) and we join with `\n\n` where the template joins with `\n` |
| `internlm2` | llama.cpp's Jinja renders an extra turn ours does not; the template's default-system branch is the likely cause |
| `Phi-3-mini` | we insert a newline after `<\|user\|>` where llama.cpp inserts a space, which re-tokenizes `SYS` (`317,21554` against `14816,29903`) |

None of these is a BOS-class silent-position bug: each is a whitespace or
turn-structure difference in the rendered template. All four are reachable from
`python scripts/jinja-bigtea-vs-llamacpp.py <model.gguf>`, which prints both
sides' token IDs.

**Phi-3 is the one to start on** — a whitespace difference that changes the
tokenization of the surrounding text is the kind that moves output, and Phi-3 is
also the only model in the parity sweep with near-ties.
