---
topic: "Bigtea's chat framing against llama.cpp's, both paths, on token IDs — two silent bugs found and fixed, three models where we invent a template llama.cpp does not"
status: measured 2026-08-15, three rendering differences open
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

## Phi-3: not Jinja at all, and the mechanism is worth remembering

Feeding **both engines the same literal string** ruled out the template and put
it in the tokenizer:

```
input      <|user|>\nSYS\nHI<|end|>\n<|assistant|>\n
llama.cpp  [1, 32010, 317, 21554, 13, 17628, 32007, 32001]                       8 tokens
bigtea     [1, 32010, 29871, 13, 14816, 29903, 13, 17628, 32007, 29871, 13, ...] 14
```

llama.cpp drops whitespace **following a special token**
(`LLAMA_TOKEN_ATTR_RSTRIP`, applied in `tokenizer_st_partition`). SPM then
prefixes the fragment with `▁`, so the *next word* tokenizes differently — `▁SYS`
as `317,21554` against our `▁`, `\n`, `SY`, `S`.

**The attribute is not in the container.** `llama-vocab.cpp` sets it from the
model's *name*:

```cpp
} else if (_contains_any(model_name, {"phi-3", "phi3"})) {
    for (auto id : cache_special_tokens) {
        _set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, true);
    }
    for (const auto * token : {"<unk>", "<s>", "<|endoftext|>"}) {
        _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, false);
    }
}
```

Nothing in the tokenizer metadata separates a Phi-3 vocabulary from any other SPM
one, so matching the reference means keying on the same string it keys on —
alongside `<mask>` LSTRIP rules keyed on `jina-v2-*` and `modern-bert`. **A
tokenizer whose behaviour depends on `general.name` is the kind of fact that
costs a day to rediscover.**

Fixed and tested, including the negative case: a model *not* named phi-3 keeps
its whitespace, because applying this everywhere would silently change every SPM
chat model. Phi-3 now agrees on **both** paths; agreement went 6 → 7 of 14.

**Why the parity sweep never saw it**: that sweep uses plain prompts with no
special tokens, so no fragment ever follows one. None of the 104 prompts behind
"102 exact" could have caught this, and it affects every chat-framed request the
server handles.

## internlm2: the detector asked whether the word appears, again

`mentions_system_role` tested for the literal `'system'`. ChatML templates never
write it:

```jinja
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}
```

The role is **interpolated**, so every role is handled — system included — and
the word is nowhere in the template. We reported "no system branch", merged the
system turn into the user turn, and rendered **two** turns where llama.cpp
rendered three.

**The role has to be emitted, not merely compared, and that is the whole
difficulty.** Phi-3 also contains `['role']`, inside `{% if message['role'] ==
'user' %}`, and genuinely has no system branch — it must still be merged. A
substring test anywhere in the template fixes internlm2 by breaking Phi-3. So the
check scans `{{ … }}` blocks only: where a template *emits* rather than
*decides*.

That is the second guard in one day that asked "does the word appear?" instead
of "does this template handle a system turn?" — the first being the Gemma
polyfill, which fired only when the template never mentions system and so could
not see the one template that mentions it in order to **reject** it. Same
question, two wrong answers, opposite directions.

## Where it settled: 9 of 14, and why the last five are not bugs

| | |
|---|---|
| `OLMo`, `starcoder2`, `all-MiniLM` | **no chat template.** llama.cpp passes the text through; we impose `System:/User:/Assistant:`. A product decision, recorded above |
| `Falcon3`, `tinyllama` | **family path only — our `--jinja` matches llama.cpp's `--jinja` on both.** Two hardcoded shortcuts disagreeing |

The last row is worth being precise about rather than counting as a defect.
Both models resolve to our `Zephyr` family renderer, which emits
`<|role|>\ncontent<eos>\n`. That matches **tinyllama's** template, which does
emit an EOS between turns, and not **Falcon3's**, which does not — and llama.cpp
classifies the two differently, so its shortcut disagrees with ours on Falcon3
and with the *template* on tinyllama.

**One hardcoded renderer cannot be right for two models whose templates differ,
which is the entire argument for `--jinja`.** We have a `ChatFormat::Falcon3` arm
that reproduces llama.cpp's `LLM_CHAT_TEMPLATE_FALCON_3` exactly
(`<|role|>\ncontent\n`); the detector sends Falcon3 to `Zephyr` before reaching
it. Re-ordering detection to llama.cpp's rule (`<|assistant|>` + `<|user|>` +
`</s>` → Falcon-3, checked *before* zephyr) is the fix, and it is deferred rather
than guessed: tinyllama's template appears to satisfy the same gate, and a
detector change that silently re-classifies a model whose framing is currently
correct is exactly the trade this file has now recorded twice.

**The exact path already agrees.** Anyone who needs byte-parity with the
reference on these two models has `--jinja` today.

## Earlier: three genuine rendering differences

| model | what differs |
|---|---|
| `Falcon3-1B` | llama.cpp's Jinja emits a trailing newline after every turn; ours does not — a block-trimming difference, not a content one |
| `gemma-2-2b` | the system text is merged into the first user turn (Gemma has no system role) and we join with `\n\n` where the template joins with `\n` |
| `internlm2` | llama.cpp's Jinja renders an extra turn ours does not; the template's default-system branch is the likely cause |

**Phi-3 was the fourth of these and turned out not to belong in the list at all**
— it was the tokenizer, not the template, which is why feeding both engines the
same literal string was the step that mattered. The remaining three are genuinely
in the rendering, and none is a silent-position bug of the BOS or RSTRIP kind.

All three are reachable from `python scripts/jinja-bigtea-vs-llamacpp.py
<model.gguf>`, which prints both sides' token IDs. **Rule out the tokenizer
first**: render the prompt, feed the identical string to both engines with `-p`,
and see whether they still disagree. Two of the three bugs found today were on
the far side of that check.
