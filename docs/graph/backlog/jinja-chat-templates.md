---
ticket: evaluate chat templates instead of matching them by family
status: open, scoped — census done 2026-08-11
links: [../research/gemma-was-running-silu-2026-08-11.md, lts-parity-criteria.md]
---

## What is missing

`--jinja` is the last capability gap in the CLI that is not GPU, not a draft
model, and not an adapter. llama.cpp *evaluates* `tokenizer.chat_template` with
a real Jinja engine. This build matches the template against 54 known families
by substring and applies a hardcoded renderer.

The family renderers are **verified byte-identical to llama.cpp for 52 of its
54 names**, so this is not a correctness gap on models anyone has. It is a
coverage gap on models nobody has tried: a finetune with a novel template falls
back to `ChatFormat::Generic`, and the runner says so rather than guessing —
which is the right failure, but it is still a failure.

## Why it has not been built, and the rule that must hold if it is

From `chat.rs`, unchanged since the module was written:

> Evaluating Jinja properly means a whole expression language, and **a
> half-implemented one silently produces the wrong framing**, which is the
> failure mode this project is most expensive at.

That is still the governing constraint. **An engine that does not understand a
construct must refuse the template and fall back to family matching, never
guess.** A wrong framing does not error — the model answers fluently, having
been handed a prompt shape it has never seen, and no test that checks "did it
produce a string" can see it.

## The subset is bounded, and here is the census

Every `tokenizer.chat_template` in `C:/Projects/models/` — **12 templates**:

| construct | uses |
|---|---:|
| `{% if %}` / `{% endif %}` | 123 |
| `{% set %}` | 98 |
| `{% else %}` | 40 |
| `{% for %}` / `{% endfor %}` | 31 |
| `{% elif %}` | 21 |
| `loop.index0` | 20 |
| `loop.last` | 12 |
| `loop.first` | 10 |
| `namespace()` | 10 |
| `raise_exception()` | 6 |
| `strftime_now()` | 1 |
| filters: `tojson` 15, `trim` 6, `length` 5 | 26 |
| operators: `in`, `not`, `is defined`, `is string`, `is not none` | 30 |

That is the whole language these templates use. **No macros, no imports, no
inheritance, no custom filters beyond three.** It is a weekend, not a quarter —
comparable to `bigtea-grammar`, which is a self-contained crate with no
dependencies at all.

Counting caveat: a naive `| filter` regex also matches the pipe inside
`<|im_start|>`, so `filter:im_start` and friends in the raw census are
artefacts. The three above are the real ones.

## Shape

A new crate, `bigtea-jinja`, with no dependencies — same as `bigtea-grammar`.
Lexer, parser, evaluator. The evaluator's environment is exactly what a chat
template gets: `messages` (a list of maps), `bos_token`, `eos_token`,
`add_generation_prompt`, and whatever `{% set %}` introduces.

**`raise_exception` must actually fail the render**, not print. Templates use it
to reject conversations they cannot express — a system turn where none is
allowed, alternating-role violations — and swallowing it produces exactly the
wrong framing those templates exist to prevent.

`strftime_now` needs a clock, which makes a render non-reproducible. One
template of twelve uses it; take the current time and record that the output is
not byte-stable for that model rather than freezing a fake date.

## Acceptance

The oracle already exists: `crates/bigtea-tokenizer/tests/chat-templates.txt` is
llama.cpp's own rendering of all 54 templates, captured token by token.

1. For every container on disk, the **Jinja-evaluated** output must equal the
   **family-matched** output. They are two independent implementations of the
   same thing, and 52 of the 54 family renderers are already verified against
   llama.cpp — so agreement is a real cross-check rather than a self-check.
2. Where they disagree, llama.cpp with `--jinja` decides, and the command line
   goes in the commit.
3. A template using anything outside the subset must **fall back with a named
   reason**, and a test must assert that — the fallback is the safety property,
   so it needs a test more than the happy path does.

## MEASURED 2026-08-11: the census was wrong, and the method was the reason

The crate is built and the acceptance test run. **3 of 15 containers agree, 9
are refused, 3 differ.** Not close to wiring `--jinja`.

The census above counted **statement tags** — `{% for %}`, `{% if %}` — because
that is what a regex over `{%-?\s*(\w+)` finds. It saw none of the
**expression** forms, and those are where the language actually lives. Each
round of fixes revealed the next one:

| round | added | next thing it hit |
|---|---|---|
| 1 | the censused subset | `messages[1:]` — slices |
| 2 | slices | `namespace(a=1, b=2)` — keyword arguments |
| 3 | kwargs | `messages\|length - 1` — subtraction |
| 4 | `-` and `%` | `range(ns.last_query_index, -1, -1)` |
| 5 | ? | a multi-line string literal inside `{{ }}` |

**"No macros, no imports, no inheritance — a weekend, not a quarter" was wrong**,
and wrong in a specific, repeatable way: I measured the part of the language
that is easy to grep for and concluded the whole language was that size.

What the 9 refusals actually are:

- **5** are DeepSeek-V4-Flash's five shards, i.e. **one** template using
  `for k, v in ...`.
- **2** are Qwen3's `range()` loop over prior turns.
- **1** is Gemma-3's multi-line string.
- **1 is not a bug at all**: Gemma-2's template *raises* on a system turn, and
  the engine correctly propagated it. **That means the family matcher silently
  accepts a conversation Gemma's own template forbids** — worth its own look,
  and found only because the Jinja path refused.

## What this changes

The crate stays. It is 34 tests of real evaluation, it builds ggml-free, and
every construct it does not know is refused by name — the safety property held
through all five rounds, which is the part that actually mattered.

`--jinja` stays declined. The gap between "a template engine exists" and "the
flag can be claimed" is exactly the 12 containers it cannot render, and shipping
the flag now would mean falling back on 80% of the models on this machine while
the help text promised evaluation.

**Next round is a real re-census**, on expressions rather than tags: extract
every `{{ … }}` and `{% … %}` body from all 12 templates, and enumerate the
distinct syntactic forms. That is the measurement that should have come first.

## Round 6-9, same day: 3 agree -> 6, and a PANIC on a real container

Driving off the acceptance test instead of a census closed most of the gap:

| added | because |
|---|---|
| slices `messages[1:]` | 4 templates drop the system turn that way |
| keyword arguments | `namespace(multi_step_tool=true, ...)` |
| `-` and `%` | `messages\|length - 1`, `loop.index0 % 2 == 0` |
| negative literals | `range(n, -1, -1)` has no left operand for a binary minus |
| inline conditional | `(first_user_prefix if loop.first else "")` |
| `range()` | Qwen3 walks backwards over prior turns |
| `is true` / `is false` | Qwen3. **Not the same as falsy** — `is false` asks whether the value IS the boolean, so an empty string must not satisfy it |
| tuple unpacking | DeepSeek-V4-Flash |

**The most important find was a panic, not a missing feature.** Keyword matching
sliced by byte offset, and DeepSeek's template is full of U+FF5C (`｜`):

```
end byte index 3 is not a char boundary; it is inside '｜'
```

A real container **crashed** the parser rather than being refused by it. That
breaks the crate's entire premise — the caller can fall back from a refusal and
cannot fall back from a crash. Fixed by matching through `str::get`, with a
regression test that feeds it the exact characters.

### Where it stands

**6 agree, 6 refused, 3 differ** of 15 containers — and 15 containers is about
11 distinct templates, since DeepSeek's five shards share one.

The 6 refusals are **two** templates: Gemma-2 (a correct `raise_exception`) and
DeepSeek's five shards, which want `'' + true`. Jinja itself raises on that,
so the refusal may be right and llama.cpp's `--jinja` has to settle it.

### The differences are the interesting part now

Llama-3.2, family vs Jinja:

```
family: "<|start_header_id|>system<|end_header_id|>

SYS<|eot_id|>..."
jinja : "<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023

         Today Date: 26 Jul 2024

SYS<|eot_id|>..."
```

**The Jinja rendering is the more faithful one.** Llama-3's own template emits
that preamble and the hardcoded renderer — ours *and* llama.cpp's — drops it. So
this is no longer only a coverage gap: the family path is losing content the
model's template specifies, on a model that is verified.

That has to be settled against `llama-completion --jinja` before either path
changes, and it is the next thing to do here.
