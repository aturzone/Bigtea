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
