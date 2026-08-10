---
topic: Every one of llama.cpp's 182 flags, categorised against what Bigtea has — the real denominator for "all of its options", replacing an estimate that was never checked
status: audited 2026-08-10, tracked
links: [lts-parity-criteria.md]
---

`lts-parity-criteria.md` said llama.cpp has "~100" CLI flags. That was a guess.
The real list, from `llama-completion --help` on build `daef2b3`:

```
$ llama-completion --help | grep -oE '\-\-[a-zA-Z0-9][a-zA-Z0-9-]*' | sort -u | wc -l
182
```

| bucket | flags | state |
|---|---:|---|
| **have** | **58** | done |
| **samplers** | 22 | **13 done 2026-08-10**, DRY (5) + `--samplers` ordering (3) left |
| interaction / prompt handling | 22 | **done 2026-08-11** — including a real REPL |
| runtime / threading / memory | 31 | mostly gap; several are meaningless here |
| RoPE, YaRN, context shift | 15 | **9 done 2026-08-11**; the other 6 refused, see below |
| logging | 13 | gap, cheap |
| **GPU** | 15 | **won't** — no backend to apply them to |
| fetch / Hugging Face | 9 | partly covered by `bigtea-pull`, different spelling |
| reasoning / speculative draft | 8 | gap |
| KV cache type / prompt cache | 7 | gap; `--cache-type-k/v` is real and substantial |
| chat template | 6 | 2 done (detection), `--jinja` won't |
| LoRA / control vectors | 5 | gap |
| grammar / JSON schema | 4 | gap |
| meta (`--help`, `--version`) | 4 | 1 done |

**"All of them" is not the right target and this table is why.** Fifteen are
GPU-only on an engine with no GPU backend; several more (`--no-mmap`,
`--mlock`, `--direct-io`) describe a loading strategy Bigtea does not use
because it owns residency itself, which is the entire design. Implementing
those as no-ops that accept the flag and change nothing would be worse than not
having them — that is precisely the failure `-t` had for weeks.

The honest goal: **every flag that means something for a CPU runner that owns
its own residency**, which is roughly 120 of the 182.

## Done 2026-08-10 — samplers, 13 of them

`--typical`/`--typical-p`, `--top-nsigma`/`--top-n-sigma`, `--dynatemp-range`,
`--dynatemp-exp`, `--xtc-probability`, `--xtc-threshold`, `--mirostat`,
`--mirostat-ent`, `--mirostat-lr`, `--logit-bias`, `--ignore-eos`, on top of the
existing temperature/top-k/top-p/min-p/penalties.

**Three real bugs surfaced while wiring them**, all of the same shape — a flag
accepted, echoed, and silently doing nothing:

1. `is_greedy()` short-circuited to the raw argmax, so `--logit-bias` and
   `--ignore-eos` were ignored at temperature 0, which is Bigtea's default.
2. `--mirostat 2` alone produced **byte-identical output to greedy**, twice:
   once through `is_greedy`, then again through the temperature-0 early return.
   llama.cpp's default temperature is 0.8 and ours is 0, so "mirostat with no
   other flags" is the normal way to ask for it and it did nothing.
3. Drawing XTC's random number unconditionally would have shifted the seeded
   stream for every existing `--seed` run that never asked for XTC.

Caught by tests and by running the flags against a real model and reading the
output. **Two of the three are invisible in any test that only checks the
process exits zero**, which is what makes this category worth the care.

## Done 2026-08-11 — interaction, and Bigtea has a REPL

`-i`/`--interactive`, `-cnv`/`--conversation`, `-st`/`--single-turn`,
`--multiline-input`, `--in-prefix`, `--in-suffix`, `--in-prefix-bos`,
`-sys`/`--system-prompt`, `--system-prompt-file`, `-co`/`--color`,
`--simple-io`, `--display-prompt`/`--no-display-prompt`, `-sp`/`--special`,
`--print-token-count`, `--verbose-prompt`, `-e`/`--escape`/`--no-escape`,
`-r`/`--reverse-prompt`.

**The KV cache is what makes this worth having**: a turn costs only its new
tokens, because everything said so far is already in the cache. Verified as a
real conversation rather than a mechanism:

```
$ bigtea-run <llama-3.2-1b> "Name the capital of France in one word." \
    -n 24 -cnv -sys "You are terse. Answer with one word only."
chat       llama3 template
Paris.
> What is the capital of Japan?
Tokyo.
```

Two things that would otherwise be silent:

- **`--escape` is on by default**, matching llama.cpp, so a prompt containing a
  backslash-n is two lines. Checked by token id rather than by eye: `198` (a
  real newline) with it, `1734` (a literal two-character sequence) with
  `--no-escape`.
- **Stop sequences reset per turn.** Carried over, a stop string from an earlier
  answer ends the next one instantly, and the session looks hung.

`--keep` is deliberately **not** accepted. It controls what survives a context
shift, and Bigtea has no context shift — accepting it would be a flag that does
nothing, which is the exact failure this audit exists to prevent.

## Done 2026-08-11 — RoPE and YaRN, 9 of the 15

`--rope-freq-base`, `--rope-freq-scale`, `--rope-scale`, `--rope-scaling`,
`--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-fast`,
`--yarn-beta-slow`, `--yarn-orig-ctx`.

These were nearly free and had been sitting there: `RopeParams` already carried
all six YaRN fields and `rope()` set exactly one of them, so ggml's `rope_ext`
was being handed defaults for the rest on every model. The container is now read
for `rope.scaling.factor`, `rope.scaling.type`, `attn_factor`, `beta_fast`,
`beta_slow` and `original_context_length`, and the flags override that.

**`--rope-scale` is the reciprocal of `--rope-freq-scale`** — llama.cpp's is a
multiplier on the *context*, ours on the *frequency*. Storing it unconverted
inverts every long-context model, silently.

Overrides are **printed**, not applied quietly:

```
$ bigtea-run <llama-3.2-1b> ... --rope-freq-base 50000 --rope-scale 2
rope       overridden: freq_base 500000 -> 50000, freq_scale 1 -> 0.5
```

RoPE is the setting most likely to turn a working model into a fluent-but-wrong
one, and 500000 is Llama-3.2's real base — visible here only because the line is
printed.

**The other six are refused, not accepted:** `--grp-attn-n`/`-w` (self-extend,
not implemented), `--context-shift`/`--no-context-shift` and `--defrag-thold`
(no context shift and no KV fragmentation to threshold), `--swa-full` (we always
keep the full window cache, so the flag has nothing to switch).

## Next batches, in order

1. **DRY + `--samplers` ordering (8)** — finishes the sampler bucket. DRY needs
   n-gram suffix matching and sequence breakers; the ordering flag needs the
   chain to become data rather than a fixed sequence of calls.
2. **RoPE / context (15)** — `--rope-freq-base`, `--rope-freq-scale`,
   `--rope-scaling`, YaRN. Cheap, and needed for any model whose container
   disagrees with its training context.
4. **Logging (13)** — `--log-file`, `--log-disable`, `--verbose`, timestamps.
   Cheap and mechanical.
5. **KV cache types (2)** — `--cache-type-k/v`. Real work and real value: it
   halves KV memory, which is the axis this project competes on.
6. **Grammar / JSON schema (4)** — previously marked `won't for LTS`. Reopened
   because an agent calling a local model wants constrained output more than it
   wants most of the above.
