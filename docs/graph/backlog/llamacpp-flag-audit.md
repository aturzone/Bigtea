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
| **have** | **74** | done |
| **samplers** | 22 | **21 done**; only `--backend-sampling` left, and it is a GPU concept |
| interaction / prompt handling | 22 | **done 2026-08-11** — including a real REPL |
| runtime / threading / memory | 31 | mostly gap; several are meaningless here |
| RoPE, YaRN, context shift | 15 | **9 done 2026-08-11**; the other 6 refused, see below |
| logging | 13 | **11 done 2026-08-11**; status moved to stderr |
| **GPU** | 15 | **won't** — no backend to apply them to |
| fetch / Hugging Face | 9 | partly covered by `bigtea-pull`, different spelling |
| reasoning / speculative draft | 8 | gap |
| KV cache type / prompt cache | 7 | **`--cache-type-k/v` done 2026-08-11**; prompt cache (5) left |
| chat template | 6 | 2 done (detection), `--jinja` won't |
| LoRA / control vectors | 5 | gap |
| grammar / JSON schema | 4 | gap |
| meta (`--help`, `--version`) | 4 | 2 done |

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

## Done 2026-08-11 — logging, and the bug underneath it

`--log-disable`, `--log-file`, `--log-timestamps`/`--no-log-timestamps`,
`--log-prefix`/`--no-log-prefix`, `-v`/`--verbose`/`--log-verbose`,
`--verbosity`/`--log-verbosity`, `--perf`/`--no-perf`, `--version`.

**The flags are the smaller half. The real change is that status now goes to
stderr.** Everything the runner says about itself — shape, residency, prefill
timing — is diagnostics; the generated text is output. They shared stdout, so
`bigtea-run … > answer.txt` captured a 16-line header along with the answer and
there was no way to separate them.

```
$ bigtea-run <llama-3.2-1b> "The capital of France is" -n 6 --log-disable 2>/dev/null
 Paris. The capital of France

$ bigtea-run … --log-file bt.log 2>/dev/null | head -3
 Paris. The capital of France
   [bt.log: 16 lines, starting "model      llama (direct (cache bypassed))"]

$ bigtea-run … --log-timestamps --log-prefix
   0.152 I model      llama (direct (cache bypassed))
```

`--version` is handled **before the positional model path is taken**. Parsed
with the other flags it became the path, and the runner reported that it could
not open a file called `--version`.

Two of the thirteen are refused: `--log-colors` (status goes to a stream that
may be a file; llama.cpp's colour applies to a level marker we render as one
character) and `--no-host`, which is not a logging flag at all.

## Done 2026-08-11 — DRY, the sampler that actually breaks a loop

`--dry-multiplier`, `--dry-base`, `--dry-allowed-length`,
`--dry-penalty-last-n`, `--dry-sequence-breaker`.

DRY asks a narrower question than a repeat penalty. A repeat penalty punishes a
token for having appeared, which also suppresses the ordinary reuse prose is
made of. DRY looks for a *sequence* replaying and penalises only the token that
would continue it, growing the penalty geometrically with how long the run
already is.

```
$ bigtea-run <llama-3.2-1b> "The sea is blue. The sea is blue. The sea is blue. The sea is" -n 14
 blue. The sea is blue. The sea is blue. The sea      <- stuck

$ ... --dry-multiplier 1.5
 ... blue. (Repeat ad infinitum)  This is a classic example
```

**Sequence breakers are what stop it penalising structure.** A match may not
cross a newline, quote, colon or asterisk, or a list is punished for having the
shape of a list. They arrive as text and the sampler works in token ids, so they
are resolved once the vocabulary exists; a breaker that is not a single token in
this vocabulary is **skipped rather than approximated**.

One test caught the author, not the code: a fixture written as
`[9,1,2,3,4,9,1,2,3]` has `9 1 2 3` repeating, so the match is four long and the
penalty is `base^2`, not `base^1`. The assertion was wrong and the
implementation was right — both cases are pinned now.

## Done 2026-08-11 — `--samplers` chain ordering; the sampler bucket is closed

`--samplers`, `--sampler-seq`, `--sampling-seq` (three spellings of one flag).

The chain was a fixed sequence of calls; it is now a `Vec<SamplerStage>` walked
in order. Same seed, same model, different order, different answer:

```
$ ... --temp 1.5 --top-p 0.5 --seed 9 --samplers "top_k;typ_p;top_p;min_p;xtc;temperature"
 vast and unpredictable, and its vastness is mirrored in t

$ ... --temp 1.5 --top-p 0.5 --seed 9 --samplers "temperature;top_p"
 turbulent, if we are successful it will determine us. Som
```

That is the whole point of the flag: a hot temperature flattens the
distribution, so `top_p 0.5` *after* it keeps a different set than before it.
Neither order is more correct and people ask for both.

**What is not reorderable, stated rather than papered over:** the penalties, DRY
and top-n-sigma act on **logits** and always run first; the six stages above act
on probabilities. That is also where llama.cpp puts them in its own default
chain, so the constraint costs nothing in practice.

**An unknown stage refuses the whole run** rather than dropping that stage:

```
$ ... --samplers "top_k;top_q"
bigtea-run: --samplers: unknown stage "top_q"
  known stages: top_k, typ_p, top_p, min_p, xtc, temperature
  penalties, dry and top_n_sigma act on logits and always run first
$ echo $?
2
```

A typo that silently removed a filter would be the same class of failure as a
flag that does nothing — the user believes a constraint is active when it is
not.

`--backend-sampling` is the one sampler flag left and it is **won't**: it moves
sampling onto the GPU, and there is no GPU backend to move it to.

## Done 2026-08-11 - `--cache-type-k/v`, a quantised KV cache

`--cache-type-k`/`-ctk`, `--cache-type-v`/`-ctv`, taking `f16` (default) or
`q8_0`. This is the one flag in the list that changes what the engine *is* able
to do rather than how it is driven: the KV cache is the memory that grows with
context, and it is the axis this project competes on.

```
$ bigtea-run <llama-3.2-1b> "The capital of France is" -n 10
kv cache   15 positions, 0.5 MiB, f16

$ bigtea-run <llama-3.2-1b> "The capital of France is" -n 10 -ctk q8_0 -ctv q8_0
kv cache   15 positions, 0.2 MiB, q8_0
```

**And the quality cost is measured, not asserted** - which is what the
perplexity work earlier today was for:

| KV storage | perplexity | bytes/value |
|---|---:|---:|
| f16 | 29.0909 | 2.00 |
| q8_0 | 28.9047 | 1.0625 |

**0.64% apart on 189 scored tokens.** q8_0 landing slightly *lower* is noise at
that sample size, not an improvement, and it must not be quoted as one.

Three things that would have been silent:

- **A block may not span two heads.** Quantisation runs row by row, where a row
  is `head_dim`; a block straddling a head boundary applies one head's scale to
  another head's values, which is fluent nonsense rather than an error.
- **`head_dim` must be a multiple of 32**, or a row does not hold whole blocks.
  Every architecture here uses 64, 128 or 256, but one that did not falls back
  to f16 **and says so** rather than being misquantised.
- **`is_consistent()` was counting values where the vectors now hold bytes.** It
  passed under f16 by coincidence and failed immediately under q8_0 - the test
  that caught it existed already and was checking the right thing.

K and V share one type because ggml's banded attention asserts
`k->type == v->type`; accepting different ones would work until that path was
reached. Both spellings are accepted and the last wins.

`q4_0` is **not** offered. ggml has the kernels, but the accuracy cost at 4 bits
in attention is real and unmeasured here, and offering a type without the
perplexity number beside it is the thing this audit exists to prevent.

## Next batches, in order
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
