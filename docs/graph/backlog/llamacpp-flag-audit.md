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
| **have** | 21 | done |
| **samplers** | 22 | **13 done 2026-08-10**, DRY (5) + `--samplers` ordering (3) left |
| interaction / prompt handling | 22 | gap — the next batch |
| runtime / threading / memory | 31 | mostly gap; several are meaningless here |
| RoPE, YaRN, context shift | 15 | gap; `--rope-freq-base`/`--rope-scale` are cheap and real |
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

## Next batches, in order

1. **Interaction (22)** — `--interactive`, `--conversation`, `--in-prefix`,
   `--in-suffix`, `--system-prompt`, `--color`, `--escape`, `--special`,
   `--print-token-count`, `--verbose-prompt`, `--keep`. This is what makes a
   local model usable from a terminal rather than benchmarkable.
2. **DRY + `--samplers` ordering (8)** — finishes the sampler bucket. DRY needs
   n-gram suffix matching and sequence breakers; the ordering flag needs the
   chain to become data rather than a fixed sequence of calls.
3. **RoPE / context (15)** — `--rope-freq-base`, `--rope-freq-scale`,
   `--rope-scaling`, YaRN. Cheap, and needed for any model whose container
   disagrees with its training context.
4. **Logging (13)** — `--log-file`, `--log-disable`, `--verbose`, timestamps.
   Cheap and mechanical.
5. **KV cache types (2)** — `--cache-type-k/v`. Real work and real value: it
   halves KV memory, which is the axis this project competes on.
6. **Grammar / JSON schema (4)** — previously marked `won't for LTS`. Reopened
   because an agent calling a local model wants constrained output more than it
   wants most of the above.
