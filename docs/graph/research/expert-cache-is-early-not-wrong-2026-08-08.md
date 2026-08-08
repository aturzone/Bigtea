---
topic: R1 — the expert cache is built and correct, and buys nothing until the KV cache exists. Off by default, with the measurement that says so.
status: resolved
links: [routing-prefill-predicts-generation-2026-08-08.md, routing-skew-is-per-prompt-2026-08-08.md, ../backlog/next-session-handoff.md]
---

R1 was the payoff for R0 and R0.1: wire the frequency-gated expert cache — the
policy that took Qwen3 from a 17% to a 70% hit rate — into the V4-Flash path.
It is built, it is verified against the llama.cpp oracle, and **on today's engine
it is a regression.** That is not a defect in the cache. It is a fact about what
a pass currently reads, and it reorders the roadmap.

## The number nobody had measured: distinct experts per pass

Every estimate in this project used **3.21 GiB of routed experts per token**,
which is 6 experts × 43 layers. That figure is correct for a *single-token* step
and wrong for everything Bigtea actually runs, because expert reads are
**deduplicated per block across the whole batch** — `read_expert_slices` is given
`unique`, not one entry per selection.

Measured from the routing captures:

| tokens in the pass | selections/layer | **distinct** experts/layer | read per pass |
|---:|---:|---:|---:|
| 1 (needs a KV cache) | 6 | 6 | **3.21 GiB** |
| 17 | 102 | 39.7 | 21 GiB |
| 166 | 996 | 122.8 | **66 GiB** |

Selections grow 10x from 17 to 166 tokens; distinct reads grow 3x. **That gap is
the whole story.** A cache is sized against the third column, not the second, and
the third column is 66 GiB.

## What the cache did

Same prompt, same seed, `--cache 0` against the auto-sized budget. Correctness
first, because on this architecture a wrong cache returns fluent nonsense rather
than an error:

```
short: IDENTICAL — the cache did not change the answer
long:  IDENTICAL — the cache did not change the answer
```

| run | cache | hit rate | evictions | prefill | generation |
|---|---|---:|---:|---:|---:|
| 17 tokens | none | — | — | 18.2s | 0.049 tok/s |
| 17 tokens | 1.51 GiB | **4.1%** | 2000 | 19.3s | 0.050 tok/s |
| 166 tokens | none | — | — | **64.5s** | 0.015 tok/s |
| 166 tokens | 1.75 GiB | **1.9%** | 2694 | **75.3s** | 0.015 tok/s |

**1.9–4.1% hits, no measurable generation gain, and a 17% slower prefill.** The
slowdown is the admission copy, paid on every miss the policy decides to keep.
2694 evictions against 2516 hits is a cache thrashing, exactly as expected when
the working set is 38x the budget.

This is what "2% of 66 GiB" looks like when you run it instead of arguing
about it.

## Why it is early rather than wrong

The moment a step stops re-reading the whole sequence, the working set collapses
from 66 GiB to **3.21 GiB** — and R0.1 measured that a set warmed on the prompt
covers **86%** of the routing the generated tokens then ask for. At that point
the same 1.5 GiB holds a real share of what is needed and the arithmetic inverts.

```
today          a step reads 122.8 experts/layer   1.5 GiB cache = 2%   useless
after R3       a step reads 6 experts/layer       1.5 GiB cache = 47%  useful
```

**So R1 is not the next task; R3 is.** R1 is already done and waiting for it.

### The same engine already demonstrates this, on the other architecture

Qwen3-30B-A3B **has** a KV cache, so a generated token there needs one token's
experts rather than the whole sequence's. Run in the same session, same machine,
same policy:

```
cache      8.48 GiB for experts
streaming  streamed 4.87 GiB over 5475 expert reads, 4125 cache hits (43%), 0 evictions
generated  6 tokens in 4.1s (1.46 tok/s)
```

**43% hits and zero evictions**, against V4-Flash's 1.9-4.1% and thousands of
evictions with a cache a fifth the size. The difference is not the policy and not
the model's routing — it is whether a step re-reads the sequence. That is the
whole argument for doing R3 first, and it is already running in this repository.

## Decisions this fixed in the code

- **The cache is off by default.** Turning it on would ship a measured 17%
  prefill regression for a measured 0% gain. `--cache <GiB>` forces it, and the
  runner prints why it is off rather than staying silent.
- **Frequency must be weighted by selections, not requests.** Because reads are
  deduplicated, an expert chosen by ninety tokens and one chosen by a single
  token arrive as one request each. Unweighted, every count ties at 1 on a long
  prompt, nothing can ever beat an incumbent, and the cache freezes permanently
  on whatever layer 0 loaded first. This was found by reasoning about the
  mechanism before the benchmark ran, and it would have been invisible in the
  results — a frozen cache and a useless cache score the same 2%.
- **Nothing is pre-loaded**, per R0: a hot set chosen in advance covers 37.5% of
  an unseen subject against 25% for caching at random.
- **The cache owns its memory** — heap this process holds, not pages the kernel
  can reclaim. Past ~6 GiB on Qwen3 a page-cache-backed 71%-hit cache was the
  *slowest* configuration ever measured here.

## Verification

`the_expert_cache_does_not_change_the_answer`, container-backed, against the real
144 GB model: two prefills through one cache, asserting the cold pass admits, the
warm pass actually hits (a cache that never hit would pass an equality check
trivially), then argmax equality and the llama.cpp oracle sum. Plus nine unit
tests on the policy itself, including both directions of the frequency gate and
the budget bound.

## Caveats

- **One machine, one model, two cache sizes.** On a 48-80 GiB desktop the budget
  would be 20-40x larger and the conclusion could differ even before R3 — but
  nobody has such a machine here, so that is unmeasured, not implied.
- **The 17% prefill regression is the admission copy**, inferred from where the
  time went rather than profiled directly.
- **Hit rate is not tok/s.** Both are reported above, next to the footprint,
  because this project has already measured a 71%-hit cache being its slowest
  configuration.
