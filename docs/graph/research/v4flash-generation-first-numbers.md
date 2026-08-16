---
topic: V4-Flash generates — at 0.042 tok/s, ten times slower than llama.cpp
status: open
links: [zero-copy-expert-reads.md, v4flash-speed-budget.md, head-to-head-llamacpp-2026-08-05.md, ../backlog/lts-0-0-0.md]
---

The T0 gate of `lts-0-0-0.md` says: *generation exists and beats llama.cpp;
if we do not win, say so and do not ship the claim.*

**Generation exists. We do not win.**

## Measured, 2026-08-07

```
./target/release/chaos-run.exe \
  C:/Projects/models/v4flash/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf \
  "The capital of France is" -n 5

resident   loaded 101 tensors, 3.97 GiB of 3.97 GiB budget in 3.4s
           3.40 GiB will be re-read from disk on EVERY token
prefill    5 tokens in 20.9s (0.24 tok/s)
output      Paris.",
    "label
generate   4 tokens in 95.6s (0.042 tok/s, 23.9s per token)
```

llama.cpp on the same model and machine: **0.45 tok/s** (command line and output
in `head-to-head-llamacpp-2026-08-05.md`). We are **10.7x slower**.

The output is coherent — the model continues into what looks like JSON, which is
what an un-templated completion of that prompt should do.

## Why: the loop has no KV cache, and that costs I/O, not attention

Each generated token re-runs the whole sequence. The obvious objection is the
quadratic attention cost, and that is **not** the problem at these lengths — a
forward pass is dominated by reading routed experts, which is paid per *pass*,
not per token.

The real cost is expert selection:

| pass | unique expert slices per layer | expert bytes |
|---|---|---|
| 5 tokens together | ~26 of 256 | ~15.9 GiB |
| 1 token with a cache | 6 of 256 | ~3.3 GiB |

**A KV cache is worth ~3-4x here, and it buys it on disk traffic rather than on
arithmetic.** That is the opposite of the usual reason to want one, and it is
worth writing down because it changes what "adding a KV cache" is for.

## The profile, re-measured per phase

`CHAOS_BLOCK_TIMING=1`, block 42 of a 5-token pass:

```
arena 0.00   dense 0.08 (147 MiB)   qkv 0.01   attn 0.01   tail 0.10   ffn 0.30
                                                                       total 0.54
```

Scaled to the whole 43-block, 20.4s forward pass:

| phase | share | note |
|---|---|---|
| expert reads (inside `ffn`) | ~47% | 9.5s — still the largest single item |
| **`layer_tail` + `moe_routing`** | **~21%** | **4.3s, and nobody has looked at it** |
| dense reads | ~17% | 3.4s — would be **0** with full residency |
| expert compute (rest of `ffn`) | ~17% | 3.4s |
| attention + Q/KV | ~4% | 0.9s — the part everyone assumes is expensive |
| arena allocation | ~0% | the 1 GiB per-block arena costs nothing measurable |

Two things here were not expected.

**Attention is 4%.** On a model whose distinguishing feature is three kinds of
compressed attention, the attention is a rounding error. Optimising it would be
worthless.

**`layer_tail` + `moe_routing` is 21%** — as much as all the dense I/O — and it
is a handful of small ops: the post hyper-connection, the FFN gates, `ffn_norm`,
and the router's `argsort_top_k` over 256 experts. There is no obvious reason
for it to cost 0.10s per block. This is the next thing to measure, and it is
listed as a measurement rather than a hypothesis on purpose.

## Threads: a hypothesis raised and refuted in one command

```
CHAOS_THREADS=4    prefill 5 tokens in 20.4s
CHAOS_THREADS=12   prefill 5 tokens in 20.5s
CHAOS_THREADS=20   prefill 5 tokens in 20.7s
```

Five times the threads, no change. The natural explanation was **threadpool
churn**: `Context::compute` calls `ggml_graph_compute_with_ctx`, which builds a
fresh threadpool every call, and the forward pass makes 32 such calls per block
— **1376 create/join cycles per pass**. If each cost ~12 ms that is the entire
non-I/O budget, and it would explain the flat curve exactly, since more threads
would buy parallelism and pay for it in spawn cost.

It is wrong. Extending the sweep downwards refutes it in one line:

```
CHAOS_THREADS=1    prefill 5 tokens in 94.8s     <- no threads spawned at all
CHAOS_THREADS=2    prefill 5 tokens in 23.0s
CHAOS_THREADS=12   prefill 5 tokens in 20.0s
```

One thread spawns nothing, so under the threadpool theory it should have been
the *fastest* configuration. It is **4.7x the slowest**. The work is real
arithmetic; it simply stops scaling after two threads.

So the shape is: a large parallel component that saturates by ~2-4 threads
(the ops are small at 5 tokens and cannot use 12 cores), on top of ~13s of I/O
that no thread count touches. Going past 4 threads buys nothing and is not
where the time is.

**Nought for four now.** Parallel expert reads were slower; contextual sparsity
was absent; residency was "likely a large multiple" and was 22%; threadpool
churn was the obvious explanation and was not the explanation. The pattern is
consistent enough to be a rule: on this project, measure first, and extend the
sweep past where the answer seems obvious.

## What would have to be true to reach 0.45 tok/s

Multiplying the levers against the measured 23.9 s/token:

| lever | effect | confidence |
|---|---|---|
| KV cache (6 slices/layer, not 26) | expert I/O ÷ 3-4 | high — arithmetic on measured counts |
| Full residency (needs ~4 GiB more free RAM) | dense I/O → 0 | high, but it is the **user's** RAM |
| Expert cache across generated tokens | routing overlaps between adjacent tokens | unmeasured on this model |
| Compute | now 42% and unexamined | unknown |

The first two together plausibly reach ~3-5 s/token, i.e. **0.2-0.33 tok/s** —
which would be within sight of 0.45 but still short of it. Reaching parity, let
alone a win, requires the compute half as well, and nothing has profiled it yet.

**No claim of beating llama.cpp is supportable today, and none is being made.**

## The structural claim that is untouched by this

The long-context argument from `head-to-head-llamacpp-2026-08-05.md` — that
llama.cpp's dense weights get evicted by cold expert traffic and ours do not —
is about a regime none of these numbers touch. It remains the strongest
available claim and it remains unmeasured on V4-Flash.

## Next

1. **Profile the compute half.** It is 42% and nobody has looked at it. 430
   `compute()` calls per forward pass, each building and running a graph, is a
   suspicious shape — but that is a hypothesis, and this node exists partly to
   record that hypotheses here have a poor track record.
2. KV cache for the raw window (bounded at `sliding_window` = 128 entries) and
   the compressed summaries (bounded at 256 blocks). All 43 layers together are
   ~33 MB of state, so the cache is small; the risk is correctness, since a
   wrong cache on this architecture yields fluent nonsense rather than an error.
   It needs its own oracle capture at two consecutive positions.
