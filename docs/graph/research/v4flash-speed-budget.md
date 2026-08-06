---
topic: What V4-Flash speed is physically available on this machine, and which levers actually reach it
status: open
links: [head-to-head-llamacpp-2026-08-05.md, v4flash-port-recon.md, hardware-profiling.md]
---

Written after the first V4-Flash prefill was profiled and one optimisation failed. The point
is to stop guessing which change is worth making by deriving the bound first.

## The bound

Generation reads **3.21 GiB of routed experts per token** — 6 of 256 experts, three tensors
each, across 43 blocks. Nothing changes that except reading fewer bytes.

```
cold, no cache:   3.21 GiB / 2.79 GB/s  =  1.15 s/token  =  0.87 tok/s   ← hard ceiling
llama.cpp today:                                             0.45 tok/s   ← measured
```

So **llama.cpp is beatable by about 2x, and 2 tok/s is not reachable cold by any amount of
kernel work.** Anyone promising more is promising fewer bytes, not faster code.

Reading fewer bytes means cache hits. With hit rate `h` the effective read is
`3.21 * (1 - h)` GiB:

| hit rate | GiB/token | s/token | tok/s |
|---|---|---|---|
| 0% | 3.21 | 1.15 | 0.87 |
| 50% | 1.61 | 0.58 | 1.7 |
| 70% | 0.96 | 0.35 | 2.9 |
| 90% | 0.32 | 0.12 | 8.7 |

**That table is the whole opportunity.** It is also where the trap is, and this project has
already been caught by it once: *"past ~6 GiB the expert cache reaches 71% hits and is the
slowest configuration measured, because cached bytes get paged out — a hit becomes a page
fault wearing a disguise."* Hit rate only converts to speed while the cache stays resident.

## Which means the binding constraint is RAM, not code

```
resident (always-read) weights   7.38 GiB   must stay pinned
useful expert cache              4+ GiB     to reach 50-70% hits
                                 ------
                                 ~12 GiB    to run V4-Flash well
machine total                    15.7 GiB
usable for weights                7.5 GiB   ← with Dota 2 closed; 4.2 with it open
```

**This is knife-edge, and it moves with what is open.** With Dota 2 running, usable was 4.2
GiB and the resident set could not fit at all. With it closed, 7.5 GiB — the 7.38 GiB
resident set fits with 0.1 GiB to spare and **no room for an expert cache**. Closing the
remaining editors and chat apps takes it to ~10.0 GiB, which is the first point at which a
~2.5 GiB cache becomes possible at all.

So the honest reading: **residency is reachable today; a cache large enough to matter is not,
without either freeing more or adding RAM.** `bigtea-probe --quick` reports exactly what to
close and what it is worth. It is the single largest speed lever available and costs nothing
to pull.

## Prefill, where the measurements are

Measured, 5 tokens, 43 blocks, 31.4s total:

```
expert reads  15.4s   49%   11.5 GiB
dense reads    7.1s   23%    6.2 GiB — the resident set, re-read 43 times
compute        8.9s   28%
```

Three changes, with what the arithmetic says each is worth:

1. **Stop re-reading the resident set.** 7.1s → ~0.2s. Bounded gain: **~7s**. Needs the
   resident set to fit, which needs the RAM above.
2. **Stop copying every slice twice.** `read_tensor_range` allocates, `extend_from_slice`
   concatenates, `bind` takes an `Arc`. 11.5 GiB copied twice at DDR5 speeds is several
   seconds of the 15.4. Bounded gain: **~5s**, and cheap to measure before building.
3. **Overlap reads with compute.** They are strictly serial today. Attention needs no expert
   weights, so block *N*'s expert reads can run under block *N*'s attention. Bounded gain:
   **~min(reads, compute) ≈ 9s**.

Together: 31.4s → roughly 10s, about **3x**. That is the honest prefill headroom before
anything clever.

**What is NOT worth doing**, from the failed attempt: parallelising the slice reads. Each
slice is ~12.7 MiB, so the workload was never latency-bound; twelve threads made it 14%
slower.

## The one structural advantage that is still unproven

Everything above is engineering both engines could do. The design difference is this:
llama.cpp mmaps all 144 GB and lets the kernel's LRU decide, so its dense weights compete
with 137 GiB of cold expert traffic and get evicted — `bigtea-model-info` projects the dense
re-read per token climbing from 0.06 GiB at 4k context to **7.38 GiB at 128k**, which is the
entire dense set being re-read every token. Bigtea pins that set and it never moves.

At 4k context that is worth little. At 128k it is worth 7.38 GiB/token — more than twice the
expert traffic. **That is the regime where the gap should be structural rather than
incremental, and it has never been measured.** Measuring it needs generation, which does not
exist yet.

## Order of work this implies

1. **Free RAM** (probe already says what to close). Largest lever, zero engineering.
2. Time the read separately from the concatenation — confirm the copy cost before removing it.
3. Residency: load the resident set once, not per block.
4. Overlap expert reads with attention compute.
5. Generation, then the expert cache, then the long-context comparison that the whole thesis
   rests on.

Steps 2-4 are ~3x on prefill. Step 5 is where "better than llama.cpp" is either proven or
retracted.
