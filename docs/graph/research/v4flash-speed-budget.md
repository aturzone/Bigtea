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

## The expert step, decomposed (2026-08-06, measured)

The 15.4s "expert reads" figure was a timer around read *and* copy *and* bind. Split:

```
pure disk read    11.3s   11.5 GiB at 1.02 GiB/s
copies + bind      5.9s   34% of the step
                  ------
                  17.1s
```

And the disk's own ceiling, measured by `bigtea-probe` (not assumed):

```
sequential, 8 MiB blocks, sized above RAM:   2.55 GB/s
expert slices, 12.7 MiB at scattered offsets: 1.10 GB/s   ← 43% of it
```

Two separate problems, both worth real seconds:

1. **34% of the step is memcpy.** `read_tensor_range` allocates a `Vec`, `extend_from_slice`
   concatenates into the compact stack, `bind` takes an `Arc<[u8]>`. Reading directly into one
   pre-sized buffer removes a full copy of 11.5 GiB. **Worth ~5.9s**, and it needs no new
   idea — the project's own facts list already says memcpy was the largest cost in generation.
2. **The disk is running at 43% of its own sequential rate.** 12.7 MiB is a large read, so
   this is not per-request latency — parallelising it made things *worse*. It is scattered
   offsets across five shards with no readahead, under cache-bypassing direct I/O. **Worth up
   to ~6.5s** if it can be closed, and the first thing to check is whether `bigtea-io` splits
   these into smaller physical reads.

Together the expert step could go 17.1s → ~5s.

## Can V4-Flash run well in 8 GiB?

Yes, with one trade. The arithmetic:

```
per-block dense weights   145 MiB x 43   =  6.09 GiB   must be resident
token_embd                                  0.53 GiB   NOT needed resident: get_rows touches
                                                       only the prompt's rows, ~10 KB
output.weight                               0.53 GiB   read once per token, keep resident
activations + arena                       ~0.50 GiB
                                            --------
                                            7.12 GiB   fits 8 GiB, ~0.9 GiB left for cache
```

**Dropping `token_embd` from residency is free** — it is a lookup table, and a forward pass
reads only the rows for its own tokens. That alone is what makes 8 GiB feasible at all.

Speed then follows the expert bytes, and only those:

| expert bits | GiB/token | tok/s | vs llama.cpp 0.45 |
|---|---|---|---|
| 4.25 (MXFP4, today) | 3.21 | 0.87 | 1.9x |
| 3.0 | 2.27 | 1.23 | 2.7x |
| 2.5 | 1.89 | 1.48 | 3.3x |

Plus ~0.9 GiB of cache (roughly 70 slices of 11,008) — a small hit rate, maybe 10-15%,
worth another ~1.15x.

**So ~1.5 tok/s in 8 GiB is reachable, at 2.5-bit routed experts.** That is 3.3x llama.cpp
and the first configuration that would be genuinely usable.

The trade is quality, and `sub-2bit-k3-fixed-hardware.md` is unambiguous about the shape of
it: **scalar 2-bit collapses** (GPTQ/AWQ hit 10^4-10^6 perplexity), so this needs
additive/residual VQ in the AQLM family, which reaches 2-3 bpw on dense 7B-70B — **never
demonstrated on an MoE this size**. It also needs a CPU decode kernel. That is a research
project, not an afternoon.

**What does not need a research project: 4.25-bit experts at 0.87 tok/s, which is already
1.9x llama.cpp**, once residency, the copy removal, and the read-rate gap are done. That is
the honest 8 GiB target to aim at first.

## The ceiling by hardware class — where the speed actually is

The 8 GiB question is the wrong one to optimise for. **Below ~150 GiB of RAM the bottleneck is
disk and the ceiling is low no matter what the code does.** Above it, the model stops touching
disk at all and the ceiling jumps by an order of magnitude. The interesting question is what
V4-Flash can do on a machine that fits it.

Per token V4-Flash touches **10.59 GiB of weights**: 7.38 GiB always-read plus 3.21 GiB of
routed experts. Which tier that traffic comes from is the entire story.

| class | RAM | what streams | bottleneck | ceiling |
|---|---|---|---|---|
| this laptop | 15.7 GiB (7.5 usable) | all 3.21 GiB of experts | NVMe 2.55 GB/s | **0.87 tok/s** |
| desktop, 64 GiB | ~50 GiB cache | ~40% of experts | mixed | **~2 tok/s** |
| desktop, 192 GiB DDR5 dual | nothing | RAM ~60 GB/s | dequant | **~5.6 tok/s** |
| workstation, 8-channel DDR5 | nothing | RAM ~300 GB/s | dequant | **~28 tok/s** |

Two things follow, and they change what is worth building.

**1. The laptop ceiling is 0.87 tok/s and no engineering passes it.** 3.21 GiB per token over a
2.55 GB/s disk is 1.26 s/token. That is 1.9x llama.cpp's measured 0.45 — a real win, and the
most this machine will ever give. Chasing 5 tok/s here is chasing a number the hardware cannot
produce.

**2. Once the model fits RAM the bottleneck stops being I/O and becomes dequantisation**, which
is exactly where Bigtea currently *loses* to llama.cpp on Qwen3 (1.07 vs 2.16 tok/s,
generation). Expert compute there is neither barrier-bound nor bandwidth-bound — 2.4 GB/s
against DDR5 — it is unpacking Q4_K one block at a time while llama.cpp interleaves rows so
several unpack per SIMD op. **On a big machine that gap is the whole race.** The repacking work
already scoped in `CLAUDE.md` item 3 is not a laptop optimisation; it is what decides the
desktop and workstation numbers above.

## What "fully winning against llama.cpp" requires, concretely

Three separate wins, in the order they become provable:

**a. Short context, disk-bound — 1.9x, needs only engineering already scoped.** Build
generation, pin the resident set, remove the copy, close the 43% read-rate gap. 0.87 vs 0.45.

**b. Long context — the structural win, and the only one llama.cpp cannot copy.** llama.cpp
mmaps all 144 GB, so its dense weights compete with 137 GiB of cold expert traffic and get
evicted. Its projected dense re-read climbs to 7.38 GiB/token at 128k. Then:

```
llama.cpp @128k:  (7.38 dense re-read + 3.21 experts) / 2.55 GB/s  =  4.2 s/token  =  0.24 tok/s
Bigtea    @128k:   3.21 experts only                  / 2.55 GB/s  =  1.26 s/token =  0.79 tok/s
                                                                                       ~3.3x
```

**That gap widens with context and is a property of the design, not the code.** It is also
completely unmeasured, because generation does not exist yet.

**c. In-RAM, compute-bound — currently a LOSS, ~2x behind on Qwen3.** Requires expert repacking
on cache admission. Until that lands, any claim about big machines is a claim we would lose.

The order matters: (a) and (b) are winnable with work already understood. (c) is where the
"big bang" lives, and it is a kernel problem, not an I/O problem. Sub-2-bit experts multiply
whichever regime you are in — but per `sub-2bit-k3-fixed-hardware.md`, scalar 2-bit collapses
and the VQ methods that work have never been shown on an MoE this size.

## The "1.06 GiB/s disk" is not the disk (2026-08-06)

Every measurement in this node called `read_tensor_range` "disk time". It is not. The path is
`direct.rs:118-151`:

```rust
let mut buf = AlignedBuf::new(span);        // allocate (and the OS zeroes fresh pages)
let got = self.read_exact_at(&mut buf, ...)?;  // the actual read
Ok(buf[skip..skip + len].to_vec())          // ALLOCATE AGAIN AND COPY EVERYTHING
```

**Every read copies all of its bytes one extra time inside the I/O layer**, on top of the two
copies already identified higher up (`extend_from_slice` into the compact stack, then `bind`
into an `Arc<[u8]>`). That is **three full copies of 11.5 GiB per prefill**, plus the
allocation and page-zeroing of both buffers.

This reframes every number here:

- The 1.06 GiB/s "expert read rate" is *allocate + read + copy*, not read.
- The 43% gap against 2.55 GB/s sequential is therefore not a seek penalty or an access-shape
  problem, and was never evidence for one.
- **It also explains why parallelising the reads made things worse**: twelve threads
  contending on the allocator and memory bandwidth, not on the disk.

That is the third time in this project that memcpy has turned out to be the cost — after the
Qwen3 generation profile and the expert-step split above. It is already the entry in
`CLAUDE.md` that says to profile first, and it has now been rediscovered twice since.

### The fix, and what it is worth

`read_at` should fill a caller-owned buffer rather than returning a fresh `Vec`. The expert
path then reads each slice **directly into its final position in the compact stack**, which
removes both the `to_vec` and the `extend_from_slice` — two of the three copies — and the
second allocation.

If the disk is genuinely near 2.55 GB/s once the copies are gone, 3.21 GiB/token becomes
**1.35 s/token = 0.74 tok/s**, against a currently-measured effective rate that would put
generation nearer 0.33. **This single change is plausibly the difference between losing to
llama.cpp's 0.45 tok/s and beating it**, and it is ordinary engineering with no quality risk
and no research.

**Measure it first**: time `read_exact_at` alone versus `read_at` as a whole. If the copy is
not the gap, the API change is not worth making.
