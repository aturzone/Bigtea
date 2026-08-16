# The GPU is a dense-model dial. On a streaming MoE model it is a 4.3x loss

**2026-08-16.** Qwen3-30B-A3B-Q4_K_M (17.28 GiB container, 48 blocks, 128
experts), RTX 3050 6 GB via Vulkan, i7-13650HX with 15.7 GiB RAM. Three runs per
point, medians, `-n 4`.

`-ngl` is a smooth, monotonic win on a dense model
([ngl-frontier](ngl-frontier-2026-08-16.md): 1.79x prefill, 1.40x generation on
Qwen3-4B). On the model this project actually exists for — one that does not fit
in RAM and streams its experts from disk — **it makes things worse, by a lot.**

| | prefill tok/s | generation tok/s |
|---|---:|---:|
| CPU only | 1.30 | **2.61** |
| `-ngl 12` (of 48) | 1.30 | 1.44 |
| `-ngl 48` | 1.09 | **0.61** |

Generation runs, showing the spread is not the story: `[2.58 2.66 2.61]`,
`[1.44 1.42 1.45]`, `[0.61 0.59 0.61]`. Under 2% either side of the median.

**4.3x slower at full offload.** Prefill is flat.

## Why, and it is not a bug

Two facts about this model, both already in `CLAUDE.md`:

- **76% of a token is disk.** A single-token step reads 3.21 GiB of expert
  weights; the run above streamed 4.58 GiB over 5157 reads at a 29% cache hit
  rate.
- **The experts run on the host, whatever `-ngl` says.** They are streamed into
  host memory per block and their FFN builds its own CPU context. `-ngl` places
  only *resident* weights — attention, norms, the router.

So offloading moves the small part and leaves the large part where it was. What
it adds is a host round trip for the activation at **every one of 48 blocks**,
plus the KV cache crossing the bus in both directions. That is pure latency
against work the card never does.

The 0.93 GiB uploaded at `-ngl 99` is the whole resident set — 5% of what a
token actually reads.

## What this rules out, and what it leaves

**It rules out the GPU as the answer for huge models on this design.** Not
"needs tuning": the card cannot help with a cost it never touches, and putting
the experts on it is not available either — this model's experts are ~16 GiB
against 5.11 GiB of VRAM, which is the same wall that made the model stream in
the first place.

It leaves the levers that were already the honest ones for this class of model,
and `v4flash-has-no-slack-2026-08-10.md` has already closed the byte-reduction
half of that list. What survives is disk and scheduling, not arithmetic.

## The rule this generalises to

**A speedup measured on a model that fits does not transfer to one that does
not.** The two are bound by different resources, and every GPU number this
project has published — 25.6x on a kernel, 1.33-1.52x on a Qwen3-4B prefill,
1.79x on the `-ngl` frontier — was measured on a model that fits. None of them
predicted this one.

`chaos-run` now warns when a device is opened on a model that streams experts,
with the measurement in the message. Finding this out four minutes into a 17 GiB
run is worse than being told.
