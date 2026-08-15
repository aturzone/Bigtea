# Phase 2: keep activations on the device across layers

**Status: open, designed, sized, not started.** Approved route: one context per
pass. Explicitly **not** a device-only duplicate of the layer body — this project
has deleted five dead forward paths and must not grow a sixth.

Links: [../research/phase-a-the-card-at-1.7x-2026-08-15.md](../research/phase-a-the-card-at-1.7x-2026-08-15.md) ·
[../research/mixed-residency-segfaults-2026-08-15.md](../research/mixed-residency-segfaults-2026-08-15.md)

## What it is worth, measured not guessed

From `bigtea-gpubench`'s per-operation counters on a 512-token Qwen3-4B prefill:

| | seconds | calls |
|---|---:|---:|
| graph compute | 1.80 | 110 submissions |
| upload | 1.04 | |
| download | 0.66 | |
| realize (device allocation) | 0.64 | 110 allocations |

Transfers are 36% of device time and allocation another 14%. Removing most of
both, and collapsing 110 graph submissions toward ~36, should take the ratio
from **1.33–1.52x to roughly 2.5–3x**.

It will **not** approach llama.cpp's 2042 pp512, which runs one graph for the
whole pass with no host round trips at all. Say so in the node when it lands.

## The actual obstacle, which is not the context

`forward_cached_inner` carries the activation as `x: Vec<f32>` — **host** memory
— and that is not incidental. It is consumed as a host slice in several places:

- pushed into the KV cache per layer (`cache.push`, which takes bytes)
- handed to the MoE router and the expert loop, which read host vectors because
  streamed expert bytes land in host memory
- sliced for the output head: `&x[last..]`

So Phase 2 is **not** "hoist the context out of the loop". It is changing what
`x` *is* — something like

```rust
enum Activation<'a> {
    Host(Vec<f32>),
    Device(Tensor<'a>),
}
```

and giving it the operations the function currently gets for free from `Vec`.
Every consumer above has to state which it needs, and the ones that genuinely
need host bytes (the cache push, the expert loop) become explicit download
points rather than accidental ones.

## Order of work

1. **One context for the pass, device path only.** On the CPU the per-layer
   arenas exist so memory is reused; a single pass-wide arena would grow host
   memory for every architecture. On a device the context is `no_alloc` and
   holds metadata only, so a pass-wide context costs nothing there.
2. **`x` stays a tensor across the layer boundary.** For a dense model there is
   *no* host work between layers, so this is the whole win: 36 downloads and 36
   uploads of a 5.24 MB activation disappear.
3. **`q` stops round-tripping.** Build attention in the same context as QKV. The
   KV cache push still needs `k`/`v` downloaded, but `q` — the largest single
   tensor at 8.4 MB — never leaves.
4. **Only then** consider fusing across layers into fewer graph submissions.

## Acceptance

- **CPU output byte-identical** on the parity prompts across all thirteen
  entries in `VERIFIED_ARCHITECTURES`. This is the gate; the port that produced
  Phase A cleared it once and this must clear it again.
- `bigtea-gpubench --repeat 3` reports the new ratio, with the warm-up
  discarded, and the per-operation counters show the transfers actually gone
  rather than merely a better wall clock.
- Device and CPU logit checksums still agree.

## Traps already paid for

- **A tensor written before its context is realized is a segfault, not an
  error.** Three of them in one session. Any builder that both creates graph
  nodes and writes into them must be split — see the mixed-residency node.
- **A mixed host/device context builds correctly and then dies at compute**, so
  step 1 cannot be a partial migration where some tensors stay host-bound
  inside the same context.
- **Do not measure with anything expensive inside the timed region.** The first
  version of the harness reloaded 2.32 GiB per run and produced a 2.5x spread in
  the CPU baseline.
