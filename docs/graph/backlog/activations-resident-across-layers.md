# Phase 2: keep activations on the device across layers

**Status: step 0 DONE (2026-08-16), 1.33-1.52x -> 2.49-2.65x. Steps 1-3 open.**

`ggml_gallocr` is in, wired at all seven call sites via `Compute::realize_graph`,
and the reuse is measured: a seven-tensor chain plans into 3 MB against a naive
7 MB. Logit checksums are unchanged, so the speedup cost no correctness.

**What remains is transfers — upload 2.01s, download 1.85s across three runs.**
Steps 1-3 below all need the same thing: the QKV and attention graphs sharing a
context, so `q` (8.4 MB each way, 605 MB per prefill) stops round-tripping and
the residual can be a graph op. Today attention builds in a scratch-buffer
context sized for the CPU path's memory reuse, and merging them naively would
grow host arenas for every architecture. That is the open design question, and
it is the reason steps 1-3 did not follow step 0 immediately.

**Original status:** open, designed, sized, not started. Approved route: one context per
pass. Explicitly **not** a device-only duplicate of the layer body — this project
has deleted five dead forward paths and must not grow a sixth.

Links: [../research/phase-a-device-prefill-2026-08-15.md](../research/phase-a-device-prefill-2026-08-15.md) ·
[../research/mixed-residency-segfaults-2026-08-15.md](../research/mixed-residency-segfaults-2026-08-15.md)

## What it is worth, measured not guessed

From `chaos-gpubench`'s per-operation counters on a 512-token Qwen3-4B prefill:

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

## The naive route does not fit in VRAM, and here is the arithmetic

**Labelled arithmetic, not a measurement** — but it is decisive enough to change
the design before anyone writes code.

Keeping `x` resident across layers means the tensor produced by layer N must
still be alive when layer N+1's graph is built. With
`ggml_backend_alloc_ctx_tensors_from_buft` that means keeping every layer's
context *and its whole device buffer* alive for the pass, because allocation is
per context and all-or-nothing.

Per-layer intermediates at 512 tokens on Qwen3-4B (n_embd 2560, n_head 32,
head_dim 128, n_ff 9728), counting only the large ones:

| | approx |
|---|---:|
| QKV: q/k/v plus normed and roped variants | ~38 MB |
| attention: permuted q/k/v, mask, output | ~25 MB |
| FFN: gate and up at 9728x512, act, down | ~60 MB |
| **per layer** | **~120 MB** |
| **x 36 layers** | **~4.3 GB** |

Free VRAM after the 2.32 GiB of weights is **2.79 GiB**. It does not fit, and it
does not fit by enough that trimming will not save it.

**So the route is `ggml_gallocr`, not a pass-wide context.** That is the API
llama.cpp uses and the reason it runs a whole-model graph in a modest buffer:
it computes a memory plan that reuses allocations for tensors whose lifetimes do
not overlap.

```c
ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);
bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);
```

This is a **new FFI surface and a different allocation model** from the one
`Compute::realize` wraps today, and it should be built and tested on its own —
against a graph whose answer is already known — before the forward pass is
rewritten on top of it. `ggml_gallocr_get_buffer_size` is what proves the reuse
is real rather than assumed.

## Order of work

0. **`ggml_gallocr` first, on its own.** New FFI, new allocation model, proved
   against a known answer and against `ggml_gallocr_get_buffer_size` showing
   real reuse. Everything below depends on it; see the arithmetic above.
1. **The residual add becomes a graph op.** Today it is a host loop —
   `for (dst, v) in ffn_input.iter_mut().zip(attn_out)` — which forces a
   download of the attention output and an upload of the FFN input on every
   layer. It is `ctx.add` once both are in one graph.
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
- `chaos-gpubench --repeat 3` reports the new ratio, with the warm-up
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
