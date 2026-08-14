# The smallest honest first slice of a GPU tier

**2026-08-11.** Scoping node, written **before** any GPU code, as instructed.
Status: **open** — one recommendation, and it is not the one that was guessed.

Links: [v4flash-has-no-slack-2026-08-10.md](v4flash-has-no-slack-2026-08-10.md) ·
[r2-overlap-2026-08-11.md](r2-overlap-2026-08-11.md) ·
[../backlog/the-big-bang.md](../backlog/the-big-bang.md) ·
[expert-cache-is-early-not-wrong-2026-08-08.md](expert-cache-is-early-not-wrong-2026-08-08.md)

## The hypothesis handed over

> my guess is "N dense layers resident in VRAM, experts still streamed to
> host", but measure before believing me.

Measured. **It does not survive on either MoE model on this machine**, for two
different reasons. A third slice does survive, and it needs neither a CUDA ggml
nor a second binding path.

## The hardware, and blocker (a) is worse than recorded

```
gpu   NVIDIA GeForce RTX 3050 6GB Laptop GPU   6.0 GiB   [nvidia-smi]
ram   15.7 GiB total, 8.6 GiB available
      driver 610.74
```

`STATUS.md` says a VRAM tier needs "a CUDA-enabled ggml". The actual state is a
step further back:

```
$ which nvcc
which: no nvcc in (...)
$ ls build/ggml/src/
ggml-base.a  ggml-cpu.a  ggml.a          # no ggml-cuda.a
```

**There is no CUDA toolkit on this machine at all**, only a CUDA-capable driver.
So the first cost of any GPU work is a toolkit install and a second ggml build
configured `-DGGML_CUDA=ON`, before a line of Rust is written. That is not a
reason not to do it; it is a reason not to describe it as a flag.

## Blocker (b), named exactly

Every weight in this engine is a **host pointer written straight into
`tensor->data`**. One function, one line:

```rust
// crates/bigtea-ggml/src/weights.rs:286
let ptr = data.as_bytes().as_ptr() as *mut c_void;
unsafe { tensor.set_data_ptr(ptr) };
```

That is the whole memory design. `ggml-cuda` cannot be handed a host pointer:
a device tensor is allocated through a CUDA buffer type and filled with
`ggml_backend_tensor_set`, which **copies**. And a copy is exactly what this
project cannot afford — the note in `CLAUDE.md` about needing 2× the model is
about this line.

So a GPU path is not a flag on `bind_shared`; it is a second implementation of
it, plus a `ggml_backend_sched` to run a graph whose tensors live on two
devices. Anything that silently falls back to CPU when the device allocation
fails reproduces the failure mode this project keeps paying for.

## Why "N dense layers in VRAM" fails, twice

`bigtea-model-info --budget 6`, the two MoE models on disk:

| model | always-read (dense) | routed experts | total |
|---|---:|---:|---:|
| DeepSeek-V4-Flash-UD-Q4_K_XL | **7.38 GiB** | 137.06 GiB | 144.44 GiB |
| Qwen3-30B-A3B-Q4_K_M | **0.93 GiB** | 16.35 GiB | 17.28 GiB |

**V4-Flash: the dense half does not fit.** 7.38 GiB of always-read weights
against 6.0 GiB of VRAM. "N dense layers" therefore means a *partial* residency
with a mixed-device graph and a scheduler — which is the largest possible first
slice, not the smallest.

**Qwen3-30B: it fits with 5 GiB spare, and there is almost nothing to move.**
One run, `-n 4`, this build:

```
prefill    5 tokens in 4.6s
generated  4 tokens in 1.9s
streaming  resident 0.93 GiB, streamed 4.61 GiB over 5190 expert reads in 2.8s,
           2124 cache hits (29%)
time: 2.8s disk, 0.2s qkv, 0.2s attention, 0.1s ffn,
      2.0s expert compute, 0.0s slice copies, 0.1s other
```

Of 5.4 s accounted: **disk 52%, expert compute 37%, and the entire dense path —
qkv + attention + ffn — 9%.** Moving that 9% to a GPU is a **1.10x ceiling**,
and it is below the 1.4x already sitting unclaimed in R2's read/compute overlap.

Moving the *expert* matmuls instead would address 37%, but the experts are the
bytes that stream, so it needs ~1.15 GiB/token pushed over PCIe. This project
has already built and reverted the same shape: making the selected experts
contiguous cost ~1.02 GB/token, was byte-identical, and went **1.34 → 1.27
tok/s**. A PCIe copy is that transfer with a bus in the middle.

## The slice that does survive: VRAM as a read cache, not a compute device

Put nothing on the GPU. Use its 6 GiB as a **second tier of the expert cache**,
in front of the disk and behind host RAM. An expert slice is read once from
NVMe, kept in VRAM, and copied back to host on a hit.

Why this is the smallest honest slice:

* **It does not touch `bind_shared` at all.** No weight is ever a device tensor.
  Bytes land in host memory in exactly the layout the CPU path already binds, so
  blocker (b) is sidestepped rather than solved.
* **It needs no `ggml-cuda`.** Allocation and two `cudaMemcpy` directions is the
  entire surface — the CUDA *runtime*, not a second ggml build. (It still needs
  the toolkit for `cuda.h` and the driver API, so blocker (a) stands.)
* **It cannot silently fall back and be wrong.** A cache miss is a disk read,
  which is the current behaviour. Correctness is unaffected by the tier existing
  or not, so this is the one GPU change whose failure mode is *slower*, not
  *wrong*.
* **It is the frontier sweep's missing axis.** The tok/s-versus-RAM curve (item
  1 in `STATUS.md`'s Next) is exactly the measurement this feeds: VRAM is
  another N GiB of owned residency, and only an engine that owns residency can
  place it.

### The arithmetic, and which parts are not measured

*Measured this session:* 4.61 GiB streamed over 4 tokens = **1.15 GiB/token** at
2.8 s, i.e. ~1.65 GiB/s effective on a warm-ish cache. Per-handle NVMe reads
measured earlier at **2.65 GiB/s**.

*Not measured — no CUDA toolkit, so this is arithmetic and must be labelled:*
a laptop RTX 3050 is PCIe 3.0 x8, nominally ~7.9 GB/s, realistically ~6 GB/s
for pinned host memory. If that holds, a VRAM hit is **~2.3x faster than the
disk read it replaces**.

*Coverage, which is the part that decides everything:* 5 GiB of usable VRAM
against a **16.35 GiB** expert bank is 31% of Qwen3-30B — but against
**137 GiB** it is 3.6% of V4-Flash. So:

> **The VRAM cache pays where VRAM is a meaningful fraction of the expert bank.**
> That is the 20–70 GiB class of model, not the 144 GiB one. On V4-Flash it is
> noise, and V4-Flash is the model this project talks about most.

Combined with the host cache, Qwen3-30B would have 6.26 + 5 = **11.26 GiB of
owned cache against a 16.35 GiB bank on a 15.7 GiB machine** — 69% coverage,
which no `mmap` engine can arrange because it cannot be told to use exactly N
GiB of either tier.

And the standing warning applies to all of it: **hit rate is not a success
metric.** Past ~6 GiB the host cache reached 71% hits and was the *slowest*
configuration measured, because cached bytes got paged out and a "hit" became a
page fault in disguise. VRAM does not page — which is the strongest argument for
this slice and the thing to verify first, not last.

## Recommendation

1. **Do not open a PR that half-binds.** Neither dense-layers-in-VRAM variant is
   a small first slice: one does not fit, the other is worth 1.10x.
2. **First measurable step is not GPU code.** It is the frontier sweep on
   Qwen3-30B — tok/s against host cache size, 1 GiB to 8 GiB — because the VRAM
   tier's value is a *point on that curve* and there is no curve yet. It needs
   no toolkit and no new code.
3. **Then, if the curve is still climbing at the host RAM ceiling**, the VRAM
   read cache is the slice worth building, on a 20–70 GiB model, with the PCIe
   bandwidth measured rather than assumed.
4. **If the curve has flattened**, the VRAM tier is dead on arrival for the same
   reason the byte-reduction roadmap closed: more cache is not the lever.

Step 2 is the honest next action, and it is not a GPU ticket.

## Open questions

- Real pinned-host↔device bandwidth on this laptop. Unmeasurable until the
  toolkit is installed; every number above that depends on it is labelled.
- Whether 6 GiB of VRAM is usable or whether the desktop compositor keeps a
  slice. `nvidia-smi` reports 259 MiB used at idle, so ~5.7 GiB, but a display
  driver under load is not idle.
- Whether a VRAM cache's eviction policy can be the frequency-gated admission
  that took the host cache 17% → 70%, or whether the smaller tier wants
  something else. Untested either way.
