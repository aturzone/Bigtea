# The iGPU has more memory than the card, no copy problem, and is slower than the CPU

**2026-08-15.** A negative, written down because it is the *attractive* idea and
it will be re-proposed by anyone who runs `--list-devices` and reads the memory
column. It cost one command to kill.

Links: [gpu-the-card-works-vulkan-not-cuda-2026-08-15.md](gpu-the-card-works-vulkan-not-cuda-2026-08-15.md) ·
[gpu-tier-smallest-honest-slice-2026-08-11.md](gpu-tier-smallest-honest-slice-2026-08-11.md) ·
[v4flash-has-no-slack-2026-08-10.md](v4flash-has-no-slack-2026-08-10.md)

## Why it looks like the better device

Vulkan reports two, and on paper the *integrated* one wins the two things this
project actually cares about:

```
Vulkan0: Intel(R) RaptorLake-S Mobile Graphics Controller (8045 MiB, 7387 MiB free)
         uma: 1 | fp16: 1 | bf16: 0 | matrix cores: none
Vulkan1: NVIDIA GeForce RTX 3050 6GB Laptop GPU          (6001 MiB, 5233 MiB free)
         uma: 0 | fp16: 1 | bf16: 1 | matrix cores: NV_coopmat2
```

**More free memory than the discrete card** — 7387 MiB against 5233 — on an
engine whose entire design problem is that weights do not fit. And **`uma: 1`**,
meaning device memory *is* host memory, so the copy that defines the whole GPU
ticket would not exist there. The obvious proposal writes itself: put the
streamed experts on the integrated device, skip `ggml_backend_tensor_set`
entirely, keep the discrete card for dense compute.

## It is slower than the CPU

Qwen3-4B-Q4_K_M, llama.cpp `daef2b3`, same session as the main GPU measurement:

```bash
./build-vulkan/bin/llama-bench.exe -m Qwen3-4B-Q4_K_M.gguf --device Vulkan0 -ngl 99 -p 512 -n 128 -r 2
```

| config | pp512 | tg128 |
|---|---:|---:|
| **Intel iGPU, `-ngl 99`** | **38.13 ± 2.09** | **3.26 ± 0.03** |
| CPU, best of 4/20 threads | 79.65 ± 5.93 | 6.39 ± 0.08 |
| RTX 3050, `-ngl 99` | 2042.60 ± 5.52 | 56.53 ± 0.04 |

**0.48x on prefill and 0.51x on generation against the CPU path we already
have.** Not a smaller win — a loss, on both axes, at every size measured.

## Why, and this is the part that generalises

`matrix cores: none`. There is no tensor-core equivalent, so the matmuls run on
shader ALUs against a CPU path that has AVX2, FMA, `LLAMAFILE` kernels and
`REPACK` layouts tuned for exactly this.

And the UMA property that makes the copy free is the same property that makes
the compute slow: **an integrated GPU shares the DRAM the CPU path is already
saturating.** `CLAUDE.md` records that generation wants 2–4 threads because it
saturates DRAM at that point. Moving that work to a device on the *same* memory
bus does not add bandwidth; it just runs the same starved workload on weaker
arithmetic.

> **The rule: a UMA device removes the copy, not the bottleneck.** Zero-copy is
> only worth having if the thing you avoided copying to can compute faster than
> where it already was. Here it cannot, so "skip the upload" buys a 2x slower
> engine.

## What would change this

- **A device with matrix cores on a UMA bus** — an Apple M-series or a Strix
  Halo class part, where the integrated GPU has both the shared memory *and*
  competitive arithmetic. The reasoning above is about *this* iGPU, not about
  integrated graphics as a category.
- **A workload that is bandwidth-bound on the device but not on the host**,
  which is the opposite of what was measured.

Neither describes this machine, so: **the RTX 3050 is the only compute device
here, and the iGPU is not a second tier.**
