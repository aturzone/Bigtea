---
topic: The "disk" time in a V4-Flash prefill was one third memcpy, and the rest was misaligned
status: resolved
links: [v4flash-speed-budget.md, lts-0-0-0.md, head-to-head-llamacpp-2026-08-05.md]
---

T0.1 of `lts-0-0-0.md`. The epic said: *measure `read_exact_at` alone against
`read_at` first — if the copy is not the gap, do not make the API change.*
Measured. It was the gap, and there was a second, larger one behind it.

## What a byte of expert weight cost, before

`bind_expert_slices` → `Model::read_tensor_range` → `DirectFile::read_at`:

| # | where | what happens to every byte |
|---|---|---|
| 0 | `AlignedBuf::new` | `alloc_zeroed` of the aligned superset |
| 1 | `read_exact_at` | the actual transfer — the only necessary step |
| 2 | `read_at` → `.to_vec()` | allocate again, copy |
| 3 | `bind_expert_slices` → `extend_from_slice` | copy into the compact stack |
| 4 | `WeightSet::bind` → `Arc<[u8]>: From<Vec<u8>>` | allocate again, copy |

**Three full copies**, one of them purely to change the shape of a pointer, on
11.5 GiB per prefill.

## Measured: `crates/chaos-model/tests/expert_read_cost.rs`

120 slices of `blk.5.ffn_up_exps.weight` (0.53 GiB), each a different expert so
no read can be served by a previous one, cache-bypassing throughout:

```
(a) read_tensor_range + extend + Arc     0.62s   0.80 GiB/s   3 copies
      of which the read call itself      0.42s   1.20 GiB/s
(b) read_range_into, aligned stack       0.42s   1.20 GiB/s   100.00% copied
(c) read_range_into, skew 2816           0.32s   1.58 GiB/s     0.09% copied
```

Copies are **34-36% of the time attributed to "disk"**, reproducible across runs.

## The second finding, which is the bigger one

Step (b) is the obvious fix — read into a caller-owned aligned buffer — and it
still copies **every byte**. The benchmark printed why:

```
slice % 4096 = 0,  tensor offset % 4096 = 2816
```

GGUF pads tensor data to `general.alignment`, which defaults to **32**, not to a
sector. So V4-Flash's expert data begins 2816 bytes into a sector, and direct
I/O can only transfer when the file offset and the memory address agree modulo
the sector size. A conventionally aligned destination can *never* match one, so
every byte bounces through a scratch buffer no matter how the API is shaped.

**The fix is to misalign the destination on purpose.** `SkewedBuf::new(len,
2816)` hands out a buffer whose first byte sits at an address ≡ 2816 (mod 4096);
now the residues agree and the drive writes into the caller's memory. Only the
two edge sectors of each slice bounce — 4096 bytes out of 4.25 MiB, **0.09%**,
and constant in the length of the read rather than proportional to it.

This generalises: every slice of a stacked expert tensor is the same size, and
that size is a sector multiple, so **one skew serves the whole stack**.

## End to end on the real model

Same binary, same prompt, same machine, back to back, `git stash` between:

```
                      run 1    run 2
before   prefill 5 tokens   32.4s    33.0s     0.15 tok/s
after    prefill 5 tokens   24.1s    23.7s     0.21 tok/s
```

**1.38x**, and the model still emits `11111 " Paris"`. All 14 container-backed
forward tests still match llama.cpp's element sums, so nothing was traded for
this.

## Why the end-to-end gain is smaller than the microbenchmark's 2.0x

Because expert reads are ~49% of a prefill. 2.0x on half the work is ~1.33x
overall, which is what was measured. The remaining time is dense reads (23%,
addressed by T0.2 residency) and compute (28%).

## What changed

- `chaos-io`: `SkewedBuf`, and `DirectFile::read_at_into`, which splits a read
  into `[head fragment][directly transferred middle][tail fragment]` and returns
  **how many bytes it had to copy** — not a bool, because a 4 MiB slice with two
  bounced edge sectors is 99.9% direct and a bool would hide that.
- `chaos-model`: `read_range_into`.
- `chaos-ggml`: `WeightSet::bind` takes `impl WeightBytes` — anything heap-owned
  that derefs to `[u8]` — instead of `impl Into<Arc<[u8]>>`, so the caller's own
  allocation is kept rather than copied into a differently-shaped one.
- `chaos-arch`: `bind_expert_slices` reads each slice straight into its final
  position in a skewed stack.

## For anyone porting this elsewhere

The 2816 is not a constant to hard-code — it is `file_offset % 4096` and differs
per tensor and per container. `SkewedBuf::skew_for(loc.file_offset)` computes it.
A container written with `general.alignment = 4096` would make the skew zero and
this machinery a no-op, which is the right behaviour: it costs nothing when
alignment already agrees.
