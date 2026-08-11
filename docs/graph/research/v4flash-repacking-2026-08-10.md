---
topic: R9.1 — repacking the V4-Flash dense path repacks nothing on x86, and the attempt found a crash that was already shipping
status: resolved
links: [../backlog/lts-parity-criteria.md, qwen3-4b-vs-llamacpp-2026-08-10.md, v4flash-vs-llamacpp-2026-08-07.md]
---

Weight repacking is worth **1.42x on Qwen3-4B prefill** and is on by default,
but only on the dense path — `StreamingRunner::load_resident`. V4-Flash binds
through `ResidentSet` + `bind_dense`, so it never saw it. The ticket was: the
same win is sitting there.

**It is not.** On x86 there is nothing to repack in this model, and the number
is **0 tensors, 0 bytes, not 1.42x**. What the attempt did find is a null
dereference that has been shipping in the dense path since repacking landed.

## Why nothing repacks

`ggml_repack_get_optimal_repack_type` decides, and it branches on the **CPU** as
well as the tensor. For `Q8_0` (`ggml/src/ggml-cpu/repack.cpp`) the only
branches are `ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()`,
`ggml_cpu_has_neon() && ggml_cpu_has_dotprod()`, and RISC-V. **There is no x86
branch at all.** `Q2_K` needs AVX-512.

Every dense tensor in `DeepSeek-V4-Flash-UD-Q4_K_XL` with a repackable shape is
`Q8_0`:

| tensor | count | type | shape |
|---|---:|---|---|
| `attn_q_a` | 43 | Q8_0 | 4096 × 1024 |
| `attn_q_b` | 43 | Q8_0 | 1024 × 32768 |
| `attn_kv` | 43 | Q8_0 | 4096 × 512 |
| `attn_output_b` | 43 | Q8_0 | 8192 × 4096 |
| `ffn_gate_shexp` | 43 | Q8_0 | 4096 × 2048 |
| `ffn_up_shexp` | 43 | Q8_0 | 4096 × 2048 |
| `ffn_down_shexp` | 43 | Q8_0 | 2048 × 4096 |
| `attn_compressor_kv` | 41 | Q8_0 | 4096 × 1024 or 512 |
| `attn_compressor_gate` | 41 | Q8_0 | 4096 × 1024 or 512 |
| `output` | 1 | Q8_0 | 4096 × 129280 |
| `ffn_gate_inp` | 43 | BF16 | — nothing to pack |
| `hc_attn_fn`, `hc_ffn_fn` | 86 | F32 | — nothing to pack |

The container is "Q4_K_XL": the routed experts are Q4_K and everything
always-read is upcast to Q8_0. **Repacking helps exactly the tensors this
quantisation does not use it for.**

Measured, 3 GiB residency budget: `42 offered, 42 declined, 0 repacked`.

So this is not a gap against llama.cpp. It is a property of the container and
the instruction set, and it would reverse on an ARM machine with `matmul_int8`.

## llama.cpp cannot load this model with repacking on at all

Asked the same question — repacking enabled, its default — llama.cpp does not
get a smaller win. It **fails to load**:

```
llama-completion -m DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf \
                 -c 512 -t 12 -n 1 -p "The capital of France is" --no-warmup

E ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 147169738752
E alloc_tensor_range: failed to allocate CPU_REPACK buffer of size 147169738752
E llama_model_load: error loading model: unable to allocate CPU_REPACK buffer
E llama_model_load_from_file_impl: failed to load model
E cmn  common_init_: failed to load model
```

147,169,738,752 bytes is **137 GiB — the whole model**, routed experts included.
llama.cpp's repack buffer is allocated as one range up front, so on a container
this size the choice is all of it in RAM or none of it. That is why every
V4-Flash figure this project has recorded against llama.cpp passes
`--no-repack`, a detail that had been noted as a quirk without the reason being
established.

Bigtea's repacking is per tensor and backed by residency, so the same container
loads, reports `0 repacked`, and runs. **That is a difference in kind, not in
speed**: no tokens per second are won, and `--no-repack` gets llama.cpp running
too. It is worth recording only because "repacking on this model" reads like an
open opportunity until you try it on both engines.

## The crash that was already shipping

`ggml`'s repack buffer does not decline. `init_tensor` sets `tensor->extra` to
whatever the chooser returned — `nullptr` when there is no kernel — and returns
`GGML_STATUS_SUCCESS`. `set_tensor` then does:

```c
auto tensor_traits = (ggml::cpu::repack::tensor_traits_base *) tensor->extra;
auto OK            = tensor_traits->repack(tensor, data, size);
```

A null there is a null dereference: `STATUS_ACCESS_VIOLATION`, no assert, no
error code, no output, process gone. It killed the whole V4-Flash test binary on
the first `Q8_0` tensor offered.

`is_repackable` cannot prevent it, because it is a shape-and-type check and the
answer depends on the CPU. The fix is to ask ggml and then **read what it
decided** — `extra` is checked after `ggml_backend_tensor_alloc` and before
`ggml_backend_tensor_set`, and a null means decline.

**This was reachable from the dense path too.** `load_resident` offers every
resident tensor, and `is_repackable` accepts `Q8_0` and `Q2_K`. Any
`*.Q8_0.gguf` — an ordinary thing to download — would have ended `bigtea-run`
with an access violation on x86 before printing a token. None of the Q4_K_M
containers on this machine contain a `Q8_0` 2-D weight, which is the only reason
it had not been seen.

## What was built anyway, and why it was kept

The V4-Flash path could not have used `bind_repacked` even where the kernels
exist. It owns an arena **per block** — chaining 43 blocks into one `ggml`
context costs hundreds of megabytes each — so it builds a fresh context and a
fresh `WeightSet` for every block of every pass. Rearranging there would redo
the whole always-read set 43 times *per token*: not a smaller win, a large loss.

So the rearrangement is hoisted to load time (`RepackedDense`) and each block
re-attaches by pointing a fresh tensor at bytes that are already in order —
`ggml_backend_tensor_alloc`, which re-runs `init_tensor` and re-hangs the traits
without calling `set_tensor`, so no bytes move.

Verified numerically on x86 with `Q4_K`, which *does* have a kernel here:
ggml's repacked matmul against ggml's ordinary matmul on the same weights,
agreeing to 1e-3 relative, **bound into two separate contexts from one
rearrangement** — because a mechanism that only worked once would pass a
single-bind test and fail in the runner at block 2.

Residency is handed over rather than duplicated: `ResidentSet::take` removes a
tensor as it is rearranged and the original is dropped, so the peak is one
tensor rather than a second whole set. On a 15.7 GiB machine holding a 7.38 GiB
always-read set there is no second copy to give.

## The allow-list, and why it is not an exclusion list

The dense path excludes three known-bad uses and repacks the rest. Here the
default has to be the other way round, because repacking interleaves rows and
**every use except a 2-D `mul_mat` weight reads the bytes by position** — none
of which fail:

| tensor | use | repack |
|---|---|---|
| `token_embd` | `get_rows` by token id | no |
| `attn_compressor_ape` | `get_rows` by within-block position | no |
| `ffn_gate_tid2eid` | `get_rows` by token id | no |
| `*_hc_scale`, `*_hc_base` | `view_1d` at a byte offset | no |
| `attn_output_a` | `reshape_3d` into a grouped `mul_mat` | no |
| `attn_sinks` | sinks argument of `flash_attn_ext` | no |
| `blk.*.ffn_*_exps` | routed, streamed from disk | **never** |
| 14 others | `mul_mat(w, x)`, 2-D | yes |

Four of those are things the ticket did not name, and `attn_output_a` is the
one that would have been missed by reading names rather than call sites: it sits
beside `attn_output_b` and differs only in being reshaped into three dimensions
first. Guessing from names would have been wrong on roughly a fifth of this
architecture's tensors, silently.

The routed experts are the load-bearing exclusion. They are bound zero-copy from
a pointer into the mapped container, one slice at a time, and never reach
`bind_dense` at all — that is what lets a 144 GB model run on a 15.7 GiB
machine. Repacking them would need the whole bank in RAM.

## What this changes

- **No speed claim.** V4-Flash prefill and generation are unchanged on x86;
  0 tensors were rearranged. Do not quote 1.42x for this model.
- A crash reachable from a plain `*.Q8_0.gguf` on the dense path is fixed.
- The machinery is in place and correct, so an ARM build gets the win for free
  and any future container that ships Q4_K always-read weights does too.

## Cost

`RepackedDense` + `Repacked` + `ResidentSet::take`/`put_back`, 4 new ggml unit
tests, 6 allow-list unit tests, 2 container-backed tests.
