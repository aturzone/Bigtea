---
ticket: `rope_freqs.weight` is in the container and this build ignores it
status: open — found 2026-08-11, fix is in qwen3.rs/stream.rs
owner: the r13 session (their files)
links: [../research/parity-measured-2026-08-11.md]
---

## The bug

Llama-3.x containers ship a **`rope_freqs.weight`** tensor. llama.cpp passes it
to `ggml_rope_ext` as `freq_factors`; this build passes `None`.

```
$ python  # tensor names in Llama-3.2-1B-Instruct-Q4_K_M.gguf
rope tensors: ['rope_freqs.weight']
```

llama.cpp, `src/models/llama.cpp`:

```cpp
layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i),
                                 {n_rot/2}, TENSOR_NOT_REQUIRED | ...);
...
ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, ...);
Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, ...);
```

Ours, `qwen3.rs` and `stream.rs`:

```rust
ctx.rope_ext(&q, positions, None, c.head_dim as i32, c.rope_type, 0, rope)?
//                          ^^^^ freq_factors
```

**The parameter is already plumbed.** It is an `Option` and every call site
passes `None`, so the change is loading the tensor and passing `Some`.

## How it was found, and why it took this long

The eight-prompt parity sweep. Llama-3.2-1B:

```
FAIL  Llama-3.2-1B-Instruct  SELECT name, COUNT(*) FROM users WHERE
  chaos   :  age > 18 AND gender = 'male' GROUP BY name;
  llama.cpp:  age > 18 GROUP BY name HAVING COUNT(*) > 1;
```

llama.cpp is **stable** on that prompt across `-fa on`, `-fa off`,
`--no-repack` and `-t 4`, so it is not a near-tie. The divergence starts about
five tokens in.

The three-prompt sweep never reached it. `llama` has been in
`VERIFIED_ARCHITECTURES` since the beginning and TinyLlama passes 8/8 — because
**TinyLlama is Llama-2 and has no `rope_freqs.weight`**. One container in a
family exercising the tensor and another not is exactly the gap a small prompt
set leaves.

## What is affected

Every Llama-3.x container that ships the tensor — which is the whole 3.1/3.2/3.3
line, since it is how their long-context RoPE scaling is expressed. Llama-2 and
TinyLlama are unaffected.

**It is not correct on any of them today**, and `llama` should be read as
"verified on Llama-2-shaped containers" until this lands.

## The fix

1. `required_tensors()` gains `rope_freqs.weight` as **optional** — it is
   `TENSOR_NOT_REQUIRED` in llama.cpp and absent from Llama-2, so requiring it
   would refuse the models that currently work. This project has already made
   that exact mistake once, with QK-norm, and it refused the entire Llama family
   before a byte was read.
2. Load it into the `WeightSet` when present.
3. Pass `Some(&t)` at the four `rope_ext` call sites instead of `None`.

llama.cpp stores it per layer with `TENSOR_DUPLICATED` after layer 0, i.e. one
tensor shared by every block. Reading `blk.0`'s once is enough.

## Acceptance

`scripts/parity-check.sh` on Llama-3.2-1B at 32 tokens: the SQL prompt must
move from FAIL to ok, and **TinyLlama must stay 8/8** — it takes the
`freq_factors = None` path and must keep taking it.

Note that Llama-3.2-1B reports **four unstable prompts of eight**, more than any
other container measured. Those are near-ties in the reference and are not this
bug; do not chase them.
