# Qwen3.6-27B is `qwen35` and it generates nonsense

**Status**: open, found 2026-08-19.
**Links**: [`qwen35-gated-delta-net`](qwen35-gated-delta-net.md) ·
[`hard-won-facts`](../reference/hard-won-facts.md)

## The symptom

```
chaos-run Qwen3.6-27B-Q4_K_M.gguf "The capital of France is" -n 6 --temp 0
  ทัน ทัน ทัน ทัน ทัน ทัน
generated  6 tokens in 126.3s (0.05 tok/s)
```

Exit 0. No warning, no assert, no non-finite value. **This is the project's own
central hazard**, and it was walked into: a sweep of all twelve installed models
recorded "twelve of twelve ok" because every one exited 0, and the outputs were
not read. Qwen3.5-0.8B, the same architecture at 24 layers instead of 64, is
byte-identical to llama.cpp at three prompt lengths.

The sweep now requires the word `Paris` in the continuation, which is a
correctness check rather than a liveness one.

## Ruled out, by measurement

**The key-head broadcast.** The strongest hypothesis, because it is exactly the
kind of thing 0.8B cannot expose: the 27B has `group_count 16` key heads and
`time_step_rank 48` value heads, where the 0.8B has 16 and 16. A missing
broadcast of q and k up to the value head count is a no-op at 1:1 and fatal at
1:3, and `qwen35.rs` carries a *comment* saying the fused op broadcasts on its
own — asserted, never checked.

Checked now. `gated_delta_net_and_the_key_head_broadcast` calls
`ggml_gated_delta_net` twice at a 2:6 ratio with ramped inputs, once with narrow
q/k and once with q/k repeated by hand through `repeat_4d`, and the two agree to
1e-5. **The op does broadcast; the caller is right not to.** The test stays,
because a comment asserting a behaviour is not the same as a test of it.

**Every shape.** Read from both containers and compared against what
`SsmConfig` computes:

| | 0.8B | 27B | rule |
|---|---|---|---|
| `attn_qkv` | 6144 | 10240 | `2*key_dim + value_dim` ✓ |
| `ssm_conv1d` | 4×6144 | 4×10240 | same ✓ |
| `ssm_norm` | 128 | 128 | `head_v_dim = inner/tsr` ✓ |
| `attn_q` | 4096 | 12288 | `2 * head_count * key_length` ✓ |
| `attn_output` in | 2048 | 6144 | `head_count * key_length` ✓ |
| `ssm_a`, `ssm_dt.bias` | 16 | 48 | `time_step_rank` ✓ |
| tensors | 320 | 851 | 48 recurrent + 16 attention + 3 ✓ |

The first four blocks' tensor names are identical between the two files, so the
layer structure is the same and no tensor is being ignored. `rope.dimension_sections`
is `[11, 11, 10, 0]` and `key_length` is 256 in both.

**Repacking.** The last structural difference -- the 0.8B is Q8_0 throughout,
the 27B is Q4_K 10.44 GiB + Q6_K 4.14 + Q5_K 0.97, and repacking rearranges
quantised tensors into the CPU kernels' layout. Ruled out:

```
CHAOS_NO_REPACK=1 chaos-run Qwen3.6-27B-Q4_K_M.gguf "The capital of France is" -n 6 --temp 0
  ทัน ทัน ...
generated  6 tokens in 206.1s (0.03 tok/s)
```

Same nonsense, 1.6x slower for it. Not the quantisation layout.

## What is left to try, in order

1. **A layer diff against llama.cpp on the 27B itself**, the way the 0.8B was
   done — `llama-eval-callback` per layer, by value and by sum. Slower per
   iteration but it is the method that has worked twice.
2. **Bisect by layer count.** If a 64-layer container is wrong and a 24-layer one
   is right, the first attention layer is index 3 in both, so the difference is
   not "which layer is recurrent". Something that accumulates over depth, or a
   width that only the wider model exercises.

## Shipped ahead of the fix

**An architecture name is not a shape.** `VERIFIED_ARCHITECTURES` is
per-architecture, and this is the first case where one string covers a container
that is exact and a container that is nonsense. `verified_block_counts` now
records what was actually diffed — `qwen35` at 24 blocks — and
`why_shape_is_unverified` turns a mismatch into a line both binaries print
before a token is produced:

```
shape      UNVERIFIED -- qwen35 was diffed against llama.cpp at 24 blocks, and
           this container has 64. The forward pass may be WRONG with no error
           anywhere ... Treat the output as unverified
```

It **warns rather than refuses**, deliberately: the policy Atur asked for is run
it and say what is known, and a size that has not been checked is not the same as
a size known to fail. Said before generation, because afterwards it reads as an
excuse for output the user has already started trusting.
