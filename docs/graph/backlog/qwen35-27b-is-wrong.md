# Qwen3.6-27B generates nonsense — and so does llama.cpp on the same file

**Status**: cause located outside this engine, 2026-08-19. Not a Chaos bug.
**Links**: [`qwen35-gated-delta-net`](qwen35-gated-delta-net.md) ·
[`hard-won-facts`](../reference/hard-won-facts.md)

## What both engines do

Same container, same prompt, greedy, CPU only:

| engine | continuation of "The capital of France is" |
|---|---|
| Chaos | `ทัน ทัน ทัน ทัน ทัน ทัน` |
| **llama.cpp** | **`333333`** |

Neither says Paris. **The reference implementation fails on this file too**, so
the port is not what is wrong, and every conclusion below follows from that.

Commands, so this is citable:

```
chaos-run Qwen3.6-27B-Q4_K_M.gguf "The capital of France is" -n 6 --temp 0
llama-completion -m Qwen3.6-27B-Q4_K_M.gguf -p "The capital of France is" -n 6 --temp 0 -ngl 0 -t 8
```

## Where it breaks, in both

`llama-eval-callback` against Chaos's `CHAOS_DUMP_LAYERS=1`, sums over all five
prompt tokens:

| tensor | Chaos | llama.cpp |
|---|---|---|
| `l_out-0` | 59.1449 | 59.1446 |
| `l_out-1` | 77.2332 | 77.2331 |
| `l_out-2` | 81.5851 | 81.5852 |
| `l_out-3` | 123.8178 | 123.8176 |
| `l_out-4` | 128.1730 | 128.1734 |
| `attn_residual-5` | **1009342.38** | **1009345.31** |
| `attn_post_norm-5` | 58.0360 | 58.0360 |
| `l_out-5` | **NaN** | **NaN** |

**Identical to five significant figures for five layers, then NaN in both.** The
residual reaches 1.009e6 at layer 5 — three orders of magnitude above layer 4's
128 — and the layer's output overflows from there. llama.cpp's dump shows the
same collapse in more detail: `ffn_up-5` is finite at 377.58 while `ffn_gate-5`,
`ffn_swiglu-5` and `linear_attn_qkv_mixed-5` are all NaN.

Agreement this exact, up to and including the failure, is the strongest evidence
the `qwen35` port is faithful that has been collected — stronger than the 0.8B
diff, because it reproduces a pathological case step for step.

## Ruled out, each by measurement

- **The container is not truncated.** 851 of 851 tensors readable,
  16,817,244,384 bytes, and `expected_file_bytes` agrees.
- **No f32 weight holds a NaN.** All 449 f32 tensors scanned: every value
  finite. So nothing is *stored* broken; the overflow happens in arithmetic.
- **Not repacking.** `CHAOS_NO_REPACK=1` gives the same nonsense, 1.6x slower.
- **Not the key-head broadcast.** The 27B has 16 key heads and 48 value heads
  where the 0.8B has 16 and 16, so a missing broadcast would be invisible at
  0.8B and fatal here. `gated_delta_net_and_the_key_head_broadcast` calls the
  fused op at a 2:6 ratio with ramped inputs, narrow against hand-repeated, and
  they agree to 1e-5. The op broadcasts; the caller is right not to.
- **Not the tokenizer.** Both containers declare `gpt2` / `qwen35`, 248320
  tokens, 247587 merges, the same eos. The 27B adds `bos_token_id` and
  `add_bos_token = False`, neither of which is reachable here.
- **Not a shape.** Every one checked against what `SsmConfig` computes:
  `attn_qkv` 10240 = `2*key_dim + value_dim`, `ssm_conv1d` 10240, `ssm_norm`
  128, `attn_q` 12288 = `2 * head_count * key_length`, `attn_output` 6144,
  `ssm_a` 48, and 851 tensors = 48 recurrent + 16 attention blocks plus three.

## So what is it

One of two things, and this machine can separate them only with a second file:

1. **This quantisation.** `Q4_K 10.44 GiB + Q6_K 4.14 + Q5_K 0.97`, an Unsloth
   imatrix build. A block whose scale decodes to something enormous would push
   layer 5's residual to 1e6 without storing a single non-finite f32.
2. **llama.cpp's own `qwen35` implementation at 64 blocks**, faithfully
   reproduced here.

**The test that separates them is a different quantisation of the same model** —
`UD-Q2_K_XL` at 10.7 GB, which has the side benefit of fitting this machine. If
it generates correctly it was the file; if it fails identically it is the
architecture at this depth, and that belongs upstream rather than here.

## Shipped ahead of that

`catalogue::verified_block_counts` records what was diffed — `qwen35` at 24
blocks — and `why_shape_is_unverified` turns a mismatch into a line
`chaos-run`, `chaos-serve` and the app all show before a token is produced. It
names the container rather than the engine, because the measurement says the
engine agrees with the reference:

```
shape      UNVERIFIED -- qwen35 was diffed against llama.cpp at 24 blocks and
           this container has 64. Answers may be WRONG with no error anywhere.
           The known 64-block container, Qwen3.6-27B-Q4_K_M, overflows to NaN at
           layer 5 -- in llama.cpp as well as here, from identical layer sums --
           so a different quantisation of it is worth trying before this engine
           is suspected
```

It **warns rather than refuses**: the policy is to run what can be run and say
what is known, and a size that has not been checked is not the same as a size
known to fail.
