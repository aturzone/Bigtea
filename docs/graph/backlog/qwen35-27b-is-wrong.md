# Qwen3.6-27B generates nonsense, and matching llama.cpp is not a defence

**Status**: open. Narrowed, not fixed, 2026-08-19.
**Links**: [`qwen35-gated-delta-net`](qwen35-gated-delta-net.md) ·
[`hard-won-facts`](../reference/hard-won-facts.md)

## What both engines do

Same container, same prompt, greedy, CPU only:

| engine | continuation of "The capital of France is" |
|---|---|
| Chaos | `ทัน ทัน ทัน ทัน ทัน ทัน` |
| **llama.cpp** | **`333333`** |

Neither says Paris.

**That llama.cpp fails too is a clue, not an excuse.** It narrows the search — it
does not mean Chaos is allowed to be wrong here. Chaos is judged on whether it
answers correctly, and on this container it does not. What the agreement buys is
knowing *where* to look: not in the parts both engines compute identically.

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

Agreement this exact says the port reproduces llama.cpp faithfully **including
its behaviour here**. That is worth knowing and it is not a result: reproducing a
wrong answer precisely is still a wrong answer.

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

One of two things. Both are this project's problem to solve — the second one
especially, because "llama.cpp does it this way" is not a specification:

1. **This quantisation.** `Q4_K 10.44 GiB + Q6_K 4.14 + Q5_K 0.97`, an Unsloth
   imatrix build. A block whose scale decodes to something enormous would push
   layer 5's residual to 1e6 without storing a single non-finite f32.
2. **The way this architecture is implemented at 64 blocks**, in llama.cpp and
   copied faithfully here. If that is the answer, the fix is to work from Qwen's
   own model definition rather than from another implementation of it — a
   residual that legitimately reaches 1.009e6 by layer 5 is something the correct
   forward pass must survive, and finding out how it does is the job.

**The test that separates them is a different quantisation** —
`Qwen3.8-27B-UD-Q2_K_XL.gguf` at 9.94 GiB, which is also the model Atur actually
wants and the only 27B size that fits this machine. If it generates correctly the
first cause was the file. If it fails the same way, the answer is in how this
architecture is implemented at 64 blocks, and then the work is to derive it from
**Qwen's own model definition** rather than from another engine's version of it.

Qwen3.6 comes out of the catalogue, the tests and the docs once 3.8 generates
correctly — Atur's instruction, and the right shape for it: one supported member
of the family that works, rather than two that do not.

## Shipped ahead of the fix

`catalogue::verified_block_counts` records the shape that was checked — `qwen35`
at 24 blocks — and `why_shape_is_unverified` turns a mismatch into a line
`chaos-run`, `chaos-serve` and the app all show before a token is produced.

**It names no other project.** Whether llama.cpp fails too is a clue for whoever
is fixing this; to the person waiting for an answer it is an excuse, and a test
asserts the string never mentions it:

```
shape      UNVERIFIED -- this architecture has been checked at 24 blocks and this
           container has 64, so answers may be WRONG with no error anywhere. The
           64-block build we have tested, Qwen3.6-27B-Q4_K_M, overflows to NaN
           part way through and produces nonsense. A smaller quantisation is
           worth trying, and the failure is being worked on
```

It **warns rather than refuses**: the policy is to run what can be run and say
what is known, and a size that has not been checked is not the same as a size
known to fail.
