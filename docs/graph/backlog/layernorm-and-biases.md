---
topic: The next tier of architectures is blocked on ONE shared feature — LayerNorm with biases — not on per-model quirks
status: scoped 2026-08-11, not started
links: [lts-parity-criteria.md, ../research/gemma3-not-verified-2026-08-11.md]
---

StableLM and StarCoder2 were downloaded, run and diagnosed. **Neither is a
per-model quirk.** Both fail for the same reason, and so will most of the
remaining list, so the honest unit of work is the feature and not the model.

## What they actually produce today

```
bigtea-run -m stablelm-2-1_6b-chat.Q4_K_M.gguf -p "The capital of France is" -n 10 --force
  -> ��地なutorsemie路emieemieا起
```

The qwen2 signature: fluent-shaped CJK noise. Both are correctly refused without
`--force`, so nothing ships wrong — but `--force` runs them into a path that
cannot express them.

## The two things the dense path does not have

**1. LayerNorm.** `bigtea-ggml` binds `ggml_rms_norm` and **not `ggml_norm`**.
RMSNorm scales by the root-mean-square and has weight only; LayerNorm subtracts
the mean, divides by the standard deviation, and has **weight and bias**. They
are not interchangeable, and substituting one is not an error — it is the noise
above.

Both containers carry `attn_norm.bias` and `ffn_norm.bias`, which is the tell:
**a norm with a bias is a LayerNorm.** Also note the metadata key is
`attention.layer_norm_epsilon`, not `attention.layer_norm_rms_epsilon`.

**2. Biases on projections.** There is no bias support anywhere in the dense
path — `grep '\.bias'` over `stream.rs` and `qwen3.rs` returns nothing. Both
models need them:

| | stablelm | starcoder2 |
|---|---|---|
| `attn_q/k/v.bias` | yes | yes |
| `attn_output.bias` | — | yes |
| `ffn_up/down.bias` | — | yes |
| `ffn_gate` | present (SwiGLU) | **absent — plain MLP, GELU** |
| head_dim | 64 (2048/32) | 128 (3072/24), GQA 24:2 |
| `rope.dimension_count` | **16, not 64 — partial RoPE** | not declared |

## Two extras beyond the shared feature

- **StableLM rotates only 16 of its 64 head dimensions.**
  `stablelm.rope.dimension_count = 16`, so `n_rot` is 16 and the remaining 48
  pass through unrotated. Our path passes `head_dim` as `n_rot` unconditionally.
- **StarCoder2's FFN is not gated.** No `ffn_gate`, so it is
  `down(gelu(up(x)))` rather than `down(silu(gate(x)) * up(x))`. Feeding a
  non-gated model through the SwiGLU path needs a tensor that does not exist, so
  this one should fail loudly rather than quietly — worth checking which it does.

## Why this is the right unit of work

LayerNorm + biases is not two models' worth of value. It is the shared shape of
**falcon, gpt2, gptneox, bloom, phi2, starcoder, stablelm and starcoder2** — the
whole pre-LLaMA lineage plus several current ones. Implementing it once moves
the architecture count by more than the three or four models anyone would
verify in a sitting.

Order to build it:

1. Bind `ggml_norm` in `bigtea-ggml` beside `rms_norm`.
2. A `norm_kind` on the config, chosen by whether `attn_norm.bias` exists —
   asked of the container, like `qk_norm` and `fused_qkv` already are, rather
   than by architecture name.
3. Optional bias on every projection, again detected from the container.
4. `n_rot` from `rope.dimension_count` where declared, defaulting to `head_dim`.
5. Non-gated FFN with GELU when `ffn_gate` is absent.

Then verify **against llama.cpp per model**, one commit each, as the
architecture loop requires. Steps 1-4 are what StableLM needs; 1-3 and 5 are
StarCoder2.

## What was ruled out

Not the tokenizer: both declare `tokenizer.ggml.model = gpt2`, which is
supported, and StarCoder2 declares a `rope.freq_base` we already read. Not the
RoPE convention: both are NeoX and already mapped in `rope_type_for`. The
failure is entirely in the block.
