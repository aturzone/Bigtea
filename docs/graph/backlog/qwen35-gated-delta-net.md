# Running Qwen3.5 / Qwen3.6 — a gated delta net, not an attention model

> Blocks: `qwen3.6-27b`, `qwen3.6-35b-a3b`, and every later Qwen on this
> architecture. Atur has asked for this three times, so it is written down
> properly rather than answered again with "not supported".

**Status: not started.** Everything below was measured from the container Atur
already has on disk (`Qwen3.6-27B-Q4_K_M.gguf`, 16.8 GB) and from llama.cpp's
own implementation, which exists and is the oracle to diff against.

## Why re-downloading cannot help

The file is fine. `chaos-run` refuses it because `general.architecture` reads
`qwen35`, and that is not one of the thirteen architectures this engine has
diffed against llama.cpp. The refusal is the *correct* behaviour: a wrong
forward pass here would answer in fluent English and be wrong, with no error.

## What the container says it is

From `chaos-meta` on the file itself:

```
qwen35.block_count               64
qwen35.full_attention_interval    4
qwen35.attention.head_count      24     head_count_kv     4
qwen35.attention.key_length     256     value_length    256
qwen35.embedding_length        5120     feed_forward_length 17408
qwen35.rope.dimension_count      64     rope.dimension_sections [4 items]
qwen35.ssm.conv_kernel            4     ssm.state_size  128
qwen35.ssm.group_count           16     ssm.inner_size 6144
qwen35.ssm.time_step_rank        48
```

and every block carries `ssm_conv1d`, `ssm_a`, `ssm_alpha`, `ssm_beta`,
`ssm_dt.bias`, `ssm_norm`, `ssm_out`, `attn_gate` and a fused `attn_qkv` of
`[5120, 10240]`.

Those numbers decompose exactly:

| | |
|---|---|
| key heads × head dim | `16 × 128 = 2048` |
| value heads × head dim | `48 × 128 = 6144` |
| q + k + v | `2048 + 2048 + 6144 = 10240` = the `attn_qkv` width |
| conv channels | the same 10240 = `ssm_conv1d`'s second dimension |

**So `attn_qkv` in these blocks is not attention's QKV.** It is the delta net's
input projection, and `ssm_conv1d` is a causal depthwise convolution over all of
it. This is Qwen3-Next's **gated delta net**.

## Which layers are which

llama.cpp's `src/models/qwen35.cpp` decides it as:

```cpp
hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
```

With `full_attention_interval = 4` and 64 blocks that is **48 recurrent layers
and 16 attention layers**. Three quarters of the model is not attention.

## The four pieces of work

1. **A recurrent memory, beside the KV cache.** Each recurrent layer needs, per
   sequence: a conv window of `[4, 10240]` and a state of
   `[128, 128, 48]` — head_v_dim × head_k_dim × value heads. That is ~3 MB per
   layer per sequence at f32, ~150 MB across 48 layers. A KV cache cannot stand
   in for it: the state is *carried*, not appended, so eviction, reuse and
   rollback all behave differently. This is the piece that touches the most
   existing code.
2. **The delta net itself.** ggml has `ggml_gated_delta_net`
   (`ggml/include/ggml.h:2592`) and `ggml_ssm_conv`, so the arithmetic exists
   and does not have to be written from scratch — but neither is bound in
   `chaos-ggml` yet, and `ggml_softplus` and `ggml_sigmoid` are needed with
   them. Follow `build_layer_attn_linear` in `qwen35.cpp`.
3. **Interleaved multimodal RoPE** (`ggml_rope_multi`, four sections) for the 16
   attention layers, which also gate their output through a sigmoid of the
   second half of a fused Q/gate projection.
4. **MTP / NextN.** `LLM_KV_NEXTN_PREDICT_LAYERS` adds decoder blocks past the
   main stack. They are loaded and *not executed* in a plain forward pass, so
   the minimum viable port is to skip them — but the loader has to know they
   exist or it will fail on unexpected tensors.

## How it gets verified

The same way every other architecture here did, and no other way: capture
llama.cpp's per-layer activations for a fixed prompt on this exact file, diff
against ours, and only then add `qwen35` to `VERIFIED_ARCHITECTURES`. Membership
of that list is the whole meaning of "Chaos can run it".

**The 5-token / 165-token / 2048-token prompt-length trap applies here too**:
prefill and single-token generation take different paths through a recurrent
layer, and a port that works on one says nothing about the other.

## Honest size

Bigger than any architecture added so far, because it is the first one that
changes what *memory* means rather than only what a layer computes. The
arithmetic is a day; the recurrent cache and its verification are the rest.

## What to tell a user meanwhile

The catalogue already does: the model is listed, and `why_not_runnable` says
"three of its four layers are a gated delta net with recurrent state, not
attention, and Chaos has no recurrent memory yet". That sentence is in the app's
refusal dialog and in `chaos-run`'s output.
