# Running Qwen3.5 / Qwen3.6 — a gated delta net, not an attention model

> Blocks: `qwen3.6-27b`, `qwen3.6-35b-a3b`, and every later Qwen on this
> architecture. Atur has asked for this three times, so it is written down
> properly rather than answered again with "not supported".

**Status: the arithmetic is bound and tested; the layer wiring is not written.**
Everything below was measured from the containers Atur has (`Qwen3.6-27B-Q4_K_M`,
16.8 GB) and from llama.cpp's own implementation, which exists and is the oracle.

## Qwen3.8-27B is the same architecture — this is not a two-model problem

Atur asked for Qwen3.8-27B *instead of* 3.6. It does not route around any of
this. Read from `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf` with
`tools/gguf-always-read.py`:

```
gguf v3, 866 tensors, 51 metadata keys
architecture                  qwen35
tensor bytes          17,912,397,824      (dense: nothing streams)
```

Upstream it is `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`,
`pipeline_tag: image-text-to-text` — **3.6's gated delta net with a vision tower
added**, plus a separate `mmproj-F16.gguf` (928 MB) for the vision half. So 3.8
is strictly more work than 3.6, not less, and text-only inference on it needs
exactly the port below. Both are in the catalogue; a test asserts they agree on
the architecture, so this stops being true loudly rather than quietly.

Note the fit: 17.9 GB dense against 15.7 GB of RAM. Even once it runs, **3.8 does
not fit this laptop at Q4** — `UD-Q2_K_XL` at 10.7 GB is the one that would.

## Done, and verified (2026-08-19)

**The fused op exists and is in the archive this project already links.** That
was the open question that made this look like a research project rather than a
port; it is answered.

- `ggml_gated_delta_net` — **the entire chunked delta rule in one op**, taking
  the carried state and returning the scores followed by state snapshots. Also
  `ggml_ssm_conv`, `ggml_l2_norm`, `ggml_rope_multi`, `ggml_view_4d`,
  `ggml_reshape_4d`, `ggml_cont_4d`, `ggml_repeat_4d`. All present in
  `ggml-base.a` at the pinned build (`nm --defined-only`), all now bound in
  `chaos-ggml`.
- **Three numeric tests, because a wrong FFI declaration does not fail to
  compile — it mis-reads arguments and returns confident numbers.**
  `l2_norm` divides a row of four 2s to 0.5 (not the 1.0 `rms_norm` would give);
  `ssm_conv` sums its rolling window to 10 then 14; `gated_delta_net` returns
  `S*H*T*N + S*S*H*N` finite values with the carried state moved off zero.

**The state does not need in-graph cache writes.** Chaos already keeps its KV
cache host-side as `Vec<u8>` and binds it per layer (`kv.rs`), and `stream.rs`
hands `Vec<f32>` between phases. So the recurrent state is a host-side
`Vec<f32>` per layer, bound as an input and read back after `compute` — no
`ggml_cpy` into a cache view, which is most of what
`delta-net-base.cpp:build_conv_state` spends its length on.

## Left to write

The integration point is the per-layer loop at `crates/chaos-arch/src/stream.rs`
around line 2107: each layer already crosses the ggml boundary as plain vectors,
so a recurrent layer can run its own graph and return `attn_out` without
touching the attention phases at all.

1. `recurrent.rs` — conv window `[3, 10240]` and state `[128, 128, 48]` per
   layer per sequence. ~150 MB across the 48 recurrent layers at f32.
2. The delta-net layer graph, following `build_layer_attn_linear`: `wqkv` and
   `wqkv_gate` matmuls, `ssm_beta`→sigmoid, `ssm_alpha`+`ssm_dt`→softplus→
   `*ssm_a`, conv over the window, silu, split q/k/v, `l2_norm` on q and k,
   `repeat_4d` q and k from 16 heads to 48, the fused op, gated `ssm_norm` by
   `silu(z)`, `ssm_out`.
3. `Qwen3Config` fields for the `qwen35.ssm.*` keys and `is_recurrent(il)`.
4. mRoPE on the 16 attention layers, plus their fused Q/gate projection and the
   sigmoid output gate.
5. `n_layer_nextn` tolerated: the MTP block is loaded and not executed, so the
   loader must not trip over its tensors.

**None of it ships until it is diffed against llama.cpp on Atur's own
container.** `llama-completion` at temperature 0 is the oracle, and the
prompt-length trap applies: prefill and single-token generation take different
paths through a recurrent layer, so agreement on one proves nothing about the
other.

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

## The four pieces, in detail

1. **A recurrent memory, beside the KV cache.** Each recurrent layer needs, per
   sequence: a conv window of `[4, 10240]` and a state of
   `[128, 128, 48]` — head_v_dim × head_k_dim × value heads. That is ~3 MB per
   layer per sequence at f32, ~150 MB across 48 layers. A KV cache cannot stand
   in for it: the state is *carried*, not appended, so eviction, reuse and
   rollback all behave differently. This is the piece that touches the most
   existing code.
2. **The delta net itself.** `ggml_gated_delta_net`
   (`ggml/include/ggml.h:2592`) and `ggml_ssm_conv` — **both now bound and
   tested**, along with `l2_norm` and the 4-D view/reshape/repeat helpers.
   `softplus` and `sigmoid` were already there. Follow
   `build_layer_attn_linear` in `qwen35.cpp` and `build_delta_net_fused`.
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
changes what *memory* means rather than only what a layer computes — but smaller
than it looked before the ops were checked. **The delta rule is one op, not a
chunked scan to reimplement.** What remains is the layer graph, the state cache,
the config, and the verification, and the verification is the half that decides
whether it ships.

## What to tell a user meanwhile

The catalogue already does: the model is listed, and `why_not_runnable` says
"three of its four layers are a gated delta net with recurrent state, not
attention, and Chaos has no recurrent memory yet". That sentence is in the app's
refusal dialog and in `chaos-run`'s output.
