---
topic: DeepSeek-V4-Flash — what the container actually says, and why this model is the one worth porting
status: open
links: [head-to-head-llamacpp-2026-08-05.md, moe-landscape-2026-08.md]
---

Read from `DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf` on 2026-08-05 with
`chaos-meta` and `chaos-model-info`. Everything below is from the container, not from a
model card or an advisory.

## Why this is the critical path

Chaos currently loses to llama.cpp on Qwen3-30B-A3B at every context length, and the reason is
structural: that model *nearly fits*, so the kernel's page cache — elastic, free, and using all
available RAM — beats a fixed hand-managed budget. There is no version of that fight we win.

V4-Flash inverts it, and the container says so:

```
always-read        7.38 GiB   read on every token
routed expert    137.06 GiB   read only when selected
total            144.44 GiB
routing        6 of 256 experts per token -> 3.21 GiB of experts read per token
```

**The 7.38 GiB of always-read weights fit in this machine's ~11 GiB of free RAM.** That is the
whole thesis in one line. Chaos pins them and they are never read again. llama.cpp mmaps all
144 GB and lets LRU decide, so its dense weights compete with 137 GiB of cold expert traffic for
the same page cache and get evicted — `chaos-model-info` projects the dense *re-read* per token
climbing from 0.06 GiB at 4k context to 7.38 GiB at 128k, which is the entire dense set being
re-read every token.

This also confirms the retracted claim's origin: 147,169,738,752 bytes = 137.06 GiB is **exactly
the routed-expert total**. llama.cpp's `--repack` tries to allocate that as one buffer outside
the mmap, which is why the default flags fail and `--no-repack` works.

### The bar, and the ceiling

llama.cpp runs it at **0.45 tok/s** (`--no-repack -c 512`, measured).

Chaos's floor is set by physics: 3.21 GiB of experts per token at the 2.79 GB/s this NVMe gives
across threads is ~1.15s/token, so **~0.87 tok/s with a cold cache and perfect streaming** —
roughly 2x llama.cpp. Any expert cache hits push it higher; 256 experts with 6 used is a skew
our frequency-gated admission should exploit better than LRU, which is the one policy result we
have already proven on Qwen3 (70% vs 17% at equal budget).

Neither figure is usable for agent work — 0.87 tok/s is still a 10-minute answer. Worth being
honest that this port proves the *design*, and does not by itself produce a coding assistant.

## What the architecture needs (all from container metadata)

```
block_count 43              embedding_length 4096       context_length 1,048,576
attention.head_count 64     head_count_kv 1             key_length 512, value_length 512
attention.q_lora_rank 1024  output_lora_rank 1024       output_group_count 8
attention.sliding_window 128
attention.compress_ratios   [44 per-layer values]       compress_rope_freq_base 160000
attention.indexer.head_count 64, key_length 128, top_k 512
expert_count 256            expert_used_count 6         expert_shared_count 1
expert_feed_forward_length 2048                         expert_gating_func 4
expert_weights_scale 1.5    expert_weights_norm true
hyper_connection.count 4    sinkhorn_iterations 20      epsilon 1e-6
rope.scaling type "yarn", factor 16, original_context_length 65536, dimension_count 64
swiglu_clamp_exp / swiglu_clamp_shexp  [43 per-layer values each]
hash_layer_count 3
1328 tensors in shard 0 (of 5)
```

Distinct pieces of work, roughly in dependency order:

1. **MLA-style compressed attention.** `head_count_kv 1` with `key_length`/`value_length` 512
   and `q_lora_rank`/`output_lora_rank` 1024 means Q and the KV cache are low-rank projections,
   not the plain per-head K/V that `KvCache` stores today. The cache layout has to change:
   the current `n_kv_heads * head_dim` per position does not describe this.
2. **YaRN RoPE scaling.** `rope_ext` is already bound and takes ext_factor/beta_fast/beta_slow;
   this is mostly plumbing the container's yarn parameters through instead of defaults.
3. **Per-layer compression ratios** — 44 values for 43 blocks (the off-by-one needs checking
   against llama.cpp's reader before assuming which is which).
4. **The sparse attention indexer** (64 heads, key length 128, top_k 512) selecting which keys
   each query attends to. This is the piece with no analogue anywhere in Chaos today.
5. **Hyper-connections with 20 Sinkhorn iterations**, replacing plain residual addition.
6. **MoE differences from Qwen3**: a shared expert always active, `expert_gating_func 4`
   (sigmoid rather than softmax — verify against llama.cpp), an explicit `expert_weights_scale`
   of 1.5, and per-layer SwiGLU clamping.
7. `hash_layer_count 3` — unexplained, needs reading llama.cpp's `deepseek4` loader.

The streaming machinery underneath — residency, direct I/O, expert grouping, frequency-gated
caching, parallel reads — is architecture-independent and already works. This port is attention
and routing, not plumbing.

## Suggested staging

Do not port all of it at once. In order, each step verifiable on its own:

1. Container + tensor-name verification, `arch.verify()` passing on all 1328 tensors across
   5 shards, no forward pass. Cheap, and catches naming surprises early.
2. Dense path with plain attention and no indexer — wrong output, but proves shapes, shard
   resolution and residency at 144 GB.
3. MLA attention + YaRN, checked against llama.cpp's logits for the same prompt. **A wrong
   forward pass here produces fluent nonsense, not a crash**, so compare numbers, not vibes.
4. MoE with shared expert and the gating differences.
5. Indexer and hyper-connections last — the model may produce plausible text without them,
   which makes it dangerously easy to declare victory early.

## CORRECTION BLOCK (2026-08-05, later the same day)

Three of the items scoped above are cheaper than this node claimed, and one is
harder. All checked against `llama.cpp/src/models/deepseek4.cpp` and the
container rather than reasoned about.

**Cheaper — ggml already implements them.** This build's *public* `ggml.h`
exposes `ggml_dsv4_hc_pre`, `ggml_dsv4_hc_post`, `ggml_dsv4_hc_comb` (whose
`n_iter` argument is the Sinkhorn iteration count) and
`ggml_lightning_indexer`, plus `ggml_flash_attn_ext_add_sinks` for the
per-head sinks. **Hyper-connections and the sparse indexer are FFI bindings,
not implementations** — items 4 and 5 of the scoping list above were the two
flagged as hardest and both largely evaporate.

**Harder — attention is not one thing.** The model dispatches to *three*
different attention builders, chosen per layer: 2 raw, 20 heavily-compressed,
21 compressed-sparse. Implementing one and applying it throughout gives fluent
output that is wrong on half the model.

**Resolved open questions.** `hash_layer_count 3` means the three layers
carrying `ffn_gate_tid2eid`. `compress_ratios` having 44 entries for 43 blocks
is not an off-by-one in the manifest — it is indexed per layer as
`dsv4_compress_ratios[il]` and selects the RoPE base, so only the first 43 are
consulted.

**A numerical reference now exists.**
`crates/chaos-arch/tests/fixtures/v4flash-layer0-oracle.txt` holds the shape
and element-sum of every tensor in the prologue and layer 0, captured with
`llama-eval-callback` on the real container. That is the oracle the forward
pass gets built against. It already caught one thing invisible in the shapes:
the attention output is **de-roped** (`rope_back`) before the grouped output
projection.

## CORRECTION BLOCK (2026-08-06) — the oracle was blind to RoPE, and now is not

The previous block announced a numerical reference. That reference had a hole,
found while building against it and recorded here rather than left implicit.

**The one-token capture cannot validate RoPE, at all.** The prompt was `"Hi"` —
a single token at position 0, where the rotation is the identity. In that trace
`q_pe` has exactly the same sum as its input, and so does `kv_pe`. Every RoPE
implementation passes those two rows, *including one that does nothing*. The
decoupled rotation is one of the five things this architecture makes easy to get
wrong, and it was the one the oracle silently exempted.

**Closed by a second capture at five tokens** —
`tests/fixtures/v4flash-layer0-oracle-5tok.txt`, from
`-p "The capital of France is"` (ids 671, 6102, 294, 8760, 344), same flags.
Positions 0..4 make the rotation real:

```
q_norm-0 (view)  {64, 64, 5}    695.835632  ->  q_pe-0     4082.126465
kv_norm-0 (view) {64,  1, 5}     24.049295  ->  kv_pe-0      76.641815
attn_raw (view)  {64, 64, 5}   3432.786621  ->  ROPE_BACK    28.466785
```

Chaos now matches all three-plus-25 checkpoints through the end of the KV
projection. The one-token fixture is kept: matching two independent inputs is
stronger than matching one.

RoPE parameters for layer 0, from `deepseek4.cpp:822-829` rather than from the
container's top-level keys: `dsv4_compress_ratios[0] == 0`, so it takes the
*uncompressed* path — plain `freq_base` 10000, `freq_scale` 1.0, and scaling
switched off entirely (`ext_factor` 0, `attn_factor` 1, both betas 0,
`n_ctx_orig` 0). **The container's YaRN settings apply to the other 41 layers,
not this one.** deepseek4 also maps to `LLAMA_ROPE_TYPE_NORM`, not NEOX.

**`expert_gating_func 4` is resolved** — `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`
(`llama-hparams.h:18`), i.e. `sqrt(softplus(logits))`, which the trace confirms
as `MUL_MAT -> SOFTPLUS -> SQRT`. Neither softmax nor sigmoid, and the five-token
capture reaches far enough to check it: it covers all of layer 0 through the MoE
and the shared expert.

**A library bug this found.** `Tensor::to_vec_f32` read `nelements` floats
straight off the data pointer, ignoring strides. A decoupled-RoPE view is 64 of
every 512 dims and therefore *not* contiguous, so the readback returned the
right count of plausible floats and all of them the wrong ones — making a
correct graph look broken. Now stride-aware, with two hand-checkable unit tests
that need no container.

## The whole forward pass is verified (2026-08-06)

**All 43 blocks plus the output head match llama.cpp**, from a token id to a 129280-wide
logit vector — roughly sixty checkpoints per layer. At the two-token prompt `"Hello there"`
the model predicts `","`, which is the sanity check on top of the numbers.

Reached by two things, neither of them more architecture code:

* **A two-token capture.** The compressed attention builders are guarded on their compressed
  caches being populated (`deepseek4.cpp:1049-1063`); at two tokens those caches are empty,
  so every layer falls through to `build_raw_attention`, which was already built. A *shorter*
  capture reached further than a longer one.
* **Per-layer contexts.** Chaining layers in one `ggml` context costs ~640 MiB of arena each.
  Each layer now builds its own arena and `WeightSet`, runs, hands its residual streams out
  as a plain `Vec<f32>`, and drops everything — so depth is free, and the `Vec` boundary is
  what makes the drop sound. (Freeing weights inside one context is not: every `compute`
  rebuilds the graph through its sources, so a dropped buffer reads freed memory
  *successfully*.) It is also the shape the streaming runner needs.

**This is not a complete implementation.** At any real prompt length 41 of 43 blocks take a
different attention path — the compressors and the lightning indexer are still unbuilt, and
the sliding window and SwiGLU clamp bounds are still unreached. See
`v4flash-compressed-attention.md`.

What the build found on the way, none of which is derivable from tensor shapes:

- **Layers 0-2 do not use top-k routing.** `ffn_moe_topk-0` is
  `GET_ROWS(blk.0.ffn_gate_tid2eid.weight{6, 129280}, inp_tokens)` — the six experts are a
  lookup on the *token id*. That is what `hash_layer_count 3` means operationally. The router
  probabilities are still computed, but only to weight experts already chosen. **For the
  streaming design this is a genuine opportunity: on those three layers the expert set is
  knowable from the token id before any compute runs**, so their reads can be issued as early
  as tokenisation.
- The gate is `sqrt(softplus(x))`; weights are renormalised over the selected six only, then
  scaled by 1.5, with the divisor clamped at the smallest F16 normal (6.103515625e-5).
- The SwiGLU clamp is **asymmetric on the gate**: `(-inf, 10]` for the gate, `[-10, 10]` for
  up, in an `LLM_ARCH_DEEPSEEK4` branch (`llama-graph.cpp:2050-2057`).
- The post hyper-connection replaces the residual add entirely:
  `x[dst] = f(x)*post[dst] + sum_src x[src]*comb[dst, src]`. `pre` ends with
  `scale_bias(x, 1, hc_eps)` and `post` with `scale(x, 2.0)` — different tails, same shape.
  The FFN's gates come from a *second* mixes matmul against `hc_ffn_fn`, not the attention
  block's.
- Attention: `kv` is K *and* V; the cache is F16 so the reference sum is the rounded one;
  `n_kv` is padded to 256 and the unused slots need `-inf`; per-head sinks shift the result
  ~10%.

**The routed experts are done, via partial reads.** Binding all three stacked tensors for
layer 0 is 3.19 GiB and did not fit (5.2 GiB available, 4.2 usable). Instead the experts the
five tokens actually route to are fetched with `read_tensor_range` and packed into a compact
stack with the ids remapped:

```
routed          29 of 256 experts, 30 slots
expert slices   0.36 GiB read (all 256 would be 3.19 GiB)
```

**8.9x fewer bytes for identical arithmetic**, and this is the first time the partial-read
path has been checked against a reference. It is not a test convenience — it is what the
runner has to do. Per-expert contributions are asserted individually, so a mis-slotted
expert cannot hide inside the total.

**Layers compose.** Layer 1 has no embedding and no `hc_init`; it consumes `l_last-0`
directly, so `hc_mixes-1 = -3428.892578` is one number standing in for the correctness of
all of layer 0. Layer 1 is also the *second* `Raw` layer, so it runs the same code with
entirely different weights — an implementation accidentally fitted to layer 0 has nowhere
to hide. Only layer 1's entry and first matmul are checked; running it fully needs the
remaining helpers parameterised by layer index.

**Per-layer hyper-parameters are read now.** `swiglu_clamp_exp`, `swiglu_clamp_shexp` and
`compress_ratios` are arrays, and `Deepseek4Config` previously read none of them.
`rope_for_layer(il)` picks the branch the way `deepseek4.cpp:822-829` does, and the forward
tests call it rather than a local copy of the rules — so the checkpoints exercise shipped
code.

### Holes recorded rather than papered over

- **The SwiGLU clamp bounds are unverified.** At five tokens neither is reached: the clamped
  sums equal the unclamped ones. The capture confirms the shape of the computation, not the
  numbers. Same class as the one-token RoPE hole.
- **The sliding window is unverified.** Raw layers are SWA layers with window 128
  (`GGML_ASSERT(hparams.is_swa(il))`), but five tokens never reach back that far, so the test
  uses a plain causal mask. Needs a capture longer than 128 tokens.
- ~~`swiglu_clamp_exp` / `swiglu_clamp_shexp` are per-layer arrays and `Deepseek4Config` reads
  neither.~~ **Fixed** — along with `compress_ratios` and `rope_for_layer(il)`.
- **The compressed RoPE branch is transcribed, not verified.** `rope_for_layer` returns YaRN
  parameters for the 41 compressed layers, taken from `deepseek4.cpp` — but the oracle stops
  inside layer 1 and both Raw layers are uncompressed, so no capture has ever exercised it.

## Open questions

- `compress_ratios` has 44 entries for 43 blocks. Off-by-one, or an extra leading/trailing value?
  (Partly answered above: only the first 43 are consulted.)
- Is the shared expert always-read (and therefore resident) or routed? If resident, it adds to
  the 7.38 GiB — `chaos-model-info` already counts it somewhere and that needs confirming.
  The five-token trace shows `ffn_shexp-0` running unconditionally, which is consistent with
  resident, but the byte accounting has not been re-checked.
- **Still unverified: the other two attention kinds.** Layer 0 is `Raw`. The 20
  heavily-compressed and 21 compressed-sparse layers have no oracle rows yet — the capture stops
  where layer 1 begins. Matching layer 0 says nothing about them. Note the same 5-token dump in
  the scratchpad *does* contain layers 1-42; extracting layer 1 (also `Raw`) is nearly free,
  but a compressed layer needs its rows pulled out too before that path can be built.

### Resolved and struck from this list

- ~~`expert_gating_func 4` — which function?~~ `sqrt(softplus(x))`, see the 2026-08-06 block.
- ~~Does `hash_layer_count 3` mean three layers use a different attention type entirely?~~ No —
  it is the three layers carrying `ffn_gate_tid2eid`, see the 2026-08-05 block.
