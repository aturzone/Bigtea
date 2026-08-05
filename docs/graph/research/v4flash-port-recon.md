---
topic: DeepSeek-V4-Flash — what the container actually says, and why this model is the one worth porting
status: open
links: [head-to-head-llamacpp-2026-08-05.md, moe-landscape-2026-08.md]
---

Read from `DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf` on 2026-08-05 with
`bigtea-meta` and `bigtea-model-info`. Everything below is from the container, not from a
model card or an advisory.

## Why this is the critical path

Bigtea currently loses to llama.cpp on Qwen3-30B-A3B at every context length, and the reason is
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
whole thesis in one line. Bigtea pins them and they are never read again. llama.cpp mmaps all
144 GB and lets LRU decide, so its dense weights compete with 137 GiB of cold expert traffic for
the same page cache and get evicted — `bigtea-model-info` projects the dense *re-read* per token
climbing from 0.06 GiB at 4k context to 7.38 GiB at 128k, which is the entire dense set being
re-read every token.

This also confirms the retracted claim's origin: 147,169,738,752 bytes = 137.06 GiB is **exactly
the routed-expert total**. llama.cpp's `--repack` tries to allocate that as one buffer outside
the mmap, which is why the default flags fail and `--no-repack` works.

### The bar, and the ceiling

llama.cpp runs it at **0.45 tok/s** (`--no-repack -c 512`, measured).

Bigtea's floor is set by physics: 3.21 GiB of experts per token at the 2.79 GB/s this NVMe gives
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
   each query attends to. This is the piece with no analogue anywhere in Bigtea today.
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

## Open questions

- `expert_gating_func 4` — which function? Read llama.cpp's `deepseek4` implementation rather
  than guessing; getting this wrong degrades quality silently.
- `compress_ratios` has 44 entries for 43 blocks. Off-by-one, or an extra leading/trailing value?
- Does `hash_layer_count 3` mean three layers use a different attention type entirely?
- Is the shared expert always-read (and therefore resident) or routed? If resident, it adds to
  the 7.38 GiB — `bigtea-model-info` already counts it somewhere and that needs confirming.
