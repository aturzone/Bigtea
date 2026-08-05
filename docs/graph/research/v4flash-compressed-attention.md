---
topic: DeepSeek-V4-Flash — the 41 compressed layers, mapped from the trace before building them
status: open
links: [v4flash-port-recon.md]
---

Layers 0 and 1 are verified against llama.cpp. They are also the only two `Raw` layers, so
**41 of 43 blocks run code that does not exist yet**. This node maps what those blocks
actually compute, read from the captured trace and `deepseek4.cpp` rather than reasoned
about, so the build has its reference before it starts.

Fixtures already extracted from the existing five-token capture — no new llama.cpp run is
needed to start:

```
crates/bigtea-arch/tests/fixtures/v4flash-layer2-oracle-5tok.txt   CSA + indexer, 273 rows
crates/bigtea-arch/tests/fixtures/v4flash-layer3-oracle-5tok.txt   HCA,           133 rows
```

## The layer pattern, which nothing in the metadata announces

```
layer 0, 1     Raw                     2 layers
layer 2,4,6…   CompressedSparse (CSA)  21 layers   even
layer 3,5,7…   HeavilyCompressed (HCA) 20 layers   odd
```

**After the first two, the kind alternates.** A layer loop that picks one builder and applies
it throughout is wrong on half the model and produces fluent text while doing it.

`compress_ratios` is not a compressed/uncompressed flag — **its value names the builder**:
0 raw, `DSV4_CSA_RATIO` 4, `DSV4_HCA_RATIO` 128 (`deepseek4.cpp:187-188`). That gives two
fully independent readings of the same fact, since `Deepseek4Model::attention_kind` derives
it instead from which tensors a block ships. `deepseek4_container.rs` now asserts the two
agree on all 43 layers; a divergence means one reading is wrong and some layer would run
through the wrong attention.

## HCA — the simpler kind, but *not* the first one reachable

133 rows against Raw's 121, so it looks like the obvious next step. It is not: layer 3 sits
behind layer 2, and layers chain. See the corrected order at the bottom of this node.

Over a Raw layer it adds exactly three ops, all state maintenance:

```
hca_state_kv-3         MUL_MAT  {512, 5}   attn_compressor_kv   · x
hca_state_score-3      MUL_MAT  {512, 5}   attn_compressor_gate · x
hca_state_score_ape-3  ADD      {512, 5}   + attn_compressor_ape[state_pos]
```

written into `dsv4_hca_state_{kv,score}_l3` `{512, 128}` with `SET_ROWS`. Everything else is
the Raw path already built.

**At five tokens the compression never runs.** `hca_state_compress` appears zero times: the
block size is 128, so the state is written and never consumed. So an HCA layer at this
prompt length *is* the Raw path plus two matmuls, and the existing capture can confirm the
state writes but not the compression. Same family of hole as the one-token RoPE capture —
proving `build_hca_compressed_kv_from_state` needs a prompt of at least 128 tokens.

## CSA — the hardest kind, and the one with no analogue here

273 rows. It carries **two** compressor states, not one: `csa_state_*` feeding attention and
`lid_state_*` feeding the indexer, each with its own kv/score/ape projections and its own
persistent tensor. The indexer chain:

```
lid_q / lid_q_pe / lid_q_rot     the indexer's own rotated query
lid_k
lid_weights -> lid_score_masked -> lid_top_k
csa_top_k_mask -> csa_lid_kq_mask         the mask attention finally uses
csa_raw_k + csa_comp_k -> csa_k_all       raw and compressed keys concatenated
attn_csa_lid
```

Unlike HCA, **CSA does compress at five tokens** — ratio 4, so `csa_state_compress` (9 rows)
and `lid_state_compress` (11 rows) both run. `lid_state_compress_rot` applies a Hadamard
rotation (`llama_mul_mat_hadamard`) that no other path uses. `ggml_lightning_indexer` is
already bound, so the indexer scoring itself is a call, not an implementation.

## Two holes in the *verified* work that these layers expose

Both are cases where passing tests cover less than they appear to.

**The compressed RoPE branch has never run.** `Deepseek4Config::rope_for_layer` returns
YaRN parameters (base 160000) for all 41 compressed layers, transcribed from
`deepseek4.cpp:822-829` and checked against nothing — both `Raw` layers are uncompressed.
Layer 3 is where it becomes checkable, and the rotation is real there:

```
q_norm-3 (view)  {64, 64, 5}  4240.427734  ->  q_pe-3   3631.148682
kv_norm-3 (view) {64,  1, 5}   -94.538872  ->  kv_pe-3  -103.840263
```

**The normal MoE routing path has never run either.** Layers 0-2 are the three
`hash_layer_count` layers and select experts by token-id lookup. The other 40 use:

```
ffn_moe_probs_biased-3  ADD      {256, 5}   probs + exp_probs_b
ffn_moe_argsort-3       ARGSORT  {256, 5}
ffn_moe_topk-3          VIEW     {6, 5}     first 6 of the argsort
ffn_moe_weights-3       GET_ROWS {1, 6, 5}  from the *unbiased* probs
```

**The bias steers selection only** — the weights are gathered from the unbiased
probabilities. Applying it to both is the natural mistake, changes every expert weight, and
changes no shape. Note also this is `ARGSORT` plus a view, not a `TOP_K`, and `top_k does
not return indices in score order` is already a rediscovered-the-hard-way entry in
`CLAUDE.md`.

## Order — CORRECTED

An earlier version of this node said to do layer 3 (HCA) before layer 2 (CSA), on the
grounds that HCA is much simpler. **That order is impossible.** Layers chain: layer 3's
input is layer 2's output, and there is no way to seed it — the trace gives element sums,
not tensors. The alternation puts CSA at every even layer, so **the simpler kind sits behind
the harder one and CSA must be built first.**

Recorded rather than quietly fixed, because the mistake is instructive: "smallest next step"
was chosen by looking at the two kinds in isolation and not at what feeds what.

1. ~~Finish layer 1~~ **done** — both `Raw` layers run in full through one generic layer
   function, and the helpers now take a block index plus a table of that layer's sums.
2. ~~The seam into layer 2~~ **done** — a layer's *entry* (hyper-connection gates,
   `attn_norm`) does not depend on which attention follows, so it was checkable with no CSA
   code at all: `attn_norm-2` = 5.640476 is exactly what CSA attention will consume. When CSA
   is built, only the attention itself is new.
3. **Layer 2's CSA attention** — the actual next build, and unavoidably the hard one.
   Everything before and after it in the block is already verified, so the work is bounded to
   `attn_norm-2` → `attn_out-2`.
4. **Layer 3 (HCA)**, which then follows almost for free and closes both holes above.
5. **A capture longer than 128 tokens**, to make HCA's compression observable at all — at
   five tokens `hca_state_compress` never runs.

### A shortcut worth considering for step 3

CSA compresses every 4 tokens. **A capture of 3 or fewer tokens would reduce layer 2 to CSA
without any compression**, leaving only the indexer over raw keys — a strictly smaller first
target, and positions 1-2 still make RoPE checkable. Untried, but cheap to find out.
