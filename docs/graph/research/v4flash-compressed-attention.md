---
topic: DeepSeek-V4-Flash — the 41 compressed layers, mapped from the trace before building them
status: open
links: [v4flash-port-recon.md]
---

**All 43 blocks and the output head are verified against llama.cpp** at a two-token prompt.
But only layers 0-1 are `Raw`: every other layer passes because at two tokens the compressed
builders never fire (see below). **The compressed attention itself is still unbuilt, on 41 of
43 blocks, and at any real prompt length those blocks take a path that does not exist.** This node maps what those blocks compute, read from the captured trace and
`deepseek4.cpp` rather than reasoned about, so the build has its reference before it starts.

Fixtures already extracted from the existing five-token capture — no new llama.cpp run is
needed to start:

```
crates/chaos-arch/tests/fixtures/v4flash-layer2-oracle-5tok.txt   CSA + indexer, 273 rows
crates/chaos-arch/tests/fixtures/v4flash-layer3-oracle-5tok.txt   HCA,           133 rows
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

## Prompt length decides what is observable — measured, not reasoned

The single most useful planning fact from this whole port. Each row is a capture that exists
in `tests/fixtures/`.

| tokens | Raw attn | CSA compress | CSA attn | HCA compress | HCA attn | indexer selects |
|---|---|---|---|---|---|---|
| 1  | ✅ (RoPE is identity) | — | — | — | — | — |
| 2  | ✅ | writes only | **falls back to Raw** | writes only | **falls back to Raw** | — |
| 5  | ✅ | ✅ ratio 4 | ✅ | writes only | **falls back to Raw** | inert |
| 165| ✅ (layers 0-1) | ✅ | ✅ | ✅ ratio 128 | ✅ | inert |
| >2048 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

The compressed builders are guarded on their compressed caches being non-empty, so **a layer
silently runs a different attention at different prompt lengths**. Two tokens was what let
all 43 blocks be verified before any compressed attention existed; 165 is what makes HCA
observable at all; and nothing below ~2048 can exercise the indexer, because
`n_top_k = min(n_lid, indexer_top_k)` selects everything until `n_lid` exceeds 512.

**Whether >2048 is verifiable on this machine is a separate question, and probably no**: at
that length every layer routes to all 256 experts, which is 3.19 GiB of expert weight per
layer. Per-layer contexts bound that to one layer at a time, so it is not hopeless, but it is
right at the edge of the ~5 GiB free here.

## HCA — the map (still unbuilt)

**It needs its own compressor.** `build_hca_compressed_kv_from_state` is a *different*
function from the overlap compressor CSA uses, and its state is `{512, 128}` — head-wide,
not `2*head` — so it is not the same code with a different ratio. That is the one thing the
five-token map got wrong by omission.

Its attention half, by contrast, is exactly CSA's: `concat(raw_k, hca_k, 2)`,
`concat(raw_mask, hca_mask, 0)`, then the same fused kernel — no indexer, no top-k mask. At
165 tokens `hca_k_all` is `{512, 1, 512}`, identical to `csa_k_all`. So building HCA is one
new compressor plus a call into machinery that already exists and is verified.

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

## CSA — BUILT (2026-08-06), except the part that cannot be checked

Verified against llama.cpp on layer 2 at five tokens:

```
csa_comp-2       5.163465    the attention compressor  (head 512)
lid_comp-2      27.113394    the indexer's compressor  (head 128)
lid_comp_rot-2  -8.839936    Walsh-Hadamard, regenerated from scratch
lid_q_pe-2      17.747589    the indexer's RoPE, and this one really rotates
csa_k_all-2    125.152557    raw window + compressed summaries
flash_attn-2  4314.597656    CSA attention
```

Three things this settled that the map below only guessed at:

* **The overlap compressor is one function run twice** — 512-wide for attention,
  128-wide for the indexer — differing only in weights and head width. The nope/pe split
  must follow *that* head (128-64), not the attention head's 448; using the wrong one aborts
  ggml on the indexer and would be silently correct on the attention side.
* **The Hadamard rotation is generated, never stored.** Nothing in the container holds it.
  llama.cpp builds an orthonormal Walsh-Hadamard at cache-init for DeepSeek indexers
  unconditionally, in a `static` helper whose `ggml_` prefix makes it look like a ggml op it
  is not. Sylvester's construction scaled by `1/sqrt(n)`; it is its own inverse.
* **The persistent cache was not on the critical path** — see the state section below.

### What could not be checked, and why

**The indexer's selection is inert below ~2048 tokens.**
`n_top_k = min(n_lid, indexer_top_k)` = `min(256, 512)` = 256, so it selects *every*
compressed slot; `build_top_k_mask` then leaves exactly the visibility mask. Making the
sparsity bite needs `n_lid > 512`, so more than 512 completed blocks, so over 2048 tokens.

**And the trace could not help even if it did fire.** `lid_score_masked`, `csa_top_k_mask`
and `csa_lid_kq_mask` all sum to `-inf`; `lid_top_k` sums to 163200 for *any* permutation of
256. Four rows with no discriminating power. `attn_csa_lid` is the only number in the
indexer's half of the layer that can fail, so it is the one asserted.

**The compressors' own RoPE is unverified.** `comp_pos` is the block's start position, 0 for
the first block, so at five tokens that rotation is the identity — the same shape of hole the
one-token capture had. It needs a prompt long enough for a second block (≥8 tokens).

## HCA — the map, from before any of it was built

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

## Two holes in the *verified* work — BOTH NOW CLOSED (2026-08-06)

Closed not by building the compressed attention, but by capturing a **shorter** prompt.
The compressed builders are guarded on their compressed caches being populated
(`deepseek4.cpp:1049-1063`); at **two tokens** those caches are empty, so layers 2 and 3
fall through to `build_raw_attention` — already built, already verified. Their compressor
projections still run; nothing reads them at that length.

Layers 0-3 now run end to end at two tokens, finishing on `next_norm-3` = 427.686554, the
exact tensor layer 4 receives. Both items below executed for the first time. Kept as written
because the *reasoning* still applies to the compressed paths, which remain unbuilt.

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

## The CSA compressor, and why its state is tractable after all

`build_overlap_compressed_kv_from_state` (`deepseek4.cpp`) reads a *persistent* state through
index tensors — `state_read_idxs`, `state_write_idxs`, `comp_pos` — that llama.cpp computes in
`llama-kv-cache-dsv4.cpp` (1978 lines), not in the graph. That looked like the blocker: port a
whole cache class before a single number can be checked.

**It is not, for a prefill from an empty cache**, which is the only case that needs to work
first. Read from `llama-kv-cache-dsv4.cpp:437-535`, the scheme is:

```
state_pos[i]   = pos % ratio            position within the current block
n_visible[i]   = (pos + 1) / ratio      compressed entries this token may attend to
n_kv           = max(n_visible)
a block completes when (pos + 1) % ratio == 0
  state_write_idxs <- cache_off + pos/ratio
  state_write_pos  <- source_start = pos + 1 - ratio
  overlap reads: prev window [source_start - ratio, source_start)
                 cur  window [source_start, pos]
```

and `state_source_idx(pos)` resolves to the appended zero row when `pos < 0`, to
`state_rows + i` when the position is in the current ubatch, and only otherwise to the
persistent ring. **On a fresh prefill every position is in the current ubatch**, so the first
two cases cover everything and the ring never gets read.

At five tokens with `ratio = 4` that gives exactly `n_blocks = 1`, `n_kv = 1`: the state is
`{1024, 8}` of zeros concatenated with the 5 current rows and a zero row appended, indices
`8..11` for the current window and `13` (the zero row) four times for the previous one.
The trace agrees — `node_265 GET_ROWS {1024, 8}` is `2*ratio*n_blocks`.

**The `{1024, …}` width is two entries per row**, not one: `kv_state->ne[0] == 2*n_embd_head`,
with `kv_prev` reading the first 512 of the first `n_read` rows and `kv_cur` the *second* 512
of the next `n_read`. That is the "overlap" — each compressed entry summarises `2*ratio` raw
positions, two windows deep.

The arithmetic itself is ordinary: softmax the scores, multiply, `sum_rows`, RMS-norm with
`attn_compressor_norm`, then RoPE the trailing 64 dims **at the block position** with the
compressed base. Every step has a trace row.

So the build is: construct the index tensors directly for the prefill case (a dozen lines,
no cache class), then follow the fourteen graph steps. The persistent ring only becomes
necessary for generation, where positions from earlier ubatches are read back.

## Order — CORRECTED

An earlier version of this node said to do layer 3 (HCA) before layer 2 (CSA), on the
grounds that HCA is much simpler. **That order is impossible.** Layers chain: layer 3's
input is layer 2's output, and there is no way to seed it — the trace gives element sums,
not tensors. The alternation puts CSA at every even layer, so **the simpler kind sits behind
the harder one and CSA must be built first.**

Recorded rather than quietly fixed, because the mistake is instructive: "smallest next step"
was chosen by looking at the two kinds in isolation and not at what feeds what.

0. ~~Both holes~~ **closed** — see above. A shorter capture reached further than a longer
   one, which is the opposite of the intuition that produced the original ordering.
1. ~~Finish layer 1~~ **done** — both `Raw` layers run in full through one generic layer
   function, and the helpers now take a block index plus a table of that layer's sums.
2. ~~The seam into layer 2~~ **done** — a layer's *entry* (hyper-connection gates,
   `attn_norm`) does not depend on which attention follows, so it was checkable with no CSA
   code at all: `attn_norm-2` = 5.640476 is exactly what CSA attention will consume. When CSA
   is built, only the attention itself is new.
3. **Layer 2's CSA attention** — the actual next build, and unavoidably the hard one.
   Everything before and after it in the block is already verified, so the work is bounded to
   `attn_norm-2` → `attn_out-2`.
4. **Layer 3's HCA attention**, which then follows almost for free — its only additions over
   Raw are two matmuls and an APE add.
5. **A capture longer than 128 tokens**, to make HCA's compression observable at all — at
   five tokens `hca_state_compress` never runs, and at two neither compressor is read.
6. **Per-layer contexts**, so depth stops costing arena. See below; it is also the shape the
   streaming runner needs, so it is not scaffolding.

### The shortcut, tried and better than expected

The guess was that ≤3 tokens would reduce CSA to "the indexer over raw keys". It does more
than that: **at two tokens the compressed builders do not run at all.** The whole of layers
2 and 3 goes through the Raw path.

### What stops this from covering all 43 layers today

Memory, not correctness. Four layers in one `ggml` context needs a 2.5 GiB arena, and each
layer's routed experts add ~150 MiB of slices at two tokens. Eight layers would want ~5 GiB
of arena alone, against 5.2 GiB free on this machine.

Freeing a layer's weights as the chain advances is **not** safe as the code stands: every
`compute` rebuilds the graph back through its sources, and a dropped weight buffer leaves a
dangling pointer that reads freed memory *successfully*.

The fix is a **per-layer context**: give each layer its own arena and `WeightSet`, seed it
from the previous layer's output as a plain `Vec<f32>`, and drop the whole thing before the
next. That bounds memory to one layer regardless of depth — and it is what the streaming
runner has to do anyway, so it is not scaffolding. That is the next structural step, and it
is what would let all 43 layers be verified in one run.
