---
topic: R13 — Gemma-3 answers plausibly and still disagrees with llama.cpp; two real causes found and fixed, at least one left
status: open — gemma3 NOT added to VERIFIED_ARCHITECTURES
links: [../backlog/lts-parity-criteria.md, gemma2-sliding-window-2026-08-10.md]
---

Gemma-3-1B loads through the dense path with no missing tensor and answers
`"The capital of France is"` with `"Paris."`. **It is still wrong**, and it is
not on the verified list.

```
llama-completion -m gemma-3-1b-it-Q4_K_M.gguf -p "The capital of France is" \
                 -n 10 --temp 0 --no-warmup -no-cnv
  -> Paris.\n\nThe largest city in the world by

bigtea-run -m ... -p "The capital of France is" -n 10 --force
  before -> \n Paris.\n\nThe city of Rome is the capital
  after  -> \n Paris.\n\nIt's a city known for
```

Both are fluent. Both name Paris. Neither is llama.cpp. This is the fourth time
an architecture has loaded cleanly and answered wrongly — after gemma2
(`himſelf`), qwen2 (CJK noise) and the near-miss where phi3 failed loudly
instead.

## Two causes found, both real, both fixed here

**1. The sliding-window layer pattern was hardcoded to Gemma-2's.** The mask
choice was `il % 2 == 0` — Gemma-2 alternates one windowed layer with one full
one. **Gemma-3 is five local to one global** (`set_swa_pattern(6)` in
`gemma3.cpp`), so four layers in six were getting full attention where they
should have been windowed. It is now `layer_is_windowed(il)`, driven by a
`swa_period` read from `attention.sliding_window_pattern` and defaulting per
architecture.

**2. Gemma-3 rotates its windowed and full layers at different frequencies.**
Local layers use `rope.freq_base_swa`, whose llama.cpp default is **10000**;
global layers use the declared `rope.freq_base`, which is **1e6** here. One base
for both leaves every sliding layer rotated at 100x the right frequency — the
same class of bug as the 1e6 RoPE default that broke Phi-3.

The default deserves a note: `rope_freq_base_swa` falls back to the **ordinary
base**, not to llama.cpp's 10000. Only Gemma-3 splits the two, and defaulting
every windowed architecture to 10000 would silently re-rotate Gemma-2's sliding
layers — a regression with no symptom but a wrong answer.

Both fixes are correct and needed regardless of what remains. Neither was enough.

## What has been ruled out

| candidate | checked | verdict |
|---|---|---|
| logit soft-caps | container declares neither key; config defaults to 0 | not it — Gemma-3 dropped them |
| attention scale | llama.cpp uses `1/sqrt(n_embd_head_k)` for 1B; ours is `1/sqrt(head_dim)` and `head_dim` reads `attention.key_length` = 256 | equal |
| `head_dim` from `n_embd / n_head` | would be 288, not 256 | already read from `key_length` |
| QK norm | `attn_q_norm`/`attn_k_norm` present and detected; applied before RoPE, as in `gemma3.cpp` | present |
| post-norms | `post_attention_norm`/`post_ffw_norm` present and detected | present |
| embedding scale | `arch.starts_with("gemma")` covers gemma3 | applied |
| RoPE convention | NeoX, already mapped | correct |

`n_embd` (1152) is deliberately **not** `n_head * head_dim` (4 x 256 = 1024) on
this model, which is worth knowing before reading the shapes.

## What is left

Unknown. The remaining difference is small enough that the model still names
Paris and large enough to change the next clause, which is the hardest kind to
find by reading. The next step is element sums per block against llama.cpp — the
method that settled the V4-Flash port — rather than more inspection.

Candidates not yet excluded: the ordering of the Q pre-scale against the
attention scale (mathematically equal, but not in f32), whether `n_rot` should
be `head_dim` here, and the exact interaction of the windowed mask with the
first `window` positions.

## Why it is not on the list anyway

`VERIFIED_ARCHITECTURES` means "run against llama.cpp and checked", not "loads
and produces words". Gemma-3 loads and produces words. Adding it would make
`--force` unnecessary for a model that answers wrongly, and the whole point of
the list is that a user cannot tell the difference from the output.

## Regression check

All five architectures that have containers here were re-run after the change
and are unchanged: gemma2 `**Paris**.`, phi3 `Paris.`, qwen3 `Paris. The
capital of Germany is Berlin`, llama `Paris. The capital of France is Paris`,
tinyllama `Paris.`. 410 workspace tests, clippy and fmt clean.
