---
topic: Gemma-3 loads, answers plausibly, and is wrong — a fourth architecture caught by the reference rather than by an error
status: measured, NOT supported
links: [gemma2-sliding-window-2026-08-11.md, lts-parity-criteria.md]
---

`gemma-3-1b-it-Q4_K_M` loads through the generic dense path with no missing
tensor. Shape reads correctly: 26 layers, 1152 embd, 4 heads (1 kv), head_dim
256, QK-norm detected. It answers, and the answer looks fine.

```
$ bigtea-run -m gemma-3-1b-it-Q4_K_M.gguf -p "The capital of France is" -n 10 --force
 Paris.

The city of Rome is the capital

$ llama-completion -m ... -p "The capital of France is" -n 10 --temp 0 -no-cnv
The capital of France is Paris.

The largest city in the world by
```

**The first four tokens agree and then it diverges.** `Paris.` plus two
newlines is identical; everything after is different text. A run that stopped at
`-n 3` would have looked like a pass.

This is the **fourth** architecture to load cleanly and be wrong — after
Gemma-2's `himſelf`, Qwen2's `睢已经是成人istentation`, and the pre-tokenizer
that split every BPE container with DeepSeek's rule. It is why
`VERIFIED_ARCHITECTURES` refuses anything whose output has not been read
against the reference.

## What is almost certainly missing

Gemma-3 is not Gemma-2 with a different size. Two changes matter here and
neither announces itself:

1. **The sliding-window pattern is 5:1, not 1:1.** Gemma-2 alternates one
   windowed layer with one full-attention layer. Gemma-3 uses five windowed to
   one global. Our `il % 2 == 0` rule therefore applies the window to the wrong
   layers — and below the window length the two are identical, which is why a
   short prompt cannot reveal it.
2. **RoPE base differs per layer kind.** Gemma-3's local layers use 10,000 and
   its global layers use 1,000,000. A single `rope_freq_base` for the whole
   model is wrong for five layers out of six.

Also: Gemma-3 **drops** the attention and final logit soft-caps that Gemma-2
has, so carrying them over would be a third error.

## What it needs

A per-layer attention description rather than a per-model one: which layers are
windowed, and what RoPE base each uses. That is a small structural change to
`Qwen3Config` — `sliding_window` and `rope_freq_base` become functions of the
layer index — and it is the same shape the next architecture with mixed
attention will need.

**Do not add `gemma3` to `VERIFIED_ARCHITECTURES` until the continuation matches
llama.cpp for at least 32 tokens**, not 4.

## Update: the pattern is fixed, and gemma3 still does not match

`Qwen3Config` now carries `swa_pattern` and `rope_freq_base_swa`, and
`is_swa_layer(il)` / `rope_base_for(il)` replace the hardcoded `il % 2 == 0`.
Gemma-2 is pattern 2, Gemma-3 pattern 6 — one rule for both, and **Gemma-2's
output is byte-identical before and after**, at 10 tokens and at 5201 (above its
window, which is the case that can actually tell).

The container supplies what was expected:

```
gemma3.attention.sliding_window = 512
gemma3.rope.freq_base           = 1000000.0
```

with no `sliding_window_pattern` and no `rope.freq_base_swa`, so the
architecture defaults (6 and 10,000) apply. **Gemma-3 still diverges**, so at
least one more thing differs. Remaining suspects, in the order worth checking:

1. **RoPE scaling applied to the global layers only.** Gemma-3 scales the 1M
   base on its global layers and not its local ones; a single scale for the
   model is wrong for one layer in six.
2. **`query_pre_attn_scalar`.** Gemma-3 divides Q by `sqrt(query_pre_attn_scalar)`
   rather than by `sqrt(head_dim)`. They coincide at 256 for the 1B and do
   **not** coincide for the 4B and 27B, so a check that passes here would still
   be wrong at the next size.
3. **The final norm before the output projection**, which Gemma-3 places
   differently from Gemma-2.

The infrastructure is worth having regardless: every later architecture with
mixed local/global attention needs exactly this, and it is now one rule rather
than a parity hardcoded per model.
