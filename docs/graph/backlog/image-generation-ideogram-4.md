# Generating images — what Ideogram 4 would actually take

> Atur asked for Ideogram 4 in the release. It **is** open-weight, so "we cannot
> get it" was never the answer. This is the real answer.

**Status: two of four foundations built and proven, 2026-08-19. The engine
itself is not started.**

## Done, and how it was checked

| piece | state |
|---|---|
| PNG output | **done** — round-trip test, Windows GDI+ decoding exact pixels, visual render |
| safetensors reading | **done** — against the real 251-tensor VAE header |
| all four files fetchable | **done** — `chaos-pull` gets them, 16.65 GB total |
| the denoiser forward pass | not started |
| the sampler loop | not started |
| text conditioning | not started |
| VAE decode | not started |

## Numbers, read from the containers rather than guessed

**The denoiser** (`ideogram4-Q4_0.gguf`, 458 tensors, **zero metadata keys** —
identified by tensor names via `catalogue::architecture_from_tensors`):

- 34 layers, hidden 4608
- `attention.qkv.weight [4608, 13824]` — fused QKV, 3 x 4608
- `norm_q` / `norm_k` are `[256]`, so head_dim 256 and **18 heads**
- SwiGLU: `w1`/`w3` `[4608, 12288]`, `w2` `[12288, 4608]`
- **sandwich norms**: `attention_norm1` *and* `attention_norm2`, `ffn_norm1` and
  `ffn_norm2`
- `adaln_modulation [512, 18432]` — 18432 = 4 x 4608, so **four modulation
  signals** per layer from a **512-wide** conditioning vector
- `t_embedding.mlp_in` — the timestep embedding no language model has
- `llm_cond_proj`, `llm_cond_norm` — where the text encoder's hidden states enter
- `input_proj [128, 4608]` and `final_layer.linear [4608, 128]` — **128 patch
  channels**

**The autoencoder** (`flux2-vae.safetensors`, 251 tensors, F32 + one I64 scalar):

- `decoder.conv_in.weight [512, 32, 3, 3]` — **32 latent channels**
- `decoder.conv_out.bias [3]` — ends in RGB
- `decoder.mid_block.{resnets,attentions}` — convolutions, group norms and
  attention

**The two agree**: 32 latent channels x a 2x2 patch = 128, which is exactly the
denoiser's patch-channel count. Two repositories, one undocumented interface.

## The gate that is not obvious

`black-forest-labs/FLUX.2-dev` is **gated** and answers 401 without an accepted
licence. `Comfy-Org/flux2-dev` mirrors the same weights ungated, and that is what
the catalogue points at.


## What is true

Ideogram 4.0 is a 9.3B **diffusion transformer**, open-weight since 3 June 2026,
with GGUF conversions on Hugging Face made by leejet — the author of
`stable-diffusion.cpp`, which is built on the same ggml this project links.

It is in the catalogue now, listed and refused, alongside the Qwen3.5 entries.

## Why Chaos cannot run it today

Chaos is a token loop: embed a token, run a stack of attention and FFN blocks,
sample the next token, repeat. An image is not that shape. From leejet's own
README, one image needs **four** components:

| part | file | what it does |
|---|---|---|
| the denoiser | `ideogram4-Q4_0.gguf`, 5.64 GB | predicts noise at each step |
| its unconditional twin | `ideogram4_uncond-Q4_0.gguf`, 5.64 GB | the other half of classifier-free guidance |
| the text encoder | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | turns the prompt into conditioning |
| the VAE | `flux2_ae.safetensors` | turns the final latent into pixels |

Chaos has code for **one** of those four, and only sort of: Qwen3-VL is a vision
LLM, not one of the thirteen verified architectures.

Missing outright:

- a **sampler loop** — Euler / DPM++ over 20–50 steps, with the guidance
  combination of the two denoisers at every step;
- **conv2d, group norm and upsampling**, for the VAE decoder. Not one op in the
  entire language-model path needs any of them;
- **`.safetensors` reading**. The VAE is not a GGUF;
- **PNG encoding**, since nothing here has ever written an image.

## The container proves the point

`tools/gguf-always-read.py` on `ideogram4-Q4_0.gguf` reports **458 tensors and
zero metadata keys**. There is no `general.architecture` to dispatch on and no
tokenizer inside — it is a bag of weights addressed to another program, not a
model container in the sense the rest of the catalogue means. Dense, so all
5.64 GB is read on *every one* of the sampler's steps.

## If it is wanted, the two honest routes

1. **Link `stable-diffusion.cpp`** the way ggml is linked: prebuilt, borrowed
   for its arithmetic, with Chaos owning the download, the residency and the
   command line. Smallest path to a working image, and it is the same ggml
   underneath. Costs the project its "one engine, ours" story.
2. **Write the pipeline** — DiT forward pass, sampler, VAE decoder, PNG writer.
   Consistent with how everything else here was done, and larger than the
   Qwen3.5 port by some way.

Either is a project, not a task. Neither is started, and nothing in the app
pretends otherwise.

## What the app says meanwhile

Ideogram 4 appears under AVAILABLE with its real size and this refusal: *"it is
an image model -- a diffusion transformer needing a sampler, a separate text
encoder and a VAE, none of which Chaos has"*.
