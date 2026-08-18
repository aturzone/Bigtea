# Generating images — what Ideogram 4 would actually take

> Atur asked for Ideogram 4 in the release. It **is** open-weight, so "we cannot
> get it" was never the answer. This is the real answer.

**Status: not started, and it is a second engine rather than a feature.**

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
