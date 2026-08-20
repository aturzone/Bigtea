# Generating images — what Ideogram 4 would actually take

> Atur asked for Ideogram 4 in the release. It **is** open-weight, so "we cannot
> get it" was never the answer. This is the real answer.

**Status: the whole pipeline runs end to end and produces a coherent
photographic image, 2026-08-20. It does not yet follow the prompt closely.**

## Done, and how it was checked

| piece | state |
|---|---|
| PNG output | **done** — round-trip test, Windows GDI+ decoding exact pixels, visual render |
| safetensors reading | **done** — against the real 251-tensor VAE header |
| all four files fetchable | **done** — `chaos-pull` gets them, 16.65 GB total |
| **VAE encode + decode** | **done — 36.09 dB round trip on a real photograph** |
| **the denoiser forward pass** | **done — 34-layer DiT, scored 0.85 against a real latent** |
| **the sampler loop** | **done — logit-normal schedule, Euler, CFG** |
| **text conditioning** | **done — Qwen3-VL, verified by " Paris"** |
| prompt adherence | **weak, open** |

## How the denoiser is checked, since a picture cannot check it

A rectified-flow model at noise level `sigma` sees
`latent*(1-sigma) + noise*sigma` and must predict `noise - latent`. **Both terms
are known**: the autoencoder's encoder — already verified to 36 dB by round trip
— turns a real photograph into a real latent, and the noise is ours. So the
model's answer is scored by cosine similarity against the truth, with no image
involved. `examples/try-velocity.rs`.

Scored against each half separately, which is what made it diagnostic:

| | cos(v) | cos(-latent) | cos(noise) |
|---|---|---|---|
| first working version | 0.57 | 0.16 | 0.80 |
| + latent normalisation | 0.73 | 0.15 | 0.84 |
| + correct attention scale | 0.79 | 0.20 | 0.88 |
| + at 512x512 instead of 256 | **0.85** | 0.34 | 0.83 |

**`cos(noise)` high with `cos(-latent)` near zero** was the signature that found
both bugs: noise is permutation-invariant and image content is not, so a model
that reports the noise and misses the image is being fed something it cannot
read.

## Three things the reference implementation gets wrong or leaves out

Read from `leejet/stable-diffusion.cpp`, then corrected against measurement:

1. **`kv_scale` is not the attention scale.** Ideogram 4 passes `1.f / 128.f`,
   which reads exactly like a scale for head_dim 256. The helper multiplies k and
   v by it, divides the softmax scale by it, and divides the output back out — an
   F16 overflow guard that cancels exactly. The real scale is
   `1/sqrt(head_dim)`. Using 1/128 makes the softmax eight times too flat.
2. **The autoencoder's latent normalisation is never applied.**
   `bn.running_mean` and `bn.running_var` are in the file, 128 wide to match the
   packed channel count, and `stable-diffusion.cpp` reads neither. Applying them
   is worth 0.17 to 0.49 at sigma 0.3.
3. **`rope_interleaved = false`** means element `f` rotates with `f + head_dim/2`,
   not with its neighbour.

Being faithful to the reference and being right are not the same thing, and only
a measurement the reference does not make can tell them apart.

## What the prompt does, measured at three sizes

At **256x256** the pipeline drew a coherent dark stone-wall scene for "a red
apple on a white table" — a real photograph of the wrong thing. At **512x512**,
16 steps, cfg 5, the same prompt drew a **red mass on white surfaces beside a
wooden shelf**: colour and scene follow the prompt, the object's *form* does not.
So the conditioning works and the geometry is the weak part.

The numbers behind that: two prompts as different as "a red apple on a white
table" and "a snowy mountain range at sunrise" move the predicted velocity by 14%
(cosine 0.9897), where the *unconditional twin* — different weights entirely — is
0.89 away. Fourteen per cent sounds small and is not: the velocity is dominated
by "remove the noise", which is the same whatever the prompt, and guidance
multiplies the difference.

## What is still open

**Form.** The subject comes out as a flat saturated region rather than an object.
Untested candidates, in the order worth trying:

1. **Resolution.** Everything here has been measured below the 1024x1024 the
   model was trained for, and the same code scored 0.79 at 256 against 0.85 at
   512 on one photograph. `stable-diffusion.cpp`'s own issue #1648 reports
   Ideogram 4 collapsing at 1024 and being correct at 1536 and 2048, so this
   model is unusually resolution-sensitive.
2. **Guidance strength.** Flat saturated regions are what too much of it looks
   like on a flow model.
3. **Prompt shape.** Ideogram 4 is trained on elaborate structured-JSON prompts;
   a seven-word one may genuinely condition weakly.
4. The thirteen hidden-state layers, which are read from the reference and not
   independently checked.

## The autoencoder is verified, and here is what that word is doing

`crates/chaos-image/src/vae.rs` builds both halves as one ggml graph each. The
decoder is what a pipeline needs; **the encoder was written so that the decoder
could be checked without a reference implementation.**

    powershell -File scripts/image-to-ppm.ps1 -In photo.jpg -Out photo.ppm -Size 256
    cargo run --release -p chaos-image --example vae-roundtrip -- photo.ppm

Encode a photograph, take the distribution's mean, decode it, and score the
result against the input. The two halves are separately trained weights over one
shared latent space, so **neither can compensate for a bug in the other**.

**Measured, four different 256x256 photographs, one session:**

| image | PSNR |
|---|---|
| Spotlight/img14 | 36.09 dB |
| Spotlight/img50 | 36.29 dB |
| ThemeB/img25 | 36.49 dB |
| ThemeA/img22 | 40.89 dB |

**And the control, which is the part that makes those numbers mean something.**
Each of these is a plausible way to get the port wrong; each was introduced
deliberately and measured against the same 256x256 input, baseline 36.09 dB:

| deliberate error | PSNR |
|---|---|
| `group_norm` without its per-channel scale and shift | 16.77 dB |
| downsampler padded symmetrically instead of `(0,1,0,1)` | 14.60 dB |
| mid-block attention skipped entirely | 31.93 dB |
| convolution kernels not dimension-reversed | **ggml aborts** |

Three of the four survive as a picture. All three would have passed "it looks
like a photograph". The attention row is the narrow one — 4.16 dB — and it is
why the regression test's threshold is 35 rather than 25.

The suite carries the round trip at 128x128 as `#[ignore]`d tests in
`crates/chaos-image/tests/vae_roundtrip.rs`; they **panic rather than skip** when
the 336 MB file is absent.

## The three tensors that are not in either half

The file has 251 tensors and the two graphs name 248. The remainder is a
BatchNorm — `bn.running_mean`, `bn.running_var`, `bn.num_batches_tracked` —
holding the **latent normalisation**, which is what earlier VAEs did with a
scalar `scaling_factor`.

They are **128-wide, not 32**: that is the patchified channel count, 32 latent
channels times a 2x2 patch, which is exactly what the denoiser consumes. A round
trip never touches them, because encode and decode are inverses whatever the
normalisation is — which is what makes the round trip a fair test of this port
and *not* a test of the interface to the denoiser. **Leaving them out of the
denoiser is the next chance to produce a confident, plausible, wrong image.**

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
