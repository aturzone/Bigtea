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

## The ceiling was memory, and planning the graph removed it

**1024x1024 works now.** It did not before, and the reason was not the weights.

A `Context`'s arena allocates every tensor in a graph and frees none of them, so
a graph pays for every intermediate it ever writes even though the live set is a
handful. `ggml_gallocr` plans the graph and gives the same buffer to tensors
whose lifetimes do not overlap — which is what the *device* path here has always
done, and what the host path never did.

| | before | after | measured |
|---|---|---|---|
| decode 256x256 | 3.69 GiB | **0.20 GiB** | 18.2x, bit-identical |
| decode 1024x1024 | 52 GiB | **3.4 GiB** | |
| denoise 1024x1024, per layer | 14.6 GiB | **2.0 GiB** | 2.0 GiB resident |

**Bit-identical, not merely close**: 0 of 196,608 pixels differ on a real
photograph, and the denoiser's velocity score is unchanged to four decimals
(0.7920 / 0.7483 / 0.6067 / 0.4723). Reuse that changed an answer would be
aliasing, and an image is exactly where nobody would notice.

Two things bit on the way, both worth keeping:

- **A `no_alloc` context cannot hold a weight.** Copying bytes into a tensor with
  no storage is a segmentation fault. The autoencoder now puts weights in an
  ordinary context and the graph in a planned one; the denoiser already bound its
  weights zero-copy, so it needed no such split.
- **The plan owns the buffer the answer lives in.** Returning the output tensor
  and reading it after the `GraphAllocator` drops is a use-after-free. The helper
  reads before returning.

## At 1024x1024 it is photorealistic, and that was the whole of it

The first 1024x1024 generation — impossible before the graphs were planned —
produced a **photorealistic human face**: skin texture, individual hairs,
eyebrows, catchlights, even an attempt at rendered text. At 512 the same code
produced flat facets. **The model is trained at 1024 and up and is not usable
below it**, exactly as `stable-diffusion.cpp`'s issue #1648 reports.

## Three things that were not bugs

The face came out upside down, which looked like an orientation bug. It was not:

- **The autoencoder does not flip.** White in the top-left quadrant encodes to a
  latent whose top-left quadrant is distinct, and decodes back to top-left at
  brightness 253. `examples/try-orientation.rs` — a round trip could never have
  shown this, because an encoder that flips and a decoder that flips back cancel
  exactly.
- **The denoiser does not flip either.** Decoding its estimate of the clean image
  from a latent at 50% noise gives back the input photograph, recognisable and
  the right way up. `CHAOS_X0_PNG=out.png` on `try-velocity`. This is also the
  strongest evidence the denoiser is right: from half noise it recovers a real
  image.
- **The thirteen layers are the right thirteen.** Reading `v` as the block index
  rather than `v - 1` asks for block 36 of a 36-block model, and fails saying so.

An unsteered model simply drew an inverted face. Which leads to the last one:

## The prompt has to be structured JSON

Ideogram 4 is conditioned on elaborate nested descriptions, and a phrase is far
outside that. Measured on one latent, two prompts as different from each other
as can be:

| prompt style | cos(A, B) | how much the prompt moves it |
|---|---|---|
| seven words | 0.9897 | 1.0% |
| structured JSON | 0.9667 | **3.3%** |

Three times the effect for the same pair of ideas, written the way the model
expects. **The conditioning path is not broken; it was being fed something the
model was never trained on.** `examples/try-conditioning.rs` holds both.

## What is still open

A full 1024x1024 generation from a structured JSON prompt has **not** been run —
it is roughly two hours on this machine, and every measurement above was made
without it. That is the demo, and it is the next thing to do.
