//! Image generation for Chaos: the parts of a diffusion pipeline that are not a
//! token loop.
//!
//! # Why this is a separate crate
//!
//! Everything else here embeds a token, runs a stack of attention and feed-
//! forward blocks, samples the next token, and repeats. An image is not that
//! shape. It is a *latent* refined over twenty to fifty steps by a denoiser that
//! is conditioned on text and on the step number, and then decoded to pixels by
//! an autoencoder that is all convolutions. Putting that in `chaos-arch` would
//! mean one crate holding two unrelated engines.
//!
//! # What one image needs
//!
//! Read from `leejet/ideogram-4-GGUF`'s own reference command line, not guessed:
//!
//! | part | file | status |
//! |---|---|---|
//! | denoiser | `ideogram4-Q4_0.gguf` (5.26 GiB) | on disk, shape mapped |
//! | unconditional twin | `ideogram4_uncond-Q4_0.gguf` | not fetched |
//! | text encoder | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | not fetched |
//! | autoencoder | `ae.safetensors` (FLUX.2) | not fetched, and not a GGUF |
//!
//! The denoiser, read from the container: **34 layers, hidden 4608, 18 heads of
//! 256, fused QKV, SwiGLU at 12288, sandwich norms, adaptive layer norm with
//! four modulation signals from a 512-wide conditioning vector, 128 patch
//! channels in and out, and zero metadata keys** — so it is identified by its
//! tensor names, which is what `catalogue::architecture_from_tensors` does.
//!
//! # Order of work, and why
//!
//! [`png`] is first and is finished. It is the only piece with no dependency on
//! any of the other three files, so it can be written and *tested* today rather
//! than waiting for a pipeline to produce something to write. The same is true
//! of the safetensors reader that comes next.
//!
//! The engine itself is last, because until the text encoder and the
//! autoencoder are in hand there is nothing to check a denoising step against —
//! and a diffusion pipeline that is subtly wrong produces a plausible-looking
//! image, which is this project's oldest hazard wearing different clothes.

pub mod dit;
pub mod flow;
pub mod pipeline;
pub mod png;
pub mod rope3d;
pub mod safetensors;
pub mod text;
pub mod vae;
