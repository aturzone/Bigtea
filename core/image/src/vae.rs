//! The FLUX.2 autoencoder: a photo in, a latent out, and pixels back again.
//!
//! # The one thing that makes this tractable
//!
//! **PyTorch's memory layout and ggml's are the same buffer read from opposite
//! ends.** A contiguous PyTorch conv weight is `[OC, IC, KH, KW]` with the last
//! dimension fastest; ggml describes the same bytes as `ne = [KW, KH, IC, OC]`
//! with `ne[0]` fastest. `ggml_conv_2d_direct` wants exactly `[KW, KH, IC, OC]`.
//!
//! So **no transpose is needed anywhere** — the safetensors bytes bind directly.
//! The same holds for the data: PyTorch `[N, C, H, W]` is ggml `[W, H, C, N]`,
//! which is what the convolution expects. Getting this wrong is the classic way
//! to produce a picture of confident nonsense, and it is worth stating before any
//! code rather than discovering at the end.
//!
//! # Why the encoder is here, when only the decoder is needed to make an image
//!
//! A diffusion pipeline never runs the encoder. It is written anyway because it
//! is the only way to check the decoder **without a reference implementation**:
//! encode a real photograph, decode the latent, and compare against the input.
//!
//! The two halves are separately trained sets of weights over a shared latent
//! space, so a bug in either one wrecks the reconstruction and neither can
//! quietly compensate for the other. A transposed kernel, a group norm without
//! its scale, the mid-block attention contracted over the wrong axis, or the
//! downsampler's off-centre padding done symmetrically — each of those turns a
//! good reconstruction into a bad one. **That is the point.** "It looks like a
//! picture" is not evidence; a subtly wrong pipeline still produces a plausible
//! picture, and that is this project's oldest hazard.
//!
//! # The architecture, read from the file rather than assumed
//!
//! ```text
//! encoder                                   decoder
//!   conv_in     3x3    3 -> 128               post_quant_conv 1x1  32 -> 32
//!   down 0: 2 x resnet(128),   /2             conv_in         3x3  32 -> 512
//!   down 1: resnet(128->256),  /2             mid: resnet, attention, resnet
//!           resnet(256)                       up 0: 3 x resnet(512), x2
//!   down 2: resnet(256->512),  /2             up 1: 3 x resnet(512), x2
//!           resnet(512)                       up 2: resnet(512->256), x2
//!   down 3: 2 x resnet(512)                         2 x resnet(256)
//!   mid: resnet, attention, resnet            up 3: resnet(256->128)
//!   norm(512), silu, conv_out 512 -> 64             2 x resnet(128)
//!   quant_conv  1x1   64 -> 64                norm(128), silu, conv_out -> 3
//! ```
//!
//! Three down/upsamplers is **8x**, so a 1024x1024 image has a 128x128 latent —
//! which is what the reference command line's `-H 1024 -W 1024` implies.
//!
//! The encoder emits **64** channels, not 32: `DiagonalGaussianDistribution`
//! splits them into a 32-channel mean and a 32-channel log-variance. The mean is
//! the latent a deterministic round-trip wants, and [`Vae::latent_mean`] takes
//! it. Sampling the distribution would add noise that has nothing to do with
//! whether this port is right.
//!
//! A `ResnetBlock2D` is `norm1, silu, conv1, norm2, silu, conv2`, plus a 1x1
//! `conv_shortcut` on the residual when the channel count changes. **`norm1`
//! normalises the input channel count and `norm2` the output** — read from the
//! file: `decoder.up_blocks.2.resnets.0` has `norm1 [512]` and `norm2 [256]`.
//!
//! # Two traps that produce a finite, correctly-shaped, wrong answer
//!
//! **`ggml_group_norm` normalises and stops.** Diffusers then scales and shifts
//! per channel, so [`Vae::group_norm`] multiplies by the weight and adds the
//! bias itself, broadcast along `ne[2]`. Leaving that out costs nothing visible
//! until the reconstruction is measured.
//!
//! **The encoder's downsampler pads on one side.** Diffusers' `Downsample2D`
//! applies `F.pad(x, (0, 1, 0, 1))` — right and bottom only — and *then*
//! convolves with stride 2 and no padding. A symmetric `pad = 1` gives an output
//! of the same shape, shifted half a pixel at every one of the three levels.
//! [`Vae::downsample`] uses [`chaos_ggml::Context::pad`], which is asymmetric
//! for exactly this reason.

use crate::safetensors::{Dtype, SafeTensors};
use chaos_ggml::{Context, Tensor};

/// Groups `AutoencoderKL` normalises over. Diffusers' default, and the file
/// carries no configuration to say otherwise.
const GROUPS: i32 = 32;

/// Epsilon in every group norm here, matching diffusers' `AutoencoderKL`.
const EPS: f32 = 1e-6;

/// The factor three down/upsamplers make, so an image is this much wider than
/// its latent in each direction.
pub const SCALE: usize = 8;

/// Channels in the latent. The encoder emits twice this — mean and log-variance.
pub const LATENT_CHANNELS: i64 = 32;

/// What went wrong, named specifically enough to fix.
#[derive(Debug)]
pub enum Error {
    /// A tensor the autoencoder needs is absent.
    Missing(String),
    /// A tensor is present with the wrong dtype.
    NotF32 { name: String, dtype: Dtype },
    /// The file does not hold this tensor's bytes.
    NoData(String),
    /// ggml refused.
    Ggml(chaos_ggml::GgmlError),
    /// The latent's shape is not usable.
    BadLatent(String),
    /// The image's shape is not usable.
    BadImage(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Missing(n) => write!(f, "the autoencoder has no tensor {n:?}"),
            Error::NotF32 { name, dtype } => {
                write!(f, "{name} is {dtype:?}; this autoencoder reads F32")
            }
            Error::NoData(n) => write!(f, "{n} has no bytes in this file -- a partial download?"),
            Error::Ggml(e) => write!(f, "ggml: {e:?}"),
            Error::BadLatent(m) => write!(f, "bad latent: {m}"),
            Error::BadImage(m) => write!(f, "bad image: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<chaos_ggml::GgmlError> for Error {
    fn from(e: chaos_ggml::GgmlError) -> Self {
        Error::Ggml(e)
    }
}

/// The autoencoder's weights, bound into a ggml context as the graph asks for
/// them.
///
/// Binding is lazy on purpose: building only the decoder's graph never touches
/// an encoder tensor, so a decode-only run pays for 138 tensors rather than 251.
pub struct Vae<'a> {
    st: &'a SafeTensors,
    file: &'a [u8],
    ctx: &'a Context,
    /// Where weights are allocated, which is usually `ctx` itself.
    ///
    /// **Separate so the graph can be planned.** `ggml_gallocr` needs a
    /// `no_alloc` context to place its tensors in, and a `no_alloc` context
    /// cannot hold a weight — copying bytes into a tensor with no storage is a
    /// segmentation fault, which is how this field came to exist. Weights go in
    /// an ordinary context, the graph in a planned one, and ggml is happy to
    /// walk a graph whose tensors come from both.
    wctx: &'a Context,
}

/// A group norm's two per-channel vectors.
struct GroupNorm<'a> {
    weight: Tensor<'a>,
    bias: Tensor<'a>,
}

/// A convolution's kernel and bias.
struct Conv<'a> {
    weight: Tensor<'a>,
    bias: Tensor<'a>,
}

impl<'a> Vae<'a> {
    /// Bind against an already-parsed file. Nothing is read until a graph is
    /// built.
    pub fn new(st: &'a SafeTensors, file: &'a [u8], ctx: &'a Context) -> Self {
        Vae {
            st,
            file,
            ctx,
            wctx: ctx,
        }
    }

    /// Build the graph in `ctx` and put the weights in `wctx`.
    ///
    /// For [`decode_planned`], where `ctx` is `no_alloc` and could not hold one.
    pub fn split(st: &'a SafeTensors, file: &'a [u8], ctx: &'a Context, wctx: &'a Context) -> Self {
        Vae {
            st,
            file,
            ctx,
            wctx,
        }
    }

    /// One tensor, with its dimensions **reversed, not transposed**.
    ///
    /// See the module docs: the bytes are already in the order ggml wants, and
    /// only the description of them differs.
    fn bind(&self, name: &str) -> Result<Tensor<'a>, Error> {
        let e = self
            .st
            .get(name)
            .ok_or_else(|| Error::Missing(name.to_string()))?;
        if e.dtype != Dtype::F32 {
            return Err(Error::NotF32 {
                name: name.to_string(),
                dtype: e.dtype,
            });
        }
        let bytes = self
            .st
            .bytes_of(self.file, e)
            .ok_or_else(|| Error::NoData(name.to_string()))?;
        let mut ne = [1i64; 4];
        for (i, d) in e.shape.iter().rev().enumerate() {
            ne[i] = *d as i64;
        }
        let t = self.wctx.new_f32_4d(ne[0], ne[1], ne[2], ne[3])?;
        t.set_bytes(bytes)?;
        Ok(t)
    }

    fn conv(&self, prefix: &str) -> Result<Conv<'a>, Error> {
        Ok(Conv {
            weight: self.bind(&format!("{prefix}.weight"))?,
            bias: self.bind(&format!("{prefix}.bias"))?,
        })
    }

    fn norm(&self, prefix: &str) -> Result<GroupNorm<'a>, Error> {
        Ok(GroupNorm {
            weight: self.bind(&format!("{prefix}.weight"))?,
            bias: self.bind(&format!("{prefix}.bias"))?,
        })
    }

    /// Group norm, then the per-channel scale and shift ggml's op leaves out.
    fn group_norm(&self, x: &Tensor<'a>, n: &GroupNorm<'a>) -> Result<Tensor<'a>, Error> {
        let h = self.ctx.group_norm(x, GROUPS, EPS)?;
        // The [C] vectors become [1, 1, C, 1] so they broadcast along channels.
        let c = n.weight.ne()[0];
        let w = self.ctx.reshape_4d(&n.weight, [1, 1, c, 1])?;
        let b = self.ctx.reshape_4d(&n.bias, [1, 1, c, 1])?;
        let h = self.ctx.mul(&h, &w)?;
        Ok(self.ctx.add(&h, &b)?)
    }

    /// A 3x3 convolution with `padding = 1`, which keeps the spatial size.
    fn conv3(&self, x: &Tensor<'a>, c: &Conv<'a>) -> Result<Tensor<'a>, Error> {
        let h = self
            .ctx
            .conv_2d_direct(&c.weight, x, (1, 1), (1, 1), (1, 1))?;
        self.add_bias(&h, &c.bias)
    }

    /// A 1x1 convolution: a per-pixel channel mix, no padding.
    fn conv1(&self, x: &Tensor<'a>, c: &Conv<'a>) -> Result<Tensor<'a>, Error> {
        let h = self
            .ctx
            .conv_2d_direct(&c.weight, x, (1, 1), (0, 0), (1, 1))?;
        self.add_bias(&h, &c.bias)
    }

    /// Add a `[C]` bias to a `[W, H, C, N]` activation.
    fn add_bias(&self, x: &Tensor<'a>, bias: &Tensor<'a>) -> Result<Tensor<'a>, Error> {
        let c = bias.ne()[0];
        let b = self.ctx.reshape_4d(bias, [1, 1, c, 1])?;
        Ok(self.ctx.add(x, &b)?)
    }

    /// `norm1, silu, conv1, norm2, silu, conv2`, plus the residual.
    fn resnet(&self, x: &Tensor<'a>, prefix: &str, shortcut: bool) -> Result<Tensor<'a>, Error> {
        let n1 = self.norm(&format!("{prefix}.norm1"))?;
        let n2 = self.norm(&format!("{prefix}.norm2"))?;
        let c1 = self.conv(&format!("{prefix}.conv1"))?;
        let c2 = self.conv(&format!("{prefix}.conv2"))?;

        let h = self.group_norm(x, &n1)?;
        let h = self.ctx.silu(&h)?;
        let h = self.conv3(&h, &c1)?;
        let h = self.group_norm(&h, &n2)?;
        let h = self.ctx.silu(&h)?;
        let h = self.conv3(&h, &c2)?;

        // The residual takes the 1x1 shortcut only when the channels change.
        let skip = if shortcut {
            let cs = self.conv(&format!("{prefix}.conv_shortcut"))?;
            self.conv1(x, &cs)?
        } else {
            *x
        };
        Ok(self.ctx.add(&skip, &h)?)
    }

    /// Nearest-neighbour 2x, then the block's own 3x3 convolution.
    fn upsample(&self, x: &Tensor<'a>, prefix: &str) -> Result<Tensor<'a>, Error> {
        let up = self.ctx.upscale_nearest(x, 2)?;
        let c = self.conv(&format!("{prefix}.upsamplers.0.conv"))?;
        self.conv3(&up, &c)
    }

    /// Zero-pad right and bottom, then a stride-2 3x3 convolution with **no**
    /// padding of its own.
    ///
    /// The asymmetry is diffusers', not a convenience: see the module docs.
    fn downsample(&self, x: &Tensor<'a>, prefix: &str) -> Result<Tensor<'a>, Error> {
        let padded = self.ctx.pad(x, [1, 1, 0, 0])?;
        let c = self.conv(&format!("{prefix}.downsamplers.0.conv"))?;
        let h = self
            .ctx
            .conv_2d_direct(&c.weight, &padded, (2, 2), (0, 0), (1, 1))?;
        self.add_bias(&h, &c.bias)
    }

    /// Self-attention over spatial positions, at 512 channels and one head.
    ///
    /// `AutoencoderKL`'s mid-block attention is single-head over `h*w`
    /// positions: the `to_*` weights are `[512, 512]`, with no head dimension to
    /// split. The residual is taken **before** the group norm, which is where
    /// diffusers' `AttnProcessor` takes it.
    fn mid_attention(&self, x: &Tensor<'a>, half: &str) -> Result<Tensor<'a>, Error> {
        let p = format!("{half}.mid_block.attentions.0");
        let n = self.norm(&format!("{p}.group_norm"))?;
        let h = self.group_norm(x, &n)?;

        let ne = h.ne();
        let (w, hh, c) = (ne[0], ne[1], ne[2]);
        let positions = w * hh;
        // [W, H, C, 1] is already [positions, C] when read as 2-D, because w is
        // fastest and c is slowest. Transposing gives [C, positions], one row of
        // channels per position, which is what a [C, C] matmul consumes.
        let flat = self.ctx.reshape_2d(&h, positions, c)?;
        let flat = self
            .ctx
            .cont_2d(&self.ctx.transpose(&flat)?, [c, positions])?;

        let q = self.linear(&flat, &format!("{p}.to_q"))?;
        let k = self.linear(&flat, &format!("{p}.to_k"))?;
        let v = self.linear(&flat, &format!("{p}.to_v"))?;

        // mul_mat contracts ne[0], so this is scores[j, i] = k_j . q_i, and
        // soft_max normalises over ne[0] = j, which is the key axis. One head
        // means dim_head is the full channel count, so the scale is 1/sqrt(C).
        let scores = self.ctx.mul_mat(&k, &q)?;
        let scaled = self.ctx.scale(&scores, 1.0 / (c as f32).sqrt())?;
        let probs = self.ctx.soft_max(&scaled)?;
        // v is [C, positions]; transposing makes the contraction run over
        // positions, giving out[c, i] = sum_j v[c, j] * probs[j, i].
        let vt = self.ctx.cont_2d(&self.ctx.transpose(&v)?, [positions, c])?;
        let out = self.ctx.mul_mat(&vt, &probs)?;

        let out = self.linear(&out, &format!("{p}.to_out.0"))?;
        // Back to [W, H, C, 1] and add the residual.
        let out = self
            .ctx
            .cont_2d(&self.ctx.transpose(&out)?, [positions, c])?;
        let out = self.ctx.reshape_4d(&out, [w, hh, c, 1])?;
        Ok(self.ctx.add(x, &out)?)
    }

    /// `y = W x + b` for a `[C, C]` weight over `[C, positions]`.
    fn linear(&self, x: &Tensor<'a>, prefix: &str) -> Result<Tensor<'a>, Error> {
        let w = self.bind(&format!("{prefix}.weight"))?;
        let b = self.bind(&format!("{prefix}.bias"))?;
        let y = self.ctx.mul_mat(&w, x)?;
        let n = b.ne()[0];
        let bb = self.ctx.reshape_2d(&b, n, 1)?;
        Ok(self.ctx.add(&y, &bb)?)
    }

    /// `resnet, attention, resnet` — the same block in both halves.
    fn mid_block(&self, x: &Tensor<'a>, half: &str) -> Result<Tensor<'a>, Error> {
        let h = self.resnet(x, &format!("{half}.mid_block.resnets.0"), false)?;
        let h = self.mid_attention(&h, half)?;
        self.resnet(&h, &format!("{half}.mid_block.resnets.1"), false)
    }

    /// Build the graph from a `[W, H, 3, 1]` image to `[W/8, H/8, 64, 1]`
    /// moments — the mean and log-variance the latent is drawn from.
    pub fn encode(&self, image: &Tensor<'a>) -> Result<Tensor<'a>, Error> {
        let ne = image.ne();
        if ne[2] != 3 {
            return Err(Error::BadImage(format!(
                "expected 3 channels, got {}",
                ne[2]
            )));
        }
        let s = SCALE as i64;
        if ne[0] % s != 0 || ne[1] % s != 0 {
            return Err(Error::BadImage(format!(
                "{}x{} is not a multiple of {s}; three stride-2 convolutions cannot halve it evenly",
                ne[0], ne[1]
            )));
        }

        let cin = self.conv("encoder.conv_in")?;
        let mut h = self.conv3(image, &cin)?;

        for b in 0..4 {
            for r in 0..2 {
                // Only the first resnet of a block that changes channels has one.
                let shortcut = r == 0 && (b == 1 || b == 2);
                h = self.resnet(
                    &h,
                    &format!("encoder.down_blocks.{b}.resnets.{r}"),
                    shortcut,
                )?;
            }
            if b < 3 {
                h = self.downsample(&h, &format!("encoder.down_blocks.{b}"))?;
            }
        }

        h = self.mid_block(&h, "encoder")?;

        let non = self.norm("encoder.conv_norm_out")?;
        let h = self.group_norm(&h, &non)?;
        let h = self.ctx.silu(&h)?;
        let cout = self.conv("encoder.conv_out")?;
        let h = self.conv3(&h, &cout)?;
        let qc = self.conv("quant_conv")?;
        self.conv1(&h, &qc)
    }

    /// The first 32 of the 64 encoded channels: the distribution's mean.
    ///
    /// A pipeline samples `mean + std * noise`; a round-trip wants the mean, so
    /// that a difference in the output is this port's fault and not the noise's.
    pub fn latent_mean(&self, moments: &Tensor<'a>) -> Result<Tensor<'a>, Error> {
        let ne = moments.ne();
        if ne[2] != 2 * LATENT_CHANNELS {
            return Err(Error::BadLatent(format!(
                "expected {} channels of moments, got {}",
                2 * LATENT_CHANNELS,
                ne[2]
            )));
        }
        // Channels are ne[2] and the buffer is contiguous, so the mean is the
        // leading half at offset zero -- no copy, just a shorter description.
        let (_, nb) = moments.dims_and_strides();
        let view = self.ctx.view_4d(
            moments,
            [ne[0], ne[1], LATENT_CHANNELS, 1],
            [nb[1], nb[2], nb[3]],
            0,
        )?;
        Ok(self
            .ctx
            .cont_4d(&view, [ne[0], ne[1], LATENT_CHANNELS, 1])?)
    }

    /// Build the graph from a `[w, h, 32, 1]` latent to an `[8w, 8h, 3, 1]`
    /// image.
    pub fn decode(&self, latent: &Tensor<'a>) -> Result<Tensor<'a>, Error> {
        if latent.ne()[2] != LATENT_CHANNELS {
            return Err(Error::BadLatent(format!(
                "expected {LATENT_CHANNELS} channels, got {}",
                latent.ne()[2]
            )));
        }
        let pqc = self.conv("post_quant_conv")?;
        let h = self.conv1(latent, &pqc)?;
        let cin = self.conv("decoder.conv_in")?;
        let mut h = self.conv3(&h, &cin)?;

        h = self.mid_block(&h, "decoder")?;

        for b in 0..4 {
            for r in 0..3 {
                let shortcut = r == 0 && (b == 2 || b == 3);
                h = self.resnet(&h, &format!("decoder.up_blocks.{b}.resnets.{r}"), shortcut)?;
            }
            if b < 3 {
                h = self.upsample(&h, &format!("decoder.up_blocks.{b}"))?;
            }
        }

        let non = self.norm("decoder.conv_norm_out")?;
        let h = self.group_norm(&h, &non)?;
        let h = self.ctx.silu(&h)?;
        let cout = self.conv("decoder.conv_out")?;
        self.conv3(&h, &cout)
    }
}

/// Every tensor the decoder needs.
///
/// Separate from building the graph so that "is this file complete and the right
/// shape?" can be answered without allocating an arena big enough to decode
/// anything — which is worth answering on its own, and is what a partial
/// download looks like.
pub fn decoder_tensors() -> Vec<String> {
    let mut v = vec![
        "post_quant_conv.weight".to_string(),
        "post_quant_conv.bias".to_string(),
        "decoder.conv_in.weight".to_string(),
        "decoder.conv_in.bias".to_string(),
    ];
    push_mid(&mut v, "decoder");
    for b in 0..4 {
        for r in 0..3 {
            push_resnet(&mut v, &format!("decoder.up_blocks.{b}.resnets.{r}"));
        }
        if b < 3 {
            v.push(format!("decoder.up_blocks.{b}.upsamplers.0.conv.weight"));
            v.push(format!("decoder.up_blocks.{b}.upsamplers.0.conv.bias"));
        }
    }
    // Only where the channel count changes, which the file says is blocks 2 and
    // 3 -- listing shortcuts for 0 and 1 would report a complete file as broken.
    for b in [2, 3] {
        v.push(format!(
            "decoder.up_blocks.{b}.resnets.0.conv_shortcut.weight"
        ));
        v.push(format!(
            "decoder.up_blocks.{b}.resnets.0.conv_shortcut.bias"
        ));
    }
    v.push("decoder.conv_norm_out.weight".into());
    v.push("decoder.conv_norm_out.bias".into());
    v.push("decoder.conv_out.weight".into());
    v.push("decoder.conv_out.bias".into());
    v
}

/// Every tensor the encoder needs.
///
/// Two resnets per block, not three, and the shortcuts fall on blocks 1 and 2 —
/// both differences from the decoder, and both read from the file.
pub fn encoder_tensors() -> Vec<String> {
    let mut v = vec![
        "quant_conv.weight".to_string(),
        "quant_conv.bias".to_string(),
        "encoder.conv_in.weight".to_string(),
        "encoder.conv_in.bias".to_string(),
    ];
    push_mid(&mut v, "encoder");
    for b in 0..4 {
        for r in 0..2 {
            push_resnet(&mut v, &format!("encoder.down_blocks.{b}.resnets.{r}"));
        }
        if b < 3 {
            v.push(format!(
                "encoder.down_blocks.{b}.downsamplers.0.conv.weight"
            ));
            v.push(format!("encoder.down_blocks.{b}.downsamplers.0.conv.bias"));
        }
    }
    for b in [1, 2] {
        v.push(format!(
            "encoder.down_blocks.{b}.resnets.0.conv_shortcut.weight"
        ));
        v.push(format!(
            "encoder.down_blocks.{b}.resnets.0.conv_shortcut.bias"
        ));
    }
    v.push("encoder.conv_norm_out.weight".into());
    v.push("encoder.conv_norm_out.bias".into());
    v.push("encoder.conv_out.weight".into());
    v.push("encoder.conv_out.bias".into());
    v
}

fn push_resnet(v: &mut Vec<String>, prefix: &str) {
    for part in ["norm1", "norm2", "conv1", "conv2"] {
        v.push(format!("{prefix}.{part}.weight"));
        v.push(format!("{prefix}.{part}.bias"));
    }
}

fn push_mid(v: &mut Vec<String>, half: &str) {
    for r in 0..2 {
        push_resnet(v, &format!("{half}.mid_block.resnets.{r}"));
    }
    for part in ["group_norm", "to_q", "to_k", "to_v"] {
        v.push(format!("{half}.mid_block.attentions.0.{part}.weight"));
        v.push(format!("{half}.mid_block.attentions.0.{part}.bias"));
    }
    v.push(format!("{half}.mid_block.attentions.0.to_out.0.weight"));
    v.push(format!("{half}.mid_block.attentions.0.to_out.0.bias"));
}

/// Bytes of ggml arena a whole-graph decode needs for a `w * h` **output**.
///
/// # Why this is the ceiling on image size, and the weights are not
///
/// The denoisers stream a layer at a time, so their 5.26 GiB never has to fit.
/// The decoder is the opposite: it is one graph, ggml's context has no
/// liveness-based reuse, and its last two blocks work at **full output
/// resolution**. `up_blocks.3` alone holds three resnets of roughly eighteen
/// 128-channel planes each.
///
/// Measured rather than derived: 256x256 completes inside 3.6 GiB, and 768x768
/// exhausted a 29.5 GiB pool — 50 KiB per output pixel, and it scales with the
/// pixel count, not with the model.
///
/// | output | arena |
/// |---|---|
/// | 256x256 | 3.7 GiB |
/// | 512x512 | 13.1 GiB |
/// | 768x768 | 29.5 GiB — **too large for a 15.7 GiB machine** |
///
/// **The fix is a graph allocator, not a bigger machine.** Almost every tensor
/// here is dead a step after it is written, so allocating with reuse would cut
/// this by most of an order of magnitude. Until then this function exists so a
/// caller can refuse *before* spending an hour denoising a latent it will not be
/// able to decode — which is exactly how it was found.
pub fn decode_arena_bytes(w: usize, h: usize) -> usize {
    (512 << 20) + w * h * 51 * 1024
}

/// Bytes a **planned** decode needs for a `w * h` output.
///
/// Measured: a 256x256 decode plans to 0.20 GiB where the unplanned graph wanted
/// 3.69 GiB, and the plan scales with the pixel count like the graph does. About
/// 3.3 KiB per output pixel against 51, so the sizes that were impossible are
/// now ordinary:
///
/// | output | unplanned | planned |
/// |---|---|---|
/// | 256x256 | 3.7 GiB | 0.20 GiB |
/// | 512x512 | 13.1 GiB | 0.79 GiB |
/// | 1024x1024 | 52 GiB | 3.2 GiB |
pub fn decode_planned_bytes(w: usize, h: usize) -> usize {
    (64 << 20) + w * h * 3400
}

/// Decode a latent to pixels with the graph **planned**, so buffers are reused.
///
/// # Why this exists, and why it is the only way large images work
///
/// A `Context`'s arena allocates every tensor in a graph and frees none of them.
/// The decoder's last block works at full output resolution, so a whole-graph
/// decode asks for **51 KiB per output pixel** — 29.5 GiB at 768x768, which
/// aborted the process after an hour of denoising.
///
/// Almost every one of those tensors is dead one step after it is written.
/// `ggml_gallocr` plans the graph and hands the same buffer to tensors whose
/// lifetimes do not overlap, which is what the device path has always done.
/// Measured on a 40-step chain it was **81x smaller and bit-identical**.
///
/// The ordering is the part that bites: **the input is written after `alloc`,
/// never before**, because it has no storage until then. Weights are untouched —
/// `ggml_gallocr` skips any tensor that already has a data pointer, which is
/// exactly what binding zero-copy leaves behind.
///
/// Returns the `[W, H, 3, 1]` pixels and the bytes the plan needed.
pub fn decode_planned(
    st: &SafeTensors,
    file: &[u8],
    latent: &[f32],
    w: i64,
    h: i64,
    threads: usize,
) -> Result<(Vec<f32>, usize), Error> {
    // Only tensor structs, the graph and the compute work buffer come from here;
    // tensor data comes from the plan.
    // The weights need real storage; the graph must not have any of its own.
    let wctx = Context::new(768 << 20)?;
    let ctx = Context::new_no_alloc(256 << 20)?;
    let v = Vae::split(st, file, &ctx, &wctx);
    let z = ctx.new_f32_4d(w, h, LATENT_CHANNELS, 1)?;
    let out = v.decode(&z)?;

    let galloc = chaos_ggml::GraphAllocator::for_cpu()?;
    galloc.reserve(&ctx, &[&out])?;
    galloc.alloc(&ctx, &[&out])?;

    // After `alloc`. See above.
    z.set_f32(latent)?;
    ctx.compute(&out, threads)?;
    Ok((out.to_vec_f32(), galloc.buffer_bytes()))
}

/// The latent normalisation the file carries beside the autoencoder.
///
/// FLUX.2 keeps a BatchNorm's running statistics — `bn.running_mean` and
/// `bn.running_var` — and they are **128-wide, the packed channel count**, not
/// the autoencoder's 32. They are what older autoencoders did with a scalar
/// `scaling_factor`: they put the latent in the space the diffusion model was
/// trained on.
///
/// Returns `(mean, variance)`, or `None` if the file has no such tensors.
pub fn latent_stats(st: &SafeTensors, file: &[u8]) -> Option<(Vec<f32>, Vec<f32>)> {
    let read = |name: &str| -> Option<Vec<f32>> {
        let e = st.get(name)?;
        if e.dtype != Dtype::F32 {
            return None;
        }
        let b = st.bytes_of(file, e)?;
        Some(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    };
    Some((read("bn.running_mean")?, read("bn.running_var")?))
}

/// Epsilon in the latent normalisation. PyTorch's `BatchNorm2d` default.
pub const BN_EPS: f32 = 1e-5;

/// `(z - mean) / sqrt(var + eps)`, per packed channel, in place.
///
/// `packed` is `[gw, gh, 128]` in ggml order, so a channel is a contiguous
/// plane of `gw * gh` values.
pub fn normalize_latent(packed: &mut [f32], mean: &[f32], var: &[f32]) {
    let plane = packed.len() / mean.len();
    for (c, chunk) in packed.chunks_mut(plane).enumerate() {
        let s = 1.0 / (var[c] + BN_EPS).sqrt();
        for v in chunk {
            *v = (*v - mean[c]) * s;
        }
    }
}

/// [`normalize_latent`] reversed, for the latent the sampler hands back.
pub fn denormalize_latent(packed: &mut [f32], mean: &[f32], var: &[f32]) {
    let plane = packed.len() / mean.len();
    for (c, chunk) in packed.chunks_mut(plane).enumerate() {
        let s = (var[c] + BN_EPS).sqrt();
        for v in chunk {
            *v = *v * s + mean[c];
        }
    }
}

/// Fold each 2x2 block of a 32-channel latent into one cell of 128 channels.
///
/// **The denoiser never sees the autoencoder's latent directly.** FLUX.2's
/// autoencoder is 8x, but Ideogram 4 consumes a 16x grid: the pipeline packs
/// every 2x2 patch of latent into the channel dimension first. That is why the
/// denoiser's `input_proj` is 128 wide and the autoencoder's `conv_in` is 32.
///
/// The channel numbering is `px + 2*py + 4*ae` — the patch offsets in the fast
/// bits and the latent channel in the slow ones. Every other arrangement gives a
/// latent of exactly the right shape, so this was derived from the reference and
/// then **checked against the denoiser**, scoring its velocity on a real
/// photograph at sigma 0.6 and 0.2:
///
/// | channel order | cos(v) at 0.6 | at 0.2 |
/// |---|---|---|
/// | `px + 2*py + 4*ae` (this one) | **0.748** | **0.529** |
/// | patch axes swapped | 0.744 | 0.495 |
/// | latent channel fastest | 0.530 | 0.267 |
///
/// The swapped variant is only slightly worse, which is the point: a derivation
/// that lands on it would look right and cost a little of every image.
///
/// `[w, h, 32]` in, `[w/2, h/2, 128]` out, both in ggml order with width
/// fastest.
pub fn pack_latent(latent: &[f32], w: usize, h: usize, ae: usize, p: usize) -> Vec<f32> {
    let (gw, gh) = (w / p, h / p);
    let mut out = vec![0.0f32; gw * gh * ae * p * p];
    for c in 0..ae {
        for y in 0..h {
            for x in 0..w {
                let ch = (x % p) + p * (y % p) + p * p * c;
                let dst = (x / p) + gw * (y / p) + gw * gh * ch;
                out[dst] = latent[x + w * y + w * h * c];
            }
        }
    }
    out
}

/// [`pack_latent`] reversed: `[w, h, 128]` back to `[2w, 2h, 32]`.
pub fn unpack_latent(packed: &[f32], gw: usize, gh: usize, ae: usize, p: usize) -> Vec<f32> {
    let (w, h) = (gw * p, gh * p);
    let mut out = vec![0.0f32; w * h * ae];
    for c in 0..ae {
        for y in 0..h {
            for x in 0..w {
                let ch = (x % p) + p * (y % p) + p * p * c;
                let src = (x / p) + gw * (y / p) + gw * gh * ch;
                out[x + w * y + w * h * c] = packed[src];
            }
        }
    }
    out
}

/// Turn 8-bit RGB into the `[W, H, 3, 1]` planar float image the encoder wants.
///
/// Diffusers' `VaeImageProcessor` normalises to `[-1, 1]`, so this is
/// `2 * (p / 255) - 1` — the exact inverse of [`to_rgb8`].
pub fn from_rgb8(rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
    let plane = w * h;
    let mut out = vec![0.0f32; plane * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let p = rgb.get((y * w + x) * 3 + c).copied().unwrap_or(0);
                out[c * plane + y * w + x] = (p as f32 / 255.0) * 2.0 - 1.0;
            }
        }
    }
    out
}

/// Turn a decoder's `[W, H, 3, 1]` output into 8-bit RGB.
///
/// The decoder emits roughly `[-1, 1]`, so the mapping is `(x + 1) / 2`, and
/// values outside it are clamped rather than wrapped — a wrap turns a slightly
/// over-bright pixel black, which looks like a bug in the model rather than in
/// the conversion.
pub fn to_rgb8(pixels: &[f32], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    let plane = w * h;
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = pixels.get(c * plane + y * w + x).copied().unwrap_or(0.0);
                let v = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
                out[(y * w + x) * 3 + c] = (v * 255.0).round() as u8;
            }
        }
    }
    out
}

/// Peak signal-to-noise ratio in dB between two 8-bit images.
///
/// The round-trip's score. Identical images have no error, which is infinite dB
/// rather than a division by zero — a real autoencoder never reaches it, so a
/// literal `inf` in a report means the two buffers are the same buffer.
pub fn psnr(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f32::INFINITY;
    }
    (10.0 * (255.0f64 * 255.0 / mse).log10()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tensor lists name shortcuts only where channels change, and the two
    /// halves disagree about where that is.
    ///
    /// Listing them everywhere would report a complete file as broken; omitting
    /// them would let a genuinely incomplete file through. The file says decoder
    /// blocks 2 and 3, encoder blocks 1 and 2.
    #[test]
    fn the_required_lists_match_where_shortcuts_exist() {
        let d = decoder_tensors();
        for b in [2, 3] {
            assert!(
                d.contains(&format!(
                    "decoder.up_blocks.{b}.resnets.0.conv_shortcut.weight"
                )),
                "decoder block {b} changes channels and needs a shortcut"
            );
        }
        for b in [0, 1] {
            assert!(
                !d.iter()
                    .any(|n| n.contains(&format!("up_blocks.{b}.resnets.0.conv_shortcut"))),
                "decoder block {b} keeps its channel count and has no shortcut"
            );
        }

        let e = encoder_tensors();
        for b in [1, 2] {
            assert!(
                e.contains(&format!(
                    "encoder.down_blocks.{b}.resnets.0.conv_shortcut.weight"
                )),
                "encoder block {b} changes channels and needs a shortcut"
            );
        }
        for b in [0, 3] {
            assert!(
                !e.iter()
                    .any(|n| n.contains(&format!("down_blocks.{b}.resnets.0.conv_shortcut"))),
                "encoder block {b} keeps its channel count and has no shortcut"
            );
        }

        // Three resamplers each, not four: the last block does not resample.
        assert_eq!(
            d.iter().filter(|n| n.contains("upsamplers")).count(),
            6,
            "three upsamplers, weight and bias each"
        );
        assert_eq!(
            e.iter().filter(|n| n.contains("downsamplers")).count(),
            6,
            "three downsamplers, weight and bias each"
        );

        // Three resnets per decoder block against two per encoder block is the
        // asymmetry `layers_per_block + 1` produces, and it is worth asserting
        // because getting it wrong names tensors that do exist, just wrong ones.
        assert_eq!(d.iter().filter(|n| n.contains(".conv1.weight")).count(), 14);
        assert_eq!(e.iter().filter(|n| n.contains(".conv1.weight")).count(), 10);

        assert!(d.contains(&"post_quant_conv.weight".to_string()));
        assert!(e.contains(&"quant_conv.weight".to_string()));
    }

    /// `to_rgb8` maps the decoder's range and clamps outside it.
    #[test]
    fn rgb_conversion_maps_and_clamps() {
        // One pixel, three planes: -1 is black, 0 is mid, 1 is white.
        let px = to_rgb8(&[-1.0, 0.0, 1.0], 1, 1);
        assert_eq!(px, vec![0, 128, 255]);
        // Outside the range clamps rather than wrapping. A wrap would turn an
        // over-bright pixel black, which reads as a model failure.
        let px = to_rgb8(&[-9.0, 9.0, 0.0], 1, 1);
        assert_eq!(px, vec![0, 255, 128]);
        // Row-major, planes in channel order.
        let px = to_rgb8(&[1.0, -1.0, -1.0, 1.0, -1.0, -1.0], 2, 1);
        assert_eq!(px.len(), 6);
        assert_eq!(&px[..3], &[255, 0, 0], "first pixel is red-ish");
    }

    /// `from_rgb8` is `to_rgb8` backwards, which the round-trip relies on: a
    /// mismatch here would be charged to the model.
    #[test]
    fn rgb_conversion_round_trips_through_the_planar_form() {
        let rgb: Vec<u8> = (0..4 * 3 * 3).map(|i| (i * 7 % 256) as u8).collect();
        let planar = from_rgb8(&rgb, 4, 3);
        assert_eq!(planar.len(), 4 * 3 * 3);
        assert_eq!(to_rgb8(&planar, 4, 3), rgb, "8 -> f32 -> 8 must be exact");
        // And the planar layout really is planar: channel 0 of pixel (1, 0) is
        // the second element, not the fourth.
        assert!((planar[1] - ((rgb[3] as f32 / 255.0) * 2.0 - 1.0)).abs() < 1e-6);
    }

    /// Packing folds 2x2 blocks into channels and unpacking undoes it exactly.
    #[test]
    fn the_latent_pack_is_an_exact_permutation() {
        let (w, h, ae, p) = (4usize, 6usize, 3usize, 2usize);
        let latent: Vec<f32> = (0..w * h * ae).map(|i| i as f32).collect();
        let packed = pack_latent(&latent, w, h, ae, p);
        assert_eq!(
            packed.len(),
            latent.len(),
            "a fold moves values, not counts"
        );
        assert_eq!(unpack_latent(&packed, w / p, h / p, ae, p), latent);

        // The documented numbering: the top-left pixel of every 2x2 block keeps
        // channel 4c, and its right neighbour lands on 4c + 1.
        let (gw, gh) = (w / p, h / p);
        for c in 0..ae {
            let top_left = packed[gw * gh * (p * p * c)];
            assert_eq!(top_left, latent[w * h * c], "c = {c}");
            let right = packed[gw * gh * (1 + p * p * c)];
            assert_eq!(right, latent[1 + w * h * c], "px = 1 is channel 4c + 1");
            let below = packed[gw * gh * (p + p * p * c)];
            assert_eq!(below, latent[w + w * h * c], "py = 1 is channel 4c + 2");
        }
    }

    /// PSNR is a score, so its two ends need to mean what a report says they do.
    #[test]
    fn psnr_scores_identity_and_disagreement() {
        let a = vec![10u8, 200, 30, 40];
        assert!(
            psnr(&a, &a).is_infinite(),
            "identical buffers have no error"
        );
        // A uniform error of 1 level is 20*log10(255) = 48.13 dB.
        let b: Vec<u8> = a.iter().map(|v| v + 1).collect();
        assert!((psnr(&a, &b) - 48.13).abs() < 0.05, "{}", psnr(&a, &b));
        // Black against white is the worst case, 0 dB.
        assert!(psnr(&[0, 0, 0, 0], &[255, 255, 255, 255]).abs() < 1e-3);
        // Mismatched lengths are a caller error, not a score.
        assert_eq!(psnr(&a, &[1, 2]), 0.0);
    }
}
