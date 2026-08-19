//! The Ideogram 4 denoiser: a diffusion transformer, 34 layers of it.
//!
//! # What this is, next to the token loop the rest of Chaos runs
//!
//! A language model consumes tokens and predicts the next one. This consumes a
//! *noisy image* and predicts the direction to move it in. The blocks look
//! familiar — attention, a gated feed-forward, RMS norms — but four things are
//! not:
//!
//! 1. **Every layer is modulated by the timestep.** A 512-wide conditioning
//!    vector becomes four signals per layer: two scales applied before attention
//!    and feed-forward, and two `tanh` gates applied to their outputs. A
//!    language model has nothing like it.
//! 2. **Sandwich norms.** Each sub-layer is normalised on the way in *and* on
//!    the way out — `attention_norm1` before, `attention_norm2` after.
//! 3. **The sequence is text then image.** Words and image patches sit in one
//!    attention, told apart by a learned two-row indicator and by their rotary
//!    positions, and the text half is discarded at the end.
//! 4. **Three position axes.** See [`crate::rope3d`].
//!
//! # The container tells you nothing
//!
//! `ideogram4-Q4_0.gguf` has **zero metadata keys** — no architecture, no layer
//! count, no head count. Every number in [`Config`] was read off the tensor
//! index: `attention.qkv.weight [4608, 13824]` is three times 4608 so the QKV is
//! fused, `attention.norm_q.weight [256]` gives head_dim 256 and therefore 18
//! heads, `adaln_modulation.weight [512, 18432]` is four times 4608 from a
//! 512-wide vector. The layer count is the largest `layers.N.` in the index.
//!
//! # Three details that produce a picture, and a wrong one
//!
//! **The attention scale is `1/128`, not `1/sqrt(256)`.** head_dim is 256, so
//! the textbook scale would be 1/16. Ideogram uses 1/128 and so does this.
//!
//! **The timestep runs backwards and the output is negated.** The timestep is
//! `1000 * (1 - sigma)` and the final tensor is multiplied by -1. The two
//! cancel; implementing one without the other walks away from the image.
//!
//! **The final layer applies `silu` to a vector that already went through one.**
//! `adaln_input` is `silu(adaln_proj(t))`, and the final layer's modulation is
//! `adaln_modulation(silu(adaln_input))`. The blocks do *not* do this. It is
//! matched deliberately rather than tidied.
//!
//! # Memory
//!
//! One ggml context per layer, not one for the whole model. The hidden state is
//! carried across as a `Vec<f32>`; everything else is dropped with the context.
//! A single arena holding 34 layers of activations at 4096 tokens would want
//! tens of gigabytes, and **an exhausted ggml arena aborts the process** rather
//! than returning an error.

use chaos_ggml::{Context, Tensor, WeightSet};
use chaos_gguf::GgmlType;
use chaos_model::Model;

use crate::rope3d;

/// ggml's F32, for tensors used elementwise rather than as a matmul operand.
const F32: GgmlType = GgmlType(0);
/// ggml's BF16, which is what most of this container's small tensors are.
const BF16: GgmlType = GgmlType(30);

/// Shapes, all read from the tensor index rather than from metadata.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub emb_dim: i64,
    pub num_layers: u32,
    pub num_heads: i64,
    pub intermediate: i64,
    pub adaln_dim: i64,
    /// Patch channels in and out: 32 latent channels times a 2x2 patch.
    pub in_channels: i64,
    /// Width of the text encoder's stacked hidden states: 13 layers of 4096.
    pub llm_features: i64,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub patch: i64,
    pub ae_channels: i64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            emb_dim: 4608,
            num_layers: 34,
            num_heads: 18,
            intermediate: 12288,
            adaln_dim: 512,
            in_channels: 128,
            llm_features: 53248,
            rope_theta: rope3d::ROPE_THETA,
            norm_eps: 1e-5,
            patch: 2,
            ae_channels: 32,
        }
    }
}

impl Config {
    pub fn head_dim(&self) -> i64 {
        self.emb_dim / self.num_heads
    }

    /// Count the layers in a container, since no metadata key says.
    pub fn from_model(model: &Model) -> Self {
        let mut n = 0u32;
        for name in model.tensor_names() {
            if let Some(rest) = name.strip_prefix("layers.") {
                if let Some((num, _)) = rest.split_once('.') {
                    if let Ok(i) = num.parse::<u32>() {
                        n = n.max(i + 1);
                    }
                }
            }
        }
        let mut c = Config::default();
        if n > 0 {
            c.num_layers = n;
        }
        c
    }

    /// Every tensor the forward pass binds, for a completeness check that needs
    /// no arena.
    pub fn required_tensors(&self) -> Vec<String> {
        let mut v = vec![
            "input_proj.weight".to_string(),
            "input_proj.bias".to_string(),
            "llm_cond_norm.weight".to_string(),
            "llm_cond_proj.weight".to_string(),
            "llm_cond_proj.bias".to_string(),
            "t_embedding.mlp_in.weight".to_string(),
            "t_embedding.mlp_in.bias".to_string(),
            "t_embedding.mlp_out.weight".to_string(),
            "t_embedding.mlp_out.bias".to_string(),
            "adaln_proj.weight".to_string(),
            "adaln_proj.bias".to_string(),
            "embed_image_indicator.weight".to_string(),
            "final_layer.linear.weight".to_string(),
            "final_layer.linear.bias".to_string(),
            "final_layer.adaln_modulation.weight".to_string(),
            "final_layer.adaln_modulation.bias".to_string(),
        ];
        for i in 0..self.num_layers {
            for (t, _) in block_tensors(i) {
                v.push(t);
            }
        }
        v
    }
}

/// The thirteen tensors of one block, and whether each is used elementwise.
///
/// Elementwise means "convert BF16 to F32 at bind time": ggml's `mul` and `add`
/// want F32, while `mul_mat` reads Q4_0 and BF16 directly.
fn block_tensors(layer: u32) -> Vec<(String, bool)> {
    let p = format!("layers.{layer}");
    [
        ("adaln_modulation.weight", false),
        ("adaln_modulation.bias", true),
        ("attention.qkv.weight", false),
        ("attention.o.weight", false),
        ("attention.norm_q.weight", true),
        ("attention.norm_k.weight", true),
        ("attention_norm1.weight", true),
        ("attention_norm2.weight", true),
        ("ffn_norm1.weight", true),
        ("ffn_norm2.weight", true),
        ("feed_forward.w1.weight", false),
        ("feed_forward.w2.weight", false),
        ("feed_forward.w3.weight", false),
    ]
    .iter()
    .map(|(t, f)| (format!("{p}.{t}"), *f))
    .collect()
}

/// What went wrong.
#[derive(Debug)]
pub enum Error {
    Missing(String),
    Model(String),
    Ggml(chaos_ggml::GgmlError),
    Shape(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Missing(n) => write!(f, "the denoiser has no tensor {n:?}"),
            Error::Model(m) => write!(f, "{m}"),
            Error::Ggml(e) => write!(f, "ggml: {e:?}"),
            Error::Shape(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<chaos_ggml::GgmlError> for Error {
    fn from(e: chaos_ggml::GgmlError) -> Self {
        Error::Ggml(e)
    }
}

/// Widen BF16 to F32.
///
/// BF16 is the top 16 bits of an F32, so this is a shift — no table, no
/// rounding. Only the elementwise tensors are converted; `llm_cond_proj` alone
/// would cost a gigabyte, and `ggml_mul_mat` reads BF16 without help.
fn bf16_to_f32(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for pair in bytes.chunks_exact(2) {
        out.extend_from_slice(&[0, 0, pair[0], pair[1]]);
    }
    out
}

/// Read one tensor and bind it, widening BF16 when it is used elementwise.
fn bind<'c>(
    model: &Model,
    ctx: &'c Context,
    set: &mut WeightSet<'c>,
    name: &str,
    as_f32: bool,
) -> Result<(), Error> {
    let loc = model
        .location(name)
        .ok_or_else(|| Error::Missing(name.to_string()))?
        .clone();
    let data = model
        .read_tensor(name)
        .map_err(|e| Error::Model(format!("{name}: {e}")))?;
    if as_f32 && loc.ty == BF16 {
        set.bind(ctx, name, F32, &loc.dims, bf16_to_f32(&data))?;
    } else {
        set.bind(ctx, name, loc.ty, &loc.dims, data)?;
    }
    Ok(())
}

fn get<'c>(set: &WeightSet<'c>, name: &str) -> Result<Tensor<'c>, Error> {
    set.get(name)
        .copied()
        .ok_or_else(|| Error::Missing(name.to_string()))
}

/// `y = W x (+ b)` for a `[in, out]` weight over `[in, tokens]`.
fn linear<'c>(
    ctx: &'c Context,
    w: &Tensor<'c>,
    b: Option<&Tensor<'c>>,
    x: &Tensor<'c>,
) -> Result<Tensor<'c>, Error> {
    let y = ctx.mul_mat(w, x)?;
    match b {
        Some(b) => {
            let n = b.ne()[0];
            let bb = ctx.reshape_2d(b, n, 1)?;
            Ok(ctx.add(&y, &bb)?)
        }
        None => Ok(y),
    }
}

/// RMS norm over `ne[0]`, then the learned per-channel scale.
fn rms_norm<'c>(
    ctx: &'c Context,
    x: &Tensor<'c>,
    w: &Tensor<'c>,
    eps: f32,
) -> Result<Tensor<'c>, Error> {
    let h = ctx.rms_norm(x, eps)?;
    Ok(ctx.mul(&h, w)?)
}

/// `x * (1 + scale)`, broadcast over every token.
fn modulate<'c>(ctx: &'c Context, x: &Tensor<'c>, scale: &Tensor<'c>) -> Result<Tensor<'c>, Error> {
    let s = ctx.reshape_2d(scale, scale.ne()[0], 1)?;
    let scaled = ctx.mul(x, &s)?;
    Ok(ctx.add(x, &scaled)?)
}

/// Rotate `[head_dim, tokens, heads]` by the interleaved rotary table.
///
/// Consecutive pairs `(x[2f], x[2f+1])` rotate together; `cos` and `sin` are
/// `[1, head_dim/2, tokens, 1]` and broadcast over heads.
///
/// The reference builds this from `repeat` and strided views. Splitting the pair
/// into two views, rotating, and concatenating is the same arithmetic with fewer
/// copies — and the shapes stay readable, which matters because a transposed
/// rotation is invisible until it is a picture.
fn apply_rope<'c>(
    ctx: &'c Context,
    x: &Tensor<'c>,
    cos: &Tensor<'c>,
    sin: &Tensor<'c>,
) -> Result<Tensor<'c>, Error> {
    let ne = x.ne();
    let (head_dim, tokens, heads) = (ne[0], ne[1], ne[2]);
    let half = head_dim / 2;

    let pairs = ctx.reshape_4d(x, [2, half, tokens, heads])?;
    let (_, nb) = pairs.dims_and_strides();
    // ne[0] = 1 picks one element of each pair; the strides stay the pair
    // tensor's own, so the view walks the same grid one element at a time.
    let even = ctx.view_4d(&pairs, [1, half, tokens, heads], [nb[1], nb[2], nb[3]], 0)?;
    let odd = ctx.view_4d(
        &pairs,
        [1, half, tokens, heads],
        [nb[1], nb[2], nb[3]],
        nb[0],
    )?;

    // (a, b) -> (a cos + b sin, b cos - a sin): the reference's
    // [cos, -sin; sin, cos] read down its columns. There is no `ggml_sub`
    // bound here, so the subtraction is an add of a negation.
    let neg_sin = ctx.scale(sin, -1.0)?;
    let out0 = ctx.add(&ctx.mul(&even, cos)?, &ctx.mul(&odd, sin)?)?;
    let out1 = ctx.add(&ctx.mul(&odd, cos)?, &ctx.mul(&even, &neg_sin)?)?;
    let out0 = ctx.cont_4d(&out0, [1, half, tokens, heads])?;
    let out1 = ctx.cont_4d(&out1, [1, half, tokens, heads])?;

    let joined = ctx.concat(&out0, &out1, 0)?;
    Ok(ctx.reshape_4d(&joined, [head_dim, tokens, heads, 1])?)
}

/// The sinusoidal timestep embedding, computed here rather than in the graph.
///
/// One timestep is one vector, so building it in Rust costs nothing and avoids
/// binding another ggml op. Two details are not the textbook ones: the timestep
/// is scaled by **10** first, and ggml emits cosines before sines while Ideogram
/// wants **sines first** — the reference chunks the result in two and swaps the
/// halves, which is what the ordering below does directly.
pub fn timestep_embedding(t: f32, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let scaled = t * 10.0;
    let mut out = vec![0.0f32; dim];
    for j in 0..half {
        let freq = (-max_period.ln() * j as f32 / half as f32).exp();
        let arg = scaled * freq;
        out[j] = arg.sin();
        out[half + j] = arg.cos();
    }
    out
}

/// Reorder a packed latent `[gw, gh, 128]` into denoiser tokens `[128, gw*gh]`.
///
/// **A pure index permutation, so it lives here and not in the graph.** The
/// packed latent numbers its channels `px + 2*py + 4*ae`; a token numbers them
/// `ae + 32*px + 64*py`. The reference does this with a reshape and a four-axis
/// permute, where naming the two patch axes the wrong way round transposes every
/// 2x2 block — a defect that survives as a picture and is invisible in a shape.
pub fn tokens_from_latent(packed: &[f32], gw: usize, gh: usize, ae: usize, p: usize) -> Vec<f32> {
    let ch = ae * p * p;
    let cells = gw * gh;
    let mut out = vec![0.0f32; cells * ch];
    for c in 0..ae {
        for py in 0..p {
            for px in 0..p {
                let src_c = px + p * py + p * p * c;
                let dst_c = c + ae * px + ae * p * py;
                for y in 0..gh {
                    for x in 0..gw {
                        let src = x + gw * y + gw * gh * src_c;
                        let dst = dst_c + ch * (x + gw * y);
                        out[dst] = packed[src];
                    }
                }
            }
        }
    }
    out
}

/// [`tokens_from_latent`] reversed.
pub fn latent_from_tokens(tokens: &[f32], gw: usize, gh: usize, ae: usize, p: usize) -> Vec<f32> {
    let ch = ae * p * p;
    let cells = gw * gh;
    let mut out = vec![0.0f32; cells * ch];
    for c in 0..ae {
        for py in 0..p {
            for px in 0..p {
                let dst_c = px + p * py + p * p * c;
                let src_c = c + ae * px + ae * p * py;
                for y in 0..gh {
                    for x in 0..gw {
                        let dst = x + gw * y + gw * gh * dst_c;
                        let src = src_c + ch * (x + gw * y);
                        out[dst] = tokens[src];
                    }
                }
            }
        }
    }
    out
}

/// The denoiser, reading its weights from a container one layer at a time.
pub struct Denoiser {
    model: Model,
    pub config: Config,
    threads: usize,
}

/// Everything the forward pass needs that is not in the container.
pub struct Inputs<'a> {
    /// The noisy latent, **already packed**: `[grid_w, grid_h, 128]` in ggml
    /// order, width fastest. See [`crate::vae::pack_latent`].
    pub latent: &'a [f32],
    pub grid_w: i64,
    pub grid_h: i64,
    /// The timestep, already `1000 * (1 - sigma)`.
    pub timestep: f32,
    /// The text encoder's stacked hidden states, `[llm_features, tokens]`, or
    /// empty — the unconditional pass carries no text at all, which is why it
    /// needs its own set of weights rather than an empty prompt.
    pub context: &'a [f32],
    pub context_len: usize,
}

impl Denoiser {
    pub fn open(model: Model, threads: usize) -> Self {
        let config = Config::from_model(&model);
        Denoiser {
            model,
            config,
            threads,
        }
    }

    /// Which required tensors the container does not have.
    pub fn missing(&self) -> Vec<String> {
        self.config
            .required_tensors()
            .into_iter()
            .filter(|n| self.model.location(n).is_none())
            .collect()
    }

    /// Run the denoiser and return the velocity, packed like the input latent.
    pub fn forward(&self, inp: &Inputs<'_>) -> Result<Vec<f32>, Error> {
        self.forward_with(inp, &mut |_, _| {})
    }

    /// As [`Self::forward`], reporting `(layer, total)` as each block finishes.
    ///
    /// A 34-layer pass over a streamed 5 GiB container takes minutes; a run with
    /// no output is indistinguishable from a hang.
    pub fn forward_with(
        &self,
        inp: &Inputs<'_>,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, Error> {
        let c = self.config;
        let image_tokens = (inp.grid_w * inp.grid_h) as usize;
        if inp.latent.len() != image_tokens * c.in_channels as usize {
            return Err(Error::Shape(format!(
                "a {}x{} grid at {} channels is {} values, got {}",
                inp.grid_w,
                inp.grid_h,
                c.in_channels,
                image_tokens * c.in_channels as usize,
                inp.latent.len()
            )));
        }
        let total = inp.context_len + image_tokens;

        // The rotary table and the indicator are the same for all 34 layers, so
        // they are built once.
        let ids = rope3d::positions(inp.context_len, inp.grid_h as usize, inp.grid_w as usize);
        let pe = rope3d::table(
            &ids,
            c.head_dim() as usize,
            c.rope_theta,
            rope3d::MROPE_SECTION,
        );
        let half = (c.head_dim() / 2) as usize;
        let mut cos = Vec::with_capacity(total * half);
        let mut sin = Vec::with_capacity(total * half);
        for p in 0..total {
            for f in 0..half {
                let base = (p * half + f) * 4;
                cos.push(pe[base]); // [0] is cos
                sin.push(pe[base + 2]); // [2] is +sin
            }
        }

        let (mut h, adaln) = self.prologue(inp, image_tokens)?;
        for layer in 0..c.num_layers {
            h = self.block(layer, &h, &adaln, &cos, &sin, total)?;
            progress(layer + 1, c.num_layers);
        }
        self.epilogue(&h, &adaln, inp.context_len, inp.grid_w, inp.grid_h)
    }

    /// Arena for a stage, sized from the token count.
    ///
    /// Generous on purpose: ggml **aborts** on an exhausted arena and takes the
    /// process with it, and the attention scores alone are `heads * tokens^2`.
    fn arena(&self, tokens: usize) -> usize {
        let c = self.config;
        let per_token = (5 * c.emb_dim + 2 * c.intermediate) as usize * 4;
        let scores = c.num_heads as usize * tokens * tokens * 4;
        (1 << 30) + tokens * per_token * 8 + scores * 6
    }

    /// Project the latent and the text, add the indicator, and build the
    /// modulation vector every layer shares.
    fn prologue(
        &self,
        inp: &Inputs<'_>,
        image_tokens: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let c = self.config;
        let total = (inp.context_len + image_tokens) as i64;
        let ctx = Context::new(self.arena(total as usize))?;
        let mut set = WeightSet::new();

        for (n, widen) in [
            ("input_proj.weight", false),
            ("input_proj.bias", true),
            ("t_embedding.mlp_in.weight", false),
            ("t_embedding.mlp_in.bias", true),
            ("t_embedding.mlp_out.weight", false),
            ("t_embedding.mlp_out.bias", true),
            ("adaln_proj.weight", false),
            ("adaln_proj.bias", true),
            ("embed_image_indicator.weight", true),
        ] {
            bind(&self.model, &ctx, &mut set, n, widen)?;
        }

        // -- the image half ---------------------------------------------------
        let toks = tokens_from_latent(
            inp.latent,
            inp.grid_w as usize,
            inp.grid_h as usize,
            c.ae_channels as usize,
            c.patch as usize,
        );
        let img = ctx.new_f32_2d(c.in_channels, image_tokens as i64)?;
        img.set_f32(&toks)?;
        let img = linear(
            &ctx,
            &get(&set, "input_proj.weight")?,
            Some(&get(&set, "input_proj.bias")?),
            &img,
        )?;

        // -- the text half, if there is one -----------------------------------
        let h = if inp.context_len > 0 {
            for (n, widen) in [
                ("llm_cond_norm.weight", true),
                ("llm_cond_proj.weight", false),
                ("llm_cond_proj.bias", true),
            ] {
                bind(&self.model, &ctx, &mut set, n, widen)?;
            }
            let txt = ctx.new_f32_2d(c.llm_features, inp.context_len as i64)?;
            txt.set_f32(inp.context)?;
            // eps 1e-6 here, not the 1e-5 the blocks use.
            let txt = rms_norm(&ctx, &txt, &get(&set, "llm_cond_norm.weight")?, 1e-6)?;
            let txt = linear(
                &ctx,
                &get(&set, "llm_cond_proj.weight")?,
                Some(&get(&set, "llm_cond_proj.bias")?),
                &txt,
            )?;
            // Text first, image second; the epilogue slices the text back off.
            ctx.concat(&txt, &img, 1)?
        } else {
            img
        };

        // -- which half is which ----------------------------------------------
        let ind = rope3d::image_indicator(inp.context_len, image_tokens);
        let ids = ctx.new_i32_1d(total)?;
        ids.set_i32(&ind)?;
        let ind_emb = ctx.get_rows(&get(&set, "embed_image_indicator.weight")?, &ids)?;
        let h = ctx.add(&h, &ind_emb)?;

        // -- the timestep, and the vector every layer modulates on -------------
        let emb = timestep_embedding(inp.timestep, c.emb_dim as usize, 10000.0);
        let t = ctx.new_f32_2d(c.emb_dim, 1)?;
        t.set_f32(&emb)?;
        let t = linear(
            &ctx,
            &get(&set, "t_embedding.mlp_in.weight")?,
            Some(&get(&set, "t_embedding.mlp_in.bias")?),
            &t,
        )?;
        let t = ctx.silu(&t)?;
        let t = linear(
            &ctx,
            &get(&set, "t_embedding.mlp_out.weight")?,
            Some(&get(&set, "t_embedding.mlp_out.bias")?),
            &t,
        )?;
        let adaln = linear(
            &ctx,
            &get(&set, "adaln_proj.weight")?,
            Some(&get(&set, "adaln_proj.bias")?),
            &t,
        )?;
        let adaln = ctx.silu(&adaln)?;

        ctx.compute_many(&[&h, &adaln], self.threads)?;
        Ok((h.to_vec_f32(), adaln.to_vec_f32()))
    }

    /// One transformer block, in its own context.
    fn block(
        &self,
        layer: u32,
        h_in: &[f32],
        adaln_in: &[f32],
        cos: &[f32],
        sin: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Error> {
        let c = self.config;
        let ctx = Context::new(self.arena(tokens))?;
        let mut set = WeightSet::new();
        for (name, widen) in block_tensors(layer) {
            bind(&self.model, &ctx, &mut set, &name, widen)?;
        }
        let p = format!("layers.{layer}");
        let n = tokens as i64;

        let h = ctx.new_f32_2d(c.emb_dim, n)?;
        h.set_f32(h_in)?;
        let adaln = ctx.new_f32_2d(c.adaln_dim, 1)?;
        adaln.set_f32(adaln_in)?;

        let half = c.head_dim() / 2;
        let cos_t = ctx.new_f32_4d(1, half, n, 1)?;
        cos_t.set_f32(cos)?;
        let sin_t = ctx.new_f32_4d(1, half, n, 1)?;
        sin_t.set_f32(sin)?;

        // -- the four modulation signals --------------------------------------
        let m = linear(
            &ctx,
            &get(&set, &format!("{p}.adaln_modulation.weight"))?,
            Some(&get(&set, &format!("{p}.adaln_modulation.bias"))?),
            &adaln,
        )?;
        let chunk = |i: i64| -> Result<Tensor<'_>, Error> {
            let v = ctx.view_2d(&m, c.emb_dim, 1, 0, (i * c.emb_dim) as usize * 4)?;
            Ok(ctx.cont_2d(&v, [c.emb_dim, 1])?)
        };
        let scale_msa = chunk(0)?;
        let gate_msa = ctx.tanh(&chunk(1)?)?;
        let scale_mlp = chunk(2)?;
        let gate_mlp = ctx.tanh(&chunk(3)?)?;

        // -- attention, inside its sandwich -----------------------------------
        let a = rms_norm(
            &ctx,
            &h,
            &get(&set, &format!("{p}.attention_norm1.weight"))?,
            c.norm_eps,
        )?;
        let a = modulate(&ctx, &a, &scale_msa)?;
        let a = self.attention(&ctx, &set, &p, &a, &cos_t, &sin_t, n)?;
        let a = rms_norm(
            &ctx,
            &a,
            &get(&set, &format!("{p}.attention_norm2.weight"))?,
            c.norm_eps,
        )?;
        let h = ctx.add(&h, &ctx.mul(&a, &gate_msa)?)?;

        // -- feed-forward, inside its own -------------------------------------
        let f = rms_norm(
            &ctx,
            &h,
            &get(&set, &format!("{p}.ffn_norm1.weight"))?,
            c.norm_eps,
        )?;
        let f = modulate(&ctx, &f, &scale_mlp)?;
        let w1 = linear(
            &ctx,
            &get(&set, &format!("{p}.feed_forward.w1.weight"))?,
            None,
            &f,
        )?;
        let w3 = linear(
            &ctx,
            &get(&set, &format!("{p}.feed_forward.w3.weight"))?,
            None,
            &f,
        )?;
        let f = ctx.mul(&ctx.silu(&w1)?, &w3)?;
        let f = linear(
            &ctx,
            &get(&set, &format!("{p}.feed_forward.w2.weight"))?,
            None,
            &f,
        )?;
        let f = rms_norm(
            &ctx,
            &f,
            &get(&set, &format!("{p}.ffn_norm2.weight"))?,
            c.norm_eps,
        )?;
        let out = ctx.add(&h, &ctx.mul(&f, &gate_mlp)?)?;

        ctx.compute(&out, self.threads)?;
        Ok(out.to_vec_f32())
    }

    /// Fused QKV, per-head RMS norm on q and k, rotary, then plain attention.
    #[allow(clippy::too_many_arguments)]
    fn attention<'c>(
        &self,
        ctx: &'c Context,
        set: &WeightSet<'c>,
        p: &str,
        x: &Tensor<'c>,
        cos: &Tensor<'c>,
        sin: &Tensor<'c>,
        n: i64,
    ) -> Result<Tensor<'c>, Error> {
        let c = self.config;
        let (hd, nh) = (c.head_dim(), c.num_heads);

        let qkv = linear(
            ctx,
            &get(set, &format!("{p}.attention.qkv.weight"))?,
            None,
            x,
        )?;
        let (_, nb) = qkv.dims_and_strides();
        let part = |i: i64| -> Result<Tensor<'c>, Error> {
            let v = ctx.view_2d(&qkv, c.emb_dim, n, nb[1], (i * c.emb_dim) as usize * 4)?;
            let v = ctx.cont_2d(&v, [c.emb_dim, n])?;
            // [emb, tokens] -> [head_dim, heads, tokens] -> [head_dim, tokens, heads]
            let v = ctx.reshape_4d(&v, [hd, nh, n, 1])?;
            Ok(ctx.cont_4d(&ctx.permute(&v, [0, 2, 1, 3])?, [hd, n, nh, 1])?)
        };
        let q = part(0)?;
        let k = part(1)?;
        let v = part(2)?;

        let q = rms_norm(
            ctx,
            &q,
            &get(set, &format!("{p}.attention.norm_q.weight"))?,
            c.norm_eps,
        )?;
        let k = rms_norm(
            ctx,
            &k,
            &get(set, &format!("{p}.attention.norm_k.weight"))?,
            c.norm_eps,
        )?;
        let q = apply_rope(ctx, &q, cos, sin)?;
        let k = apply_rope(ctx, &k, cos, sin)?;

        // scores[j, i, head] = k_j . q_i, softmaxed over j. **1/128, not 1/16.**
        let scores = ctx.mul_mat(&k, &q)?;
        let scores = ctx.scale(&scores, 1.0 / 128.0)?;
        let probs = ctx.soft_max(&scores)?;

        // v is [head_dim, tokens, heads]; contracting over tokens needs it
        // transposed, the same trick the autoencoder's attention uses.
        let vt = ctx.cont_4d(&ctx.permute(&v, [1, 0, 2, 3])?, [n, hd, nh, 1])?;
        let out = ctx.mul_mat(&vt, &probs)?;

        let out = ctx.cont_4d(&ctx.permute(&out, [0, 2, 1, 3])?, [hd, nh, n, 1])?;
        let out = ctx.reshape_2d(&out, c.emb_dim, n)?;
        linear(
            ctx,
            &get(set, &format!("{p}.attention.o.weight"))?,
            None,
            &out,
        )
    }

    /// The final layer, the text slice, and the negation.
    fn epilogue(
        &self,
        h_in: &[f32],
        adaln_in: &[f32],
        context_len: usize,
        grid_w: i64,
        grid_h: i64,
    ) -> Result<Vec<f32>, Error> {
        let c = self.config;
        let image_tokens = (grid_w * grid_h) as usize;
        let tokens = context_len + image_tokens;
        let ctx = Context::new(self.arena(tokens))?;
        let mut set = WeightSet::new();
        for (n, widen) in [
            ("final_layer.linear.weight", false),
            ("final_layer.linear.bias", true),
            ("final_layer.adaln_modulation.weight", false),
            ("final_layer.adaln_modulation.bias", true),
        ] {
            bind(&self.model, &ctx, &mut set, n, widen)?;
        }

        let n = tokens as i64;
        let h = ctx.new_f32_2d(c.emb_dim, n)?;
        h.set_f32(h_in)?;
        let adaln = ctx.new_f32_2d(c.adaln_dim, 1)?;
        adaln.set_f32(adaln_in)?;

        // A second silu on a vector that already had one -- see the module docs.
        let scale = linear(
            &ctx,
            &get(&set, "final_layer.adaln_modulation.weight")?,
            Some(&get(&set, "final_layer.adaln_modulation.bias")?),
            &ctx.silu(&adaln)?,
        )?;
        // LayerNorm with no affine parameters, which is how the container shows
        // it: there is no `final_layer.norm_final.weight` to bind.
        let x = ctx.norm(&h, 1e-6)?;
        let x = modulate(&ctx, &x, &scale)?;
        let x = linear(
            &ctx,
            &get(&set, "final_layer.linear.weight")?,
            Some(&get(&set, "final_layer.linear.bias")?),
            &x,
        )?;

        // Drop the text tokens: only the image half is an image.
        let (_, nb) = x.dims_and_strides();
        let x = ctx.view_2d(
            &x,
            c.in_channels,
            image_tokens as i64,
            nb[1],
            context_len * nb[1],
        )?;
        let x = ctx.cont_2d(&x, [c.in_channels, image_tokens as i64])?;
        // The sampler wants a velocity toward the image; the model emits the
        // opposite. See the module docs.
        let x = ctx.scale(&x, -1.0)?;
        ctx.compute(&x, self.threads)?;

        Ok(latent_from_tokens(
            &x.to_vec_f32(),
            grid_w as usize,
            grid_h as usize,
            c.ae_channels as usize,
            c.patch as usize,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shapes have to agree with each other and with the autoencoder's.
    #[test]
    fn the_config_is_self_consistent() {
        let c = Config::default();
        assert_eq!(c.head_dim(), 256, "4608 over 18 heads");
        assert_eq!(c.emb_dim % c.num_heads, 0);
        assert_eq!(c.emb_dim * 3, 13824, "the fused QKV is three of them");
        assert_eq!(c.emb_dim * 4, 18432, "four modulation signals");
        // 32 latent channels in a 2x2 patch is the denoiser's 128, and 32 is
        // exactly what the autoencoder's latent carries. Two repositories, one
        // undocumented interface.
        assert_eq!(c.ae_channels * c.patch * c.patch, c.in_channels);
        assert_eq!(c.ae_channels, crate::vae::LATENT_CHANNELS);
        assert_eq!(
            c.llm_features,
            4096 * 13,
            "13 layers of a 4096-wide encoder"
        );
    }

    /// The tensor list covers every layer and the scaffolding around them.
    #[test]
    fn the_required_list_covers_the_whole_model() {
        let v = Config::default().required_tensors();
        assert_eq!(v.len(), 34 * 13 + 16, "{}", v.len());
        assert!(v.contains(&"layers.33.attention.qkv.weight".to_string()));
        assert!(!v.contains(&"layers.34.attention.qkv.weight".to_string()));
        assert!(v.contains(&"embed_image_indicator.weight".to_string()));
        // No `norm_final` weight: that LayerNorm has no affine parameters, so
        // asking for one would report a complete container as broken.
        assert!(!v.iter().any(|n| n.contains("norm_final")));
    }

    /// BF16 is the top half of an F32, so widening is a shift.
    #[test]
    fn bf16_widens_by_shifting() {
        let out = bf16_to_f32(&0x3F80u16.to_le_bytes());
        assert_eq!(f32::from_le_bytes(out.try_into().unwrap()), 1.0);
        let out = bf16_to_f32(&0xC000u16.to_le_bytes());
        assert_eq!(f32::from_le_bytes(out.try_into().unwrap()), -2.0);
        assert_eq!(bf16_to_f32(&[0, 0, 0, 0]).len(), 8);
    }

    /// The patch reordering is a permutation, so it must lose nothing and
    /// invert exactly.
    #[test]
    fn the_patch_reordering_is_an_exact_permutation() {
        let (gw, gh, ae, p) = (3usize, 2usize, 4usize, 2usize);
        let n = gw * gh * ae * p * p;
        let packed: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let toks = tokens_from_latent(&packed, gw, gh, ae, p);
        assert_eq!(toks.len(), n);
        // A permutation moves values, never invents or drops them.
        let mut a = packed.clone();
        let mut b = toks.clone();
        a.sort_by(f32::total_cmp);
        b.sort_by(f32::total_cmp);
        assert_eq!(a, b, "values were lost or duplicated");

        assert_eq!(latent_from_tokens(&toks, gw, gh, ae, p), packed);

        // And the documented channel mapping: packed channel px + 2py + 4c
        // becomes token channel c + 4px + 8py at ae = 4.
        let ch = ae * p * p;
        for c in 0..ae {
            for py in 0..p {
                for px in 0..p {
                    let src_c = px + p * py + p * p * c;
                    let dst_c = c + ae * px + ae * p * py;
                    // Cell (1, 0) of the grid.
                    let src = 1 + gw * gh * src_c;
                    let dst = dst_c + ch;
                    assert_eq!(toks[dst], packed[src], "c={c} px={px} py={py}");
                }
            }
        }
    }

    /// The timestep embedding puts sines first, which is the swap the reference
    /// performs after ggml emits cosines first.
    #[test]
    fn the_timestep_embedding_leads_with_sines() {
        let dim = 8;
        let emb = timestep_embedding(1.0, dim, 10000.0);
        assert_eq!(emb.len(), dim);
        // Frequency 0 has exponent 0, so freq = 1 and the argument is t * 10.
        assert!((emb[0] - 10.0f32.sin()).abs() < 1e-6, "{}", emb[0]);
        assert!(
            (emb[dim / 2] - 10.0f32.cos()).abs() < 1e-6,
            "{}",
            emb[dim / 2]
        );
        // Every pair is on the unit circle, which a mixed-up ordering breaks.
        for j in 0..dim / 2 {
            let r = emb[j].powi(2) + emb[dim / 2 + j].powi(2);
            assert!((r - 1.0).abs() < 1e-5, "pair {j} is not a unit vector");
        }
        // At t = 0 every sine is 0 and every cosine is 1.
        let zero = timestep_embedding(0.0, dim, 10000.0);
        assert!(zero[..dim / 2].iter().all(|v| v.abs() < 1e-9));
        assert!(zero[dim / 2..].iter().all(|v| (v - 1.0).abs() < 1e-9));
    }
}
