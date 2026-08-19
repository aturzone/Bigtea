//! The text encoder: a language model used for what it *thinks*, not what it says.
//!
//! # Why a whole language model, and why not the one Chaos already runs
//!
//! Ideogram 4 conditions on Qwen3-VL-8B — but never on its output tokens. It
//! takes the **hidden states of thirteen layers**, stacks them into a 53248-wide
//! vector per token, and feeds that to the denoiser's `llm_cond_proj`. The model
//! is a description-understander, not a generator, and it is never sampled.
//!
//! That is why this is a separate forward pass rather than a flag on
//! `chaos-arch`'s. What is needed is thirteen intermediate tensors and no
//! logits, no KV cache, no sampling, and no generation loop — the entire prompt
//! goes through once. Threading that through a streaming token loop built for
//! the opposite shape would complicate the engine that runs every other model
//! here in order to serve one that is not a chat model at all.
//!
//! The container makes it easy: `Qwen3-VL-8B-Instruct-Q4_K_M.gguf` holds **no
//! vision tensors** — 399 of them, which is 36 blocks of 11 plus an embedding, a
//! norm and an output head. It is a Qwen3 text model wearing a `qwen3vl`
//! architecture name.
//!
//! # Which thirteen, and in what order
//!
//! `{1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36}` — every third layer, and
//! then the last. Read from the reference implementation, where the test is
//! `out_layers.contains(i + 1)` for block `i`, so the value `v` names the output
//! of block `v - 1`. **All thirteen are un-normalised block outputs**: the
//! reference only appends the `output_norm`-ed tensor when the set contains
//! `num_layers + 1`, which is 37 and is not in the list. Applying the final norm
//! to the last one is a plausible mistake that changes 1/13th of the
//! conditioning.
//!
//! # The layout the denoiser wants
//!
//! **Layer-fastest.** The reference concatenates the thirteen along the hidden
//! dimension and then permutes so that, for each of the 4096 hidden channels,
//! the thirteen layer values sit next to each other: `index = layer + 13 * h`.
//! Concatenating them the obvious way — one layer's 4096 values after another's
//! — gives a vector of exactly the right length, in the wrong order, and a
//! picture that has nothing to do with the prompt.
//!
//! # Positions
//!
//! Qwen3-VL uses three-axis rotary positions like the denoiser does. For text
//! there are no image patches, so all three axes carry the same number and the
//! whole thing collapses to ordinary 1-D RoPE — which is what
//! [`crate::rope3d`]'s own test asserts, and why `rope_ext` is enough here.

use chaos_ggml::{Context, RopeParams, Tensor, WeightSet};
use chaos_gguf::{GgmlType, Value};
use chaos_model::Model;

/// ggml's NeoX rotary mode: pairs are `(x[i], x[i + d/2])`, not adjacent.
const ROPE_TYPE_NEOX: i32 = 2;

/// The thirteen layers Ideogram 4 reads, as the reference names them: value `v`
/// is the output of block `v - 1`.
pub const OUT_LAYERS: [u32; 13] = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36];

/// The prompt template. No system message, and the assistant turn is opened but
/// never filled — the model is being asked to *understand*, not to reply.
pub fn wrap_prompt(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

/// Shapes, from the container's own metadata this time.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub blocks: u32,
    pub hidden: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub ffn: i64,
    pub eps: f32,
    pub rope_theta: f32,
}

#[derive(Debug)]
pub enum Error {
    Missing(String),
    Meta(String),
    Model(String),
    Ggml(chaos_ggml::GgmlError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Missing(n) => write!(f, "the text encoder has no tensor {n:?}"),
            Error::Meta(m) => write!(f, "{m}"),
            Error::Model(m) => write!(f, "{m}"),
            Error::Ggml(e) => write!(f, "ggml: {e:?}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<chaos_ggml::GgmlError> for Error {
    fn from(e: chaos_ggml::GgmlError) -> Self {
        Error::Ggml(e)
    }
}

impl Config {
    pub fn from_model(model: &Model) -> Result<Self, Error> {
        let meta = model.metadata();
        let arch = meta
            .get("general.architecture")
            .and_then(Value::as_str)
            .unwrap_or("");
        let u = |k: &str| -> Result<u64, Error> {
            meta.get(&format!("{arch}.{k}"))
                .and_then(Value::as_u64)
                .ok_or_else(|| Error::Meta(format!("no {arch}.{k} in the container")))
        };
        let f = |k: &str, d: f32| -> f32 {
            meta.get(&format!("{arch}.{k}"))
                .and_then(Value::as_f32)
                .unwrap_or(d)
        };
        let hidden = u("embedding_length")? as i64;
        let heads = u("attention.head_count")? as i64;
        Ok(Config {
            blocks: u("block_count")? as u32,
            hidden,
            heads,
            kv_heads: u("attention.head_count_kv")? as i64,
            // `key_length` rather than hidden/heads: Qwen3 head_dim is 128 while
            // 4096/32 is also 128, but the two are not the same field and other
            // sizes in the family disagree.
            head_dim: u("attention.key_length")
                .map(|v| v as i64)
                .unwrap_or(hidden / heads),
            ffn: u("feed_forward_length")? as i64,
            eps: f("attention.layer_norm_rms_epsilon", 1e-6),
            rope_theta: f("rope.freq_base", 5_000_000.0),
        })
    }

    pub fn required_tensors(&self) -> Vec<String> {
        let mut v = vec!["token_embd.weight".to_string()];
        for i in 0..self.blocks {
            for t in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                v.push(format!("blk.{i}.{t}"));
            }
        }
        v
    }
}

fn bind<'c>(
    model: &Model,
    ctx: &'c Context,
    set: &mut WeightSet<'c>,
    name: &str,
) -> Result<(), Error> {
    let loc = model
        .location(name)
        .ok_or_else(|| Error::Missing(name.to_string()))?
        .clone();
    let data = model
        .read_tensor(name)
        .map_err(|e| Error::Model(format!("{name}: {e}")))?;
    // Norm weights are stored F32 in this container, so nothing needs widening.
    set.bind(ctx, name, loc.ty, &loc.dims, data)?;
    Ok(())
}

fn get<'c>(set: &WeightSet<'c>, name: &str) -> Result<Tensor<'c>, Error> {
    set.get(name)
        .copied()
        .ok_or_else(|| Error::Missing(name.to_string()))
}

fn rms_norm<'c>(
    ctx: &'c Context,
    x: &Tensor<'c>,
    w: &Tensor<'c>,
    eps: f32,
) -> Result<Tensor<'c>, Error> {
    Ok(ctx.mul(&ctx.rms_norm(x, eps)?, w)?)
}

/// The text encoder, streamed one block at a time like the denoiser.
pub struct TextEncoder {
    model: Model,
    pub config: Config,
    threads: usize,
}

/// What one prompt produced.
pub struct Encoded {
    /// `[13 * hidden, tokens]`, layer-fastest — the denoiser's `context`.
    pub hidden: Vec<f32>,
    pub tokens: usize,
    pub width: usize,
}

impl TextEncoder {
    pub fn open(model: Model, threads: usize) -> Result<Self, Error> {
        let config = Config::from_model(&model)?;
        Ok(TextEncoder {
            model,
            config,
            threads,
        })
    }

    pub fn missing(&self) -> Vec<String> {
        self.config
            .required_tensors()
            .into_iter()
            .filter(|n| self.model.location(n).is_none())
            .collect()
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Run the prompt through and stack the thirteen layers.
    pub fn encode(
        &self,
        ids: &[u32],
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Encoded, Error> {
        let c = self.config;
        let t = ids.len() as i64;
        let wanted: Vec<u32> = OUT_LAYERS.iter().map(|v| v - 1).collect();

        // The embedding, in its own context: `token_embd` is 373 MB and is not
        // wanted alive beside thirty-six blocks of weights.
        let mut x = {
            let ctx = Context::new(self.arena(t as usize))?;
            let mut set = WeightSet::new();
            bind(&self.model, &ctx, &mut set, "token_embd.weight")?;
            let idt = ctx.new_i32_1d(t)?;
            let signed: Vec<i32> = ids.iter().map(|v| *v as i32).collect();
            idt.set_i32(&signed)?;
            let e = ctx.get_rows(&get(&set, "token_embd.weight")?, &idt)?;
            ctx.compute(&e, self.threads)?;
            e.to_vec_f32()
        };

        let mut captured: Vec<Vec<f32>> = Vec::with_capacity(OUT_LAYERS.len());
        for i in 0..c.blocks {
            x = self.block(i, &x, t)?;
            if wanted.contains(&i) {
                captured.push(x.clone());
            }
            progress(i + 1, c.blocks);
        }
        if captured.len() != OUT_LAYERS.len() {
            return Err(Error::Meta(format!(
                "captured {} layers, wanted {} -- the container has {} blocks and the \
                 layer list runs to {}",
                captured.len(),
                OUT_LAYERS.len(),
                c.blocks,
                OUT_LAYERS[OUT_LAYERS.len() - 1]
            )));
        }

        Ok(Encoded {
            hidden: interleave(&captured, c.hidden as usize, t as usize),
            tokens: t as usize,
            width: captured.len() * c.hidden as usize,
        })
    }

    /// The most likely next tokens after `ids` — **for checking this forward
    /// pass, not for generating anything.**
    ///
    /// The conditioning path never needs a logit. But a text encoder that is
    /// subtly wrong produces hidden states that are finite, correctly shaped and
    /// meaningless, and the picture that results is the only place it would show
    /// up. Running the same weights as a language model and requiring that
    /// "The capital of France is" continues with " Paris" is a check that costs
    /// one extra matmul and cannot be passed by accident.
    pub fn probe_next(&self, ids: &[u32], top: usize) -> Result<Vec<(u32, f32)>, Error> {
        let c = self.config;
        let t = ids.len() as i64;

        let mut x = {
            let ctx = Context::new(self.arena(t as usize))?;
            let mut set = WeightSet::new();
            bind(&self.model, &ctx, &mut set, "token_embd.weight")?;
            let idt = ctx.new_i32_1d(t)?;
            let signed: Vec<i32> = ids.iter().map(|v| *v as i32).collect();
            idt.set_i32(&signed)?;
            let e = ctx.get_rows(&get(&set, "token_embd.weight")?, &idt)?;
            ctx.compute(&e, self.threads)?;
            e.to_vec_f32()
        };
        for i in 0..c.blocks {
            x = self.block(i, &x, t)?;
        }

        // Only the last token matters, so the output head runs on one row
        // rather than on the whole prompt.
        let last = &x[((t - 1) as usize) * c.hidden as usize..];
        if std::env::var_os("CHAOS_DEBUG_TEXT").is_some() {
            let r = (last.iter().map(|v| v * v).sum::<f32>() / last.len() as f32).sqrt();
            eprintln!("[debug] last hidden: {} values, rms {r:.6}", last.len());
        }
        let ctx = Context::new(3 << 30)?;
        let mut set = WeightSet::new();
        bind(&self.model, &ctx, &mut set, "output_norm.weight")?;
        // Some containers tie the output head to the embedding; this one ships
        // an `output.weight`, and falling back keeps the check working either way.
        let head = if self.model.location("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        bind(&self.model, &ctx, &mut set, head)?;

        let h = ctx.new_f32_2d(c.hidden, 1)?;
        h.set_f32(last)?;
        let h = rms_norm(&ctx, &h, &get(&set, "output_norm.weight")?, c.eps)?;
        let logits = ctx.mul_mat(&get(&set, head)?, &h)?;
        ctx.compute(&logits, self.threads)?;

        let v = logits.to_vec_f32();
        if std::env::var_os("CHAOS_DEBUG_TEXT").is_some() {
            let r = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
            let nz = v.iter().filter(|x| **x != 0.0).count();
            eprintln!(
                "[debug] logits: {} values, rms {r:.6}, nonzero {nz}",
                v.len()
            );
        }
        let mut idx: Vec<u32> = (0..v.len() as u32).collect();
        idx.sort_by(|a, b| v[*b as usize].total_cmp(&v[*a as usize]));
        Ok(idx
            .into_iter()
            .take(top)
            .map(|i| (i, v[i as usize]))
            .collect())
    }

    /// Root-mean-square after the embedding and after each of the first
    /// `blocks` blocks.
    ///
    /// The same idea as `CHAOS_DUMP_LAYERS` on the language-model path: when a
    /// forward pass produces plausible-looking nothing, the first question is
    /// *which stage* stopped producing numbers, and a per-stage magnitude
    /// answers it in one run.
    pub fn debug_rms(&self, ids: &[u32], blocks: u32) -> Result<Vec<(String, f32)>, Error> {
        let stats = |v: &[f32]| {
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            let maxabs = v
                .iter()
                .filter(|x| x.is_finite())
                .fold(0.0f32, |a, b| a.max(b.abs()));
            (bad, maxabs)
        };
        let c = self.config;
        let t = ids.len() as i64;
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();

        let mut x = {
            let ctx = Context::new(self.arena(t as usize))?;
            let mut set = WeightSet::new();
            bind(&self.model, &ctx, &mut set, "token_embd.weight")?;
            let idt = ctx.new_i32_1d(t)?;
            let signed: Vec<i32> = ids.iter().map(|v| *v as i32).collect();
            idt.set_i32(&signed)?;
            let e = ctx.get_rows(&get(&set, "token_embd.weight")?, &idt)?;
            ctx.compute(&e, self.threads)?;
            e.to_vec_f32()
        };
        let (b, m) = stats(&x);
        let mut out = vec![(format!("embedding    max {m:>12.3} bad {b}"), rms(&x))];
        for i in 0..blocks.min(c.blocks) {
            x = self.block(i, &x, t)?;
            let (b, m) = stats(&x);
            out.push((format!("block {i:<8} max {m:>12.3} bad {b}"), rms(&x)));
        }
        Ok(out)
    }

    fn arena(&self, tokens: usize) -> usize {
        let c = self.config;
        let per_token = (4 * c.hidden + 3 * c.ffn) as usize * 4;
        let scores = c.heads as usize * tokens * tokens * 4;
        (768 << 20) + tokens * per_token * 8 + scores * 4
    }

    /// One Qwen3 block: norm, attention with per-head QK norm, norm, SwiGLU.
    fn block(&self, i: u32, x_in: &[f32], t: i64) -> Result<Vec<f32>, Error> {
        let c = self.config;
        let ctx = Context::new(self.arena(t as usize))?;
        let mut set = WeightSet::new();
        let p = format!("blk.{i}");
        for n in [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            bind(&self.model, &ctx, &mut set, &format!("{p}.{n}"))?;
        }

        let x = ctx.new_f32_2d(c.hidden, t)?;
        x.set_f32(x_in)?;

        let pos = ctx.new_i32_1d(t)?;
        let ps: Vec<i32> = (0..t as i32).collect();
        pos.set_i32(&ps)?;

        // A causal mask, added before the softmax. Building it here rather than
        // calling `soft_max_ext` keeps this to ops the crate already had.
        let mask = ctx.new_f32_2d(t, t)?;
        let mut m = vec![0.0f32; (t * t) as usize];
        for q in 0..t {
            for k in 0..t {
                if k > q {
                    m[(q * t + k) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        mask.set_f32(&m)?;

        // -- attention --------------------------------------------------------
        let h = rms_norm(
            &ctx,
            &x,
            &get(&set, &format!("{p}.attn_norm.weight"))?,
            c.eps,
        )?;
        let q = ctx.mul_mat(&get(&set, &format!("{p}.attn_q.weight"))?, &h)?;
        let k = ctx.mul_mat(&get(&set, &format!("{p}.attn_k.weight"))?, &h)?;
        let v = ctx.mul_mat(&get(&set, &format!("{p}.attn_v.weight"))?, &h)?;

        let q_raw = ctx.reshape_4d(&q, [c.head_dim, c.heads, t, 1])?;
        let k = ctx.reshape_4d(&k, [c.head_dim, c.kv_heads, t, 1])?;
        let v = ctx.reshape_4d(&v, [c.head_dim, c.kv_heads, t, 1])?;

        // Per-head RMS norm on q and k -- Qwen3's distinguishing feature, and
        // the reason `attn_q_norm.weight` is [128] rather than [4096].
        let q_norm = rms_norm(
            &ctx,
            &q_raw,
            &get(&set, &format!("{p}.attn_q_norm.weight"))?,
            c.eps,
        )?;
        let q = q_norm;
        let k = rms_norm(
            &ctx,
            &k,
            &get(&set, &format!("{p}.attn_k_norm.weight"))?,
            c.eps,
        )?;

        let rope = RopeParams {
            freq_base: c.rope_theta,
            ..RopeParams::default()
        };
        let q = ctx.rope_ext(&q, &pos, None, c.head_dim as i32, ROPE_TYPE_NEOX, 0, rope)?;
        let k = ctx.rope_ext(&k, &pos, None, c.head_dim as i32, ROPE_TYPE_NEOX, 0, rope)?;

        // [head_dim, heads, tokens] -> [head_dim, tokens, heads]
        let q = ctx.cont_4d(&ctx.permute(&q, [0, 2, 1, 3])?, [c.head_dim, t, c.heads, 1])?;
        let k = ctx.cont_4d(
            &ctx.permute(&k, [0, 2, 1, 3])?,
            [c.head_dim, t, c.kv_heads, 1],
        )?;
        let v = ctx.cont_4d(
            &ctx.permute(&v, [0, 2, 1, 3])?,
            [c.head_dim, t, c.kv_heads, 1],
        )?;

        // mul_mat broadcasts the 8 key heads across the 32 query heads, which is
        // grouped-query attention with no gather.
        let scores = ctx.mul_mat(&k, &q)?;
        let scores = ctx.scale(&scores, 1.0 / (c.head_dim as f32).sqrt())?;
        let scores = ctx.add(&scores, &ctx.reshape_4d(&mask, [t, t, 1, 1])?)?;
        let probs = ctx.soft_max(&scores)?;

        let vt = ctx.cont_4d(
            &ctx.permute(&v, [1, 0, 2, 3])?,
            [t, c.head_dim, c.kv_heads, 1],
        )?;
        let o = ctx.mul_mat(&vt, &probs)?;
        let o = ctx.cont_4d(&ctx.permute(&o, [0, 2, 1, 3])?, [c.head_dim, c.heads, t, 1])?;
        let o = ctx.reshape_2d(&o, c.head_dim * c.heads, t)?;
        let o = ctx.mul_mat(&get(&set, &format!("{p}.attn_output.weight"))?, &o)?;
        let x = ctx.add(&x, &o)?;

        // -- feed-forward ------------------------------------------------------
        let h = rms_norm(
            &ctx,
            &x,
            &get(&set, &format!("{p}.ffn_norm.weight"))?,
            c.eps,
        )?;
        let g = ctx.mul_mat(&get(&set, &format!("{p}.ffn_gate.weight"))?, &h)?;
        let u = ctx.mul_mat(&get(&set, &format!("{p}.ffn_up.weight"))?, &h)?;
        let f = ctx.mul(&ctx.silu(&g)?, &u)?;
        let f = ctx.mul_mat(&get(&set, &format!("{p}.ffn_down.weight"))?, &f)?;
        let out = ctx.add(&x, &f)?;

        if std::env::var("CHAOS_DEBUG_BLOCK").ok().as_deref() == Some(i.to_string().as_str()) {
            let probe = [
                ("h(attn_norm)", &h),
                ("q_raw", &q_raw),
                ("q_norm", &q_norm),
                ("q", &q),
                ("k", &k),
                ("scores", &scores),
                ("probs", &probs),
                ("o", &o),
                ("ffn", &f),
                ("out", &out),
            ];
            let refs: Vec<&chaos_ggml::Tensor<'_>> = probe.iter().map(|(_, t)| *t).collect();
            ctx.compute_many(&refs, self.threads)?;
            for (name, t) in &probe {
                let v = t.to_vec_f32();
                let bad = v.iter().filter(|x| !x.is_finite()).count();
                let m = v
                    .iter()
                    .filter(|x| x.is_finite())
                    .fold(0.0f32, |a, b| a.max(b.abs()));
                let r = (v
                    .iter()
                    .filter(|x| x.is_finite())
                    .map(|x| x * x)
                    .sum::<f32>()
                    / v.len() as f32)
                    .sqrt();
                eprintln!("  [blk {i}] {name:<14} max {m:>14.3}  rms {r:>12.4}  bad {bad}");
            }
        }

        ctx.compute(&out, self.threads)?;
        Ok(out.to_vec_f32())
    }
}

/// Stack the captured layers **layer-fastest**: `index = layer + n * hidden`.
///
/// The reference concatenates along the hidden dimension and then permutes.
/// Doing the permutation while writing costs nothing and makes the order
/// checkable — see the test, which is the only thing standing between this and
/// a 53248-wide vector that is correct in length and shuffled in content.
pub fn interleave(layers: &[Vec<f32>], hidden: usize, tokens: usize) -> Vec<f32> {
    let n = layers.len();
    let mut out = vec![0.0f32; tokens * hidden * n];
    for (l, layer) in layers.iter().enumerate() {
        for t in 0..tokens {
            for h in 0..hidden {
                out[t * hidden * n + h * n + l] = layer[t * hidden + h];
            }
        }
    }
    out
}

/// The ggml type of a tensor, for reporting.
pub fn type_of(model: &Model, name: &str) -> Option<GgmlType> {
    model.location(name).map(|l| l.ty)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Thirteen layers, every third then the last, and each names a block one
    /// lower than its own number.
    #[test]
    fn the_layer_list_matches_the_reference() {
        assert_eq!(OUT_LAYERS.len(), 13, "13 layers of 4096 is 53248");
        assert_eq!(OUT_LAYERS[0], 1, "the first is block 0's output");
        assert_eq!(OUT_LAYERS[12], 36, "the last is block 35's output");
        // Every third up to 34, then 36 -- 35 is deliberately skipped.
        for w in OUT_LAYERS[..12].windows(2) {
            assert_eq!(w[1] - w[0], 3, "{w:?}");
        }
        assert_eq!(OUT_LAYERS[11], 34);
        assert!(!OUT_LAYERS.contains(&35), "block 34's output is not used");
        // 37 would mean "the output_norm-ed final state", and it is not asked
        // for -- all thirteen are raw block outputs.
        assert!(!OUT_LAYERS.contains(&37), "the final norm is never applied");
        // The blocks they name all exist in a 36-block model.
        assert!(OUT_LAYERS.iter().all(|v| *v >= 1 && *v <= 36));
    }

    /// The prompt template opens an assistant turn it never fills.
    #[test]
    fn the_template_is_the_reference_one() {
        let p = wrap_prompt("a cat");
        assert_eq!(
            p,
            "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
        );
        // No system message: Ideogram's own template has none, and adding the
        // usual "You are a helpful assistant" changes every hidden state.
        assert!(!p.contains("system"));
    }

    /// The stacking is layer-fastest, which is the whole point of doing it here.
    #[test]
    fn layers_are_stacked_with_the_layer_index_fastest() {
        // Two layers, three hidden channels, two tokens.
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let out = interleave(&[a, b], 3, 2);
        assert_eq!(out.len(), 2 * 3 * 2);
        // Token 0: channel 0 from both layers, then channel 1 from both, ...
        assert_eq!(&out[..6], &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        // Token 1 follows, not interleaved with token 0.
        assert_eq!(&out[6..], &[4.0, 40.0, 5.0, 50.0, 6.0, 60.0]);
    }

    /// The width the denoiser expects, from the two containers' own numbers.
    #[test]
    fn thirteen_layers_of_this_encoder_is_the_denoisers_input_width() {
        assert_eq!(
            OUT_LAYERS.len() as i64 * 4096,
            crate::dit::Config::default().llm_features
        );
    }
}
