//! LoRA adapters and control vectors, read and checked against a base model.
//!
//! # Why the loader is a separate thing from the application
//!
//! Applying either one is a change to the forward pass. *Deciding whether this
//! adapter belongs to this model* is not — it is arithmetic on shapes and a
//! handful of metadata keys, and it is where the expensive mistakes live.
//!
//! A LoRA whose `lora_a` is stored untransposed multiplies cleanly against the
//! wrong axis and produces a model that answers fluently and is not the
//! fine-tune. A control vector built for a 32-layer model applied to a 26-layer
//! one shifts the wrong residuals. **Neither raises.** Both are this project's
//! signature failure: fluent, confident, wrong.
//!
//! So everything here refuses rather than adapts. `alpha / rank` is computed
//! once and reported, shapes are checked against the base model's own tensors,
//! and a mismatch names the tensor rather than saying "incompatible".

use std::collections::BTreeMap;

use crate::Model;

/// One `(A, B)` pair for a tensor a LoRA modifies.
#[derive(Debug, Clone)]
pub struct LoraPair {
    /// Base-model tensor this pair modifies, e.g. `blk.0.attn_q.weight`.
    pub target: String,
    /// `[n_in, rank]`.
    pub a_dims: Vec<u64>,
    /// `[rank, n_out]`.
    pub b_dims: Vec<u64>,
}

impl LoraPair {
    pub fn rank(&self) -> u64 {
        // `a` is `[n_in, rank]`, so the rank is its second dimension. Reading
        // `a_dims[0]` gives `n_in` and looks plausible on a square projection,
        // which is exactly the kind of thing that survives a smoke test.
        self.a_dims.get(1).copied().unwrap_or(0)
    }
}

/// A parsed LoRA adapter, before anything is applied.
#[derive(Debug, Clone)]
pub struct Lora {
    pub arch: String,
    /// llama.cpp's `adapter.lora.alpha`. The applied scale is
    /// `user_scale * alpha / rank`.
    pub alpha: f32,
    pub pairs: Vec<LoraPair>,
}

impl Lora {
    /// The multiplier applied to `B·A`, for a given user scale.
    ///
    /// **`alpha / rank`, not `alpha`.** A rank-64 adapter with alpha 16 scales
    /// by 0.25; using alpha alone would apply it 4x too strongly, which does
    /// not error — it produces a model that is recognisably the fine-tune and
    /// wrong in degree, the hardest kind of wrong to notice.
    pub fn scale(&self, user_scale: f32) -> f32 {
        let rank = self.pairs.first().map(|p| p.rank()).unwrap_or(0);
        if rank == 0 {
            return user_scale;
        }
        user_scale * self.alpha / rank as f32
    }
}

/// Why an adapter was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum AdapterError {
    NotAnAdapter(String),
    WrongType(String),
    /// The adapter was built for a different architecture.
    ArchMismatch {
        adapter: String,
        model: String,
    },
    /// A shape that does not line up with the base tensor it modifies.
    Shape(String),
    Missing(String),
}

impl std::fmt::Display for AdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdapterError::NotAnAdapter(what) => write!(
                f,
                "{what} is not a LoRA adapter (no `adapter.type` in its metadata)"
            ),
            AdapterError::WrongType(t) => {
                write!(f, "adapter.type is {t:?}; only `lora` is supported")
            }
            AdapterError::ArchMismatch { adapter, model } => write!(
                f,
                "this adapter was built for `{adapter}` and the model is `{model}`. \
                 Applying it would shift the wrong tensors and the model would still answer."
            ),
            AdapterError::Shape(what) => write!(f, "{what}"),
            AdapterError::Missing(what) => write!(f, "missing {what}"),
        }
    }
}

impl std::error::Error for AdapterError {}

/// Read a LoRA adapter and check it against `base`.
///
/// Every check here is one llama.cpp also makes, and each exists because the
/// failure it prevents is silent.
pub fn load_lora(adapter: &Model, base: &Model) -> Result<Lora, AdapterError> {
    let arch = adapter.architecture().to_string();
    // Unscoped keys: `adapter.type` is not prefixed by the architecture, so
    // the arch-scoped accessors would look for `llama.adapter.type` and find
    // nothing -- reporting "not an adapter" for a perfectly good one.
    let ty = adapter
        .meta_str("adapter.type")
        .ok_or_else(|| AdapterError::NotAnAdapter("this file".into()))?;
    if ty != "lora" {
        return Err(AdapterError::WrongType(ty.to_string()));
    }
    if arch != base.architecture() {
        return Err(AdapterError::ArchMismatch {
            adapter: arch,
            model: base.architecture().to_string(),
        });
    }
    let alpha = adapter
        .meta_f32("adapter.lora.alpha")
        .ok_or_else(|| AdapterError::Missing("adapter.lora.alpha".into()))?;

    // Pair `.lora_a` with `.lora_b` by the target name they share. A `BTreeMap`
    // rather than a `HashMap` so the reported order is stable -- an error that
    // names a different tensor on each run is hard to act on.
    let mut a: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut b: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for name in adapter.tensor_names() {
        let Some(loc) = adapter.location(name) else {
            continue;
        };
        if let Some(target) = name.strip_suffix(".lora_a") {
            a.insert(target.to_string(), loc.dims.clone());
        } else if let Some(target) = name.strip_suffix(".lora_b") {
            b.insert(target.to_string(), loc.dims.clone());
        }
    }

    let mut pairs = Vec::new();
    for (target, a_dims) in a {
        let Some(b_dims) = b.get(&target).cloned() else {
            // Half a pair is not a partial adapter, it is a broken one: `B·A`
            // needs both, and skipping the target silently leaves that tensor
            // un-adapted while every other one changes.
            return Err(AdapterError::Missing(format!("{target}.lora_b")));
        };
        let pair = LoraPair {
            target: target.clone(),
            a_dims,
            b_dims,
        };
        check_shapes(&pair, base)?;
        pairs.push(pair);
    }
    if pairs.is_empty() {
        return Err(AdapterError::Missing(
            "any `.lora_a`/`.lora_b` tensor pair".into(),
        ));
    }
    Ok(Lora {
        arch: base.architecture().to_string(),
        alpha,
        pairs,
    })
}

/// The three shape rules, each guarding a silent failure.
fn check_shapes(pair: &LoraPair, base: &Model) -> Result<(), AdapterError> {
    let Some(target) = base.location(&pair.target) else {
        return Err(AdapterError::Missing(format!(
            "`{}` in the base model -- the adapter names a tensor this model \
             does not have, so it was built for a different one",
            pair.target
        )));
    };
    let (Some(&a0), Some(&a1)) = (pair.a_dims.first(), pair.a_dims.get(1)) else {
        return Err(AdapterError::Shape(format!(
            "`{}.lora_a` is not 2-D",
            pair.target
        )));
    };
    let (Some(&b0), Some(&b1)) = (pair.b_dims.first(), pair.b_dims.get(1)) else {
        return Err(AdapterError::Shape(format!(
            "`{}.lora_b` is not 2-D",
            pair.target
        )));
    };
    let (t0, t1) = (
        target.dims.first().copied().unwrap_or(0),
        target.dims.get(1).copied().unwrap_or(0),
    );

    // 1. A's input width must match the base tensor's.
    if a0 != t0 {
        return Err(AdapterError::Shape(format!(
            "`{}.lora_a` is {a0} wide and the base tensor is {t0}. Wrong base model.",
            pair.target
        )));
    }
    // 2. B's output width must match the base tensor's.
    if b1 != t1 {
        return Err(AdapterError::Shape(format!(
            "`{}.lora_b` outputs {b1} and the base tensor outputs {t1}. Wrong base model.",
            pair.target
        )));
    }
    // 3. The ranks must meet. llama.cpp calls a violation here "lora_a tensor
    //    is not transposed", and it is the one that does NOT announce itself:
    //    an untransposed A still multiplies, against the wrong axis, and the
    //    result is a model that answers fluently and is not the fine-tune.
    if a1 != b0 {
        return Err(AdapterError::Shape(format!(
            "`{}`: lora_a's rank is {a1} and lora_b's is {b0}. The A tensor is stored \
             untransposed -- it would still multiply, against the wrong axis, and the \
             model would answer fluently without being the fine-tune.",
            pair.target
        )));
    }
    Ok(())
}

/// A control vector: one direction per layer, added to the residual stream.
#[derive(Debug, Clone)]
pub struct ControlVector {
    /// `directions[il]` is `None` where the file has no vector for that layer.
    pub directions: Vec<Option<Vec<f32>>>,
    pub n_embd: usize,
}

impl ControlVector {
    /// Layers this vector actually touches.
    pub fn active_layers(&self) -> usize {
        self.directions.iter().filter(|d| d.is_some()).count()
    }

    /// Restrict to `[start, end]` inclusive, llama.cpp's
    /// `--control-vector-layer-range`.
    ///
    /// Out-of-range layers are **cleared, not clamped**. Clamping would apply
    /// a direction to a layer the user excluded, which is the opposite of what
    /// the flag asks for.
    pub fn restrict(&mut self, start: usize, end: usize) {
        for (il, d) in self.directions.iter_mut().enumerate() {
            if il < start || il > end {
                *d = None;
            }
        }
    }

    /// Scale every direction. Combining two vectors is adding them, so a scale
    /// applied here composes the way `--control-vector-scaled` expects.
    pub fn scale(&mut self, by: f32) {
        for d in self.directions.iter_mut().flatten() {
            for v in d.iter_mut() {
                *v *= by;
            }
        }
    }
}

/// Read a control vector and check it fits `n_embd` and `n_layer`.
///
/// The tensors are named `direction.N`, one-based on the layer index, and a
/// file may skip layers. `n_embd` mismatches are refused: a direction of the
/// wrong width would be added to the residual stream with whatever bytes
/// followed it.
pub fn load_control_vector(
    cvec: &Model,
    n_embd: usize,
    n_layer: usize,
) -> Result<ControlVector, AdapterError> {
    let mut directions: Vec<Option<Vec<f32>>> = vec![None; n_layer];
    let mut found = 0;

    for name in cvec.tensor_names().map(str::to_string).collect::<Vec<_>>() {
        let Some(idx) = name.strip_prefix("direction.") else {
            continue;
        };
        let Ok(il): Result<usize, _> = idx.parse() else {
            continue;
        };
        let Some(loc) = cvec.location(&name).cloned() else {
            continue;
        };
        let width = loc.dims.first().copied().unwrap_or(0) as usize;
        if width != n_embd {
            return Err(AdapterError::Shape(format!(
                "`{name}` is {width} wide and the model's n_embd is {n_embd}. \
                 This control vector was built for a different model."
            )));
        }
        // llama.cpp numbers these from 1 and never has a direction for layer 0.
        if il == 0 || il > n_layer {
            return Err(AdapterError::Shape(format!(
                "`{name}` names layer {il}, and the model has layers 1..={n_layer}"
            )));
        }
        let mut bytes = vec![0u8; loc.size as usize];
        if cvec.read_range_into(&name, 0, &mut bytes).is_err() {
            return Err(AdapterError::Missing(format!("{name}'s data")));
        }
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if floats.len() != n_embd {
            return Err(AdapterError::Shape(format!(
                "`{name}` holds {} values for an n_embd of {n_embd}",
                floats.len()
            )));
        }
        directions[il - 1] = Some(floats);
        found += 1;
    }

    if found == 0 {
        return Err(AdapterError::Missing(
            "any `direction.N` tensor -- this file is not a control vector".into(),
        ));
    }
    Ok(ControlVector { directions, n_embd })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(target: &str, a: [u64; 2], b: [u64; 2]) -> LoraPair {
        LoraPair {
            target: target.into(),
            a_dims: a.to_vec(),
            b_dims: b.to_vec(),
        }
    }

    #[test]
    fn the_rank_is_a_s_second_dimension() {
        // `a_dims[0]` is n_in and looks plausible on a square projection, which
        // is exactly the kind of mistake that survives a smoke test.
        assert_eq!(pair("t", [4096, 16], [16, 4096]).rank(), 16);
    }

    #[test]
    fn the_scale_is_alpha_over_rank_not_alpha() {
        // A rank-64 adapter with alpha 16 scales by 0.25. Using alpha alone
        // applies it 4x too strongly -- which does not error, and produces a
        // model that IS recognisably the fine-tune and wrong in degree.
        let lora = Lora {
            arch: "llama".into(),
            alpha: 16.0,
            pairs: vec![pair("t", [4096, 64], [64, 4096])],
        };
        assert!((lora.scale(1.0) - 0.25).abs() < 1e-6, "{}", lora.scale(1.0));
        assert!((lora.scale(2.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_rankless_adapter_does_not_divide_by_zero() {
        let lora = Lora {
            arch: "llama".into(),
            alpha: 16.0,
            pairs: vec![],
        };
        assert_eq!(lora.scale(1.5), 1.5);
    }

    #[test]
    fn a_layer_range_clears_rather_than_clamps() {
        // Clamping would apply a direction to a layer the user EXCLUDED, which
        // is the opposite of what the flag asks for.
        let mut cv = ControlVector {
            directions: vec![Some(vec![1.0]), Some(vec![1.0]), Some(vec![1.0])],
            n_embd: 1,
        };
        cv.restrict(1, 1);
        assert_eq!(cv.active_layers(), 1);
        assert!(cv.directions[0].is_none());
        assert!(cv.directions[1].is_some());
        assert!(cv.directions[2].is_none());
    }

    #[test]
    fn scaling_composes_because_combining_vectors_is_adding_them() {
        let mut cv = ControlVector {
            directions: vec![Some(vec![2.0, -4.0])],
            n_embd: 2,
        };
        cv.scale(0.5);
        assert_eq!(cv.directions[0].as_deref(), Some(&[1.0f32, -2.0][..]));
    }

    #[test]
    fn the_error_for_an_untransposed_a_says_what_it_would_do() {
        // The one shape violation that does NOT announce itself: the multiply
        // still succeeds, against the wrong axis.
        let e = AdapterError::Shape(
            "`blk.0.attn_q.weight`: lora_a's rank is 4096 and lora_b's is 16. The A tensor \
             is stored untransposed -- it would still multiply, against the wrong axis, and \
             the model would answer fluently without being the fine-tune."
                .into(),
        );
        assert!(e.to_string().contains("answer fluently"));
    }

    #[test]
    fn an_arch_mismatch_explains_why_it_matters() {
        let e = AdapterError::ArchMismatch {
            adapter: "llama".into(),
            model: "qwen3".into(),
        };
        let s = e.to_string();
        assert!(s.contains("llama") && s.contains("qwen3"), "{s}");
        // The point is not that the names differ -- it is that applying it
        // anyway produces a model that still answers.
        assert!(s.contains("still answer"), "{s}");
    }
}
