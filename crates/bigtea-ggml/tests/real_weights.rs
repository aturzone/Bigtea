//! The join: real weights off disk, turned into real numbers.
//!
//! Every layer below this has been verified in isolation — the container
//! parses, the loader reads the right bytes, ggml links. This checks that the
//! composition is correct, which is the part unit tests cannot see: bytes read
//! from a real 144 GiB model, dequantized by ggml's own kernel for the type
//! they were actually stored in, coming out as plausible weights.
//!
//! "Plausible" is doing real work here. A wrong offset, a mis-sized block or a
//! type mix-up usually still *produces* numbers — they are simply garbage.
//! Trained neural-network weights have a recognisable shape: finite, centred
//! near zero, small in magnitude, and not all identical. Garbage rarely is.

use std::path::PathBuf;

use bigtea_ggml::{available, dequantize, type_info};
use bigtea_model::Model;

const DEFAULT_PATH: &str =
    r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf";

fn model() -> Option<Model> {
    let p = std::env::var("BIGTEA_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_PATH));
    p.exists().then(|| Model::open_split(&p).expect("open model"))
}

/// Statistics that separate trained weights from garbage.
struct Stats {
    finite: usize,
    total: usize,
    mean: f64,
    max_abs: f32,
    distinct_sample: usize,
}

fn stats(values: &[f32]) -> Stats {
    let finite = values.iter().filter(|v| v.is_finite()).count();
    let sum: f64 = values.iter().filter(|v| v.is_finite()).map(|&v| v as f64).sum();
    let max_abs = values
        .iter()
        .filter(|v| v.is_finite())
        .fold(0f32, |m, &v| m.max(v.abs()));
    // Cheap proxy for "not a constant buffer".
    let mut sample: Vec<u32> = values.iter().take(512).map(|v| v.to_bits()).collect();
    sample.sort_unstable();
    sample.dedup();
    Stats {
        finite,
        total: values.len(),
        mean: if finite > 0 { sum / finite as f64 } else { 0.0 },
        max_abs,
        distinct_sample: sample.len(),
    }
}

#[test]
fn dequantizes_a_real_quantized_tensor_into_plausible_weights() {
    if !available() {
        eprintln!("skipping: built without ggml (set GGML_LIB_DIR)");
        return;
    }
    let Some(m) = model() else {
        eprintln!("skipping: no model on disk");
        return;
    };

    // A genuinely quantized tensor, not an f32 one -- the quantized path is
    // what carries 137 of the model's 144 GiB.
    let Some(name) = m
        .tensor_names()
        .find(|n| {
            m.location(n)
                .is_some_and(|l| l.ty.is_quantized() && l.size > 0)
                && m.is_available(n).unwrap_or(false)
        })
        .map(str::to_string)
    else {
        eprintln!("skipping: no quantized tensor available");
        return;
    };

    let loc = m.location(&name).expect("located").clone();
    let raw = m.read_tensor(&name).expect("read tensor");
    let elements = loc.elements();

    let info = type_info(loc.ty).expect("ggml knows the type");
    let values = dequantize(loc.ty, &raw, elements as usize).expect("dequantize");

    assert_eq!(values.len(), elements as usize);

    let s = stats(&values);
    eprintln!(
        "{name}: {} [{}] {} elements -> mean {:.5}, max|w| {:.4}, {} distinct in first 512",
        info.name, loc.ty, s.total, s.mean, s.max_abs, s.distinct_sample
    );

    // Every value must be a number. NaN or infinity here means a misread.
    assert_eq!(s.finite, s.total, "{name} produced non-finite values");
    // Trained weights are not a constant buffer; a wrong offset often is.
    assert!(s.distinct_sample > 8, "{name} looks like a constant buffer");
    // Weights of a trained transformer sit near zero with small magnitude.
    // Garbage from a bad offset is typically orders of magnitude larger.
    assert!(
        s.max_abs < 100.0,
        "{name} has implausibly large weights (max |w| = {})",
        s.max_abs
    );
    assert!(
        s.mean.abs() < 1.0,
        "{name} has an implausible mean ({:.4})",
        s.mean
    );
}

#[test]
fn dequantization_is_deterministic() {
    // The same bytes must always give the same numbers. If they do not, the
    // kernel is reading uninitialised memory somewhere.
    if !available() {
        eprintln!("skipping: built without ggml");
        return;
    }
    let Some(m) = model() else {
        eprintln!("skipping: no model on disk");
        return;
    };
    let Some(name) = m
        .tensor_names()
        .find(|n| {
            m.location(n).is_some_and(|l| l.ty.is_quantized())
                && m.is_available(n).unwrap_or(false)
        })
        .map(str::to_string)
    else {
        return;
    };

    let loc = m.location(&name).expect("located").clone();
    let raw = m.read_tensor(&name).expect("read");
    let a = dequantize(loc.ty, &raw, loc.elements() as usize).expect("first");
    let b = dequantize(loc.ty, &raw, loc.elements() as usize).expect("second");
    assert_eq!(a, b, "dequantization is not deterministic");
}

#[test]
fn a_corrupted_buffer_does_not_silently_produce_the_same_weights() {
    // Guards against the failure that would be hardest to notice: bytes being
    // ignored, so wrong data still yields plausible-looking output.
    if !available() {
        eprintln!("skipping: built without ggml");
        return;
    }
    let Some(m) = model() else {
        eprintln!("skipping: no model on disk");
        return;
    };
    let Some(name) = m
        .tensor_names()
        .find(|n| {
            m.location(n).is_some_and(|l| l.ty.is_quantized())
                && m.is_available(n).unwrap_or(false)
        })
        .map(str::to_string)
    else {
        return;
    };

    let loc = m.location(&name).expect("located").clone();
    let raw = m.read_tensor(&name).expect("read");
    let original = dequantize(loc.ty, &raw, loc.elements() as usize).expect("original");

    let mut tampered = raw.clone();
    for b in tampered.iter_mut().take(64) {
        *b = !*b;
    }
    let changed = dequantize(loc.ty, &tampered, loc.elements() as usize).expect("tampered");

    assert_ne!(
        original, changed,
        "flipping bytes changed nothing -- the input is not actually being read"
    );
}
