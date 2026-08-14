//! Checking that a container's *values* are finite — llama.cpp's
//! `--check-tensors`.
//!
//! # What this catches that opening the file does not
//!
//! Container parsing validates *structure*: names, offsets, shapes, that the
//! data section is long enough. All of that can be perfect while the numbers
//! inside are ruined — a truncated download resumed against a changed file, a
//! bad disk, a quantiser that divided by zero on an all-zero block. The
//! symptom is not a crash. The first `NaN` reaching a softmax makes every
//! probability `NaN`, `argmax` returns index 0, and the model emits the same
//! token forever. That reads as "the model is broken", and the search starts
//! in the forward pass rather than in the file.
//!
//! # Why the check is per-type and not "scan for NaN"
//!
//! Most of a quantised container is *not* floats. A `Q4_K` block is 144 bytes
//! of packed 4-bit integers and scales; scanning it as `f32` would interpret
//! index bytes as exponents and report garbage. Only the parts that are
//! genuinely floating-point can be checked, and for the quantised types those
//! are the **f16 scale fields** — which is exactly where a bad quantise shows
//! up, because a zero or infinite scale is what makes a whole block explode.
//!
//! So this is honest about its reach: it reports how many tensors it could
//! check *fully*, how many only by their scales, and how many not at all.

use crate::{GgmlType, Model};

/// What a validation pass found.
#[derive(Debug, Default)]
pub struct Report {
    /// Tensors whose every value was checked.
    pub checked_fully: usize,
    /// Quantised tensors where only the block scales could be checked.
    pub checked_scales: usize,
    /// Types this build cannot inspect. Counted, not ignored.
    pub unchecked: usize,
    /// `(tensor name, what was wrong)`, first few only.
    pub problems: Vec<(String, String)>,
    pub bytes_read: u64,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.problems.is_empty()
    }
}

/// Type ids, matching ggml's enum.
const F32: u32 = 0;
const F16: u32 = 1;
const BF16: u32 = 30;

/// Block layout of the quantised types this can reach: `(bytes per block,
/// values per block, byte offsets of the f16 scale fields within a block)`.
///
/// Taken from ggml's `block_q*` structs. A type absent here is counted as
/// unchecked rather than guessed at — reading the wrong two bytes as a scale
/// would invent failures, and a validator that cries wolf is worse than none.
fn block_layout(ty: u32) -> Option<(usize, usize, &'static [usize])> {
    Some(match ty {
        2 => (18, 32, &[0]),    // Q4_0:  d
        3 => (20, 32, &[0, 2]), // Q4_1:  d, m
        6 => (22, 32, &[0]),    // Q5_0:  d
        7 => (24, 32, &[0, 2]), // Q5_1:  d, m
        8 => (34, 32, &[0]),    // Q8_0:  d
        // Q4_K and Q5_K carry `d` and `dmin` at the START of the block; Q6_K
        // carries its single `d` at the END. Read ggml-common.h rather than
        // assuming a shape: putting Q4_K's scales at the tail made this
        // validator report a *healthy* Qwen2 container as damaged, because
        // packed 4-bit quants at offset 140 happened to read as inf.
        12 => (144, 256, &[0, 2]), // block_q4_K: d, dmin, scales[12], qs[128]
        13 => (176, 256, &[0, 2]), // block_q5_K: d, dmin, scales[12], qh, qs
        14 => (210, 256, &[208]),  // block_q6_K: ql, qh, scales[16], d
        _ => return None,
    })
}

/// Interpret two little-endian bytes as an f16 and say whether it is finite.
///
/// Written out rather than pulled from a crate: the exponent/mantissa split is
/// six lines and a dependency here would be the only one in this crate.
fn f16_is_finite(lo: u8, hi: u8) -> bool {
    let bits = u16::from_le_bytes([lo, hi]);
    // All-ones exponent is inf or NaN in IEEE 754 half precision.
    (bits >> 10) & 0x1f != 0x1f
}

/// The smallest unit a scan may start on, in bytes.
///
/// A read that begins mid-block misreads every block after it, so this is what
/// the chunk size must be a multiple of.
fn stride(ty: GgmlType) -> u64 {
    match ty.0 {
        F32 => 4,
        F16 | BF16 => 2,
        t => block_layout(t).map(|(b, _, _)| b as u64).unwrap_or(1),
    }
}

/// Read every tensor and check its values are finite.
///
/// `max_problems` bounds the report: a container that is entirely corrupt would
/// otherwise produce one line per tensor, and the first few say the same thing.
pub fn check(model: &Model, max_problems: usize) -> Report {
    let mut report = Report::default();
    let names: Vec<String> = model.tensor_names().map(str::to_string).collect();

    for name in names {
        let Some(loc) = model.location(&name).cloned() else {
            continue;
        };
        // Read in chunks: a single expert tensor can be gigabytes, and a
        // validator that needs the model resident defeats the point of a
        // runner whose whole design is not holding it.
        //
        // **The chunk must be a whole number of blocks.** 8 MiB is not a
        // multiple of 210 (`Q6_K`) or 144 (`Q4_K`), so the second chunk starts
        // mid-block and every block in it is read at the wrong offset -- which
        // made this report a healthy `token_embd.weight` as damaged at block
        // 246754, i.e. exactly where the first chunk ended. An earlier comment
        // here claimed the misaligned tail was "skipped rather than misread";
        // it is the next chunk's alignment that matters, not the tail's.
        const TARGET: u64 = 8 << 20;
        let unit = stride(loc.ty);
        let chunk = (TARGET / unit).max(1) * unit;
        let mut buf = vec![0u8; chunk.min(loc.size.max(1)) as usize];
        let mut offset = 0u64;
        let mut bad: Option<String> = None;

        while offset < loc.size && bad.is_none() {
            let want = chunk.min(loc.size - offset) as usize;
            let slice = &mut buf[..want];
            if model.read_range_into(&name, offset, slice).is_err() {
                bad = Some(format!("unreadable at byte {offset}"));
                break;
            }
            report.bytes_read += want as u64;
            bad = scan(loc.ty, slice, offset);
            offset += want as u64;
        }

        match (bad, classify(loc.ty)) {
            (Some(why), _) => {
                if report.problems.len() < max_problems {
                    report.problems.push((name, why));
                }
            }
            (None, Reach::Full) => report.checked_fully += 1,
            (None, Reach::Scales) => report.checked_scales += 1,
            (None, Reach::None) => report.unchecked += 1,
        }
    }
    report
}

enum Reach {
    Full,
    Scales,
    None,
}

fn classify(ty: GgmlType) -> Reach {
    match ty.0 {
        F32 | F16 | BF16 => Reach::Full,
        t if block_layout(t).is_some() => Reach::Scales,
        _ => Reach::None,
    }
}

/// Scan one chunk. Returns a description of the first problem found.
///
/// `base` is the chunk's offset in the tensor, so a block boundary that falls
/// mid-chunk is handled by only scanning whole blocks — a partial block at the
/// end is picked up by the next chunk, and `CHUNK` is a multiple of every block
/// size involved only by luck, so the tail is skipped rather than misread.
fn scan(ty: GgmlType, data: &[u8], base: u64) -> Option<String> {
    match ty.0 {
        F32 => {
            for (i, c) in data.chunks_exact(4).enumerate() {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if !v.is_finite() {
                    return Some(format!("{v} at element {}", base / 4 + i as u64));
                }
            }
            None
        }
        F16 | BF16 => {
            // BF16's exponent is in the same top bits as f32's, so the
            // all-ones test is the same shape; for f16 it is the standard one.
            for (i, c) in data.chunks_exact(2).enumerate() {
                let finite = if ty.0 == F16 {
                    f16_is_finite(c[0], c[1])
                } else {
                    u16::from_le_bytes([c[0], c[1]]) & 0x7f80 != 0x7f80
                };
                if !finite {
                    return Some(format!("non-finite at element {}", base / 2 + i as u64));
                }
            }
            None
        }
        t => {
            let (bytes, _, scales) = block_layout(t)?;
            for (b, block) in data.chunks_exact(bytes).enumerate() {
                for &off in scales {
                    if !f16_is_finite(block[off], block[off + 1]) {
                        return Some(format!(
                            "non-finite block scale at block {}",
                            base / bytes as u64 + b as u64
                        ));
                    }
                }
            }
            None
        }
    }
}

/// One line summarising a report, for the runner's header.
pub fn summary(r: &Report) -> String {
    format!(
        "{} tensors fully, {} by block scales, {} not inspectable ({:.2} GiB read)",
        r.checked_fully,
        r.checked_scales,
        r.unchecked,
        r.bytes_read as f64 / (1u64 << 30) as f64
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ty(t: u32) -> GgmlType {
        GgmlType(t)
    }

    #[test]
    fn a_finite_f32_run_is_clean() {
        let mut data = Vec::new();
        for v in [0.0f32, 1.5, -2.25, 1e30] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        assert!(scan(ty(F32), &data, 0).is_none());
    }

    #[test]
    fn a_nan_is_found_and_its_index_is_reported() {
        let mut data = Vec::new();
        for v in [1.0f32, 2.0, f32::NAN] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let e = scan(ty(F32), &data, 0).expect("NaN must be caught");
        assert!(e.contains("element 2"), "{e}");
    }

    #[test]
    fn an_infinity_is_a_problem_too() {
        // Not pedantry: an inf reaching a softmax is as fatal as a NaN, and a
        // quantiser that divides by a zero scale produces inf, not NaN.
        let data = f32::INFINITY.to_le_bytes();
        assert!(scan(ty(F32), &data, 0).is_some());
    }

    #[test]
    fn the_index_is_absolute_across_chunks() {
        // A NaN in the second chunk must not be reported as element 0 -- the
        // whole value of this check is telling you WHERE.
        let data = f32::NAN.to_le_bytes();
        let e = scan(ty(F32), &data, 4096).unwrap();
        assert!(e.contains("element 1024"), "{e}");
    }

    #[test]
    fn f16_halves_are_recognised() {
        // 0x7c00 is +inf in half precision; 0x3c00 is 1.0.
        assert!(f16_is_finite(0x00, 0x3c));
        assert!(!f16_is_finite(0x00, 0x7c));
        // 0xfc00 is -inf -- the value this project writes into attention masks,
        // so it MUST be recognised as non-finite rather than passed over.
        assert!(!f16_is_finite(0x00, 0xfc));
    }

    #[test]
    fn a_q4_k_block_is_checked_by_its_two_scales() {
        // Offsets 0 and 2 -- from ggml-common.h's `block_q4_K`, not from what
        // the layout looks like it ought to be. The first version of this test
        // asserted the tail, agreed with the code, and both were wrong: the
        // validator then called a healthy container damaged. A test written
        // from the same assumption as the code proves only that they match.
        let mut block = vec![0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3c;
        block[2] = 0x00;
        block[3] = 0x3c;
        assert!(scan(ty(12), &block, 0).is_none());
        // An infinite d is exactly what a divide-by-zero quantise leaves.
        block[1] = 0x7c;
        let e = scan(ty(12), &block, 0).unwrap();
        assert!(e.contains("block 0"), "{e}");
    }

    #[test]
    fn an_unknown_quant_is_counted_not_guessed() {
        // Reading the wrong two bytes as a scale would invent failures, and a
        // validator that cries wolf is worse than no validator.
        assert!(block_layout(99).is_none());
        assert!(matches!(classify(ty(99)), Reach::None));
        assert!(scan(ty(99), &[0xff; 64], 0).is_none());
    }

    #[test]
    fn the_summary_says_what_it_could_not_reach() {
        let r = Report {
            checked_fully: 2,
            checked_scales: 3,
            unchecked: 1,
            bytes_read: 1 << 30,
            ..Report::default()
        };
        let s = summary(&r);
        assert!(s.contains("1 not inspectable"), "{s}");
        assert!(r.ok());
    }
}
