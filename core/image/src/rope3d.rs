//! Three-axis rotary positions for the denoiser, and the table they become.
//!
//! # Why an image needs three position axes
//!
//! A language model's token has one position. A denoiser's token is either a
//! word or a *patch of an image*, and a patch has a row and a column that both
//! matter. Ideogram 4 gives every token three coordinates — `(t, y, x)` — and
//! splits the rotary frequencies between them.
//!
//! Text tokens get `(i, i, i)`: all three axes agree, which makes the rotation
//! for a word identical to ordinary 1-D RoPE. Image tokens get
//! `(OFFSET, OFFSET + y, OFFSET + x)` with `OFFSET = 65536`, far past any text
//! length, so no patch can be confused with a word.
//!
//! # The interleave, which is the part that is easy to get wrong
//!
//! The frequencies are **not** split into three contiguous blocks. They are
//! dealt out round-robin: frequency `f` belongs to axis `f % 3`, and only while
//! `f < 3 * section[axis]`. Past that every remaining frequency falls back to
//! the time axis. With sections `[24, 20, 20]` over 128 frequency pairs that is
//! 20 for `y`, 20 for `x`, and **88** for `t` — not 24, because `t` also
//! collects everything above 60.
//!
//! A contiguous split gives a table of the same shape, full of finite numbers,
//! and an image whose left half does not know about its right half. This is the
//! reason this module exists separately and is tested on its own.
//!
//! # What comes out
//!
//! A flat `[2, 2, head_dim / 2, positions]` buffer in ggml order: a 2x2 rotation
//! matrix `[cos, -sin, sin, cos]` for every (position, frequency-pair). The
//! denoiser multiplies rather than calling `ggml_rope`, because no ggml rope
//! mode implements this interleave.

/// Where image positions start, so that no patch coordinate can collide with a
/// text position. Ideogram's own constant.
pub const IMAGE_POSITION_OFFSET: f32 = 65536.0;

/// Frequencies dealt to `(t, y, x)` before the remainder falls back to `t`.
pub const MROPE_SECTION: [usize; 3] = [24, 20, 20];

/// The rotary base. Large because the position numbers are large.
pub const ROPE_THETA: f32 = 5_000_000.0;

/// Which axis owns frequency pair `f`, given the sections.
///
/// Separate from the table builder so the interleave can be asserted directly:
/// it is the one rule here that a wrong implementation still runs with.
pub fn axis_of(freq: usize, half_dim: usize, section: [usize; 3]) -> usize {
    let axis = freq % 3;
    // Axis 0 is the default and also the fallback, so it is never "past its
    // section" -- the check only demotes axes 1 and 2.
    if axis == 0 {
        return 0;
    }
    let length = (section[axis] * 3).min(half_dim);
    if freq < length {
        axis
    } else {
        0
    }
}

/// The `(t, y, x)` coordinate of every token: text first, then the image grid
/// in row-major order.
pub fn positions(context_len: usize, grid_h: usize, grid_w: usize) -> Vec<[f32; 3]> {
    let mut ids = Vec::with_capacity(context_len + grid_h * grid_w);
    for i in 0..context_len {
        let p = i as f32;
        ids.push([p, p, p]);
    }
    for y in 0..grid_h {
        for x in 0..grid_w {
            ids.push([
                IMAGE_POSITION_OFFSET,
                IMAGE_POSITION_OFFSET + y as f32,
                IMAGE_POSITION_OFFSET + x as f32,
            ]);
        }
    }
    ids
}

/// The rotation table for one axis: `angle = pos * theta^(-2f / head_dim)`.
fn angles(pos: &[f32], head_dim: usize, theta: f32) -> Vec<Vec<f32>> {
    let half = head_dim / 2;
    pos.iter()
        .map(|p| {
            (0..half)
                .map(|f| {
                    // The reference writes this as a linspace from 0 to
                    // (dim - 2)/dim over `half` points, which simplifies
                    // exactly to 2f/dim. Asserted in the tests.
                    let exponent = 2.0 * f as f32 / head_dim as f32;
                    *p * theta.powf(-exponent)
                })
                .collect()
        })
        .collect()
}

/// Build the `[2, 2, head_dim / 2, positions]` table the attention multiplies by.
///
/// Layout, fastest dimension first: `[cos, -sin, sin, cos]` per frequency pair,
/// then frequency pairs, then positions.
// `p` indexes three parallel tables and `f` the frequency inside each; neither
// is an iteration over one slice that clippy could rewrite.
#[allow(clippy::needless_range_loop)]
pub fn table(ids: &[[f32; 3]], head_dim: usize, theta: f32, section: [usize; 3]) -> Vec<f32> {
    assert!(head_dim % 2 == 0, "head_dim must be even");
    let half = head_dim / 2;

    // One angle table per axis, then pick per frequency.
    let per_axis: Vec<Vec<Vec<f32>>> = (0..3)
        .map(|a| {
            let pos: Vec<f32> = ids.iter().map(|id| id[a]).collect();
            angles(&pos, head_dim, theta)
        })
        .collect();

    let mut out = vec![0.0f32; ids.len() * half * 4];
    for (p, _) in ids.iter().enumerate() {
        for f in 0..half {
            let a = axis_of(f, half, section);
            let angle = per_axis[a][p][f];
            let (s, c) = angle.sin_cos();
            let base = (p * half + f) * 4;
            out[base] = c;
            out[base + 1] = -s;
            out[base + 2] = s;
            out[base + 3] = c;
        }
    }
    out
}

/// `0` for a text token and `1` for an image token — the index into the
/// denoiser's two-row `embed_image_indicator`.
///
/// Two learned vectors, added to every token, telling the model which half of
/// the sequence it is looking at. Omitting them leaves the shapes correct.
pub fn image_indicator(context_len: usize, image_tokens: usize) -> Vec<i32> {
    let mut v = vec![1i32; context_len + image_tokens];
    for slot in v.iter_mut().take(context_len) {
        *slot = 0;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The round-robin interleave, which a contiguous split would get wrong
    /// while still producing a correctly shaped table.
    #[test]
    fn frequencies_are_dealt_round_robin_then_fall_back_to_time() {
        let half = 128;
        // Below 60, strict round robin: 0 -> t, 1 -> y, 2 -> x.
        assert_eq!(axis_of(0, half, MROPE_SECTION), 0);
        assert_eq!(axis_of(1, half, MROPE_SECTION), 1);
        assert_eq!(axis_of(2, half, MROPE_SECTION), 2);
        assert_eq!(axis_of(57, half, MROPE_SECTION), 0);
        assert_eq!(axis_of(58, half, MROPE_SECTION), 1);
        assert_eq!(axis_of(59, half, MROPE_SECTION), 2);
        // At and above 3 * 20 = 60, y and x are exhausted and t takes over.
        assert_eq!(axis_of(60, half, MROPE_SECTION), 0);
        assert_eq!(axis_of(61, half, MROPE_SECTION), 0, "y is past its section");
        assert_eq!(axis_of(62, half, MROPE_SECTION), 0, "x is past its section");
        assert_eq!(axis_of(127, half, MROPE_SECTION), 0);

        // The counts are 20 / 20 / 88 -- not 24 / 20 / 20, which is what
        // reading the section list as a partition would give.
        let mut n = [0usize; 3];
        for f in 0..half {
            n[axis_of(f, half, MROPE_SECTION)] += 1;
        }
        assert_eq!(n, [88, 20, 20], "t also collects every frequency above 60");
        assert_eq!(n.iter().sum::<usize>(), half);
    }

    /// The exponent really is `2f / head_dim`, which is what lets the linspace
    /// in the reference be written as a division here.
    #[test]
    fn the_frequency_exponent_matches_the_reference_linspace() {
        let (dim, half) = (256usize, 128usize);
        // linspace(0, (dim - 2) / dim, half): step = ((dim - 2)/dim) / (half - 1).
        let step = ((dim as f64 - 2.0) / dim as f64) / (half as f64 - 1.0);
        for f in 0..half {
            let from_linspace = step * f as f64;
            let ours = 2.0 * f as f64 / dim as f64;
            assert!((from_linspace - ours).abs() < 1e-12, "f = {f}");
        }
    }

    /// Text positions rise by one; image positions live far above them.
    #[test]
    fn image_positions_cannot_collide_with_text_positions() {
        let ids = positions(3, 2, 2);
        assert_eq!(ids.len(), 3 + 4);
        assert_eq!(ids[0], [0.0, 0.0, 0.0]);
        assert_eq!(ids[2], [2.0, 2.0, 2.0], "text agrees on all three axes");
        // Row-major over the grid, and every image token shares one t.
        assert_eq!(ids[3], [65536.0, 65536.0, 65536.0]);
        assert_eq!(ids[4], [65536.0, 65536.0, 65537.0], "x moves first");
        assert_eq!(ids[5], [65536.0, 65537.0, 65536.0], "then y");
        assert_eq!(ids[6], [65536.0, 65537.0, 65537.0]);
        // No text length this side of 65536 can reach an image position.
        assert!(ids[..3].iter().all(|p| p[0] < IMAGE_POSITION_OFFSET));
    }

    /// The table is a stack of 2x2 rotation matrices, in the documented order.
    #[test]
    fn the_table_holds_rotation_matrices_in_ggml_order() {
        let head_dim = 4;
        let ids = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let t = table(&ids, head_dim, 10000.0, MROPE_SECTION);
        assert_eq!(t.len(), 2 * (head_dim / 2) * 4);

        // Position 0 is the identity rotation at every frequency: cos 0 = 1.
        assert_eq!(&t[0..4], &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(&t[4..8], &[1.0, 0.0, 0.0, 1.0]);

        // Position 1, frequency 0: exponent 0, so the angle is exactly 1 rad.
        let (s, c) = 1.0f32.sin_cos();
        let base = (head_dim / 2) * 4;
        assert!((t[base] - c).abs() < 1e-6);
        assert!((t[base + 1] + s).abs() < 1e-6, "the -sin slot");
        assert!((t[base + 2] - s).abs() < 1e-6, "the +sin slot");
        assert!((t[base + 3] - c).abs() < 1e-6);
        // Which is a rotation matrix: its determinant is 1.
        let det = t[base] * t[base + 3] - t[base + 1] * t[base + 2];
        assert!((det - 1.0).abs() < 1e-6, "{det}");
    }

    /// A text-only sequence rotates identically on all three axes, so it is
    /// ordinary 1-D RoPE — which is what makes the text encoder reusable.
    #[test]
    fn text_only_positions_reduce_to_one_dimensional_rope() {
        let ids = positions(5, 0, 0);
        let three_axis = table(&ids, 64, ROPE_THETA, MROPE_SECTION);
        // Force every frequency onto the time axis and compare.
        let one_axis = table(&ids, 64, ROPE_THETA, [64, 0, 0]);
        assert_eq!(three_axis, one_axis, "text ids agree on t, y and x");
    }

    /// The indicator marks text and image, and is the length of the sequence.
    #[test]
    fn the_indicator_splits_text_from_image() {
        let v = image_indicator(2, 3);
        assert_eq!(v, vec![0, 0, 1, 1, 1]);
        // The unconditional pass has no text at all, which is the case that
        // would silently index a zero-length prefix.
        assert_eq!(image_indicator(0, 3), vec![1, 1, 1]);
    }
}
