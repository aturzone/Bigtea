//! The artwork, all of it vector-derived and strictly two-colour.
//!
//! The logo comes from `assets/logo.svg` by way of `tools/rasterise-logo.py`,
//! which already produces the bytes the terminal banner prints. **Included from
//! the other crate rather than copied**: two committed copies of a generated
//! array is two things to regenerate, and the second one is always the one
//! nobody remembers.
//!
//! Everything else here is drawn from coordinates at paint time rather than
//! shipped as pixels, so it stays sharp at any window size and cannot introduce
//! a colour. That is the whole palette: `#000000` and `#FFFFFF`, nothing
//! between. A design with two values has no greys to get wrong, and it is what
//! Atur asked for.

// The generated luminance bitmap: `LOGO_W`, `LOGO_H`, `LOGO`.
// Three levels up, not two: this file is `gui/app/src/`, so `../../..` is the
// workspace root. It was `../../chaos-arch/...` when both crates sat in a flat
// `crates/`, and the move to buckets put one more directory between them.
include!("../../../core/arch/src/logo_bitmap.rs");

// The mark's outlines: `Poly`, `POLYS`. Geometry, not pixels.
include!("logo_vector.rs");

/// Threshold the luminance ramp to pure black and white.
///
/// The rasteriser antialiases, which is right for a terminal printing shaded
/// half-blocks and wrong here: a two-colour design with a grey fringe is a
/// three-colour design. Mid-grey is the cut, so a pixel the rasteriser thought
/// was more ink than paper becomes ink.
pub fn logo_mono() -> Vec<bool> {
    LOGO.iter().map(|&l| l < 128).collect()
}

pub fn logo_size() -> (usize, usize) {
    (LOGO_W, LOGO_H)
}

/// The mark at `n` pixels square, as ink coverage.
///
/// **Filled from outlines every time, never resampled.** The window used to
/// draw a 56x56 bitmap stretched to 30 pixels, and then a 256x256 one filtered
/// down; both are a bitmap at heart and both lose something at a size they were
/// not made for. `logo_vector.rs` carries the flattened paths instead, so this
/// scan-converts the actual geometry at whatever size is asked for and there is
/// no intermediate image anywhere.
///
/// Returns coverage: 0 is paper, 255 is full ink. The caller blends its own
/// foreground through it, which is how one mark works on a light page and a
/// dark one without a second asset.
pub fn logo_coverage(n: usize) -> Vec<u8> {
    let n = n.max(1);
    // 8x8 subsamples per output pixel. The scanline fill below is hard-edged,
    // so supersampling is the only source of antialiasing, and the number of
    // subsamples is the number of grey levels an edge can take.
    //
    // **16 levels was not enough for this mark.** It is a sun of two dozen fine
    // rays, and at 44px a ray is about one pixel wide -- so every ray edge
    // landed on one of sixteen steps and the whole mark read as notched. Atur's
    // report was "low quality logo on top", twice. 64 levels is smooth at that
    // size; the cost is four times a rasterisation that happens once and is
    // then cached by `logo_scaled`.
    const SS: usize = 8;
    let w = n * SS;
    let mut grid = vec![0u8; w * w];

    for poly in POLYS {
        fill_polygon(&mut grid, w, poly, 255u8 * u8::from(poly.ink));
    }

    // Box-downsample the subsample grid to the requested size.
    let mut out = vec![0u8; n * n];
    for y in 0..n {
        for x in 0..n {
            let mut sum = 0u32;
            for sy in y * SS..(y + 1) * SS {
                for sx in x * SS..(x + 1) * SS {
                    sum += u32::from(grid[sy * w + sx]);
                }
            }
            out[y * n + x] = (sum / (SS * SS) as u32) as u8;
        }
    }
    out
}

/// Scan-convert one closed polygon into `grid`, nonzero winding.
///
/// The same rule the SVG uses and the same one `tools/rasterise-logo.py`
/// implements: a span is inside when the accumulated winding is not zero, which
/// is what leaves the holes in the mark actually hollow. An even-odd fill would
/// look correct on most of these paths and wrong on the ones that overlap.
fn fill_polygon(grid: &mut [u8], w: usize, poly: &Poly, value: u8) {
    let scale = w as f32;
    // **Every subpath of this path, in one scanline pass.** SVG fills a path as
    // a single region, and with no `fill-rule` the default is nonzero -- so a
    // subpath winding against the first cuts a *hole*. Filling the subpaths one
    // at a time fills those holes, and this mark's eyes are a hole in a white
    // shape: they vanished. Collecting crossings from all contours before
    // resolving the winding is what makes a hole a hole.
    if poly.contours.iter().all(|c| c.len() < 3) {
        return;
    }
    // Row range this path can touch, so a small one does not walk the whole
    // bitmap.
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for &(_, y) in poly.contours.iter().flat_map(|c| c.iter()) {
        lo = lo.min(y);
        hi = hi.max(y);
    }
    let y0 = ((lo * scale).floor().max(0.0)) as usize;
    let y1 = ((hi * scale).ceil().min(scale)) as usize;

    let mut xs: Vec<(f32, i32)> = Vec::with_capacity(16);
    for row in y0..y1.min(w) {
        let yc = row as f32 + 0.5;
        xs.clear();
        for pts in poly.contours {
            if pts.len() < 3 {
                continue;
            }
            for i in 0..pts.len() {
                let (x0, ay) = pts[i];
                let (x1, by) = pts[(i + 1) % pts.len()];
                let (ay, by) = (ay * scale, by * scale);
                // Half-open in y, so a vertex shared by two edges is counted
                // once and horizontal edges drop out entirely.
                if (ay <= yc && by > yc) || (by <= yc && ay > yc) {
                    let t = (yc - ay) / (by - ay);
                    let x = (x0 + t * (x1 - x0)) * scale;
                    xs.push((x, if by > ay { 1 } else { -1 }));
                }
            }
        }
        if xs.len() < 2 {
            continue;
        }
        xs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut wind = 0;
        for k in 0..xs.len() - 1 {
            wind += xs[k].1;
            if wind == 0 {
                continue;
            }
            let a = (xs[k].0 + 0.5).max(0.0) as usize;
            let b = ((xs[k + 1].0 + 0.5).max(0.0) as usize).min(w);
            for px in a..b {
                grid[row * w + px] = value;
            }
        }
    }
}

/// The mark at `n` pixels, cached.
///
/// The rail repaints on every generated token, and rasterising 44 polygons each
/// time would be work done sixteen times a second for a picture that has not
/// changed. One size is in flight at once, so a single slot is the whole cache.
pub fn logo_scaled(n: usize) -> Vec<u8> {
    use std::sync::Mutex;
    static CACHE: Mutex<Option<(usize, Vec<u8>)>> = Mutex::new(None);
    let n = n.max(1);
    let mut c = match CACHE.lock() {
        Ok(c) => c,
        // A poisoned lock here means another thread panicked mid-render; the
        // mark is not worth propagating that, so draw it again.
        Err(e) => e.into_inner(),
    };
    if let Some((have, bits)) = c.as_ref() {
        if *have == n {
            return bits.clone();
        }
    }
    let bits = logo_coverage(n);
    *c = Some((n, bits.clone()));
    bits
}

/// A stroke in a glyph, in a 0..1 square, to be scaled at paint time./// A stroke in a glyph, in a 0..1 square, to be scaled at paint time.
pub type Stroke = (f32, f32, f32, f32);

/// Line art for the app's controls, as coordinates rather than pixels.
///
/// Each is a set of strokes in a unit square. Drawn with a white pen on black,
/// they read at any size and add no third value to the palette.
pub mod glyph {
    use super::Stroke;

    /// A right-pointing triangle outline: run.
    pub const PLAY: &[Stroke] = &[
        (0.25, 0.15, 0.85, 0.5),
        (0.85, 0.5, 0.25, 0.85),
        (0.25, 0.85, 0.25, 0.15),
    ];

    /// A square: stop.
    pub const STOP: &[Stroke] = &[
        (0.25, 0.25, 0.75, 0.25),
        (0.75, 0.25, 0.75, 0.75),
        (0.75, 0.75, 0.25, 0.75),
        (0.25, 0.75, 0.25, 0.25),
    ];

    /// A downward arrow into a tray: fetch.
    pub const DOWNLOAD: &[Stroke] = &[
        (0.5, 0.15, 0.5, 0.62),
        (0.28, 0.42, 0.5, 0.64),
        (0.72, 0.42, 0.5, 0.64),
        (0.2, 0.82, 0.8, 0.82),
    ];

    /// Three sliders: settings.
    pub const GEAR: &[Stroke] = &[
        (0.15, 0.3, 0.85, 0.3),
        (0.15, 0.5, 0.85, 0.5),
        (0.15, 0.7, 0.85, 0.7),
        (0.35, 0.22, 0.35, 0.38),
        (0.6, 0.42, 0.6, 0.58),
        (0.3, 0.62, 0.3, 0.78),
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_logo_has_both_values() {
        let m = logo_mono();
        assert_eq!(m.len(), LOGO_W * LOGO_H);
        let ink = m.iter().filter(|&&b| b).count();
        assert!(ink > 0, "thresholding left no ink at all");
        assert!(ink < m.len(), "thresholding left no paper at all");
    }

    /// Strokes stay inside the unit square, or they paint over their neighbours.
    /// The geometry has to be geometry: enough polygons, enough points, and
    /// both kinds of fill. A file with only ink paths would draw a solid disc.
    #[test]
    fn the_outlines_are_present_and_have_both_fills() {
        assert!(POLYS.len() > 20, "only {} polygons", POLYS.len());
        let pts: usize = POLYS
            .iter()
            .flat_map(|p| p.contours.iter())
            .map(|c| c.len())
            .sum();
        assert!(pts > 2000, "only {pts} points; the flattener has drifted");
        assert!(POLYS.iter().any(|p| p.ink), "nothing is inked");
        assert!(
            POLYS.iter().any(|p| !p.ink),
            "nothing is paper -- the holes in the mark would fill in"
        );
        // **At least one path has more than one contour, or the hole logic is
        // untested by the art itself.** Path 12 of this mark is a white shape
        // with a hole, and that hole is the eyes: filling its subpaths
        // separately erased them.
        assert!(
            POLYS.iter().any(|p| p.contours.len() > 1),
            "no path has a second contour; the nonzero-winding hole is untested"
        );
        for p in POLYS {
            for c in p.contours {
                assert!(c.len() >= 3, "a contour with {} points", c.len());
            }
            for &(x, y) in p.contours.iter().flat_map(|c| c.iter()) {
                // The unit square is the *ink* box, and the art's white backing
                // plate is deliberately larger than it -- so geometry outside
                // is normal and the rasteriser clips it. What would not be
                // normal is a coordinate far enough out to mean the
                // normalisation is wrong rather than the plate generous.
                assert!(
                    (-0.5..=1.5).contains(&x) && (-0.5..=1.5).contains(&y),
                    "({x}, {y}) is nowhere near the mark; the ink box is wrong"
                );
            }
        }
        // The inked geometry itself has to sit inside, or the mark is off-centre.
        for p in POLYS.iter().filter(|p| p.ink) {
            for &(x, y) in p.contours.iter().flat_map(|c| c.iter()) {
                assert!(
                    (-0.02..=1.02).contains(&x) && (-0.02..=1.02).contains(&y),
                    "an inked point at ({x}, {y}) falls outside the mark's own box"
                );
            }
        }
    }

    /// **Resolution independence, checked rather than claimed.** Rendering at
    /// 64 and box-filtering to 32 must land close to rendering at 32 directly.
    /// A bitmap master resampled to both would agree trivially; outlines filled
    /// at each size agreeing is what says the geometry is doing the work.
    #[test]
    fn the_mark_is_the_same_shape_at_any_size() {
        let small = logo_coverage(32);
        let big = logo_coverage(64);
        let mut folded = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                let mut sum = 0u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        sum += u32::from(big[(y * 2 + dy) * 64 + x * 2 + dx]);
                    }
                }
                folded[y * 32 + x] = (sum / 4) as u8;
            }
        }
        let diff: u32 = small
            .iter()
            .zip(&folded)
            .map(|(a, b)| u32::from(a.abs_diff(*b)))
            .sum();
        let mean = diff / (32 * 32);
        assert!(
            mean < 24,
            "the shape differs by {mean}/255 on average between sizes"
        );
    }

    /// A size of zero must not panic; the rail computes it from window metrics.
    #[test]
    fn a_degenerate_size_is_survivable() {
        assert_eq!(logo_scaled(0).len(), 1);
    }

    #[test]
    fn glyphs_are_inside_their_box() {
        for (name, set) in [
            ("PLAY", glyph::PLAY),
            ("STOP", glyph::STOP),
            ("DOWNLOAD", glyph::DOWNLOAD),
            ("GEAR", glyph::GEAR),
        ] {
            assert!(!set.is_empty(), "{name} has no strokes");
            for &(x0, y0, x1, y1) in set {
                for v in [x0, y0, x1, y1] {
                    assert!(
                        (0.0..=1.0).contains(&v),
                        "{name} leaves the unit square: {v}"
                    );
                }
            }
        }
    }
}
