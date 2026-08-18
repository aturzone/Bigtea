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
include!("../../chaos-arch/src/logo_bitmap.rs");

// The GUI master: `HI_W`, `HI_H`, `HI` -- ink coverage at 256x256.
include!("logo_hi.rs");

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
/// **This is why the logo stopped looking like a blob.** The window used to
/// draw the 56x56 terminal bitmap, thresholded to one bit and stretched by
/// `StretchDIBits` to 30 pixels -- a nearest-neighbour scale of an already
/// aliased source. Here a 256x256 antialiased master is box-filtered to the
/// exact size wanted, so every output pixel averages at least a 4x4 footprint
/// and the rays survive.
///
/// Returns coverage, not colour: 0 is paper, 255 is full ink. The caller blends
/// its own foreground through it, which is how the same mark works on a light
/// page and a dark one without a second asset.
pub fn logo_scaled(n: usize) -> Vec<u8> {
    let n = n.max(1);
    let mut out = vec![0u8; n * n];
    for y in 0..n {
        let y0 = y * HI_H / n;
        let y1 = (((y + 1) * HI_H) / n).max(y0 + 1).min(HI_H);
        for x in 0..n {
            let x0 = x * HI_W / n;
            let x1 = (((x + 1) * HI_W) / n).max(x0 + 1).min(HI_W);
            let mut sum = 0u32;
            let mut count = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    sum += u32::from(HI[sy * HI_W + sx]);
                    count += 1;
                }
            }
            out[y * n + x] = (sum / count.max(1)) as u8;
        }
    }
    out
}

/// A stroke in a glyph, in a 0..1 square, to be scaled at paint time.
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
    /// The high-resolution master must actually be high resolution, and must
    /// carry the mid-tones that thresholding destroys.
    #[test]
    fn the_gui_master_is_antialiased() {
        assert_eq!(HI.len(), HI_W * HI_H);
        assert!(HI_W >= 256, "the GUI master is only {HI_W}px");
        let mid = HI.iter().filter(|&&v| (24..232).contains(&v)).count();
        assert!(
            mid > HI.len() / 200,
            "only {mid} pixels are partial ink; this master has been              thresholded and will look like the 56px one did"
        );
    }

    /// Scaling must preserve the mark: ink at the size the window draws it.
    #[test]
    fn the_mark_survives_being_scaled_down() {
        for n in [16, 24, 30, 32, 48, 64] {
            let m = logo_scaled(n);
            assert_eq!(m.len(), n * n);
            let ink: u32 = m.iter().map(|&v| u32::from(v)).sum();
            let mean = ink / (n * n) as u32;
            assert!(
                (8..200).contains(&mean),
                "at {n}px the mark averages {mean}/255 -- it has become a                  blank square or a solid block"
            );
            // A box filter over an antialiased master has to produce greys; if
            // it does not, something has thresholded on the way through.
            let partial = m.iter().filter(|&&v| (24..232).contains(&v)).count();
            assert!(
                partial > 0,
                "at {n}px every pixel is pure ink or pure paper"
            );
        }
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
