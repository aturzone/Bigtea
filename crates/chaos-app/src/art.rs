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
