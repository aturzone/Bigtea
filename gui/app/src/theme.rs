//! The design tokens, in one place.
//!
//! **Tokens, not literals.** Nothing outside this file names a colour. That is
//! the rule the previous window broke everywhere -- `BLACK` and `WHITE` were
//! spelled out at forty call sites, so the palette could not change without
//! forty edits, and the two that were missed are how a "two-value design" ended
//! up with a grey list box.
//!
//! The values come from Hermes' own `apps/desktop/DESIGN.md` and `styles.css`,
//! which Atur asked this app to follow: near-black `#17171A` rather than pure
//! black, a light ground by default, hairlines in four descending strengths,
//! and one accent used sparingly. The accent itself is Atur's `#0000F2` in
//! place of Hermes' `#0053FD`.
//!
//! This module is plain data and has no Win32 in it, so the palette and the
//! type scale are testable on any machine.

/// A colour as `0x00BBGGRR`, the layout `RGB()` produces and every GDI call
/// expects. The constructor takes `(r, g, b)` in the order everyone writes
/// them, because reversing the two by hand is invisible in a greyscale design
/// and obvious only once a colour is added.
pub type Rgb = u32;

pub const fn rgb(r: u8, g: u8, b: u8) -> Rgb {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16)
}

/// Which way round the palette runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Hermes' default, and this app's: a light ground with near-black text.
    Light,
    Dark,
}

impl Mode {
    pub fn toggled(self) -> Self {
        match self {
            Mode::Light => Mode::Dark,
            Mode::Dark => Mode::Light,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Light => "light",
            Mode::Dark => "dark",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "light" => Some(Mode::Light),
            "dark" => Some(Mode::Dark),
            _ => None,
        }
    }
}

/// Every colour the window may use.
///
/// Named by *role*, not by value: `chrome` rather than `light_grey`, so the
/// dark palette is the same struct with different numbers and no call site
/// knows which one it is drawing with.
#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub mode: Mode,

    /// The page itself.
    pub bg: Rgb,
    /// Navigation rail and the running strip -- the frame around the page.
    pub chrome: Rgb,
    /// A control's resting fill, and a row's hover.
    pub soft: Rgb,
    /// A pressed control, and a selected row.
    pub soft_active: Rgb,

    /// Body text.
    pub fg: Rgb,
    /// Labels, units, and anything supporting the line above it.
    pub fg_secondary: Rgb,
    /// Hints, and text that is deliberately quiet.
    pub fg_tertiary: Rgb,
    /// Text on top of `accent`.
    pub on_accent: Rgb,

    /// The one accent, as a fill.
    pub accent: Rgb,
    /// The accent as *text* or a rule. Identical to `accent` on a light ground;
    /// lightened on a dark one, where `#0000F2` on near-black measures 2.11:1 and
    /// simply cannot be read.
    pub accent_text: Rgb,
    /// A wash of the accent -- the active navigation row's fill.
    pub accent_soft: Rgb,

    /// Hairlines, in descending strength. `stroke_3` is the default divider.
    pub stroke_1: Rgb,
    pub stroke_2: Rgb,
    pub stroke_3: Rgb,
    pub stroke_4: Rgb,

    /// Running, and anything that succeeded.
    pub green: Rgb,
    /// Stopped, failed, destructive.
    pub red: Rgb,
    /// Working, and anything provisional.
    pub yellow: Rgb,
}

/// Hermes' light theme, with `#0000F2` for its `#0053FD`.
pub const LIGHT: Theme = Theme {
    mode: Mode::Light,
    bg: rgb(0xFF, 0xFF, 0xFF),
    // Hermes' sidebar seed is a blue-tinted near-white (`#f3f7ff`); this is the
    // same idea mixed from the accent actually in use.
    chrome: rgb(0xF5, 0xF5, 0xFC),
    soft: rgb(0xED, 0xED, 0xF8),
    soft_active: rgb(0xDE, 0xDE, 0xF4),

    // Not `#000000`. Hermes' foreground is `#17171A`, and the difference is the
    // difference between text and a hole in the screen.
    fg: rgb(0x17, 0x17, 0x1A),
    fg_secondary: rgb(0x5A, 0x5A, 0x63),
    fg_tertiary: rgb(0x8A, 0x8A, 0x93),
    on_accent: rgb(0xFF, 0xFF, 0xFF),

    accent: rgb(0x00, 0x00, 0xF2),
    accent_text: rgb(0x00, 0x00, 0xF2),
    accent_soft: rgb(0xE7, 0xE7, 0xFE),

    stroke_1: rgb(0xC4, 0xC4, 0xD2),
    stroke_2: rgb(0xD8, 0xD8, 0xE4),
    stroke_3: rgb(0xE7, 0xE7, 0xF0),
    stroke_4: rgb(0xF1, 0xF1, 0xF7),

    green: rgb(0x1F, 0x8A, 0x65),
    red: rgb(0xCF, 0x2D, 0x56),
    yellow: rgb(0xC0, 0x85, 0x32),
};

/// The dark palette. Ground values are Hermes' own dark seeds -- chrome
/// `#0D0D0E`, sidebar `#0A0A0B`, card `#161618`.
pub const DARK: Theme = Theme {
    mode: Mode::Dark,
    bg: rgb(0x0D, 0x0D, 0x0E),
    chrome: rgb(0x0A, 0x0A, 0x0B),
    soft: rgb(0x1B, 0x1B, 0x1F),
    soft_active: rgb(0x26, 0x26, 0x2C),

    fg: rgb(0xF0, 0xF0, 0xF3),
    fg_secondary: rgb(0xA2, 0xA2, 0xAE),
    fg_tertiary: rgb(0x70, 0x70, 0x7C),
    on_accent: rgb(0xFF, 0xFF, 0xFF),

    // The fill stays the brand blue -- white on `#0000F2` is 9.2:1 either way.
    accent: rgb(0x00, 0x00, 0xF2),
    // As text it cannot: on `#0D0D0E` it measures 2.11:1. Lightened until it
    // reads, which is what Hermes does to its own accents in dark mode.
    accent_text: rgb(0x7A, 0x7A, 0xFF),
    accent_soft: rgb(0x16, 0x16, 0x3A),

    stroke_1: rgb(0x45, 0x45, 0x50),
    stroke_2: rgb(0x33, 0x33, 0x3C),
    stroke_3: rgb(0x24, 0x24, 0x2B),
    stroke_4: rgb(0x18, 0x18, 0x1D),

    // Hermes lightens red and green in dark mode for the same reason.
    green: rgb(0x55, 0xA5, 0x83),
    red: rgb(0xE7, 0x5E, 0x78),
    yellow: rgb(0xD6, 0xA5, 0x5E),
};

/// The installer's palette: Hermes' own setup, which is a different surface
/// from its desktop and looks it.
///
/// Read out of `apps/bootstrap-installer/src/styles.css`: a navy ground
/// (`--theme-background-seed: #0d2f86`), cream type (`--theme-foreground:
/// #ffe6cb`), and a card a shade lighter than the ground. **The accent here is
/// the cream, not the brand blue** -- `#0000F2` on `#0d2f86` is 1.1:1 and
/// invisible, which is the whole reason Hermes puts its wordmark in cream and
/// the blue behind it.
pub const SETUP: Theme = Theme {
    mode: Mode::Dark,
    bg: rgb(0x0D, 0x2F, 0x86),
    chrome: rgb(0x09, 0x28, 0x6F),
    soft: rgb(0x12, 0x37, 0x8F),
    soft_active: rgb(0x1B, 0x45, 0xA4),

    fg: rgb(0xFF, 0xE6, 0xCB),
    // Cream mixed toward the navy ground, rather than an unrelated grey.
    fg_secondary: rgb(0xC8, 0xB6, 0xAE),
    fg_tertiary: rgb(0x93, 0x8E, 0x9C),
    on_accent: rgb(0x0D, 0x2F, 0x86),

    accent: rgb(0xFF, 0xE6, 0xCB),
    accent_text: rgb(0xFF, 0xE6, 0xCB),
    accent_soft: rgb(0x1B, 0x45, 0xA4),

    stroke_1: rgb(0x8A, 0x8C, 0xB4),
    stroke_2: rgb(0x5A, 0x67, 0xA4),
    stroke_3: rgb(0x30, 0x4A, 0x95),
    stroke_4: rgb(0x1B, 0x3C, 0x8C),

    green: rgb(0x7A, 0xD1, 0xA8),
    red: rgb(0xE8, 0x8A, 0x7E),
    yellow: rgb(0xF0, 0xC9, 0x7A),
};

/// Display faces for the installer wordmark, best first.
///
/// Hermes sets its wordmark in `Collapse`, which is Nous Research's own and not
/// ours to redistribute. These are the high-contrast serifs Windows machines
/// actually have; `win32::first_available_face` picks the first that is
/// installed, because `CreateFontW` substitutes silently and would otherwise
/// turn a display serif into the UI font with no indication.
pub const FACE_DISPLAY: &[&str] = &[
    "Bodoni MT",
    "Playfair Display",
    "Didot",
    "Sitka Display",
    "Constantia",
    "Georgia",
    "Times New Roman",
];

pub const fn theme(mode: Mode) -> Theme {
    match mode {
        Mode::Light => LIGHT,
        Mode::Dark => DARK,
    }
}

// -- type --------------------------------------------------------------------

/// The UI face.
///
/// Hermes' `--dt-font-sans` begins `'Segoe WPC', 'Segoe UI'` -- on Windows its
/// interface *is* the system face. So "use Hermes' font" needs nothing
/// installed and nothing shipped. (Its display face, `Collapse`, is Nous
/// Research's own and is not ours to redistribute; the wordmark here is the
/// Chaos logo, which we drew.)
pub const FACE_UI: &str = "Segoe UI";

/// Numbers, endpoints, sizes, throughput, and the transcript.
///
/// **Every measurement is monospaced**, so a column of them lines up and a
/// digit changing does not reflow the line beside it. Consolas ships with
/// Windows; Cascadia Mono does too on 11, but only Consolas is guaranteed.
pub const FACE_MONO: &str = "Consolas";

/// Three sizes, and no more.
///
/// The old window ran everything at 14-15px, which is why nothing led the eye:
/// with one size, hierarchy has to come from position alone. Negative because
/// `CreateFontW` reads a negative height as *character* height in pixels, which
/// is the number a designer means; positive is cell height including leading.
pub mod size {
    /// The page title. One per page, and nothing else is this big.
    pub const DISPLAY: i32 = -30;
    /// A section heading inside a page.
    pub const HEADING: i32 = -19;
    /// Body text, controls, list rows.
    pub const BODY: i32 = -15;
    /// Units, hints, and the second line of a two-line row.
    pub const SMALL: i32 = -12;
    /// Measurements and endpoints.
    pub const MONO: i32 = -13;
    /// The wordmark in the navigation rail.
    pub const MARK: i32 = -17;
}

/// `CreateFontW` weights.
pub mod weight {
    pub const REGULAR: i32 = 400;
    pub const MEDIUM: i32 = 500;
    pub const BOLD: i32 = 700;
}

/// Spacing and sizes, so no page invents its own.
pub mod metric {
    /// The navigation rail. Wide enough for the longest destination plus the
    /// active rule, narrow enough to leave the page the screen.
    pub const RAIL: i32 = 208;
    /// The strip along the bottom that says what is running. On every page.
    pub const STRIP: i32 = 52;
    /// A page's side padding.
    pub const INSET: i32 = 28;
    /// The gap between a heading and what it heads.
    pub const GAP: i32 = 12;
    /// A control's height. One value, so a row of them lines up.
    pub const CONTROL: i32 = 30;
    /// A button's height -- taller than a text box, because it is pressed.
    pub const BUTTON: i32 = 32;
    /// A row in a list. Two lines of text fit.
    pub const ROW: i32 = 40;
    /// A navigation destination.
    pub const NAV_ROW: i32 = 34;
    /// One row of a dropdown's list.
    pub const COMBO_ROW: i32 = 26;
    /// How many rows of a dropdown are visible before it scrolls.
    ///
    /// **This is not decoration -- it is the reason a dropdown opens at all.**
    /// A combo box is sized by its *dropped* height, and Windows gives the
    /// closed box only what its item height needs. Size one to a single row and
    /// the list gets nothing, which is indistinguishable from a control that
    /// ignores the mouse. The longest list on the settings page offers seven.
    pub const COMBO_VISIBLE: i32 = 8;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RGB()` is `0x00BBGGRR`. Getting it backwards swaps red and blue, which
    /// is invisible in greyscale -- both palettes here have colour in them
    /// precisely so the mistake would show, but a test is cheaper than an eye.
    #[test]
    fn a_colour_packs_blue_into_the_high_byte() {
        assert_eq!(rgb(0x00, 0x00, 0xF2), 0x00F2_0000);
        assert_eq!(rgb(0xF2, 0x00, 0x00), 0x0000_00F2);
        // Greys are palindromes, which is why they prove nothing.
        assert_eq!(rgb(0x17, 0x17, 0x1A), 0x001A_1717);
    }

    /// The accent Atur asked for, in both palettes, as a fill.
    #[test]
    fn the_accent_is_atur_s_blue() {
        for t in [LIGHT, DARK] {
            assert_eq!(t.accent, rgb(0x00, 0x00, 0xF2), "{:?}", t.mode);
        }
    }

    /// Relative luminance, per WCAG, for the contrast checks below.
    fn luminance(c: Rgb) -> f64 {
        let ch = |v: u32| {
            let s = (v & 0xFF) as f64 / 255.0;
            if s <= 0.03928 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        // Unpack the 0x00BBGGRR layout back out.
        0.2126 * ch(c) + 0.7152 * ch(c >> 8) + 0.0722 * ch(c >> 16)
    }

    fn contrast(a: Rgb, b: Rgb) -> f64 {
        let (x, y) = (luminance(a), luminance(b));
        let (hi, lo) = if x > y { (x, y) } else { (y, x) };
        (hi + 0.05) / (lo + 0.05)
    }

    /// **Body text must be readable in both palettes.** 4.5:1 is the WCAG AA
    /// threshold for normal text; this is the check that stops a palette tweak
    /// quietly making the app unusable.
    #[test]
    fn text_reads_against_its_ground() {
        for t in [LIGHT, DARK] {
            for (name, fg, bg) in [
                ("fg on bg", t.fg, t.bg),
                ("fg on chrome", t.fg, t.chrome),
                ("fg on soft", t.fg, t.soft),
                ("secondary on bg", t.fg_secondary, t.bg),
                ("on_accent on accent", t.on_accent, t.accent),
                ("accent_text on bg", t.accent_text, t.bg),
                ("accent_text on chrome", t.accent_text, t.chrome),
            ] {
                let c = contrast(fg, bg);
                assert!(
                    c >= 4.5,
                    "{:?}: {name} is {c:.2}:1, below the 4.5:1 floor",
                    t.mode
                );
            }
        }
    }

    /// Tertiary text is deliberately quiet, but "quiet" is 3:1, not invisible.
    #[test]
    fn even_the_quietest_text_is_visible() {
        for t in [LIGHT, DARK] {
            let c = contrast(t.fg_tertiary, t.bg);
            assert!(c >= 3.0, "{:?}: tertiary is {c:.2}:1", t.mode);
        }
    }

    /// **This is why `accent_text` exists.** `#0000F2` on the dark ground is
    /// about 2.11:1 -- unreadable -- so a single `accent` token used for both
    /// fills and text would have been fine in light and broken in dark. If
    /// someone collapses the two tokens back into one, this fails.
    #[test]
    fn the_raw_accent_would_be_unreadable_on_the_dark_ground() {
        // The bound is 3.0 rather than the measured 2.11, so a small palette
        // tweak does not fail this; anything that genuinely made the raw accent
        // readable would clear 3.0 by a wide margin.
        let raw = contrast(DARK.accent, DARK.bg);
        assert!(
            raw < 3.0,
            "the raw accent is {raw:.2}:1 on the dark ground -- if it has become \
             readable, accent_text has stopped earning its place"
        );
        assert!(contrast(DARK.accent_text, DARK.bg) >= 4.5);
    }

    /// Status colours have to be distinguishable from body text, or "running"
    /// green reads as ordinary text and the one signal on the strip is lost.
    #[test]
    fn status_colours_are_not_just_the_text_colour() {
        for t in [LIGHT, DARK] {
            for (name, c) in [("green", t.green), ("red", t.red), ("yellow", t.yellow)] {
                assert!(
                    contrast(c, t.bg) >= 3.0,
                    "{:?}: {name} is {:.2}:1 against the page",
                    t.mode,
                    contrast(c, t.bg)
                );
                assert_ne!(c, t.fg, "{:?}: {name} is the body colour", t.mode);
            }
        }
    }

    /// Hairlines must descend. A "four strengths" scale whose members are out
    /// of order is four names for one thing.
    #[test]
    fn hairlines_descend_in_strength() {
        for t in [LIGHT, DARK] {
            let s = [t.stroke_1, t.stroke_2, t.stroke_3, t.stroke_4];
            for w in s.windows(2) {
                let (a, b) = (contrast(w[0], t.bg), contrast(w[1], t.bg));
                assert!(
                    a > b,
                    "{:?}: strokes are not in descending order ({a:.2} then {b:.2})",
                    t.mode
                );
            }
        }
    }

    /// Three sizes, distinct, in order. The old window's failure was one size
    /// for everything; a scale whose steps are too close is the same failure
    /// with more code.
    #[test]
    fn the_type_scale_actually_steps() {
        // Negative heights: bigger text is a *smaller* number.
        let scale = [size::DISPLAY, size::HEADING, size::BODY, size::SMALL];
        for w in scale.windows(2) {
            assert!(w[0] < w[1], "{:?} does not step down", w);
            assert!(
                (w[1] - w[0]) >= 3,
                "{:?} and {:?} are too close to read as different",
                w[0],
                w[1]
            );
        }
    }

    /// The installer's ground is the reason its accent is cream: the brand blue
    /// on navy is unreadable, and a token table that let someone use it there
    /// would produce an invisible wordmark.
    #[test]
    fn the_setup_palette_reads_on_its_navy() {
        for (name, fg, bg) in [
            ("fg on bg", SETUP.fg, SETUP.bg),
            ("fg on soft", SETUP.fg, SETUP.soft),
            ("secondary on bg", SETUP.fg_secondary, SETUP.bg),
            ("on_accent on accent", SETUP.on_accent, SETUP.accent),
        ] {
            let c = contrast(fg, bg);
            assert!(c >= 4.5, "setup: {name} is {c:.2}:1");
        }
        assert!(
            contrast(LIGHT.accent, SETUP.bg) < 2.0,
            "the brand blue has become readable on the installer ground; if so,              SETUP.accent no longer needs to be the cream"
        );
    }

    /// A display face list that is empty, or that leads with something no
    /// Windows machine has and ends there, gives a wordmark in the UI font.
    #[test]
    fn the_display_face_list_ends_somewhere_universal() {
        assert!(!FACE_DISPLAY.is_empty());
        assert_eq!(
            *FACE_DISPLAY.last().unwrap(),
            "Times New Roman",
            "the fallback must be a face that ships with every Windows"
        );
    }

    #[test]
    fn a_mode_round_trips_through_its_name() {
        for m in [Mode::Light, Mode::Dark] {
            assert_eq!(Mode::parse(m.as_str()), Some(m));
        }
        assert_eq!(Mode::parse("  DARK "), Some(Mode::Dark));
        assert_eq!(Mode::parse("purple"), None);
        assert_eq!(Mode::Light.toggled(), Mode::Dark);
        assert_eq!(Mode::Dark.toggled(), Mode::Light);
    }

    /// The rail has to fit the longest destination at body size, or the
    /// navigation clips -- which is exactly how the old sidebar failed.
    #[test]
    fn the_rail_fits_the_longest_destination() {
        let longest = ["CHAT", "MODELS", "MONITOR", "SETTINGS"]
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap() as i32;
        // Segoe UI at 15px averages under 9px a character for capitals.
        let needed = longest * 9 + metric::INSET * 2;
        assert!(
            metric::RAIL >= needed,
            "the rail is {}px and needs {needed}px",
            metric::RAIL
        );
    }
}
