//! The logo the CLI prints when it starts.
//!
//! # How a picture gets into a terminal without a dependency
//!
//! `assets/logo.svg` is 43 closed Bezier paths. It is rasterised **offline** by
//! `tools/rasterise-logo.py` into [`logo_bitmap`] -- 3 KB of luminance bytes,
//! committed -- so the crate carries no SVG parser, no image decoder, no build
//! script and no external dependency. This workspace has zero of those and the
//! banner was not going to be the first one.
//!
//! Printing uses the upper half-block `U+2580`. A cell has a foreground and a
//! background colour, so one cell is **two vertical pixels**: `fg` paints the
//! top half, `bg` the bottom. That is twice the vertical resolution of any
//! one-glyph-per-pixel scheme and as close to pixel-perfect as a terminal gets.
//!
//! # When it does not print
//!
//! A banner that appears in a log file or a pipe is a bug. It is skipped when:
//!
//! - `NO_COLOR` is set to anything at all (<https://no-color.org>);
//! - `CHAOS_NO_BANNER` is set;
//! - **either** stdout or stderr is not a terminal. Gating on stdout alone
//!   would still write escape codes into `2> log`; gating on stderr alone would
//!   still print for `chaos-run > answer.txt`, where the user asked for the
//!   answer and not for decoration;
//! - the terminal is too small to hold the smallest size;
//! - status output is silenced (`--log-disable`, `--verbosity 0`).
//!
//! It goes to **stderr**, with the rest of the diagnostics, because stdout is
//! the generated text and nothing else.

use crate::logo_bitmap::{LOGO, LOGO_H, LOGO_W};

/// Widths the master bitmap is box-filtered down to, largest first.
///
/// Not arbitrary: rendered to PNG and looked at. At 56 the eye, the hands and
/// the individual rays all survive; by 40 the rays read but the centre is
/// softening; by 28 it is a recognisable sun and nothing finer. Below that it
/// is a grey blob and the banner is not worth the rows, so there is no fourth
/// entry and a terminal that cannot hold 28 columns gets no logo.
const SIZES: [usize; 3] = [56, 40, 28];

/// Print the logo to stderr, or do nothing if this is not the place for it.
pub fn print() {
    if !wanted() {
        return;
    }
    let (cols, rows) = terminal_size();
    let Some(width) = SIZES.into_iter().find(|&w| {
        let h = height_for(w);
        // Two pixels to a row, and leave four rows for the first status lines
        // so the banner never fills the screen on its own.
        w + 2 <= cols && h.div_ceil(2) + 4 <= rows
    }) else {
        return;
    };
    eprint!("{}", render(width, height_for(width)));
    // Centred under the logo, dim, one line. The version belongs here because
    // it is the first thing asked about any bug report, and `--version` is not
    // what a user types before noticing something is wrong.
    // The long form wraps in a narrow terminal, and a wrapped tagline looks
    // like a bug rather than a strapline, so it is dropped rather than folded.
    let long = format!("chaos {VERSION}  --  a runner for models larger than RAM");
    let mark = if long.chars().count() <= cols.saturating_sub(2) {
        long
    } else {
        format!("chaos {VERSION}")
    };
    let pad = width.saturating_sub(mark.chars().count()) / 2;
    eprintln!("\x1b[2m{:pad$}{mark}\x1b[0m\n", "", pad = pad);
}

/// The workspace version, so the banner cannot drift from `Cargo.toml`.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Whether a banner belongs here at all.
///
/// `NO_COLOR` wins over `CHAOS_BANNER=1`: a forced banner is still colour, and
/// the standard says a user who set that variable does not want any.
fn wanted() -> bool {
    if std::env::var_os("NO_COLOR").is_some() || std::env::var_os("CHAOS_NO_BANNER").is_some() {
        return false;
    }
    if !crate::log::enabled(1) {
        return false;
    }
    // The escape hatch for `chaos-run ... 2>&1 | less -R`, and for anyone
    // taking a screenshot through a pipe. Opt-in, because the default has to be
    // the safe one.
    if std::env::var_os("CHAOS_BANNER").is_some() {
        return true;
    }
    is_terminal(Stream::Stdout) && is_terminal(Stream::Stderr)
}

/// The master is 56x54, so a narrower render keeps that aspect ratio.
fn height_for(width: usize) -> usize {
    (width * LOGO_H).div_ceil(LOGO_W)
}

/// Box-filter the master down to `w` x `h` and emit it as half-block rows.
///
/// Kept separate from [`print`] and free of I/O so a test can assert on the
/// bytes -- a banner is the one thing nobody notices is broken.
fn render(w: usize, h: usize) -> String {
    let px = resample(w, h);
    // Roughly 24 bytes of escape per changed cell, and most cells repeat.
    let mut out = String::with_capacity(w * h * 8);
    for row in (0..h).step_by(2) {
        let (mut last_fg, mut last_bg) = (usize::MAX, usize::MAX);
        for x in 0..w {
            let fg = px[row * w + x] as usize;
            // An odd height leaves the bottom half of the last row unpainted;
            // the logo's background is white, so that is what belongs there.
            let bg = if row + 1 < h {
                px[(row + 1) * w + x] as usize
            } else {
                255
            };
            // Re-emitting an unchanged colour is correct but triples the output,
            // and a slow terminal shows it repainting.
            if fg != last_fg {
                out.push_str(&format!("\x1b[38;2;{fg};{fg};{fg}m"));
                last_fg = fg;
            }
            if bg != last_bg {
                out.push_str(&format!("\x1b[48;2;{bg};{bg};{bg}m"));
                last_bg = bg;
            }
            out.push('\u{2580}');
        }
        out.push_str("\x1b[0m\n");
    }
    out
}

/// Area-average the master into a `w` x `h` grid.
///
/// Averaging rather than sampling matters here: the logo is one-pixel rays on
/// white, and nearest-neighbour at these ratios drops whole rays instead of
/// dimming them.
fn resample(w: usize, h: usize) -> Vec<u8> {
    if w == LOGO_W && h == LOGO_H {
        return LOGO.to_vec();
    }
    let mut out = vec![255u8; w * h];
    for y in 0..h {
        let (y0, y1) = span(y, h, LOGO_H);
        for x in 0..w {
            let (x0, x1) = span(x, w, LOGO_W);
            let mut sum = 0usize;
            let mut n = 0usize;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    sum += LOGO[sy * LOGO_W + sx] as usize;
                    n += 1;
                }
            }
            out[y * w + x] = (sum / n.max(1)) as u8;
        }
    }
    out
}

/// The half-open source range one destination index covers, never empty.
fn span(i: usize, dst: usize, src: usize) -> (usize, usize) {
    let a = i * src / dst;
    let b = ((i + 1) * src).div_ceil(dst).min(src);
    (a, b.max(a + 1))
}

enum Stream {
    Stdout,
    Stderr,
}

#[cfg(windows)]
fn is_terminal(s: Stream) -> bool {
    // `GetConsoleMode` fails on a handle that is not a console, which is the
    // documented way to ask. -11 is STD_OUTPUT_HANDLE, -12 is STD_ERROR_HANDLE.
    extern "system" {
        fn GetStdHandle(n: u32) -> isize;
        fn GetConsoleMode(h: isize, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: isize, mode: u32) -> i32;
    }
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    let id = match s {
        Stream::Stdout => -11i32 as u32,
        Stream::Stderr => -12i32 as u32,
    };
    unsafe {
        let h = GetStdHandle(id);
        let mut mode = 0u32;
        if GetConsoleMode(h, &mut mode) == 0 {
            return false;
        }
        // Old conhost windows do not interpret escape sequences until asked,
        // and printing a screenful of raw `\x1b[38;2;...` is worse than
        // printing nothing. If it cannot be turned on, treat it as not a
        // terminal for our purposes.
        if mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING == 0
            && SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0
        {
            return false;
        }
        true
    }
}

#[cfg(not(windows))]
fn is_terminal(s: Stream) -> bool {
    extern "C" {
        fn isatty(fd: i32) -> i32;
    }
    let fd = match s {
        Stream::Stdout => 1,
        Stream::Stderr => 2,
    };
    unsafe { isatty(fd) == 1 }
}

/// Columns and rows, or a conservative default.
///
/// `COLUMNS`/`LINES` are consulted first on every platform because they are the
/// only portable answer and a user who exports them means them. The Windows
/// console API is the fallback there; elsewhere the fallback is 80x24, which is
/// deliberately small enough that guessing wrong shrinks the banner rather than
/// wrapping it.
fn terminal_size() -> (usize, usize) {
    let env = |k: &str| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
    };
    if let (Some(c), Some(r)) = (env("COLUMNS"), env("LINES")) {
        return (c, r);
    }
    console_size().unwrap_or((80, 24))
}

#[cfg(windows)]
fn console_size() -> Option<(usize, usize)> {
    #[repr(C)]
    #[derive(Default)]
    struct Rect {
        left: i16,
        top: i16,
        right: i16,
        bottom: i16,
    }
    #[repr(C)]
    #[derive(Default)]
    struct Info {
        size_x: i16,
        size_y: i16,
        cursor_x: i16,
        cursor_y: i16,
        attributes: u16,
        window: Rect,
        max_x: i16,
        max_y: i16,
    }
    extern "system" {
        fn GetStdHandle(n: u32) -> isize;
        fn GetConsoleScreenBufferInfo(h: isize, info: *mut Info) -> i32;
    }
    // The *window* is what the user can see; the buffer is usually far taller.
    unsafe {
        let mut info = Info::default();
        if GetConsoleScreenBufferInfo(GetStdHandle(-11i32 as u32), &mut info) == 0 {
            return None;
        }
        let w = (info.window.right - info.window.left + 1).max(0) as usize;
        let h = (info.window.bottom - info.window.top + 1).max(0) as usize;
        (w > 0 && h > 0).then_some((w, h))
    }
}

#[cfg(not(windows))]
fn console_size() -> Option<(usize, usize)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every destination pixel must cover at least one source pixel, or a whole
    /// row of the logo silently disappears at some size.
    #[test]
    fn every_span_is_non_empty_at_every_size() {
        for &w in &SIZES {
            let h = height_for(w);
            for i in 0..w {
                let (a, b) = span(i, w, LOGO_W);
                assert!(b > a, "empty column span at width {w}, index {i}");
            }
            for i in 0..h {
                let (a, b) = span(i, h, LOGO_H);
                assert!(b > a, "empty row span at height {h}, index {i}");
            }
        }
    }

    #[test]
    fn a_render_is_one_half_block_per_column_and_half_the_rows() {
        for &w in &SIZES {
            let h = height_for(w);
            let out = render(w, h);
            let lines: Vec<&str> = out.lines().collect();
            assert_eq!(lines.len(), h.div_ceil(2), "row count at width {w}");
            for line in lines {
                assert_eq!(
                    line.chars().filter(|c| *c == '\u{2580}').count(),
                    w,
                    "column count at width {w}"
                );
            }
        }
    }

    /// The logo is black on white. If the render came out uniform, the bitmap
    /// or the resampler is broken -- and a uniform block still *looks* like a
    /// deliberate design, which is why this is asserted rather than eyeballed.
    #[test]
    fn the_render_has_both_ink_and_paper() {
        let px = resample(40, height_for(40));
        assert!(px.iter().any(|&v| v < 64), "no dark pixels");
        assert!(px.iter().any(|&v| v > 200), "no light pixels");
    }

    /// The master must survive being asked for its own size unchanged.
    #[test]
    fn resampling_to_the_master_size_is_the_master() {
        assert_eq!(resample(LOGO_W, LOGO_H), LOGO.to_vec());
    }

    #[test]
    fn the_aspect_ratio_is_kept_within_a_pixel() {
        for &w in &SIZES {
            let want = w as f64 * LOGO_H as f64 / LOGO_W as f64;
            assert!((height_for(w) as f64 - want).abs() <= 1.0, "width {w}");
        }
    }
}
