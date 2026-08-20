//! Watching a model load, the same way downloads are watched.
//!
//! Atur: *"the Chaos app needs a progress bar to really show us progress"*. The
//! window said **"loading -- a large model takes a while"** and then nothing,
//! for as long as it took. On a 144 GB container that is minutes of a window
//! that looks broken, which is the same complaint that produced
//! [`crate::download`] and deserves the same answer.
//!
//! # Measured, not parsed
//!
//! `chaos-serve` reads the model's always-read weights into memory before it
//! answers anything, so **the server's working set is the progress**. The
//! catalogue already knows what that set weighs. Dividing one by the other is a
//! percentage that needs no protocol, no extra output to parse, and no
//! cooperation from the process being watched — the same reasoning that made
//! bytes-on-disk the download's progress.
//!
//! It also survives what parsing would not: a server started with no console, a
//! future version that prints differently, or one that is already part-way
//! through when the window attaches.
//!
//! # What it is honest about
//!
//! **This is not a completion percentage, it is a residency percentage**, and
//! the two part company at the end. The last few per cent cover work that is not
//! reading weights — building the graph, warming the sampler — so the bar
//! reaches 99 and waits. It is deliberately capped below 100 for exactly that
//! reason: a bar that sits full while the app is still busy is worse than one
//! that sits at 99, because only the second is telling the truth.
//!
//! Readiness is still "does it answer". The bar says how far along it looks.

use crate::models::human_size;

/// A model being loaded by a `chaos-serve` we started.
#[derive(Clone, Debug)]
pub struct Loading {
    /// What the user picked, for the line on screen.
    pub label: String,
    /// What the catalogue says has to end up resident, or **0 when it is not
    /// known**.
    ///
    /// Only the catalogue knows this, and an installed container may not be in
    /// it. The honest answer then is bytes with no percentage: a denominator
    /// guessed from the file size would read 5% for the whole load of a 144 GB
    /// mixture-of-experts, whose resident set is 7 GiB of it.
    pub resident: u64,
    /// The server's working set right now.
    pub rss: u64,
    /// Seconds since it was started.
    pub elapsed: f64,
    /// Set when the server answers.
    pub ready: bool,
}

/// The highest the bar goes before the server actually answers.
///
/// See the module docs: the tail of a load is not weights.
const CEILING: u32 = 99;

impl Loading {
    pub fn new(label: String, resident: u64) -> Self {
        Self {
            label,
            resident,
            rss: 0,
            elapsed: 0.0,
            ready: false,
        }
    }

    /// How far along it looks, `0..=99` until it answers.
    pub fn percent(&self) -> u32 {
        if self.ready {
            return 100;
        }
        if self.resident == 0 {
            return 0;
        }
        let p = (self.rss.min(self.resident) as u128 * 100 / self.resident as u128) as u32;
        p.min(CEILING)
    }

    /// Bytes a second into memory, over this load.
    pub fn rate(&self) -> f64 {
        if self.elapsed < 1.0 || self.rss == 0 {
            return 0.0;
        }
        self.rss as f64 / self.elapsed
    }

    /// Seconds left at the current rate, or `None` when there is nothing to go
    /// on. **Never a guess**, like the download's.
    pub fn eta_secs(&self) -> Option<u64> {
        let rate = self.rate();
        if rate <= 0.0 || self.ready {
            return None;
        }
        let left = self.resident.saturating_sub(self.rss) as f64;
        (left > 0.0).then(|| (left / rate) as u64)
    }

    /// The whole thing as one line.
    pub fn line(&self) -> String {
        if self.ready {
            return format!("{} — ready", self.label);
        }
        // Before the first sample there is nothing honest to show but the fact
        // that it started.
        if self.rss == 0 {
            return format!("loading {} — starting", self.label);
        }
        // Unknown total: report what is loaded, claim no fraction of it.
        if self.resident == 0 {
            let mut s = format!("loading {} — {} resident", self.label, human_size(self.rss));
            let rate = self.rate();
            if rate > 0.0 {
                s.push_str(&format!("  ·  {}/s", human_size(rate as u64)));
            }
            return s;
        }
        let mut s = format!(
            "loading {}  {}%  ·  {} of {} resident",
            self.label,
            self.percent(),
            human_size(self.rss),
            human_size(self.resident)
        );
        let rate = self.rate();
        if rate > 0.0 {
            s.push_str(&format!("  ·  {}/s", human_size(rate as u64)));
        }
        if let Some(eta) = self.eta_secs() {
            s.push_str(&format!(
                "  ·  {} left",
                crate::download::human_duration(eta)
            ));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l(resident: u64, rss: u64, elapsed: f64) -> Loading {
        Loading {
            label: "qwen3-4b".into(),
            resident,
            rss,
            elapsed,
            ready: false,
        }
    }

    /// The percentage is residency, and it is capped below 100 until the server
    /// answers.
    #[test]
    fn the_bar_stops_at_99_until_it_is_actually_ready() {
        assert_eq!(l(1000, 0, 0.0).percent(), 0);
        assert_eq!(l(1000, 500, 1.0).percent(), 50);
        // The tail of a load is not weights, so a full read is still not ready.
        assert_eq!(l(1000, 1000, 1.0).percent(), 99);
        // A working set larger than the catalogue's figure is normal -- there is
        // more in a process than weights -- and must not read as 140%.
        assert_eq!(l(1000, 1400, 1.0).percent(), 99);
        let mut done = l(1000, 1000, 1.0);
        done.ready = true;
        assert_eq!(done.percent(), 100);
        // A model with no known resident size reports nothing rather than
        // dividing by zero.
        assert_eq!(l(0, 500, 1.0).percent(), 0);
    }

    /// An unknown total gets bytes and no invented percentage.
    #[test]
    fn an_unknown_resident_size_claims_no_fraction() {
        let s = l(0, 3_000_000_000, 3.0).line();
        assert!(s.contains("resident"), "{s}");
        assert!(s.contains("/s"), "{s}");
        assert!(!s.contains('%'), "a percentage of an unknown total: {s}");
        assert!(!s.contains("left"), "an estimate with no total: {s}");
    }

    /// An estimate is only offered when there is something to base it on.
    #[test]
    fn there_is_no_estimate_without_a_rate() {
        assert_eq!(l(1000, 0, 0.0).eta_secs(), None, "nothing has loaded yet");
        assert_eq!(l(1000, 100, 0.5).eta_secs(), None, "too early to say");
        // 500 bytes in 1s, 500 to go.
        assert_eq!(l(1000, 500, 1.0).eta_secs(), Some(1));
        let mut done = l(1000, 500, 1.0);
        done.ready = true;
        assert_eq!(done.eta_secs(), None, "ready has nothing left");
    }

    /// The line says what it knows and no more.
    #[test]
    fn the_line_reports_only_what_it_has() {
        // Before the first sample: no invented percentage.
        let s = l(1000, 0, 0.0).line();
        assert!(s.contains("starting"), "{s}");
        assert!(!s.contains('%'), "{s}");

        let s = l(4_000_000_000, 2_000_000_000, 4.0).line();
        assert!(s.contains("50%"), "{s}");
        assert!(s.contains("resident"), "{s}");
        assert!(s.contains("/s"), "{s}");
        assert!(s.contains("left"), "{s}");

        let mut done = l(1000, 1000, 1.0);
        done.ready = true;
        assert_eq!(done.line(), "qwen3-4b — ready");
    }
}
