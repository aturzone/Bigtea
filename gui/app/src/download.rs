//! Watching a download that another process is doing.
//!
//! Atur: *"percent downloads of models"*. The app shelled out to `chaos-pull`
//! with no console and then said "downloading" until it finished, which for a
//! 155 GB container is an hour of a window that looks broken.
//!
//! **Progress is measured from the files themselves, not parsed from `curl`.**
//! `chaos-pull` resumes with `-C -` straight into the final filenames, so the
//! bytes on disk *are* the progress, and the catalogue already knows the total.
//! Reading a directory is also the one method that survives the downloader
//! changing its output, running in a console we cannot see, or being restarted.
//!
//! No Win32 here, so the arithmetic and the wording are testable.

use crate::models::human_size;
use std::path::PathBuf;

/// A download in flight.
#[derive(Clone, Debug)]
pub struct Download {
    /// What the user asked for, for the line on screen.
    pub label: String,
    /// Every file it will produce. A five-shard container is five.
    pub files: Vec<PathBuf>,
    /// What the catalogue says the whole thing weighs.
    pub total: u64,
    /// Bytes present when it started, so a resume does not report the part it
    /// already had as though this run fetched it.
    pub start_bytes: u64,
    /// Bytes present now.
    pub done_bytes: u64,
    /// Seconds since it started.
    pub elapsed: f64,
    /// Set when `chaos-pull` exits, whatever it says.
    pub finished: bool,
}

impl Download {
    pub fn new(label: String, files: Vec<PathBuf>, total: u64) -> Self {
        let start = bytes_on_disk(&files);
        Self {
            label,
            files,
            total,
            start_bytes: start,
            done_bytes: start,
            elapsed: 0.0,
            finished: false,
        }
    }

    pub fn percent(&self) -> u32 {
        if self.total == 0 {
            return 0;
        }
        // Clamped: a container can be a few bytes larger than the catalogue
        // records, and "101%" reads as a bug in everything else too.
        ((self.done_bytes.min(self.total) * 100) / self.total) as u32
    }

    /// Bytes a second, over this run only.
    pub fn rate(&self) -> f64 {
        let fetched = self.done_bytes.saturating_sub(self.start_bytes) as f64;
        if self.elapsed < 1.0 || fetched <= 0.0 {
            return 0.0;
        }
        fetched / self.elapsed
    }

    /// Seconds left at the current rate, or `None` while there is nothing to
    /// go on. **Never a guess**: a rate of zero produces no estimate rather
    /// than an infinity dressed up as a number.
    pub fn eta_secs(&self) -> Option<u64> {
        let rate = self.rate();
        if rate <= 0.0 {
            return None;
        }
        let left = self.total.saturating_sub(self.done_bytes) as f64;
        (left > 0.0).then(|| (left / rate) as u64)
    }

    /// The whole thing as one line.
    pub fn line(&self) -> String {
        if self.finished {
            return format!("{} — {}", self.label, human_size(self.done_bytes));
        }
        let mut s = format!(
            "{}  {}%  ·  {} of {}",
            self.label,
            self.percent(),
            human_size(self.done_bytes),
            human_size(self.total)
        );
        let rate = self.rate();
        if rate > 0.0 {
            s.push_str(&format!("  ·  {}/s", human_size(rate as u64)));
        }
        if let Some(eta) = self.eta_secs() {
            s.push_str(&format!("  ·  {} left", human_duration(eta)));
        }
        s
    }
}

/// How many bytes of these files exist right now.
///
/// A missing file counts as zero rather than an error: the second shard of five
/// does not exist until the first is done, and that is the normal case.
pub fn bytes_on_disk(files: &[PathBuf]) -> u64 {
    files
        .iter()
        .filter_map(|f| std::fs::metadata(f).ok())
        .map(|m| m.len())
        .sum()
}

/// A rough duration, for an estimate that is rough by nature.
pub fn human_duration(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs}s"),
        // Up to an hour, not up to ninety minutes: 3600 rounding to "60m"
        // rather than "1h" is what this range got wrong the first time.
        60..=3599 => format!("{}m", (secs + 30) / 60),
        _ => {
            let h = secs / 3600;
            let m = (secs % 3600) / 60;
            if m == 0 {
                format!("{h}h")
            } else {
                format!("{h}h {m}m")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dl(total: u64, start: u64, done: u64, elapsed: f64) -> Download {
        Download {
            label: "m".into(),
            files: vec![],
            total,
            start_bytes: start,
            done_bytes: done,
            elapsed,
            finished: false,
        }
    }

    #[test]
    fn percent_is_of_the_whole_download() {
        assert_eq!(dl(100, 0, 0, 0.0).percent(), 0);
        assert_eq!(dl(100, 0, 25, 1.0).percent(), 25);
        assert_eq!(dl(100, 0, 100, 1.0).percent(), 100);
    }

    /// A container can be marginally larger than the catalogue records, and a
    /// progress bar past the end reads as a bug in the whole app.
    #[test]
    fn percent_never_exceeds_a_hundred() {
        assert_eq!(dl(100, 0, 140, 1.0).percent(), 100);
        assert_eq!(dl(0, 0, 500, 1.0).percent(), 0);
    }

    /// **A resume must not claim credit for bytes it did not fetch.** Half a
    /// 155 GB container already on disk would otherwise report an absurd rate
    /// in the first second and an ETA of nothing.
    #[test]
    fn the_rate_counts_only_this_run() {
        let d = dl(1000, 400, 500, 10.0);
        assert_eq!(d.rate(), 10.0, "100 bytes over 10s, not 500");
        assert_eq!(d.percent(), 50, "percent is still of the whole file");
    }

    /// No rate yet means no estimate, rather than a number made up from one.
    #[test]
    fn there_is_no_eta_without_a_rate() {
        assert_eq!(dl(1000, 0, 0, 0.5).eta_secs(), None, "too early to say");
        assert_eq!(dl(1000, 0, 0, 30.0).eta_secs(), None, "nothing fetched yet");
        assert_eq!(dl(1000, 0, 1000, 10.0).eta_secs(), None, "already finished");
        assert_eq!(dl(1000, 0, 500, 10.0).eta_secs(), Some(10));
    }

    /// The line has to carry every number a user needs, and none of the ones
    /// that are not known yet.
    #[test]
    fn the_line_says_what_is_known_and_no_more() {
        let early = dl(16_817_244_384, 0, 0, 0.2).line();
        assert!(early.contains("0%"), "{early}");
        assert!(!early.contains("left"), "an ETA before any bytes: {early}");
        assert!(!early.contains("/s"), "a rate before any bytes: {early}");

        let mid = dl(16_817_244_384, 0, 6_000_000_000, 600.0).line();
        assert!(mid.contains("35%"), "{mid}");
        assert!(mid.contains("6.00 GB") || mid.contains("6 GB"), "{mid}");
        assert!(mid.contains("/s"), "no rate: {mid}");
        assert!(mid.contains("left"), "no estimate: {mid}");
    }

    #[test]
    fn a_finished_download_stops_estimating() {
        let mut d = dl(100, 0, 100, 10.0);
        d.finished = true;
        let l = d.line();
        assert!(!l.contains("left") && !l.contains('%'), "{l}");
    }

    #[test]
    fn durations_are_rounded_the_way_people_read_them() {
        assert_eq!(human_duration(45), "45s");
        assert_eq!(human_duration(90), "2m");
        assert_eq!(human_duration(3600), "1h");
        assert_eq!(human_duration(5400), "1h 30m");
    }

    /// Files that do not exist yet count as zero: the second shard of five is
    /// absent until the first finishes, and that is the ordinary case.
    #[test]
    fn missing_files_are_zero_not_an_error() {
        let missing = vec![PathBuf::from("no-such-file-anywhere.gguf")];
        assert_eq!(bytes_on_disk(&missing), 0);
    }
}
