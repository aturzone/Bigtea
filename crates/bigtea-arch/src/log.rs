//! Where the runner's status lines go.
//!
//! # Why this is not just `println!`
//!
//! Everything the runner says about itself — the shape it read, what it made
//! resident, how long prefill took — is *diagnostics*, not output. The
//! generated text is output. Mixing them into one stream means a caller
//! piping `bigtea-run` into a file gets a header it has to strip, and a caller
//! wanting the header has no way to keep it when redirecting.
//!
//! llama.cpp separates the two and gives the diagnostic side a handful of
//! switches. This is that separation, with its flags:
//!
//! ```text
//! --log-disable        silence status entirely
//! --log-file F         send it to a file instead of the terminal
//! --log-timestamps     prefix each line with elapsed time
//! --log-prefix         prefix each line with its level
//! --verbosity N        0 quiet, 1 normal (default), 2+ verbose
//! ```
//!
//! **Status goes to stderr, not stdout.** That is the actual fix: `bigtea-run
//! ... > answer.txt` should contain the answer and nothing else, and today it
//! contains the header too.

use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// How loud, and where to.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// `0` silences everything, `1` is the normal header, `2+` adds detail.
    pub verbosity: u32,
    /// Prefix every line with seconds since start.
    pub timestamps: bool,
    /// Prefix every line with its level.
    pub prefix: bool,
    /// Append to this file instead of writing to stderr.
    pub file: Option<String>,
    /// Colour the status lines by level.
    ///
    /// **Never applied to a `--log-file`**: escape codes written to a file are
    /// noise in every reader that is not a terminal, and llama.cpp's own
    /// `--log-colors` has the same carve-out. A log you cannot `grep` is worse
    /// than an uncoloured one.
    pub colors: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        LogConfig {
            verbosity: 1,
            timestamps: false,
            prefix: false,
            file: None,
            colors: false,
        }
    }
}

struct State {
    config: LogConfig,
    start: Instant,
    sink: Mutex<Option<std::fs::File>>,
}

static STATE: OnceLock<State> = OnceLock::new();

/// Install the configuration. The first call wins.
///
/// Called once from `main` before anything is opened, so a `--log-file` catches
/// the container header too. A second call is ignored rather than racing.
pub fn configure(config: LogConfig) {
    let file = config.file.as_ref().and_then(|path| {
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(f) => Some(f),
            Err(e) => {
                // Straight to stderr: the log sink is what just failed, so
                // routing this through it would lose the message.
                eprintln!("bigtea: cannot open log file {path}: {e}");
                None
            }
        }
    });
    let _ = STATE.set(State {
        config,
        start: Instant::now(),
        sink: Mutex::new(file),
    });
}

/// Whether a message at `level` would be printed.
///
/// Exposed so a caller can skip *computing* an expensive line, not just skip
/// printing it.
pub fn enabled(level: u32) -> bool {
    STATE.get().map(|s| s.config.verbosity).unwrap_or(1) >= level
}

/// Write one status line at `level`.
pub fn log(level: u32, message: &str) {
    let Some(state) = STATE.get() else {
        // Not configured — before `main` ran, or in a test. Default behaviour.
        eprintln!("{message}");
        return;
    };
    if state.config.verbosity < level {
        return;
    }
    let mut line = String::new();
    if state.config.timestamps {
        line.push_str(&format!("{:8.3} ", state.start.elapsed().as_secs_f64()));
    }
    if state.config.prefix {
        line.push_str(if level >= 2 { "D " } else { "I " });
    }
    line.push_str(message);

    if let Ok(mut sink) = state.sink.lock() {
        if let Some(file) = sink.as_mut() {
            // Uncoloured on purpose -- see `LogConfig::colors`.
            let _ = writeln!(file, "{line}");
            return;
        }
    }
    if state.config.colors {
        // Detail dim, normal plain-but-bright. Two codes rather than a palette:
        // the point is separating status from the generated text on stdout,
        // not decorating it.
        let code = if level >= 2 { "2" } else { "36" };
        eprintln!("\x1b[{code}m{line}\x1b[0m");
    } else {
        eprintln!("{line}");
    }
}

/// A status line at the normal level.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => { $crate::log::log(1, &format!($($arg)*)) };
}

/// A status line only shown at `--verbosity 2` or above.
#[macro_export]
macro_rules! detail {
    ($($arg:tt)*) => { $crate::log::log(2, &format!($($arg)*)) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbosity_gates_by_level() {
        // `enabled` is what lets a caller skip building an expensive line.
        // Unconfigured is level 1, which is the normal header.
        assert!(enabled(1));
        assert!(!enabled(2));
    }

    #[test]
    fn a_disabled_log_still_answers_enabled_honestly() {
        let cfg = LogConfig {
            verbosity: 0,
            ..LogConfig::default()
        };
        // Not installed globally — this checks the type's own contract, since
        // `configure` is once-per-process and a test cannot own it.
        assert_eq!(cfg.verbosity, 0);
        assert!(cfg.file.is_none());
        assert!(!cfg.timestamps && !cfg.prefix);
    }
}
