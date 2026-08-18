//! What the app remembers between runs.
//!
//! Every field here was a box you had to retype each time the window opened,
//! which is not a settings page -- it is a form. They persist to a small text
//! file beside the models directory.
//!
//! **The format is `key = value`, one per line, and unknown keys are kept.**
//! A settings file that silently drops what it does not recognise makes a
//! downgrade destructive: run an older build once and the newer build's
//! preferences are gone. Parsing is hand-rolled because the workspace has no
//! serialisation crate and this is thirty lines.

use crate::theme::Mode;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Everything the window lets you change.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Expert cache budget in GiB. `None` means "let the engine measure".
    pub cache_gib: Option<f64>,
    /// Generation threads. `None` means measured.
    pub threads: Option<u32>,
    /// Prefill threads, which want the opposite of generation threads.
    pub threads_batch: Option<u32>,
    /// Where the local server listens.
    pub port: u16,
    /// Context cap, in tokens. `None` means the model's own limit.
    pub context: Option<u32>,
    /// Layers to put on the GPU. `None` means none; `Some(99)` means all.
    pub ngl: Option<u32>,
    /// Where models are looked for, overriding the default.
    pub models_dir: Option<String>,
    /// Let the engine choose device, offload and cache from the machine.
    pub auto: bool,
    /// Run an architecture that has not been diffed against llama.cpp.
    pub force: bool,
    /// Light or dark. Persisted because a window that forgets which way round
    /// it is every launch is not a preference, it is a flicker.
    pub mode: Mode,
    /// Keys read but not understood, preserved on write.
    unknown: BTreeMap<String, String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            cache_gib: None,
            threads: None,
            threads_batch: None,
            // Not 8080: that is the first port every other local tool takes,
            // and a collision here looks like the model failing to load.
            port: 8231,
            context: None,
            ngl: None,
            models_dir: None,
            auto: false,
            force: false,
            // Hermes' desktop defaults to light, and so does this.
            mode: Mode::Light,
            unknown: BTreeMap::new(),
        }
    }
}

/// `%USERPROFILE%\.chaos\settings.txt`, beside the models rather than inside
/// the install -- so an upgrade or an uninstall never takes preferences with it.
pub fn path() -> PathBuf {
    let base = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join(".chaos").join("settings.txt")
}

impl Settings {
    pub fn parse(text: &str) -> Self {
        let mut s = Settings::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let (k, v) = (k.trim(), v.trim());
            match k {
                "cache_gib" => s.cache_gib = v.parse().ok(),
                "threads" => s.threads = v.parse().ok(),
                "threads_batch" => s.threads_batch = v.parse().ok(),
                "port" => s.port = v.parse().unwrap_or(s.port),
                "context" => s.context = v.parse().ok(),
                "ngl" => s.ngl = v.parse().ok(),
                "models_dir" => {
                    s.models_dir = (!v.is_empty()).then(|| v.to_string());
                }
                "auto" => s.auto = truthy(v),
                "force" => s.force = truthy(v),
                "mode" => s.mode = Mode::parse(v).unwrap_or(s.mode),
                _ => {
                    s.unknown.insert(k.to_string(), v.to_string());
                }
            }
        }
        s
    }

    pub fn render(&self) -> String {
        let mut out =
            String::from("# Chaos settings. Written by chaos-app; safe to edit by hand.\n");
        let opt = |name: &str, v: Option<String>| match v {
            Some(v) => format!("{name} = {v}\n"),
            // Written as an empty value rather than omitted, so the file shows
            // every setting that exists and what it is currently not set to.
            None => format!("{name} =\n"),
        };
        out.push_str(&opt("cache_gib", self.cache_gib.map(|v| format!("{v}"))));
        out.push_str(&opt("threads", self.threads.map(|v| v.to_string())));
        out.push_str(&opt(
            "threads_batch",
            self.threads_batch.map(|v| v.to_string()),
        ));
        out.push_str(&format!("port = {}\n", self.port));
        out.push_str(&opt("context", self.context.map(|v| v.to_string())));
        out.push_str(&opt("ngl", self.ngl.map(|v| v.to_string())));
        out.push_str(&opt("models_dir", self.models_dir.clone()));
        out.push_str(&format!("auto = {}\n", self.auto));
        out.push_str(&format!("force = {}\n", self.force));
        out.push_str(&format!("mode = {}\n", self.mode.as_str()));
        for (k, v) in &self.unknown {
            out.push_str(&format!("{k} = {v}\n"));
        }
        out
    }

    pub fn load() -> Self {
        std::fs::read_to_string(path())
            .map(|t| Self::parse(&t))
            .unwrap_or_default()
    }

    /// Write, creating the directory. Returns the error text for the status
    /// line rather than swallowing it -- a settings page that cannot save and
    /// does not say so is worse than one that does not exist.
    pub fn save(&self) -> Result<(), String> {
        let p = path();
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        std::fs::write(&p, self.render()).map_err(|e| format!("cannot write {}: {e}", p.display()))
    }

    /// Back to measured everything, without touching the view preference or
    /// any key a newer build wrote.
    ///
    /// A method rather than `..Default::default()` at the call site, because
    /// `unknown` is private: struct-update syntax from outside this module
    /// would not compile, and making the field public to allow it would let any
    /// caller drop the keys it exists to preserve.
    pub fn reset_engine(&mut self) {
        let keep_mode = self.mode;
        let keep_unknown = std::mem::take(&mut self.unknown);
        *self = Settings {
            mode: keep_mode,
            unknown: keep_unknown,
            ..Settings::default()
        };
    }

    /// The arguments these settings imply, for `chaos-serve`.
    ///
    /// One place, so the window and any future headless mode cannot disagree
    /// about what a setting means.
    pub fn serve_args(&self, model: &str) -> Vec<String> {
        let mut a = vec![model.to_string(), "--port".into(), self.port.to_string()];
        if let Some(c) = self.cache_gib {
            a.push("--cache".into());
            a.push(format!("{c}"));
        }
        if let Some(t) = self.threads {
            a.push("-t".into());
            a.push(t.to_string());
        }
        if let Some(t) = self.threads_batch {
            a.push("-tb".into());
            a.push(t.to_string());
        }
        if let Some(c) = self.context {
            a.push("-c".into());
            a.push(c.to_string());
        }
        if let Some(n) = self.ngl {
            a.push("-ngl".into());
            a.push(n.to_string());
        }
        if self.auto {
            a.push("--auto".into());
        }
        if self.force {
            a.push("--force".into());
        }
        a
    }
}

fn truthy(v: &str) -> bool {
    matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_round_trip_preserves_everything() {
        let s = Settings {
            cache_gib: Some(6.5),
            threads: Some(4),
            threads_batch: Some(20),
            port: 9001,
            context: Some(4096),
            ngl: Some(99),
            models_dir: Some(r"D:\models".into()),
            auto: true,
            force: true,
            mode: Mode::Dark,
            ..Settings::default()
        };
        assert_eq!(Settings::parse(&s.render()), s);
    }

    #[test]
    fn defaults_round_trip_too() {
        let s = Settings::default();
        assert_eq!(Settings::parse(&s.render()), s);
    }

    /// **A downgrade must not destroy preferences.** An older build that does
    /// not know a key has to write it back untouched, or running it once loses
    /// whatever the newer one stored.
    #[test]
    fn unknown_keys_survive() {
        let s = Settings::parse("port = 1234\nsomething_new = 7\n");
        assert_eq!(s.port, 1234);
        assert!(s.render().contains("something_new = 7"));
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let s = Settings::parse("# a note\n\n   \nport = 4321\n");
        assert_eq!(s.port, 4321);
    }

    /// A corrupt value must not silently become zero -- port 0 binds a random
    /// port and the endpoint the window shows would be a lie.
    #[test]
    fn a_bad_port_keeps_the_default() {
        assert_eq!(Settings::parse("port = banana").port, 8231);
    }

    #[test]
    fn an_empty_value_means_unset() {
        let s = Settings::parse("cache_gib =\nthreads =\n");
        assert!(s.cache_gib.is_none() && s.threads.is_none());
    }

    #[test]
    fn a_missing_file_gives_defaults() {
        assert_eq!(Settings::parse(""), Settings::default());
    }

    #[test]
    fn serve_args_carry_only_what_is_set() {
        let s = Settings::default();
        let a = s.serve_args("qwen3");
        assert_eq!(a, vec!["qwen3", "--port", "8231"]);
    }

    #[test]
    fn serve_args_include_every_setting() {
        let s = Settings {
            cache_gib: Some(8.0),
            threads: Some(4),
            threads_batch: Some(20),
            context: Some(2048),
            ngl: Some(99),
            auto: true,
            force: true,
            ..Settings::default()
        };
        let a = s.serve_args("m").join(" ");
        for expected in [
            "--cache 8",
            "-t 4",
            "-tb 20",
            "-c 2048",
            "-ngl 99",
            "--auto",
            "--force",
        ] {
            assert!(a.contains(expected), "{expected} missing from {a}");
        }
    }

    /// The model always comes first: `chaos-serve` takes it positionally.
    #[test]
    fn the_model_is_the_first_argument() {
        assert_eq!(Settings::default().serve_args("mymodel")[0], "mymodel");
    }

    /// A reset returns every engine setting to measured, and touches neither
    /// the theme nor a key this build does not understand.
    #[test]
    fn a_reset_keeps_the_theme_and_the_unknown_keys() {
        let mut s = Settings::parse(
            "cache_gib = 9
threads = 12
port = 9999
mode = dark
from_the_future = 7
",
        );
        s.reset_engine();
        assert_eq!(s.cache_gib, None, "cache was not reset");
        assert_eq!(s.threads, None, "threads were not reset");
        assert_eq!(s.port, Settings::default().port, "the port was not reset");
        assert_eq!(s.mode, Mode::Dark, "the reset flipped the lights");
        assert!(
            s.render().contains("from_the_future = 7"),
            "the reset dropped a key a newer build wrote"
        );
    }

    /// Settings live outside the install, so upgrading or uninstalling Chaos
    /// cannot take them with it.
    #[test]
    fn settings_are_not_inside_the_install() {
        let p = path();
        assert!(p.ends_with("settings.txt"));
        assert!(!p
            .to_string_lossy()
            .to_lowercase()
            .contains("localappdata\\chaos\\bin"));
    }
}
