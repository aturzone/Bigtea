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
    /// The key `/v1/*` requires, or `None` for no key at all.
    ///
    /// **Off by default, deliberately.** The server binds `127.0.0.1` only, so
    /// a key is not what keeps a stranger out -- what keeps them out is that
    /// there is no route in. Turning it on by default would also break every
    /// agent already pointed at an existing install. It exists because many
    /// OpenAI-compatible clients insist on sending a key, and because a shared
    /// machine is a real thing.
    pub api_key: Option<String>,
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
            api_key: None,
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
                "api_key" => s.api_key = (!v.is_empty()).then(|| v.to_string()),
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
        out.push_str(&opt("api_key", self.api_key.clone()));
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
        // **`-ngl` is deliberately not sent.** `chaos-serve` refuses it -- its
        // dense path binds weights straight into host memory rather than through
        // the runner's device loader, so there is nowhere on a card to put them
        // yet -- and an unknown flag is an error there now rather than something
        // silently swallowed. Sending it would stop the server from starting.
        //
        // It was sent for three releases and did nothing whatsoever, which is
        // the whole reason the flag is refused loudly today. The setting stays
        // in the file so it survives the wiring; the GPU list says what is true.
        let _ = self.ngl;
        if self.auto {
            a.push("--auto".into());
        }
        if self.force {
            a.push("--force".into());
        }
        if let Some(k) = &self.api_key {
            a.push("--api-key".into());
            a.push(k.clone());
        }
        a
    }

    /// The endpoint, and what to send with it.
    ///
    /// One place, so the line the window shows, the string COPY ENDPOINT puts
    /// on the clipboard, and what the chat client actually sends cannot
    /// disagree -- which is precisely how an endpoint panel comes to advertise
    /// something that does not work.
    pub fn endpoint(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
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
            api_key: Some("deadbeef".into()),
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
            "--auto",
            "--force",
        ] {
            assert!(a.contains(expected), "{expected} missing from {a}");
        }
        // **And nothing the server refuses.** `-ngl` was sent for three
        // releases and silently dropped; it is a hard error there now, so
        // sending it would stop the server from starting at all.
        assert!(
            !a.contains("-ngl"),
            "chaos-serve refuses -ngl -- sending it kills the server: {a}"
        );
    }

    /// The model always comes first: `chaos-serve` takes it positionally.
    #[test]
    fn the_model_is_the_first_argument() {
        assert_eq!(Settings::default().serve_args("mymodel")[0], "mymodel");
    }

    /// A key reaches the server, or it is a decoration on a page.
    #[test]
    fn a_key_is_passed_to_the_server() {
        let mut s = Settings::default();
        assert!(
            !s.serve_args("m").iter().any(|a| a == "--api-key"),
            "no key is set, so none must be passed"
        );
        s.api_key = Some("abc123".into());
        let a = s.serve_args("m");
        let i = a
            .iter()
            .position(|x| x == "--api-key")
            .expect("no --api-key");
        assert_eq!(a[i + 1], "abc123");
    }

    /// The endpoint is built in one place and follows the port.
    #[test]
    fn the_endpoint_follows_the_port() {
        let s = Settings::parse("port = 9313");
        assert_eq!(s.endpoint(), "http://127.0.0.1:9313/v1");
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
