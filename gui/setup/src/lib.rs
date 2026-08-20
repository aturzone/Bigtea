//! The parts of installing that are not a window.
//!
//! Kept out of `main.rs` so they can be tested anywhere: an installer is
//! exactly the program you cannot afford to debug on a user's machine, and the
//! decisions it makes -- where to put things, what a previous install left
//! behind, what to remove -- are all plain data.

use std::path::{Path, PathBuf};

/// The default install location: per-user, so no administrator prompt.
///
/// `%LOCALAPPDATA%\Chaos`. Program Files would need elevation, and elevation
/// for something that writes only its own folder is a habit worth not teaching.
pub fn default_prefix() -> PathBuf {
    std::env::var("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(r"C:\Chaos"))
        .join("Chaos")
}

/// Where models go. Never inside the prefix: uninstalling must not be able to
/// delete a 155 GB download, and an upgrade must not touch it.
pub fn default_models_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(r"C:\"))
        .join(".chaos")
        .join("models")
}

pub fn bin_dir(prefix: &Path) -> PathBuf {
    prefix.join("bin")
}

/// What the last install wrote, so an upgrade can remove what this one drops.
pub fn manifest_path(prefix: &Path) -> PathBuf {
    prefix.join("installed-files.txt")
}

/// Add `dir` to a PATH string, without duplicating it.
///
/// Compares case-insensitively and ignores a trailing separator, because
/// `C:\X` and `c:\x\` are the same directory and appending both is how a PATH
/// grows unboundedly across upgrades.
pub fn path_with(existing: &str, dir: &str) -> String {
    let norm = |s: &str| s.trim().trim_end_matches('\\').to_lowercase();
    let target = norm(dir);
    if existing
        .split(';')
        .any(|e| !e.trim().is_empty() && norm(e) == target)
    {
        return existing.to_string();
    }
    if existing.trim().is_empty() {
        dir.to_string()
    } else {
        format!("{};{}", existing.trim_end_matches(';'), dir)
    }
}

/// Remove `dir` from a PATH string, leaving everything else in order.
pub fn path_without(existing: &str, dir: &str) -> String {
    let norm = |s: &str| s.trim().trim_end_matches('\\').to_lowercase();
    let target = norm(dir);
    existing
        .split(';')
        .filter(|e| !e.trim().is_empty() && norm(e) != target)
        .collect::<Vec<_>>()
        .join(";")
}

/// What is already installed at `prefix`, if anything.
///
/// Read from the manifest the last install wrote plus the recorded version, so
/// an upgrade can say what it is replacing rather than silently overwriting.
pub struct Existing {
    pub version: Option<String>,
    pub files: usize,
}

pub fn existing_install(prefix: &Path) -> Option<Existing> {
    let bin = bin_dir(prefix);
    if !bin.exists() {
        return None;
    }
    let files = std::fs::read_dir(&bin).map(|d| d.count()).unwrap_or(0);
    if files == 0 {
        return None;
    }
    Some(Existing {
        version: std::fs::read_to_string(version_path(prefix))
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty()),
        files,
    })
}

/// Where the installed version is recorded, so an upgrade can name it.
pub fn version_path(prefix: &Path) -> PathBuf {
    prefix.join("version.txt")
}

/// How an upgrade should describe itself.
pub fn upgrade_line(before: Option<&Existing>, to: &str) -> String {
    match before {
        None => format!("Installing Chaos {to}."),
        Some(e) => match &e.version {
            Some(v) if v == to => format!("Reinstalling Chaos {to} over the same version."),
            Some(v) => format!("Upgrading Chaos {v} -> {to}."),
            None => format!("Upgrading an older install ({} files) to {to}.", e.files),
        },
    }
}

/// Everything the install writes into the prefix itself, outside `bin`.
///
/// **One list, so both ends agree.** The uninstall had its own copy and
/// `version.txt` was not in it, so the prefix was never empty, `remove_dir`
/// always failed, and a stale version file was left behind claiming Chaos was
/// installed when nothing was. That leftover was found on a real machine, not
/// in a test.
pub fn prefix_files(prefix: &Path) -> Vec<PathBuf> {
    vec![
        manifest_path(prefix),
        version_path(prefix),
        prefix.join("setup.log"),
    ]
}

/// What the primary button says, and the line above it, before anything is
/// written.
///
/// The counterpart to [`upgrade_line`], which describes the same thing
/// afterwards. Both exist because the report alone was not enough: a user who
/// installs a new version over an old one wants to know it is an update
/// *before* pressing the button.
pub fn welcome_action(before: Option<&Existing>, to: &str) -> (&'static str, String) {
    match before {
        None => ("INSTALL", String::new()),
        Some(e) => match &e.version {
            Some(v) if v == to => (
                "REINSTALL",
                format!("Chaos {v} is already installed here. This reinstalls it."),
            ),
            Some(v) => (
                "UPDATE",
                format!("Chaos {v} is installed here. This updates it to {to}."),
            ),
            // Installed before the version file existed, so the file count is
            // all there is to go on. Still an update, and still worth saying.
            None => (
                "UPDATE",
                format!(
                    "An older Chaos ({} files) is installed here. This updates it to {to}.",
                    e.files
                ),
            ),
        },
    }
}

/// Which of the previously installed names this version no longer ships.
pub fn stale(previous: &[String], incoming: &[String]) -> Vec<String> {
    previous
        .iter()
        .filter(|p| !incoming.iter().any(|i| i.eq_ignore_ascii_case(p)))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_is_not_duplicated() {
        let p = r"C:\a;C:\b";
        assert_eq!(path_with(p, r"C:\a"), p);
        assert_eq!(
            path_with(p, r"c:\A\"),
            p,
            "case and trailing slash differ, directory does not"
        );
        assert_eq!(path_with(p, r"C:\c"), r"C:\a;C:\b;C:\c");
    }

    #[test]
    fn an_empty_path_becomes_just_the_directory() {
        assert_eq!(path_with("", r"C:\x"), r"C:\x");
        assert_eq!(path_with("   ", r"C:\x"), r"C:\x");
    }

    #[test]
    fn removal_keeps_order_and_drops_empties() {
        assert_eq!(path_without(r"C:\a;C:\b;C:\c", r"C:\b"), r"C:\a;C:\c");
        assert_eq!(path_without(r"C:\a;;C:\b", r"C:\a"), r"C:\b");
    }

    /// An upgrade must delete a binary this version dropped, or a stale one
    /// stays on PATH shadowing nothing and confusing every later report.
    #[test]
    fn an_upgrade_finds_what_to_remove() {
        let prev = vec!["chaos-run.exe".to_string(), "old-tool.exe".to_string()];
        let now = vec!["chaos-run.exe".to_string(), "chaos-app.exe".to_string()];
        assert_eq!(stale(&prev, &now), vec!["old-tool.exe".to_string()]);
    }

    #[test]
    fn nothing_is_stale_when_nothing_was_dropped() {
        let v = vec!["a.exe".to_string()];
        assert!(stale(&v, &v).is_empty());
    }

    #[test]
    fn an_upgrade_says_what_it_replaces() {
        let from = Existing {
            version: Some("0.0.4".into()),
            files: 17,
        };
        assert_eq!(
            upgrade_line(Some(&from), "0.0.6"),
            "Upgrading Chaos 0.0.4 -> 0.0.6."
        );
    }

    #[test]
    fn a_fresh_install_does_not_claim_to_upgrade() {
        assert_eq!(upgrade_line(None, "0.0.6"), "Installing Chaos 0.0.6.");
    }

    /// Re-running the same installer is a legitimate thing to do -- repairing a
    /// broken install -- and should not be described as an upgrade.
    #[test]
    fn the_same_version_is_a_reinstall() {
        let same = Existing {
            version: Some("0.0.6".into()),
            files: 17,
        };
        assert!(upgrade_line(Some(&same), "0.0.6").contains("Reinstalling"));
    }

    /// Installs from before the version file existed still have to be handled.
    #[test]
    fn an_unversioned_install_is_still_an_upgrade() {
        let old = Existing {
            version: None,
            files: 12,
        };
        let line = upgrade_line(Some(&old), "0.0.6");
        assert!(line.contains("older install"), "{line}");
        assert!(line.contains("12 files"), "{line}");
    }

    #[test]
    fn the_version_file_sits_beside_the_manifest() {
        let p = Path::new("X:/prefix");
        assert_eq!(version_path(p).parent(), manifest_path(p).parent());
    }

    /// Models must never live inside the prefix, or uninstalling could delete
    /// a download measured in hundreds of gigabytes.
    #[test]
    fn models_are_not_inside_the_prefix() {
        let p = default_prefix();
        let m = default_models_dir();
        assert!(!m.starts_with(&p), "{m:?} is inside {p:?}");
    }
}

// -- the install, as steps ---------------------------------------------------

/// One thing the installer does, and how it went.
///
/// The install used to be a single call that returned a paragraph when it was
/// over. On a fast machine that is fine; on a slow one it is a frozen window,
/// and either way it never says *what* it is doing. Hermes' installer shows a
/// named list with a tick and a duration against each line, which is both nicer
/// and far easier to send back when something fails -- so this is that.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// What the user sees. A verb in the present participle: "Writing chaos-run".
    pub label: String,
    pub state: StepState,
    /// How long it took, once it is done.
    pub millis: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepState {
    Waiting,
    Running,
    Done,
    Failed,
}

impl Step {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            state: StepState::Waiting,
            millis: 0,
        }
    }
}

/// The whole run: the steps, and where it got to.
#[derive(Clone, Debug, Default)]
pub struct Progress {
    pub steps: Vec<Step>,
    /// Set when a step fails, and the reason. The run stops here.
    pub failure: Option<String>,
    /// The closing report, once every step is done.
    pub report: Vec<String>,
}

impl Progress {
    pub fn done_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.state == StepState::Done)
            .count()
    }

    /// Completion as a percentage, for the bar.
    pub fn percent(&self) -> u32 {
        if self.steps.is_empty() {
            return 0;
        }
        (self.done_count() * 100 / self.steps.len()) as u32
    }

    pub fn finished(&self) -> bool {
        self.failure.is_some() || (!self.steps.is_empty() && self.done_count() == self.steps.len())
    }

    /// The step currently running, for the line the eye should be on.
    pub fn running(&self) -> Option<usize> {
        self.steps
            .iter()
            .position(|s| s.state == StepState::Running)
    }
}

/// A duration as the installer prints it: `405ms`, `1.2s`.
///
/// Milliseconds up to a second because most of these steps are a file write and
/// "0.4s" reads as slower than it is; seconds beyond, because "12480ms" does
/// not.
pub fn human_millis(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}

#[cfg(test)]
mod step_tests {
    use super::*;

    #[test]
    fn an_empty_run_is_not_finished_and_is_not_complete() {
        let p = Progress::default();
        assert_eq!(p.percent(), 0);
        assert!(!p.finished(), "a run with no steps must not report success");
    }

    #[test]
    fn percent_tracks_completed_steps() {
        let mut p = Progress {
            steps: (0..4).map(|i| Step::new(format!("s{i}"))).collect(),
            ..Progress::default()
        };
        assert_eq!(p.percent(), 0);
        p.steps[0].state = StepState::Done;
        assert_eq!(p.percent(), 25);
        p.steps[1].state = StepState::Running;
        assert_eq!(p.percent(), 25, "a running step is not a completed one");
        for s in p.steps.iter_mut() {
            s.state = StepState::Done;
        }
        assert_eq!(p.percent(), 100);
        assert!(p.finished());
    }

    /// **A failed run must never read as finished-successfully.** It is
    /// finished, but `failure` is what the screen keys off.
    #[test]
    fn a_failure_stops_the_run_and_is_visible() {
        let mut p = Progress {
            steps: (0..3).map(|i| Step::new(format!("s{i}"))).collect(),
            ..Progress::default()
        };
        p.steps[0].state = StepState::Done;
        p.steps[1].state = StepState::Failed;
        p.failure = Some("cannot write chaos-run.exe".into());
        assert!(p.finished());
        assert!(p.failure.is_some());
        assert!(p.percent() < 100, "a failed run must not show 100%");
    }

    #[test]
    fn the_running_step_is_findable() {
        let mut p = Progress {
            steps: (0..3).map(|i| Step::new(format!("s{i}"))).collect(),
            ..Progress::default()
        };
        assert_eq!(p.running(), None);
        p.steps[1].state = StepState::Running;
        assert_eq!(p.running(), Some(1));
    }

    #[test]
    fn durations_read_as_durations() {
        assert_eq!(human_millis(0), "0ms");
        assert_eq!(human_millis(405), "405ms");
        assert_eq!(human_millis(999), "999ms");
        assert_eq!(human_millis(1000), "1.0s");
        assert_eq!(human_millis(12480), "12.5s");
    }

    #[test]
    fn the_uninstall_removes_the_version_file() {
        // The bug this guards: `version.txt` was written by the install and not
        // removed by the uninstall, so the prefix could never be deleted and a
        // stale version file said Chaos was installed when it was not.
        let dir = Path::new("C:").join("nowhere");
        let names: Vec<String> = prefix_files(&dir)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"version.txt".to_string()), "{names:?}");
        assert!(
            names.contains(&"installed-files.txt".to_string()),
            "{names:?}"
        );
        assert!(names.contains(&"setup.log".to_string()), "{names:?}");
    }

    #[test]
    fn a_second_install_says_it_is_an_update() {
        // Nothing there: the plain case.
        let (label, notice) = welcome_action(None, "0.0.9");
        assert_eq!(label, "INSTALL");
        assert!(notice.is_empty());

        // An older version: the button has to say so before it is pressed.
        let old = Existing {
            version: Some("0.0.8".into()),
            files: 9,
        };
        let (label, notice) = welcome_action(Some(&old), "0.0.9");
        assert_eq!(label, "UPDATE");
        assert!(
            notice.contains("0.0.8") && notice.contains("0.0.9"),
            "{notice}"
        );

        // The same version is a reinstall, and saying "update" there would be a
        // lie the user could check.
        let same = Existing {
            version: Some("0.0.9".into()),
            files: 9,
        };
        assert_eq!(welcome_action(Some(&same), "0.0.9").0, "REINSTALL");

        // Installed before version.txt existed: still an update.
        let unknown = Existing {
            version: None,
            files: 7,
        };
        let (label, notice) = welcome_action(Some(&unknown), "0.0.9");
        assert_eq!(label, "UPDATE");
        assert!(notice.contains('7'), "{notice}");
    }
}
