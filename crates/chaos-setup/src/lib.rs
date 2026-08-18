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
