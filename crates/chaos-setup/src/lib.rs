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

    /// Models must never live inside the prefix, or uninstalling could delete
    /// a download measured in hundreds of gigabytes.
    #[test]
    fn models_are_not_inside_the_prefix() {
        let p = default_prefix();
        let m = default_models_dir();
        assert!(!m.starts_with(&p), "{m:?} is inside {p:?}");
    }
}
