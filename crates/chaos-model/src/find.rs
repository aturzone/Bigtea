//! Finding a model by name instead of by path.
//!
//! # Why this exists
//!
//! Every command in this project used to begin with an absolute path to a
//! `.gguf` file, which on Windows means something like
//! `C:\Users\you\.chaos\models\Qwen3-30B-A3B-Q4_K_M.gguf` typed by hand. For a
//! 144 GB five-shard container it means knowing which shard to name. That is a
//! wall in front of the first thing a new user does, and it is a wall made
//! entirely of typing.
//!
//! So a name that is not a path is looked up: `chaos-run qwen3` finds
//! `Qwen3-30B-A3B-Q4_K_M.gguf` if exactly one model matches, and lists the
//! candidates if several do. **An existing path always wins** — the lookup only
//! runs when the argument is not a file, so nothing that worked before changes.

use std::path::{Path, PathBuf};

/// The user's home, whatever this platform calls it.
fn home() -> Option<PathBuf> {
    let key = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var(key).ok().map(PathBuf::from)
}

/// `models_dir` from `~/.chaos/settings.txt`, which the app writes.
///
/// **Read here rather than in the app** so that setting it moves `chaos-run`
/// and `chaos-serve` too. It was written to the file and consulted by nothing,
/// which is the worst of the three possible behaviours: the setting looked like
/// it worked. The file is `key = value` lines; anything else is skipped.
fn dirs_from_settings() -> Vec<PathBuf> {
    let Some(file) = home().map(|h| h.join(".chaos").join("settings.txt")) else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(file) else {
        return Vec::new();
    };
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        if k.trim() == "models_dir" {
            return split_dirs(v.trim());
        }
    }
    Vec::new()
}

/// One setting, several folders.
///
/// **A machine with a big model on a second drive is the normal case, not an
/// exotic one** -- a 144 GB container does not live beside a 2 GB one. So both
/// `CHAOS_MODELS` and the `models_dir` setting take a list in the platform's
/// own separator (`;` on Windows, `:` elsewhere), which is what `split_paths`
/// implements and what a `PATH` has always meant. A single folder still reads
/// as a single folder, so nothing that already worked changes.
fn split_dirs(value: &str) -> Vec<PathBuf> {
    std::env::split_paths(value)
        .filter(|p| !p.as_os_str().is_empty())
        .collect()
}

/// Where models are looked for, in order.
///
/// Two of these exist for real reasons rather than by accident: `install.ps1`
/// creates `~/.chaos/models` for files the user drops in by hand, and
/// [`crate::download::cache_dir`] is where `chaos-pull` writes what it fetches.
/// Searching both means a user never has to know which one a given file came
/// from.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(value) = std::env::var("CHAOS_MODELS") {
        dirs.extend(split_dirs(&value));
    }
    dirs.extend(dirs_from_settings());
    if let Some(home) = home() {
        dirs.push(home.join(".chaos").join("models"));
    }
    dirs.push(crate::download::cache_dir());
    dirs.push(PathBuf::from("models"));
    // `Vec::dedup` only removes *adjacent* duplicates, so the cache directory
    // appearing both in the setting and by default survived it twice.
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|d| seen.insert(d.clone()));
    dirs
}

/// One model on disk: the shard to open, and the name to show a user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Found {
    pub path: PathBuf,
    /// The file stem, with any `-00001-of-00005` suffix removed.
    pub label: String,
}

/// Every model in every search directory, sorted by label.
///
/// **A sharded container appears once, as its first shard.** Listing five
/// shards of one model as five models is both wrong and the more confusing of
/// the two possible mistakes — the runner discovers the rest from any one of
/// them, so only the first is ever the thing to open.
pub fn list() -> Vec<Found> {
    let mut out: Vec<Found> = Vec::new();
    for dir in model_dirs() {
        scan_into(&dir, true, &mut out);
    }
    out.sort_by_key(|f| f.label.to_lowercase());
    out
}

/// Every `.gguf` directly in `dir`, and -- once -- in each of its children.
///
/// **A big sharded model lives in its own folder.** Five files that together
/// weigh 144 GB are not dropped beside a 2 GB one; they are put in
/// `models/v4flash/`, which is where `chaos-pull` puts a multi-shard fetch and
/// where a user who moved one by hand puts it too. A scan that stopped at the
/// top level reported "no models installed" with the model plainly there.
///
/// One level, not a walk: a models folder pointed at a whole drive would
/// otherwise read every directory on it.
fn scan_into(dir: &Path, descend: bool, out: &mut Vec<Found>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut children = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if descend {
                children.push(path);
            }
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let (label, shard) = split_shard(stem);
        if shard.is_some_and(|n| n != 1) {
            continue; // a later shard of a container already listed
        }
        if out.iter().any(|f| f.label == label) {
            continue; // the same model in two search directories
        }
        out.push(Found { path, label });
    }
    children.sort();
    for c in children {
        scan_into(&c, false, out);
    }
}

/// `("Model-00002-of-00005")` becomes `("Model", Some(2))`.
///
/// Written by hand rather than with a regex because this workspace has no
/// dependencies, and the shape is fixed enough that a parser is three lines.
fn split_shard(stem: &str) -> (String, Option<u32>) {
    // ...-NNNNN-of-NNNNN
    let mut parts = stem.rsplitn(4, '-');
    let total = parts.next();
    let of = parts.next();
    let index = parts.next();
    let head = parts.next();
    if let (Some(head), Some(index), Some("of"), Some(total)) = (head, index, of, total) {
        if index.len() == 5 && total.len() == 5 {
            if let Ok(n) = index.parse::<u32>() {
                return (head.to_string(), Some(n));
            }
        }
    }
    (stem.to_string(), None)
}

/// Why a name could not be turned into exactly one model.
#[derive(Debug)]
pub enum FindError {
    /// Nothing matched. Carries everything that *is* available, so the caller
    /// can show it rather than making the user go looking.
    NotFound { available: Vec<Found> },
    /// Several matched, so guessing would be a coin flip on a 144 GB read.
    Ambiguous { matches: Vec<Found> },
}

/// The labels a catalogue name's own downloads would appear under, lowercased.
///
/// One per quantisation, most-preferred first, taken from the filenames the
/// entry itself generates rather than from any second rule about naming.
fn catalogue_labels(name: &str) -> Vec<String> {
    let Some(e) = crate::catalogue::find(name) else {
        return Vec::new();
    };
    e.quants
        .iter()
        .filter_map(|q| e.files(q).into_iter().next())
        .map(|f| split_shard(f.trim_end_matches(".gguf")).0.to_lowercase())
        .collect()
}

/// Turn a user's argument into a path to open.
///
/// An existing path is returned unchanged — including a relative one, and
/// including a file with no `.gguf` extension — so this can be called
/// unconditionally without changing any behaviour that already worked. Only
/// when the argument is not a file is it treated as a name and matched, first
/// exactly against a label, then as a case-insensitive substring.
pub fn resolve(arg: &str) -> Result<PathBuf, FindError> {
    let direct = Path::new(arg);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }
    let available = list();
    let wanted = arg.trim_end_matches(".gguf").to_lowercase();

    // Exact first: a user who typed the whole label means that model, even if
    // it is also a substring of a longer one.
    if let Some(f) = available.iter().find(|f| f.label.to_lowercase() == wanted) {
        return Ok(f.path.clone());
    }

    // **The catalogue name is a name too.** `chaos-pull v4flash` fetches
    // `DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf`, and then `chaos-run
    // v4flash` said "no model called v4flash" while listing the file it had
    // just downloaded. The name a user is told to type must be the name that
    // works, so a catalogue entry is resolved to the labels its own filenames
    // produce. Tried after the exact label match, so a file actually called
    // `v4flash.gguf` still wins.
    for label in catalogue_labels(&wanted) {
        if let Some(f) = available.iter().find(|f| f.label.to_lowercase() == label) {
            return Ok(f.path.clone());
        }
    }

    let matches: Vec<Found> = available
        .iter()
        .filter(|f| f.label.to_lowercase().contains(&wanted))
        .cloned()
        .collect();
    match matches.len() {
        1 => Ok(matches[0].path.clone()),
        0 => Err(FindError::NotFound { available }),
        _ => Err(FindError::Ambiguous { matches }),
    }
}

/// The message to print for a [`FindError`], including where it looked.
///
/// Built here rather than in each binary so `chaos-run` and `chaos-serve`
/// cannot drift into telling a user two different things about the same
/// directories.
pub fn explain(arg: &str, err: &FindError) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    match err {
        FindError::Ambiguous { matches } => {
            let _ = writeln!(s, "{arg:?} matches {} models:", matches.len());
            for f in matches {
                let _ = writeln!(s, "    {}", f.label);
            }
            let _ = write!(s, "\n  Name one of them, or give the path.");
        }
        FindError::NotFound { available } if available.is_empty() => {
            let _ = writeln!(s, "no model called {arg:?}, and no models found at all.");
            let _ = writeln!(s, "\n  Looked in:");
            for d in model_dirs() {
                let _ = writeln!(s, "    {}", d.display());
            }
            let _ = write!(
                s,
                "\n  Put a .gguf file in the first of those, or give a path.\n  \
                 Chaos downloads nothing on its own."
            );
        }
        FindError::NotFound { available } => {
            let _ = writeln!(s, "no model called {arg:?}. Available:");
            for f in available {
                let _ = writeln!(s, "    {}", f.label);
            }
            let _ = write!(s, "\n  Any unique part of a name works.");
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **One setting, several folders.** A 144 GB container does not live
    /// beside a 2 GB one, so the folder list has to be a list -- and the
    /// separator is the platform's own, the one a `PATH` has always used.
    #[test]
    fn a_folder_setting_may_name_several() {
        let sep = if cfg!(windows) { ";" } else { ":" };
        let joined = format!("{}{sep}{}", "one", "two");
        let dirs = split_dirs(&joined);
        assert_eq!(dirs.len(), 2, "{dirs:?}");
        assert_eq!(dirs[0], PathBuf::from("one"));
        assert_eq!(dirs[1], PathBuf::from("two"));
    }

    /// A single folder still reads as a single folder, so nothing that already
    /// worked changes. On Windows this also means a drive letter survives:
    /// splitting on `:` would have cut `C:\\models` in half.
    #[test]
    fn one_folder_stays_one_folder() {
        let one = if cfg!(windows) {
            "C:\\models"
        } else {
            "/models"
        };
        assert_eq!(split_dirs(one), vec![PathBuf::from(one)]);
    }

    #[test]
    fn an_empty_setting_names_no_folder() {
        assert!(split_dirs("").is_empty());
    }

    /// `Vec::dedup` only removes *adjacent* duplicates, so the same directory
    /// named by the setting and again by default survived it and every model
    /// in it was skipped the second time by label -- which worked, but only by
    /// accident. The search order must hold each directory exactly once.
    #[test]
    fn a_directory_is_searched_once() {
        let dirs = model_dirs();
        let mut seen = std::collections::HashSet::new();
        for d in &dirs {
            assert!(seen.insert(d.clone()), "{d:?} is in the search order twice");
        }
    }

    /// **The name a user is told to type must be the name that works.**
    /// `chaos-pull v4flash` writes `DeepSeek-V4-Flash-UD-...gguf`, and then
    /// `chaos-run v4flash` answered "no model called v4flash" while listing
    /// that very file. The catalogue name has to resolve to what it fetched.
    #[test]
    fn a_catalogue_name_maps_to_the_label_it_downloads_as() {
        assert_eq!(
            catalogue_labels("v4flash"),
            vec!["deepseek-v4-flash-ud-q4_k_xl".to_string()]
        );
        assert_eq!(
            catalogue_labels("qwen3-8b"),
            vec!["qwen3-8b-q4_k_m".to_string()]
        );
        // One per quantisation, so either download resolves.
        assert_eq!(catalogue_labels("qwen3-32b").len(), 2);
        assert!(catalogue_labels("not-in-the-catalogue").is_empty());
    }

    /// Every catalogue name must produce at least one label, or `chaos-run
    /// <name>` silently falls through to substring matching for that entry.
    #[test]
    fn every_catalogue_name_produces_a_label() {
        for e in crate::catalogue::CATALOGUE {
            let labels = catalogue_labels(e.name);
            assert!(!labels.is_empty(), "{} produces no label", e.name);
            for l in labels {
                assert!(!l.contains("-of-"), "{} kept a shard suffix: {l}", e.name);
            }
        }
    }

    /// A model in its own folder is found, one level down and no further.
    ///
    /// This is how a five-shard container is stored, and a scan that stopped at
    /// the top level reported "no models installed" with 145 GB plainly there.
    #[test]
    fn a_model_in_its_own_folder_is_found() {
        let root = std::env::temp_dir().join("chaos-find-nested-9c2");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("big")).unwrap();
        std::fs::create_dir_all(root.join("big").join("deeper")).unwrap();
        std::fs::write(root.join("top.gguf"), b"x").unwrap();
        std::fs::write(root.join("big").join("nested.gguf"), b"x").unwrap();
        std::fs::write(root.join("big").join("deeper").join("far.gguf"), b"x").unwrap();

        let mut out = Vec::new();
        scan_into(&root, true, &mut out);
        let labels: Vec<&str> = out.iter().map(|f| f.label.as_str()).collect();
        assert!(labels.contains(&"top"), "{labels:?}");
        assert!(labels.contains(&"nested"), "{labels:?}");
        // Two levels down is not searched: a models folder pointed at a whole
        // drive would otherwise read every directory on it.
        assert!(!labels.contains(&"far"), "{labels:?}");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A sharded container in a subfolder still appears once, as shard one.
    #[test]
    fn a_sharded_container_in_a_subfolder_is_one_entry() {
        let root = std::env::temp_dir().join("chaos-find-nested-shards-9c2");
        let _ = std::fs::remove_dir_all(&root);
        let sub = root.join("v4flash");
        std::fs::create_dir_all(&sub).unwrap();
        for i in 1..=5 {
            std::fs::write(sub.join(format!("Big-{i:05}-of-00005.gguf")), b"x").unwrap();
        }

        let mut out = Vec::new();
        scan_into(&root, true, &mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].label, "Big");
        assert!(out[0].path.ends_with("Big-00001-of-00005.gguf"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_shard_suffix_is_stripped_and_numbered() {
        assert_eq!(
            split_shard("DeepSeek-V4-Flash-UD-Q4_K_XL-00002-of-00005"),
            ("DeepSeek-V4-Flash-UD-Q4_K_XL".into(), Some(2))
        );
        assert_eq!(
            split_shard("DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005"),
            ("DeepSeek-V4-Flash-UD-Q4_K_XL".into(), Some(1))
        );
    }

    /// A single-file model must not lose part of its name to the shard parser.
    /// `Qwen3-30B-A3B-Q4_K_M` has four `-` and would be mangled by a naive
    /// split.
    #[test]
    fn an_unsharded_name_is_left_alone() {
        for name in [
            "Qwen3-30B-A3B-Q4_K_M",
            "Llama-3.2-1B-Instruct-Q4_K_M",
            "model",
            "a-b-of-c",
            "x-00001-of-005",
        ] {
            assert_eq!(split_shard(name), (name.to_string(), None), "{name}");
        }
    }

    /// The lookup must never fire for something that is already a file, or a
    /// user with a model outside the search directories would stop being able
    /// to open it.
    #[test]
    fn an_existing_path_wins_over_any_name_matching() {
        let dir = std::env::temp_dir().join("chaos-find-test");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("plain.gguf");
        std::fs::write(&file, b"x").unwrap();
        let got = resolve(file.to_str().unwrap()).unwrap();
        assert_eq!(got, file);
        let _ = std::fs::remove_file(&file);
    }

    #[test]
    fn model_dirs_are_distinct_and_non_empty() {
        let dirs = model_dirs();
        assert!(!dirs.is_empty());
        for (i, a) in dirs.iter().enumerate() {
            for b in dirs.iter().skip(i + 1) {
                assert_ne!(a, b, "duplicate search directory");
            }
        }
    }
}
