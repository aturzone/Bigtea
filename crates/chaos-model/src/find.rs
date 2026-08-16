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

/// Where models are looked for, in order.
///
/// Two of these exist for real reasons rather than by accident: `install.ps1`
/// creates `~/.chaos/models` for files the user drops in by hand, and
/// [`crate::download::cache_dir`] is where `chaos-pull` writes what it fetches.
/// Searching both means a user never has to know which one a given file came
/// from.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(dir) = std::env::var("CHAOS_MODELS") {
        dirs.push(PathBuf::from(dir));
    }
    let home = if cfg!(windows) {
        std::env::var("USERPROFILE").ok()
    } else {
        std::env::var("HOME").ok()
    };
    if let Some(home) = home {
        dirs.push(PathBuf::from(home).join(".chaos").join("models"));
    }
    dirs.push(crate::download::cache_dir());
    dirs.push(PathBuf::from("models"));
    dirs.dedup();
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
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
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
    }
    out.sort_by_key(|f| f.label.to_lowercase());
    out
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
