//! Fetching a container by name or URL, for `-hf` and `--model-url`.
//!
//! # Why this is not just "shell out to curl"
//!
//! It is that, but the interesting part is what happens around it. A download
//! that half-succeeds is the worst outcome available here: a truncated `.gguf`
//! parses far enough to report a plausible architecture and then fails deep in
//! a forward pass, or — worse — an HTTP error page saved under a `.gguf` name.
//! That is not hypothetical; it is why `bigtea-pull` passes `--fail`, and a 401
//! on a gated repo is exactly how it happens.
//!
//! So every download here is followed by a **magic-number check**, and a file
//! that fails it is deleted rather than left on disk to be picked up by the
//! next run and misdiagnosed as a corrupt model.
//!
//! # Why a cache at all
//!
//! `-hf` is how most people get a model, and re-downloading 17 GiB because a
//! flag changed is not acceptable. Files land in one place, keyed by repo and
//! filename, and `--offline` makes that cache the only source — which is the
//! honest way to run without a network rather than discovering halfway that
//! something wanted to phone home.

use std::path::{Path, PathBuf};
use std::process::Command;

/// GGUF's magic number, little-endian `"GGUF"`.
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Where fetched containers live.
///
/// `BIGTEA_CACHE` wins, then the platform cache directory. Not the current
/// directory: a 17 GiB file dropped wherever the shell happened to be is a
/// surprise, and the second surprise is downloading it again from elsewhere.
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BIGTEA_CACHE") {
        return PathBuf::from(dir);
    }
    let base = if cfg!(windows) {
        std::env::var("LOCALAPPDATA").ok()
    } else {
        std::env::var("XDG_CACHE_HOME")
            .ok()
            .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.cache")))
    };
    PathBuf::from(base.unwrap_or_else(|| ".".into()))
        .join("bigtea")
        .join("models")
}

/// Split llama.cpp's `-hf` shorthand into a repo and an optional filename.
///
/// It accepts `owner/name`, `owner/name:quant` and `owner/name/file.gguf`, and
/// so does this. The `:quant` form cannot be resolved without listing the repo,
/// which needs the HF API rather than a plain GET — so it is reported as such
/// instead of being guessed into a filename that 404s.
pub fn parse_hf(spec: &str) -> Result<(String, Option<String>), String> {
    let (spec, quant) = match spec.split_once(':') {
        Some((r, q)) => (r, Some(q.to_string())),
        None => (spec, None),
    };
    let parts: Vec<&str> = spec.split('/').collect();
    match parts.len() {
        2 => {
            if let Some(q) = quant {
                return Err(format!(
                    "`{spec}:{q}`: resolving a quant name to a file needs the Hugging Face \
                     listing API, which this build does not call. Pass the filename with \
                     --hf-file, or use `owner/name/file.gguf`."
                ));
            }
            Ok((spec.to_string(), None))
        }
        n if n > 2 => {
            let repo = parts[..2].join("/");
            let file = parts[2..].join("/");
            Ok((repo, Some(file)))
        }
        _ => Err(format!(
            "`{spec}` is not a Hugging Face repo. Expected `owner/name`, \
             `owner/name/file.gguf`, or --hf-repo with --hf-file."
        )),
    }
}

/// The URL a repo file resolves to.
pub fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

/// Where a URL's download lands.
pub fn cache_path(repo: Option<&str>, file: &str) -> PathBuf {
    let name = file.rsplit('/').next().unwrap_or(file);
    match repo {
        // Flattened, because a repo name contains a `/` and nesting it makes
        // the cache hard to inspect by hand -- which is the first thing anyone
        // does when a download looks wrong.
        Some(r) => cache_dir().join(format!("{}--{name}", r.replace('/', "--"))),
        None => cache_dir().join(name),
    }
}

/// Whether `path` starts with GGUF's magic number.
///
/// The check that separates "downloaded" from "usable". An HTTP error page and
/// a half-written file both fail it; a valid container passes it in four bytes.
pub fn looks_like_gguf(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).is_ok() && magic == GGUF_MAGIC
}

/// What a resolved `-hf` / `--model-url` turned into.
#[derive(Debug)]
pub struct Fetched {
    pub path: PathBuf,
    /// True when the file was already present and nothing was downloaded.
    pub cached: bool,
}

/// Fetch `url` into the cache, or return the cached copy.
///
/// `token` is used only as a bearer header and is **never printed**, including
/// in the error path — a failed download is exactly when someone pastes the
/// output into an issue.
pub fn fetch(
    url: &str,
    dest: &Path,
    token: Option<&str>,
    offline: bool,
) -> Result<Fetched, String> {
    if dest.exists() && looks_like_gguf(dest) {
        return Ok(Fetched {
            path: dest.to_path_buf(),
            cached: true,
        });
    }
    if offline {
        return Err(format!(
            "--offline and {} is not in the cache. Remove --offline to download it.",
            dest.display()
        ));
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {parent:?}: {e}"))?;
    }
    if Command::new("curl").arg("--version").output().is_err() {
        return Err(
            "curl was not found on PATH, and it is how Bigtea downloads. Install curl, \
             or fetch the file by hand and pass its path."
                .into(),
        );
    }

    let mut cmd = Command::new("curl");
    // `-C -` resumes a partial file; `-L` follows the CDN redirect Hugging Face
    // issues; `--fail` turns an HTTP error into a failure rather than a saved
    // error page, which is how a 401 on a gated repo becomes a corrupt .gguf.
    cmd.args([
        "-L",
        "--fail",
        "-C",
        "-",
        "--retry",
        "5",
        "--retry-delay",
        "5",
    ]);
    if let Some(t) = token
        .map(str::to_string)
        .or_else(|| std::env::var("HF_TOKEN").ok())
    {
        cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
    }
    cmd.arg("-o").arg(dest).arg(url);

    let status = cmd
        .status()
        .map_err(|e| format!("could not run curl: {e}"))?;
    if !status.success() {
        return Err(format!(
            "curl failed on {url} ({status}). Re-run to resume; if the repo is gated, \
             pass --hf-token or set HF_TOKEN."
        ));
    }
    if !looks_like_gguf(dest) {
        // Deleted rather than left behind: a file that is not a container must
        // not survive to be re-read next run and misdiagnosed as a corrupt
        // model. Most often it is an HTML error page with a .gguf name.
        let _ = std::fs::remove_file(dest);
        return Err(format!(
            "{url} did not return a GGUF container (the first four bytes are not `GGUF`). \
             The partial file has been removed. A gated repo usually returns an HTML page \
             here -- pass --hf-token."
        ));
    }
    Ok(Fetched {
        path: dest.to_path_buf(),
        cached: false,
    })
}

/// Every container in the cache, with its size.
pub fn cached_files() -> Vec<(PathBuf, u64)> {
    let Ok(entries) = std::fs::read_dir(cache_dir()) else {
        return Vec::new();
    };
    let mut out: Vec<(PathBuf, u64)> = entries
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            let len = e.metadata().ok()?.len();
            p.extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
                .then_some((p, len))
        })
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_shorthand_splits_repo_from_file() {
        assert_eq!(
            parse_hf("unsloth/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf").unwrap(),
            (
                "unsloth/gemma-3-1b-it-GGUF".into(),
                Some("gemma-3-1b-it-Q4_K_M.gguf".into())
            )
        );
        assert_eq!(
            parse_hf("unsloth/gemma-3-1b-it-GGUF").unwrap(),
            ("unsloth/gemma-3-1b-it-GGUF".into(), None)
        );
    }

    #[test]
    fn a_quant_suffix_is_refused_rather_than_guessed() {
        // `:Q4_K_M` needs the listing API to become a filename. Guessing
        // `<name>-Q4_K_M.gguf` is right for some repos and a 404 for others,
        // and a 404 saved under a .gguf name is the exact failure this module
        // exists to prevent.
        let e = parse_hf("unsloth/gemma-3-1b-it-GGUF:Q4_K_M").unwrap_err();
        assert!(e.contains("--hf-file"), "{e}");
    }

    #[test]
    fn a_bare_name_is_not_a_repo() {
        assert!(parse_hf("gemma").is_err());
    }

    #[test]
    fn the_url_is_the_resolve_endpoint() {
        assert_eq!(
            hf_url("a/b", "c.gguf"),
            "https://huggingface.co/a/b/resolve/main/c.gguf"
        );
    }

    #[test]
    fn a_repo_name_is_flattened_into_the_filename() {
        // Nesting `owner/name/` would make the cache a directory tree, and the
        // first thing anyone does with a suspect download is look at it.
        let p = cache_path(Some("owner/name"), "m.gguf");
        assert!(p.to_string_lossy().contains("owner--name--m.gguf"), "{p:?}");
    }

    #[test]
    fn a_file_that_is_not_gguf_is_not_mistaken_for_one() {
        let dir = std::env::temp_dir().join("bigtea-dl-test");
        let _ = std::fs::create_dir_all(&dir);
        let bad = dir.join("notgguf.gguf");
        std::fs::write(&bad, b"<html>401</html>").unwrap();
        assert!(!looks_like_gguf(&bad));
        let good = dir.join("ok.gguf");
        std::fs::write(&good, b"GGUF\x03\x00\x00\x00").unwrap();
        assert!(looks_like_gguf(&good));
        let _ = std::fs::remove_file(bad);
        let _ = std::fs::remove_file(good);
    }

    #[test]
    fn offline_refuses_rather_than_downloading() {
        let missing = cache_dir().join("definitely-not-here-9f3a.gguf");
        let e = fetch("https://example.invalid/x.gguf", &missing, None, true).unwrap_err();
        assert!(e.contains("--offline"), "{e}");
    }
}
