//! What the app knows about the models on the machine.
//!
//! The discovery itself is `chaos_model::find`, shared with `chaos-run` and
//! `chaos-serve` so the three cannot disagree about where models live -- that
//! disagreement was a real bug once, when the installer created one directory
//! and the downloader wrote to another.

use std::path::PathBuf;

/// A model as the list shows it.
pub struct Entry {
    pub label: String,
    pub path: PathBuf,
    /// Total bytes across every shard, or `None` if it could not be measured.
    pub bytes: Option<u64>,
    /// Why this container cannot be loaded, if it cannot.
    ///
    /// **A half-downloaded model looks exactly like a finished one in a list**
    /// -- right name, right extension, valid header -- and the only way a user
    /// found out was to press LOAD and read whatever the engine said three
    /// seconds later. Two of the models on this machine were in that state.
    pub incomplete: Option<String>,
}

/// Human-readable size, at the precision the number deserves.
///
/// Two decimals below 10, one below 100, none above: "144 GB" is more useful
/// than "144.42 GB", and "9.34 GB" is more useful than "9 GB".
pub fn human_size(bytes: u64) -> String {
    const K: f64 = 1000.0;
    let b = bytes as f64;
    let (v, unit) = if b >= K * K * K {
        (b / (K * K * K), "GB")
    } else if b >= K * K {
        (b / (K * K), "MB")
    } else if b >= K {
        (b / K, "kB")
    } else {
        return format!("{bytes} B");
    };
    if v < 10.0 {
        format!("{v:.2} {unit}")
    } else if v < 100.0 {
        format!("{v:.1} {unit}")
    } else {
        format!("{v:.0} {unit}")
    }
}

/// Where the app puts what it downloads.
///
/// The same `~/.chaos/models` the installer creates and `find` searches first
/// after `CHAOS_MODELS`, so a download appears in the list without any
/// configuration. Getting this wrong once already cost a release: the installer
/// made one directory and the downloader wrote to another.
pub fn default_dir() -> PathBuf {
    // The *first* place `chaos_model::find` looks, so a download lands where
    // the list will show it. Asking `find` rather than re-deriving the order is
    // the point: the two disagreeing is the bug this doc comment describes, and
    // a second copy of the rule is how it came back.
    chaos_model::find::model_dirs()
        .into_iter()
        .next()
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// Every shard of a split container, so the size is the model's and not
/// shard one's. `find` reports the first shard; the rest sit beside it.
fn total_bytes(first: &std::path::Path) -> Option<u64> {
    let name = first.file_name()?.to_str()?;
    let dir = first.parent()?;
    // `-00001-of-00005.gguf` -> count them all. Anything else is one file.
    let Some(idx) = name.rfind("-00001-of-") else {
        return std::fs::metadata(first).ok().map(|m| m.len());
    };
    let stem = &name[..idx];
    let mut total = 0u64;
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let n = e.file_name();
        let Some(n) = n.to_str() else { continue };
        if n.starts_with(stem) && n.ends_with(".gguf") {
            if let Ok(m) = e.metadata() {
                total += m.len();
            }
        }
    }
    (total > 0).then_some(total)
}

/// Everything discoverable, ready to display.
pub fn list() -> Vec<Entry> {
    chaos_model::find::list()
        .into_iter()
        .map(|f| Entry {
            bytes: total_bytes(&f.path),
            incomplete: chaos_model::complete::why_incomplete(&f.path),
            label: f.label,
            path: f.path,
        })
        .collect()
}

/// One line for the list box: name on the left, size on the right.
pub fn row(e: &Entry) -> String {
    let size = match e.bytes {
        Some(b) => format!("{}   {}", e.label, human_size(b)),
        None => e.label.clone(),
    };
    // Said in the list, not only on the model's own page: the list is where the
    // choice is made, and "9.00 GB" beside a file holding 911 MB is a lie.
    match &e.incomplete {
        Some(_) => format!("{size}   (unfinished)"),
        None => size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_read_the_way_a_person_would_write_them() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(999), "999 B");
        assert_eq!(human_size(1_500), "1.50 kB");
        assert_eq!(human_size(9_990_000_000), "9.99 GB");
        assert_eq!(human_size(17_300_000_000), "17.3 GB");
        assert_eq!(human_size(144_400_000_000), "144 GB");
    }

    /// The precision must change with the magnitude, or a 144 GB model reads
    /// as "144.42 GB" and a 9 GB one as "9 GB".
    #[test]
    fn precision_falls_as_the_number_grows() {
        assert_eq!(human_size(1_230_000_000).matches('.').count(), 1);
        assert_eq!(human_size(123_000_000_000).matches('.').count(), 0);
    }

    #[test]
    fn a_row_without_a_size_is_still_a_row() {
        let e = Entry {
            label: "qwen3".into(),
            path: "x".into(),
            bytes: None,
            incomplete: None,
        };
        assert_eq!(row(&e), "qwen3");
    }

    /// The list must say so, because the list is where the choice is made.
    #[test]
    fn an_unfinished_download_says_so_in_the_row() {
        let e = Entry {
            label: "qwen3-14b".into(),
            path: "x".into(),
            bytes: Some(911_499_264),
            incomplete: Some("the download did not finish".into()),
        };
        assert!(row(&e).contains("unfinished"));
    }
}
