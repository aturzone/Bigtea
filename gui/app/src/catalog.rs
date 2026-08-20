//! What the app offers to download, and whether this machine could run it.
//!
//! The list comes from `chaos_model::catalogue`, shared with `chaos-pull`, so
//! the window and the CLI cannot disagree about what exists.
//!
//! **The number that decides a download is not the download size.** A 155 GB
//! container runs on a 16 GB machine because the routed experts stream; what
//! must fit is the always-read set. Showing only "155 GB" next to a model would
//! tell a user it is impossible when it is the thing this project exists to do,
//! so both figures are surfaced and the verdict is computed from the right one.

use crate::models::{self, human_size};

pub struct Offer {
    pub name: String,
    pub quant: String,
    /// Total download.
    pub bytes: u64,
    /// What has to stay in memory for it to run at all.
    pub always_read: u64,
    pub shards: u32,
    pub arch: String,
    /// Why Chaos cannot run this yet, if it cannot.
    ///
    /// **Listed rather than hidden.** A catalogue that shows only what works
    /// answers "where is the model I read about?" with silence, and the honest
    /// answer is a sentence: this container needs something the engine does not
    /// implement. Hiding it also means the next person asks again.
    pub unsupported: Option<&'static str>,
    /// Adult content. Marked in the list, and confirmed before a download.
    pub adult: bool,
}

/// What must stay resident for an installed model, if the catalogue knows it.
///
/// Matched on the container's file stem, because that is all an installed model
/// carries: `Qwen3-VL-8B-Instruct-Q4_K_M.gguf` against the catalogue's stem and
/// quant. **`None` rather than a guess** — the caller shows bytes without a
/// percentage instead, and a denominator taken from the file size would report
/// a 144 GB mixture-of-experts as 5% loaded for its whole load.
pub fn resident_for(stem: &str) -> Option<u64> {
    // The catalogue already stores the filename template each entry downloads
    // to, so this is that comparison and not a guess at how names are spelled.
    let want = stem.trim_end_matches(".gguf").to_ascii_lowercase();
    for e in chaos_model::catalogue::CATALOGUE {
        for q in e.quants {
            for f in e.files(q) {
                let name = chaos_model::catalogue::Entry::local_name(&f)
                    .trim_end_matches(".gguf")
                    .to_ascii_lowercase();
                // Shards end `-00001-of-00005`; the stem on screen may be any
                // one of them, so a prefix match is what identifies the model.
                if !name.is_empty() && (want == name || want.starts_with(&name)) {
                    return Some(q.always_read_bytes);
                }
            }
        }
    }
    None
}

/// Everything fetchable, flattened to one row per quantisation.
pub fn offers() -> Vec<Offer> {
    let mut out = Vec::new();
    for e in chaos_model::catalogue::CATALOGUE {
        for q in e.quants {
            out.push(Offer {
                name: e.name.to_string(),
                quant: q.name.to_string(),
                bytes: q.bytes,
                always_read: q.always_read_bytes,
                shards: q.shards,
                arch: e.arch.to_string(),
                unsupported: chaos_model::catalogue::why_not_runnable(e.arch),
                adult: e.adult,
            });
        }
    }
    out
}

/// How a machine with `free` bytes of memory would fare.
///
/// **None of these mean "no".** This runner exists to run models larger than
/// memory: DeepSeek-V4-Flash is 144 GB and generates correct text on a 15.7 GiB
/// laptop. The three cases are three *speeds*, and naming the slowest one
/// `TooBig` — which the window showed as "too big for this machine" — told the
/// user a model would not work when it demonstrably does.
pub enum Verdict {
    /// Everything fits; nothing streams.
    Resident,
    /// The always-read set fits, so it runs and the experts stream from disk.
    Streams,
    /// The always-read set does not fit either, so those weights are re-read
    /// from disk on every token. Slow — and it runs.
    Rereads,
}

pub fn verdict(o: &Offer, free_bytes: u64) -> Verdict {
    if o.bytes <= free_bytes {
        Verdict::Resident
    } else if o.always_read <= free_bytes {
        Verdict::Streams
    } else {
        Verdict::Rereads
    }
}

/// One line for the list, its columns joined.
pub fn row(o: &Offer, free_bytes: u64) -> String {
    columns(o, free_bytes).join(&models::COLUMN_SEP.to_string())
}

/// A downloadable offer as its parts: what it is, then what it costs.
///
/// **Separate columns, for the reason the installed list has them.** Built as
/// one string with spaces between, the row was drawn with a single
/// `DT_END_ELLIPSIS` and the *tail* went first -- so "needs 16.5 GB - slow,
/// re-reads" became "needs 16.5 GB - sl..." and the verdict, which is the one
/// thing the row exists to say, was the one thing cut. Each measurement now
/// gets its own right edge and the name keeps what is left.
pub fn columns(o: &Offer, free_bytes: u64) -> Vec<String> {
    // An unsupported architecture outranks the fit verdict: "streams" is true
    // and useless if the engine will refuse the container on load.
    let mark = if o.unsupported.is_some() {
        // One word, not "not supported yet". It is the widest column in the
        // list and it appears on one row, where it cost that row's name its
        // tail: "qwen3-30b-a3b Q4_K_M" read as "qwen3-30b-a3b ...". The
        // sentence with the actual reason is on the model's own panel, which is
        // where somebody who cares reads it.
        "unsupported"
    } else {
        match verdict(o, free_bytes) {
            Verdict::Resident => "fits",
            Verdict::Streams => "streams",
            // Not "too big". It runs; the weights come back off the disk.
            Verdict::Rereads => "slow, re-reads",
        }
    };
    // Before the size, because it decides whether to read the rest of the row.
    let flag = if o.adult { "  [18+]" } else { "" };
    let shards = if o.shards > 1 {
        format!(" [{} files]", o.shards)
    } else {
        String::new()
    };
    let size = human_size(o.bytes);
    let needs = human_size(o.always_read);
    let mut v = vec![
        format!("{}{} {}", o.name, flag, o.quant),
        format!("{size}{shards}"),
    ];
    // **Both numbers, but only when they are two numbers.** The download size
    // and the resident requirement are different questions -- one is disk, the
    // other is whether it runs -- and on a *dense* model they have the same
    // answer, because every weight is always read. Printing "9.00 GB   needs
    // 9.00 GB" spent a column saying nothing on twenty rows out of
    // twenty-seven, and the width it cost came out of the name.
    //
    // What is left is the case the column exists for: 155 GB that needs 7.92,
    // 18.6 GB that needs 998 MB. That is the whole idea of this engine, and
    // now it is the only place the second number appears.
    if needs != size {
        v.push(format!("needs {needs}"));
    }
    v.push(mark.to_string());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `[18+]` marker is in the row the window renders.
    ///
    /// It was in `chaos-pull --list` first and not here, so the window listed
    /// adult models with no warning at all -- found by printing the rows through
    /// this function rather than by looking at the window, which is the only way
    /// it was going to be found.
    #[test]
    fn an_adult_offer_is_marked_in_the_row() {
        let mut o = offer(1 << 20, 1 << 20);
        assert!(
            !row(&o, 1 << 30).contains("18+"),
            "not marked when it is not"
        );
        o.adult = true;
        assert!(row(&o, 1 << 30).contains("[18+]"), "{}", row(&o, 1 << 30));
    }

    fn offer(bytes: u64, always: u64) -> Offer {
        Offer {
            name: "m".into(),
            quant: "q".into(),
            bytes,
            always_read: always,
            adult: false,
            shards: 1,
            arch: "a".into(),
            unsupported: None,
        }
    }

    /// An architecture the engine cannot run outranks the fit verdict: telling
    /// someone a 22 GB container "streams" is true and useless if loading it
    /// will be refused.
    #[test]
    fn an_unsupported_model_says_so_instead_of_its_fit() {
        let mut o = offer(22_000_000_000, 2_600_000_000);
        o.unsupported = Some("needs a rope mode Chaos does not implement");
        let r = row(&o, 8_000_000_000);
        assert!(r.contains("unsupported"), "{r}");
        assert!(!r.contains("streams"), "{r}");
    }

    /// Every entry the real catalogue offers carries a verdict one way or the
    /// other, and the newest Qwen containers are present rather than hidden.
    ///
    /// **The dense ones are offered as runnable now**, verified against
    /// llama.cpp on Qwen3.5-0.8B — the same `qwen35` architecture at 24 layers.
    /// The MoE variant is still marked, because its routed path is untested.
    #[test]
    fn the_catalogue_lists_the_new_qwen_and_marks_it() {
        let all = offers();
        let dense = all
            .iter()
            .find(|o| o.name == "qwen3.8-27b")
            .expect("the newest Qwen is not offered at all");
        assert!(
            dense.unsupported.is_none(),
            "qwen3.8 is `qwen35`, which is implemented and verified"
        );
        let moe = all
            .iter()
            .find(|o| o.name == "qwen3.6-35b-a3b")
            .expect("the MoE variant is not offered at all");
        assert!(
            moe.unsupported.is_some(),
            "qwen35moe's routed path has never been run here"
        );
        assert!(
            all.iter()
                .any(|o| o.name == "v4flash" && o.unsupported.is_none()),
            "V4-Flash must still be offered as runnable"
        );
    }

    /// The whole point of the project: a container far larger than memory still
    /// runs, and the app must say so rather than calling it impossible.
    #[test]
    fn a_model_ten_times_your_ram_still_streams() {
        let v4 = offer(155_000_000_000, 7_925_000_000);
        assert!(matches!(verdict(&v4, 10_000_000_000), Verdict::Streams));
        assert!(row(&v4, 10_000_000_000).contains("streams"));
    }

    /// The resident lookup matches the names actually on disk.
    ///
    /// An installed model carries only its file stem, so this is string
    /// matching, and string matching that silently misses shows a loading line
    /// with no percentage — which looks like the feature is broken rather than
    /// like the catalogue does not know the model.
    #[test]
    fn the_resident_lookup_matches_real_filenames() {
        // Names as `chaos-pull` writes them.
        for stem in [
            "Qwen3-VL-8B-Instruct-Q4_K_M",
            "Llama-3.2-1B-Instruct-Q4_K_M",
            "gemma-3-4b-it-Q4_K_M",
        ] {
            assert!(
                resident_for(stem).is_some_and(|b| b > 0),
                "no resident size for {stem}"
            );
        }
        // And it does not invent one for something that is not in the
        // catalogue, because a wrong denominator is worse than none.
        assert_eq!(resident_for("something-nobody-ships-Q4_K_M"), None);
        assert_eq!(resident_for(""), None);
    }

    #[test]
    fn it_is_too_big_only_when_the_always_read_set_does_not_fit() {
        let v4 = offer(155_000_000_000, 7_925_000_000);
        assert!(matches!(verdict(&v4, 4_000_000_000), Verdict::Rereads));
    }

    #[test]
    fn a_small_model_is_reported_as_resident() {
        let small = offer(800_000_000, 800_000_000);
        assert!(matches!(verdict(&small, 10_000_000_000), Verdict::Resident));
    }

    /// Both numbers appear when they are two numbers, and one when they are one.
    ///
    /// The download size and the resident requirement are different questions
    /// and a user needs each -- but on a dense model they have the same answer,
    /// and a column that says "9.00 GB   needs 9.00 GB" is width taken from the
    /// name to repeat a number.
    #[test]
    fn the_row_carries_size_and_requirement() {
        let v4 = offer(155_000_000_000, 7_925_000_000);
        let r = row(&v4, 10_000_000_000);
        assert!(r.contains("155 GB"), "{r}");
        // 7.925 lands just under the halfway point as a float, so it formats
        // down. Pinned as it actually behaves rather than as it reads.
        assert!(r.contains("7.92 GB"), "{r}");

        // A dense model reads every weight, so the two are the same number and
        // it is printed once.
        let dense = offer(9_000_000_000, 9_000_000_000);
        let c = columns(&dense, 20_000_000_000);
        assert_eq!(
            c.iter().filter(|p| p.contains("9.00 GB")).count(),
            1,
            "{c:?}"
        );
        assert!(!row(&dense, 20_000_000_000).contains("needs"), "{c:?}");
    }

    #[test]
    fn a_split_container_says_how_many_files() {
        let mut v4 = offer(155_000_000_000, 7_925_000_000);
        v4.shards = 5;
        assert!(row(&v4, 10_000_000_000).contains("[5 files]"));
    }

    /// The verdict is its own column, so it cannot be the thing that truncates.
    ///
    /// Drawn as one string it was: "needs 16.5 GB - slow, re-reads" came out as
    /// "needs 16.5 GB - sl...", cutting the one word the row exists to say.
    #[test]
    fn the_verdict_is_a_column_of_its_own() {
        let v4 = offer(155_000_000_000, 7_925_000_000);
        let c = columns(&v4, 10_000_000_000);
        assert!(c.len() >= 4, "{c:?}");
        assert_eq!(c[c.len() - 1], "streams", "the verdict is the last column");
        assert!(c[0].contains(&v4.name), "the name is the first column");
        // No column carries the separator, so splitting the row recovers them.
        for part in &c {
            assert!(!part.contains(models::COLUMN_SEP), "{part:?}");
        }
        assert_eq!(
            row(&v4, 10_000_000_000)
                .split(models::COLUMN_SEP)
                .collect::<Vec<_>>(),
            c.iter().map(String::as_str).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_catalogue_is_not_empty() {
        assert!(!offers().is_empty(), "nothing is offered for download");
    }
}
