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

use crate::models::human_size;

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

/// One line for the list.
pub fn row(o: &Offer, free_bytes: u64) -> String {
    // An unsupported architecture outranks the fit verdict: "streams" is true
    // and useless if the engine will refuse the container on load.
    let mark = if o.unsupported.is_some() {
        "not supported yet"
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
    format!(
        "{}{} {}   {}{}   needs {} - {}",
        o.name,
        flag,
        o.quant,
        human_size(o.bytes),
        shards,
        human_size(o.always_read),
        mark
    )
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
        assert!(r.contains("not supported"), "{r}");
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

    /// Both numbers appear, because the download size and the requirement are
    /// different questions and a user needs each.
    #[test]
    fn the_row_carries_size_and_requirement() {
        let v4 = offer(155_000_000_000, 7_925_000_000);
        let r = row(&v4, 10_000_000_000);
        assert!(r.contains("155 GB"), "{r}");
        // 7.925 lands just under the halfway point as a float, so it formats
        // down. Pinned as it actually behaves rather than as it reads.
        assert!(r.contains("7.92 GB"), "{r}");
    }

    #[test]
    fn a_split_container_says_how_many_files() {
        let mut v4 = offer(155_000_000_000, 7_925_000_000);
        v4.shards = 5;
        assert!(row(&v4, 10_000_000_000).contains("[5 files]"));
    }

    #[test]
    fn the_catalogue_is_not_empty() {
        assert!(!offers().is_empty(), "nothing is offered for download");
    }
}
