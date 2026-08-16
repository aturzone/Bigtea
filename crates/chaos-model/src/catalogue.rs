//! Which models Chaos knows about, and how to fetch them.
//!
//! # Why a checked-in table rather than a server
//!
//! Resolving `v4flash` to five shard URLs needs a mapping from somewhere. A
//! remote index would mean running a service, versioning its schema, and
//! handling its outage — for data that changes a few times a year. A table in
//! the repository is reviewable in a diff, versioned with the code that reads
//! it, and works offline.
//!
//! # Models live on Hugging Face
//!
//! Not on GitHub. GitHub carries the code and the release binaries; the weights
//! are HF repositories, and for DeepSeek-V4-Flash specifically an Unsloth
//! dynamic quant split across five shards. A downloader written against the
//! GitHub API would be pointed at the wrong service entirely.

/// One downloadable quantisation of one model.
#[derive(Debug, Clone, Copy)]
pub struct Quant {
    /// As it appears in the filename, e.g. `UD-Q4_K_XL`.
    pub name: &'static str,
    /// Total bytes across every shard.
    pub bytes: u64,
    /// How many files the container is split into.
    pub shards: u32,
    /// Bytes that must stay resident: attention, routers, embeddings, shared
    /// experts. **This, not the total, decides whether a machine can run it** —
    /// the routed experts stream, the always-read set cannot.
    pub always_read_bytes: u64,
}

/// A model Chaos can fetch by name.
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    /// The short name a user types.
    pub name: &'static str,
    /// `owner/repo` on Hugging Face.
    pub repo: &'static str,
    /// Filename stem; `{stem}-{i:05}-of-{n:05}.gguf` for a split container, or
    /// `{stem}.gguf` when `shards == 1`.
    pub stem: &'static str,
    /// Architecture, so a fit can be predicted before a byte is downloaded.
    pub arch: &'static str,
    pub quants: &'static [Quant],
}

impl Entry {
    /// The files this quant is made of, as repo-relative paths.
    pub fn files(&self, quant: &Quant) -> Vec<String> {
        let stem = self.stem.replace("{quant}", quant.name);
        if quant.shards <= 1 {
            return vec![format!("{stem}.gguf")];
        }
        (1..=quant.shards)
            .map(|i| format!("{stem}-{i:05}-of-{:05}.gguf", quant.shards))
            .collect()
    }

    /// Where a file is fetched from.
    ///
    /// `resolve/main` rather than `blob`: the former streams the file, the
    /// latter returns an HTML page, and the mistake shows up as a `.gguf` that
    /// parses as HTML several gigabytes later.
    pub fn url(&self, file: &str) -> String {
        format!("https://huggingface.co/{}/resolve/main/{file}", self.repo)
    }

    pub fn quant(&self, name: &str) -> Option<&Quant> {
        self.quants
            .iter()
            .find(|q| q.name.eq_ignore_ascii_case(name))
    }
}

const GIB: u64 = 1 << 30;

/// Everything Chaos can fetch by name.
///
/// Sizes are recorded so a fit can be predicted **before** a download starts —
/// the whole point of asking is to be told "this will not fit, and here is the
/// one that will" rather than finding out after 144 GB.
pub const CATALOGUE: &[Entry] = &[
    Entry {
        name: "v4flash",
        repo: "unsloth/DeepSeek-V4-Flash-GGUF",
        stem: "DeepSeek-V4-Flash-{quant}",
        arch: "deepseek4",
        quants: &[Quant {
            name: "UD-Q4_K_XL",
            bytes: 155_095_240_320,
            shards: 5,
            always_read_bytes: 7_925_000_000,
        }],
    },
    Entry {
        name: "qwen3-30b-a3b",
        repo: "unsloth/Qwen3-30B-A3B-GGUF",
        stem: "Qwen3-30B-A3B-{quant}",
        arch: "qwen3moe",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 18_554_000_000,
            shards: 1,
            always_read_bytes: 1_000_000_000,
        }],
    },
];

pub fn find(name: &str) -> Option<&'static Entry> {
    CATALOGUE.iter().find(|e| e.name.eq_ignore_ascii_case(name))
}

/// What a download would cost and whether the result will run here.
#[derive(Debug)]
pub struct Plan {
    pub entry: &'static Entry,
    pub quant: &'static Quant,
    pub files: Vec<String>,
    pub total_bytes: u64,
    /// Bytes still to fetch, given what is already on disk.
    pub remaining_bytes: u64,
    pub disk_free_bytes: u64,
    pub usable_ram_bytes: u64,
}

impl Plan {
    pub fn fits_on_disk(&self) -> bool {
        self.disk_free_bytes >= self.remaining_bytes
    }

    /// Whether the always-read set fits in RAM.
    ///
    /// This is the question that decides if a model is usable, and it is not
    /// "does the model fit" — Chaos's entire design is for models that do not.
    pub fn always_read_fits(&self) -> bool {
        self.usable_ram_bytes >= self.quant.always_read_bytes
    }

    pub fn shortfall_bytes(&self) -> u64 {
        self.quant
            .always_read_bytes
            .saturating_sub(self.usable_ram_bytes)
    }
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_container_names_every_shard() {
        let e = find("v4flash").expect("v4flash is in the catalogue");
        let q = e.quant("UD-Q4_K_XL").expect("quant");
        let files = e.files(q);
        assert_eq!(files.len(), 5);
        assert_eq!(files[0], "DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf");
        assert_eq!(files[4], "DeepSeek-V4-Flash-UD-Q4_K_XL-00005-of-00005.gguf");
    }

    #[test]
    fn a_single_file_container_has_no_shard_suffix() {
        let e = find("qwen3-30b-a3b").expect("in the catalogue");
        let q = e.quant("Q4_K_M").expect("quant");
        assert_eq!(e.files(q), vec!["Qwen3-30B-A3B-Q4_K_M.gguf"]);
    }

    #[test]
    fn urls_resolve_the_file_rather_than_its_web_page() {
        // `blob` returns HTML. The mistake surfaces as a .gguf that fails to
        // parse after several gigabytes have been written.
        let e = find("v4flash").expect("in the catalogue");
        let url = e.url("x.gguf");
        assert!(url.contains("/resolve/main/"), "{url}");
        assert!(!url.contains("/blob/"), "{url}");
    }

    #[test]
    fn lookup_is_case_insensitive_because_users_type() {
        assert!(find("V4Flash").is_some());
        assert!(find("v4flash").is_some());
        assert!(find("nope").is_none());
    }

    #[test]
    fn fit_is_decided_by_the_always_read_set_not_the_container() {
        let e = find("v4flash").expect("in the catalogue");
        let q = e.quant("UD-Q4_K_XL").expect("quant");
        let plan = Plan {
            entry: e,
            quant: q,
            files: e.files(q),
            total_bytes: q.bytes,
            remaining_bytes: q.bytes,
            disk_free_bytes: 600 * GIB,
            // Far less than the 144 GB container, and that is the point.
            usable_ram_bytes: 10 * GIB,
        };
        assert!(plan.fits_on_disk());
        assert!(
            plan.always_read_fits(),
            "10 GiB of RAM must be enough for a 144 GB model — streaming the \
             experts is the whole design"
        );
        assert_eq!(plan.shortfall_bytes(), 0);
    }

    #[test]
    fn a_machine_too_small_reports_the_shortfall() {
        let e = find("v4flash").expect("in the catalogue");
        let q = e.quant("UD-Q4_K_XL").expect("quant");
        let plan = Plan {
            entry: e,
            quant: q,
            files: e.files(q),
            total_bytes: q.bytes,
            remaining_bytes: q.bytes,
            disk_free_bytes: 600 * GIB,
            usable_ram_bytes: 4 * GIB,
        };
        assert!(!plan.always_read_fits());
        assert!(plan.shortfall_bytes() > 3 * GIB);
    }
}
