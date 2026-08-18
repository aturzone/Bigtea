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
            // **Measured, not estimated.** This was `7_925_000_000` -- a round
            // number that turned out to be within 0.06% of the truth, but a
            // guess all the same, and it is the figure the whole project's
            // headline rests on. Summed from all five shards' tensor tables
            // with `tools/gguf-always-read.py`: shard 1 carries only metadata,
            // and 129 expert tensors across shards 2-5 hold 147 GB of the
            // 155 GB, leaving 4-6% resident per shard.
            always_read_bytes: 7_920_157_020,
        }],
    },
    Entry {
        name: "qwen3-30b-a3b",
        repo: "unsloth/Qwen3-30B-A3B-GGUF",
        stem: "Qwen3-30B-A3B-{quant}",
        arch: "qwen3moe",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 18_554_098_112,
            shards: 1,
            // Measured from the tensor table, replacing a round
            // `1_000_000_000` that happened to be accurate to 0.25%. 144
            // expert tensors hold 17.55 GB of the 18.55 GB of weights, so 5%
            // stays resident -- which is why an 18 GB model runs on a laptop.
            always_read_bytes: 997_554_176,
        }],
    },
    // -- dense models -------------------------------------------------------
    //
    // **Every byte of a dense model is always-read**, so `always_read_bytes`
    // equals `bytes` for all of these. That is not laziness: a dense container
    // has no routed experts to stream, so the streaming trick this project is
    // built on does nothing for it and the whole file has to fit. Writing the
    // real number here is what makes the app say "too big" honestly instead of
    // promising a 20 GB model will stream on a 16 GB machine.
    //
    // Sizes were read from the Hugging Face tree API rather than estimated, and
    // every repo, filename and byte count below was verified to resolve before
    // it was added -- a wrong stem here is a download button that 404s.
    // Architectures are all in `VERIFIED_ARCHITECTURES`.
    Entry {
        name: "qwen3-32b",
        repo: "unsloth/Qwen3-32B-GGUF",
        stem: "Qwen3-32B-{quant}",
        arch: "qwen3",
        quants: &[
            Quant {
                name: "Q4_K_M",
                bytes: 19_762_150_048,
                shards: 1,
                always_read_bytes: 19_762_150_048,
            },
            Quant {
                name: "Q5_K_M",
                bytes: 23_214_832_288,
                shards: 1,
                always_read_bytes: 23_214_832_288,
            },
        ],
    },
    Entry {
        name: "qwen3-14b",
        repo: "unsloth/Qwen3-14B-GGUF",
        stem: "Qwen3-14B-{quant}",
        arch: "qwen3",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 9_001_753_984,
            shards: 1,
            always_read_bytes: 9_001_753_984,
        }],
    },
    Entry {
        name: "qwen3-8b",
        repo: "unsloth/Qwen3-8B-GGUF",
        stem: "Qwen3-8B-{quant}",
        arch: "qwen3",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 5_027_784_512,
            shards: 1,
            always_read_bytes: 5_027_784_512,
        }],
    },
    Entry {
        name: "qwen3-4b",
        repo: "unsloth/Qwen3-4B-GGUF",
        stem: "Qwen3-4B-{quant}",
        arch: "qwen3",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 2_497_281_312,
            shards: 1,
            always_read_bytes: 2_497_281_312,
        }],
    },
    Entry {
        name: "gemma3-27b",
        repo: "unsloth/gemma-3-27b-it-GGUF",
        stem: "gemma-3-27b-it-{quant}",
        arch: "gemma3",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 16_546_688_736,
            shards: 1,
            always_read_bytes: 16_546_688_736,
        }],
    },
    Entry {
        name: "gemma3-12b",
        repo: "unsloth/gemma-3-12b-it-GGUF",
        stem: "gemma-3-12b-it-{quant}",
        arch: "gemma3",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 7_300_778_336,
            shards: 1,
            always_read_bytes: 7_300_778_336,
        }],
    },
    Entry {
        name: "gemma3-4b",
        repo: "unsloth/gemma-3-4b-it-GGUF",
        stem: "gemma-3-4b-it-{quant}",
        arch: "gemma3",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 2_489_894_016,
            shards: 1,
            always_read_bytes: 2_489_894_016,
        }],
    },
    Entry {
        name: "llama3.2-3b",
        repo: "unsloth/Llama-3.2-3B-Instruct-GGUF",
        stem: "Llama-3.2-3B-Instruct-{quant}",
        arch: "llama",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 2_019_377_600,
            shards: 1,
            always_read_bytes: 2_019_377_600,
        }],
    },
    Entry {
        name: "llama3.2-1b",
        repo: "unsloth/Llama-3.2-1B-Instruct-GGUF",
        stem: "Llama-3.2-1B-Instruct-{quant}",
        arch: "llama",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 807_694_368,
            shards: 1,
            always_read_bytes: 807_694_368,
        }],
    },
    Entry {
        name: "qwen2.5-coder-7b",
        repo: "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
        stem: "Qwen2.5-Coder-7B-Instruct-{quant}",
        arch: "qwen2",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 4_683_073_504,
            shards: 1,
            always_read_bytes: 4_683_073_504,
        }],
    },
    Entry {
        name: "phi4",
        repo: "unsloth/phi-4-GGUF",
        stem: "phi-4-{quant}",
        arch: "phi3",
        quants: &[Quant {
            name: "Q4_K_M",
            bytes: 8_890_306_112,
            shards: 1,
            always_read_bytes: 8_890_306_112,
        }],
    },
    // -- Qwen 3.5/3.6, which this engine cannot run yet ----------------------
    //
    // **Listed on purpose, and listed as unrunnable.** Leaving them out was
    // answering "where is the new Qwen?" with silence, and the answer is not
    // "it does not exist".
    //
    // It is that these are **hybrid** models, which the container states
    // outright: `Qwen3.6-27B-Q4_K_M.gguf` carries `qwen35.ssm.conv_kernel 4`,
    // `ssm.state_size 128`, `ssm.group_count 16`, `ssm.time_step_rank 48` and
    // `full_attention_interval 4`, and every block holds `ssm_conv1d`,
    // `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias` and `ssm_norm`. Following
    // llama.cpp's `qwen35.cpp`, a layer is recurrent when `(i + 1) % 4 != 0` --
    // so **48 of the 64 layers are a gated delta net, not attention**. Those
    // need a recurrent state cache (a conv window and an `[128, 128, 48]` state
    // per sequence per layer) which this engine does not have; a KV cache
    // cannot stand in for it. The remaining 16 attention layers additionally
    // want interleaved multimodal RoPE, four sections wide.
    //
    // `qwen3.rs` refuses them by name rather than rotating the wrong way and
    // producing fluent, confident nonsense. The route in is written down:
    // `docs/graph/backlog/qwen35-gated-delta-net.md`.
    //
    // Every byte below was read from the container's own tensor table with
    // `tools/gguf-always-read.py`, which range-fetches the header rather than
    // the model. The MoE resident figures are measured, not scaled from a
    // sibling -- 11% at Q4 and 14% at Q2, which are not the same fraction and
    // could not have been guessed from one another.
    Entry {
        name: "qwen3.6-27b",
        repo: "unsloth/Qwen3.6-27B-GGUF",
        stem: "Qwen3.6-27B-{quant}",
        arch: "qwen35",
        quants: &[
            Quant {
                name: "Q4_K_M",
                bytes: 16_817_244_384,
                shards: 1,
                // Dense: every byte is always-read. Measured at 16_806_250_496
                // of tensor data; the file is larger by its header.
                always_read_bytes: 16_817_244_384,
            },
            Quant {
                name: "UD-Q4_K_XL",
                bytes: 17_612_564_704,
                shards: 1,
                always_read_bytes: 17_612_564_704,
            },
        ],
    },
    Entry {
        name: "qwen3.6-35b-a3b",
        repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
        stem: "Qwen3.6-35B-A3B-{quant}",
        arch: "qwen35moe",
        quants: &[
            Quant {
                name: "UD-Q4_K_XL",
                bytes: 22_360_456_160,
                shards: 1,
                // 120 expert tensors hold 19.7 GB of the 22.3 GB of weights.
                always_read_bytes: 2_678_180_352,
            },
            Quant {
                name: "UD-Q2_K_XL",
                bytes: 12_290_628_576,
                shards: 1,
                always_read_bytes: 1_751_935_488,
            },
        ],
    },
    // -- Ideogram 4, which is an image model and not a language model --------
    //
    // Open-weight since 3 June 2026: a 9.3B diffusion transformer, and the
    // GGUF conversions below are the ones `stable-diffusion.cpp` reads.
    //
    // **Listed so that "where is Ideogram?" has an answer, and listed as
    // unrunnable because it is a different kind of program.** Chaos is a token
    // loop: embed, attend, sample the next token. An image comes out of a
    // sampler loop over a denoiser, and Ideogram's needs four parts, not one --
    // this transformer, a second *unconditional* copy of it for classifier-free
    // guidance, Qwen3-VL-8B as the text encoder, and a VAE to turn latents into
    // pixels. Three of those four Chaos has no code for.
    //
    // The container says so itself: `tools/gguf-always-read.py` reports **458
    // tensors and zero metadata keys**, so there is no `general.architecture`
    // to dispatch on and no tokenizer inside. It is a bag of weights for
    // another engine, not a model container in the sense the rest of this table
    // means.
    Entry {
        name: "ideogram-4",
        repo: "leejet/ideogram-4-GGUF",
        stem: "ideogram4-{quant}",
        arch: "ideogram4",
        quants: &[Quant {
            name: "Q4_0",
            // Measured from the repository, not the model card.
            bytes: 5_643_820_832,
            shards: 1,
            // Dense: a diffusion transformer has no routed experts, so every
            // byte is read on every one of the sampler's steps.
            always_read_bytes: 5_643_820_832,
        }],
    },
];

/// Architectures this engine implements *and* has diffed against llama.cpp.
///
/// Kept here rather than in `chaos-arch` because the catalogue is what the app
/// and `chaos-pull` consult before a download starts, and neither depends on
/// the engine. `chaos-arch` depends on *this* crate, so a test up there asserts
/// the two lists agree -- which is what stops this copy going stale.
pub const RUNNABLE_ARCHS: &[&str] = &[
    "baichuan",
    "deepseek4",
    "gemma",
    "gemma2",
    "gemma3",
    "internlm2",
    "llama",
    "olmo",
    "phi3",
    "qwen2",
    "qwen3",
    "stablelm",
    "starcoder2",
];

/// Why a model cannot run here, or `None` if it can.
///
/// **A catalogue that only lists what works is a catalogue that cannot answer
/// "where is the model I read about".** Every entry is listed; the ones this
/// engine has not implemented say so, in the sentence a user needs, instead of
/// being quietly missing or -- far worse -- downloading 22 GB and then
/// producing fluent nonsense.
pub fn why_not_runnable(arch: &str) -> Option<&'static str> {
    if RUNNABLE_ARCHS.contains(&arch) {
        return None;
    }
    Some(match arch {
        "qwen35" | "qwen35moe" => {
            "three of its four layers are a gated delta net with recurrent \
             state, not attention, and Chaos has no recurrent memory yet"
        }
        "ideogram4" => {
            "it is an image model -- a diffusion transformer needing a sampler, \
             a separate text encoder and a VAE, none of which Chaos has"
        }
        "qwen3moe" => "its forward pass does not yet match llama.cpp exactly",
        _ => "this architecture has never been diffed against llama.cpp",
    })
}

pub fn find(name: &str) -> Option<&'static Entry> {
    CATALOGUE.iter().find(|e| e.name.eq_ignore_ascii_case(name))
}

/// The catalogue entry a file on disk came from, matched by its filename.
///
/// **This is what turns "your download stopped half way" into a button that
/// finishes it.** A model already on disk is known by its file, not by the name
/// it was fetched under, and resuming needs the entry and the quantisation back
/// again. Matching on the filename the entry itself would produce is exact --
/// no fuzzy stem comparison, no guessing which quant a `Q4_K_M` in the name
/// meant.
///
/// Takes a plain file name, with or without the `.gguf`, and with or without a
/// `-00001-of-00005` shard suffix.
pub fn find_by_file(file: &str) -> Option<(&'static Entry, &'static Quant)> {
    let want = file.trim_end_matches(".gguf").to_lowercase();
    for e in CATALOGUE {
        for q in e.quants {
            for f in e.files(q) {
                if f.trim_end_matches(".gguf").to_lowercase() == want {
                    return Some((e, q));
                }
            }
        }
    }
    None
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

#[cfg(test)]
mod runnable_tests {
    use super::*;

    /// Every catalogue entry either runs or says why not -- there is no third
    /// state, and a `None` for an architecture nobody implemented would be a
    /// download button that produces nonsense.
    #[test]
    fn every_entry_is_either_runnable_or_explained() {
        for e in CATALOGUE {
            match why_not_runnable(e.arch) {
                None => assert!(
                    RUNNABLE_ARCHS.contains(&e.arch),
                    "{} claims to run on {}, which is not in RUNNABLE_ARCHS",
                    e.name,
                    e.arch
                ),
                Some(why) => {
                    assert!(!why.is_empty(), "{} has an empty reason", e.name);
                    assert!(
                        why.len() > 20,
                        "{}: {why:?} is too short to be an explanation",
                        e.name
                    );
                }
            }
        }
    }

    /// The newest Qwen containers are present *and* known to be unrunnable.
    /// If someone implements the architecture, this test is the reminder to
    /// move them -- it fails once they become runnable.
    #[test]
    fn the_new_qwen_models_are_listed_and_refused() {
        for name in ["qwen3.6-27b", "qwen3.6-35b-a3b"] {
            let e = find(name).unwrap_or_else(|| panic!("{name} is not listed"));
            let why = why_not_runnable(e.arch)
                .unwrap_or_else(|| panic!("{name} now claims to run; move it and update this"));
            // Names the *blocker*, not a symptom: 48 of the 64 layers are a
            // gated delta net with recurrent state. The mRoPE the older message
            // pointed at is real but is the smaller half of the work.
            assert!(why.contains("recurrent"), "{name}: {why}");
        }
    }

    /// A file on disk resolves back to the entry that would have fetched it.
    ///
    /// This is what makes DOWNLOAD able to *finish* an interrupted fetch: the
    /// app has a filename and needs the name and quantisation back.
    #[test]
    fn a_filename_resolves_to_its_catalogue_entry() {
        let (e, q) = find_by_file("phi-4-Q4_K_M.gguf").expect("phi-4 is listed");
        assert_eq!(e.name, "phi4");
        assert_eq!(q.name, "Q4_K_M");

        // Without the extension, and case-insensitively.
        assert!(find_by_file("QWEN3-8B-Q4_K_M").is_some());

        // A shard of a split container resolves too, from shard one.
        let (e, q) = find_by_file("DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf")
            .expect("v4flash shard one is listed");
        assert_eq!(e.name, "v4flash");
        assert_eq!(q.shards, 5);

        assert!(find_by_file("something-nobody-ships.gguf").is_none());
    }

    /// Every entry's own filenames must resolve to it, or the resume path is a
    /// coin flip on whichever entry happens to be listed first.
    #[test]
    fn every_entry_resolves_from_its_own_filenames() {
        for e in CATALOGUE {
            for q in e.quants {
                let first = &e.files(q)[0];
                let (got_e, got_q) =
                    find_by_file(first).unwrap_or_else(|| panic!("{first} resolves to nothing"));
                assert_eq!(got_e.name, e.name, "{first}");
                assert_eq!(got_q.name, q.name, "{first}");
            }
        }
    }

    /// Ideogram 4 is listed and refused for being a different kind of model.
    ///
    /// It is open-weight and it is on Hugging Face, so "we do not have it" was
    /// never the answer. The answer is that an image needs a sampler, a text
    /// encoder and a VAE, and this engine is a token loop.
    #[test]
    fn the_image_model_is_listed_and_refused_as_an_image_model() {
        let e = find("ideogram-4").expect("ideogram-4 is listed");
        let why = why_not_runnable(e.arch).expect("it must not claim to run");
        assert!(why.contains("image"), "{why}");
    }

    /// **The resident figure is what decides a download.** For a dense model it
    /// is the whole file; for a Mixture-of-Experts it must be a small fraction,
    /// or the entry is claiming a 22 GB model needs 22 GB resident and the app
    /// will call it impossible on exactly the machines it was built for.
    #[test]
    fn moe_entries_have_a_measured_resident_fraction() {
        for e in CATALOGUE {
            let moe = e.arch.contains("moe") || e.arch == "deepseek4";
            for q in e.quants {
                let pct = q.always_read_bytes * 100 / q.bytes.max(1);
                if moe {
                    assert!(
                        pct < 40,
                        "{} {}: {pct}% resident -- that is not a streaming model",
                        e.name,
                        q.name
                    );
                    assert!(
                        q.always_read_bytes % 1_000_000 != 0,
                        "{} {}: {} looks rounded rather than measured",
                        e.name,
                        q.name,
                        q.always_read_bytes
                    );
                } else {
                    assert_eq!(
                        q.always_read_bytes, q.bytes,
                        "{} {} is dense, so every byte is always-read",
                        e.name, q.name
                    );
                }
            }
        }
    }
}
