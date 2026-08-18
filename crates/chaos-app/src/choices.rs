//! Settings as choices, computed from the machine Chaos is running on.
//!
//! Atur: *"in config must be have selectable options — too much of users do not
//! know about configs about systems... that options must be set and selectable
//! based on the system Chaos runs on"*. He is right, and the empty text box was
//! the worst possible answer: it asks a question most people cannot answer, and
//! the one number a knowledgeable user would type is different on every machine.
//!
//! So each setting offers a short list built from the actual hardware — core
//! count, physical memory — with the measured default first and a sentence
//! against every option saying what it costs. **The list is generated, not
//! hardcoded**: "4 threads" is a sensible suggestion on this laptop and a silly
//! one on a 64-core workstation.
//!
//! No Win32 here, so all of it is testable.

/// What the machine looks like, as far as these choices care.
#[derive(Clone, Copy, Debug)]
pub struct Machine {
    /// Logical processors.
    pub cores: u32,
    /// Physical memory, total.
    pub total_ram: u64,
    /// Physical memory free right now.
    pub free_ram: u64,
    /// Whether a usable GPU was detected.
    pub gpu: bool,
}

impl Machine {
    /// A conservative stand-in when nothing could be measured. Never used to
    /// *decide* anything -- it only keeps the lists non-empty.
    pub fn unknown() -> Self {
        Self {
            cores: 4,
            total_ram: 8 << 30,
            free_ram: 4 << 30,
            gpu: false,
        }
    }
}

/// One selectable value.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    /// What goes in the box. Empty means "let Chaos measure".
    pub value: String,
    /// What the user reads.
    pub label: String,
    /// One line on what it costs or buys.
    pub note: String,
}

impl Choice {
    fn new(value: &str, label: impl Into<String>, note: impl Into<String>) -> Self {
        Self {
            value: value.to_string(),
            label: label.into(),
            note: note.into(),
        }
    }
}

/// Generation threads.
///
/// **Generation wants 2-4, not every core**, which is the single most
/// counter-intuitive setting here and the one a user is most likely to get
/// wrong by reaching for the biggest number. The list says so.
pub fn threads(m: Machine) -> Vec<Choice> {
    let mut out = vec![Choice::new(
        "",
        "Measured",
        "Chaos picks from your core count. Recommended.",
    )];
    let mut seen = vec![];
    for n in [2u32, 4, 8, m.cores / 2, m.cores] {
        let n = n.clamp(1, m.cores.max(1));
        if seen.contains(&n) {
            continue;
        }
        seen.push(n);
        let note = match n {
            1 => "One core. Slowest, but leaves the machine usable.".to_string(),
            2..=4 => "Where generation is fastest on most machines.".to_string(),
            _ if n >= m.cores => format!(
                "All {n} cores. Usually *slower* for generation -- the threads \
                 contend for memory bandwidth."
            ),
            _ => format!("{n} cores. Try it against 4; more is not always faster."),
        };
        out.push(Choice::new(&n.to_string(), format!("{n} threads"), note));
    }
    out
}

/// Prefill threads, which want the opposite of generation threads.
pub fn threads_batch(m: Machine) -> Vec<Choice> {
    let cores = m.cores.max(1);
    let mut out = vec![Choice::new(
        "",
        "Measured",
        "Chaos picks from your core count. Recommended.",
    )];
    let mut seen = vec![];
    for n in [cores, cores * 3 / 4, cores / 2, 4] {
        let n = n.clamp(1, cores);
        if seen.contains(&n) {
            continue;
        }
        seen.push(n);
        let note = if n >= cores {
            "Every core. Prefill scales with cores, unlike generation."
        } else {
            "Fewer cores, so the machine stays responsive while a long prompt loads."
        };
        out.push(Choice::new(&n.to_string(), format!("{n} threads"), note));
    }
    out
}

/// The expert cache budget, in GiB.
///
/// Offered as a share of *free* memory rather than total: what matters is what
/// is available now, and a laptop with 16 GB and a browser open has 6.
pub fn cache_gib(m: Machine) -> Vec<Choice> {
    let free = m.free_ram / (1 << 30);
    let mut out = vec![Choice::new(
        "",
        "Measured",
        "Chaos sizes the cache from free memory each time it loads. Recommended.",
    )];
    let mut seen: Vec<u64> = vec![];
    for g in [free / 4, free / 2, free * 3 / 4] {
        if g < 1 || seen.contains(&g) {
            continue;
        }
        seen.push(g);
        let pct = g
            .checked_mul(100)
            .and_then(|v| v.checked_div(free))
            .unwrap_or(0);
        out.push(Choice::new(
            &g.to_string(),
            format!("{g} GiB"),
            format!(
                "About {pct}% of the {free} GiB free now. More cache means fewer \
                 disk reads, up to a point -- the curve flattens."
            ),
        ));
    }
    out
}

/// Context length.
pub fn context() -> Vec<Choice> {
    vec![
        Choice::new(
            "",
            "The model's own limit",
            "Whatever the container declares. Recommended.",
        ),
        Choice::new("2048", "2048 tokens", "Small and quick. Short chats."),
        Choice::new("4096", "4096 tokens", "A long answer, or a short file."),
        Choice::new(
            "8192",
            "8192 tokens",
            "A whole source file. Uses noticeably more memory.",
        ),
        Choice::new(
            "16384",
            "16384 tokens",
            "Large. The key/value cache grows with this, and it is resident.",
        ),
    ]
}

/// GPU layers.
pub fn ngl(m: Machine) -> Vec<Choice> {
    if !m.gpu {
        return vec![Choice::new(
            "",
            "No GPU detected",
            "Chaos found no usable GPU, so everything runs on the processor.",
        )];
    }
    vec![
        Choice::new("", "None", "Everything on the processor. Recommended here."),
        Choice::new(
            "99",
            "All of them",
            "Every layer on the GPU. Only helps if the model fits in video memory.",
        ),
        Choice::new(
            "20",
            "20 layers",
            "A split. Useful when the model is slightly larger than video memory.",
        ),
    ]
}

/// The choices for a settings field, by its control id.
///
/// One function, so the window does not carry a second copy of which box gets
/// which list.
pub fn for_field(id: i32, m: Machine) -> Option<Vec<Choice>> {
    use crate::nav;
    Some(match id {
        nav::ID_THREADS => threads(m),
        nav::ID_THREADS_BATCH => threads_batch(m),
        nav::ID_CACHE => cache_gib(m),
        nav::ID_CONTEXT => context(),
        nav::ID_NGL => ngl(m),
        // The port and the models folder are typed, not chosen: there is no
        // short list of sensible ports, and a folder is a folder.
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn laptop() -> Machine {
        Machine {
            cores: 8,
            total_ram: 16 << 30,
            free_ram: 6 << 30,
            gpu: false,
        }
    }

    fn workstation() -> Machine {
        Machine {
            cores: 64,
            total_ram: 256 << 30,
            free_ram: 200 << 30,
            gpu: true,
        }
    }

    /// **Every list leads with "measured".** The whole point is that a user who
    /// knows nothing can pick the first item and be right.
    #[test]
    fn the_measured_default_is_always_first() {
        for m in [laptop(), workstation()] {
            for id in [
                crate::nav::ID_THREADS,
                crate::nav::ID_THREADS_BATCH,
                crate::nav::ID_CACHE,
                crate::nav::ID_CONTEXT,
                crate::nav::ID_NGL,
            ] {
                let list = for_field(id, m).expect("a list");
                assert!(!list.is_empty());
                assert_eq!(
                    list[0].value, "",
                    "field {id} does not lead with the measured default"
                );
            }
        }
    }

    /// The lists must actually differ by machine, or they are hardcoded advice
    /// wearing a computed label.
    #[test]
    fn the_options_depend_on_the_machine() {
        let small = threads(laptop());
        let big = threads(workstation());
        assert_ne!(
            small.iter().map(|c| c.value.clone()).collect::<Vec<_>>(),
            big.iter().map(|c| c.value.clone()).collect::<Vec<_>>(),
            "an 8-core laptop and a 64-core workstation got the same thread list"
        );
        assert!(big.iter().any(|c| c.value == "64"));
        assert!(!small.iter().any(|c| c.value == "64"));
    }

    /// A cache larger than free memory is the one suggestion that is actively
    /// harmful: it evicts the resident weights the model needs every token.
    #[test]
    fn no_cache_option_exceeds_free_memory() {
        for m in [laptop(), workstation()] {
            let free = m.free_ram / (1 << 30);
            for c in cache_gib(m) {
                if let Ok(g) = c.value.parse::<u64>() {
                    assert!(g <= free, "{g} GiB offered against {free} GiB free");
                }
            }
        }
    }

    /// No duplicates: "4 threads" twice is a list nobody trusts.
    #[test]
    fn options_are_distinct() {
        for m in [laptop(), workstation(), Machine::unknown()] {
            for id in [
                crate::nav::ID_THREADS,
                crate::nav::ID_THREADS_BATCH,
                crate::nav::ID_CACHE,
            ] {
                let list = for_field(id, m).expect("a list");
                let mut values: Vec<&str> = list.iter().map(|c| c.value.as_str()).collect();
                values.sort_unstable();
                let before = values.len();
                values.dedup();
                assert_eq!(before, values.len(), "field {id} repeats an option");
            }
        }
    }

    /// Every option explains itself. A dropdown of bare numbers is the text box
    /// again, with fewer choices.
    #[test]
    fn every_option_carries_a_reason() {
        for m in [laptop(), workstation()] {
            for id in [
                crate::nav::ID_THREADS,
                crate::nav::ID_THREADS_BATCH,
                crate::nav::ID_CACHE,
                crate::nav::ID_CONTEXT,
                crate::nav::ID_NGL,
            ] {
                for c in for_field(id, m).expect("a list") {
                    assert!(!c.label.is_empty(), "field {id} has an unlabelled option");
                    assert!(
                        c.note.len() > 15,
                        "field {id} option {:?} explains nothing",
                        c.label
                    );
                }
            }
        }
    }

    /// A machine with a single core must still produce a usable list rather
    /// than an empty one or a zero.
    #[test]
    fn a_single_core_machine_still_gets_choices() {
        let m = Machine {
            cores: 1,
            total_ram: 2 << 30,
            free_ram: 1 << 30,
            gpu: false,
        };
        for list in [threads(m), threads_batch(m)] {
            assert!(list.len() >= 2);
            for c in list {
                if let Ok(n) = c.value.parse::<u32>() {
                    assert!(n >= 1, "a thread count of {n} was offered");
                }
            }
        }
    }

    /// With no GPU the list says so instead of offering layers that would do
    /// nothing.
    #[test]
    fn no_gpu_means_no_gpu_options() {
        let list = ngl(laptop());
        assert_eq!(list.len(), 1);
        assert!(list[0].label.contains("No GPU"));
        assert!(
            ngl(workstation()).len() > list.len(),
            "a machine with a GPU must be offered more than one option"
        );
    }

    /// The port and the models folder are typed, not chosen.
    #[test]
    fn free_text_fields_have_no_list() {
        for id in [crate::nav::ID_PORT, crate::nav::ID_MODELS_DIR] {
            assert!(for_field(id, laptop()).is_none());
        }
    }
}
