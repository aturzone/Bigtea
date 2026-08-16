//! Loading always-read weights into RAM, against the real model.

use std::path::PathBuf;

use chaos_model::{Model, ResidentSet, SkipReason};

const GIB: u64 = 1 << 30;
const DEFAULT_PATH: &str =
    r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf";

fn model() -> Option<Model> {
    let p = std::env::var("CHAOS_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_PATH));
    p.exists().then(|| Model::open_split(&p).expect("open"))
}

#[test]
fn a_zero_budget_loads_nothing_and_says_so() {
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    let (set, report) = ResidentSet::load(&m, 0).expect("load");
    assert!(set.is_empty());
    assert_eq!(report.loaded_bytes, 0);
    assert!(!report.complete());
    assert!(report.skipped_over_budget > 0);
}

#[test]
fn the_budget_is_never_exceeded() {
    // Exceeding it makes the OS swap, which is slower than the streaming it
    // was meant to replace -- so this is the load's central guarantee.
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    for budget in [GIB / 4, GIB, 2 * GIB] {
        let (set, report) = ResidentSet::load(&m, budget).expect("load");
        assert!(
            set.bytes() <= budget,
            "loaded {} bytes against a {budget} budget",
            set.bytes()
        );
        assert_eq!(set.bytes(), report.loaded_bytes);
    }
}

#[test]
fn only_always_read_tensors_are_made_resident() {
    // Routed experts stream by design; loading them would defeat the whole
    // approach and blow any budget.
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    let (set, _) = ResidentSet::load(&m, 2 * GIB).expect("load");
    for name in m.tensor_names() {
        if set.contains(name) {
            let loc = m.location(name).expect("located");
            assert!(
                !loc.routed_expert,
                "{name} is a routed expert and must not be resident"
            );
        }
    }
}

#[test]
fn loaded_tensors_have_exactly_their_declared_size() {
    // A short read would put wrong bytes into the model and produce silently
    // wrong output, which is worse than failing.
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    let (set, _) = ResidentSet::load(&m, GIB).expect("load");
    let mut checked = 0;
    for name in m.tensor_names() {
        if let Some(bytes) = set.get(name) {
            let loc = m.location(name).expect("located");
            assert_eq!(bytes.len() as u64, loc.size, "{name} loaded at wrong size");
            checked += 1;
        }
    }
    assert!(checked > 0, "expected to load at least one tensor");
}

#[test]
fn skips_are_attributed_to_a_real_reason() {
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    let (set, report) = ResidentSet::load(&m, GIB / 2).expect("load");
    for s in set.skipped() {
        assert!(s.size > 0);
        assert!(matches!(
            s.reason,
            SkipReason::OverBudget | SkipReason::NotDownloaded
        ));
    }
    // Accounting must balance: everything planned is loaded or skipped.
    let skipped: u64 = set.skipped().iter().map(|s| s.size).sum();
    assert_eq!(
        skipped,
        report.skipped_over_budget + report.skipped_not_downloaded
    );
}

#[test]
fn load_reports_a_plausible_throughput() {
    // An independent check on the probe's bandwidth figure: this is real
    // bytes off the real device through the real read path.
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    let (_, report) = ResidentSet::load(&m, GIB).expect("load");
    if report.loaded_bytes > (64 << 20) {
        let gbps = report.bytes_per_sec() / 1e9;
        assert!(
            gbps > 0.05 && gbps < 60.0,
            "implausible load throughput: {gbps:.2} GB/s"
        );
    }
}

#[test]
fn progress_callback_is_monotonic_and_bounded() {
    let Some(m) = model() else {
        eprintln!("skipping: no container");
        return;
    };
    let mut last = 0u64;
    let mut total_seen = 0u64;
    let (set, _) = ResidentSet::load_with_progress(&m, GIB, |done, total| {
        assert!(done >= last, "progress went backwards");
        assert!(done <= total, "progress exceeded the total");
        last = done;
        total_seen = total;
    })
    .expect("load");
    assert_eq!(last, set.bytes());
    assert!(total_seen >= set.bytes());
}
