//! Residency policy: what goes in RAM, what streams, and why.

use bigtea_gguf::{GgmlType, TensorInfo};
use bigtea_plan::{plan_layout, Placement, GIB};

/// One always-read tensor of roughly `bytes`, using F32 (1 element = 4 bytes).
fn dense(name: &str, bytes: u64) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        dims: vec![bytes / 4],
        ty: GgmlType(0), // F32
        offset: 0,
    }
}

/// One routed-expert tensor. The `_exps` suffix is how llama.cpp names the
/// stacked routed experts of a layer.
fn expert(name: &str, bytes: u64) -> TensorInfo {
    TensorInfo {
        name: format!("{name}_exps.weight"),
        dims: vec![bytes / 4],
        ty: GgmlType(0),
        offset: 0,
    }
}

#[test]
fn always_read_weights_are_made_resident_first() {
    let tensors = vec![
        dense("blk.0.attn_q.weight", 2 * GIB),
        expert("blk.0.ffn_gate", 40 * GIB),
        dense("token_embd.weight", 1 * GIB),
    ];
    let layout = plan_layout(&tensors, 8 * GIB);

    assert!(layout.all_always_read_resident());
    assert_eq!(layout.ram_used_bytes, 3 * GIB);
    // The expert pool never becomes resident at this machine class.
    let experts_resident = layout
        .placed
        .iter()
        .filter(|p| p.routed && p.placement == Placement::ResidentRam)
        .count();
    assert_eq!(experts_resident, 0);
}

#[test]
fn experts_always_stream_even_with_a_huge_budget() {
    // Even absurd RAM does not make the planner pin a 40 GiB pool, because
    // that is not the regime this tool exists for.
    let tensors = vec![dense("attn", 1 * GIB), expert("ffn_gate", 40 * GIB)];
    let layout = plan_layout(&tensors, 500 * GIB);
    for p in &layout.placed {
        if p.routed {
            assert_eq!(p.placement, Placement::StreamFromDisk);
        }
    }
}

#[test]
fn shortfall_is_reported_when_always_read_exceeds_the_budget() {
    let tensors = vec![
        dense("a", 4 * GIB),
        dense("b", 4 * GIB),
        dense("c", 4 * GIB),
    ];
    let layout = plan_layout(&tensors, 9 * GIB);

    assert!(!layout.all_always_read_resident());
    assert_eq!(layout.always_read_bytes, 12 * GIB);
    // Two fit, one does not.
    assert_eq!(layout.ram_used_bytes, 8 * GIB);
    assert_eq!(layout.always_read_shortfall_bytes, 4 * GIB);
    assert!(layout
        .notes
        .iter()
        .any(|n| n.contains("did not fit") && n.contains("every token")));
}

#[test]
fn resident_bytes_never_exceed_the_budget() {
    // The budget is a hard ceiling: exceeding it means the OS swaps, which is
    // slower than the streaming it was meant to replace.
    for budget_gib in [0u64, 1, 3, 7, 100] {
        let tensors = vec![dense("a", 2 * GIB), dense("b", 3 * GIB), dense("c", 5 * GIB)];
        let layout = plan_layout(&tensors, budget_gib * GIB);
        assert!(
            layout.ram_used_bytes <= budget_gib * GIB,
            "budget {budget_gib} GiB was exceeded"
        );
    }
}

#[test]
fn accounting_is_conserved() {
    let tensors = vec![
        dense("a", 2 * GIB),
        dense("b", 3 * GIB),
        expert("e", 20 * GIB),
    ];
    let layout = plan_layout(&tensors, 4 * GIB);

    // Every always-read byte is either resident or counted as shortfall.
    assert_eq!(
        layout.ram_used_bytes + layout.always_read_shortfall_bytes,
        layout.always_read_bytes
    );
    // Every tensor gets a placement.
    assert_eq!(layout.placed.len(), tensors.len());
}

#[test]
fn expert_cache_is_only_worthwhile_above_one_working_set() {
    // The whole reason spare RAM usually goes unused at this tier.
    let tensors = vec![dense("a", 2 * GIB), expert("e", 40 * GIB)];
    let layout = plan_layout(&tensors, 6 * GIB);
    assert_eq!(layout.spare_ram_bytes(), 4 * GIB);

    // A working set larger than the spare budget: caching buys nothing.
    assert!(!layout.expert_cache_is_worthwhile(5 * GIB));
    // Small enough to actually hold a token's worth: now it pays.
    assert!(layout.expert_cache_is_worthwhile(3 * GIB));
    // Degenerate input must not claim a benefit.
    assert!(!layout.expert_cache_is_worthwhile(0));
}

#[test]
fn zero_budget_streams_everything_without_panicking() {
    let tensors = vec![dense("a", 2 * GIB), expert("e", 8 * GIB)];
    let layout = plan_layout(&tensors, 0);
    assert_eq!(layout.ram_used_bytes, 0);
    assert_eq!(layout.always_read_shortfall_bytes, 2 * GIB);
    assert!(layout
        .placed
        .iter()
        .all(|p| p.placement == Placement::StreamFromDisk));
}

#[test]
fn unsized_tensors_are_excluded_and_flagged_not_silently_dropped() {
    let tensors = vec![
        dense("good", 1 * GIB),
        TensorInfo {
            name: "mystery".into(),
            dims: vec![1024],
            ty: GgmlType(9999), // unknown type
            offset: 0,
        },
    ];
    let layout = plan_layout(&tensors, 8 * GIB);
    assert_eq!(layout.placed.len(), 1, "unsized tensors are not placed");
    assert!(layout.notes.iter().any(|n| n.contains("unknown type")));
}

#[test]
fn empty_container_is_handled() {
    let layout = plan_layout(&[], 8 * GIB);
    assert_eq!(layout.ram_used_bytes, 0);
    assert_eq!(layout.always_read_bytes, 0);
    assert!(layout.all_always_read_resident());
}
