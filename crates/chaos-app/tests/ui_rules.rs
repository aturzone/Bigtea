//! The app's rules, tested where they can be.
//!
//! A Win32 window cannot be unit-tested: there is no way to assert that a
//! button looks pressed. What *can* be tested is everything the window decides
//! before it draws -- which rows to show, which controls are live, what the
//! endpoint is, which files a delete would remove -- and that is where the bugs
//! have actually been.
//!
//! **This file also encodes the rule that made the app unusable**, as a source
//! check rather than a runtime one, because the failure is a `RefCell` panic
//! under `panic = "abort"`: instant, silent process death that no test harness
//! can observe.

use chaos_app::settings::Settings;
use chaos_app::{catalog, models};

/// The source of the window, for the checks that have to be textual.
fn main_rs() -> String {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/src/main.rs");
    std::fs::read_to_string(p).expect("cannot read main.rs")
}

/// **The bug that made every click fatal.**
///
/// `WM_CTLCOLOR*` handlers borrow `UI`. Any `SendMessageW`, `EnableWindow` or
/// `SetWindowTextW` issued while a borrow is live can dispatch one of those
/// synchronously, and a `RefCell` double borrow under `panic = "abort"` kills
/// the process with no message.
///
/// A textual check is crude, but the alternative is discovering it again by
/// clicking, which is how it was found the first time. It looks for the shape
/// that was wrong: a `borrow_mut()` and a window call inside one `UI.with`.
#[test]
fn no_window_call_happens_while_the_state_is_mutably_borrowed() {
    let src = main_rs();
    let mut offenders = Vec::new();

    for (i, _) in src.match_indices("UI.with(") {
        // Take the closure body: from this call to the matching depth-0 close.
        let rest = &src[i..];
        let mut depth = 0usize;
        let mut end = rest.len();
        for (j, c) in rest.char_indices() {
            match c {
                '{' | '(' => depth += 1,
                '}' | ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = j;
                        break;
                    }
                }
                _ => {}
            }
        }
        let body = &rest[..end];
        if !body.contains("borrow_mut()") {
            continue;
        }
        for call in [
            "SendMessageW(",
            "EnableWindow(",
            "SetWindowTextW(",
            "InvalidateRect(",
            "MoveWindow(",
        ] {
            if body.contains(call) {
                let line = src[..i].matches(char::from(10)).count() + 1;
                offenders.push(format!("line {line}: {call} inside a borrow_mut"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "a Win32 call is made while UI is mutably borrowed, which is an \
         instant silent abort the moment Windows re-enters:\n  {}",
        offenders.join("\n  ")
    );
}

/// Every command the window routes must exist as a function, or a button does
/// nothing and says nothing.
#[test]
fn every_button_is_wired_to_something() {
    let src = main_rs();
    for (id, func) in [
        ("ID_LOAD", "load_model"),
        ("ID_UNLOAD", "unload_model"),
        ("ID_SEND", "send_prompt"),
        ("ID_REFRESH", "rescan"),
        ("ID_GET", "download_selected"),
        ("ID_DELETE", "delete_selected"),
    ] {
        assert!(
            src.contains(&format!("({id}, BN_CLICKED)")),
            "{id} is never handled in WM_COMMAND"
        );
        assert!(
            src.contains(&format!("fn {func}")),
            "{func}, which {id} calls, does not exist"
        );
    }
}

/// Closing the window must stop the child engine. Without this, quitting Chaos
/// leaves a model resident with no window left to stop it from.
#[test]
fn closing_the_window_stops_the_engine() {
    let src = main_rs();
    let destroy = src.find("WM_DESTROY =>").expect("no WM_DESTROY handler");
    // A generous window: the handler carries a long comment explaining why it
    // exists, and a tight slice would miss the call and fail for the wrong
    // reason -- which it did.
    let tail = &src[destroy..destroy + 1200.min(src.len() - destroy)];
    assert!(
        tail.contains("stop_server()"),
        "WM_DESTROY does not stop the server; closing the window would leak it"
    );
}

/// A crash has to leave evidence: with `panic = "abort"` and no console there
/// is otherwise nothing at all.
#[test]
fn a_panic_is_reported() {
    let src = main_rs();
    assert!(src.contains("set_hook"), "no panic hook is installed");
    assert!(
        src.contains("chaos-app-crash.log"),
        "the panic hook does not write a log file"
    );
}

/// The sidebar has to grow with the window, which is what stopped the model
/// rows being clipped mid-word.
#[test]
fn the_sidebar_is_responsive_and_bounded() {
    let src = main_rs();
    assert!(
        src.contains("fn sidebar_for("),
        "the sidebar width is not computed"
    );
    // A fixed width is exactly what was wrong; make sure it did not come back.
    assert!(
        !src.contains("const SIDEBAR: i32"),
        "the fixed-width sidebar is back"
    );
}

// -- the pure logic, tested directly -----------------------------------------

/// The endpoint the window advertises must be the port the server is told to
/// bind. Showing one and binding another sends every client to nothing.
#[test]
fn the_advertised_port_is_the_bound_port() {
    // Built through `parse`, which is how the app itself loads settings, and
    // which the private-by-design `unknown` field makes the only route from
    // outside the crate.
    let cfg = Settings::parse("port = 9999");
    let args = cfg.serve_args("m");
    let i = args.iter().position(|a| a == "--port").expect("no --port");
    assert_eq!(args[i + 1], "9999");
    assert!(
        main_rs().contains("http://127.0.0.1:{}/v1"),
        "the URL is not shown"
    );
}

/// A dense model cannot stream, so the fit verdict must use the resident
/// requirement -- otherwise the app calls a 155 GB streaming model impossible
/// and a 20 GB dense one easy, which is backwards.
#[test]
fn the_fit_verdict_uses_the_resident_requirement() {
    let sixteen_gb = 16_000_000_000u64;
    let mut streams = None;
    let mut too_big = None;
    for o in catalog::offers() {
        let row = catalog::row(&o, sixteen_gb);
        if o.bytes > sixteen_gb && o.always_read < sixteen_gb {
            streams = Some(row.clone());
        }
        if o.always_read > sixteen_gb {
            too_big = Some(row);
        }
    }
    assert!(
        streams
            .expect("no streaming model in the catalogue")
            .contains("streams"),
        "a model larger than memory but with a small resident set must stream"
    );
    assert!(
        too_big
            .expect("no oversized model in the catalogue")
            .contains("too big"),
        "a model whose resident set does not fit must say so"
    );
}

/// Sizes are what a user reads first; they must not regress to raw bytes.
#[test]
fn sizes_are_readable() {
    assert_eq!(models::human_size(155_095_240_320), "155 GB");
    assert_eq!(models::human_size(807_694_368), "808 MB");
}

/// Settings must survive the round trip the window relies on.
#[test]
fn what_is_typed_is_what_comes_back() {
    let cfg = Settings::parse(
        "cache_gib = 6
threads = 4
port = 8231
",
    );
    assert_eq!(Settings::parse(&cfg.render()), cfg);
}
