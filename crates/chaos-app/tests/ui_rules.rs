//! The app's rules, tested where they can be.
//!
//! A Win32 window cannot be unit-tested: there is no way to assert that a
//! button looks pressed. What *can* be tested is everything the window decides
//! before it draws -- which page owns a control, which rows to show, what the
//! endpoint is, which files a delete would remove -- and that is where the bugs
//! have actually been.
//!
//! **This file also encodes the two rules that are invisible at runtime**, as
//! source checks rather than assertions:
//!
//! 1. No Win32 call while `UI` is mutably borrowed. The failure is a `RefCell`
//!    double borrow under `panic = "abort"`: instant, silent process death that
//!    no harness can observe.
//! 2. No colour named outside `theme.rs`. The failure is not a crash at all --
//!    it is a palette that cannot be changed, which is how the previous window
//!    ended up with controls that ignored the theme.

use chaos_app::nav::{self, Page};
use chaos_app::settings::Settings;
use chaos_app::theme;
use chaos_app::{catalog, models};
use std::collections::{HashMap, HashSet};

fn source(name: &str) -> String {
    let p = format!("{}/src/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {p}: {e}"))
}

fn main_rs() -> String {
    source("main.rs")
}

/// A source slice with its comments removed.
///
/// Every check here looks for code, and a comment that *explains* a rule
/// contains the same words the rule forbids -- `child()` says "no `WS_VISIBLE`"
/// in as many words, and without this the test that enforces it fails on its
/// own explanation.
fn code_only(src: &str) -> String {
    src.lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(
            "
",
        )
}

/// The body of one `fn`, from its signature to the matching close brace.
fn function_body<'a>(src: &'a str, signature: &str) -> &'a str {
    let start = src
        .find(signature)
        .unwrap_or_else(|| panic!("{signature} is not in main.rs"));
    let rest = &src[start..];
    let open = rest.find('{').expect("no body");
    let mut depth = 0usize;
    for (i, c) in rest[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &rest[open..open + i];
                }
            }
            _ => {}
        }
    }
    rest
}

// -- the two invisible rules --------------------------------------------------

/// **The bug that made every click fatal.**
///
/// `WM_CTLCOLOR*` handlers borrow `UI`. Any window call issued while a borrow is
/// live can dispatch one of those synchronously, and a `RefCell` double borrow
/// under `panic = "abort"` kills the process with no message.
///
/// A textual check is crude, but the alternative is discovering it again by
/// clicking, which is how it was found the first time. It looks for the shape
/// that was wrong: a `borrow_mut()` and a window call inside one `UI.with`.
#[test]
fn no_window_call_happens_while_the_state_is_mutably_borrowed() {
    let src = main_rs();
    let mut offenders = Vec::new();

    for (i, _) in src.match_indices("UI.with(") {
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
        // Anything that can re-enter the window procedure. `ShowWindow` and
        // `SetFocus` are on the list because `show_page` calls both, and a
        // borrow left open around them would be the same abort in a new place.
        for call in [
            "SendMessageW(",
            "EnableWindow(",
            "SetWindowTextW(",
            "InvalidateRect(",
            "MoveWindow(",
            "ShowWindow(",
            "SetFocus(",
            "DestroyWindow(",
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

/// **Tokens, not literals.** `theme.rs` owns every colour in the app.
///
/// The previous window spelled `BLACK` and `WHITE` at forty call sites, so the
/// palette could not be changed without forty edits -- and the ones that were
/// missed are why controls came up in the system's greys. A colour constructed
/// anywhere else is that bug starting again.
#[test]
fn no_colour_is_named_outside_the_theme() {
    let src = main_rs();
    let mut offenders = Vec::new();
    for (n, (line, code)) in src.lines().zip(code_only(&src).lines()).enumerate() {
        for needle in ["rgb(", "RGB(", "0x00FF", "CreateSolidBrush(0"] {
            if code.contains(needle) {
                offenders.push(format!("line {}: {}", n + 1, line.trim()));
            }
        }
        // The old two-value palette's constants, which still exist in `win32`
        // for the installer's use and must not come back here.
        for word in ["BLACK", "WHITE"] {
            if code
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .any(|w| w == word)
            {
                offenders.push(format!("line {}: {}", n + 1, line.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "main.rs names a colour; every colour belongs to theme.rs:\n  {}",
        offenders.join("\n  ")
    );
}

// -- the shell ----------------------------------------------------------------

/// Every control `nav` declares must actually be created, or `show_page` reveals
/// a window that does not exist and the page is silently short of a control.
///
/// The id *names* are read out of `nav.rs` itself rather than listed here, so
/// this cannot go stale when a control is added.
#[test]
fn every_declared_control_is_created() {
    let nav_src = source("nav.rs");
    let mut by_value: HashMap<i32, String> = HashMap::new();
    for line in nav_src.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("pub const ID_") else {
            continue;
        };
        let Some((name, tail)) = rest.split_once(": i32 = ") else {
            continue;
        };
        if let Ok(v) = tail.trim_end_matches(';').trim().parse::<i32>() {
            by_value.insert(v, format!("ID_{name}"));
        }
    }
    assert!(
        by_value.len() >= 20,
        "only {} ids were parsed out of nav.rs; the parser has drifted",
        by_value.len()
    );

    let src = main_rs();
    let build = function_body(&src, "unsafe fn build_controls(");
    let mut missing = Vec::new();
    for p in nav::PAGES {
        for &id in nav::controls(p) {
            let name = &by_value[&id];
            // The settings boxes and toggles are built by iterating `FIELDS`
            // and `TOGGLES`, so their ids never appear by name.
            let built_by_loop = nav::FIELDS.iter().chain(nav::TOGGLES).any(|f| f.id == id);
            if !built_by_loop && !build.contains(name.as_str()) {
                missing.push(format!("{name} ({:?})", p));
            }
        }
    }
    for id in nav::SHELL_CONTROLS {
        let name = &by_value[&id];
        // The rail buttons are built by iterating `PAGES`.
        if nav::PAGES.iter().any(|&p| nav::nav_id(p) == id) {
            continue;
        }
        if !build.contains(name.as_str()) {
            missing.push(format!("{name} (shell)"));
        }
    }
    assert!(
        missing.is_empty(),
        "declared in nav.rs but never created in build_controls: {}",
        missing.join(", ")
    );
    assert!(
        build.contains("for f in nav::FIELDS") && build.contains("for f in nav::TOGGLES"),
        "the settings controls are no longer built from nav::FIELDS/TOGGLES, \
         so the loop this test trusts is gone"
    );
    assert!(
        build.contains("for p in nav::PAGES"),
        "the rail buttons are no longer built from nav::PAGES"
    );
}

/// **Controls are created hidden.** `show_page` is the only thing that reveals
/// one; if `child()` passed `WS_VISIBLE`, every page's controls would be on
/// screen at once and stacked on top of each other.
#[test]
fn controls_are_created_hidden_so_one_page_owns_the_screen() {
    let src = main_rs();
    let body = code_only(function_body(&src, "unsafe fn child("));
    assert!(
        body.contains("WS_CHILD | style"),
        "child() no longer creates controls with WS_CHILD alone"
    );
    assert!(
        !body.contains("WS_VISIBLE"),
        "child() passes WS_VISIBLE, so every page's controls show at once"
    );
    let show = code_only(function_body(&src, "fn show_page("));
    assert!(
        show.contains("SW_HIDE") && show.contains("SW_SHOW"),
        "show_page does not hide and show"
    );
    assert!(
        !show.contains("DestroyWindow"),
        "show_page destroys controls; visibility is not lifecycle, and the \
         transcript would be lost on every page change"
    );
}

/// Every command the window routes must exist as a function, or a button does
/// nothing and says nothing.
#[test]
fn every_button_is_wired_to_something() {
    let src = main_rs();
    for (id, func) in [
        ("nav::ID_LOAD", "load_model"),
        ("nav::ID_UNLOAD", "unload_model"),
        ("nav::ID_SEND", "send_prompt"),
        ("nav::ID_REFRESH", "rescan"),
        ("nav::ID_GET", "download_selected"),
        ("nav::ID_DELETE", "delete_selected"),
        ("nav::ID_CLEAR", "clear_chat"),
        ("nav::ID_SAVE", "save_settings"),
        ("nav::ID_RESET", "reset_settings"),
        ("nav::ID_COPY_ENDPOINT", "copy_endpoint"),
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

/// **Every menu command is handled.** A menu that lists something the window
/// ignores is worse than one that does not list it.
#[test]
fn every_menu_command_is_handled() {
    let nav_src = source("nav.rs");
    let src = main_rs();
    let mut checked = 0;
    for line in nav_src.lines() {
        let Some(rest) = line.trim().strip_prefix("pub const IDM_") else {
            continue;
        };
        let Some((name, _)) = rest.split_once(':') else {
            continue;
        };
        let full = format!("nav::IDM_{name}");
        // The page commands are routed through `page_of_menu` rather than by
        // name, which is the point of that function.
        if name.starts_with("PAGE_") {
            assert!(
                src.contains("page_of_menu("),
                "the page menu commands are no longer routed"
            );
            checked += 1;
            continue;
        }
        assert!(
            src.contains(&format!("{full} =>")),
            "{full} is in the menu but never handled"
        );
        checked += 1;
    }
    assert!(checked >= 15, "only {checked} menu commands were checked");
}

/// The window must refuse to shrink past the point where the rail plus a page
/// has nowhere to put anything -- which is how the old sidebar came to clip
/// model names mid-word.
#[test]
fn the_window_enforces_a_minimum_size() {
    let src = main_rs();
    assert!(
        src.contains("WM_GETMINMAXINFO"),
        "no minimum size is enforced"
    );
    assert!(
        src.contains("ptMinTrackSize"),
        "WM_GETMINMAXINFO is handled but sets no minimum"
    );
    // A fixed-width sidebar holding content is exactly what was wrong before.
    assert!(
        !src.contains("fn sidebar_for("),
        "the scaling content sidebar is back; content belongs on a page"
    );
    // The rail is fixed-width because it holds four words, not content -- so it
    // has to stay a small share of even the smallest allowed window. Read from
    // the source rather than written twice, which also keeps the comparison
    // from folding to a constant.
    let min_w: i32 = src
        .lines()
        .find_map(|l| l.trim().strip_prefix("const MIN_W: i32 = "))
        .and_then(|v| v.trim_end_matches(';').parse().ok())
        .expect("MIN_W is not declared in main.rs");
    assert!(
        theme::metric::RAIL * 3 < min_w,
        "the rail is {}px of a {min_w}px minimum window; it has become a          content panel rather than navigation",
        theme::metric::RAIL
    );
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

/// **One primary action per page.** Hermes' rule, and the answer to six buttons
/// of identical weight: the eye needs somewhere to start.
#[test]
fn each_page_marks_exactly_one_primary_action() {
    let src = main_rs();
    let body = function_body(&src, "fn weight_of(");
    let primaries = body.matches("=> Weight::Primary").count();
    assert!(
        (3..=4).contains(&primaries),
        "{primaries} controls claim Weight::Primary; Chat, Models and Settings \
         have one each (Models' follows the tab), and Monitor has none"
    );
    assert!(
        body.contains("page == Page::Chat")
            && body.contains("page == Page::Settings")
            && body.contains("page == Page::Models"),
        "a primary action is claimed without naming the page it is primary on"
    );
}

// -- the pure logic, tested directly ------------------------------------------

/// The endpoint the window advertises must be the port the server is told to
/// bind. Showing one and binding another sends every client to nothing.
#[test]
fn the_advertised_port_is_the_bound_port() {
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
        // A container the engine cannot run reports that instead of its fit,
        // which is the right precedence and would otherwise fail this test for
        // the wrong reason.
        if o.unsupported.is_some() {
            continue;
        }
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

/// Settings must survive the round trip the window relies on, including the
/// theme -- a window that forgets which way round it is on every launch is not
/// a preference, it is a flicker.
#[test]
fn what_is_typed_is_what_comes_back() {
    let cfg = Settings::parse(
        "cache_gib = 6
threads = 4
port = 8231
mode = dark
",
    );
    assert_eq!(cfg.mode, theme::Mode::Dark);
    assert_eq!(Settings::parse(&cfg.render()), cfg);
}

/// The page titles and the rail labels are different strings on purpose -- one
/// is a heading, the other is navigation -- but they must describe the same
/// place, or the rail sends you somewhere the title denies.
#[test]
fn a_rail_label_and_its_page_title_agree() {
    for p in nav::PAGES {
        assert_eq!(
            p.label().to_lowercase(),
            p.title().to_lowercase(),
            "the rail says {:?} and the page says {:?}",
            p.label(),
            p.title()
        );
    }
}

/// Chat is the home surface, so it is the page the window opens on.
#[test]
fn the_window_opens_on_chat() {
    let src = main_rs();
    assert!(
        src.contains("show_page(Page::Chat)"),
        "the window does not open on Chat"
    );
    assert_eq!(nav::PAGES[0], Page::Chat);
}

/// Nothing may be discovered by clicking: every settings row carries a hint,
/// and every hint says what leaving the box empty will do.
#[test]
fn every_setting_explains_itself() {
    let mut without_empty_case = Vec::new();
    for f in nav::FIELDS {
        let h = f.hint.to_lowercase();
        if !h.contains("empty") {
            without_empty_case.push(f.label);
        }
    }
    // `port` has no empty case -- it always has a value -- and neither does the
    // models folder, whose hint names the default path instead.
    let allowed: HashSet<&str> = ["port"].into_iter().collect();
    let unexplained: Vec<_> = without_empty_case
        .into_iter()
        .filter(|l| !allowed.contains(l))
        .collect();
    assert!(
        unexplained.is_empty(),
        "these settings never say what empty means: {unexplained:?}"
    );
}

/// **A model the engine cannot run must be refused before a server starts.**
///
/// Without this the app spawned `chaos-serve`, the server refused the container
/// and exited, and the window went on showing a green dot and an endpoint — so
/// the next message came back "connection actively refused", which reads as a
/// networking fault rather than as "this model does not work".
#[test]
fn an_unrunnable_architecture_is_refused_before_a_server_is_started() {
    let src = main_rs();
    let body = function_body(&src, "fn load_model()");
    let guard = body
        .find("why_not_runnable")
        .expect("load_model does not check whether the architecture can run");
    let spawn = body
        .find("cmd.spawn()")
        .expect("load_model no longer spawns a server");
    assert!(
        guard < spawn,
        "the architecture is checked after the server is started, which is no \
         check at all"
    );
    assert!(
        body.contains("architecture_of"),
        "the architecture is not read from the container; a filename saying \
         Qwen3.6 does not tell you the header says qwen35"
    );
}

/// A child that has exited is not a running model. The window used to keep the
/// dot and the endpoint after `chaos-serve` died.
#[test]
fn a_dead_server_stops_being_reported_as_running() {
    let src = main_rs();
    assert!(
        src.contains("try_wait()"),
        "nothing ever notices that the engine process has exited"
    );
    let tick = function_body(&src, "WM_TIMER =>");
    assert!(
        tick.contains("try_wait()"),
        "the check for a dead child is not on the timer, so it only happens \
         when something else already went wrong"
    );
}

/// **The window must clip its children.** The timer repaints once a second for
/// the uptime and the download bar; without clipping, every child control
/// repaints with it and the whole window flashes continuously.
#[test]
fn the_window_clips_its_children_so_it_does_not_flicker() {
    assert!(
        main_rs().contains("WS_CLIPCHILDREN"),
        "the main window does not clip its children, so every timer tick \
         repaints the transcript, the list and every box"
    );
}

/// **What a button does depends on which model is selected, so the button
/// states have to be recomputed when the selection changes.**
///
/// They were not. `LBN_SELCHANGE` repainted the page beside the list and left
/// LOAD and DOWNLOAD showing whatever the *first* row had decided at startup —
/// so an unfinished download offered LOAD (which then refused) and hid
/// DOWNLOAD (which is the thing that fixes it). Measured through
/// `IsWindowEnabled` from outside the process before and after.
#[test]
fn changing_the_selection_re_decides_which_buttons_are_live() {
    let src = main_rs();
    let arm = src
        .split("(nav::ID_LIST, LBN_SELCHANGE)")
        .nth(1)
        .expect("the list's selection-change arm must exist");
    // Up to the next match arm.
    let arm = &arm[..arm.find("(_,").unwrap_or(arm.len().min(400))];
    assert!(
        arm.contains("sync_enabled()"),
        "LBN_SELCHANGE must call sync_enabled(), or the buttons describe the \
         previously selected model:\n{arm}"
    );
}
