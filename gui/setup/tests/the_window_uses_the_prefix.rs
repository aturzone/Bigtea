//! The installer window must act on the prefix it was given.
//!
//! Read from the source, because the window cannot be opened in CI. That is a
//! weaker check than driving it -- and it is the one that would have caught
//! this, which a type signature alone did not.

fn source() -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    std::fs::read_to_string(p).expect("read gui/setup/src/main.rs")
}

/// **`--prefix` was parsed, honoured in silent mode, and dropped by the
/// window.** `run()` filled the box with `default_prefix()` unconditionally, so
/// `chaos-setup --prefix D:\Chaos` opened a window that would install to, and
/// uninstall from, `%LOCALAPPDATA%\Chaos` instead.
///
/// That is not a cosmetic bug. It removes the wrong directory, and it did:
/// found by scripting the UNINSTALL button against a throwaway prefix and
/// watching it delete the real installation.
#[test]
fn the_window_is_given_the_prefix_rather_than_assuming_it() {
    let s = source();
    assert!(
        s.contains("pub fn run(prefix: &std::path::Path)"),
        "run() does not take a prefix, so the window cannot honour --prefix"
    );
    assert!(
        s.contains("setup::run(&prefix)"),
        "main() does not pass the parsed prefix to the window"
    );
    // And the box is filled from it.
    let i = s.find("let prefix_text =").expect("no prefix box text");
    let line = &s[i..s[i..].find('\n').map_or(s.len(), |n| i + n)];
    assert!(
        !line.contains("default_prefix()"),
        "the box is still filled with the default: {line}"
    );
}

/// **Closing the window stopped quitting in v0.0.12.** Chaos hides to the
/// notification area, so "I closed it" no longer means the process is gone --
/// and a running executable keeps its own file open. Uninstall then leaves
/// `chaos-app.exe` behind and install cannot replace it, which from the outside
/// is "the uninstall button does not work".
#[test]
fn install_and_uninstall_close_a_running_chaos_first() {
    let s = source();
    assert!(
        s.contains("fn stop_running_chaos()"),
        "nothing closes a running Chaos before touching its files"
    );
    for f in ["fn do_install()", "fn do_uninstall()"] {
        let i = s.find(f).unwrap_or_else(|| panic!("no {f}"));
        let body = &s[i..(i + 700).min(s.len())];
        assert!(
            body.contains("stop_running_chaos()"),
            "{f} acts on the files without closing a running Chaos"
        );
    }
    // It asks the app to quit rather than killing it: a terminated app leaves
    // its model resident and its tray icon on screen.
    let i = s.find("fn stop_running_chaos()").unwrap();
    let body = &s[i..(i + 1600).min(s.len())];
    assert!(
        body.contains("IDM_EXIT"),
        "the running Chaos is stopped some way other than its own Exit command"
    );
}
