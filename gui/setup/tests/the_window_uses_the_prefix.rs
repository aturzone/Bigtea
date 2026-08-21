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
        // A generous window: in `do_uninstall` the call sits *after* the
        // confirmation dialog, which is where it belongs. Closing somebody's
        // running Chaos before they have said yes would be its own bug.
        let body = &s[i..(i + 2400).min(s.len())];
        let stop = body
            .find("stop_running_chaos()")
            .unwrap_or_else(|| panic!("{f} acts on the files without closing a running Chaos"));
        if let Some(act) = body
            .find("uninstall_to(")
            .or_else(|| body.find("install_to("))
        {
            assert!(
                stop < act,
                "{f} closes Chaos after touching the files, which is too late"
            );
        }
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

/// **Uninstall asks before it destroys anything.**
///
/// This button had no confirmation at all: one click and every binary was
/// gone, with no dialog, no undo, and nothing on screen beforehand saying what
/// was about to happen. It sits on the same screen as the primary button --
/// which is where somebody who came to *update* presses it by mistake, and is
/// then left with no Chaos and no idea why.
///
/// That is not hypothetical. It happened to Atur, and to a script of mine
/// pointed at what it thought was a throwaway prefix.
#[test]
fn uninstall_confirms_first() {
    let s = source();
    let i = s.find("fn do_uninstall()").expect("no do_uninstall");
    let body = &s[i..(i + 2000).min(s.len())];
    let ask = body
        .find("MB_YESNO")
        .expect("uninstall does not ask anything");
    let act = body
        .find("uninstall_to(")
        .expect("uninstall never uninstalls");
    assert!(
        ask < act,
        "the confirmation comes after the removal, which is not a confirmation"
    );
    assert!(
        body.contains("IDYES"),
        "the answer is not checked, so No would delete anyway"
    );
    // And it says what survives, because that is the question a person has.
    assert!(
        body.contains("KEPT"),
        "the dialog does not say the models are kept"
    );
}

/// The primary button names what it will do, and UNINSTALL is not offered when
/// there is nothing to uninstall.
///
/// Atur: *"if chaos already in system why uninstall?? if not why update and not
/// install??"* -- the label logic was already right (INSTALL on an empty
/// prefix, UPDATE over something older); what was wrong was a second button
/// beside it that could only ever do nothing.
#[test]
fn uninstall_is_hidden_when_nothing_is_installed() {
    let s = source();
    let i = s
        .find("let welcome = screen == Screen::Welcome;")
        .expect("no sync_screen");
    let body = &s[i..(i + 1400).min(s.len())];
    assert!(
        body.contains("existing_install(&prefix_value())"),
        "the uninstall button is shown without asking whether anything is installed"
    );
}

/// A cross-process `SendMessageW` blocks until the other window handles it.
///
/// Chaos busy loading a 7 GiB model is not pumping messages, so the installer
/// would sit there painting nothing for as long as the load took -- which from
/// the outside is a setup where "nothing works".
#[test]
fn closing_a_running_chaos_cannot_hang_the_installer() {
    let s = source();
    let i = s
        .find("fn stop_running_chaos()")
        .expect("no stop_running_chaos");
    let body = &s[i..(i + 1800).min(s.len())];
    assert!(
        body.contains("SendMessageTimeoutW"),
        "a plain SendMessageW here can block the installer indefinitely"
    );
    assert!(
        body.contains("SMTO_ABORTIFHUNG"),
        "the send does not abort on a hung window"
    );
}
