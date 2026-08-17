//! The served page, checked for the ways it can be broken while still
//! returning 200.
//!
//! A page is not like a function: every failure here is silent. It references a
//! CDN and works on the developer's machine and nowhere offline; it loses a
//! closing tag and renders as a blank rectangle; it drops the element the script
//! reaches for and throws in the console where nobody is looking. None of that
//! shows up as a failing request, so it has to be asserted here.

use chaos_arch::ui::PAGE;

/// **The constraint the whole project runs under.** Chaos downloads nothing on
/// its own, and a page that fetches a font, a script or a stylesheet from
/// somewhere else is broken on exactly the machine this is built for: one with
/// a large model on disk and no working network.
#[test]
fn nothing_is_fetched_from_outside_the_binary() {
    for probe in [
        "http://",
        "https://",
        "//cdn",
        "src=\"/",
        "@import",
        "googleapis",
        "unpkg",
        "jsdelivr",
        "cdnjs",
    ] {
        assert!(
            !PAGE.contains(probe),
            "the page reaches outside the binary for `{probe}`; it must be self-contained"
        );
    }
}

/// Relative fetches to our own origin are the exception, and the only two.
#[test]
fn it_talks_to_our_own_endpoints() {
    assert!(PAGE.contains("'/v1/chat/completions'"));
    assert!(PAGE.contains("'/v1/models'"));
}

/// Every element the script looks up must exist, or the page throws on load and
/// shows nothing. Cheap to check, and it is the mistake a later edit makes.
#[test]
fn every_element_the_script_reaches_for_exists() {
    for id in [
        "log", "input", "form", "send", "stat", "main", "empty", "model",
    ] {
        assert!(
            PAGE.contains(&format!("id=\"{id}\"")),
            "the script calls getElementById('{id}') but no element declares it"
        );
        assert!(
            PAGE.contains(&format!("getElementById('{id}')")),
            "element '{id}' is declared but never used -- remove it or wire it up"
        );
    }
}

/// Tags balance. Not a parser -- just the containers whose loss blanks the page.
#[test]
fn the_document_closes_what_it_opens() {
    for tag in [
        "html", "head", "body", "header", "main", "footer", "style", "script", "form",
    ] {
        // `<head` is a prefix of `<header`, so a bare substring count reports
        // two opens for one tag. Only `<tag>` or `<tag ` starts this element.
        let open =
            PAGE.matches(&format!("<{tag}>")).count() + PAGE.matches(&format!("<{tag} ")).count();
        let close = PAGE.matches(&format!("</{tag}>")).count();
        assert_eq!(open, close, "<{tag}> opened {open} times, closed {close}");
    }
}

/// The page is served with a `Content-Length` counted in **bytes**. Any
/// non-ASCII character makes `str::len()` disagree with what a naive reader
/// would guess from the character count, and a wrong length truncates the page
/// or hangs the browser waiting for bytes that never come. Escapes keep it
/// ASCII; this pins that.
#[test]
fn the_page_is_ascii_so_content_length_cannot_be_wrong() {
    assert!(
        PAGE.is_ascii(),
        "the page contains non-ASCII; use an HTML entity or a \\u escape instead"
    );
    assert_eq!(PAGE.len(), PAGE.chars().count());
}

/// It renders in both themes. A page that defines its colours only inside a
/// `prefers-color-scheme` block is unstyled for everyone else.
#[test]
fn both_themes_are_defined() {
    let root = PAGE.find(":root {").expect("no base palette");
    let dark = PAGE
        .find("prefers-color-scheme: dark")
        .expect("no dark palette");
    assert!(
        root < dark,
        "the base palette must come before the dark override"
    );
    for token in ["--bg", "--fg", "--dim", "--line", "--panel", "--accent"] {
        assert!(
            PAGE[..dark].contains(token),
            "{token} is only defined under a media query, so it is unset by default"
        );
    }
}

/// Streaming is the point: without it a 2.4 s/token model looks frozen for a
/// minute. The parser must hold a partial event rather than dropping it, since
/// a network chunk boundary lands mid-event routinely.
#[test]
fn it_streams_and_survives_a_split_chunk() {
    assert!(
        PAGE.contains("stream: true"),
        "the request must ask for SSE"
    );
    assert!(PAGE.contains("[DONE]"), "the terminator must be recognised");
    assert!(
        PAGE.contains("buf = parts.pop()"),
        "the remainder after the last event boundary must be carried forward, \
         or an event split across two chunks is silently lost"
    );
}

/// Text from the model goes in as **text**, never as markup.
#[test]
fn model_output_is_never_treated_as_html() {
    assert!(
        !PAGE.contains("innerHTML"),
        "innerHTML would let model output inject markup into the page"
    );
    assert!(PAGE.contains("out.textContent = answer"));
}
