//! Talking to the `chaos-serve` process the app starts.
//!
//! **Why a child process and not the engine in this binary.** The engine is
//! built inside one stack frame and every part of it borrows from the one
//! before -- weights from a context, a runner from the weights -- so hoisting it
//! into a GUI thread means making it self-referential. The alternative is a
//! second construction path, and `chaos-serve.rs` already carries the comment
//! explaining what that cost last time: *"A second code path is a second place
//! for every fix to be missing from"*, written after a server produced fluent
//! nonsense on a model the CLI got byte-identical.
//!
//! So the app drives the same binary a user would run by hand. Three things
//! fall out of that which the in-process version would not have given:
//! **unloading actually frees the memory**, a model that aborts takes its own
//! process down rather than the window, and the thing being tested is the thing
//! that ships.
//!
//! This is not a web app. There is no browser, no HTML and no webview. Two of
//! our own processes talk over a loopback socket, which is how a great many
//! native applications have always talked to their own backends.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// One piece of a streamed answer.
pub enum Event {
    /// Text to append.
    Token(String),
    /// The model finished normally.
    Done,
    /// Something went wrong; the message is for the user.
    Failed(String),
}

/// Escape a string for embedding in a JSON document.
///
/// Hand-written because the workspace has no JSON crate. Control characters
/// must be escaped or the server rejects the body -- and a prompt containing a
/// newline is the first thing anyone types.
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Pull the `delta.content` out of one SSE payload.
///
/// A hand-rolled scan rather than a parser: the shape is fixed by the server
/// next door, and the alternative is a JSON implementation to maintain. It
/// returns `None` for the role-only first chunk, which carries no content.
pub fn content_of(payload: &str) -> Option<String> {
    let i = payload.find("\"content\":\"")? + "\"content\":\"".len();
    let rest = &payload[i..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'u' => {
                    let hex: String = chars.by_ref().take(4).collect();
                    let n = u32::from_str_radix(&hex, 16).ok()?;
                    out.push(char::from_u32(n)?);
                }
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    None
}

/// Is the server up and answering?
pub fn health(port: u16) -> bool {
    let Ok(mut s) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let req = "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    if s.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = String::new();
    let _ = s.read_to_string(&mut buf);
    buf.contains("\"status\":\"ok\"")
}

/// One line of a connection report.
pub struct Check {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
}

/// Do exactly what a coding agent does, and report each step.
///
/// **A user should not need `curl` to find out whether their agent will work.**
/// Every one of these is a request an OpenAI-compatible client makes: `/health`
/// to see the server is up, `/v1/models` to learn the model's name, and one
/// tiny completion to prove the whole path -- including the API key, if one is
/// required, which is the part that silently fails everywhere else.
///
/// Blocking; call it on a worker thread.
pub fn check(port: u16, api_key: Option<&str>) -> Vec<Check> {
    let mut out = Vec::new();
    let line = |name, ok, detail: String| Check { name, ok, detail };

    // 1. Is anything listening at all?
    let up = health(port);
    out.push(line(
        "server is up",
        up,
        if up {
            format!("127.0.0.1:{port} answered /health")
        } else {
            format!("nothing answered on 127.0.0.1:{port}")
        },
    ));
    if !up {
        return out;
    }

    // 2. The call an agent makes to discover the model name.
    let models = get(port, "/v1/models", api_key);
    let named = models
        .as_deref()
        .and_then(|b| b.split("\"id\":\"").nth(1))
        .and_then(|r| r.split('"').next())
        .map(|s| s.to_string());
    out.push(match &named {
        Some(n) => line("model list", true, format!("/v1/models offers {n}")),
        None => line(
            "model list",
            false,
            match &models {
                Some(b) if b.contains("invalid api key") => {
                    "refused: the API key is wrong".to_string()
                }
                Some(_) => "/v1/models answered, but named no model".to_string(),
                None => "/v1/models did not answer".to_string(),
            },
        ),
    });

    // 3. The one that matters: a real completion, with the key if there is one.
    let mut got = String::new();
    let mut failure = None;
    chat(
        port,
        &[],
        "Reply with the single word OK.",
        8,
        api_key,
        &mut |e| match e {
            Event::Token(t) => got.push_str(&t),
            Event::Failed(m) => failure = Some(m),
            Event::Done => {}
        },
    );
    let ok = failure.is_none() && !got.trim().is_empty();
    out.push(line(
        "a real completion",
        ok,
        match (&failure, got.trim()) {
            (Some(m), _) => format!("failed: {m}"),
            (None, "") => "the model returned nothing".to_string(),
            (None, t) => format!(
                "the model replied {:?}",
                t.chars().take(40).collect::<String>()
            ),
        },
    ));
    out
}

/// A GET, with the key when one is set. Returns the body.
fn get(port: u16, path: &str, api_key: Option<&str>) -> Option<String> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).ok()?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
    let auth = match api_key {
        Some(k) if !k.is_empty() => format!("Authorization: Bearer {k}\r\n"),
        _ => String::new(),
    };
    // CRLF, not LF. Our own server is forgiving about it, but the request line
    // and headers are specified as CRLF-terminated and this is the shape a
    // report about compatibility should itself be sending.
    let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Connection: close\r\n\r\n");
    s.write_all(req.as_bytes()).ok()?;
    let mut buf = String::new();
    let _ = s.read_to_string(&mut buf);
    Some(buf)
}

/// Send a prompt and hand each token to `on`, as it arrives.
///
/// Blocking, and meant to be called on a worker thread: a 144 GB model takes
/// seconds per token, and doing this on the UI thread is a frozen window.
pub fn chat(
    port: u16,
    history: &[(String, String)],
    prompt: &str,
    max_tokens: u32,
    // The key the server was started with, if any.
    api_key: Option<&str>,
    on: &mut dyn FnMut(Event),
) {
    let mut msgs = String::new();
    for (role, text) in history {
        if !msgs.is_empty() {
            msgs.push(',');
        }
        msgs.push_str(&format!(
            r#"{{"role":"{}","content":"{}"}}"#,
            role,
            json_escape(text)
        ));
    }
    if !msgs.is_empty() {
        msgs.push(',');
    }
    msgs.push_str(&format!(
        r#"{{"role":"user","content":"{}"}}"#,
        json_escape(prompt)
    ));

    let body = format!(r#"{{"messages":[{msgs}],"stream":true,"max_tokens":{max_tokens}}}"#);

    let stream = match TcpStream::connect(("127.0.0.1", port)) {
        Ok(s) => s,
        Err(e) => {
            on(Event::Failed(format!("could not reach the model: {e}")));
            return;
        }
    };
    // No read timeout: the first token of a large model can be minutes away,
    // and a deadline here would look exactly like a crash.
    let mut w = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            on(Event::Failed(format!("socket error: {e}")));
            return;
        }
    };
    // **The window's own chat sends the key too.** A key the app displays but
    // does not use would make its own transcript the one client that cannot
    // talk to the model it just started.
    let auth = match api_key {
        Some(k) if !k.is_empty() => format!("Authorization: Bearer {k}\r\n"),
        _ => String::new(),
    };
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n{auth}Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    if w.write_all(head.as_bytes()).is_err() || w.write_all(body.as_bytes()).is_err() {
        on(Event::Failed("the model stopped listening".into()));
        return;
    }
    let _ = w.flush();

    let mut reader = BufReader::new(stream);
    // Skip the header block; the body begins after the blank line.
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                on(Event::Failed("the model closed the connection".into()));
                return;
            }
            Ok(_) => {
                if line == "\r\n" || line == "\n" {
                    break;
                }
            }
            Err(e) => {
                on(Event::Failed(format!("read failed: {e}")));
                return;
            }
        }
    }

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                on(Event::Done);
                return;
            }
            Ok(_) => {
                let t = line.trim_end();
                let Some(payload) = t.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();
                if payload == "[DONE]" {
                    on(Event::Done);
                    return;
                }
                if let Some(text) = content_of(payload) {
                    if !text.is_empty() {
                        on(Event::Token(text));
                    }
                }
            }
            Err(e) => {
                on(Event::Failed(format!("read failed: {e}")));
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_comes_out_of_a_chunk() {
        let c = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        assert_eq!(content_of(c).as_deref(), Some("Hello"));
    }

    /// The first chunk carries a role and no content, and must not be mistaken
    /// for an empty answer.
    #[test]
    fn a_role_only_chunk_yields_nothing() {
        let c = r#"{"choices":[{"delta":{"role":"assistant"}}]}"#;
        assert_eq!(content_of(c), None);
    }

    /// Escapes have to survive the round trip, or every newline the model emits
    /// arrives as a literal backslash-n in the window.
    #[test]
    fn escapes_are_decoded() {
        let c = r#"{"delta":{"content":"a\nb\t\"c\"\\d"}}"#;
        assert_eq!(content_of(c).as_deref(), Some("a\nb\t\"c\"\\d"));
    }

    #[test]
    fn unicode_escapes_are_decoded() {
        let c = r#"{"delta":{"content":"café"}}"#;
        assert_eq!(content_of(c).as_deref(), Some("café"));
    }

    /// A prompt with a quote or a newline must not break the request body --
    /// this is the first thing a real user types.
    #[test]
    fn a_prompt_survives_being_embedded() {
        let s = "say \"hi\"\nthen stop\ttabbed";
        let e = json_escape(s);
        assert!(!e.contains('\n') && !e.contains('\t'));
        assert_eq!(
            content_of(&format!(r#"{{"content":"{e}"}}"#)).as_deref(),
            Some(s)
        );
    }

    #[test]
    fn control_characters_are_escaped() {
        assert_eq!(json_escape("a\u{1}b"), "a\\u0001b");
    }
}
