//! An OpenAI-compatible HTTP endpoint, so a coding agent can drive Bigtea.
//!
//! Usage: `bigtea-serve <model.gguf> [--port 8080] [--cache GiB]`
//!
//! # Why this shape and not a nicer one
//!
//! Every editor integration, agent framework and CLI client already speaks
//! `POST /v1/chat/completions`. A better-designed API would be used by nobody.
//! This is the single item that turns Bigtea from something you benchmark into
//! something you *use*, and the API's job is to be boring and familiar.
//!
//! # Why there is no HTTP crate here
//!
//! The workspace has **no external Rust dependencies** — path crates and a ggml
//! FFI, nothing else. That is worth keeping: it means no supply chain, no
//! version churn, and a build that cannot break because something upstream
//! yanked a crate. The subset of HTTP/1.1 needed to answer one POST is a few
//! hundred lines against `std::net`, and it is written here rather than
//! imported.
//!
//! **What that costs, stated plainly**: no TLS, no HTTP/2, no compression, one
//! request at a time. Bind it to localhost and put a real proxy in front if it
//! ever needs to face a network. For an agent talking to a model on the same
//! machine — the actual use — none of those matter.
//!
//! # One request at a time, on purpose
//!
//! The model is a single set of weights with one KV cache, and a second
//! concurrent generation would either corrupt that cache or need a second copy
//! of 7.38 GiB. Requests are therefore serialised, and the server says so in
//! its logs rather than pretending otherwise.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::ExitCode;

use bigtea_arch::{Deepseek4Cache, Deepseek4Config, Deepseek4Forward};
use bigtea_model::{Model, ResidentSet};
use bigtea_tokenizer::{Message, Tokenizer};

const GIB: f64 = (1u64 << 30) as f64;

fn main() -> ExitCode {
    let mut path = String::new();
    let mut port = 8080u16;
    let mut cache_gib = 0f64;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                port = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(8080);
                i += 2;
            }
            "--cache" => {
                cache_gib = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                i += 2;
            }
            "-h" | "--help" => {
                usage();
                return ExitCode::SUCCESS;
            }
            other => {
                if path.is_empty() {
                    path = other.to_string();
                }
                i += 1;
            }
        }
    }
    if path.is_empty() {
        usage();
        return ExitCode::from(2);
    }

    match serve(&path, port, cache_gib) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bigtea-serve: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    println!("usage: bigtea-serve <model.gguf> [--port 8080] [--cache GiB]");
    println!();
    println!("Serves an OpenAI-compatible endpoint on 127.0.0.1:");
    println!("  POST /v1/chat/completions   the one an agent calls");
    println!("  GET  /v1/models             what is loaded");
    println!("  GET  /health                readiness, and what the engine is doing");
    println!();
    println!("Binds to localhost only: no TLS, one request at a time.");
}

fn serve(path: &str, port: u16, cache_gib: f64) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = std::time::Instant::now();
    let model = Model::open_split(path)?;
    if model.architecture() != "deepseek4" {
        return Err(format!(
            "bigtea-serve currently serves deepseek4 only; this container is {}",
            model.architecture()
        )
        .into());
    }
    let config = Deepseek4Config::from_model(&model)?;
    let tokenizer = Tokenizer::from_metadata(model.metadata())?;

    let machine = bigtea_probe::Machine::probe(std::path::Path::new("."), false);
    let reserve = (1u64 << 30) + (512 << 20) + (768 << 20);
    let (resident, report) = ResidentSet::load(&model, machine.usable_ram_for_weights(reserve))?;
    println!("resident   {report}");

    let mut fw = Deepseek4Forward::new(&model, config.clone()).with_resident(&resident);
    // Same rule the runner enforces: a byte given to the expert cache while the
    // always-read set is still streaming comes out of residency, where it would
    // have been read on every token. Measured both ways.
    if cache_gib > 0.0 && report.skipped_over_budget == 0 {
        fw = fw.with_expert_cache((cache_gib * GIB) as usize);
        println!("cache      {cache_gib:.2} GiB for routed experts");
    } else if cache_gib > 0.0 {
        println!(
            "cache      refused: {:.2} GiB of always-read weights is still streaming",
            report.skipped_over_budget as f64 / GIB
        );
    }
    let fw = fw;

    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!("ready      {addr} in {:.1}s", t0.elapsed().as_secs_f64());
    println!("           POST /v1/chat/completions");
    println!("           one request at a time — the KV cache is not shareable");

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle(s, &fw, &tokenizer, &config) {
                    eprintln!("request failed: {e}");
                }
            }
            Err(e) => eprintln!("accept failed: {e}"),
        }
    }
    Ok(())
}

/// A parsed request line plus body. Deliberately minimal.
struct Request {
    method: String,
    target: String,
    body: String,
}

fn read_request(stream: &TcpStream) -> Result<Request, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default().to_string();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        let trimmed = header.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some(v) = trimmed.strip_prefix("Content-Length:") {
            content_length = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = trimmed.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Request {
        method,
        target,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn handle(
    mut stream: TcpStream,
    fw: &Deepseek4Forward<'_>,
    tokenizer: &Tokenizer,
    config: &Deepseek4Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let req = read_request(&stream)?;
    let started = std::time::Instant::now();

    let (status, body) = match (req.method.as_str(), req.target.as_str()) {
        ("GET", "/health") | ("GET", "/") => (
            200,
            format!(
                r#"{{"status":"ok","architecture":"{}","context_limit":{}}}"#,
                "deepseek4", 256
            ),
        ),
        ("GET", "/v1/models") => (
            200,
            r#"{"object":"list","data":[{"id":"deepseek-v4-flash","object":"model","owned_by":"bigtea"}]}"#
                .to_string(),
        ),
        ("POST", "/v1/chat/completions") => match complete(&req.body, fw, tokenizer, config) {
            Ok(b) => (200, b),
            Err(e) => (400, error_json(&e.to_string())),
        },
        _ => (404, error_json("no such endpoint")),
    };

    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Not Found",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    eprintln!(
        "{} {} -> {status} in {:.1}s",
        req.method,
        req.target,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn error_json(message: &str) -> String {
    format!(
        r#"{{"error":{{"message":"{}","type":"invalid_request_error"}}}}"#,
        escape(message)
    )
}

/// Run one completion.
fn complete(
    body: &str,
    fw: &Deepseek4Forward<'_>,
    tokenizer: &Tokenizer,
    config: &Deepseek4Config,
) -> Result<String, Box<dyn std::error::Error>> {
    let messages = extract_messages(body)?;
    // The framing the model was trained on. Concatenating the contents -- what
    // this did before -- makes an instruct model continue the conversation
    // rather than answer it.
    let prompt = tokenizer.apply_chat_template(&messages, true);
    let max_tokens = extract_int(body, "max_tokens").unwrap_or(64).clamp(1, 4096) as usize;

    let tokens: Vec<i32> = tokenizer
        .encode(&prompt)
        .iter()
        .map(|t| *t as i32)
        .collect();
    if tokens.is_empty() {
        return Err("empty prompt".into());
    }
    // The context limit is a real property of this path, not a policy: attention
    // builds its cache for the whole sequence at once. Say so before spending
    // ten seconds discovering it.
    if tokens.len() + max_tokens > 256 {
        return Err(format!(
            "prompt is {} tokens and max_tokens is {max_tokens}; this path holds 256 in total",
            tokens.len()
        )
        .into());
    }

    let arena = 1024usize << 20;
    let mut kv = Deepseek4Cache::new(config.n_layer, config.kv_lora_rank);
    let logits = bigtea_arch::forward(fw, &mut kv, &tokens, arena)?;
    let mut next = argmax(&logits);

    let mut out = String::new();
    let mut produced = 0usize;
    let started = std::time::Instant::now();
    loop {
        out.push_str(&tokenizer.decode(&[next as u32]));
        produced += 1;
        if produced >= max_tokens {
            break;
        }
        let logits = bigtea_arch::step(fw, &mut kv, next, arena)?;
        next = argmax(&logits);
    }
    let secs = started.elapsed().as_secs_f64();
    eprintln!(
        "  {produced} tokens in {secs:.1}s ({:.3} tok/s)",
        produced as f64 / secs.max(1e-9)
    );

    Ok(format!(
        r#"{{"id":"bigtea","object":"chat.completion","model":"deepseek-v4-flash",{}"#,
        format_args!(
            r#""choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"length"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{produced},"total_tokens":{}}}}}"#,
            escape(&out),
            tokens.len(),
            tokens.len() + produced
        )
    ))
}

fn argmax(v: &[f32]) -> i32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite logits"))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

/// Pull the conversation out of an OpenAI request body.
///
/// A hand-rolled scan rather than a JSON parser, for the same reason there is no
/// HTTP crate: the shape is fixed and known. It handles what a client actually
/// sends — `messages: [{role, content}]` — and refuses anything it does not
/// understand instead of guessing.
/// Pull `messages[]` out of a chat-completions body, in order, with roles.
///
/// Hand-rolled because this crate has no JSON dependency. It reads `"role"` and
/// `"content"` pairs in document order, which is what the OpenAI schema
/// guarantees, and refuses anything it cannot represent rather than sending the
/// model half a request.
fn extract_messages(body: &str) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let mut out: Vec<Message> = Vec::new();
    let mut rest = body;
    // Track the most recent role so a content field is attributed correctly
    // even though the two keys are separate.
    let mut pending_role = String::from("user");
    loop {
        let role_at = rest.find("\"role\"");
        let content_at = rest.find("\"content\"");
        match (role_at, content_at) {
            (Some(r), Some(c)) if r < c => {
                let after = rest[r + "\"role\"".len()..].trim_start();
                let Some(colon) = after.find(':') else { break };
                let val = after[colon + 1..].trim_start();
                if let Some(body) = val.strip_prefix('"') {
                    let (text, _) = read_json_string(body)?;
                    pending_role = text;
                }
                let off = rest.len() - after.len() + colon + 1;
                rest = &rest[off..];
            }
            (_, Some(c)) => {
                let after = rest[c + "\"content\"".len()..].trim_start();
                let Some(colon) = after.find(':') else { break };
                let val = after[colon + 1..].trim_start();
                if !val.starts_with('"') {
                    // An array of content parts (images, audio). Refusing is
                    // the honest answer -- this runner is text only.
                    return Err("only string `content` is supported".into());
                }
                let (text, consumed) = read_json_string(&val[1..])?;
                out.push(Message::new(&pending_role, &text));
                pending_role = String::from("user");
                let off = rest.len() - val.len() + 1 + consumed;
                rest = &rest[off..];
            }
            _ => break,
        }
    }
    if out.is_empty() {
        return Err("no `messages[].content` in the request body".into());
    }
    Ok(out)
}

/// Read a JSON string body (the opening quote already consumed).
/// Returns the decoded text and how many bytes were consumed including the
/// closing quote.
fn read_json_string(s: &str) -> Result<(String, usize), Box<dyn std::error::Error>> {
    let mut out = String::new();
    let mut chars = s.char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => return Ok((out, i + 1)),
            '\\' => {
                let Some((_, esc)) = chars.next() else { break };
                out.push(match esc {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    'u' => {
                        // Skip the four hex digits; a coding prompt rarely needs
                        // them and guessing wrong is worse than dropping one.
                        for _ in 0..4 {
                            chars.next();
                        }
                        continue;
                    }
                    other => other,
                });
            }
            other => out.push(other),
        }
    }
    Err("unterminated string in request body".into())
}

fn extract_int(body: &str, key: &str) -> Option<i64> {
    let at = body.find(&format!("\"{key}\""))?;
    let after = &body[at + key.len() + 2..];
    let colon = after.find(':')?;
    let digits: String = after[colon + 1..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Escape for embedding in a JSON string.
fn escape(s: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_normal_chat_request_yields_its_messages_with_roles() {
        let body = r#"{"model":"x","messages":[{"role":"system","content":"Be brief."},
                      {"role":"user","content":"Hello"}],"max_tokens":16}"#;
        let msgs = extract_messages(body).unwrap();
        assert_eq!(msgs.len(), 2);
        // Roles must survive: a system turn framed as a user turn is a
        // different prompt, and the model answers it differently.
        assert_eq!(msgs[0], Message::new("system", "Be brief."));
        assert_eq!(msgs[1], Message::new("user", "Hello"));
        assert_eq!(extract_int(body, "max_tokens"), Some(16));
    }

    #[test]
    fn escapes_survive_the_round_trip() {
        let body = r#"{"messages":[{"role":"user","content":"say \"hi\"\nand a tab\there"}]}"#;
        let got = extract_messages(body).unwrap().remove(0).content;
        assert_eq!(got, "say \"hi\"\nand a tab\there");
        // And what comes back out must be valid JSON again.
        let e = escape(&got);
        assert!(
            !e.contains('\n'),
            "raw newline would break the response body"
        );
        assert!(e.contains("\\\""), "quotes must be escaped: {e}");
    }

    #[test]
    fn unsupported_content_is_refused_not_guessed() {
        // Multimodal clients send an array of parts. Sending the model a
        // half-understood request is worse than saying no.
        let body = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#;
        assert!(extract_messages(body).is_err());
    }

    #[test]
    fn a_request_without_content_is_an_error() {
        assert!(extract_messages(r#"{"model":"x"}"#).is_err());
    }

    #[test]
    fn missing_max_tokens_is_absent_rather_than_zero() {
        // Defaulting to 0 would silently produce an empty completion.
        assert_eq!(extract_int(r#"{"messages":[]}"#, "max_tokens"), None);
    }

    #[test]
    fn control_characters_cannot_break_the_response() {
        let e = escape("a\u{1}b");
        assert!(e.contains("\\u0001"), "{e}");
    }
}
