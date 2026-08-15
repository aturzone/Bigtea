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

use bigtea_arch::{
    architecture_is_verified, Deepseek4Cache, Deepseek4Config, Deepseek4Forward, Qwen3Config,
    Qwen3Model, Sampler, SamplerConfig, VERIFIED_ARCHITECTURES,
};
use bigtea_ggml::{Context, WeightSet};
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
            // Read by `configured_threads`, which every graph evaluation calls.
            // Set here rather than threaded through `serve` because the engines
            // are constructed several call-frames down.
            "-t" | "--threads" => {
                if let Some(t) = args.get(i + 1).and_then(|v| v.parse::<usize>().ok()) {
                    std::env::set_var("BIGTEA_THREADS", t.to_string());
                }
                i += 2;
            }
            "-tb" | "--threads-batch" => {
                if let Some(t) = args.get(i + 1).and_then(|v| v.parse::<usize>().ok()) {
                    std::env::set_var("BIGTEA_THREADS_BATCH", t.to_string());
                }
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
    println!("usage: bigtea-serve <model.gguf> [--port 8080] [--cache GiB] [-t N] [-tb N]");
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
    let tokenizer = Tokenizer::from_metadata(model.metadata())?;
    // Same rule as the runner: refuse an architecture nobody has checked rather
    // than serving confident nonsense to an agent that cannot tell.
    if !architecture_is_verified(model.architecture()) {
        return Err(format!(
            "{:?} is not an architecture this build has been verified against              (verified: {}). It may load and answer WRONG with no error.              bigtea-run --force will run it; the server will not, because a client              has no way to see that the answer is unsound.",
            model.architecture(),
            VERIFIED_ARCHITECTURES.join(", "),
        )
        .into());
    }
    println!("model      {} ({})", model.architecture(), model.io_mode());
    let format = tokenizer.chat_format();
    if format.is_known() {
        println!("chat       {} template", format.name());
    } else {
        println!("chat       template not recognised -- using a plain framing;");
        println!("           the model may not respond as an assistant.");
    }

    // Two engines, chosen by architecture, because V4-Flash shares almost none
    // of its graph with the dense path. Both are set up here so the borrowed
    // state outlives the request loop.
    if model.architecture() == "deepseek4" {
        let config = Deepseek4Config::from_model(&model)?;
        let machine = bigtea_probe::Machine::probe(std::path::Path::new("."), false);
        let reserve = (1u64 << 30) + (512 << 20) + (768 << 20);
        let (mut resident, report) =
            ResidentSet::load(&model, machine.usable_ram_for_weights(reserve))?;
        println!("resident   {report}");

        // Rearranged once, at load, and re-bound per block — see `RepackedDense`.
        let repacked = bigtea_arch::RepackedDense::build(&mut resident, &model)?;
        let (n_repacked, repacked_bytes, _) = repacked.stats();
        if n_repacked > 0 {
            println!(
                "repacked   {n_repacked} tensors, {:.2} GiB in the CPU kernels' layout",
                repacked_bytes as f64 / GIB
            );
        }

        let mut fw = Deepseek4Forward::new(&model, config.clone())
            .with_resident(&resident)
            .with_repacked(&repacked);
        // Same rule the runner enforces: a byte given to the expert cache while
        // the always-read set is still streaming comes out of residency, where
        // it would have been read on every token. Measured both ways.
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
        let engine = Engine::Deepseek4 {
            fw: &fw,
            config: &config,
        };
        return run_loop(engine, &tokenizer, port, t0);
    }

    // Dense: Llama, Mistral, Qwen and everything else the qwen3 path covers.
    let config = Qwen3Config::from_model(&model)?;
    let arch = Qwen3Model::new(config.clone());
    arch.verify(&model)?;
    println!(
        "shape      {} layers, {} embd, {} heads ({} kv)",
        config.n_layer, config.n_embd, config.n_head, config.n_head_kv
    );
    if !config.rope_type_is_known {
        println!(
            "           NOTE: {:?} is not an architecture this build has verified;",
            model.architecture()
        );
        println!("           its RoPE layout is assumed. Fluent-but-wrong output points here.");
    }

    let weight_ctx = Context::new_no_alloc(64 << 20)?;
    let mut weights = WeightSet::new();
    let mut bound = 0u64;
    for name in arch.required_tensors() {
        let loc = model
            .location(&name)
            .ok_or_else(|| format!("missing tensor {name}"))?
            .clone();
        let data = model.read_tensor(&name)?;
        bound += data.len() as u64;
        weights.bind(&weight_ctx, &name, loc.ty, &loc.dims, data)?;
    }
    // Tied embeddings: many small models ship no separate output projection and
    // reuse the embedding table. Binding it only when present is what lets
    // those containers load at all.
    if model.location("output.weight").is_some() && weights.get("output.weight").is_none() {
        let loc = model.location("output.weight").expect("checked").clone();
        let data = model.read_tensor("output.weight")?;
        bound += data.len() as u64;
        weights.bind(&weight_ctx, "output.weight", loc.ty, &loc.dims, data)?;
    }
    println!(
        "weights    {} tensors, {:.2} GiB bound in {:.1}s (zero-copy)",
        weights.len(),
        bound as f64 / GIB,
        t0.elapsed().as_secs_f64()
    );

    // `general.name` if the container carries one, else the file stem -- a
    // client's model id should mean something.
    let name = model
        .metadata()
        .get("general.name")
        .and_then(bigtea_gguf::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| {
            std::path::Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "bigtea-model".to_string())
        });
    // The SAME forward the CLI uses, and that is the whole point of this
    // change. The old dense path here called `arch.build_graph` -- a second
    // implementation that never received the QKV bias, the Gemma activation,
    // the post-norms or the soft caps, all of which live in `stream.rs`. Qwen2
    // through this server produced fluent nonsense while `bigtea-run` on the
    // same container was byte-identical to llama.cpp.
    //
    // A second code path is a second place for every fix to be missing from.
    let runner = bigtea_arch::StreamingRunner::new(&model, config.clone(), 1 << 30);
    let engine = Engine::Dense {
        runner: std::cell::RefCell::new(runner),
        weights: &weights,
        name: &name,
        config: config.clone(),
    };
    run_loop(engine, &tokenizer, port, t0)
}

/// Accept and answer requests, one at a time.
fn run_loop(
    engine: Engine<'_>,
    tokenizer: &Tokenizer,
    port: u16,
    t0: std::time::Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!("ready      {addr} in {:.1}s", t0.elapsed().as_secs_f64());
    println!("           POST /v1/chat/completions");
    println!(
        "           context {} tokens, one request at a time",
        engine.context_limit()
    );

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle(s, &engine, tokenizer) {
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
    engine: &Engine<'_>,
    tokenizer: &Tokenizer,
) -> Result<(), Box<dyn std::error::Error>> {
    let req = read_request(&stream)?;
    let started = std::time::Instant::now();

    let (status, body) = match (req.method.as_str(), req.target.as_str()) {
        ("GET", "/health") | ("GET", "/") => (
            200,
            format!(
                r#"{{"status":"ok","model":"{}","context_limit":{}}}"#,
                engine.model_name(),
                engine.context_limit()
            ),
        ),
        ("GET", "/v1/models") => (
            200,
            format!(
                r#"{{"object":"list","data":[{{"id":"{}","object":"model","owned_by":"bigtea"}}]}}"#,
                engine.model_name()
            ),
        ),
        ("POST", "/v1/chat/completions") => {
            let params = Params::from_body(&req.body);
            if params.stream {
                // Streaming owns the socket: headers go out first, then one
                // event per token, so a client sees words appear instead of
                // waiting for the whole answer. Nothing more may be written
                // afterwards, so this returns early.
                return stream_completion(stream, &req, engine, tokenizer, &params, started);
            }
            match generate(&req.body, engine, tokenizer, &params, &mut |_| Ok(())) {
                Ok((text, prompt_tokens, produced, finish)) => (
                    200,
                    completion_json(engine.model_name(), &text, prompt_tokens, produced, finish),
                ),
                Err(e) => (400, error_json(&e.to_string())),
            }
        }
        // The legacy completions endpoint: same engine, no chat framing. Some
        // clients and most autocomplete integrations still speak only this.
        ("POST", "/v1/completions") => {
            let params = Params::from_body(&req.body);
            match generate_raw(&req.body, engine, tokenizer, &params, &mut |_| Ok(())) {
                Ok((text, prompt_tokens, produced, finish)) => (
                    200,
                    format!(
                        r#"{{"id":"bigtea","object":"text_completion","model":"{}","choices":[{{"index":0,"text":"{}","finish_reason":"{}"}}],"usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":{produced},"total_tokens":{}}}}}"#,
                        engine.model_name(),
                        escape(&text),
                        finish.as_str(),
                        prompt_tokens + produced
                    ),
                ),
                Err(e) => (400, error_json(&e.to_string())),
            }
        }
        // Embeddings are a different computation, not a cheaper completion:
        // they need the model's hidden state rather than its logits. This used
        // to be a 501 saying the graph returns only logits -- true of what it
        // returned, false about what it computed. See `embed`.
        ("POST", "/v1/embeddings") => match embed(&req.body, engine, tokenizer) {
            Ok((vectors, prompt_tokens)) => (
                200,
                embeddings_json(engine.model_name(), &vectors, prompt_tokens),
            ),
            Err(e) => (400, error_json(&e.to_string())),
        },
        _ => (404, error_json("no such endpoint")),
    };

    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        501 => "Not Implemented",
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

/// Answer a `stream: true` request as server-sent events.
///
/// The status line and headers are written **before** generation starts, which
/// is the entire point: a client that waits for `Content-Length` cannot show
/// anything until the last token. There is no length to send, so the response
/// ends by closing the connection after `data: [DONE]`.
///
/// An error after the headers are out cannot become a 400 — the status is
/// already committed — so it is delivered as a final chunk carrying the message
/// and then `[DONE]`, which is what the OpenAI clients expect.
fn stream_completion(
    mut stream: TcpStream,
    req: &Request,
    engine: &Engine<'_>,
    tokenizer: &Tokenizer,
    params: &Params,
    started: std::time::Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    // Built by concatenation rather than as one multi-line literal: HTTP header
    // lines are CRLF-separated with no leading whitespace, and a literal that
    // wraps in the source is an easy way to ship indented headers that stricter
    // clients reject.
    let headers = concat!(
        "HTTP/1.1 200 OK\r\n",
        "Content-Type: text/event-stream\r\n",
        "Cache-Control: no-cache\r\n",
        "Access-Control-Allow-Origin: *\r\n",
        "Connection: close\r\n",
        "\r\n",
    );
    stream.write_all(headers.as_bytes())?;
    // The role arrives in its own first chunk, before any content, which is
    // what the OpenAI streaming schema specifies.
    stream.write_all(
        concat!(
            r#"data: {"id":"bigtea","object":"chat.completion.chunk","#,
            r#""choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            "\n\n",
        )
        .as_bytes(),
    )?;
    stream.flush()?;

    let mut sink = stream.try_clone()?;
    let result = generate(&req.body, engine, tokenizer, params, &mut |text| {
        sink.write_all(sse_chunk(text, None).as_bytes())?;
        // Flush per token or the OS buffers the whole answer and "streaming"
        // arrives all at once at the end.
        sink.flush()
    });

    match result {
        Ok((_, _, _, finish)) => {
            stream.write_all(sse_chunk("", Some(finish)).as_bytes())?;
        }
        Err(e) => {
            stream.write_all(
                sse_chunk(
                    &format!(
                        "
[error: {e}]"
                    ),
                    Some(Finish::Stop),
                )
                .as_bytes(),
            )?;
        }
    }
    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()?;
    eprintln!(
        "{} {} -> 200 (stream) in {:.1}s",
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

/// Which model this server is driving.
///
/// V4-Flash shares almost none of its graph with the dense architectures, so
/// they are separate variants rather than one configurable path — the same
/// split `bigtea-run` makes. Serving only V4-Flash was a real limitation: the
/// server is the part an editor or agent talks to, and refusing every Llama and
/// Qwen container made it useless for the models people actually run.
// The two variants differ in size by more than clippy likes, and boxing the
// larger would put an allocation on a value that lives for the whole process
// and is constructed exactly once.
#[allow(clippy::large_enum_variant)]
enum Engine<'a> {
    Deepseek4 {
        fw: &'a Deepseek4Forward<'a>,
        config: &'a Deepseek4Config,
    },
    /// Dense Llama/Qwen, through the SAME `StreamingRunner` the CLI uses.
    ///
    /// It used to call `arch.build_graph` -- a second forward implementation
    /// that never received the QKV bias, the Gemma activation, the post-norms
    /// or the soft caps, because all of those landed in `stream.rs`. Qwen2
    /// through this server produced fluent nonsense while `bigtea-run` on the
    /// same container was byte-identical to llama.cpp. A second code path is a
    /// second place for every fix to be missing from.
    Dense {
        /// Interior mutability because `run_loop` holds the engine by shared
        /// reference and a cached forward pass is inherently stateful. One
        /// request is served at a time, so there is no contention to lose.
        runner: std::cell::RefCell<bigtea_arch::StreamingRunner<'a>>,
        weights: &'a WeightSet<'a>,
        name: &'a str,
        config: Qwen3Config,
    },
}

impl Engine<'_> {
    fn model_name(&self) -> &str {
        match self {
            Engine::Deepseek4 { .. } => "deepseek-v4-flash",
            Engine::Dense { name, .. } => name,
        }
    }

    /// Tokens this path can hold in total, prompt plus generation.
    fn context_limit(&self) -> usize {
        match self {
            // **Was 256, and stayed 256 for a release after the engine stopped
            // needing it.** #61 replaced the position-indexed raw latents with
            // a 1024-slot ring, so the total sequence is no longer capped at
            // all -- what is capped is one pass, at 897 tokens, because a pass
            // must hold `window + nt - 1` distinct positions live at once.
            //
            // The server refused sequences the engine had handled for days.
            // A limit that outlives its cause is worse than no limit: it is a
            // correct-looking refusal, and nobody re-derives those.
            Engine::Deepseek4 { .. } => 897,
            // Bounded by the arena rather than by a cache. Kept modest because
            // every pass rebuilds the graph over the whole sequence.
            Engine::Dense { .. } => 2048,
        }
    }
}

/// One generation, driven by whichever engine is loaded.
///
/// Returns the logits for the next token. The two paths differ in what they
/// carry between calls, so the state lives here rather than in `generate`.
enum State {
    Deepseek4(Deepseek4Cache),
    /// The dense path now keeps a KV cache, like the CLI. Rebuilding over the
    /// whole sequence per token was quadratic AND used the unfixed graph.
    Dense(bigtea_arch::KvCache),
}

/// Everything a request asks for beyond the messages themselves.
struct Params {
    max_tokens: usize,
    sampler: SamplerConfig,
    stop: Vec<String>,
    stream: bool,
    /// From OpenAI's `response_format`. `None` means unconstrained.
    ///
    /// This is the field that makes a local model usable by an agent: without
    /// it, "reply with JSON" is a request the model may decline, and the caller
    /// finds out by failing to parse the answer.
    grammar: Option<bigtea_grammar::Grammar>,
}

impl Params {
    /// Read the OpenAI sampling fields, defaulting the way that API does.
    ///
    /// OpenAI's default temperature is 1.0, not 0.0 — a client that sends no
    /// `temperature` expects sampling, not greedy. That differs from
    /// `bigtea-run`, where greedy is right because it keeps a wrong forward
    /// pass diagnosable.
    fn from_body(body: &str) -> Self {
        // OpenAI's default temperature is 1.0, not 0.0, so the rest is taken
        // from `default()` and only that overridden — a struct literal here
        // would need updating every time a sampler is added, and forgetting
        // would silently reset one to zero rather than failing to compile.
        let mut sampler = SamplerConfig {
            temperature: 1.0,
            ..SamplerConfig::default()
        };
        if let Some(v) = extract_float(body, "temperature") {
            sampler.temperature = v as f32;
        }
        if let Some(v) = extract_float(body, "top_p") {
            sampler.top_p = v as f32;
        }
        if let Some(v) = extract_float(body, "min_p") {
            sampler.min_p = v as f32;
        }
        if let Some(v) = extract_int(body, "top_k") {
            sampler.top_k = v.max(0) as usize;
        }
        if let Some(v) =
            extract_float(body, "repetition_penalty").or(extract_float(body, "repeat_penalty"))
        {
            sampler.repeat_penalty = v as f32;
        }
        // Both are standard OpenAI fields. A client that sends them and is
        // silently ignored gets output that looks like the model repeating
        // itself, with nothing to point at.
        if let Some(v) = extract_float(body, "frequency_penalty") {
            sampler.frequency_penalty = v as f32;
        }
        if let Some(v) = extract_float(body, "presence_penalty") {
            sampler.presence_penalty = v as f32;
        }
        if let Some(v) = extract_int(body, "seed") {
            sampler.seed = v as u64;
        }
        Params {
            max_tokens: extract_int(body, "max_tokens").unwrap_or(64).clamp(1, 4096) as usize,
            sampler,
            stop: extract_string_array(body, "stop"),
            stream: extract_bool(body, "stream").unwrap_or(false),
            grammar: response_format_grammar(body),
        }
    }
}

/// Why generation ended, in the vocabulary the OpenAI API uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Finish {
    /// Hit the token budget.
    Length,
    /// The model emitted end-of-sequence, or a stop sequence was produced.
    Stop,
}

impl Finish {
    fn as_str(self) -> &'static str {
        match self {
            Finish::Length => "length",
            Finish::Stop => "stop",
        }
    }
}

/// Run one completion, handing each newly decoded piece of text to `emit`.
///
/// `emit` returning an error aborts generation — that is how a client
/// disconnecting mid-stream stops the work rather than finishing it for nobody.
fn generate(
    body: &str,
    engine: &Engine<'_>,
    tokenizer: &Tokenizer,
    params: &Params,
    emit: &mut dyn FnMut(&str) -> std::io::Result<()>,
) -> Result<(String, usize, usize, Finish), Box<dyn std::error::Error>> {
    let messages = extract_messages(body)?;
    // The framing the model was trained on. Concatenating the contents -- what
    // this did before -- makes an instruct model continue the conversation
    // rather than answer it.
    let prompt = tokenizer.apply_chat_template(&messages, true);
    run_prompt(&prompt, engine, tokenizer, params, emit)
}

/// `/v1/completions`: the caller's text verbatim, with no chat framing.
///
/// A base model or an autocomplete client wants exactly what it sent. Applying
/// a chat template here would be the mirror of the bug that made instruct
/// models answer the wrong question.
fn generate_raw(
    body: &str,
    engine: &Engine<'_>,
    tokenizer: &Tokenizer,
    params: &Params,
    emit: &mut dyn FnMut(&str) -> std::io::Result<()>,
) -> Result<(String, usize, usize, Finish), Box<dyn std::error::Error>> {
    let prompt =
        extract_json_string(body, "prompt").ok_or("no `prompt` string in the request body")?;
    run_prompt(&prompt, engine, tokenizer, params, emit)
}

/// `/v1/embeddings`: the hidden state, not the logits.
///
/// # Why this stopped being a 501
///
/// The refusal said "this runner's graph returns logits, not hidden states".
/// That was true of what the graph *returned* and false about what it computed:
/// the pre-projection hidden state is the input to the vocabulary matmul and was
/// being discarded a line later. `set_want_embedding` keeps it, at the cost of
/// one `compute` on a tensor already in the graph, and only when asked.
///
/// Taken **after `output_norm` and before the vocabulary projection**, which is
/// where llama.cpp takes it. Earlier and the vector carries a per-model scale
/// that makes similarity between two models meaningless; later and it is a
/// distribution over tokens rather than an embedding.
///
/// Each input gets a **fresh KV cache**. Sharing one would make every embedding
/// after the first a function of the texts before it — the vectors would still
/// look plausible, and they would silently encode the batch order.
///
/// One vector per input, and the total prompt tokens for the `usage` field.
type Embeddings = (Vec<Vec<f32>>, usize);

fn embed(
    body: &str,
    engine: &Engine<'_>,
    tokenizer: &Tokenizer,
) -> Result<Embeddings, Box<dyn std::error::Error>> {
    let inputs = extract_inputs(body)
        .ok_or("no `input` in the request body: expected a string or an array of strings")?;
    if inputs.is_empty() {
        return Err("`input` is empty".into());
    }

    let Engine::Dense {
        runner,
        weights,
        config,
        ..
    } = engine
    else {
        // Deepseek4 runs a different forward path whose output stage this does
        // not reach. Named rather than dressed up as a generic failure.
        return Err(
            "embeddings are implemented for the dense path only; this model uses the \
             V4-Flash path, whose forward pass does not expose a hidden state yet"
                .into(),
        );
    };

    let mut vectors = Vec::with_capacity(inputs.len());
    let mut prompt_tokens = 0usize;
    for text in &inputs {
        let tokens: Vec<u32> = tokenizer.encode(text);
        if tokens.is_empty() {
            return Err("one of the inputs is empty".into());
        }
        if tokens.len() > engine.context_limit() {
            return Err(format!(
                "an input is {} tokens and this path holds {}",
                tokens.len(),
                engine.context_limit()
            )
            .into());
        }
        prompt_tokens += tokens.len();

        let mut kv = bigtea_arch::KvCache::new(
            config.n_layer as usize,
            config.n_head_kv as usize,
            config.head_dim as usize,
        );
        let mut r = runner.borrow_mut();
        r.set_want_embedding(true);
        let _logits = r.forward_cached(weights, &mut kv, &tokens, 0)?;
        let v = r
            .last_embedding()
            .ok_or("the forward pass produced no hidden state")?;
        r.set_want_embedding(false);
        vectors.push(v);
    }
    Ok((vectors, prompt_tokens))
}

/// The shared body of both completion endpoints.
fn run_prompt(
    prompt: &str,
    engine: &Engine<'_>,
    tokenizer: &Tokenizer,
    params: &Params,
    emit: &mut dyn FnMut(&str) -> std::io::Result<()>,
) -> Result<(String, usize, usize, Finish), Box<dyn std::error::Error>> {
    let tokens: Vec<i32> = tokenizer.encode(prompt).iter().map(|t| *t as i32).collect();
    if tokens.is_empty() {
        return Err("empty prompt".into());
    }
    // A real property of the path rather than a policy, and worth stating
    // before ten seconds of loading discovers it. For the V4-Flash path this
    // is now the per-PASS bound rather than a total: the ring holds any length,
    // but one forward pass cannot cover more than `RAW_RING - window + 1`
    // positions. Since the server prefills a prompt in a single pass, the
    // prompt is what the bound applies to.
    let limit = engine.context_limit();
    if tokens.len() + params.max_tokens > limit {
        return Err(format!(
            "prompt is {} tokens and max_tokens is {}; this path holds {limit} in total",
            tokens.len(),
            params.max_tokens
        )
        .into());
    }

    let mut seq = tokens.clone();
    let mut state = match engine {
        Engine::Deepseek4 { config, .. } => {
            State::Deepseek4(Deepseek4Cache::new(config.n_layer, config.kv_lora_rank))
        }
        Engine::Dense { config, .. } => State::Dense(bigtea_arch::KvCache::new(
            config.n_layer as usize,
            config.n_head_kv as usize,
            config.head_dim as usize,
        )),
    };
    let mut logits = advance(engine, &mut state, &seq, true)?;

    let mut sampler = Sampler::new(params.sampler.clone());
    let mut history: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    // `response_format`. The vocabulary is built once as token id -> the bytes
    // that token decodes to, which is what the grammar matches against; the
    // matcher is carried across tokens rather than re-parsing the text so far,
    // because `allowed(prefix)` is quadratic in the answer's length and an
    // agent's structured reply is exactly where that shows.
    // The vocabulary outlives the constraint that borrows it, which is why it
    // is bound here rather than inside the closure.
    let vocab: Vec<Vec<u8>> = params
        .grammar
        .as_ref()
        .map(|_| {
            (0..tokenizer.vocab_size() as u32)
                .map(|id| tokenizer.decode(&[id]).into_bytes())
                .collect()
        })
        .unwrap_or_default();
    let constraint = params
        .grammar
        .as_ref()
        .map(|g| bigtea_grammar::Constraint::new(g.clone(), &vocab));
    let mut matcher = constraint.as_ref().map(|c| c.grammar().matcher());
    let mut grammar_done = false;

    apply_grammar(&constraint, &matcher, &mut logits, &mut grammar_done);
    let mut next = sampler.sample(&logits, &history) as i32;

    let mut out = String::new();
    // Bytes not yet forming a whole character. One character is often several
    // tokens, so converting each token to text on its own would emit
    // replacement characters into the stream permanently.
    let mut pending: Vec<u8> = Vec::new();
    let mut produced = 0usize;
    let mut finish = Finish::Length;
    let started = std::time::Instant::now();

    loop {
        if Some(next as u32) == tokenizer.eos {
            finish = Finish::Stop;
            break;
        }
        history.push(next as u32);
        pending.extend(tokenizer.decode_bytes(&[next as u32]));
        let good = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(e) => e.valid_up_to(),
        };
        if good > 0 {
            let text = String::from_utf8_lossy(&pending[..good]).into_owned();
            pending.drain(..good);
            out.push_str(&text);
            emit(&text)?;
        }
        produced += 1;

        // A stop sequence is checked against the accumulated text, not the
        // token, because it can straddle a token boundary.
        if let Some(cut) = params
            .stop
            .iter()
            .filter(|s| !s.is_empty())
            .find_map(|s| out.find(s.as_str()))
        {
            out.truncate(cut);
            finish = Finish::Stop;
            break;
        }
        if produced >= params.max_tokens {
            break;
        }
        // Advance the grammar by what was actually emitted, then stop if it
        // can accept nothing more -- a satisfied grammar is a finished answer.
        if let Some(m) = matcher.as_mut() {
            m.accept_str(&tokenizer.decode(&[next as u32]));
        }
        if grammar_done {
            finish = Finish::Stop;
            break;
        }
        seq.push(next);
        let mut logits = advance(engine, &mut state, &seq, false)?;
        apply_grammar(&constraint, &matcher, &mut logits, &mut grammar_done);
        if grammar_done {
            finish = Finish::Stop;
            break;
        }
        next = sampler.sample(&logits, &history) as i32;
    }

    let secs = started.elapsed().as_secs_f64();
    eprintln!(
        "  {produced} tokens in {secs:.1}s ({:.3} tok/s), finish={}",
        produced as f64 / secs.max(1e-9),
        finish.as_str()
    );
    Ok((out, tokens.len(), produced, finish))
}

/// Run the model forward and return the next token's logits.
///
/// `first` distinguishes the prompt pass from a continuation. The deepseek4
/// path is incremental — its KV cache means a step feeds one token — while the
/// dense path rebuilds over the whole sequence every time, so it needs `seq`
/// rather than the last token.
/// The server carries token ids as `i32` (the OpenAI shape); the engine wants
/// `u32`. Converted at the boundary rather than changing either side.
fn seq_u32(seq: &[i32]) -> Vec<u32> {
    seq.iter().map(|&t| t as u32).collect()
}

fn advance(
    engine: &Engine<'_>,
    state: &mut State,
    seq: &[i32],
    first: bool,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match (engine, state) {
        (Engine::Deepseek4 { fw, .. }, State::Deepseek4(kv)) => {
            let arena = 1024usize << 20;
            if first {
                Ok(bigtea_arch::forward(fw, kv, seq, arena)?)
            } else {
                // The cache already holds everything before it, so a step feeds
                // exactly the token just chosen.
                let last = *seq.last().expect("non-empty sequence");
                Ok(bigtea_arch::step(fw, kv, last, arena)?)
            }
        }
        (
            Engine::Dense {
                runner, weights, ..
            },
            State::Dense(kv),
        ) => {
            // `first` prefills the whole prompt; every step after feeds exactly
            // the token just chosen, because the cache holds the rest.
            let mut r = runner.borrow_mut();
            if first {
                Ok(r.forward_cached(weights, kv, seq_u32(seq).as_slice(), 0)?)
            } else {
                let last = *seq.last().expect("non-empty sequence") as u32;
                let pos = kv.len();
                Ok(r.forward_cached(weights, kv, &[last], pos)?)
            }
        }
        _ => Err("engine and state disagree -- this is a bug".into()),
    }
}

/// The `/v1/embeddings` response body.
///
/// Floats are written with `{:?}`, which gives Rust's shortest representation
/// that round-trips exactly. A fixed number of decimal places would silently
/// quantise every vector the server ever returns.
fn embeddings_json(model: &str, vectors: &[Vec<f32>], prompt_tokens: usize) -> String {
    let mut data = String::new();
    for (i, v) in vectors.iter().enumerate() {
        if i > 0 {
            data.push(',');
        }
        let mut nums = String::with_capacity(v.len() * 12);
        for (j, x) in v.iter().enumerate() {
            if j > 0 {
                nums.push(',');
            }
            // Non-finite values are not legal JSON. They cannot come out of a
            // healthy forward pass, so emitting 0 would hide a real fault --
            // `null` is at least visible to the client as "not a number".
            if x.is_finite() {
                nums.push_str(&format!("{x:?}"));
            } else {
                nums.push_str("null");
            }
        }
        data.push_str(&format!(
            r#"{{"object":"embedding","index":{i},"embedding":[{nums}]}}"#
        ));
    }
    format!(
        r#"{{"object":"list","model":"{model}","data":[{data}],"usage":{{"prompt_tokens":{prompt_tokens},"total_tokens":{prompt_tokens}}}}}"#
    )
}

/// The non-streaming response body.
fn completion_json(
    model: &str,
    text: &str,
    prompt_tokens: usize,
    produced: usize,
    finish: Finish,
) -> String {
    format!(
        r#"{{"id":"bigtea","object":"chat.completion","model":"{model}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"{}"}}],"usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":{produced},"total_tokens":{}}}}}"#,
        escape(text),
        finish.as_str(),
        prompt_tokens + produced
    )
}

/// One server-sent-event chunk carrying a delta.
fn sse_chunk(delta: &str, finish: Option<Finish>) -> String {
    let finish_field = match finish {
        Some(f) => format!(r#""{}""#, f.as_str()),
        None => "null".to_string(),
    };
    let delta_field = if delta.is_empty() {
        "{}".to_string()
    } else {
        format!(r#"{{"content":"{}"}}"#, escape(delta))
    };
    format!(
        "data: {{\"id\":\"bigtea\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{delta_field},\"finish_reason\":{finish_field}}}]}}\n\n"
    )
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

/// Mask the logits to what the grammar allows, and say when it is finished.
///
/// # Why an empty mask cannot simply be sampled from
///
/// Every token would be `-inf`, the argmax would be arbitrary, and the answer
/// would end looking exactly like a clean stop. Empty has two meanings and they
/// are not the same event: a grammar that has been SATISFIED admits nothing
/// more, which is success; one that is STUCK admits nothing because the text so
/// far cannot be completed, which is a truncated answer. Reporting the second as
/// the first is how a client receives half a JSON object and a `"stop"` reason.
fn apply_grammar(
    constraint: &Option<bigtea_grammar::Constraint>,
    matcher: &Option<bigtea_grammar::Matcher>,
    logits: &mut [f32],
    done: &mut bool,
) {
    let (Some(c), Some(m)) = (constraint.as_ref(), matcher.as_ref()) else {
        return;
    };
    let mask = c.allowed_from(m);
    if mask.is_empty() {
        if !m.is_complete() {
            bigtea_arch::info!(
                "serve      grammar STUCK -- no token can continue and it is not satisfied; \
                 the response is incomplete"
            );
        }
        *done = true;
        return;
    }
    mask.apply(logits);
}

/// Turn OpenAI's `response_format` into a grammar.
///
/// Two shapes are standard and both are honoured:
///
/// ```json
/// {"response_format": {"type": "json_object"}}
/// {"response_format": {"type": "json_schema", "json_schema": {"schema": { ... }}}}
/// ```
///
/// # Why a malformed schema is not silently dropped
///
/// A `response_format` that fails to compile and is then ignored produces free
/// text where the caller is parsing JSON. That failure surfaces in the client,
/// several layers from its cause, and looks like the model disobeying rather
/// than the server discarding the request. So a schema that will not compile is
/// reported here and the request is refused.
fn response_format_grammar(body: &str) -> Option<bigtea_grammar::Grammar> {
    let at = body.find("\"response_format\"")?;
    let rest = &body[at..];
    // The `type` nearest the key. Crude, and deliberately so: this server
    // parses JSON by scanning rather than carrying a parser, and a nested
    // `"type"` inside the schema itself is exactly why the FIRST one after
    // `response_format` is the one taken.
    let ty = extract_string(rest, "type")?;
    match ty.as_str() {
        // Any JSON value. llama.cpp's `--json-schema '{}'` compiles to the
        // same thing, so the two agree on what "json_object" means.
        "json_object" => bigtea_grammar::Grammar::from_json_schema("{}").ok(),
        "json_schema" => {
            // The schema sits under `json_schema.schema` in the OpenAI shape.
            // Taken as a raw substring rather than re-serialised: re-encoding
            // a schema through a scanner would change it, and a changed schema
            // is a changed contract.
            let schema = raw_object_after(rest, "\"schema\"")?;
            match bigtea_grammar::Grammar::from_json_schema(&schema) {
                Ok(g) => Some(g),
                Err(e) => {
                    bigtea_arch::info!("serve      response_format schema rejected: {e}");
                    None
                }
            }
        }
        other => {
            bigtea_arch::info!("serve      response_format type {other:?} not recognised");
            None
        }
    }
}

/// The balanced `{...}` that follows `key`, as raw text.
///
/// Brace counting rather than parsing, and it respects strings and escapes --
/// a schema containing `"pattern": "\\}"` would otherwise close the object
/// early and hand the grammar compiler a truncated document.
fn raw_object_after(body: &str, key: &str) -> Option<String> {
    let at = body.find(key)? + key.len();
    let start = body[at..].find('{')? + at;
    let bytes = body.as_bytes();
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(body[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
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

/// Read a top-level JSON string field, e.g. `"prompt"`.
fn extract_json_string(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let at = body.find(&needle)?;
    let after = body[at + needle.len()..].trim_start();
    let rest = after.strip_prefix(':')?.trim_start();
    let body = rest.strip_prefix('"')?;
    read_json_string(body).ok().map(|(s, _)| s)
}

/// `input` for `/v1/embeddings`, which OpenAI defines as **either** a string or
/// an array of strings.
///
/// Both spellings are in real client code, and a server that takes only the
/// scalar form fails on the batch one with "no input" — which reads like the
/// request was empty rather than like the shape was unsupported.
fn extract_inputs(body: &str) -> Option<Vec<String>> {
    let needle = "\"input\"";
    let at = body.find(needle)?;
    let after = body[at + needle.len()..].trim_start();
    let rest = after.strip_prefix(':')?.trim_start();

    if let Some(one) = rest.strip_prefix('"') {
        return read_json_string(one).ok().map(|(s, _)| vec![s]);
    }
    // The array form. Walked with the same string reader rather than split on
    // commas, because an input containing a comma is ordinary text.
    let mut cur = rest.strip_prefix('[')?.trim_start();
    let mut out = Vec::new();
    loop {
        if let Some(end) = cur.strip_prefix(']') {
            let _ = end;
            return Some(out);
        }
        let s = cur.strip_prefix('"')?;
        let (text, used) = read_json_string(s).ok()?;
        out.push(text);
        cur = s[used..].trim_start();
        cur = match cur.strip_prefix(',') {
            Some(next) => next.trim_start(),
            None => return cur.strip_prefix(']').map(|_| out),
        };
    }
}

/// Read a JSON number as `f64`. Accepts integers too, since `temperature: 1`
/// is legal JSON and common from hand-written clients.
fn extract_float(body: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\"");
    let at = body.find(&needle)?;
    let after = body[at + needle.len()..].trim_start();
    let rest = after.strip_prefix(':')?.trim_start();
    let end = rest
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
        })
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Read a JSON boolean. Absent and malformed are both `None`, so the caller
/// picks the default rather than this guessing one.
fn extract_bool(body: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{key}\"");
    let at = body.find(&needle)?;
    let after = body[at + needle.len()..].trim_start();
    let rest = after.strip_prefix(':')?.trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// Read `"stop"`, which the OpenAI API allows as a string **or** an array of
/// strings. Both spellings are common in the wild, so both are accepted.
/// The string value of `key`, scanning rather than parsing.
///
/// This server carries no JSON parser on purpose; every reader here is a scan.
/// Returns the FIRST match after the caller's slice start, which is what makes
/// `response_format_grammar` able to take the `type` nearest its own key rather
/// than a `"type"` nested inside a schema.
fn extract_string(body: &str, key: &str) -> Option<String> {
    let at = body.find(&format!("\"{key}\""))? + key.len() + 2;
    let rest = &body[at..];
    let colon = rest.find(':')? + 1;
    let open = rest[colon..].find('"')? + colon + 1;
    let mut out = String::new();
    let mut escaped = false;
    for c in rest[open..].chars() {
        if escaped {
            out.push(c);
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

fn extract_string_array(body: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{key}\"");
    let Some(at) = body.find(&needle) else {
        return Vec::new();
    };
    let after = body[at + needle.len()..].trim_start();
    let Some(rest) = after.strip_prefix(':') else {
        return Vec::new();
    };
    let rest = rest.trim_start();
    if let Some(one) = rest.strip_prefix('"') {
        return read_json_string(one)
            .map(|(s, _)| vec![s])
            .unwrap_or_default();
    }
    let Some(mut list) = rest.strip_prefix('[') else {
        return Vec::new();
    };
    let mut out = Vec::new();
    loop {
        list = list.trim_start();
        match list.strip_prefix('"') {
            Some(body) => match read_json_string(body) {
                Ok((s, consumed)) => {
                    out.push(s);
                    list = &body[consumed..];
                }
                Err(_) => break,
            },
            None => break,
        }
        list = list.trim_start();
        match list.strip_prefix(',') {
            Some(next) => list = next,
            None => break,
        }
    }
    out
}

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

    #[test]
    fn sampling_params_are_read_from_the_request() {
        let body = r#"{"messages":[{"role":"user","content":"hi"}],
                       "temperature":0.7,"top_p":0.9,"top_k":40,"seed":123,
                       "max_tokens":32,"stream":true}"#;
        let p = Params::from_body(body);
        assert!((p.sampler.temperature - 0.7).abs() < 1e-6);
        assert!((p.sampler.top_p - 0.9).abs() < 1e-6);
        assert_eq!(p.sampler.top_k, 40);
        assert_eq!(p.sampler.seed, 123);
        assert_eq!(p.max_tokens, 32);
        assert!(p.stream);
    }

    #[test]
    fn an_absent_temperature_defaults_to_the_openai_value_not_greedy() {
        // OpenAI's default is 1.0. Defaulting to 0.0 here would make every
        // answer from every client deterministic and flat, which is a
        // behaviour difference no caller asked for.
        let p = Params::from_body(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert!((p.sampler.temperature - 1.0).abs() < 1e-6);
        assert!(!p.stream, "stream must default to false");
        assert!(p.stop.is_empty());
    }

    #[test]
    fn stop_is_accepted_as_a_string_or_an_array() {
        // The OpenAI schema allows both and clients send both.
        let one = Params::from_body(r#"{"stop":"END","messages":[]}"#);
        assert_eq!(one.stop, vec!["END".to_string()]);
        let many = Params::from_body(
            r#"{"stop":["

","<|eot|>"],"messages":[]}"#,
        );
        assert_eq!(
            many.stop,
            vec![
                "

"
                .to_string(),
                "<|eot|>".to_string()
            ]
        );
    }

    #[test]
    fn floats_parse_whether_written_as_int_or_decimal() {
        assert_eq!(
            extract_float(r#"{"temperature":1}"#, "temperature"),
            Some(1.0)
        );
        assert_eq!(
            extract_float(r#"{"temperature":0.25}"#, "temperature"),
            Some(0.25)
        );
        assert_eq!(extract_float(r#"{"a":1}"#, "temperature"), None);
        assert_eq!(extract_bool(r#"{"stream":true}"#, "stream"), Some(true));
        assert_eq!(extract_bool(r#"{"stream":false}"#, "stream"), Some(false));
        assert_eq!(extract_bool(r#"{"x":1}"#, "stream"), None);
    }

    #[test]
    fn an_sse_chunk_is_one_event_with_a_blank_line_after_it() {
        // Two newlines terminate an event. One, and every client hangs waiting
        // for the rest of it.
        let c = sse_chunk("hi", None);
        assert!(c.starts_with("data: {"));
        assert!(
            c.ends_with(
                "

"
            ),
            "event must end with a blank line: {c:?}"
        );
        assert!(c.contains(r#""content":"hi""#));
        assert!(c.contains(r#""finish_reason":null"#));

        let last = sse_chunk("", Some(Finish::Stop));
        assert!(
            last.contains(r#""delta":{}"#),
            "the final chunk carries no content"
        );
        assert!(last.contains(r#""finish_reason":"stop""#));
    }

    #[test]
    fn a_chunk_escapes_content_that_would_break_the_event() {
        // A raw newline inside the JSON would terminate the event early and
        // the client would see a truncated object.
        let c = sse_chunk("line1\nline2\"quoted\"", None);
        let payload = c.trim_start_matches("data: ").trim_end();
        assert!(
            !payload.contains('\n'),
            "raw newline breaks the event: {payload:?}"
        );
        assert!(
            payload.contains("\\n"),
            "newline must be escaped: {payload:?}"
        );
    }

    #[test]
    fn finish_reason_distinguishes_running_out_from_stopping() {
        assert_eq!(Finish::Length.as_str(), "length");
        assert_eq!(Finish::Stop.as_str(), "stop");
        let j = completion_json("test-model", "hi", 5, 2, Finish::Stop);
        assert!(j.contains(r#""model":"test-model""#));
        assert!(j.contains(r#""finish_reason":"stop""#));
        assert!(j.contains(r#""total_tokens":7"#));
    }

    #[test]
    fn a_raw_prompt_is_read_for_the_completions_endpoint() {
        let body = r#"{"model":"x","prompt":"once upon a","max_tokens":8}"#;
        assert_eq!(
            extract_json_string(body, "prompt").as_deref(),
            Some("once upon a")
        );
        // Absent is None rather than an empty string, so the endpoint can
        // refuse instead of generating from nothing.
        assert_eq!(extract_json_string(r#"{"model":"x"}"#, "prompt"), None);
    }

    #[test]
    fn escapes_survive_a_raw_prompt() {
        let body = r#"{"prompt":"say \"hi\"
then stop"}"#;
        assert_eq!(
            extract_json_string(body, "prompt").as_deref(),
            Some(
                "say \"hi\"
then stop"
            )
        );
    }

    #[test]
    fn input_accepts_both_the_string_and_the_array_form() {
        // OpenAI defines `input` as either. A server that takes only the scalar
        // form fails the batch one with "no input", which reads like an empty
        // request rather than an unsupported shape.
        assert_eq!(
            extract_inputs(r#"{"input":"hello"}"#),
            Some(vec!["hello".to_string()])
        );
        assert_eq!(
            extract_inputs(r#"{"input":["a","b","c"]}"#),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        );
    }

    #[test]
    fn an_input_containing_a_comma_is_one_input() {
        // The array is walked with the JSON string reader rather than split on
        // commas, because a comma inside a text is ordinary.
        assert_eq!(
            extract_inputs(r#"{"input":["one, two","three"]}"#),
            Some(vec!["one, two".to_string(), "three".to_string()])
        );
    }

    #[test]
    fn an_empty_input_array_is_recognised_rather_than_rejected_as_absent() {
        // `[]` parses to zero inputs; the handler rejects it with "`input` is
        // empty", which is a different message from "no `input`".
        assert_eq!(extract_inputs(r#"{"input":[]}"#), Some(vec![]));
        assert_eq!(extract_inputs(r#"{"model":"x"}"#), None);
    }

    #[test]
    fn an_embedding_response_is_shaped_like_openais() {
        let json = embeddings_json("m", &[vec![1.0, -0.5]], 3);
        assert!(json.contains(r#""object":"list""#), "{json}");
        assert!(json.contains(r#""object":"embedding""#), "{json}");
        assert!(json.contains(r#""index":0"#), "{json}");
        assert!(json.contains("[1.0,-0.5]"), "{json}");
        assert!(json.contains(r#""prompt_tokens":3"#), "{json}");
    }

    #[test]
    fn a_non_finite_value_becomes_null_rather_than_invalid_json() {
        // NaN and inf are not legal JSON. They cannot come out of a healthy
        // forward pass, so emitting 0 would hide a fault -- `null` is at least
        // visible to the client as "not a number".
        let json = embeddings_json("m", &[vec![f32::NAN, 1.0]], 1);
        assert!(json.contains("[null,1.0]"), "{json}");
    }
}
