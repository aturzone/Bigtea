//! Generate text. The first end-to-end path through every layer of Bigtea.
//!
//! Usage: `bigtea-run <model.gguf> "prompt" [-n tokens]`
//!
//! Pipeline: container -> residency -> zero-copy weight binding -> tokenizer
//! -> forward graph -> logits -> sampling -> text.

use std::process::ExitCode;

use bigtea_arch::{
    architecture_is_verified, neg_log_prob, KvCache, Qwen3Config, Qwen3Model, Sampler,
    SamplerConfig, VERIFIED_ARCHITECTURES,
};
use bigtea_ggml::{Context, WeightSet};
use bigtea_model::{Model, ResidentSet};
use bigtea_tokenizer::{Message, Tokenizer};

const GIB: f64 = (1u64 << 30) as f64;

/// Emit generated tokens as text, holding back incomplete UTF-8.
///
/// One character is often several tokens - an emoji is four byte-fallback
/// tokens under SentencePiece, and a Persian or Chinese character is two or
/// three. Converting each token to a `String` on its own turns every incomplete
/// fragment into a replacement character permanently, so the bytes are buffered
/// and flushed only at a valid UTF-8 boundary.
struct TokenWriter {
    pending: Vec<u8>,
    colored: bool,
}

impl TokenWriter {
    fn new() -> Self {
        TokenWriter {
            pending: Vec::new(),
            colored: false,
        }
    }

    fn push(&mut self, tokenizer: &Tokenizer, id: u32) {
        use std::io::Write;
        self.pending.extend(tokenizer.decode_bytes(&[id]));
        // Longest valid prefix: everything up to a trailing partial character.
        let good = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) => e.valid_up_to(),
        };
        if good > 0 {
            let text = String::from_utf8_lossy(&self.pending[..good]).into_owned();
            print!("{text}");
            let _ = std::io::stdout().flush();
            self.pending.drain(..good);
        }
    }

    /// `push`, with `--color` and `--special` applied.
    ///
    /// Control tokens are hidden by default because a chat template's
    /// `<|im_end|>` is framing, not output, and printing it makes every answer
    /// look broken. `--special` shows them, which is what you want when the
    /// question is *why* an answer ended where it did.
    fn push_visible(&mut self, tokenizer: &Tokenizer, id: u32, ui: &Ui) {
        use std::io::Write;
        if ui.special && tokenizer.is_control(id) {
            if ui.color {
                print!("{COLOR_OFF}");
            }
            print!("{}", tokenizer.control_text(id));
            if ui.color {
                print!("{COLOR_GEN}");
            }
            let _ = std::io::stdout().flush();
            return;
        }
        if ui.color && !self.colored {
            print!("{COLOR_GEN}");
            self.colored = true;
        }
        self.push(tokenizer, id);
    }

    /// Anything still buffered at the end was genuinely malformed, so it is
    /// shown lossily rather than silently dropped.
    fn finish(&mut self) {
        if !self.pending.is_empty() {
            print!("{}", String::from_utf8_lossy(&self.pending));
            self.pending.clear();
        }
    }
}

/// RoPE settings the user supplied, each `None` unless asked for.
///
/// `Option` per field rather than a filled-in struct, because "not given" and
/// "given the same value the container has" must not be the same thing: the
/// container is right far more often than a flag is, and silently overwriting
/// its RoPE base with a default is how a long-context model starts answering
/// fluently and wrongly.
#[derive(Clone, Default)]
struct RopeOverrides {
    freq_base: Option<f32>,
    freq_scale: Option<f32>,
    scaling: Option<String>,
    ext_factor: Option<f32>,
    attn_factor: Option<f32>,
    beta_fast: Option<f32>,
    beta_slow: Option<f32>,
    orig_ctx: Option<u32>,
}

impl RopeOverrides {
    /// Apply to a config read from the container, and say what changed.
    ///
    /// Printed rather than applied quietly: RoPE is the setting most likely to
    /// turn a working model into a fluent-but-wrong one, and a user who mistyped
    /// `--rope-freq-base 1000` for `100000` should be able to see it.
    fn apply(&self, c: &mut Qwen3Config) {
        let mut changed: Vec<String> = Vec::new();
        if let Some(v) = self.freq_base {
            changed.push(format!("freq_base {} -> {v}", c.rope_freq_base));
            c.rope_freq_base = v;
        }
        if let Some(v) = self.freq_scale {
            changed.push(format!("freq_scale {} -> {v}", c.rope_freq_scale));
            c.rope_freq_scale = v;
        }
        match self.scaling.as_deref() {
            Some("none") => {
                changed.push("scaling -> none".into());
                c.rope_freq_scale = 1.0;
                c.rope_ext_factor = 0.0;
            }
            Some("linear") => {
                changed.push("scaling -> linear".into());
                c.rope_ext_factor = 0.0;
            }
            Some("yarn") => {
                changed.push("scaling -> yarn".into());
                // Only default the mix if the user did not state one; a bare
                // `--rope-scaling yarn` means "on", not "on at zero strength".
                if self.ext_factor.is_none() && c.rope_ext_factor == 0.0 {
                    c.rope_ext_factor = 1.0;
                }
            }
            _ => {}
        }
        if let Some(v) = self.ext_factor {
            c.rope_ext_factor = v;
            changed.push(format!("yarn ext_factor {v}"));
        }
        if let Some(v) = self.attn_factor {
            c.rope_attn_factor = v;
            changed.push(format!("yarn attn_factor {v}"));
        }
        if let Some(v) = self.beta_fast {
            c.rope_beta_fast = v;
            changed.push(format!("yarn beta_fast {v}"));
        }
        if let Some(v) = self.beta_slow {
            c.rope_beta_slow = v;
            changed.push(format!("yarn beta_slow {v}"));
        }
        if let Some(v) = self.orig_ctx {
            c.rope_orig_ctx = v;
            changed.push(format!("yarn orig_ctx {v}"));
        }
        if !changed.is_empty() {
            bigtea_arch::info!("rope       overridden: {}", changed.join(", "));
        }
    }
}

/// How the terminal side behaves — llama.cpp's interaction flags.
///
/// Grouped rather than passed individually because they are all "how it talks
/// to a person", they travel together, and a function taking twenty `bool`s
/// invites the argument-order bug that no test catches.
#[derive(Clone, Default)]
struct Ui {
    interactive: bool,
    /// Take a turn from the user before generating anything.
    interactive_first: bool,
    conversation: bool,
    single_turn: bool,
    multiline: bool,
    display_prompt: bool,
    color: bool,
    /// Render control tokens like `<|im_end|>` instead of hiding them.
    special: bool,
    print_token_count: bool,
    verbose_prompt: bool,
    in_prefix: String,
    in_suffix: String,
    in_prefix_bos: bool,
}

/// ANSI, and only when asked for and not writing to a pipe.
const COLOR_GEN: &str = "\x1b[32m";
const COLOR_OFF: &str = "\x1b[0m";

/// Read one turn from the user.
///
/// `Ok(None)` means end of input — Ctrl-D, or a pipe that has run out — which
/// ends the session rather than being an error.
///
/// With `--multiline-input`, a line ending in a single backslash continues onto
/// the next, because a shell prompt is the wrong place to be unable to paste a
/// paragraph.
fn read_user_turn(ui: &Ui) -> Result<Option<String>, Box<dyn std::error::Error>> {
    use std::io::{BufRead, Write};
    let mut out = String::new();
    let stdin = std::io::stdin();
    loop {
        if ui.color {
            print!("{COLOR_OFF}");
        }
        print!("\n> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            return Ok(None);
        }
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if ui.multiline {
            if let Some(head) = trimmed.strip_suffix('\\') {
                out.push_str(head);
                out.push('\n');
                continue;
            }
        }
        out.push_str(trimmed);
        // A blank first line is a request for a prompt, not a turn to send.
        if out.trim().is_empty() {
            out.clear();
            continue;
        }
        return Ok(Some(out));
    }
}

/// Interpret the backslash escapes llama.cpp's `-e` accepts.
///
/// `-p "Line one\nLine two"` is the ordinary way to write a two-line prompt on
/// a command line, and without this the model is asked about a literal
/// backslash-n. Unknown escapes are left exactly as written rather than
/// swallowed, so a Windows path in a prompt survives.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some('\'') => out.push('\''),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('x') => {
                // Exactly two hex digits, and only if both are there.
                let hex: String = chars.clone().take(2).collect();
                match u8::from_str_radix(&hex, 16) {
                    Ok(byte) if hex.len() == 2 => {
                        out.push(byte as char);
                        chars.next();
                        chars.next();
                    }
                    _ => {
                        out.push('\\');
                        out.push('x');
                    }
                }
            }
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// A saved KV cache, so a repeated prompt does not pay prefill twice.
///
/// # Why this earns its complexity
///
/// Prefill is the expensive half for anything with a long prompt: a system
/// prompt plus a document is thousands of tokens of work before the first token
/// of the answer. Re-running the same prefix every invocation is the single
/// largest avoidable cost in an agent loop, and llama.cpp's `--prompt-cache`
/// exists for exactly that.
///
/// # Format, and why every field is checked
///
/// ```text
/// "BTPC" u32 version  u64 fingerprint  u32 kv_type  u32 layers
/// u32 positions       u32 n_tokens     [u32; n_tokens] tokens
/// per layer: u64 len, k bytes, u64 len, v bytes
/// ```
///
/// The **fingerprint** is the shape the cache was built with. Restoring keys
/// computed by a different model, or with a different KV quantisation, is not
/// an error anywhere downstream — attention simply reads numbers that mean
/// nothing, and the answer is fluent and wrong. So a mismatch discards the
/// file rather than trying to use part of it.
struct PromptCache;

impl PromptCache {
    const MAGIC: &'static [u8; 4] = b"BTPC";
    const VERSION: u32 = 1;

    /// Shape the cache depends on. Any change invalidates every saved file.
    fn fingerprint(config: &Qwen3Config, kv: bigtea_arch::KvType) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64; // FNV-1a
        for part in [
            config.n_layer as u64,
            config.n_embd as u64,
            config.n_head as u64,
            config.n_head_kv as u64,
            config.head_dim as u64,
            config.vocab_size as u64,
            kv.ggml_type() as u64,
        ] {
            h ^= part;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Read a saved cache, returning its tokens and per-layer bytes.
    ///
    /// Any inconsistency returns `None`: a prompt cache is an optimisation, and
    /// failing to use one must never fail the run.
    #[allow(clippy::type_complexity)]
    fn load(path: &str, want: u64) -> Option<(Vec<u32>, Vec<(Vec<u8>, Vec<u8>)>)> {
        let data = std::fs::read(path).ok()?;
        let mut at = 0usize;
        let mut take = |n: usize| -> Option<&[u8]> {
            let end = at.checked_add(n)?;
            let out = data.get(at..end)?;
            at = end;
            Some(out)
        };
        if take(4)? != Self::MAGIC {
            return None;
        }
        let u32_at = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        if u32_at(take(4)?) != Self::VERSION {
            return None;
        }
        let fp = u64::from_le_bytes(take(8)?.try_into().ok()?);
        if fp != want {
            return None;
        }
        let _kv = u32_at(take(4)?);
        let layers = u32_at(take(4)?) as usize;
        let _positions = u32_at(take(4)?) as usize;
        let n_tokens = u32_at(take(4)?) as usize;
        let mut tokens = Vec::with_capacity(n_tokens);
        for _ in 0..n_tokens {
            tokens.push(u32_at(take(4)?));
        }
        let mut per_layer = Vec::with_capacity(layers);
        for _ in 0..layers {
            let kn = u64::from_le_bytes(take(8)?.try_into().ok()?) as usize;
            let k = take(kn)?.to_vec();
            let vn = u64::from_le_bytes(take(8)?.try_into().ok()?) as usize;
            let v = take(vn)?.to_vec();
            per_layer.push((k, v));
        }
        Some((tokens, per_layer))
    }

    /// Write the cache covering `tokens`.
    fn save(path: &str, fingerprint: u64, cache: &KvCache, tokens: &[u32]) -> std::io::Result<u64> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(Self::MAGIC);
        out.extend_from_slice(&Self::VERSION.to_le_bytes());
        out.extend_from_slice(&fingerprint.to_le_bytes());
        out.extend_from_slice(&cache.kind().ggml_type().to_le_bytes());
        out.extend_from_slice(&(cache.layers() as u32).to_le_bytes());
        out.extend_from_slice(&(cache.len() as u32).to_le_bytes());
        out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for t in tokens {
            out.extend_from_slice(&t.to_le_bytes());
        }
        for layer in 0..cache.layers() {
            let k = cache.keys(layer);
            let v = cache.values(layer);
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k);
            out.extend_from_slice(&(v.len() as u64).to_le_bytes());
            out.extend_from_slice(v);
        }
        std::fs::write(path, &out)?;
        Ok(out.len() as u64)
    }
}

/// How many leading tokens two sequences share.
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Apply `--chat-template`, refusing a name this build does not implement.
///
/// Both engine paths call it: the dense one and V4-Flash build their
/// tokenizers separately, and a flag honoured on only one of them is the
/// failure `-t` had for weeks.
fn force_chat_template(
    tokenizer: &mut Tokenizer,
    name: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(name) = name else {
        return Ok(());
    };
    match bigtea_tokenizer::ChatFormat::from_name(name) {
        Some(fmt) => {
            bigtea_arch::info!("chat       forced to the {} template", fmt.name());
            tokenizer.set_chat_format(fmt);
            Ok(())
        }
        // Refused rather than falling back to the generic framing: a template
        // silently not applied is a model answering the wrong question
        // fluently, which is this project's most expensive failure.
        None => Err(format!(
            "--chat-template: unknown template {name:?}. Known: {}",
            bigtea_tokenizer::ChatFormat::known_names().join(", ")
        )
        .into()),
    }
}

/// Pin every resident tensor in physical memory.
///
/// The ceiling is raised **once, for the whole set**, before any tensor is
/// locked: doing it per tensor would raise the quota N times and still fail on
/// the first large one, since the quota is a total rather than a per-call
/// limit.
///
/// A failure is counted rather than aborting. A partially locked residency is
/// still better than none, and the caller reports how much actually took.
fn lock_resident(weights: &WeightSet<'_>) -> bigtea_io::lock::LockReport {
    let mut report = bigtea_io::lock::LockReport::default();
    let slices = weights.bound_slices();
    let total: u64 = slices.iter().map(|s| s.len() as u64).sum();
    if let Err(e) = bigtea_io::lock::reserve_working_set(total) {
        report.failed_bytes = total;
        report.reason = e;
        return report;
    }
    for bytes in slices {
        match bigtea_io::lock::lock_bytes(bytes) {
            Ok(()) => report.locked_bytes += bytes.len() as u64,
            Err(e) => {
                report.failed_bytes += bytes.len() as u64;
                if report.reason.is_empty() {
                    report.reason = e;
                }
            }
        }
    }
    report
}

/// Parse `key=type:value` for `--override-kv`.
///
/// llama.cpp's spelling exactly, because muscle memory is the point of
/// matching a CLI. Returns `None` on anything malformed so the caller can
/// refuse the run: an override silently dropped is worse than no override,
/// since the user believes the container has been corrected.
fn parse_override(spec: &str) -> Option<(String, bigtea_gguf::Value)> {
    let (key, rest) = spec.split_once('=')?;
    let (ty, raw) = rest.split_once(':')?;
    let value = match ty.trim().to_ascii_lowercase().as_str() {
        "int" | "i32" | "i64" | "u32" | "u64" => bigtea_gguf::Value::I64(raw.trim().parse().ok()?),
        "float" | "f32" | "f64" => bigtea_gguf::Value::F32(raw.trim().parse().ok()?),
        "bool" => bigtea_gguf::Value::Bool(match raw.trim() {
            "true" | "1" => true,
            "false" | "0" => false,
            _ => return None,
        }),
        "str" | "string" => bigtea_gguf::Value::String(raw.to_string()),
        _ => return None,
    };
    Some((key.trim().to_string(), value))
}

/// Apply the model's chat template when asked, and say which one was used.
///
/// An instruct model trained on `<|im_start|>user` does not fail on raw text —
/// it continues it. Asked to "Write one sentence about the sea", Llama-3.2
/// answered "The sentence should be concise and evocative", because it was
/// completing an instruction rather than following one.
fn framed(tokenizer: &Tokenizer, prompt: &str, chat: bool, system: Option<&str>) -> String {
    // A system prompt is only meaningful inside a template — there is nowhere
    // to put it in raw completion — so asking for one implies chat framing
    // rather than being silently dropped.
    if !chat && system.is_none() {
        return prompt.to_string();
    }
    let format = tokenizer.chat_format();
    if format.is_known() {
        bigtea_arch::info!("chat       {} template", format.name());
    } else {
        // Do not pretend. An unrecognised template framed as someone else's is
        // how a model quietly answers the wrong question.
        bigtea_arch::info!("chat       template not recognised -- using a plain framing;");
        bigtea_arch::info!("           the model may not respond as an assistant.");
    }
    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(Message::new("system", sys));
    }
    messages.push(Message::new("user", prompt));
    tokenizer.apply_chat_template(&messages, true)
}

/// Perplexity over a corpus: the standard way to say a model still works.
///
/// # Why this exists
///
/// Every correctness check in this project so far has been "does it say Paris".
/// That catches a broken forward pass and nothing subtler — a slightly wrong
/// RoPE base, a rounding difference in the KV cache, or a repacked kernel that
/// is *almost* right all answer Paris. Perplexity is a number over thousands of
/// tokens, so it moves when any of those are wrong, and it is what llama.cpp's
/// `llama-perplexity` reports, so the two can be compared directly.
///
/// # The method, stated because it decides the number
///
/// The corpus is cut into chunks of `chunk_size` tokens. Each chunk starts with
/// an **empty KV cache**, and only positions in the **second half** contribute
/// `-log P(token | everything before it in this chunk)`. The result is
/// `exp(total / count)`.
///
/// Scoring the second half only is llama.cpp's rule, and it is not arbitrary:
/// token 1 of a chunk is predicted from a single token of context and token 400
/// from 400, so including the early ones measures mostly how short the context
/// was. Every scored token here has at least `chunk_size / 2` of history.
/// Scoring from position 1 instead gave 1.9232 where this gives a different
/// number on the same file — the windowing *is* the measurement.
///
/// Tokens are fed **one at a time**, which is slow and deliberate: the forward
/// pass projects only the final position through the output matrix (that was a
/// 253 GFLOP saving on prefill), so per-position logits are only available a
/// step at a time. Correct and slow beats fast and approximate for a number
/// whose whole purpose is to be compared.
///
/// **This is not bit-comparable to `llama-perplexity`** unless the chunking
/// matches: it defaults to 512 and different windowing gives a different number
/// on the same file and the same model. Compare the two only with the same
/// `--ppl-chunk` and the same corpus, and say so when quoting.
#[allow(clippy::too_many_arguments)]
fn perplexity_run(
    runner: &mut bigtea_arch::StreamingRunner<'_>,
    weights: &WeightSet<'_>,
    config: &Qwen3Config,
    tokens: &[u32],
    chunk_size: usize,
    kv_type: bigtea_arch::KvType,
    t0: std::time::Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    let vocab = config.vocab_size as usize;
    if tokens.len() < 2 {
        return Err("perplexity needs at least 2 tokens; use -f with a real corpus".into());
    }
    let mut total_nll = 0f64;
    let mut counted = 0usize;
    let mut chunks = 0usize;
    let start = std::time::Instant::now();

    for chunk in tokens.chunks(chunk_size) {
        // **Whole chunks only**, which is llama.cpp's rule. A trailing fragment
        // gives its scored tokens far less context than a full chunk does, and
        // including one took 29.25 to 33.65 on the same corpus — a 15% error
        // from a single short chunk out of four.
        if chunk.len() < chunk_size {
            break;
        }
        let mut cache = KvCache::with_type(
            config.n_layer as usize,
            config.n_head_kv as usize,
            config.head_dim as usize,
            kv_type,
        );
        // Every position is still *evaluated* — the context has to be built —
        // but only the second half is scored.
        //
        // `+ 1` matches llama.cpp exactly: it scores `n_ctx - 1 - n_ctx/2`
        // tokens per chunk, which is 63 at a context of 128, not 64. An
        // off-by-one here is invisible in the output and shifts the number.
        let first_scored = chunk.len() / 2 + 1;
        let mut logits = runner.forward_cached(weights, &mut cache, &chunk[..1], 0)?;
        for i in 1..chunk.len() {
            if logits.len() < vocab {
                return Err(format!("logits too small: {} < {vocab}", logits.len()).into());
            }
            if i >= first_scored {
                let row = &logits[logits.len() - vocab..];
                total_nll += neg_log_prob(row, chunk[i] as usize);
                counted += 1;
            }
            // The last position predicts nothing further, so do not pay for it.
            if i + 1 < chunk.len() {
                logits = runner.forward_cached(weights, &mut cache, &chunk[i..i + 1], i)?;
            }
        }
        chunks += 1;
        let ppl = (total_nll / counted as f64).exp();
        bigtea_arch::info!(
            "chunk {chunks:>4}   {counted:>7} tokens   ppl {ppl:.4}   ({:.1}s)",
            start.elapsed().as_secs_f64()
        );
    }

    if counted == 0 {
        return Err(format!(
            "no chunk reached 2 tokens: the corpus is {} tokens and --ppl-chunk is {chunk_size}",
            tokens.len()
        )
        .into());
    }
    let ppl = (total_nll / counted as f64).exp();
    println!();
    bigtea_arch::info!(
        "perplexity {ppl:.4} over {counted} tokens in {chunks} chunks of {chunk_size}"
    );
    bigtea_arch::info!(
        "           mean NLL {:.4} nats/token",
        total_nll / counted as f64
    );
    bigtea_arch::info!("total      {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Bytes of `ggml` arena a dense forward pass over `n` tokens needs.
///
/// The dominant term is attention: `n * n` scores per head, held twice (the
/// scores and their softmax), in `f32`. Everything else — activations, Q/K/V,
/// the FFN intermediates, the logits — is linear in `n` and is covered by the
/// second term plus generous slack.
///
/// Deliberately generous. Under-estimating does not return an error: `ggml`
/// calls `GGML_ASSERT` and the process dies, so the cost of being wrong is
/// asymmetric and the slack is cheap.
/// Print every hyper-parameter the forward pass actually reads, at `-v`.
///
/// Deliberately the *derived* values, not the raw metadata keys: `attn_scale`
/// and the per-layer RoPE bases are what the graph uses, and a key that was
/// present but read under the wrong name looks identical to one that was
/// absent until you print the result.
fn print_hparams(c: &Qwen3Config) {
    if !bigtea_arch::log::enabled(2) {
        return;
    }
    bigtea_arch::detail!(
        "hparams    n_layer {} n_embd {} n_head {} n_head_kv {} head_dim {} n_ff {}",
        c.n_layer,
        c.n_embd,
        c.n_head,
        c.n_head_kv,
        c.head_dim,
        c.n_ff
    );
    bigtea_arch::detail!(
        "hparams    vocab {} rms_eps {:e} attn_scale {} (1/sqrt {}, prescale_q {}) ffn_act {:?}",
        c.vocab_size,
        c.rms_eps,
        c.attn_scale(),
        c.attn_scale_dim,
        c.prescale_q,
        c.ffn_act
    );
    bigtea_arch::detail!(
        "hparams    rope base {} scale {} type {} ({}) orig_ctx {}",
        c.rope_freq_base,
        c.rope_freq_scale,
        c.rope_type,
        if c.rope_type_is_known {
            "known"
        } else {
            "guessed"
        },
        c.rope_orig_ctx
    );
    if c.sliding_window > 0 {
        // The layer list rather than the pattern number: "pattern 6" is a
        // claim, "layers 0-4 windowed, 5 global" is checkable against
        // llama.cpp's own trace.
        let windowed: Vec<u32> = (0..c.n_layer.min(12))
            .filter(|&il| c.is_swa_layer(il))
            .collect();
        bigtea_arch::detail!(
            "hparams    swa window {} pattern {} rope_swa {} first-12 windowed {:?}",
            c.sliding_window,
            c.swa_pattern,
            c.rope_freq_base_swa,
            windowed
        );
    }
    if c.attn_logit_softcap > 0.0 || c.final_logit_softcap > 0.0 {
        bigtea_arch::detail!(
            "hparams    softcap attn {} final {}",
            c.attn_logit_softcap,
            c.final_logit_softcap
        );
    }
    if c.is_moe() {
        bigtea_arch::detail!(
            "hparams    experts {} used {} n_ff_expert {}",
            c.n_expert,
            c.n_expert_used,
            c.n_ff_expert
        );
    }
    bigtea_arch::detail!(
        "hparams    qk_norm {} post_norms {} scale_embd {} attn_bias {} fused_qkv {}",
        c.qk_norm,
        c.post_norms,
        c.scale_embeddings,
        c.attn_bias,
        c.fused_qkv
    );
}

fn dense_arena_bytes(config: &Qwen3Config, n: i64) -> usize {
    let n = n.max(1) as u64;
    let layers = config.n_layer.max(1) as u64;
    // **Per layer**, and that is the whole point. One graph spans every block in
    // a single context, and `ggml` frees nothing inside a context, so all 36
    // layers' intermediates are alive at once. Sizing this for one layer is what
    // made a 651-token prompt abort.
    let per_layer = {
        // Attention scores and their softmax: n x n per head, twice.
        let scores = n * n * config.n_head as u64 * 4 * 2;
        // Activations, Q/K/V, the FFN intermediates: roughly a dozen tensors of
        // n_embd x n, plus the wider FFN ones.
        let activations = n * config.n_embd as u64 * 4 * 12 + n * config.n_ff as u64 * 4 * 3;
        // 25% over the counted tensors. The count is a reading of the graph and
        // the graph changes; being 0.2% short still aborts, and being 25% over
        // only refuses slightly sooner.
        (scores + activations) * 5 / 4
    };
    // The logits are one row now, not `n` of them — see `build_graph`.
    let head = config.vocab_size as u64 * 4 * 2;
    // `ggml_graph_compute_with_ctx` allocates the graph struct **and its
    // per-thread work buffer** out of this same arena, so the tensor data is
    // not the whole requirement. Sizing for the data alone left it 0.1% short,
    // which is still an abort.
    let data = per_layer * layers + head;
    (data + data / 8 + (512 << 20)) as usize
}

/// The longest prompt this machine can prefill in one graph, for this model.
///
/// The dense path builds one graph over every layer, so its arena grows with
/// the sequence and there is a length past which it does not fit. Saying so is
/// the difference between a clear refusal and `GGML_ASSERT` killing the process
/// with no message this code can catch.
fn dense_max_tokens(config: &Qwen3Config, budget: u64) -> i64 {
    let mut lo = 1i64;
    let mut hi = 32_768i64;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if dense_arena_bytes(config, mid) as u64 <= budget {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// A bash completion script, generated from the parser's own flag list.
///
/// Generated rather than written out, because a hand-maintained completion
/// script is a second list of flags that drifts from the first — this file has
/// already shipped a **flag count measured from the help text** that was 25
/// short of what the parser accepted, for eight commits. Anything that claims
/// to enumerate the flags has to derive them.
fn completion_bash() -> ExitCode {
    println!("# bash completion for bigtea-run -- generated, do not edit");
    println!("# install: bigtea-run --completion-bash > /etc/bash_completion.d/bigtea-run");
    println!("_bigtea_run() {{");
    println!("  local cur=\"${{COMP_WORDS[COMP_CWORD]}}\"");
    println!("  if [[ \"$cur\" == -* ]]; then");
    println!("    COMPREPLY=($(compgen -W \"{COMPLETION_FLAGS}\" -- \"$cur\"))");
    println!("  else");
    // Model paths are the only positional, and they are always .gguf.
    println!("    COMPREPLY=($(compgen -f -X '!*.gguf' -- \"$cur\") $(compgen -d -- \"$cur\"))");
    println!("  fi");
    println!("}}");
    println!("complete -o filenames -F _bigtea_run bigtea-run");
    ExitCode::SUCCESS
}

// Every flag the parser accepts, generated by `build.rs` from this file's own
// source. See `generate_flag_list` there for why it is derived rather than
// written down: a hand-kept list drifted in both directions within an hour of
// being written.
include!(concat!(env!("OUT_DIR"), "/flags.rs"));

/// The fill-in-the-middle control tokens in this vocabulary.
///
/// Read from the vocabulary's own text rather than from metadata keys, because
/// containers disagree about which keys they set (`tokenizer.ggml.fim_pre_id`
/// is common but far from universal) while the token *text* is stable across
/// every FIM model shipped so far. A model with no such tokens returns an empty
/// list, and `--infill` then says `0` rather than pretending.
///
/// Suppressing these matters more than it looks: a FIM model that emits
/// `<|fim_prefix|>` halfway through the span it is filling does not produce bad
/// prose, it produces a **corrupted file**, because the caller splices the
/// completion back between two halves that now contain a stray control token.
fn infill_tokens(tokenizer: &Tokenizer) -> Vec<u32> {
    const MARKERS: &[&str] = &[
        "fim_prefix",
        "fim_middle",
        "fim_suffix",
        "fim_pad",
        "fim_rep",
        "fim_sep",
        "fim_pre",
        "fim_suf",
        "fim_mid",
        "PRE",
        "SUF",
        "MID",
        "EOT",
    ];
    (0..tokenizer.vocab_size() as u32)
        .filter(|&id| {
            let Some(t) = tokenizer.token_text(id) else {
                return false;
            };
            // Control tokens only: a vocabulary entry that merely contains the
            // letters "PRE" is an ordinary word, and suppressing it would quietly
            // remove real vocabulary from every infill completion.
            let bracketed = (t.starts_with("<|") && t.ends_with("|>"))
                || (t.starts_with('<') && t.ends_with('>'));
            bracketed && MARKERS.iter().any(|m| t.contains(m))
        })
        .collect()
}

/// The full option list. One place, so `--help`, `-h` and a bare
/// invocation cannot drift apart.
fn usage() -> ExitCode {
    eprintln!("usage: bigtea-run <model.gguf> \"prompt\" [options]");
    eprintln!();
    eprintln!("  -n N                tokens to generate");
    eprintln!("  -f FILE             read the prompt from a file");
    eprintln!("  -b N                prefill block size");
    eprintln!("  --cache GIB         expert cache budget");
    eprintln!("  --temp T            0 = greedy (default)");
    eprintln!("  --top-k K           0 = off");
    eprintln!("  --top-p P           1.0 = off");
    eprintln!("  --min-p P           0.0 = off");
    eprintln!("  --repeat-penalty R  1.0 = off");
    eprintln!("  --frequency-penalty F  subtract F x count. 0 = off");
    eprintln!("  --presence-penalty P   subtract P if used at all. 0 = off");
    eprintln!("  --repeat-last-n N   penalty window (default 64)");
    eprintln!("  --typical P         locally typical sampling. 1.0 = off");
    eprintln!("  --top-nsigma N      keep logits within N sigma of the max. 0 = off");
    eprintln!("  --dynatemp-range R  entropy-driven temperature spread. 0 = off");
    eprintln!("  --dynatemp-exp E    how sharply it reacts (default 1.0)");
    eprintln!("  --xtc-probability P exclude top choices, chance per token. 0 = off");
    eprintln!("  --xtc-threshold T   XTC only considers tokens above this (default 0.1)");
    eprintln!("  --mirostat N        0 off, 1 v1, 2 v2 -- targets a surprise, not a mass");
    eprintln!("  --mirostat-ent TAU  target surprise in bits (default 5.0)");
    eprintln!("  --mirostat-lr ETA   mirostat learning rate (default 0.1)");
    eprintln!("  --logit-bias ID+B   nudge one token, repeatable (e.g. 42-100)");
    eprintln!("  --ignore-eos        never stop at end-of-sequence");
    eprintln!("  --dry-multiplier M  DRY repetition penalty. 0 = off");
    eprintln!("  --dry-base B        DRY growth per extra repeated token (1.75)");
    eprintln!("  --dry-allowed-length N  repeats shorter than this are free (2)");
    eprintln!("  --dry-penalty-last-n N  how far DRY looks back. 0 = all");
    eprintln!("  --dry-sequence-breaker S  a match may not cross this, repeatable");
    eprintln!("  --samplers SPEC     chain order, e.g. \"top_k;temperature;top_p\"");
    eprintln!("  -ctk, -ctv TYPE     KV cache storage: f16 (default) or q8_0");
    eprintln!("  --no-direct-io      read through the page cache (also --no-mmap)");
    eprintln!("  --direct-io         bypass the page cache (default)");
    eprintln!("  --override-kv K=T:V override one GGUF metadata entry");
    eprintln!("  --mlock             pin resident weights so the OS cannot page them out");
    eprintln!("  --chat-template N   force a chat template (chatml, llama3, gemma, ...)");
    eprintln!("  --prompt-cache F    reuse a saved KV cache for a repeated prefix");
    eprintln!("  --prompt-cache-all  also cache what was generated, not just the prompt");
    eprintln!("  --prompt-cache-ro   read the cache but never write it");
    eprintln!("  --grammar GBNF      constrain output to a GBNF grammar");
    eprintln!("  --grammar-file F    ...read from a file");
    eprintln!("  -j, --json-schema S constrain output to a JSON schema");
    eprintln!("  --json-schema-file F  ...read from a file");
    eprintln!("  -i, --interactive   keep the session open and take turns");
    eprintln!("  -cnv, --conversation  interactive, with the chat template per turn");
    eprintln!("  -st, --single-turn  one exchange, then exit");
    eprintln!("  --multiline-input   a trailing backslash continues the line");
    eprintln!("  --in-prefix S       wrap user input (non-conversation mode)");
    eprintln!("  --in-suffix S       ...and after it");
    eprintln!("  --in-prefix-bos     prepend BOS to each user turn");
    eprintln!("  -sys, --system-prompt S   system message (implies a template)");
    eprintln!("  --system-prompt-file F    ...read from a file");
    eprintln!("  -co, --color        colour the generated text");
    eprintln!("  --simple-io         no ANSI, for pipes and logs");
    eprintln!("  --no-display-prompt do not echo the prompt back");
    eprintln!("  -sp, --special      show control tokens instead of hiding them");
    eprintln!("  --print-token-count report prompt and generated counts");
    eprintln!("  --verbose-prompt    print the tokenised prompt and its ids");
    eprintln!("  -e, --escape        process backslash escapes in -p (default on)");
    eprintln!("  --no-escape         take -p literally");
    eprintln!("  -r, --reverse-prompt S    llama.cpp's name for --stop");
    eprintln!("  --rope-freq-base B  override the container's RoPE base");
    eprintln!("  --rope-freq-scale S linear RoPE scaling (1.0 = off)");
    eprintln!("  --rope-scale N      context multiplier (= 1 / freq-scale)");
    eprintln!("  --rope-scaling T    none | linear | yarn");
    eprintln!("  --yarn-ext-factor F   YaRN mix (0 = pure linear)");
    eprintln!("  --yarn-attn-factor F  YaRN magnitude correction");
    eprintln!("  --yarn-beta-fast F    YaRN high-frequency cutoff");
    eprintln!("  --yarn-beta-slow F    YaRN low-frequency cutoff");
    eprintln!("  --yarn-orig-ctx N     context the model was trained at");
    eprintln!("  --log-disable       silence the status lines");
    eprintln!("  --log-file F        write status to a file instead of stderr");
    eprintln!("  --log-timestamps    prefix each status line with elapsed time");
    eprintln!("  --log-prefix        prefix each status line with its level");
    eprintln!("  -v, --verbose       verbosity 2");
    eprintln!("  --verbosity N       0 quiet, 1 normal, 2+ verbose");
    eprintln!("  --no-perf           omit the timing summary");
    eprintln!("  --version           print the version and exit");
    eprintln!("  --perplexity        score a corpus instead of generating");
    eprintln!("  --ppl-chunk N       perplexity chunk size (default 512)");
    eprintln!("  --seed S            reproducible sampling");
    eprintln!("  --llamacpp-defaults temp 0.8, top-k 40, top-p 0.95, min-p 0.05, repeat 1.1");
    eprintln!("  --chat              apply the model's chat template to the prompt");
    eprintln!("  -t, --threads N     threads for generation (default: measured -- generation");
    eprintln!("                      is bandwidth-bound and all cores is 1.7x SLOWER)");
    eprintln!("  -tb, --threads-batch N  threads for prefill (default: all cores)");
    eprintln!("  -c, --ctx-size N    cap the context; refuses past it rather than aborting");
    eprintln!("  --stop TEXT         stop when this appears (repeatable)");
    eprintln!("  --force             run an unverified architecture anyway");
    eprintln!("  --no-repack         keep resident weights in their stored layout");
    eprintln!("                      (repacking is on by default: 1.35x prefill)");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    // `-m model.gguf` puts a flag where the positional path used to be, so the
    // first argument is only treated as the path when it is not one. Without
    // this, `bigtea-run -m x.gguf -p "hi"` tries to open a file called `-m`.
    let leads_with_flag = std::env::args()
        .nth(1)
        .map(|a| a.starts_with('-') && a != "-")
        .unwrap_or(false);
    // Before the positional model path is taken. `--version` as the first
    // argument would otherwise *be* the path, and the runner would report that
    // it cannot open a file called `--version`.
    if let Some(first) = std::env::args().nth(1) {
        if first == "--usage" {
            // llama.cpp's alias for --help. Falls through to the usage block
            // by leaving `path` unset.
            eprintln!("usage: bigtea-run <model.gguf> \"prompt\" [options]");
            eprintln!("  run with no arguments for the full option list");
            return ExitCode::from(2);
        }
        if first == "--version" {
            println!("bigtea-run {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        if first == "--help" || first == "-h" {
            // Falls into the usage block below by leaving `path` unset, so
            // there is one list rather than two that drift apart.
            return usage();
        }
        if first == "--completion-bash" {
            return completion_bash();
        }
    }
    let path_positional = if leads_with_flag { None } else { args.next() };
    if path_positional.is_none() && !leads_with_flag {
        return usage();
    }
    let mut prompt = String::new();
    let mut n_predict = 8usize;
    // A block reads nearly the whole expert set whatever its size, so larger
    // blocks amortise that over more tokens: at 4395 tokens, 512 gives 30.5
    // tok/s and 4096 gives 43.6. The limit is memory — every arena in the
    // forward pass scales with the block — so 2048 is the default and -b
    // raises it when there is RAM to spare.
    let mut prefill_block = 2048usize;
    let mut cache_budget: Option<u64> = None;
    // Greedy by default, so existing behaviour is unchanged until asked.
    let mut sampler = SamplerConfig::default();
    let mut chat = false;
    let mut threads: Option<usize> = None;
    let mut threads_batch: Option<usize> = None;
    let mut perplexity: Option<usize> = None;
    let mut ui = Ui {
        // llama.cpp echoes the prompt by default and processes backslash
        // escapes in -p by default; both match here.
        display_prompt: true,
        ..Ui::default()
    };
    let mut escape = true;
    let mut rope = RopeOverrides::default();
    let mut logcfg = bigtea_arch::log::LogConfig::default();
    // Held as text until a tokenizer exists to turn them into ids.
    let mut dry_breakers: Vec<String> = Vec::new();
    let mut kv_type = bigtea_arch::KvType::F16;
    let mut overrides: Vec<(String, bigtea_gguf::Value)> = Vec::new();
    let mut mlock = false;
    let mut prio: Option<u32> = None;
    let mut warmup = false;
    let mut infill = false;
    let mut grammar_triggers: Vec<String> = Vec::new();
    let mut chat_template: Option<String> = None;
    let mut model_flag: Option<String> = None;
    let mut grammar_src: Option<String> = None;
    let mut schema_src: Option<String> = None;
    let mut prompt_cache: Option<String> = None;
    let mut prompt_cache_all = false;
    let mut prompt_cache_ro = false;
    let mut show_perf = true;
    let mut system_prompt: Option<String> = None;
    let mut ctx_size: Option<usize> = None;
    let mut stop: Vec<String> = Vec::new();
    let mut force = false;
    // With a leading flag nothing was consumed as the path, so every argument
    // is a flag to parse.
    let rest: Vec<String> = if leads_with_flag {
        std::env::args().skip(1).collect()
    } else {
        args.collect()
    };
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "-n" | "--n-predict" | "--predict" => {
                n_predict = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(8);
                i += 2;
            }
            "--cache" => {
                cache_budget = rest
                    .get(i + 1)
                    .and_then(|v| v.parse::<f64>().ok())
                    .map(|g| (g * (1u64 << 30) as f64) as u64);
                i += 2;
            }
            "--temp" | "--temperature" => {
                sampler.temperature = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.8);
                i += 2;
            }
            "--top-k" => {
                sampler.top_k = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(40);
                i += 2;
            }
            "--top-p" => {
                sampler.top_p = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.95);
                i += 2;
            }
            "--min-p" => {
                sampler.min_p = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.05);
                i += 2;
            }
            "--repeat-penalty" => {
                sampler.repeat_penalty =
                    rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1.1);
                i += 2;
            }
            // llama.cpp's spellings, and OpenAI's semantics: frequency scales
            // with how often a token was used, presence is flat.
            "--frequency-penalty" => {
                sampler.frequency_penalty =
                    rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                i += 2;
            }
            "--presence-penalty" => {
                sampler.presence_penalty =
                    rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                i += 2;
            }
            // llama.cpp's sampler flags, its spellings, its defaults.
            "--typical" | "--typical-p" => {
                sampler.typical_p = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1.0);
                i += 2;
            }
            "--top-nsigma" | "--top-n-sigma" => {
                sampler.top_n_sigma = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                i += 2;
            }
            "--dynatemp-range" => {
                sampler.dynatemp_range =
                    rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                i += 2;
            }
            "--dynatemp-exp" => {
                sampler.dynatemp_exponent =
                    rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1.0);
                i += 2;
            }
            "--xtc-probability" => {
                sampler.xtc_probability =
                    rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                i += 2;
            }
            "--xtc-threshold" => {
                sampler.xtc_threshold = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.1);
                i += 2;
            }
            "--mirostat" => {
                sampler.mirostat = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0);
                i += 2;
            }
            // Adaptive-p: aim for a token of roughly this probability, with the
            // target moving as it observes what it actually picked. Like
            // mirostat it replaces the truncate-then-temperature tail rather
            // than joining it, so the two cannot both be on.
            "--adaptive-target" | "--adaptive-p" => {
                sampler.adaptive_p = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(-1.0);
                i += 2;
            }
            "--adaptive-decay" => {
                sampler.adaptive_decay =
                    rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.95);
                i += 2;
            }
            // Fill-in-the-middle. The ids are resolved from the vocabulary
            // after the tokenizer loads -- see `infill_tokens`.
            "--infill" => {
                infill = true;
                i += 1;
            }
            "--mirostat-ent" => {
                sampler.mirostat_tau = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(5.0);
                i += 2;
            }
            "--mirostat-lr" => {
                sampler.mirostat_eta = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.1);
                i += 2;
            }
            "--ignore-eos" => {
                sampler.ignore_eos = true;
                i += 1;
            }
            // `ID+BIAS` or `ID-BIAS`, repeatable, which is llama.cpp's spelling.
            "--logit-bias" => {
                if let Some(spec) = rest.get(i + 1) {
                    if let Some(cut) = spec.find(['+', '-']) {
                        if let (Ok(id), Ok(bias)) =
                            (spec[..cut].parse::<u32>(), spec[cut..].parse::<f32>())
                        {
                            sampler.logit_bias.push((id, bias));
                        }
                    }
                }
                i += 2;
            }
            "--repeat-last-n" => {
                sampler.repeat_last_n = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(64);
                i += 2;
            }
            "--seed" => {
                sampler.seed = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0);
                i += 2;
            }
            // Frame the prompt as a chat turn, the way the model was trained.
            // Off by default: a raw prompt is still the right thing for a base
            // model and for diagnosing the forward pass.
            "--chat" => {
                chat = true;
                i += 1;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            // On by default -- it is 1.35x faster AND agrees with llama.cpp.
            // This turns it off, for measuring the difference.
            // The default, spelled explicitly. llama.cpp has both, and a
            // script passing --repack should not be told it is unknown.
            "--repack" => {
                i += 1;
            }
            "--no-repack" => {
                std::env::set_var("BIGTEA_NO_REPACK", "1");
                i += 1;
            }
            // llama.cpp spells these -t and -c; matching its names matters more
            // than inventing better ones, because muscle memory is the whole
            // reason an OpenAI-shaped API and a familiar CLI are worth having.
            "-t" | "--threads" => {
                threads = rest
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .filter(|&t: &usize| t > 0);
                i += 2;
            }
            // Generation and prefill want opposite thread counts — one is
            // bandwidth-bound, the other compute-bound — so llama.cpp carries
            // two flags and so do we, with its spelling.
            // --- constrained decoding ------------------------------------
            "--grammar" => {
                grammar_src = rest.get(i + 1).cloned();
                i += 2;
            }
            // Lazy grammar: hold the constraint back until the model has
            // written one of these, then apply it from that point on.
            //
            // This is what makes a grammar usable for tool calling. A model
            // asked to "answer normally, or emit a JSON call" cannot do the
            // first half under a JSON grammar -- the grammar forbids prose from
            // the very first token, so the model never gets to choose. The
            // trigger lets it choose, and constrains only what follows.
            //
            // Substrings, not regexes, and the help says so. llama.cpp's
            // `--grammar-lazy-patterns` takes regexes; a half-implemented regex
            // engine that silently mismatches would arm the grammar at the
            // wrong moment, which is worse than not having the flag.
            "--grammar-lazy" | "--grammar-trigger" => {
                if let Some(t) = rest.get(i + 1) {
                    grammar_triggers.push(t.clone());
                }
                i += 2;
            }
            "--grammar-file" => {
                match rest.get(i + 1).map(std::fs::read_to_string) {
                    Some(Ok(text)) => grammar_src = Some(text),
                    _ => {
                        eprintln!(
                            "bigtea-run: --grammar-file: cannot read {:?}",
                            rest.get(i + 1).cloned().unwrap_or_default()
                        );
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            "-j" | "--json-schema" => {
                schema_src = rest.get(i + 1).cloned();
                i += 2;
            }
            "--json-schema-file" => {
                match rest.get(i + 1).map(std::fs::read_to_string) {
                    Some(Ok(text)) => schema_src = Some(text),
                    _ => {
                        eprintln!(
                            "bigtea-run: --json-schema-file: cannot read {:?}",
                            rest.get(i + 1).cloned().unwrap_or_default()
                        );
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            // Save and reuse the KV cache for a prompt prefix, so a repeated
            // prompt does not pay prefill twice.
            "--prompt-cache" => {
                prompt_cache = rest.get(i + 1).cloned();
                i += 2;
            }
            "--prompt-cache-all" => {
                prompt_cache_all = true;
                i += 1;
            }
            "--prompt-cache-ro" => {
                prompt_cache_ro = true;
                i += 1;
            }
            // Force a chat format. Two cases make this necessary rather than a
            // curiosity: a container with no template at all, and one whose
            // template this build does not recognise. Both otherwise fall back
            // to a plain framing the model was never trained on, and answer
            // fluently and wrongly.
            "--chat-template" => {
                chat_template = rest.get(i + 1).cloned();
                i += 2;
            }
            // The same thing from a file, because a real Jinja template is
            // several hundred characters of quoting that no shell survives.
            // Takes the file's *contents* as the template name/body, matching
            // llama.cpp; a name that is not recognised is still refused by
            // `force_chat_template` rather than silently ignored.
            "--chat-template-file" => {
                let Some(file) = rest.get(i + 1) else {
                    eprintln!("bigtea-run: --chat-template-file needs a file path");
                    return ExitCode::from(2);
                };
                match std::fs::read_to_string(file) {
                    Ok(text) => chat_template = Some(text.trim().to_string()),
                    Err(e) => {
                        eprintln!("bigtea-run: --chat-template-file: cannot read {file}: {e}");
                        return ExitCode::FAILURE;
                    }
                }
                i += 2;
            }
            // Pin the resident set in physical memory. Bigtea decides what
            // stays in RAM; that decision is undone if the OS pages it out.
            "--mlock" => {
                mlock = true;
                i += 1;
            }
            // Scheduling priority. Applied immediately rather than stored,
            // because the model load is itself minutes of disk work that
            // benefits. `--prio-batch` is llama.cpp's separate knob for the
            // prefill threadpool; there is one process here, so the higher of
            // the two wins and the runner says which it took -- rather than
            // accepting the second flag and quietly dropping it.
            "--prio" | "--prio-batch" => {
                let Some(level) = rest.get(i + 1).and_then(|v| v.parse::<u32>().ok()) else {
                    eprintln!("bigtea-run: {}: expected 0-3", rest[i]);
                    return ExitCode::from(2);
                };
                prio = Some(prio.map_or(level, |p: u32| p.max(level)));
                i += 2;
            }
            // A forward pass before the user's, so the first token they time is
            // not also paying for the page cache, the repack and the thread
            // ladder. **Off by default, unlike llama.cpp**, and that is a
            // deliberate difference: warming a runner whose job is streaming
            // from disk reads gigabytes, and the cold cost is the number this
            // project exists to report honestly. `--no-warmup` is the default
            // and is accepted so a llama.cpp command line runs unchanged.
            "--warmup" => {
                warmup = true;
                i += 1;
            }
            "--no-warmup" => {
                warmup = false;
                i += 1;
            }
            // --- I/O mode and metadata overrides --------------------------
            "--direct-io" => {
                std::env::set_var("BIGTEA_IO", "direct");
                i += 1;
            }
            // Also llama.cpp's --no-mmap: it means "do not let the OS page
            // cache hold the weights", which is what direct I/O already does
            // here, so the two spellings land on the same switch.
            "--no-direct-io" | "--no-mmap" => {
                std::env::set_var("BIGTEA_IO", "buffered");
                i += 1;
            }
            "--override-kv" => {
                match rest.get(i + 1).and_then(|spec| parse_override(spec)) {
                    Some(kv) => overrides.push(kv),
                    None => {
                        eprintln!(
                            "bigtea-run: --override-kv: expected key=type:value, got {:?}",
                            rest.get(i + 1).cloned().unwrap_or_default()
                        );
                        eprintln!("  types: int, float, bool, str");
                        eprintln!("  e.g. --override-kv qwen3.rope.freq_base=float:1000000");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            // --- KV cache storage type ------------------------------------
            // One type for both halves: ggml's banded attention asserts
            // k->type == v->type, so accepting different ones would work until
            // that path was reached. Both spellings are taken and the last
            // wins, which is what a user passing `-ctk q8_0 -ctv q8_0` means.
            "--cache-type-k" | "-ctk" | "--cache-type-v" | "-ctv" => {
                match rest.get(i + 1).and_then(|v| bigtea_arch::KvType::parse(v)) {
                    Some(t) => kv_type = t,
                    None => {
                        eprintln!(
                            "bigtea-run: {}: unknown cache type {:?}",
                            rest[i],
                            rest.get(i + 1).cloned().unwrap_or_default()
                        );
                        eprintln!("  known: f16, q8_0");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            // The chain order itself. Refused wholesale on an unknown name
            // rather than dropping that stage: a typo would otherwise remove a
            // filter the user is relying on, silently.
            "--samplers" | "--sampler-seq" | "--sampling-seq" => {
                if let Some(spec) = rest.get(i + 1) {
                    let mut chain = Vec::new();
                    let mut bad: Option<String> = None;
                    for name in spec.split([';', ',']).filter(|n| !n.trim().is_empty()) {
                        match bigtea_arch::SamplerStage::parse(name) {
                            Some(stage) => chain.push(stage),
                            None => {
                                bad = Some(name.trim().to_string());
                                break;
                            }
                        }
                    }
                    match bad {
                        Some(name) => {
                            // Built as separate lines: a `\` continuation in a
                            // Rust string keeps the source indentation and
                            // prints a ragged message, which is how the SSE
                            // headers went out malformed earlier.
                            eprintln!("bigtea-run: --samplers: unknown stage {name:?}");
                            eprintln!(
                                "  known stages: top_k, typ_p, top_p, min_p, xtc, temperature"
                            );
                            eprintln!(
                                "  penalties, dry and top_n_sigma act on logits and always run first"
                            );
                            return ExitCode::from(2);
                        }
                        None if chain.is_empty() => {
                            eprintln!("bigtea-run: --samplers: empty chain");
                            return ExitCode::from(2);
                        }
                        None => sampler.chain = chain,
                    }
                }
                i += 2;
            }
            // --- DRY: penalise continuing a repeat, not reusing a word -----
            "--dry-multiplier" => {
                sampler.dry_multiplier =
                    rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                i += 2;
            }
            "--dry-base" => {
                sampler.dry_base = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1.75);
                i += 2;
            }
            "--dry-allowed-length" => {
                sampler.dry_allowed_length =
                    rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(2);
                i += 2;
            }
            "--dry-penalty-last-n" => {
                sampler.dry_penalty_last_n =
                    rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--dry-sequence-breaker" => {
                if let Some(v) = rest.get(i + 1) {
                    dry_breakers.push(v.clone());
                }
                i += 2;
            }
            // --- logging: status is diagnostics, the text is output --------
            "--log-disable" => {
                logcfg.verbosity = 0;
                i += 1;
            }
            "--log-file" => {
                logcfg.file = rest.get(i + 1).cloned();
                i += 2;
            }
            "--log-timestamps" => {
                logcfg.timestamps = true;
                i += 1;
            }
            "--no-log-timestamps" => {
                logcfg.timestamps = false;
                i += 1;
            }
            // Colour is not decoration here: status goes to stderr and the
            // generated text to stdout, and in a terminal the two are
            // interleaved. Dimming the status is what makes the answer
            // findable. Suppressed for `--log-file` -- see `LogConfig::colors`.
            "--log-colors" => {
                logcfg.colors = true;
                i += 1;
            }
            "--no-log-colors" => {
                logcfg.colors = false;
                i += 1;
            }
            "--log-prefix" => {
                logcfg.prefix = true;
                i += 1;
            }
            "--no-log-prefix" => {
                logcfg.prefix = false;
                i += 1;
            }
            "-v" | "--verbose" | "--log-verbose" => {
                logcfg.verbosity = 2;
                i += 1;
            }
            "--verbosity" | "--log-verbosity" => {
                logcfg.verbosity = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1);
                i += 2;
            }
            "--perf" => {
                show_perf = true;
                i += 1;
            }
            "--no-perf" => {
                show_perf = false;
                i += 1;
            }
            "--version" => {
                println!("bigtea-run {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            // --- RoPE, for a container whose metadata is wrong or absent ---
            "--rope-freq-base" => {
                rope.freq_base = rest.get(i + 1).and_then(|v| v.parse().ok());
                i += 2;
            }
            "--rope-freq-scale" => {
                rope.freq_scale = rest.get(i + 1).and_then(|v| v.parse().ok());
                i += 2;
            }
            // llama.cpp's --rope-scale is the context multiplier, i.e. the
            // reciprocal of the frequency scale. Storing it unconverted would
            // invert the meaning of every long-context flag.
            "--rope-scale" => {
                rope.freq_scale = rest
                    .get(i + 1)
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|f| *f > 0.0)
                    .map(|f| 1.0 / f);
                i += 2;
            }
            "--rope-scaling" => {
                rope.scaling = rest.get(i + 1).cloned();
                i += 2;
            }
            "--yarn-ext-factor" => {
                rope.ext_factor = rest.get(i + 1).and_then(|v| v.parse().ok());
                i += 2;
            }
            "--yarn-attn-factor" => {
                rope.attn_factor = rest.get(i + 1).and_then(|v| v.parse().ok());
                i += 2;
            }
            "--yarn-beta-fast" => {
                rope.beta_fast = rest.get(i + 1).and_then(|v| v.parse().ok());
                i += 2;
            }
            "--yarn-beta-slow" => {
                rope.beta_slow = rest.get(i + 1).and_then(|v| v.parse().ok());
                i += 2;
            }
            "--yarn-orig-ctx" => {
                rope.orig_ctx = rest.get(i + 1).and_then(|v| v.parse().ok());
                i += 2;
            }
            // --- interaction, llama.cpp's spellings ---------------------
            "-i" | "--interactive" => {
                ui.interactive = true;
                i += 1;
            }
            // llama.cpp's: interactive, but the user speaks first. Distinct
            // from -i, which generates from the prompt and then waits.
            "-if" | "--interactive-first" => {
                ui.interactive = true;
                ui.interactive_first = true;
                i += 1;
            }
            "-cnv" | "--conversation" => {
                ui.interactive = true;
                ui.conversation = true;
                i += 1;
            }
            "--no-conversation" => {
                ui.conversation = false;
                i += 1;
            }
            "-st" | "--single-turn" => {
                ui.interactive = true;
                ui.single_turn = true;
                i += 1;
            }
            "--multiline-input" => {
                ui.multiline = true;
                i += 1;
            }
            "--in-prefix" => {
                ui.in_prefix = rest.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--in-suffix" => {
                ui.in_suffix = rest.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--in-prefix-bos" => {
                ui.in_prefix_bos = true;
                i += 1;
            }
            "--color" | "-co" => {
                ui.color = true;
                i += 1;
            }
            // A pipe is not a terminal: colour codes in a redirected file are
            // noise, and llama.cpp's --simple-io means exactly "no ANSI".
            "--simple-io" => {
                ui.color = false;
                i += 1;
            }
            "--display-prompt" => {
                ui.display_prompt = true;
                i += 1;
            }
            "--no-display-prompt" => {
                ui.display_prompt = false;
                i += 1;
            }
            "--special" | "-sp" => {
                ui.special = true;
                i += 1;
            }
            "--print-token-count" => {
                ui.print_token_count = true;
                i += 1;
            }
            "--verbose-prompt" => {
                ui.verbose_prompt = true;
                i += 1;
            }
            "-e" | "--escape" => {
                escape = true;
                i += 1;
            }
            "--no-escape" => {
                escape = false;
                i += 1;
            }
            "-sys" | "--system-prompt" => {
                system_prompt = rest.get(i + 1).cloned();
                i += 2;
            }
            "--system-prompt-file" => {
                if let Some(f) = rest.get(i + 1) {
                    system_prompt = std::fs::read_to_string(f).ok();
                }
                i += 2;
            }
            // llama.cpp's name for what we already spell --stop.
            "-r" | "--reverse-prompt" => {
                if let Some(v) = rest.get(i + 1) {
                    stop.push(v.clone());
                }
                i += 2;
            }
            // Quality, not speed: the one thing this project has never measured.
            "--perplexity" | "--ppl" => {
                perplexity = Some(perplexity.unwrap_or(512));
                i += 1;
            }
            "--ppl-chunk" => {
                perplexity = rest
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .filter(|&c: &usize| c >= 2);
                i += 2;
            }
            "-tb" | "--threads-batch" => {
                threads_batch = rest
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .filter(|&t: &usize| t > 0);
                i += 2;
            }
            "-c" | "--ctx-size" => {
                ctx_size = rest
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .filter(|&c: &usize| c > 0);
                i += 2;
            }
            "--stop" => {
                if let Some(v) = rest.get(i + 1) {
                    stop.push(v.clone());
                }
                i += 2;
            }
            // One flag for "sample the way llama.cpp does by default", so a
            // quality comparison is not silently comparing sampler settings.
            "--llamacpp-defaults" => {
                sampler = SamplerConfig::llamacpp_defaults();
                i += 1;
            }
            "-b" | "--batch-size" => {
                prefill_block = rest
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .filter(|&b: &usize| b > 0)
                    .unwrap_or(256);
                i += 2;
            }
            // A long-context prompt does not fit on a command line; Windows
            // caps it around 32k characters, well under the token counts that
            // make streaming interesting.
            // llama.cpp names the model and the prompt with flags; this
            // runner only ever took them positionally. Someone with the muscle
            // memory types `-m model.gguf -p "..."`, and matching the spelling
            // is the whole reason for copying a CLI.
            "-m" | "--model" => {
                if let Some(v) = rest.get(i + 1) {
                    model_flag = Some(v.clone());
                }
                i += 2;
            }
            "-p" | "--prompt" => {
                if let Some(v) = rest.get(i + 1) {
                    prompt = v.clone();
                }
                i += 2;
            }
            "-f" | "--file" => {
                let Some(file) = rest.get(i + 1) else {
                    eprintln!("bigtea-run: -f needs a file path");
                    return ExitCode::from(2);
                };
                match std::fs::read_to_string(file) {
                    Ok(text) => prompt = text,
                    Err(e) => {
                        eprintln!("bigtea-run: cannot read {file}: {e}");
                        return ExitCode::FAILURE;
                    }
                }
                i += 2;
            }
            // Bytes rather than text. Not the same flag as `-f` and not a
            // convenience: `read_to_string` *fails* on a file that is not valid
            // UTF-8, so a prompt captured from a binary source is unreachable
            // through `-f`. Decoded lossily here, which is what llama.cpp's
            // `--binary-file` does, so the invalid bytes become U+FFFD and the
            // tokenizer sees something well-formed instead of an error.
            "--binary-file" => {
                let Some(file) = rest.get(i + 1) else {
                    eprintln!("bigtea-run: --binary-file needs a file path");
                    return ExitCode::from(2);
                };
                match std::fs::read(file) {
                    Ok(bytes) => prompt = String::from_utf8_lossy(&bytes).into_owned(),
                    Err(e) => {
                        eprintln!("bigtea-run: --binary-file: cannot read {file}: {e}");
                        return ExitCode::FAILURE;
                    }
                }
                i += 2;
            }
            other => {
                if prompt.is_empty() {
                    prompt = other.to_string();
                }
                i += 1;
            }
        }
    }
    if prompt.is_empty() {
        prompt = "The capital of France is".into();
    }
    // llama.cpp processes backslash escapes in `-p` by default, so a prompt
    // written with a backslash-n is two lines rather than a question about a
    // backslash. `--no-escape` turns it off for prompts that contain literal
    // ones, such as a Windows path.
    if escape {
        prompt = unescape(&prompt);
    }
    let prompt = prompt;
    let Some(path) = model_flag.or(path_positional) else {
        eprintln!("bigtea-run: no model given. Pass it positionally or with -m.");
        return ExitCode::from(2);
    };

    bigtea_arch::log::configure(logcfg);
    // After the log is configured so the outcome is reported through it, and
    // before the model opens so the load itself runs at the asked-for priority.
    if let Some(level) = prio {
        match bigtea_io::lock::set_priority(level) {
            Ok(name) => bigtea_arch::info!("priority   {name}"),
            // Not fatal: a refused priority change leaves a process that still
            // runs correctly, just at the priority it already had. Saying so is
            // the point -- silently continuing is the failure mode this whole
            // audit exists to avoid.
            Err(e) => bigtea_arch::info!("priority   not changed: {e}"),
        }
    }
    match run(
        &path,
        &prompt,
        n_predict,
        prefill_block,
        cache_budget,
        sampler,
        chat,
        threads,
        threads_batch,
        perplexity,
        ui,
        system_prompt,
        rope,
        show_perf,
        dry_breakers,
        kv_type,
        overrides,
        mlock,
        chat_template,
        prompt_cache,
        prompt_cache_all,
        prompt_cache_ro,
        grammar_src,
        schema_src,
        ctx_size,
        stop,
        force,
        warmup,
        infill,
        grammar_triggers,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bigtea-run: {e}");
            ExitCode::FAILURE
        }
    }
}

/// MoE path: the dense weights stay resident, experts stream per token.
///
/// A model far larger than RAM runs here because only the always-read part is
/// held — for Qwen3-30B-A3B that is 0.93 GiB of a 17.28 GiB container.
// These are command-line options, not coupled state; a config struct here
// would add a layer without removing a decision.
#[allow(clippy::too_many_arguments)]
fn run_streaming(
    model: &Model,
    config: Qwen3Config,
    arch: &Qwen3Model,
    tokenizer: &Tokenizer,
    mut tokens: Vec<u32>,
    n_predict: usize,
    prefill_block: usize,
    cache_budget: Option<u64>,
    sampler_cfg: SamplerConfig,
    ctx_size: Option<usize>,
    stop: Vec<String>,
    perplexity: Option<usize>,
    ui: Ui,
    show_perf: bool,
    kv_type: bigtea_arch::KvType,
    mlock: bool,
    prompt_cache: Option<String>,
    prompt_cache_all: bool,
    prompt_cache_ro: bool,
    grammar: Option<bigtea_grammar::Grammar>,
    warmup: bool,
    infill: bool,
    grammar_triggers: Vec<String>,
    t0: std::time::Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    use bigtea_arch::StreamingRunner;

    // llama.cpp's `llm_load_print_meta`, and worth the lines: every wrong
    // answer this project has shipped came from a hyper-parameter that was
    // read wrongly or defaulted silently, and none of them were visible from
    // the outside. Three hours went into "gemma-2 diverges" before anyone
    // could see which scale it was actually using.
    print_hparams(&config);

    // Past this the implementation is wrong, not merely slow -- see
    // `correct_context_limit`. Refusing is the only honest option.
    let correct = config.correct_context_limit();
    if tokens.len() + n_predict > correct {
        return Err(format!(
            "prompt is {} tokens and -n is {n_predict}, past the {correct} this build can run              correctly for this model (its sliding-window attention is not implemented, so              beyond the window the local layers would attend too far -- silently)",
            tokens.len()
        )
        .into());
    }

    // A context cap the user asked for, enforced before any work rather than
    // discovered as an arena abort partway through.
    if let Some(limit) = ctx_size {
        if tokens.len() + n_predict > limit {
            return Err(format!(
                "prompt is {} tokens and -n is {n_predict}, which exceeds the -c limit of {limit}",
                tokens.len()
            )
            .into());
        }
    }

    // Size the expert cache from the RAM that is actually free, not a constant.
    //
    // A fixed 1 GiB held under 4% of this model's 18,432 expert slices, so
    // nearly every token went to disk — while ten gigabytes of memory sat
    // unused. The whole point of measuring residency is to spend what the
    // machine has. Headroom covers the OS, the resident weights, the KV cache
    // and the compute arenas; what remains is worth filling with experts.
    // Headroom is computed, not fixed. A flat 4 GiB was set when the attention
    // arena needed 1.3 GiB; fused attention cut that to ~100 MiB, and the extra
    // reserve then cost real speed — at 32 tokens, a 6 GiB cache gives 1.44
    // tok/s and 8 GiB gives 1.56. The two things that genuinely scale are the
    // KV cache, which grows with context, and the arenas, which grow with the
    // prefill block. Everything else is the OS.
    const BASE_HEADROOM: u64 = 2 * (1 << 30);
    // Two bytes per value: the KV cache is f16.
    let kv_per_position =
        (config.n_layer as u64) * (config.n_head_kv as u64) * (config.head_dim as u64) * 2 * 2;
    let kv_estimate = kv_per_position * (tokens.len() + n_predict) as u64;
    // Arenas scale with the block: activations, Q/K/V and the router, roughly
    // a dozen n_embd-by-block matrices, doubled by `arena_for`.
    let arena_estimate = (config.n_embd as u64) * (prefill_block as u64) * 4 * 24;
    let headroom = BASE_HEADROOM + kv_estimate + arena_estimate;

    let budget = match cache_budget {
        Some(bytes) => bytes,
        None => {
            let machine = bigtea_probe::Machine::probe(std::path::Path::new("."), false);
            machine
                .ram_available_bytes
                .map(|avail| avail.saturating_sub(headroom).max(1 << 30))
                .unwrap_or(1 << 30)
        }
    };
    let mut runner = StreamingRunner::new(model, config.clone(), budget as usize);
    bigtea_arch::info!(
        "cache      {:.2} GiB for experts (headroom {:.2} GiB: {:.2} kv + {:.2} arenas + 2.00 os)",
        budget as f64 / GIB,
        headroom as f64 / GIB,
        kv_estimate as f64 / GIB,
        arena_estimate as f64 / GIB
    );

    let ctx = Context::new_no_alloc(64 << 20)?;
    let mut weights = WeightSet::new();
    let load_start = std::time::Instant::now();
    let resident = runner.load_resident(&ctx, &mut weights)?;
    if mlock {
        let report = lock_resident(&weights);
        if report.ok() {
            // Says what is NOT covered, because the number is smaller than the
            // resident line above and the difference looks like a bug. Repacked
            // tensors live in `ggml`'s own arena, which this code has no
            // address for — a partial lock stated plainly beats a total that
            // quietly means something else.
            let (n_repacked, repacked) = weights.repacked();
            bigtea_arch::info!(
                "mlock      {:.2} GiB pinned in physical memory{}",
                report.locked_bytes as f64 / GIB,
                if n_repacked > 0 {
                    format!(
                        "; {:.2} GiB of repacked weights are in ggml's arena and not covered",
                        repacked as f64 / GIB
                    )
                } else {
                    String::new()
                }
            );
        } else {
            // Loud, and not fatal. A partial lock still helps, but a user who
            // asked for this must not believe it happened when it did not.
            bigtea_arch::info!(
                "mlock      FAILED for {:.2} GiB of {:.2}: {}",
                report.failed_bytes as f64 / GIB,
                (report.locked_bytes + report.failed_bytes) as f64 / GIB,
                report.reason
            );
        }
    }

    let (n_repacked, repacked_bytes) = weights.repacked();
    bigtea_arch::info!(
        "resident   {} tensors, {:.2} GiB in {:.1}s (experts stream on demand)",
        weights.len(),
        resident as f64 / GIB,
        load_start.elapsed().as_secs_f64()
    );
    if n_repacked > 0 {
        bigtea_arch::info!(
            "repacked   {n_repacked} tensors, {:.2} GiB in the CPU kernels' layout",
            repacked_bytes as f64 / GIB
        );
    }

    // Say which counts are in use and where they came from. Generation settles
    // on a measured count, and an unexplained "2" on a 20-thread machine reads
    // as a bug rather than as the 1.8x it is worth.
    bigtea_arch::info!(
        "threads    {} prefilling, generation {}",
        bigtea_arch::configured_threads_batch(),
        if std::env::var("BIGTEA_THREADS").is_ok() {
            format!("{} (-t)", bigtea_arch::configured_threads())
        } else {
            "tuned on the first tokens".to_string()
        }
    );

    let _ = arch;
    let prompt_len = tokens.len();

    let mut cache = KvCache::with_type(
        config.n_layer as usize,
        config.n_head_kv as usize,
        config.head_dim as usize,
        kv_type,
    );
    if cache.kind() != kv_type {
        bigtea_arch::info!(
            "kv cache   {} refused: head_dim {} is not a multiple of 32, using {}",
            kv_type.name(),
            config.head_dim,
            cache.kind().name()
        );
    }
    let vocab = config.vocab_size as usize;

    if let Some(chunk_size) = perplexity {
        return perplexity_run(
            &mut runner,
            &weights,
            &config,
            &tokens,
            chunk_size,
            kv_type,
            t0,
        );
    }

    // Prefill in blocks. Attention holds n_total * n_new * n_head floats for
    // scores and again for their softmax, so prefilling a long prompt in one
    // pass needs an arena quadratic in prompt length. Blocks bound it, and the
    // KV cache makes them equivalent — position 900 attends over 0..900 either
    // way.
    //
    // Block size is the central prefill trade-off: a block reads nearly every
    // expert in the model (16.35 GiB here) regardless of how many tokens are in
    // it, so doubling the block halves the disk cost per token — until the
    // attention arena, which grows with block * context, stops fitting.
    // One throwaway forward pass, on a cache that is then discarded.
    //
    // What it buys is real and measurable: the OS page cache holds the dense
    // weights, ggml's repacked copies exist, the arenas are sized, and the
    // thread ladder has one timed token to start from. What it costs is a full
    // block's worth of expert reads, which on a streaming model is gigabytes --
    // hence off by default here and on in llama.cpp. The runner says what it
    // spent so the number is attributable rather than absorbed into prefill.
    if warmup {
        let t = std::time::Instant::now();
        let mut throwaway = bigtea_arch::KvCache::with_type(
            config.n_layer as usize,
            config.n_head_kv as usize,
            config.head_dim as usize,
            cache.kind(),
        );
        // The prompt's own first token, not a synthetic one: a warmup on a
        // token the model will not see routes to different experts and warms
        // the wrong slices.
        match runner.forward_cached(&weights, &mut throwaway, &tokens[..1], 0) {
            Ok(_) => bigtea_arch::info!("warmup     1 token in {:.1}s", t.elapsed().as_secs_f64()),
            // A warmup that fails must not fail the run: it is an optimisation,
            // and the real pass is about to do the same work anyway.
            Err(e) => bigtea_arch::info!("warmup     skipped: {e}"),
        }
    }

    let prefill_start = std::time::Instant::now();
    let mut logits: Vec<f32> = Vec::new();
    let mut pos = 0usize;

    // Reuse as much of a saved cache as the prompts share.
    //
    // Reusable only up to the FIRST DIFFERING TOKEN: past it, every stored key
    // is conditioned on text that is no longer there, and attention would read
    // it without complaint. So the cache is truncated to the common prefix
    // rather than accepted or rejected whole — which is what makes it useful
    // for a prompt that was edited rather than repeated exactly.
    //
    // The last prompt token is never restored: the forward pass has to run for
    // at least one position to produce the logits that start generation.
    let fingerprint = PromptCache::fingerprint(&config, cache.kind());
    if let Some(path) = prompt_cache.as_deref() {
        if let Some((saved_tokens, layers)) = PromptCache::load(path, fingerprint) {
            let shared = common_prefix(&saved_tokens, &tokens).min(tokens.len().saturating_sub(1));
            if shared > 0 && layers.len() == cache.layers() {
                let mut ok = true;
                for (layer, (k, v)) in layers.iter().enumerate() {
                    if cache.restore_layer(layer, k, v).is_err() {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    cache.set_positions(saved_tokens.len());
                    cache.truncate_to(shared);
                    pos = shared;
                    bigtea_arch::info!(
                        "prompt cache  reused {shared} of {} tokens from {path}",
                        tokens.len()
                    );
                } else {
                    // A shape that does not divide cleanly means the file was
                    // written by a different build. Start over rather than
                    // restoring part of it.
                    cache.clear();
                    bigtea_arch::info!("prompt cache  {path} does not match this cache shape");
                }
            }
        }
    }

    for block in tokens[pos..].chunks(prefill_block) {
        logits = runner.forward_cached(&weights, &mut cache, block, pos)?;
        pos += block.len();
        debug_assert!(cache.is_consistent(), "kv cache layers fell out of step");
    }

    if let Some(path) = prompt_cache.as_deref() {
        if !prompt_cache_ro && !prompt_cache_all {
            match PromptCache::save(path, fingerprint, &cache, &tokens) {
                Ok(bytes) => bigtea_arch::info!(
                    "prompt cache  wrote {:.1} MiB for {} tokens to {path}",
                    bytes as f64 / (1 << 20) as f64,
                    tokens.len()
                ),
                // Not fatal: a cache that cannot be written is a lost
                // optimisation, not a failed run.
                Err(e) => bigtea_arch::info!("prompt cache  could not write {path}: {e}"),
            }
        }
    }
    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    bigtea_arch::info!(
        "prefill    {prompt_len} tokens in {prefill_secs:.1}s ({:.2} tok/s)",
        prompt_len as f64 / prefill_secs.max(1e-9)
    );

    if ui.verbose_prompt {
        eprintln!("prompt     {} tokens: {:?}", tokens.len(), tokens);
    }
    if ui.display_prompt {
        println!("\n{}", tokenizer.decode(&tokens));
    }
    let gen_start = std::time::Instant::now();

    let mut writer = TokenWriter::new();
    // Two things the parser cannot know, because both need the vocabulary:
    // which id is EOS (so `--ignore-eos` has something to suppress), and which
    // ids are fill-in-the-middle markers.
    let mut sampler_cfg = sampler_cfg;
    if sampler_cfg.eos.is_none() {
        sampler_cfg.eos = tokenizer.eos;
    }
    if infill {
        sampler_cfg.infill_suppress = infill_tokens(tokenizer);
        bigtea_arch::info!(
            "infill     suppressing {} FIM control tokens",
            sampler_cfg.infill_suppress.len()
        );
    }
    let mut sampler = Sampler::new(sampler_cfg);

    // Constrained decoding. The vocabulary is built once as token id -> the
    // bytes that token decodes to, which is what the grammar matches against.
    //
    // `allowed_from` with a matcher carried across tokens, not `allowed(prefix)`
    // — the latter replays the whole generated prefix through the grammar on
    // every single token, which is quadratic in the answer length.
    let grammar_vocab: Option<Vec<Vec<u8>>> = grammar.as_ref().map(|_| {
        (0..vocab)
            .map(|id| tokenizer.decode_bytes(&[id as u32]))
            .collect()
    });
    let constraint = match (&grammar, &grammar_vocab) {
        (Some(g), Some(v)) => Some(bigtea_grammar::Constraint::new(g.clone(), v)),
        _ => None,
    };
    let mut matcher = constraint.as_ref().map(|c| c.grammar().matcher());
    let mut turns = 0usize;

    // One iteration per exchange. A non-interactive run takes the `break` at
    // the bottom on its first pass, so its behaviour is exactly what it was.
    // `--interactive-first`: the user speaks before the model does. Skipping
    // the first generation rather than duplicating the turn-reading code below
    // keeps one path for appending a turn to the cache.
    let mut skip_generation = ui.interactive_first;
    loop {
        // Stop sequences are matched against the accumulated text, not the
        // token: a stop string can straddle a token boundary and per-token
        // matching would miss most of them. Reset per turn, or a stop string
        // from an earlier answer would end this one immediately.
        let mut generated_text = String::new();
        // A lazy grammar is off until a trigger appears; with no triggers the
        // grammar is armed from the first token, which is the ordinary case.
        let mut grammar_armed = grammar_triggers.is_empty();
        let this_turn = if skip_generation {
            skip_generation = false;
            0
        } else {
            n_predict
        };
        for step in 0..this_turn {
            if logits.len() < vocab {
                return Err(format!("logits too small: {} < {vocab}", logits.len()).into());
            }
            // The last token's row: a prefill returns logits for every position.
            let row = logits.len() - vocab;
            // Lazy grammars stay off until the model writes a trigger. Once
            // armed they never disarm: the point is to constrain the tail of
            // the answer, and re-checking would let the grammar switch off
            // again mid-structure.
            if !grammar_armed
                && !grammar_triggers.is_empty()
                && grammar_triggers
                    .iter()
                    .any(|t| !t.is_empty() && generated_text.contains(t.as_str()))
            {
                grammar_armed = true;
                bigtea_arch::info!(
                    "grammar    armed after {} tokens",
                    tokens.len().saturating_sub(prompt_len)
                );
            }
            if let (Some(c), Some(m)) = (constraint.as_ref(), matcher.as_ref()) {
                if !grammar_armed {
                    // Not yet triggered: sample unconstrained, and do NOT
                    // advance the matcher below either, or the grammar would
                    // be asked to parse the prose that preceded the trigger.
                    let last = &logits[row..];
                    let next = sampler.sample(last, &tokens);
                    if Some(next) == tokenizer.eos {
                        tokens.push(next);
                        break;
                    }
                    writer.push_visible(tokenizer, next, &ui);
                    tokens.push(next);
                    generated_text.push_str(&tokenizer.decode(std::slice::from_ref(&next)));
                    continue;
                }
                let mask = c.allowed_from(m);
                // **An empty mask must never be sampled from.** Every token
                // would be -inf, the argmax would be arbitrary, and generation
                // would stop looking exactly like a clean EOS.
                //
                // But empty has two meanings and they are not the same event:
                // a grammar that has *finished* admits nothing more, which is
                // the successful ending; one that is *stuck* admits nothing
                // because the text so far cannot be completed. Reporting the
                // second as if it were the first is how a truncated answer
                // passes for a complete one.
                if mask.is_empty() {
                    if m.is_complete() {
                        bigtea_arch::detail!(
                            "grammar    satisfied after {} tokens",
                            tokens.len().saturating_sub(prompt_len)
                        );
                    } else {
                        bigtea_arch::info!(
                            "grammar    STUCK after {} tokens — no token can continue, and the \
                             grammar is not satisfied. The answer is incomplete.",
                            tokens.len().saturating_sub(prompt_len)
                        );
                    }
                    break;
                }
                mask.apply(&mut logits[row..]);
            }
            let last = &logits[row..];
            let next = sampler.sample(last, &tokens);
            if Some(next) == tokenizer.eos {
                tokens.push(next);
                break;
            }
            writer.push_visible(tokenizer, next, &ui);
            tokens.push(next);
            // Advance the grammar by what was actually emitted. Done after the
            // EOS check above, so a stop token never has to parse.
            if let Some(m) = matcher.as_mut() {
                let text = tokenizer.decode(std::slice::from_ref(&next));
                m.accept_str(&text);
            }
            if !stop.is_empty() {
                generated_text.push_str(&tokenizer.decode(std::slice::from_ref(&next)));
                if stop
                    .iter()
                    .any(|s| !s.is_empty() && generated_text.contains(s.as_str()))
                {
                    break;
                }
            }

            // Only the new token needs computing; history lives in the cache.
            // Skipped on the last step — nothing would read those logits.
            if step + 1 < this_turn {
                logits = runner.forward_cached(&weights, &mut cache, &[next], pos)?;
                pos += 1;
            }
        }
        writer.finish();

        if !ui.interactive {
            break;
        }
        turns += 1;
        if ui.single_turn {
            break;
        }
        // The KV cache already holds everything said so far, so a turn costs
        // only the new tokens — which is the whole reason a REPL is worth
        // having over re-invoking the binary.
        let Some(line) = read_user_turn(&ui)? else {
            break; // EOF: Ctrl-D, or a pipe running out
        };
        let framed_turn = if ui.conversation {
            tokenizer.apply_chat_template(&[Message::new("user", &line)], true)
        } else {
            format!("{}{}{}", ui.in_prefix, line, ui.in_suffix)
        };
        let mut next_tokens = tokenizer.encode(&framed_turn);
        if ui.in_prefix_bos {
            if let Some(bos) = tokenizer.bos {
                next_tokens.insert(0, bos);
            }
        }
        if next_tokens.is_empty() {
            continue;
        }
        if ui.verbose_prompt {
            eprintln!("turn       {} tokens: {:?}", next_tokens.len(), next_tokens);
        }
        tokens.extend_from_slice(&next_tokens);
        logits = runner.forward_cached(&weights, &mut cache, &next_tokens, pos)?;
        pos += next_tokens.len();
    }

    // `--prompt-cache-all` extends the cache over what was generated too, so a
    // continued conversation resumes instead of re-reading its own answer.
    if let Some(path) = prompt_cache.as_deref() {
        if prompt_cache_all && !prompt_cache_ro {
            match PromptCache::save(path, fingerprint, &cache, &tokens) {
                Ok(bytes) => bigtea_arch::info!(
                    "prompt cache  wrote {:.1} MiB for {} tokens (prompt + generated) to {path}",
                    bytes as f64 / (1 << 20) as f64,
                    tokens.len()
                ),
                Err(e) => bigtea_arch::info!("prompt cache  could not write {path}: {e}"),
            }
        }
    }

    let secs = gen_start.elapsed().as_secs_f64();
    let produced = tokens.len().saturating_sub(prompt_len);
    if ui.print_token_count {
        println!("\ntokens     {} prompt + {produced} generated", prompt_len);
    }
    let _ = turns;
    println!("\n");
    bigtea_arch::info!(
        "generated  {produced} tokens in {secs:.1}s ({:.2} tok/s)",
        produced as f64 / secs.max(1e-9)
    );
    bigtea_arch::info!(
        "kv cache   {} positions, {:.1} MiB, {}",
        cache.len(),
        cache.bytes() as f64 / (1 << 20) as f64,
        cache.kind().name()
    );
    if show_perf {
        bigtea_arch::info!("streaming  {}", runner.stats);
    }
    // What the tuner settled on. Printed even when it did not finish, because
    // "still tuning after N tokens" explains an odd tok/s that would otherwise
    // look like a regression.
    let (settled, done) = runner.generation_threads();
    bigtea_arch::info!(
        "threads    generation used {settled}{}",
        if done { "" } else { " (still tuning)" }
    );
    bigtea_arch::info!("total      {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

// These are command-line options, not coupled state; grouping them into a
// struct would add a layer without removing a decision.
#[allow(clippy::too_many_arguments)]
fn run(
    path: &str,
    prompt: &str,
    n_predict: usize,
    prefill_block: usize,
    cache_budget: Option<u64>,
    sampler: SamplerConfig,
    chat: bool,
    threads_flag: Option<usize>,
    threads_batch_flag: Option<usize>,
    perplexity: Option<usize>,
    ui: Ui,
    system_prompt: Option<String>,
    rope: RopeOverrides,
    show_perf: bool,
    dry_breakers: Vec<String>,
    kv_type: bigtea_arch::KvType,
    overrides: Vec<(String, bigtea_gguf::Value)>,
    mlock: bool,
    chat_template: Option<String>,
    prompt_cache: Option<String>,
    prompt_cache_all: bool,
    prompt_cache_ro: bool,
    grammar_src: Option<String>,
    schema_src: Option<String>,
    ctx_size: Option<usize>,
    stop: Vec<String>,
    force: bool,
    warmup: bool,
    infill: bool,
    grammar_triggers: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = std::time::Instant::now();
    // Set once, read by every graph evaluation. A flag that only reached some
    // of them would make -t look ineffective on exactly the paths that matter.
    if let Some(t) = threads_flag {
        std::env::set_var("BIGTEA_THREADS", t.to_string());
    }
    if let Some(t) = threads_batch_flag {
        std::env::set_var("BIGTEA_THREADS_BATCH", t.to_string());
    }

    // --- container ---------------------------------------------------------
    // Built before the model is opened: a malformed grammar should fail in
    // milliseconds, not after a 17 GiB load.
    let grammar = match (grammar_src.as_deref(), schema_src.as_deref()) {
        (Some(_), Some(_)) => {
            return Err("--grammar and --json-schema are alternatives; pass one".into());
        }
        (Some(src), None) => Some(bigtea_grammar::Grammar::parse(src)?),
        (None, Some(src)) => Some(bigtea_grammar::Grammar::from_json_schema(src)?),
        (None, None) => None,
    };
    if let Some(g) = &grammar {
        bigtea_arch::info!("grammar    {} rules", g.rule_count());
    }

    let mut model = Model::open_split(path)?;
    // Applied before anything reads the metadata, and reported: a wrong
    // override is indistinguishable from a wrong container unless the run says
    // which one it used.
    for (key, value) in &overrides {
        bigtea_arch::info!("override   {key} = {value:?}");
        model.override_metadata(key, value.clone());
    }
    let model = model;

    // Refuse an architecture nobody has checked, rather than answering wrongly
    // and confidently. Gemma-2 loads through the generic dense path with no
    // error at all and replies to "The capital of France is" with "himſelf".
    if !architecture_is_verified(model.architecture()) && !force {
        // Built line by line: a multi-line format string keeps its source
        // indentation and prints a ragged message.
        let mut msg = String::new();
        msg.push_str(&format!(
            "{:?} is not an architecture this build has been verified against.
",
            model.architecture()
        ));
        msg.push_str(&format!(
            "           verified: {}
",
            VERIFIED_ARCHITECTURES.join(", ")
        ));
        msg.push_str(
            "
           It may load and generate, and be WRONG with no error.
",
        );
        msg.push_str(
            "           Gemma-2 does exactly that: it answers \"The capital of
",
        );
        msg.push_str(
            "           France is\" with \"himselff\", because it needs post-norms,
",
        );
        msg.push_str(
            "           logit soft-capping and embedding scaling this path does not
",
        );
        msg.push_str(
            "           implement -- none of which appear as a missing tensor.
",
        );
        msg.push_str(
            "
           Pass --force to run it anyway.",
        );
        return Err(msg.into());
    }

    // DeepSeek-V4-Flash shares the residency and streaming machinery but almost
    // none of the graph, so it gets its own path rather than a config branch.
    if model.architecture() == "deepseek4" {
        bigtea_arch::info!("model      {} ({})", model.architecture(), model.io_mode());
        let mut tokenizer = Tokenizer::from_metadata(model.metadata())?;
        force_chat_template(&mut tokenizer, chat_template.as_deref())?;
        let tokenizer = tokenizer;
        let prompt = &framed(
            &tokenizer,
            prompt,
            chat || ui.conversation,
            system_prompt.as_deref(),
        );
        run_deepseek4(
            &model,
            &tokenizer,
            prompt,
            n_predict,
            1024,
            cache_budget,
            sampler,
            t0,
        )?;
        return Ok(());
    }

    let mut config = Qwen3Config::from_model(&model)?;
    rope.apply(&mut config);
    let config = config;
    let arch = Qwen3Model::new(config.clone());

    bigtea_arch::info!("model      {} ({})", model.architecture(), model.io_mode());
    bigtea_arch::info!(
        "shape      {} layers, {} embd, {} heads ({} kv), head_dim {}",
        config.n_layer,
        config.n_embd,
        config.n_head,
        config.n_head_kv,
        config.head_dim
    );
    if config.is_moe() {
        bigtea_arch::info!(
            "experts    {} total, {} per token",
            config.n_expert,
            config.n_expert_used
        );
    } else {
        bigtea_arch::info!("experts    none (dense)");
    }
    bigtea_arch::info!(
        "attention  {} rope, per-head QK norm {}",
        if config.rope_type == 0 {
            "NORM"
        } else {
            "NeoX"
        },
        if config.qk_norm { "yes" } else { "no" }
    );
    if !config.rope_type_is_known {
        // Say it rather than let the user discover it in the output. Both RoPE
        // conventions run without error on either layout, so a wrong guess is
        // fluent nonsense and nothing downstream can detect it.
        bigtea_arch::info!(
            "           NOTE: {:?} is not an architecture this build has verified.",
            model.architecture()
        );
        bigtea_arch::info!("           NeoX rope and the tensor layout are assumed. If the output");
        bigtea_arch::info!("           is fluent but wrong, that assumption is the first suspect.");
    }

    // Fail on a missing tensor now, not at layer 37.
    arch.verify(&model)?;

    // --- tokenizer ---------------------------------------------------------
    let mut tokenizer = Tokenizer::from_metadata(model.metadata())?;
    force_chat_template(&mut tokenizer, chat_template.as_deref())?;
    let tokenizer = tokenizer;

    // DRY's sequence breakers arrive as text and the sampler works in ids, so
    // they can only be resolved once a vocabulary exists. Defaults are
    // llama.cpp's: a newline, a quote, a colon and an asterisk — the marks that
    // separate one structural unit from the next, and across which a "repeat"
    // is usually just a list having a shape.
    let mut sampler = sampler;
    if sampler.dry_multiplier > 0.0 {
        let wanted: Vec<String> = if dry_breakers.is_empty() {
            ["\n", ":", "\"", "*"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            dry_breakers
        };
        for text in &wanted {
            // BOS is dropped: `encode` prepends it for models that ask for one,
            // and a breaker is a piece of text, not the start of a sequence.
            // A breaker that is still not a single token cannot act as a
            // barrier, so it is skipped rather than silently matching
            // something else — "*" is one token in some vocabularies and part
            // of a merge in others.
            let ids: Vec<u32> = tokenizer
                .encode(text)
                .into_iter()
                .filter(|id| Some(*id) != tokenizer.bos)
                .collect();
            if let [only] = ids[..] {
                sampler.dry_sequence_breakers.push(only);
            }
        }
        bigtea_arch::detail!(
            "dry        {} sequence breakers resolved of {} asked for",
            sampler.dry_sequence_breakers.len(),
            wanted.len()
        );
    }
    let sampler = sampler;

    let prompt = &framed(
        &tokenizer,
        prompt,
        chat || ui.conversation,
        system_prompt.as_deref(),
    );
    let mut tokens: Vec<u32> = tokenizer.encode(prompt);
    bigtea_arch::info!("prompt     {prompt:?} -> {} tokens", tokens.len());
    if tokens.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }

    // Dense models go through the same path as MoE ones, because that is the
    // path with a **KV cache**.
    //
    // The uncached branch below rebuilds the graph over the whole sequence for
    // every token, which measured **0.67 tok/s against llama.cpp's 5.90** on
    // Qwen3-4B — 128 tokens from a 9-token prompt costs ~9,300 token-positions
    // of work. `StreamingRunner::forward_cached` computes only the new
    // position and attends over cached history; for a dense model there are no
    // routed experts, so "streaming" reduces to exactly that.
    // `BIGTEA_UNCACHED=1` keeps the old stateless path reachable, so the gain
    // can be measured rather than asserted.
    if std::env::var("BIGTEA_UNCACHED").is_err() {
        return run_streaming(
            &model,
            config,
            &arch,
            &tokenizer,
            tokens,
            n_predict,
            prefill_block,
            cache_budget,
            sampler,
            ctx_size,
            stop,
            perplexity,
            ui,
            show_perf,
            kv_type,
            mlock,
            prompt_cache,
            prompt_cache_all,
            prompt_cache_ro,
            grammar,
            warmup,
            infill,
            grammar_triggers,
            t0,
        );
    }

    // --- weights -----------------------------------------------------------
    // Metadata only: the data pointers reference buffers we own, so this arena
    // holds tensor structs rather than weights.
    let names = arch.required_tensors();
    let weight_ctx = Context::new_no_alloc(64 << 20)?;
    let mut weights = WeightSet::new();

    let load_start = std::time::Instant::now();
    let mut bound_bytes = 0u64;
    for name in &names {
        let loc = model
            .location(name)
            .ok_or_else(|| format!("missing tensor {name}"))?
            .clone();
        let data = model.read_tensor(name)?;
        bound_bytes += data.len() as u64;
        weights.bind(&weight_ctx, name, loc.ty, &loc.dims, data)?;
    }
    // The output projection is tied to the embeddings unless shipped separately.
    if model.location("output.weight").is_some() && weights.get("output.weight").is_none() {
        let loc = model.location("output.weight").expect("checked").clone();
        let data = model.read_tensor("output.weight")?;
        bound_bytes += data.len() as u64;
        weights.bind(&weight_ctx, "output.weight", loc.ty, &loc.dims, data)?;
    }
    bigtea_arch::info!(
        "weights    {} tensors, {:.2} GiB bound in {:.1}s (zero-copy)",
        weights.len(),
        bound_bytes as f64 / GIB,
        load_start.elapsed().as_secs_f64()
    );

    // Say which counts are in use and why. The generation default is a
    // measurement, and an unexplained "2" on a 20-thread machine looks like a
    // bug rather than the 1.7x it is worth.
    let threads = bigtea_arch::configured_threads();
    let threads_batch = bigtea_arch::configured_threads_batch();
    bigtea_arch::info!("threads    {threads} generating, {threads_batch} prefilling");

    // --- generate ----------------------------------------------------------
    println!("\n{prompt}");

    let mut produced = String::new();
    let gen_start = std::time::Instant::now();

    // Refuse a prompt this machine cannot prefill, with the numbers — rather
    // than letting ggml abort partway through, which is a `GGML_ASSERT` and
    // kills the process with nothing this code can catch or report.
    {
        let machine = bigtea_probe::Machine::probe(std::path::Path::new("."), false);
        let budget = machine
            .ram_available_bytes
            .unwrap_or(4 << 30)
            .saturating_sub(1 << 30);
        let need = dense_arena_bytes(&config, (tokens.len() + n_predict) as i64) as u64;
        if need > budget {
            return Err(format!(
                "this prompt is {} tokens and needs {:.2} GiB of compute arena, but only \
                 {:.2} GiB is free.\n           The dense path builds one graph over all {} \
                 layers and ggml frees nothing inside a context, so the arena grows with the \
                 sequence.\n           The longest that fits here is about {} tokens. Close \
                 some applications, or use a shorter prompt.",
                tokens.len(),
                need as f64 / GIB,
                budget as f64 / GIB,
                config.n_layer,
                dense_max_tokens(&config, budget),
            )
            .into());
        }
    }

    let mut sampler = Sampler::new(sampler);
    let mut writer = TokenWriter::new();
    for step in 0..n_predict {
        let n = tokens.len() as i64;
        // A fresh compute arena per token: intermediates are dead once the
        // token is chosen, and reclaiming them keeps peak memory flat rather
        // than growing with the sequence.
        //
        // Sized from the sequence, not fixed. A flat 2 GiB **aborted** on a
        // 651-token prompt — `ggml_new_object: not enough space`, which is a
        // `GGML_ASSERT` and kills the process rather than returning an error
        // this code could report. Attention holds `n * n * n_head` floats for
        // the scores and again for their softmax, so the requirement is
        // quadratic in the sequence and a constant can only ever be wrong at
        // some length.
        let arena = dense_arena_bytes(&config, n);
        let ctx = Context::new(arena)?;

        let tok = ctx.new_i32_1d(n)?;
        tok.set_i32(&tokens.iter().map(|&t| t as i32).collect::<Vec<_>>())?;
        let pos = ctx.new_i32_1d(n)?;
        pos.set_i32(&(0..n as i32).collect::<Vec<_>>())?;

        let logits = arch.build_graph(&ctx, &weights, &tok, &pos, n)?;
        ctx.compute(&logits, threads)?;

        // The final position's row. Greedy is still the default -- see
        // `SamplerConfig::default` -- so a wrong forward pass stays diagnosable
        // unless the caller opts into sampling.
        let all = logits.to_vec_f32();
        let vocab = config.vocab_size as usize;
        if all.len() < vocab {
            return Err(format!("logits too small: {} < vocab {}", all.len(), vocab).into());
        }
        let last = &all[all.len() - vocab..];
        let next = sampler.sample(last, &tokens);
        if Some(next) == tokenizer.eos {
            println!("\n[end of sequence at step {step}]");
            break;
        }
        writer.push(&tokenizer, next);
        produced.push_str(&tokenizer.decode(std::slice::from_ref(&next)));
        tokens.push(next);
    }

    let secs = gen_start.elapsed().as_secs_f64();
    let count = tokens.len() - tokenizer.encode(prompt).len();
    println!("\n");
    bigtea_arch::info!(
        "generated  {count} tokens in {secs:.1}s ({:.2} tok/s)",
        count as f64 / secs.max(1e-9)
    );
    bigtea_arch::info!("total      {:.1}s", t0.elapsed().as_secs_f64());
    if produced.trim().is_empty() {
        println!("\n! produced no visible text -- check the forward pass");
    }
    Ok(())
}

/// Prefill DeepSeek-V4-Flash and time it.
///
/// Separate from the Qwen3 path because almost nothing is shared: MLA attention,
/// hyper-connections instead of a residual add, two compressors, two routing
/// schemes. What *is* shared is the point of the project — residency, partial
/// reads, and the arena discipline.
///
/// **Prefill only.** Generation needs the persistent compressor ring that a
/// prefill can skip, a growing KV cache, and the expert cache; see
/// `deepseek4_forward`'s module docs. Timing this first is deliberate: if
/// prefill is slow, that changes what generation should look like.
/// Say what a residency shortfall costs and what would fix it.
///
/// This is the difference between a tool that is slow and a tool that is slow
/// *and inexplicable*. Weights that do not fit are re-read on every token
/// forever, so the shortfall is not a one-off — it is a permanent tax, and the
/// user is the only one who can decide whether closing an editor is worth
/// paying less of it. Naming the processes turns "it's slow" into a choice.
///
/// The saving is quoted as a range because a re-read costs somewhere between
/// the drive's sequential rate and what these scattered tensor reads actually
/// achieve; promising the optimistic end would be a lie the first time someone
/// timed it.
fn report_residency_shortfall(report: &bigtea_model::LoadReport, machine: &bigtea_probe::Machine) {
    if report.complete() {
        return;
    }
    let missing = report.skipped_over_budget;
    if missing == 0 {
        return; // the shortfall is undownloaded weights, not RAM
    }
    // What re-reading them costs per token, at the rate this load just achieved.
    let rate = if report.bytes_per_sec() > 0.0 {
        report.bytes_per_sec()
    } else {
        1e9
    };
    bigtea_arch::info!(
        "           {:.2} GiB will be re-read from disk on EVERY token (~{:.1}s each)",
        missing as f64 / GIB,
        missing as f64 / rate
    );

    let holders = bigtea_probe::processes::grouped(256 << 20);
    if holders.is_empty() {
        bigtea_arch::info!("           nothing large is closeable; this model needs more RAM than this machine has");
        return;
    }
    let free: u64 = holders.iter().map(|(_, b, _)| *b).sum();
    bigtea_arch::info!(
        "           closing these would free up to {:.2} GiB:",
        free as f64 / GIB
    );
    for (name, bytes, count) in holders.iter().take(4) {
        let n = if *count > 1 {
            format!(" ({count} processes)")
        } else {
            String::new()
        };
        bigtea_arch::info!("             {name:<28} {:.2} GiB{n}", *bytes as f64 / GIB);
    }
    if free >= missing {
        bigtea_arch::info!("           that is enough to make the whole model resident.");
    } else {
        bigtea_arch::info!(
            "           still {:.2} GiB short after that — a smaller quant would fit.",
            (missing - free) as f64 / GIB
        );
    }
    let _ = machine;
}

#[allow(clippy::too_many_arguments)]
fn run_deepseek4(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    n_predict: usize,
    arena_mib: usize,
    expert_cache_budget: Option<u64>,
    sampler_cfg: SamplerConfig,
    t0: std::time::Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = bigtea_arch::Deepseek4Config::from_model(model)?;
    let vocab = config.vocab_size as usize;

    let tokens: Vec<i32> = tokenizer.encode(prompt).iter().map(|t| *t as i32).collect();
    if tokens.is_empty() {
        return Err("empty prompt".into());
    }

    bigtea_arch::info!(
        "shape      {} blocks, {} embd, {} heads, {} experts ({} used, {} shared)",
        config.n_layer,
        config.n_embd,
        config.n_head,
        config.n_expert,
        config.n_expert_used,
        config.n_expert_shared
    );
    bigtea_arch::info!("prompt     {} tokens", tokens.len());

    // Hold the always-read weights in RAM. Without this every block re-reads
    // them from disk on every forward pass — 23% of a prefill, and the whole
    // cost again for each generated token.
    //
    // The budget is what the machine has free now, minus room for the compute
    // arena and the expert slices in flight. Over-estimating makes the OS swap,
    // and swapping is slower than the streaming it was meant to replace, so the
    // reserve is deliberate and what does not fit is reported rather than hidden.
    let machine = bigtea_probe::Machine::probe(std::path::Path::new("."), false);
    // Compute arena, plus the expert slices in flight, plus slack for the OS.
    // A flat constant here is either wasteful or wrong depending on the block.
    let reserve = ((arena_mib as u64) << 20) + (512 << 20) + (768 << 20);
    let budget = machine.usable_ram_for_weights(reserve);
    let (mut resident, report) = ResidentSet::load(model, budget)?;
    bigtea_arch::info!("resident   {report}");
    report_residency_shortfall(&report, &machine);

    // Rearrange the always-read weights into the layout the CPU kernels want,
    // once, before any block runs.
    //
    // It has to happen here rather than inside the block loop: V4-Flash owns an
    // arena per block and rebuilds its `WeightSet` 43 times a token, so
    // rearranging there would re-do the whole set on every one of them.
    //
    // Each tensor is taken out of the resident set as it is converted, so the
    // footprint does not double — which on a 15.7 GiB machine holding a 7.38
    // GiB always-read set is the difference between this working and swapping.
    let repack_start = std::time::Instant::now();
    let repacked = bigtea_arch::RepackedDense::build(&mut resident, model)?;
    let (n_repacked, repacked_bytes, declined) = repacked.stats();
    if n_repacked > 0 {
        bigtea_arch::info!(
            "repacked   {n_repacked} tensors, {:.2} GiB in the CPU kernels' layout, {:.1}s",
            repacked_bytes as f64 / GIB,
            repack_start.elapsed().as_secs_f64()
        );
        if declined > 0 {
            bigtea_arch::info!(
                "repacked   {declined} declined by ggml and left in their stored layout"
            );
        }
    }

    // The expert cache is **off unless asked for**, and that default is measured
    // rather than cautious.
    //
    // Expert reads are deduplicated per block across the batch, so a pass reads
    // the *distinct* experts its tokens select: 6 per layer at one token, but
    // 39.7 at 17 tokens and 122.8 at 166 — about 66 GiB in a single pass. The
    // RAM left on this machine after the 7.38 GiB always-read set is ~1.5 GiB,
    // which is 2% of that, and 2% is what it returned:
    //
    //     17 tokens,  1.51 GiB cache -> 4.1% hits, 0.049 -> 0.050 tok/s
    //     166 tokens, 1.75 GiB cache -> 1.9% hits, 0.015 -> 0.015 tok/s
    //     166-token prefill          -> 64.5s -> 75.3s, 17% SLOWER
    //
    // The slowdown is the admission copies, paid on every miss that gets kept.
    // So on today's engine the cache is a regression, and turning it on by
    // default would ship one.
    //
    // It becomes worth having the moment a step stops re-reading the whole
    // sequence — a KV-cached token needs 6 experts per layer, 3.21 GiB, and R0.1
    // measured that a set warmed on the prompt covers 86% of what generation
    // then asks for. **The cache is not wrong; it is early.**
    //
    // Nothing is ever pre-loaded: R0 measured a hot set chosen in advance
    // covering 37.5% of an unseen subject against 25% for caching at random.
    // And the cache owns its memory, because past ~6 GiB on Qwen3 a 71%-hit
    // cache backed by the page cache was the *slowest* configuration measured.
    let mut fw = bigtea_arch::Deepseek4Forward::new(model, config.clone())
        .with_resident(&resident)
        .with_repacked(&repacked);
    // The expert cache and the always-read weights compete for the same RAM, and
    // measurement says residency wins by a wide margin until it is satisfied.
    //
    // Measured 2026-08-09 with 5.7 GiB free, so 4.95 GiB of the always-read set
    // was streaming: a 2 GiB expert cache reached 12.6% hits and moved generation
    // 0.127 -> 0.134 tok/s. A resident byte is read every token by definition —
    // a 100% hit rate — so it is worth roughly 8x an expert-cache byte here, and
    // the 2 GiB spent on the cache came straight out of residency.
    //
    // So the cache is refused, with the arithmetic, until the always-read set
    // fits. It is not a weak cache; it is the wrong place to spend the byte.
    let shortfall = report.skipped_over_budget;
    let expert_budget = match expert_cache_budget {
        Some(b) if b > 0 && shortfall > 0 => {
            bigtea_arch::info!(
                "cache      refusing {:.2} GiB for experts: {:.2} GiB of always-read",
                b as f64 / GIB,
                shortfall as f64 / GIB
            );
            bigtea_arch::info!(
                "cache      weights is still streaming, and a resident byte is read"
            );
            bigtea_arch::info!("cache      every token (100%) against ~13% for a cached expert.");
            bigtea_arch::info!(
                "cache      Free ~{:.1} GiB and it becomes worth having.",
                shortfall as f64 / GIB
            );
            0
        }
        Some(b) => b,
        None => 0,
    };
    if expert_budget > 0 {
        fw = fw.with_expert_cache(expert_budget as usize);
        bigtea_arch::info!(
            "cache      {:.2} GiB for routed experts, warmed from the prompt (not pinned)",
            expert_budget as f64 / GIB
        );
    } else if shortfall == 0 && expert_cache_budget.is_none() {
        bigtea_arch::info!("cache      off. The always-read set fits, so --cache <GiB> is now");
        bigtea_arch::info!("cache      worth measuring: a cached step reads 6 experts per layer,");
        bigtea_arch::info!("cache      not the ~123 a long prefill does.");
    }
    let fw = fw;
    if !fw.indexer_is_exact(tokens.len()) {
        // Below this length skipping the indexer is exact; above it, it is not.
        println!(
            "WARNING    the lightning indexer is not implemented, and at {} tokens\n\
             WARNING    it would no longer be a no-op. These logits are APPROXIMATE.",
            tokens.len()
        );
    }
    bigtea_arch::info!("loaded     {:.1}s", t0.elapsed().as_secs_f64());

    let t_prefill = std::time::Instant::now();
    let mut seq = tokens.clone();
    // One cache for the whole session: the prompt fills it, and each generated
    // token appends a single row instead of re-running the sequence.
    let mut kv = bigtea_arch::Deepseek4Cache::new(config.n_layer, config.kv_lora_rank);
    let logits = bigtea_arch::forward(&fw, &mut kv, &seq, arena_mib << 20)?;
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    bigtea_arch::info!(
        "prefill    {} tokens in {prefill_secs:.1}s ({:.2} tok/s)",
        seq.len(),
        seq.len() as f64 / prefill_secs
    );

    let mut sampler = Sampler::new(sampler_cfg);
    let mut next = sample_next(&mut sampler, &logits, vocab, &seq);
    print!("output     {}", tokenizer.decode(&[next as u32]));
    use std::io::Write;
    let _ = std::io::stdout().flush();

    // Generate by re-running the whole sequence, one forward pass per token.
    //
    // This is the honest version of a generation loop and not the fast one. A
    // KV cache would let each token attend over the previous ones without
    // recomputing them; without it, token N costs a forward pass over N tokens.
    //
    // It is much less wasteful here than it sounds, because the cost of a
    // forward pass on this model is dominated by reading 3.21 GiB of routed
    // experts — which is paid **per pass, not per token** — and not by the
    // attention that the cache would save. The quadratic term is real but small
    // at these lengths. What it buys is a loop that is correct by construction:
    // every pass is stateless and identical to a prefill, so there is no cache
    // to get subtly wrong, and on this architecture a wrong cache produces
    // fluent nonsense rather than an error.
    let t_gen = std::time::Instant::now();
    let mut generated = 0usize;
    let mut writer = TokenWriter::new();
    while generated + 1 < n_predict {
        seq.push(next);
        // Each iteration is a fresh pass over the whole sequence. Telling the
        // routing histogram so keeps the prompt from being counted again per
        // token — and makes the per-pass difference a single token's routing.
        bigtea_arch::routing_next_pass();
        let logits = bigtea_arch::step(&fw, &mut kv, next, arena_mib << 20)?;
        next = sample_next(&mut sampler, &logits, vocab, &seq);
        generated += 1;
        writer.push(tokenizer, next as u32);
    }
    writer.finish();
    println!();

    // 137 GiB of routed experts, if the router spreads evenly. If it does not,
    // the hot set is cacheable and every byte-per-token figure changes.
    bigtea_arch::routing_report(137.06, fw.config().hash_layer_count);
    // 3.21 GiB is what one token's six-of-256 costs on this container; the
    // report scales it by how many of the six would actually be read.
    bigtea_arch::routing_weight_report(3.21);

    // Hit rate is reported **with** footprint and next to tok/s below, never
    // alone. This project has measured a 71%-hit cache being the slowest
    // configuration it had, because the cached bytes were being paged out — so a
    // hit rate on its own is not evidence of anything.
    if let Some((stats, bytes)) = fw.cache_stats() {
        bigtea_arch::info!(
            "cache      {:.1}% hits ({} of {}), {:.2} GiB resident of {:.2} GiB, \
             {} evictions, {:.1} GiB not read",
            stats.hit_rate() * 100.0,
            stats.hits,
            stats.hits + stats.misses,
            bytes as f64 / GIB,
            expert_budget as f64 / GIB,
            stats.evictions,
            stats.bytes_saved as f64 / GIB,
        );
    }

    if generated > 0 {
        let secs = t_gen.elapsed().as_secs_f64();
        println!(
            "generate   {generated} tokens in {secs:.1}s ({:.3} tok/s, {:.1}s per token)",
            generated as f64 / secs,
            secs / generated as f64
        );
    }
    Ok(())
}

/// The last position's logits, and the token drawn from them.
///
/// A prefill returns a row per position; only the final one predicts the next
/// token. Taking the whole buffer would sample from position 0 — which reads
/// as the model ignoring the prompt.
fn sample_next(sampler: &mut Sampler, logits: &[f32], vocab: usize, seq: &[i32]) -> i32 {
    let row = if logits.len() >= vocab {
        &logits[logits.len() - vocab..]
    } else {
        logits
    };
    // The repeat penalty indexes by token id; this path carries i32.
    let history: Vec<u32> = seq.iter().map(|&t| t as u32).collect();
    sampler.sample(row, &history) as i32
}
