//! Generate text. The first end-to-end path through every layer of Bigtea.
//!
//! Usage: `bigtea-run <model.gguf> "prompt" [-n tokens]`
//!
//! Pipeline: container -> residency -> zero-copy weight binding -> tokenizer
//! -> forward graph -> logits -> sampling -> text.

use std::process::ExitCode;

use bigtea_arch::{KvCache, Qwen3Config, Qwen3Model, Sampler, SamplerConfig};
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
}

impl TokenWriter {
    fn new() -> Self {
        TokenWriter {
            pending: Vec::new(),
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

    /// Anything still buffered at the end was genuinely malformed, so it is
    /// shown lossily rather than silently dropped.
    fn finish(&mut self) {
        if !self.pending.is_empty() {
            print!("{}", String::from_utf8_lossy(&self.pending));
            self.pending.clear();
        }
    }
}

/// Apply the model's chat template when asked, and say which one was used.
///
/// An instruct model trained on `<|im_start|>user` does not fail on raw text —
/// it continues it. Asked to "Write one sentence about the sea", Llama-3.2
/// answered "The sentence should be concise and evocative", because it was
/// completing an instruction rather than following one.
fn framed(tokenizer: &Tokenizer, prompt: &str, chat: bool) -> String {
    if !chat {
        return prompt.to_string();
    }
    let format = tokenizer.chat_format();
    if format.is_known() {
        println!("chat       {} template", format.name());
    } else {
        // Do not pretend. An unrecognised template framed as someone else's is
        // how a model quietly answers the wrong question.
        println!("chat       template not recognised -- using a plain framing;");
        println!("           the model may not respond as an assistant.");
    }
    tokenizer.apply_chat_template(&[Message::new("user", prompt)], true)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
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
        eprintln!("  --repeat-last-n N   penalty window (default 64)");
        eprintln!("  --seed S            reproducible sampling");
        eprintln!("  --llamacpp-defaults temp 0.8, top-k 40, top-p 0.95, min-p 0.05, repeat 1.1");
        eprintln!("  --chat              apply the model's chat template to the prompt");
        return ExitCode::from(2);
    };
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
    let rest: Vec<String> = args.collect();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "-n" => {
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
            // One flag for "sample the way llama.cpp does by default", so a
            // quality comparison is not silently comparing sampler settings.
            "--llamacpp-defaults" => {
                sampler = SamplerConfig::llamacpp_defaults();
                i += 1;
            }
            "-b" => {
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
            "-f" => {
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
    let prompt = prompt;

    match run(
        &path,
        &prompt,
        n_predict,
        prefill_block,
        cache_budget,
        sampler,
        chat,
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
    t0: std::time::Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    use bigtea_arch::StreamingRunner;

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
    println!(
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
    println!(
        "resident   {} tensors, {:.2} GiB in {:.1}s (experts stream on demand)",
        weights.len(),
        resident as f64 / GIB,
        load_start.elapsed().as_secs_f64()
    );

    let _ = arch;
    let prompt_len = tokens.len();

    let mut cache = KvCache::new(
        config.n_layer as usize,
        config.n_head_kv as usize,
        config.head_dim as usize,
    );
    let vocab = config.vocab_size as usize;

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
    let prefill_start = std::time::Instant::now();
    let mut logits: Vec<f32> = Vec::new();
    let mut pos = 0usize;
    for block in tokens.chunks(prefill_block) {
        logits = runner.forward_cached(&weights, &mut cache, block, pos)?;
        pos += block.len();
        debug_assert!(cache.is_consistent(), "kv cache layers fell out of step");
    }
    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    println!(
        "prefill    {prompt_len} tokens in {prefill_secs:.1}s ({:.2} tok/s)",
        prompt_len as f64 / prefill_secs.max(1e-9)
    );

    println!("\n{}", tokenizer.decode(&tokens));
    let gen_start = std::time::Instant::now();

    let mut writer = TokenWriter::new();
    let mut sampler = Sampler::new(sampler_cfg);
    for step in 0..n_predict {
        if logits.len() < vocab {
            return Err(format!("logits too small: {} < {vocab}", logits.len()).into());
        }
        // The last token's row: a prefill returns logits for every position.
        let last = &logits[logits.len() - vocab..];
        let next = sampler.sample(last, &tokens);
        if Some(next) == tokenizer.eos {
            break;
        }
        writer.push(tokenizer, next);
        tokens.push(next);

        // Only the new token needs computing; history lives in the cache.
        // Skipped on the last step — nothing would read those logits.
        if step + 1 < n_predict {
            logits = runner.forward_cached(&weights, &mut cache, &[next], pos)?;
            pos += 1;
        }
    }

    let secs = gen_start.elapsed().as_secs_f64();
    let produced = tokens.len() - prompt_len;
    println!("\n");
    println!(
        "generated  {produced} tokens in {secs:.1}s ({:.2} tok/s)",
        produced as f64 / secs.max(1e-9)
    );
    println!(
        "kv cache   {} positions, {:.1} MiB",
        cache.len(),
        cache.bytes() as f64 / (1 << 20) as f64
    );
    println!("streaming  {}", runner.stats);
    println!("total      {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn run(
    path: &str,
    prompt: &str,
    n_predict: usize,
    prefill_block: usize,
    cache_budget: Option<u64>,
    sampler: SamplerConfig,
    chat: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = std::time::Instant::now();

    // --- container ---------------------------------------------------------
    let model = Model::open_split(path)?;

    // DeepSeek-V4-Flash shares the residency and streaming machinery but almost
    // none of the graph, so it gets its own path rather than a config branch.
    if model.architecture() == "deepseek4" {
        println!("model      {} ({})", model.architecture(), model.io_mode());
        let tokenizer = Tokenizer::from_metadata(model.metadata())?;
        let prompt = &framed(&tokenizer, prompt, chat);
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

    let config = Qwen3Config::from_model(&model)?;
    let arch = Qwen3Model::new(config.clone());

    println!("model      {} ({})", model.architecture(), model.io_mode());
    println!(
        "shape      {} layers, {} embd, {} heads ({} kv), head_dim {}",
        config.n_layer, config.n_embd, config.n_head, config.n_head_kv, config.head_dim
    );
    if config.is_moe() {
        println!(
            "experts    {} total, {} per token",
            config.n_expert, config.n_expert_used
        );
    } else {
        println!("experts    none (dense)");
    }
    println!(
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
        println!(
            "           NOTE: {:?} is not an architecture this build has verified.",
            model.architecture()
        );
        println!("           NeoX rope and the tensor layout are assumed. If the output");
        println!("           is fluent but wrong, that assumption is the first suspect.");
    }

    // Fail on a missing tensor now, not at layer 37.
    arch.verify(&model)?;

    // --- tokenizer ---------------------------------------------------------
    let tokenizer = Tokenizer::from_metadata(model.metadata())?;
    let prompt = &framed(&tokenizer, prompt, chat);
    let mut tokens: Vec<u32> = tokenizer.encode(prompt);
    println!("prompt     {prompt:?} -> {} tokens", tokens.len());
    if tokens.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }

    if config.is_moe() {
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
    println!(
        "weights    {} tensors, {:.2} GiB bound in {:.1}s (zero-copy)",
        weights.len(),
        bound_bytes as f64 / GIB,
        load_start.elapsed().as_secs_f64()
    );

    // --- generate ----------------------------------------------------------
    println!("\n{prompt}");
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut produced = String::new();
    let gen_start = std::time::Instant::now();

    let mut sampler = Sampler::new(sampler);
    let mut writer = TokenWriter::new();
    for step in 0..n_predict {
        // A fresh compute arena per token: intermediates are dead once the
        // token is chosen, and reclaiming them keeps peak memory flat rather
        // than growing with the sequence.
        let ctx = Context::new(2 << 30)?;

        let n = tokens.len() as i64;
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
    println!(
        "generated  {count} tokens in {secs:.1}s ({:.2} tok/s)",
        count as f64 / secs.max(1e-9)
    );
    println!("total      {:.1}s", t0.elapsed().as_secs_f64());
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
    println!(
        "           {:.2} GiB will be re-read from disk on EVERY token (~{:.1}s each)",
        missing as f64 / GIB,
        missing as f64 / rate
    );

    let holders = bigtea_probe::processes::grouped(256 << 20);
    if holders.is_empty() {
        println!("           nothing large is closeable; this model needs more RAM than this machine has");
        return;
    }
    let free: u64 = holders.iter().map(|(_, b, _)| *b).sum();
    println!(
        "           closing these would free up to {:.2} GiB:",
        free as f64 / GIB
    );
    for (name, bytes, count) in holders.iter().take(4) {
        let n = if *count > 1 {
            format!(" ({count} processes)")
        } else {
            String::new()
        };
        println!("             {name:<28} {:.2} GiB{n}", *bytes as f64 / GIB);
    }
    if free >= missing {
        println!("           that is enough to make the whole model resident.");
    } else {
        println!(
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

    println!(
        "shape      {} blocks, {} embd, {} heads, {} experts ({} used, {} shared)",
        config.n_layer,
        config.n_embd,
        config.n_head,
        config.n_expert,
        config.n_expert_used,
        config.n_expert_shared
    );
    println!("prompt     {} tokens", tokens.len());

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
    let (resident, report) = ResidentSet::load(model, budget)?;
    println!("resident   {report}");
    report_residency_shortfall(&report, &machine);

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
    let mut fw = bigtea_arch::Deepseek4Forward::new(model, config.clone()).with_resident(&resident);
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
            println!(
                "cache      refusing {:.2} GiB for experts: {:.2} GiB of always-read",
                b as f64 / GIB,
                shortfall as f64 / GIB
            );
            println!("cache      weights is still streaming, and a resident byte is read");
            println!("cache      every token (100%) against ~13% for a cached expert.");
            println!(
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
        println!(
            "cache      {:.2} GiB for routed experts, warmed from the prompt (not pinned)",
            expert_budget as f64 / GIB
        );
    } else if shortfall == 0 && expert_cache_budget.is_none() {
        println!("cache      off. The always-read set fits, so --cache <GiB> is now");
        println!("cache      worth measuring: a cached step reads 6 experts per layer,");
        println!("cache      not the ~123 a long prefill does.");
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
    println!("loaded     {:.1}s", t0.elapsed().as_secs_f64());

    let t_prefill = std::time::Instant::now();
    let mut seq = tokens.clone();
    // One cache for the whole session: the prompt fills it, and each generated
    // token appends a single row instead of re-running the sequence.
    let mut kv = bigtea_arch::Deepseek4Cache::new(config.n_layer, config.kv_lora_rank);
    let logits = bigtea_arch::forward(&fw, &mut kv, &seq, arena_mib << 20)?;
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    println!(
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
        println!(
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
