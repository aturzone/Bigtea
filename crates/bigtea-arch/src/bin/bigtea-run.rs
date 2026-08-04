//! Generate text. The first end-to-end path through every layer of Bigtea.
//!
//! Usage: `bigtea-run <model.gguf> "prompt" [-n tokens]`
//!
//! Pipeline: container -> residency -> zero-copy weight binding -> tokenizer
//! -> forward graph -> logits -> sampling -> text.

use std::process::ExitCode;

use bigtea_arch::{KvCache, Qwen3Config, Qwen3Model};
use bigtea_ggml::{Context, WeightSet};
use bigtea_model::Model;
use bigtea_tokenizer::Tokenizer;

const GIB: f64 = (1u64 << 30) as f64;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: bigtea-run <model.gguf> \"prompt\" [-n tokens]");
        return ExitCode::from(2);
    };
    let mut prompt = String::new();
    let mut n_predict = 8usize;
    let mut prefill_block = 256usize;
    let mut cache_budget: Option<u64> = None;
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

    match run(&path, &prompt, n_predict, prefill_block, cache_budget) {
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
fn run_streaming(
    model: &Model,
    config: Qwen3Config,
    arch: &Qwen3Model,
    tokenizer: &Tokenizer,
    mut tokens: Vec<u32>,
    n_predict: usize,
    prefill_block: usize,
    cache_budget: Option<u64>,
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
    const HEADROOM: u64 = 4 * (1 << 30);
    let budget = match cache_budget {
        Some(bytes) => bytes,
        None => {
            let machine = bigtea_probe::Machine::probe(std::path::Path::new("."), false);
            machine
                .ram_available_bytes
                .map(|avail| avail.saturating_sub(HEADROOM).max(1 << 30))
                .unwrap_or(1 << 30)
        }
    };
    let mut runner = StreamingRunner::new(model, config.clone(), budget as usize);
    println!("cache      {:.2} GiB for experts", budget as f64 / GIB);

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

    for step in 0..n_predict {
        if logits.len() < vocab {
            return Err(format!("logits too small: {} < {vocab}", logits.len()).into());
        }
        let last = &logits[logits.len() - vocab..];
        let (best, _) = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or("empty logits")?;

        let next = best as u32;
        if Some(next) == tokenizer.eos {
            break;
        }
        print!("{}", tokenizer.decode(&[next]));
        use std::io::Write;
        std::io::stdout().flush().ok();
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
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = std::time::Instant::now();

    // --- container ---------------------------------------------------------
    let model = Model::open_split(path)?;
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

    // Fail on a missing tensor now, not at layer 37.
    arch.verify(&model)?;

    // --- tokenizer ---------------------------------------------------------
    let tokenizer = Tokenizer::from_metadata(model.metadata())?;
    let mut tokens: Vec<u32> = tokenizer.encode(prompt);
    println!("prompt     {prompt:?} -> {} tokens", tokens.len());
    if tokens.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }

    if config.is_moe() {
        return run_streaming(
            &model, config, &arch, &tokenizer, tokens, n_predict, prefill_block, cache_budget, t0,
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

        // Greedy: the highest-scoring token of the final position. Sampling
        // strategies come later; determinism is what makes a wrong forward
        // pass diagnosable.
        let all = logits.to_vec_f32();
        let vocab = config.vocab_size as usize;
        if all.len() < vocab {
            return Err(format!("logits too small: {} < vocab {}", all.len(), vocab).into());
        }
        let last = &all[all.len() - vocab..];
        let (best, _) = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or("empty logits")?;

        let next = best as u32;
        if Some(next) == tokenizer.eos {
            println!("\n[end of sequence at step {step}]");
            break;
        }
        let piece = tokenizer.decode(&[next]);
        print!("{piece}");
        use std::io::Write;
        std::io::stdout().flush().ok();
        produced.push_str(&piece);
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
