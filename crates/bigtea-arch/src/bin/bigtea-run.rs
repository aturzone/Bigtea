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
    let prompt = args.next().unwrap_or_else(|| "The capital of France is".into());
    let mut n_predict = 8usize;
    while let Some(a) = args.next() {
        if a == "-n" {
            n_predict = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);
        }
    }

    match run(&path, &prompt, n_predict) {
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
    t0: std::time::Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    use bigtea_arch::StreamingRunner;

    // 1 GiB of expert cache: enough to help on repeated routing, bounded so it
    // cannot grow into the problem streaming exists to avoid.
    let mut runner = StreamingRunner::new(model, config.clone(), 1 << 30);

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
    println!("\n{}", tokenizer.decode(&tokens));
    let gen_start = std::time::Instant::now();
    let prompt_len = tokens.len();

    // Prefill the whole prompt once, then feed one token at a time. Without
    // the cache each step re-ran every previous position, which is where the
    // 31,032 expert reads for 5 tokens came from.
    let mut cache = KvCache::new(
        config.n_layer as usize,
        config.n_head_kv as usize,
        config.head_dim as usize,
    );
    let mut pending: Vec<u32> = tokens.clone();
    let mut pos = 0usize;

    for _ in 0..n_predict {
        let logits = runner.forward_cached(&weights, &mut cache, &pending, pos)?;
        pos += pending.len();
        debug_assert!(cache.is_consistent(), "kv cache layers fell out of step");

        let vocab = config.vocab_size as usize;
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
        pending = vec![next];
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

fn run(path: &str, prompt: &str, n_predict: usize) -> Result<(), Box<dyn std::error::Error>> {
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
        return run_streaming(&model, config, &arch, &tokenizer, tokens, n_predict, t0);
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
