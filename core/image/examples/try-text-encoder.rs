//! Check the text encoder by making it finish a sentence.
//!
//! The conditioning path never samples a token — it takes thirteen layers of
//! hidden states and hands them to the denoiser. But hidden states that are
//! subtly wrong are finite, correctly shaped, and produce a picture that looks
//! like a picture. There is no way to see the error downstream.
//!
//! So the forward pass is checked the way every other model in this project is:
//! run it as a language model and require **" Paris"** after "The capital of
//! France is". One extra matmul, and attention, rotary positions, the per-head
//! QK norm, grouped-query broadcasting and the causal mask all have to be right
//! to pass it.
//!
//! ```text
//! cargo run --release -p chaos-image --example try-text-encoder
//! ```

use chaos_image::text::{self, TextEncoder};
use chaos_model::Model;
use chaos_tokenizer::Tokenizer;

const PROMPT: &str = "The capital of France is";

fn main() {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    let path = format!("{home}/.chaos/models/Qwen3-VL-8B-Instruct-Q4_K_M.gguf");

    let model = match Model::open_split(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot open {path}: {e}");
            eprintln!("  fetch it with: chaos-pull qwen3-vl-8b");
            std::process::exit(1);
        }
    };
    let tok = match Tokenizer::from_metadata(model.metadata()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer: {e:?}");
            std::process::exit(1);
        }
    };

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let enc = match TextEncoder::open(model, threads) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let c = enc.config;
    println!(
        "encoder      {} blocks, {} wide, {} heads / {} kv of {}",
        c.blocks, c.hidden, c.heads, c.kv_heads, c.head_dim
    );
    println!("rope         theta {:.0}, rms eps {}", c.rope_theta, c.eps);

    let missing = enc.missing();
    if !missing.is_empty() {
        eprintln!(
            "missing {} tensors, first {:?}",
            missing.len(),
            &missing[..1]
        );
        std::process::exit(1);
    }

    // -- the check that cannot be passed by accident --------------------------
    let ids = tok.encode(PROMPT);
    println!("\nprompt       {PROMPT:?} -> {} tokens", ids.len());
    let started = std::time::Instant::now();
    let top = match enc.probe_next(&ids, 5) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("probe: {e}");
            std::process::exit(1);
        }
    };
    println!("took         {:.1}s", started.elapsed().as_secs_f32());
    println!("top 5:");
    for (id, logit) in &top {
        println!("  {:>8}  {:>9.3}  {:?}", id, logit, tok.decode(&[*id]));
    }
    let best = tok.decode(&[top[0].0]);
    let ok = best.trim() == "Paris";
    println!(
        "\n{}",
        if ok {
            "SAYS PARIS -- attention, rotary, QK norm, GQA and the mask are all right"
        } else {
            "DOES NOT SAY PARIS -- this forward pass is wrong somewhere"
        }
    );
    if !ok {
        std::process::exit(1);
    }

    // -- and the thing the denoiser actually consumes -------------------------
    let wrapped = text::wrap_prompt("a red apple on a white table");
    let ids = tok.encode(&wrapped);
    println!("\nconditioning {:?}", wrapped);
    println!("             {} tokens", ids.len());
    let started = std::time::Instant::now();
    let out = match enc.encode(&ids, &mut |i, n| eprint!("\r  block {i}/{n}   ")) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("\nencode: {e}");
            std::process::exit(1);
        }
    };
    eprintln!();
    println!("took         {:.1}s", started.elapsed().as_secs_f32());
    println!(
        "hidden       {} values = {} tokens x {} wide",
        out.hidden.len(),
        out.tokens,
        out.width
    );
    let finite = out.hidden.iter().filter(|v| v.is_finite()).count();
    let rms = (out.hidden.iter().map(|v| v * v).sum::<f32>() / out.hidden.len() as f32).sqrt();
    println!("finite       {finite}/{}", out.hidden.len());
    println!("rms          {rms:.4}");
    if out.width != 53248 {
        eprintln!("WRONG WIDTH -- the denoiser's llm_cond_proj wants 53248");
        std::process::exit(1);
    }
    if finite != out.hidden.len() {
        eprintln!("NOT FINITE");
        std::process::exit(1);
    }
    println!("\nwidth matches the denoiser's llm_cond_proj.");
}
