//! Does the prompt change what the denoiser predicts, and by how much?
//!
//! The pipeline can produce a coherent image and still ignore what it was asked
//! for — that is what "a red apple on a white table" drawing a dark stone wall
//! looks like. The image alone cannot tell you whether the conditioning is
//! *dead* or merely *weak*, so this asks the model directly: run the
//! conditional denoiser on one latent under two very different prompts and
//! measure how far apart the two answers are.
//!
//! ```text
//! cargo run --release -p chaos-image --example try-conditioning
//! ```
//!
//! | cosine between the two | meaning |
//! |---|---|
//! | 1.0000 | the text is not reaching the model at all |
//! | ~0.99 | it reaches it and barely matters |
//! | well below | the prompt is doing real work |
//!
//! The unconditional twin is run too, as a reference point: it is a *different
//! set of weights*, so the distance between it and either conditional answer is
//! the scale on which "far apart" should be read.

use chaos_image::pipeline::Noise;
use chaos_image::{dit, flow, text, text::TextEncoder};
use chaos_model::Model;
use chaos_tokenizer::Tokenizer;

const A: &str = "a red apple on a white table";
const B: &str = "a snowy mountain range at sunrise, wide landscape photograph";

fn main() {
    let grid: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let sigma: f32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.7);

    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    let dir = std::path::Path::new(&home).join(".chaos").join("models");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());

    // -- two prompts through the text encoder --------------------------------
    let (ca, cb) = {
        let model = Model::open_split(dir.join("Qwen3-VL-8B-Instruct-Q4_K_M.gguf"))
            .expect("open the text encoder");
        let tok = Tokenizer::from_metadata(model.metadata()).expect("tokenizer");
        let enc = TextEncoder::open(model, threads).expect("text encoder");
        let run = |p: &str| {
            let ids = tok.encode(&text::wrap_prompt(p));
            let out = enc.encode(&ids, &mut |_, _| {}).expect("encode");
            println!("  {:?} -> {} tokens", p, out.tokens);
            out
        };
        println!("prompts:");
        (run(A), run(B))
    };

    let cos = |a: &[f32], b: &[f32]| -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb)
    };

    // The conditioning vectors themselves must differ before anything
    // downstream can. Same token count would be a coincidence; different is
    // normal, and then only the common prefix is comparable.
    if ca.tokens == cb.tokens {
        println!("\nconditioning cosine {:.6}", cos(&ca.hidden, &cb.hidden));
    } else {
        println!("\n(different token counts, so the vectors are not comparable)");
    }

    // -- one latent, three answers -------------------------------------------
    let c = dit::Config::default();
    let n = (grid * grid * c.in_channels) as usize;
    let x = Noise::seeded(7).normals(n);
    let t = flow::timestep_for(sigma);
    println!("\nlatent       {grid}x{grid} grid, sigma {sigma}, timestep {t:.0}");

    let cond = dit::Denoiser::open(
        Model::open_split(dir.join("ideogram4-Q4_0.gguf")).expect("open the denoiser"),
        threads,
    );
    let go = |ctx: &[f32], len: usize| {
        cond.forward(&dit::Inputs {
            latent: &x,
            grid_w: grid,
            grid_h: grid,
            timestep: t,
            context: ctx,
            context_len: len,
        })
        .expect("forward")
    };
    println!("running the conditional model on prompt A ...");
    let va = go(&ca.hidden, ca.tokens);
    println!("running the conditional model on prompt B ...");
    let vb = go(&cb.hidden, cb.tokens);

    println!("running the unconditional twin ...");
    let uncond = dit::Denoiser::open(
        Model::open_split(dir.join("ideogram4_uncond-Q4_0.gguf")).expect("open the twin"),
        threads,
    );
    let vu = uncond
        .forward(&dit::Inputs {
            latent: &x,
            grid_w: grid,
            grid_h: grid,
            timestep: t,
            context: &[],
            context_len: 0,
        })
        .expect("forward");

    println!(
        "\n  cos(A, B)      {:.6}   <- two prompts, same weights",
        cos(&va, &vb)
    );
    println!("  cos(A, uncond) {:.6}", cos(&va, &vu));
    println!("  cos(B, uncond) {:.6}", cos(&vb, &vu));

    let ab = cos(&va, &vb);
    println!();
    if ab > 0.99999 {
        println!("THE PROMPT DOES NOTHING -- the text never reaches the model.");
        std::process::exit(1);
    } else if ab > 0.99 {
        println!("The prompt reaches the model and barely moves it.");
    } else {
        println!("The prompt is doing real work.");
    }
}
