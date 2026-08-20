//! Every tensor name, shape and type in a GGUF, plus its metadata keys.
//!
//! `chaos-meta` answers "what model is this"; this answers "what is *in* it",
//! which is the question porting an architecture asks. It exists because
//! `ideogram4-Q4_0.gguf` has **zero metadata keys** — there is no
//! `general.architecture`, no layer count, no head count, nothing. Every number
//! the denoiser needs has to be read off the tensor index itself, so the index
//! has to be printable.
//!
//! ```text
//! cargo run --release -p chaos-gguf --example dump-index -- model.gguf
//! cargo run --release -p chaos-gguf --example dump-index -- model.gguf blk.0.
//! ```
//!
//! A second argument filters by name prefix, because 458 tensors do not fit on
//! a screen and the interesting question is usually one block.

use std::io::Read;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: dump-index <model.gguf> [name-filter]");
        std::process::exit(2);
    };
    let filter = args.next().unwrap_or_default();

    // Only the header is wanted, and the file may be gigabytes. Read a chunk
    // and grow it if the index does not fit -- the same doubling `chaos-model`
    // uses, for the same reason.
    let mut cap = 4 << 20;
    let gguf = loop {
        let mut buf = vec![0u8; cap];
        let mut f = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("cannot open {path}: {e}");
                std::process::exit(1);
            }
        };
        let n = f.read(&mut buf).unwrap_or(0);
        buf.truncate(n);
        match chaos_gguf::Gguf::parse(&buf) {
            Ok(g) => break g,
            Err(e) if cap < (256 << 20) => {
                cap *= 2;
                let _ = e;
            }
            Err(e) => {
                eprintln!("{path}: {e:?}");
                std::process::exit(1);
            }
        }
    };

    println!("version      {}", gguf.version);
    println!("tensors      {}", gguf.tensors.len());
    println!("metadata     {} keys", gguf.metadata.len());
    for (k, v) in gguf.metadata.iter() {
        let shown = match v.as_str() {
            Some(s) if s.len() > 60 => format!("{}...", &s[..60]),
            Some(s) => s.to_string(),
            None => format!("{v:?}").chars().take(60).collect(),
        };
        println!("  {k} = {shown}");
    }

    println!("\ntensors:");
    let mut shown = 0;
    for t in &gguf.tensors {
        if !filter.is_empty() && !t.name.contains(&filter) {
            continue;
        }
        println!(
            "  {:52} {:?} {}",
            t.name,
            t.dims,
            t.ty.name().unwrap_or("?")
        );
        shown += 1;
    }
    println!("shown {shown} of {}", gguf.tensors.len());
}
