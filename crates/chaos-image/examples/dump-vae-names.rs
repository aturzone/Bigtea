//! Every tensor name and shape in a safetensors file, sorted.
//!
//! `read-safetensors` reports the *summary* — counts, dtypes, the first eight
//! entries — which is what a partial download needs. This prints the whole
//! table, which is what porting an architecture needs: the autoencoder's block
//! structure, where its channel counts change and therefore which resnets carry
//! a `conv_shortcut` were all read out of this, rather than assumed from
//! diffusers' defaults and discovered wrong later.
//!
//! ```text
//! cargo run --release -p chaos-image --example dump-vae-names -- flux2-vae.safetensors
//! ```

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: dump-vae-names <file.safetensors>");
        std::process::exit(2);
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(1);
        }
    };
    let st = match chaos_image::safetensors::SafeTensors::parse(&bytes) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    };

    // Sorted, because the file's own order groups by nothing in particular and
    // the question is always "what does block 2 have that block 0 does not".
    let mut lines: Vec<String> = st
        .entries()
        .iter()
        .map(|e| format!("{} {:?} {:?}", e.name, e.shape, e.dtype))
        .collect();
    lines.sort();
    for line in &lines {
        println!("{line}");
    }
    println!("total {}", lines.len());
}
