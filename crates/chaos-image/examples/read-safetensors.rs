//! Read a safetensors header and report what is in it.
//!
//! Works on a partial file — a header-only range fetch is enough, which is how
//! the FLUX.2 autoencoder's table was inspected before the 300 MB of data had
//! arrived.

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: read-safetensors <file.safetensors>");
            std::process::exit(2);
        }
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

    println!("tensors      {}", st.entries().len());
    println!("data begins  {} bytes in", st.data_offset());
    for (k, v) in st.metadata() {
        println!("metadata     {k} = {v}");
    }

    let mut by_dtype: Vec<(String, usize, u64)> = Vec::new();
    for e in st.entries() {
        let key = format!("{:?}", e.dtype);
        match by_dtype.iter_mut().find(|(k, _, _)| *k == key) {
            Some((_, n, els)) => {
                *n += 1;
                *els += e.elements();
            }
            None => by_dtype.push((key, 1, e.elements())),
        }
    }
    for (d, n, els) in &by_dtype {
        println!("{d:12} {n} tensors, {els} elements");
    }

    println!("\nfirst eight entries:");
    for e in st.entries().iter().take(8) {
        println!(
            "  {:44} {:?} {:?} at {}..{}",
            e.name, e.dtype, e.shape, e.start, e.end
        );
    }

    // Only the header may be present, so this is a report rather than a check.
    let present = st
        .entries()
        .iter()
        .filter(|e| st.bytes_of(&bytes, e).is_some())
        .count();
    println!(
        "\n{present} of {} tensors have their data in this file",
        st.entries().len()
    );
}
