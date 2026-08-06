//! Dump a container's metadata keys and scalar values.
//!
//! Usage: `bigtea-meta <shard.gguf> [filter]`
//!
//! Arrays print only their length and first element — a tokenizer vocabulary
//! is 160k entries and nobody wants it on their terminal.

use std::process::ExitCode;

use bigtea_gguf::Value;
use bigtea_model::Model;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: bigtea-meta <shard.gguf> [filter]");
        return ExitCode::from(2);
    };
    let second = args.next().unwrap_or_default();
    let want_tensors = second == "--tensors";
    let filter = if want_tensors {
        args.next().unwrap_or_default().to_lowercase()
    } else {
        second.to_lowercase()
    };

    let model = match Model::open_split(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bigtea-meta: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Porting an architecture starts with knowing what tensors it actually
    // ships, which the metadata keys do not tell you.
    if want_tensors {
        let mut names: Vec<&str> = model
            .tensor_names()
            .filter(|n| filter.is_empty() || n.to_lowercase().contains(&filter))
            .collect();
        names.sort_unstable();
        for name in &names {
            let loc = model.location(name).expect("listed");
            let dims: Vec<String> = loc.dims.iter().map(|d| d.to_string()).collect();
            println!(
                "{name:<44} {:<10} [{}]{}",
                format!("{:?}", loc.ty),
                dims.join(", "),
                if loc.routed_expert { "  routed" } else { "" }
            );
        }
        println!(
            "\n{} tensors shown of {}",
            names.len(),
            model.tensor_count()
        );
        return ExitCode::SUCCESS;
    }

    for (key, value) in model.metadata() {
        if !filter.is_empty() && !key.to_lowercase().contains(&filter) {
            continue;
        }
        match value {
            Value::Array(items) => {
                let first = items
                    .first()
                    .map(render_short)
                    .unwrap_or_else(|| "-".into());
                println!("{key:<44} [array; {} items] first={first}", items.len());
            }
            other => println!("{key:<44} {}", render_short(other)),
        }
    }
    ExitCode::SUCCESS
}

fn render_short(v: &Value) -> String {
    match v {
        Value::String(s) if s.len() > 60 => format!("{:?}...", &s[..60]),
        Value::String(s) => format!("{s:?}"),
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => format!("{x}"),
        Value::F64(x) => format!("{x}"),
        Value::Bool(b) => b.to_string(),
        Value::Array(items) => format!("[array; {} items]", items.len()),
    }
}
