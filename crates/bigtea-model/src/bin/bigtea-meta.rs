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
    let filter = args.next().unwrap_or_default().to_lowercase();

    let model = match Model::open_split(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bigtea-meta: {e}");
            return ExitCode::FAILURE;
        }
    };

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
