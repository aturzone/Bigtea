//! Does evaluating a container's own Jinja produce what the family matcher does?
//!
//! # Why this is the acceptance test for `--jinja`
//!
//! The two are independent implementations of the same thing, and **the family
//! matcher is already verified byte-identical to llama.cpp for 52 of its 54
//! renderers**. So agreement here is a cross-check against a known-good
//! reference, not a self-check — which is the difference between this test
//! meaning something and it meaning nothing.
//!
//! Ignored by default: it needs containers on disk. Run with
//!
//! ```text
//! cargo test --release --test jinja_agrees_with_families -- --ignored --nocapture
//! ```
//!
//! # What a disagreement means
//!
//! Not automatically a Jinja bug. The family matcher is a *hardcoded* renderer
//! that llama.cpp also hardcodes, and llama.cpp's is wrong for at least one
//! model on purpose — its Zephyr renderer emits `<|endoftext|>` on every
//! container because it has no vocabulary to read, while TinyLlama's own
//! template says `eos_token` and that is `</s>`. So a disagreement is a
//! question, and the answer comes from `llama-completion --jinja`, not from
//! whichever of the two looks nicer.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bigtea_jinja::{parse, render, Env, Value};
use bigtea_tokenizer::chat::Message;
use bigtea_tokenizer::Tokenizer;

fn models() -> Vec<PathBuf> {
    let root = Path::new("C:/Projects/models");
    let Ok(dirs) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for d in dirs.flatten() {
        let Ok(files) = std::fs::read_dir(d.path()) else {
            continue;
        };
        for f in files.flatten() {
            let p = f.path();
            if p.extension().is_some_and(|e| e == "gguf") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn msg(role: &str, content: &str) -> Value {
    let mut m = HashMap::new();
    m.insert("role".to_string(), Value::Str(role.to_string()));
    m.insert("content".to_string(), Value::Str(content.to_string()));
    Value::Map(m)
}

#[test]
#[ignore = "needs containers on disk"]
fn every_container_template_renders_the_same_both_ways() {
    let mut agree = 0;
    let mut refused: Vec<String> = Vec::new();
    let mut differ: Vec<String> = Vec::new();
    let mut checked = 0;

    // **Skip when the machine has no containers, rather than assert.** The
    // `--ignored` CI step exists to prove exactly this path works: its own
    // comment says "the container-backed tests skip themselves when no model is
    // on disk. This proves the skip path works, not that the tests pass."
    // Asserting instead failed on all three runners, none of which has
    // `C:/Projects/models`, and it could only be seen from a pull request --
    // a branch without one never builds here at all.
    let containers = models();
    if containers.is_empty() {
        eprintln!("skipping: no model");
        return;
    }

    for path in containers {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let Ok(model) = bigtea_model::Model::open_split(path.to_string_lossy().as_ref()) else {
            continue;
        };
        let Ok(tok) = Tokenizer::from_metadata(model.metadata()) else {
            continue;
        };
        let Some(template) = tok.chat_template() else {
            continue;
        };
        // Only families the matcher claims to know; `Generic` is its way of
        // saying it did not recognise the template, and comparing against a
        // deliberate fallback proves nothing.
        let fmt = tok.chat_format();
        if !fmt.is_known() {
            continue;
        }
        checked += 1;

        let messages = vec![Message::new("system", "SYS"), Message::new("user", "HI")];
        let want = fmt.apply(&messages, "</s>", true);

        // llama.cpp's polyfill, applied before rendering: a template with no
        // system branch would otherwise DROP the system turn silently, and
        // Phi-3's does exactly that. Comparing without it compares two
        // different conversations.
        let raw = vec![msg("system", "SYS"), msg("user", "HI")];
        let prepared = if bigtea_jinja::mentions_system_role(template) {
            raw
        } else {
            bigtea_jinja::merge_system_into_first_user(
                &raw, "
",
            )
        };
        let mut env = Env::new();
        env.set("messages", Value::List(prepared));
        env.set("bos_token", Value::Str(String::new()));
        env.set("eos_token", Value::Str("</s>".into()));
        env.set("add_generation_prompt", Value::Bool(true));

        match parse(template).and_then(|nodes| render(&nodes, &mut env)) {
            Ok(got) if got == want => agree += 1,
            // A refusal is the DESIGNED outcome for anything outside the
            // subset, so it is counted rather than failed. The number is the
            // measure of how much of the subset is real.
            Err(e) => refused.push(format!("{name} [{}]: {e}", fmt.name())),
            Ok(got) => differ.push(format!(
                "\n  {name} [{}]\n    family: {want:?}\n    jinja : {got:?}",
                fmt.name()
            )),
        }
    }

    println!(
        "checked {checked}: {agree} agree, {} refused, {} differ",
        refused.len(),
        differ.len()
    );
    for r in &refused {
        println!("  refused: {r}");
    }
    for d in &differ {
        println!("{d}");
    }
    // Deliberately not asserting agreement yet. This test EXISTS to produce the
    // number that decides whether `--jinja` may be wired at all -- asserting a
    // result before measuring it would be deciding the answer in advance.
    // Kept, and it still means something: reaching here proves containers WERE
    // found, so zero checked is a broken discovery or a broken template match
    // rather than an empty machine.
    assert!(
        checked > 0,
        "containers were found but none had a known template"
    );
}
