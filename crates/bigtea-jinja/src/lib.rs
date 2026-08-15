//! The Jinja subset GGUF chat templates actually use — and a refusal for
//! everything else.
//!
//! # Why this is not a Jinja engine
//!
//! It deliberately is not. `chat.rs` has carried the same warning since it was
//! written:
//!
//! > Evaluating Jinja properly means a whole expression language, and **a
//! > half-implemented one silently produces the wrong framing**, which is the
//! > failure mode this project is most expensive at.
//!
//! That is still true, and it is the reason this crate exists in the shape it
//! does. A wrong chat framing does not error. The model answers, fluently,
//! having been handed a prompt shape it has never seen — it comments on the
//! question instead of answering it, or answers the system prompt. No test that
//! checks "did it produce a string" can see that.
//!
//! So the contract is inverted from a normal template engine's: **anything this
//! crate does not fully understand is an error**, and the caller falls back to
//! the family matcher in `bigtea-tokenizer`, whose 54 renderers are verified
//! byte-identical to llama.cpp for 52 of them. Refusing loudly loses nothing;
//! guessing loses the answer.
//!
//! # Why the subset is this one
//!
//! Not guessed. Every `tokenizer.chat_template` on disk was censused — 12
//! templates — and this is the whole language they use:
//!
//! ```text
//! if/endif 123 · set 98 · else 40 · for/endfor 31 · elif 21
//! loop.index0 20 · loop.last 12 · loop.first 10
//! namespace() 10 · raise_exception() 6 · strftime_now() 1
//! filters: tojson 15, trim 6, length 5
//! operators: in, not, is defined, is string, is not none
//! ```
//!
//! No macros, no imports, no inheritance, three filters. That bound is what
//! makes "refuse everything else" a workable policy rather than a permanent
//! fallback.

use std::collections::HashMap;
use std::fmt;

/// A value flowing through a template.
///
/// Deliberately small. A chat template's environment is a list of maps, a few
/// strings, and a bool — anything richer would be this crate accepting more
/// than it can be checked against.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Bool(bool),
    List(Vec<Value>),
    Map(HashMap<String, Value>),
    /// Jinja's `none`, and the thing `is defined` asks about.
    None,
}

impl Value {
    /// Jinja truthiness: empty string, empty list, zero and `none` are false.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Str(s) => !s.is_empty(),
            Value::Int(i) => *i != 0,
            Value::List(l) => !l.is_empty(),
            Value::Map(m) => !m.is_empty(),
            Value::None => false,
        }
    }

    /// How a value prints inside `{{ }}`.
    pub fn render(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::Int(i) => i.to_string(),
            // Jinja prints Python's spelling, and a template that concatenates
            // this into a prompt would otherwise emit `true` where the model
            // was trained on `True`.
            Value::Bool(b) => (if *b { "True" } else { "False" }).to_string(),
            Value::None => "None".to_string(),
            Value::List(_) | Value::Map(_) => json(self),
        }
    }
}

/// Minimal JSON, for the `tojson` filter.
pub(crate) fn json_public(v: &Value) -> String {
    json(v)
}

fn json(v: &Value) -> String {
    match v {
        Value::Str(s) => {
            let mut out = String::from("\"");
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        }
        Value::Int(i) => i.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::None => "null".to_string(),
        Value::List(l) => {
            let items: Vec<String> = l.iter().map(json).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Map(m) => {
            // Sorted, so a rendered template is byte-stable across runs. A
            // HashMap's order is not, and a prompt that changes between runs
            // would make every parity comparison meaningless.
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let items: Vec<String> = keys
                .iter()
                .map(|k| format!("{}: {}", json(&Value::Str((*k).clone())), json(&m[*k])))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// Does this template ever look at a `system` turn?
///
/// # Why a caller needs to know
///
/// Phi-3's template handles `user` and `assistant` and **silently drops
/// anything else**. Render a conversation with a system turn through it and the
/// system prompt simply vanishes — no error, and a model that ignores its
/// instructions for a reason nothing reports.
///
/// llama.cpp does not fix this in the template. It fixes it *before* rendering:
/// when the template has no system support it merges the system content into
/// the first user turn. `--jinja` on Phi-3 emits
/// `<|user|> SYS\nHI<|end|>`, where the template alone would emit
/// `<|user|>\nHI<|end|>`.
///
/// So evaluating the template correctly is not sufficient to match the
/// reference — the caller has to apply the same polyfill, and to do that it has
/// to ask this question first.
/// Does this template actually *accept* a system turn?
///
/// **Decided by rendering one and looking for it, not by reading the source.**
/// A template can refuse a system turn in two completely different ways, and
/// only one of them is visible lexically:
///
/// ```text
/// gemma-2:  raises   {{ raise_exception('System role not supported') }}
/// Phi-3:    DROPS IT silently -- no system branch, no error, no output
/// ```
///
/// So neither a lexical scan nor a check for [`Error::Raised`] is enough. The
/// first was measured wrong on gemma-2, which names the role only to reject it;
/// the second was measured wrong on Phi-3, which says nothing at all and simply
/// loses the turn. Both produce a prompt the model was never trained on, and
/// neither fails loudly.
///
/// The question that survives both is: **did the system content come out the
/// other side?** That is what the caller actually needs to know before deciding
/// whether to merge the system turn into the first user message.
///
/// llama.cpp settles it behaviourally too. A template that rejects or drops a
/// system turn is not broken — that is how a template *says* it has no system
/// slot — and the correct response is the merge polyfill, not an error.
pub fn supports_system_role(template: &str) -> bool {
    let Ok(nodes) = parse(template) else {
        // Not our question to answer; let the real render report it.
        return true;
    };
    // Distinctive enough that it cannot appear from the template's own text.
    const PROBE: &str = "ZQSYSPROBEQZ";
    let mut env = Env::new();
    let mut sys = std::collections::HashMap::new();
    sys.insert("role".to_string(), Value::Str("system".to_string()));
    sys.insert("content".to_string(), Value::Str(PROBE.to_string()));
    let mut usr = std::collections::HashMap::new();
    usr.insert("role".to_string(), Value::Str("user".to_string()));
    usr.insert("content".to_string(), Value::Str("U".to_string()));
    env.set(
        "messages",
        Value::List(vec![Value::Map(sys), Value::Map(usr)]),
    );
    env.set("bos_token", Value::Str(String::new()));
    env.set("eos_token", Value::Str(String::new()));
    env.set("add_generation_prompt", Value::Bool(true));

    match render(&nodes, &mut env) {
        // Rendered: supported only if the content actually survived.
        Ok(out) => out.contains(PROBE),
        // Raised: the template said no in the loud way.
        Err(Error::Raised(_)) => false,
        // Anything else is a different problem -- an unsupported construct, an
        // undefined variable -- and must not be reported as "no system role".
        Err(_) => true,
    }
}

pub fn mentions_system_role(template: &str) -> bool {
    // Naming the role literally is the obvious way to support it.
    if template.contains("'system'") || template.contains("\"system\"") {
        return true;
    }
    // **And the non-obvious way: emitting the role instead of testing it.**
    // ChatML templates write
    //
    //   {{ '<|im_start|>' + message['role'] + '\n' + message['content'] ... }}
    //
    // which handles *every* role, system included, without the word appearing
    // anywhere. internlm2 does exactly this, and asking only for the literal
    // said "no system branch" — so the system turn was merged into the user
    // turn on a template that would have rendered it correctly, and llama.cpp
    // emitted three turns where we emitted two.
    //
    // **The role has to be OUTPUT, not merely compared.** Phi-3 also contains
    // `['role']`, in `{% if message['role'] == 'user' %}` conditions, and it
    // genuinely has no system branch — it must still be merged. Testing for the
    // substring anywhere would fix internlm2 by breaking Phi-3, which is the
    // same trade the old test made in the other direction.
    output_blocks(template).any(|b| b.contains("['role']") || b.contains("[\"role\"]"))
}

/// The contents of each `{{ … }}` block, which is where a template *emits*
/// rather than *decides*.
fn output_blocks(template: &str) -> impl Iterator<Item = &str> {
    let mut rest = template;
    std::iter::from_fn(move || {
        let open = rest.find("{{")?;
        let after = &rest[open + 2..];
        // An unterminated `{{` is a broken template; the parser will say so
        // properly. Nothing more to scan here.
        let close = after.find("}}")?;
        let block = &after[..close];
        rest = &after[close + 2..];
        Some(block)
    })
}

/// Merge the system turn into the first user turn, llama.cpp's polyfill.
///
/// Returns the messages unchanged when there is no system turn or no user turn
/// to merge it into — dropping it on the floor would be the very failure this
/// exists to prevent.
pub fn merge_system_into_first_user(messages: &[Value], separator: &str) -> Vec<Value> {
    let system: Option<String> = messages.iter().find_map(|m| match m {
        Value::Map(f) if matches!(f.get("role"), Some(Value::Str(r)) if r == "system") => {
            f.get("content").map(|c| c.render())
        }
        _ => None,
    });
    let Some(system) = system else {
        return messages.to_vec();
    };
    let mut out = Vec::with_capacity(messages.len());
    let mut merged = false;
    for m in messages {
        match m {
            Value::Map(f) if matches!(f.get("role"), Some(Value::Str(r)) if r == "system") => {}
            Value::Map(f)
                if !merged && matches!(f.get("role"), Some(Value::Str(r)) if r == "user") =>
            {
                let mut f = f.clone();
                let body = f.get("content").map(|c| c.render()).unwrap_or_default();
                f.insert(
                    "content".to_string(),
                    Value::Str(format!("{system}{separator}{body}")),
                );
                out.push(Value::Map(f));
                merged = true;
            }
            other => out.push(other.clone()),
        }
    }
    // No user turn to merge into: keep the system turn rather than lose it.
    if merged {
        out
    } else {
        messages.to_vec()
    }
}

/// Why a template could not be rendered.
///
/// Every variant is a *refusal to guess*. `Unsupported` in particular is the
/// crate's whole safety property: it is what sends the caller back to the
/// family matcher rather than to a plausible-looking wrong answer.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A construct outside the censused subset. Carries what it was, so the
    /// message names the thing rather than saying "parse error".
    Unsupported(String),
    Syntax(String),
    /// The template itself called `raise_exception`. **Not a bug**: templates
    /// use it to reject conversations they cannot express, and swallowing it
    /// produces exactly the framing they exist to prevent.
    Raised(String),
    UndefinedVariable(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(what) => write!(
                f,
                "unsupported Jinja construct: {what}. This build evaluates only the subset \
                 GGUF chat templates use; the family matcher will be used instead."
            ),
            Error::Syntax(what) => write!(f, "template syntax: {what}"),
            Error::Raised(msg) => write!(f, "template rejected this conversation: {msg}"),
            Error::UndefinedVariable(name) => write!(f, "undefined variable `{name}`"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

mod lex;
mod parse;
mod render;

pub use lex::{Token, TokenKind};
pub use parse::{parse, Node};
pub use render::{render, Env};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_template_without_a_system_branch_is_detected() {
        // Phi-3's, trimmed. It handles user and assistant and drops the rest.
        let phi3 = "{% for m in messages %}{% if (m['role'] == 'user') %}x{% endif %}{% endfor %}";
        assert!(!mentions_system_role(phi3));
        let chatml =
            "{% for m in messages %}{% if m['role'] == 'system' %}y{% endif %}{% endfor %}";
        assert!(mentions_system_role(chatml));
    }

    #[test]
    fn a_template_that_emits_the_role_supports_every_role_including_system() {
        // internlm2's, and every ChatML template's: the role is INTERPOLATED,
        // so system is handled without the word appearing anywhere. Asking only
        // for the literal reported "no system branch", merged the system turn
        // into the user turn, and produced two turns where llama.cpp produced
        // three.
        let chatml = "{% for m in messages %}{{'<|im_start|>' + m['role'] + '\\n' \
                      + m['content'] + '<|im_end|>'}}{% endfor %}";
        assert!(mentions_system_role(chatml));
        assert!(mentions_system_role(
            "{% for m in messages %}{{ m[\"role\"] }}{% endfor %}"
        ));
    }

    #[test]
    fn the_role_must_be_emitted_not_merely_compared() {
        // The whole difficulty. Phi-3 also contains `['role']` -- inside an
        // `{% if %}` condition -- and genuinely has no system branch, so it
        // must still be merged. A substring test anywhere in the template would
        // fix internlm2 by breaking Phi-3, which is the same trade the old
        // version made in the other direction.
        let compares_only =
            "{% for m in messages %}{% if m['role'] == 'user' %}x{% endif %}{% endfor %}";
        assert!(!mentions_system_role(compares_only));
    }

    #[test]
    fn an_unterminated_output_block_does_not_hang_or_panic() {
        // A broken template is the parser's business to report properly; this
        // must not loop forever looking for a `}}` that never arrives.
        assert!(!mentions_system_role("{{ m['role']"));
        assert!(!mentions_system_role("{{"));
    }

    #[test]
    fn the_system_turn_is_merged_rather_than_dropped() {
        let mk = |role: &str, content: &str| {
            let mut m = HashMap::new();
            m.insert("role".to_string(), Value::Str(role.into()));
            m.insert("content".to_string(), Value::Str(content.into()));
            Value::Map(m)
        };
        let msgs = vec![mk("system", "SYS"), mk("user", "HI")];
        let out = merge_system_into_first_user(&msgs, "\n");
        assert_eq!(out.len(), 1);
        let Value::Map(f) = &out[0] else { panic!() };
        assert_eq!(f["role"], Value::Str("user".into()));
        assert_eq!(f["content"], Value::Str("SYS\nHI".into()));
    }

    #[test]
    fn with_no_user_turn_the_system_turn_survives() {
        // Losing it would be the exact failure the polyfill exists to prevent.
        let mut m = HashMap::new();
        m.insert("role".to_string(), Value::Str("system".into()));
        m.insert("content".to_string(), Value::Str("SYS".into()));
        let msgs = vec![Value::Map(m)];
        assert_eq!(merge_system_into_first_user(&msgs, "\n").len(), 1);
    }

    #[test]
    fn truthiness_follows_jinja_not_rust() {
        assert!(!Value::Str(String::new()).truthy());
        assert!(Value::Str("x".into()).truthy());
        assert!(!Value::List(vec![]).truthy());
        assert!(!Value::None.truthy());
        assert!(!Value::Int(0).truthy());
    }

    #[test]
    fn a_bool_renders_pythons_spelling() {
        // A template that writes `{{ add_generation_prompt }}` into a prompt
        // must produce `True`, not `true` -- the model was trained on the
        // former and a one-character difference is a different token.
        assert_eq!(Value::Bool(true).render(), "True");
        assert_eq!(Value::None.render(), "None");
    }

    #[test]
    fn tojson_escapes_and_sorts() {
        let mut m = HashMap::new();
        m.insert("b".to_string(), Value::Int(2));
        m.insert("a".to_string(), Value::Str("x\"y".into()));
        // Sorted: a HashMap's order is not stable, and an unstable prompt makes
        // every parity comparison meaningless.
        assert_eq!(json(&Value::Map(m)), r#"{"a": "x\"y", "b": 2}"#);
    }
}
