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
