//! JSON Schema to GBNF, for `--json-schema` and `--json-schema-file`.
//!
//! # Why the primitive rules are copied verbatim
//!
//! The built-in rules below (`string`, `number`, `object`, `space`, …) are
//! character-for-character llama.cpp's, from `common/json-schema-to-grammar.cpp`.
//! They are not obvious and they are not arbitrary — `char` in particular
//! excludes `\x7F` and `\x00-\x1F` because JSON forbids raw control characters
//! in strings, and `integral-part` is `[0] | [1-9] [0-9]{0,15}` rather than
//! `[0-9]+` because JSON forbids leading zeros. Rewriting them "more clearly"
//! is how you get a grammar that accepts `01` or a literal newline inside a
//! string, and the model's output then fails to parse in the caller.
//!
//! # What is refused, and why refusing is the safe direction
//!
//! An unimplemented keyword is **refused by name**, never ignored. Ignoring a
//! constraint yields a grammar that is *looser* than the schema asked for —
//! output that satisfies the grammar and violates the schema, which nothing
//! downstream can detect. A refusal is a message; a silent loosening is a bug
//! in someone else's parser three days later.

use std::collections::BTreeMap;

use crate::GrammarError;

/// llama.cpp's `SPACE_RULE`. The leading `|` is an empty alternative: optional
/// whitespace, but bounded, so the model cannot spend its budget on newlines.
const SPACE_RULE: &str = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

/// The built-in rules, verbatim from llama.cpp. Order here is the order they
/// are emitted in when used.
fn primitive(name: &str) -> Option<(&'static str, &'static [&'static str])> {
    Some(match name {
        "boolean" => ("(\"true\" | \"false\")", &[]),
        "decimal-part" => ("[0-9]{1,16}", &[]),
        "integral-part" => ("[0] | [1-9] [0-9]{0,15}", &[]),
        "number" => (
            "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?",
            &["integral-part", "decimal-part"],
        ),
        "integer" => ("(\"-\"? integral-part)", &["integral-part"]),
        "value" => (
            "object | array | string | number | boolean | null",
            &["object", "array", "string", "number", "boolean", "null"],
        ),
        "object" => (
            "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"",
            &["string", "value"],
        ),
        "array" => (
            "\"[\" space ( value (\",\" space value)* )? space \"]\"",
            &["value"],
        ),
        "char" => (
            "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})",
            &[],
        ),
        "string" => ("\"\\\"\" char* \"\\\"\"", &["char"]),
        "null" => ("\"null\"", &[]),
        _ => return None,
    })
}

pub fn to_gbnf(src: &str) -> Result<String, GrammarError> {
    let schema = Json::parse(src)?;
    let mut c = Converter {
        rules: BTreeMap::new(),
        defs: Vec::new(),
        counter: 0,
    };
    // `$defs` and `definitions` are resolved by name, so they are collected
    // before conversion starts -- a `$ref` may point forward.
    if let Some(Json::Obj(entries)) = schema.get("$defs").or_else(|| schema.get("definitions")) {
        c.defs = entries.clone();
    }
    c.rules.insert("space".to_string(), SPACE_RULE.to_string());
    let root = c.visit(&schema, "root")?;
    if root != "root" {
        c.rules.insert("root".to_string(), root);
    }

    // `root` first so the grammar reads top-down; the rest sorted, so the same
    // schema always produces byte-identical output and a diff means something.
    let mut out = String::new();
    if let Some(body) = c.rules.get("root") {
        out.push_str(&format!("root ::= {body}\n"));
    }
    for (name, body) in &c.rules {
        if name != "root" {
            out.push_str(&format!("{name} ::= {body}\n"));
        }
    }
    Ok(out)
}

struct Converter {
    rules: BTreeMap<String, String>,
    defs: Vec<(String, Json)>,
    counter: usize,
}

impl Converter {
    fn add_primitive(&mut self, name: &str) -> Result<String, GrammarError> {
        if self.rules.contains_key(name) {
            return Ok(name.to_string());
        }
        let (body, deps) =
            primitive(name).ok_or_else(|| GrammarError::SchemaUnsupported(name.to_string()))?;
        // Insert before recursing: `value` refers to `object`, which refers
        // back to `value`, and without this the recursion does not terminate.
        self.rules.insert(name.to_string(), body.to_string());
        for dep in deps {
            self.add_primitive(dep)?;
        }
        Ok(name.to_string())
    }

    fn fresh(&mut self, hint: &str) -> String {
        self.counter += 1;
        let hint = hint.replace(|c: char| !c.is_ascii_alphanumeric(), "-");
        format!("{}-{}", hint.trim_matches('-'), self.counter)
    }

    /// Convert `schema` and return a GBNF *expression* for it.
    fn visit(&mut self, schema: &Json, name: &str) -> Result<String, GrammarError> {
        let Json::Obj(_) = schema else {
            // `true` allows anything, `false` allows nothing. Only the former
            // has a sane grammar, and the latter is refused rather than
            // silently turned into "anything".
            return match schema {
                Json::Bool(true) => self.add_primitive("value"),
                other => Err(GrammarError::SchemaUnsupported(format!(
                    "a schema that is {other:?} rather than an object"
                ))),
            };
        };

        for keyword in [
            "allOf",
            "not",
            "if",
            "then",
            "else",
            "patternProperties",
            "pattern",
            "prefixItems",
            "uniqueItems",
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        ] {
            if schema.get(keyword).is_some() {
                return Err(GrammarError::SchemaUnsupported(keyword.to_string()));
            }
        }

        if let Some(r) = schema.get("$ref") {
            return self.visit_ref(r, name);
        }
        if let Some(Json::Arr(values)) = schema.get("enum") {
            let alts: Vec<String> = values.iter().map(json_literal).collect();
            return Ok(format!("({})", alts.join(" | ")));
        }
        if let Some(value) = schema.get("const") {
            return Ok(json_literal(value));
        }
        for key in ["oneOf", "anyOf"] {
            if let Some(Json::Arr(options)) = schema.get(key) {
                let mut alts = Vec::with_capacity(options.len());
                for (i, option) in options.iter().enumerate() {
                    let sub = self.fresh(&format!("{name}-{i}"));
                    let body = self.visit(option, &sub)?;
                    self.rules.insert(sub.clone(), body);
                    alts.push(sub);
                }
                return Ok(format!("({})", alts.join(" | ")));
            }
        }

        let ty = match schema.get("type") {
            None => {
                // No `type` and no combinator: any JSON value.
                return self.add_primitive("value");
            }
            Some(Json::Str(s)) => s.clone(),
            Some(Json::Arr(types)) => {
                // A union of primitive types, e.g. ["string", "null"].
                let mut alts = Vec::new();
                for t in types {
                    let Json::Str(t) = t else {
                        return Err(GrammarError::SchemaUnsupported(
                            "a non-string entry in a `type` array".into(),
                        ));
                    };
                    alts.push(self.add_primitive(t)?);
                }
                return Ok(format!("({})", alts.join(" | ")));
            }
            Some(other) => {
                return Err(GrammarError::SchemaUnsupported(format!(
                    "`type` that is {other:?}"
                )))
            }
        };

        match ty.as_str() {
            "object" => self.visit_object(schema, name),
            "array" => self.visit_array(schema, name),
            other => self.add_primitive(other),
        }
    }

    fn visit_ref(&mut self, r: &Json, name: &str) -> Result<String, GrammarError> {
        let Json::Str(path) = r else {
            return Err(GrammarError::SchemaUnsupported(
                "a non-string `$ref`".into(),
            ));
        };
        let key = path
            .strip_prefix("#/$defs/")
            .or_else(|| path.strip_prefix("#/definitions/"))
            .ok_or_else(|| {
                GrammarError::SchemaUnsupported(format!(
                    "`$ref` to `{path}` -- only `#/$defs/` and `#/definitions/` are resolved"
                ))
            })?;
        let rule = format!(
            "ref-{}",
            key.replace(|c: char| !c.is_ascii_alphanumeric(), "-")
        );
        if self.rules.contains_key(&rule) {
            return Ok(rule);
        }
        let target = self
            .defs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| {
                GrammarError::Schema(format!("`$ref` to `{path}`, which is not defined"))
            })?;
        // Reserve the name before recursing, so a self-referential definition
        // -- a tree node containing children of its own type -- terminates.
        self.rules.insert(rule.clone(), String::new());
        let body = self.visit(&target, name)?;
        self.rules.insert(rule.clone(), body);
        Ok(rule)
    }

    fn visit_object(&mut self, schema: &Json, name: &str) -> Result<String, GrammarError> {
        self.add_primitive("string")?;
        let properties = match schema.get("properties") {
            Some(Json::Obj(p)) => p.clone(),
            _ => Vec::new(),
        };
        if properties.is_empty() {
            // No declared properties: a free-form object.
            return self.add_primitive("object");
        }
        let required: Vec<String> = match schema.get("required") {
            Some(Json::Arr(r)) => r
                .iter()
                .filter_map(|v| match v {
                    Json::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        // `additionalProperties` defaulting to allowed would let the model emit
        // keys the schema never mentioned, which is the loosening this module
        // refuses to do quietly. Anything but an explicit `false` is refused.
        match schema.get("additionalProperties") {
            None | Some(Json::Bool(false)) => {}
            Some(_) => {
                return Err(GrammarError::SchemaUnsupported(
                    "additionalProperties other than `false`".into(),
                ))
            }
        }

        let mut parts: Vec<String> = Vec::new();
        parts.push("\"{\" space".to_string());

        // Required properties come in declaration order, separated by commas.
        // Optional ones each carry their own leading comma, so any subset is
        // expressible without the grammar admitting a trailing comma.
        let mut emitted_required = 0usize;
        for (key, sub_schema) in &properties {
            if !required.contains(key) {
                continue;
            }
            let rule = self.fresh(&format!("{name}-{key}"));
            let body = self.visit(sub_schema, &rule)?;
            self.rules.insert(rule.clone(), body);
            if emitted_required > 0 {
                parts.push("\",\" space".to_string());
            }
            parts.push(format!("{} space \":\" space {rule}", json_key(key)));
            emitted_required += 1;
        }
        for (key, sub_schema) in &properties {
            if required.contains(key) {
                continue;
            }
            let rule = self.fresh(&format!("{name}-{key}"));
            let body = self.visit(sub_schema, &rule)?;
            self.rules.insert(rule.clone(), body);
            let lead = if emitted_required > 0 || !parts.is_empty() {
                "\",\" space "
            } else {
                ""
            };
            parts.push(format!(
                "({lead}{} space \":\" space {rule})?",
                json_key(key)
            ));
        }
        parts.push("space \"}\"".to_string());
        Ok(parts.join(" "))
    }

    fn visit_array(&mut self, schema: &Json, name: &str) -> Result<String, GrammarError> {
        let Some(items) = schema.get("items") else {
            return self.add_primitive("array");
        };
        let item_rule = self.fresh(&format!("{name}-item"));
        let body = self.visit(items, &item_rule)?;
        self.rules.insert(item_rule.clone(), body);

        let min = schema.get("minItems").and_then(Json::as_u64).unwrap_or(0);
        let max = schema.get("maxItems").and_then(Json::as_u64);

        // The tail carries the comma, so an empty array has no comma to drop
        // and a one-element array has none to add.
        let repeat = match (min, max) {
            (0, None) => format!("({item_rule} (\",\" space {item_rule})*)?"),
            (n, None) if n >= 1 => {
                let head = format!("{item_rule} (\",\" space {item_rule})");
                format!("{head}{{{},}}", n.saturating_sub(1))
            }
            (n, Some(m)) => {
                if m == 0 {
                    String::new()
                } else {
                    let tail_min = n.saturating_sub(1);
                    let tail_max = m.saturating_sub(1);
                    let inner =
                        format!("{item_rule} (\",\" space {item_rule}){{{tail_min},{tail_max}}}");
                    if n == 0 {
                        format!("({inner})?")
                    } else {
                        inner
                    }
                }
            }
            _ => unreachable!("min is u64 and the (0, None) case is handled above"),
        };
        Ok(format!("\"[\" space {repeat} space \"]\""))
    }
}

/// A JSON value as a GBNF literal, for `const` and `enum`.
fn json_literal(v: &Json) -> String {
    gbnf_string(&v.to_json_text())
}

/// An object key as it appears **in the output** — quoted.
///
/// The quotes are the whole point: a property key in JSON is a string, so the
/// grammar has to match `"name"` and not `name`. Emitting the bare word is not
/// a parse error in the grammar, it just constrains the model to produce
/// unquoted keys, and every object the schema was meant to guarantee comes out
/// invalid.
fn json_key(key: &str) -> String {
    gbnf_string(&Json::Str(key.to_string()).to_json_text())
}

/// Wrap `s` as a GBNF string literal, escaping what GBNF treats specially.
fn gbnf_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------------------
// A JSON reader.
//
// The workspace has no external dependencies, so the schema is parsed here.
// Objects keep their entries in a `Vec` rather than a map because **property
// order is part of the grammar** -- required properties are emitted in
// declaration order, and a hash map would reorder them per run, making the
// generated grammar non-deterministic.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(src: &str) -> Result<Json, GrammarError> {
        let bytes: Vec<char> = src.chars().collect();
        let mut p = JsonParser { s: &bytes, at: 0 };
        p.space();
        let v = p.value()?;
        p.space();
        if p.at != bytes.len() {
            return Err(GrammarError::Schema(
                "trailing content after the JSON value".into(),
            ));
        }
        Ok(v)
    }

    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
            _ => None,
        }
    }

    /// Render back to JSON text, for `const`/`enum` literals.
    fn to_json_text(&self) -> String {
        match self {
            Json::Null => "null".into(),
            Json::Bool(b) => b.to_string(),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            Json::Str(s) => {
                let mut out = String::from("\"");
                for c in s.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        _ => out.push(c),
                    }
                }
                out.push('"');
                out
            }
            Json::Arr(items) => format!(
                "[{}]",
                items
                    .iter()
                    .map(Json::to_json_text)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Json::Obj(entries) => format!(
                "{{{}}}",
                entries
                    .iter()
                    .map(|(k, v)| format!(
                        "{}:{}",
                        Json::Str(k.clone()).to_json_text(),
                        v.to_json_text()
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

struct JsonParser<'a> {
    s: &'a [char],
    at: usize,
}

impl JsonParser<'_> {
    fn space(&mut self) {
        while matches!(self.s.get(self.at), Some(' ' | '\t' | '\n' | '\r')) {
            self.at += 1;
        }
    }

    fn eat(&mut self, c: char) -> bool {
        if self.s.get(self.at) == Some(&c) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn word(&mut self, w: &str) -> bool {
        if self.s[self.at..].starts_with(&w.chars().collect::<Vec<_>>()[..]) {
            self.at += w.chars().count();
            true
        } else {
            false
        }
    }

    fn err(&self, what: &str) -> GrammarError {
        GrammarError::Schema(format!("{what} at character {}", self.at))
    }

    fn value(&mut self) -> Result<Json, GrammarError> {
        self.space();
        match self.s.get(self.at) {
            None => Err(self.err("unexpected end of input")),
            Some('n') if self.word("null") => Ok(Json::Null),
            Some('t') if self.word("true") => Ok(Json::Bool(true)),
            Some('f') if self.word("false") => Ok(Json::Bool(false)),
            Some('"') => Ok(Json::Str(self.string()?)),
            Some('[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.space();
                if self.eat(']') {
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    self.space();
                    if self.eat(',') {
                        continue;
                    }
                    if self.eat(']') {
                        return Ok(Json::Arr(items));
                    }
                    return Err(self.err("expected `,` or `]`"));
                }
            }
            Some('{') => {
                self.at += 1;
                let mut entries = Vec::new();
                self.space();
                if self.eat('}') {
                    return Ok(Json::Obj(entries));
                }
                loop {
                    self.space();
                    let key = self.string()?;
                    self.space();
                    if !self.eat(':') {
                        return Err(self.err("expected `:`"));
                    }
                    let value = self.value()?;
                    entries.push((key, value));
                    self.space();
                    if self.eat(',') {
                        continue;
                    }
                    if self.eat('}') {
                        return Ok(Json::Obj(entries));
                    }
                    return Err(self.err("expected `,` or `}`"));
                }
            }
            Some(c) if *c == '-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(self.err(&format!("unexpected `{c}`"))),
        }
    }

    fn string(&mut self) -> Result<String, GrammarError> {
        if !self.eat('"') {
            return Err(self.err("expected a string"));
        }
        let mut out = String::new();
        loop {
            let Some(&c) = self.s.get(self.at) else {
                return Err(self.err("unterminated string"));
            };
            self.at += 1;
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let Some(&e) = self.s.get(self.at) else {
                        return Err(self.err("trailing backslash"));
                    };
                    self.at += 1;
                    out.push(match e {
                        '"' => '"',
                        '\\' => '\\',
                        '/' => '/',
                        'b' => '\u{8}',
                        'f' => '\u{c}',
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        'u' => {
                            let hex: String = self.s[self.at..(self.at + 4).min(self.s.len())]
                                .iter()
                                .collect();
                            if hex.len() != 4 {
                                return Err(self.err("short \\u escape"));
                            }
                            self.at += 4;
                            let n = u32::from_str_radix(&hex, 16)
                                .map_err(|_| self.err("bad \\u escape"))?;
                            char::from_u32(n).ok_or_else(|| self.err("bad code point"))?
                        }
                        other => return Err(self.err(&format!("unknown escape `\\{other}`"))),
                    });
                }
                _ => out.push(c),
            }
        }
    }

    fn number(&mut self) -> Result<Json, GrammarError> {
        let start = self.at;
        if self.s.get(self.at) == Some(&'-') {
            self.at += 1;
        }
        while matches!(self.s.get(self.at), Some(c) if c.is_ascii_digit()) {
            self.at += 1;
        }
        if self.s.get(self.at) == Some(&'.') {
            self.at += 1;
            while matches!(self.s.get(self.at), Some(c) if c.is_ascii_digit()) {
                self.at += 1;
            }
        }
        if matches!(self.s.get(self.at), Some('e' | 'E')) {
            self.at += 1;
            if matches!(self.s.get(self.at), Some('+' | '-')) {
                self.at += 1;
            }
            while matches!(self.s.get(self.at), Some(c) if c.is_ascii_digit()) {
                self.at += 1;
            }
        }
        let text: String = self.s[start..self.at].iter().collect();
        text.parse()
            .map(Json::Num)
            .map_err(|_| self.err("invalid number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grammar;

    /// A generated grammar is only as good as what it *refuses*, so every case
    /// here checks both directions against a real matcher.
    fn accepts(schema: &str, text: &str) -> bool {
        let g = Grammar::from_json_schema(schema).expect("converts");
        let mut m = g.matcher();
        m.accept_str(text) && m.is_complete()
    }

    #[test]
    fn json_parses() {
        assert_eq!(
            Json::parse(r#"{"a": [1, true, null], "b": "x"}"#).expect("parses"),
            Json::Obj(vec![
                (
                    "a".into(),
                    Json::Arr(vec![Json::Num(1.0), Json::Bool(true), Json::Null])
                ),
                ("b".into(), Json::Str("x".into())),
            ])
        );
    }

    #[test]
    fn a_string_schema_accepts_only_quoted_strings() {
        let s = r#"{"type": "string"}"#;
        assert!(accepts(s, r#""hello""#));
        assert!(!accepts(s, "hello"));
        assert!(!accepts(s, "123"));
    }

    /// JSON forbids leading zeros, and `[0-9]+` would allow them. This is one
    /// of the reasons the primitive rules are copied rather than rewritten.
    #[test]
    fn an_integer_may_not_have_a_leading_zero() {
        let s = r#"{"type": "integer"}"#;
        assert!(accepts(s, "0"));
        assert!(accepts(s, "-42"));
        assert!(accepts(s, "1234"));
        assert!(!accepts(s, "01"));
        assert!(!accepts(s, "1.5"));
    }

    #[test]
    fn a_number_accepts_decimals_and_exponents() {
        let s = r#"{"type": "number"}"#;
        for ok in ["0", "-1", "1.5", "1e10", "-2.5E-3"] {
            assert!(accepts(s, ok), "{ok}");
        }
        assert!(!accepts(s, "1."));
    }

    /// A raw control character inside a JSON string is invalid, and a grammar
    /// that allowed it would produce output the caller's parser rejects.
    #[test]
    fn a_string_may_not_contain_a_raw_control_character() {
        let s = r#"{"type": "string"}"#;
        assert!(accepts(s, "\"a\\nb\""), "an escaped newline is fine");
        assert!(!accepts(s, "\"a\nb\""), "a raw one is not");
    }

    #[test]
    fn an_object_requires_its_required_properties_in_order() {
        let s = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}"#;
        assert!(accepts(s, r#"{"name":"a","age":3}"#));
        assert!(
            !accepts(s, r#"{"age":3}"#),
            "a required property is missing"
        );
        assert!(!accepts(s, r#"{"name":"a","age":3,}"#), "trailing comma");
    }

    #[test]
    fn an_optional_property_may_be_present_or_absent() {
        let s = r#"{"type":"object","properties":{"name":{"type":"string"},"nick":{"type":"string"}},"required":["name"]}"#;
        assert!(accepts(s, r#"{"name":"a"}"#));
        assert!(accepts(s, r#"{"name":"a","nick":"b"}"#));
    }

    #[test]
    fn an_enum_admits_exactly_its_values() {
        let s = r#"{"enum":["red","green",1,null]}"#;
        for ok in [r#""red""#, r#""green""#, "1", "null"] {
            assert!(accepts(s, ok), "{ok}");
        }
        assert!(!accepts(s, r#""blue""#));
    }

    #[test]
    fn an_array_of_strings_needs_commas_and_no_trailing_one() {
        let s = r#"{"type":"array","items":{"type":"string"}}"#;
        assert!(accepts(s, "[]"));
        assert!(accepts(s, r#"["a"]"#));
        assert!(accepts(s, r#"["a","b"]"#));
        assert!(!accepts(s, r#"["a",]"#));
        assert!(!accepts(s, "[1]"));
    }

    #[test]
    fn min_items_is_enforced() {
        let s = r#"{"type":"array","items":{"type":"integer"},"minItems":2}"#;
        assert!(!accepts(s, "[]"));
        assert!(!accepts(s, "[1]"));
        assert!(accepts(s, "[1,2]"));
        assert!(accepts(s, "[1,2,3]"));
    }

    #[test]
    fn max_items_is_enforced() {
        let s = r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}"#;
        assert!(!accepts(s, "[]"));
        assert!(accepts(s, "[1]"));
        assert!(accepts(s, "[1,2]"));
        assert!(!accepts(s, "[1,2,3]"));
    }

    #[test]
    fn a_type_union_admits_either() {
        let s = r#"{"type":["string","null"]}"#;
        assert!(accepts(s, r#""a""#));
        assert!(accepts(s, "null"));
        assert!(!accepts(s, "1"));
    }

    #[test]
    fn one_of_admits_either_branch() {
        let s = r#"{"oneOf":[{"type":"integer"},{"type":"boolean"}]}"#;
        assert!(accepts(s, "12"));
        assert!(accepts(s, "true"));
        assert!(!accepts(s, r#""x""#));
    }

    #[test]
    fn a_ref_into_defs_resolves() {
        let s = r##"{"$defs":{"id":{"type":"integer"}},"type":"object","properties":{"a":{"$ref":"#/$defs/id"}},"required":["a"]}"##;
        assert!(accepts(s, r#"{"a":7}"#));
        assert!(!accepts(s, r#"{"a":"7"}"#));
    }

    #[test]
    fn a_const_is_exactly_itself() {
        let s = r#"{"const":"yes"}"#;
        assert!(accepts(s, r#""yes""#));
        assert!(!accepts(s, r#""no""#));
    }

    /// Ignoring a keyword yields a grammar looser than the schema, and nothing
    /// downstream can tell. Refusing is the safe direction.
    #[test]
    fn an_unimplemented_keyword_is_refused_by_name_not_ignored() {
        for (schema, keyword) in [
            (r#"{"type":"string","pattern":"^a"}"#, "pattern"),
            (r#"{"type":"integer","minimum":3}"#, "minimum"),
            (r#"{"allOf":[{"type":"string"}]}"#, "allOf"),
            (r#"{"type":"string","maxLength":5}"#, "maxLength"),
        ] {
            let e = Grammar::from_json_schema(schema).expect_err("must refuse");
            assert!(
                matches!(&e, GrammarError::SchemaUnsupported(k) if k == keyword),
                "{schema} gave {e:?}"
            );
        }
    }

    #[test]
    fn additional_properties_that_would_loosen_the_schema_are_refused() {
        let s = r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":true}"#;
        assert!(matches!(
            Grammar::from_json_schema(s),
            Err(GrammarError::SchemaUnsupported(_))
        ));
    }

    #[test]
    fn a_ref_to_something_undefined_is_an_error() {
        let s = r##"{"$ref":"#/$defs/missing"}"##;
        assert!(matches!(
            Grammar::from_json_schema(s),
            Err(GrammarError::Schema(_))
        ));
    }

    #[test]
    fn the_same_schema_always_produces_the_same_grammar() {
        let s = r#"{"type":"object","properties":{"b":{"type":"string"},"a":{"type":"integer"}},"required":["b","a"]}"#;
        let first = Grammar::gbnf_for_json_schema(s).expect("converts");
        for _ in 0..5 {
            assert_eq!(Grammar::gbnf_for_json_schema(s).expect("converts"), first);
        }
    }
}
