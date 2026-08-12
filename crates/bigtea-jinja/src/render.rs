//! Evaluating the tree, and the one function that decides whether to refuse.
//!
//! # The expression grammar, in precedence order
//!
//! ```text
//! or   := and ('or' and)*
//! and  := not ('and' not)*
//! not  := 'not' not | cmp
//! cmp  := add (('==' | '!=' | '<' | '>' | '<=' | '>=' | 'in' | 'not in'
//!               | 'is' test) add)?
//! add  := unary ('+' unary)*
//! unary:= atom trailer*
//! atom := literal | name | '(' or ')' | '[' list ']'
//! trailer := '.' name | '[' or ']' | '|' filter | '(' args ')'
//! ```
//!
//! Anything that does not fit is [`Error::Unsupported`]. That is the crate's
//! whole safety property, so the fallthroughs below are load-bearing and must
//! stay exhaustive — a construct silently evaluating to `none` would render a
//! plausible prompt from missing data, which is worse than not rendering.

use std::collections::HashMap;

use crate::parse::Node;
use crate::{Error, Result, Value};

/// The variables a chat template is given.
#[derive(Debug, Clone, Default)]
pub struct Env {
    vars: HashMap<String, Value>,
}

impl Env {
    pub fn new() -> Self {
        Env::default()
    }

    pub fn set(&mut self, name: &str, v: Value) -> &mut Self {
        self.vars.insert(name.to_string(), v);
        self
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.vars.get(name)
    }
}

/// Render `nodes` against `env`.
pub fn render(nodes: &[Node], env: &mut Env) -> Result<String> {
    let mut out = String::new();
    exec(nodes, env, &mut out)?;
    Ok(out)
}

fn exec(nodes: &[Node], env: &mut Env, out: &mut String) -> Result<()> {
    for n in nodes {
        match n {
            Node::Text(t) => out.push_str(t),
            Node::Output(e) => out.push_str(&eval(e, env)?.render()),
            Node::Set { target, expr } => {
                let v = eval(expr, env)?;
                // `ns.field = x`, the only reason `namespace()` exists: Jinja
                // scopes `set` to the loop body, so templates use a namespace
                // to carry state out of a loop. Assigning to the whole name
                // instead would lose it at `endfor`.
                if let Some((obj, field)) = target.split_once('.') {
                    let Some(Value::Map(m)) = env.vars.get_mut(obj.trim()) else {
                        return Err(Error::UndefinedVariable(obj.trim().to_string()));
                    };
                    m.insert(field.trim().to_string(), v);
                } else {
                    env.set(target, v);
                }
            }
            Node::If { arms, otherwise } => {
                let mut done = false;
                for (cond, body) in arms {
                    if eval(cond, env)?.truthy() {
                        exec(body, env, out)?;
                        done = true;
                        break;
                    }
                }
                if !done {
                    exec(otherwise, env, out)?;
                }
            }
            Node::For { var, iter, body } => {
                let items = match eval(iter, env)? {
                    Value::List(l) => l,
                    // Iterating a non-list is a template bug, not ours, and
                    // silently doing nothing would render a prompt with the
                    // turns missing.
                    other => {
                        return Err(Error::Unsupported(format!(
                            "`for` over a non-list ({other:?})"
                        )))
                    }
                };
                let n = items.len();
                // Saved and restored: Jinja's loop variable does not leak, and
                // nested loops must not see the outer `loop`.
                let saved_var = env.vars.get(var).cloned();
                let saved_loop = env.vars.get("loop").cloned();
                for (idx, item) in items.into_iter().enumerate() {
                    let mut lp = HashMap::new();
                    lp.insert("index0".to_string(), Value::Int(idx as i64));
                    lp.insert("index".to_string(), Value::Int(idx as i64 + 1));
                    lp.insert("first".to_string(), Value::Bool(idx == 0));
                    lp.insert("last".to_string(), Value::Bool(idx + 1 == n));
                    lp.insert("length".to_string(), Value::Int(n as i64));
                    env.set("loop", Value::Map(lp));
                    env.set(var, item);
                    exec(body, env, out)?;
                }
                match saved_var {
                    Some(v) => env.set(var, v),
                    None => {
                        env.vars.remove(var);
                        &mut *env
                    }
                };
                match saved_loop {
                    Some(v) => env.set("loop", v),
                    None => {
                        env.vars.remove("loop");
                        &mut *env
                    }
                };
            }
        }
    }
    Ok(())
}

// --- expressions -----------------------------------------------------------

pub fn eval(src: &str, env: &Env) -> Result<Value> {
    let mut p = P {
        s: src.as_bytes(),
        i: 0,
        src,
        env,
    };
    p.ws();
    let v = p.or()?;
    p.ws();
    if p.i < p.s.len() {
        return Err(Error::Unsupported(format!(
            "trailing `{}` in expression `{src}`",
            &src[p.i..]
        )));
    }
    Ok(v)
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
    src: &'a str,
    env: &'a Env,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() {
            self.i += 1;
        }
    }

    /// Consume `word` if it is the next token and is followed by a boundary.
    fn word(&mut self, word: &str) -> bool {
        self.ws();
        let end = self.i + word.len();
        if end <= self.s.len()
            && &self.src[self.i..end] == word
            && (end == self.s.len() || !is_name_byte(self.s[end]))
        {
            self.i = end;
            return true;
        }
        false
    }

    fn sym(&mut self, sym: &str) -> bool {
        self.ws();
        let end = self.i + sym.len();
        if end <= self.s.len() && &self.src[self.i..end] == sym {
            self.i = end;
            return true;
        }
        false
    }

    fn or(&mut self) -> Result<Value> {
        let mut left = self.and()?;
        while self.word("or") {
            let right = self.and()?;
            left = Value::Bool(left.truthy() || right.truthy());
        }
        Ok(left)
    }

    fn and(&mut self) -> Result<Value> {
        let mut left = self.not()?;
        while self.word("and") {
            let right = self.not()?;
            left = Value::Bool(left.truthy() && right.truthy());
        }
        Ok(left)
    }

    fn not(&mut self) -> Result<Value> {
        if self.word("not") {
            return Ok(Value::Bool(!self.not()?.truthy()));
        }
        self.cmp()
    }

    fn cmp(&mut self) -> Result<Value> {
        let left = self.add()?;
        self.ws();
        // `is` tests first: `is not none` must not be read as `is` then `not`.
        if self.word("is") {
            let negated = self.word("not");
            let t = self.name().ok_or_else(|| {
                Error::Unsupported(format!("`is` without a test in `{}`", self.src))
            })?;
            let r = match t.as_str() {
                // `defined` is the reason a bare name may be missing without
                // being an error -- every other path treats that as a bug.
                "defined" => !matches!(left, Value::None),
                "none" => matches!(left, Value::None),
                "string" => matches!(left, Value::Str(_)),
                "mapping" => matches!(left, Value::Map(_)),
                "iterable" => matches!(left, Value::List(_) | Value::Str(_)),
                other => {
                    return Err(Error::Unsupported(format!("test `is {other}`")));
                }
            };
            return Ok(Value::Bool(if negated { !r } else { r }));
        }
        if self.word("not") {
            // `not in`
            if !self.word("in") {
                return Err(Error::Unsupported(format!("`not` in `{}`", self.src)));
            }
            let right = self.add()?;
            return Ok(Value::Bool(!contains(&right, &left)));
        }
        if self.word("in") {
            let right = self.add()?;
            return Ok(Value::Bool(contains(&right, &left)));
        }
        for (op, f) in [
            ("==", 0usize),
            ("!=", 1),
            ("<=", 2),
            (">=", 3),
            ("<", 4),
            (">", 5),
        ] {
            if self.sym(op) {
                let right = self.add()?;
                return Ok(Value::Bool(match f {
                    0 => left == right,
                    1 => left != right,
                    _ => {
                        let (a, b) = match (&left, &right) {
                            (Value::Int(a), Value::Int(b)) => (*a, *b),
                            _ => {
                                return Err(Error::Unsupported(format!(
                                    "ordering non-integers in `{}`",
                                    self.src
                                )))
                            }
                        };
                        match f {
                            2 => a <= b,
                            3 => a >= b,
                            4 => a < b,
                            _ => a > b,
                        }
                    }
                }));
            }
        }
        Ok(left)
    }

    fn add(&mut self) -> Result<Value> {
        let mut left = self.unary()?;
        loop {
            self.ws();
            // `+` only. `-` would be ambiguous with the whitespace-control dash
            // the lexer already stripped, and no template on disk subtracts.
            if self.sym("+") {
                let right = self.unary()?;
                left = match (&left, &right) {
                    (Value::Str(a), Value::Str(b)) => Value::Str(format!("{a}{b}")),
                    (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
                    (Value::List(a), Value::List(b)) => {
                        let mut v = a.clone();
                        v.extend(b.clone());
                        Value::List(v)
                    }
                    // Jinja would coerce; coercing silently is how a template
                    // that meant to concatenate ends up printing `None`.
                    _ => {
                        return Err(Error::Unsupported(format!(
                            "`+` between {left:?} and {right:?}"
                        )))
                    }
                };
            } else {
                return Ok(left);
            }
        }
    }

    fn unary(&mut self) -> Result<Value> {
        let mut v = self.atom()?;
        loop {
            self.ws();
            if self.sym(".") {
                let Some(field) = self.name() else {
                    return Err(Error::Syntax(format!(
                        "`.` without a name in `{}`",
                        self.src
                    )));
                };
                v = index(&v, &Value::Str(field));
            } else if self.sym("[") {
                let idx = self.or()?;
                if !self.sym("]") {
                    return Err(Error::Syntax(format!("unclosed `[` in `{}`", self.src)));
                }
                v = index(&v, &idx);
            } else if self.sym("|") {
                let Some(f) = self.name() else {
                    return Err(Error::Syntax(format!(
                        "`|` without a filter in `{}`",
                        self.src
                    )));
                };
                // Filters taking arguments (`join(', ')`) are consumed and
                // refused rather than ignored -- ignoring an argument renders
                // the right shape with the wrong separator.
                if self.sym("(") {
                    let _ = self.or();
                    let _ = self.sym(")");
                    return Err(Error::Unsupported(format!("filter `{f}` with arguments")));
                }
                v = match f.as_str() {
                    "trim" => Value::Str(v.render().trim().to_string()),
                    "tojson" => Value::Str(crate::json_public(&v)),
                    "length" | "count" => Value::Int(match &v {
                        Value::List(l) => l.len() as i64,
                        Value::Str(s) => s.chars().count() as i64,
                        Value::Map(m) => m.len() as i64,
                        _ => 0,
                    }),
                    "string" => Value::Str(v.render()),
                    "first" => match &v {
                        Value::List(l) => l.first().cloned().unwrap_or(Value::None),
                        _ => Value::None,
                    },
                    "last" => match &v {
                        Value::List(l) => l.last().cloned().unwrap_or(Value::None),
                        _ => Value::None,
                    },
                    other => return Err(Error::Unsupported(format!("filter `{other}`"))),
                };
            } else {
                return Ok(v);
            }
        }
    }

    fn atom(&mut self) -> Result<Value> {
        self.ws();
        if self.i >= self.s.len() {
            return Err(Error::Syntax(format!(
                "expression ended early: `{}`",
                self.src
            )));
        }
        let c = self.s[self.i];
        if c == b'(' {
            self.i += 1;
            let v = self.or()?;
            if !self.sym(")") {
                return Err(Error::Syntax(format!("unclosed `(` in `{}`", self.src)));
            }
            return Ok(v);
        }
        if c == b'[' {
            self.i += 1;
            let mut items = Vec::new();
            loop {
                self.ws();
                if self.sym("]") {
                    break;
                }
                items.push(self.or()?);
                if !self.sym(",") && !self.sym("]") {
                    return Err(Error::Syntax(format!("bad list in `{}`", self.src)));
                }
                if self.s.get(self.i.wrapping_sub(1)) == Some(&b']') {
                    break;
                }
            }
            return Ok(Value::List(items));
        }
        if c == b'\'' || c == b'"' {
            let quote = c;
            self.i += 1;
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i] != quote {
                // Backslash escapes inside template string literals.
                if self.s[self.i] == b'\\' {
                    self.i += 1;
                }
                self.i += 1;
            }
            if self.i >= self.s.len() {
                return Err(Error::Syntax(format!(
                    "unterminated string in `{}`",
                    self.src
                )));
            }
            let raw = &self.src[start..self.i];
            self.i += 1;
            return Ok(Value::Str(unescape(raw)));
        }
        if c.is_ascii_digit() {
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            return Ok(Value::Int(self.src[start..self.i].parse().unwrap_or(0)));
        }
        let Some(name) = self.name() else {
            return Err(Error::Syntax(format!(
                "cannot read `{}` at byte {}",
                self.src, self.i
            )));
        };
        match name.as_str() {
            "true" | "True" => return Ok(Value::Bool(true)),
            "false" | "False" => return Ok(Value::Bool(false)),
            "none" | "None" => return Ok(Value::None),
            _ => {}
        }
        // A call.
        self.ws();
        if self.i < self.s.len() && self.s[self.i] == b'(' {
            self.i += 1;
            let mut args = Vec::new();
            loop {
                self.ws();
                if self.sym(")") {
                    break;
                }
                args.push(self.or()?);
                self.ws();
                if self.sym(")") {
                    break;
                }
                if !self.sym(",") {
                    return Err(Error::Syntax(format!(
                        "bad call `{name}(` in `{}`",
                        self.src
                    )));
                }
            }
            return match name.as_str() {
                // Templates use this to reject conversations they cannot
                // express. It MUST fail the render -- swallowing it produces
                // exactly the framing the template exists to prevent.
                "raise_exception" => Err(Error::Raised(
                    args.first().map(|a| a.render()).unwrap_or_default(),
                )),
                // `namespace(a=1)` -- keyword arguments are not parsed, and an
                // empty namespace is what every template on disk creates.
                "namespace" => Ok(Value::Map(HashMap::new())),
                other => Err(Error::Unsupported(format!("call to `{other}()`"))),
            };
        }
        // A bare name. Missing is `none` so `is defined` can ask about it --
        // every other use of a missing name surfaces as a render that is
        // visibly wrong rather than silently short.
        Ok(self.env.get(&name).cloned().unwrap_or(Value::None))
    }

    fn name(&mut self) -> Option<String> {
        self.ws();
        let start = self.i;
        while self.i < self.s.len() && is_name_byte(self.s[self.i]) {
            self.i += 1;
        }
        (self.i > start).then(|| self.src[start..self.i].to_string())
    }
}

fn is_name_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn unescape(raw: &str) -> String {
    let mut out = String::new();
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

fn index(v: &Value, key: &Value) -> Value {
    match (v, key) {
        (Value::Map(m), Value::Str(k)) => m.get(k).cloned().unwrap_or(Value::None),
        (Value::List(l), Value::Int(i)) => {
            let i = if *i < 0 { l.len() as i64 + i } else { *i };
            usize::try_from(i)
                .ok()
                .and_then(|i| l.get(i).cloned())
                .unwrap_or(Value::None)
        }
        _ => Value::None,
    }
}

fn contains(hay: &Value, needle: &Value) -> bool {
    match hay {
        Value::List(l) => l.contains(needle),
        Value::Map(m) => matches!(needle, Value::Str(k) if m.contains_key(k)),
        Value::Str(s) => matches!(needle, Value::Str(n) if s.contains(n.as_str())),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn env() -> Env {
        let mut e = Env::new();
        let mut m = HashMap::new();
        m.insert("role".to_string(), Value::Str("user".into()));
        m.insert("content".to_string(), Value::Str("hi".into()));
        e.set("messages", Value::List(vec![Value::Map(m)]));
        e.set("bos_token", Value::Str("<s>".into()));
        e.set("add_generation_prompt", Value::Bool(true));
        e
    }

    fn r(src: &str) -> Result<String> {
        let nodes = parse(src)?;
        render(&nodes, &mut env())
    }

    #[test]
    fn a_real_chatml_template_renders() {
        let t = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}\
                 <|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        assert_eq!(
            r(t).unwrap(),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn dotted_and_bracketed_access_agree() {
        assert_eq!(r("{{ messages[0].role }}").unwrap(), "user");
        assert_eq!(r("{{ messages[0]['role'] }}").unwrap(), "user");
    }

    #[test]
    fn loop_variables_are_available_and_do_not_leak() {
        let t =
            "{% for m in messages %}{{ loop.first }}{{ loop.last }}{{ loop.index0 }}{% endfor %}\
                 [{{ loop }}]";
        assert_eq!(r(t).unwrap(), "TrueTrue0[None]");
    }

    #[test]
    fn is_defined_distinguishes_missing_from_false() {
        assert_eq!(r("{{ nope is defined }}").unwrap(), "False");
        assert_eq!(r("{{ messages is defined }}").unwrap(), "True");
        assert_eq!(r("{{ nope is not defined }}").unwrap(), "True");
    }

    #[test]
    fn namespace_carries_state_out_of_a_loop() {
        // The whole reason `namespace()` exists: a plain `set` inside a loop is
        // scoped to the body and its value is gone at `endfor`.
        let t = "{% set ns = namespace() %}{% set ns.seen = 0 %}\
                 {% for m in messages %}{% set ns.seen = 1 %}{% endfor %}{{ ns.seen }}";
        assert_eq!(r(t).unwrap(), "1");
    }

    #[test]
    fn raise_exception_fails_the_render() {
        // Templates use it to reject conversations they cannot express.
        // Swallowing it produces exactly the framing they exist to prevent.
        let e = r("{% if true %}{{ raise_exception('no system turn') }}{% endif %}").unwrap_err();
        assert!(
            matches!(e, Error::Raised(ref m) if m.contains("no system")),
            "{e:?}"
        );
    }

    #[test]
    fn membership_and_comparison() {
        assert_eq!(r("{{ 'us' in 'user' }}").unwrap(), "True");
        assert_eq!(r("{{ 'zz' not in 'user' }}").unwrap(), "True");
        assert_eq!(r("{{ messages[0].role == 'user' }}").unwrap(), "True");
        assert_eq!(r("{{ messages | length }}").unwrap(), "1");
    }

    #[test]
    fn unknown_filters_and_calls_are_refused_not_ignored() {
        // The safety property, at expression level. A filter that silently did
        // nothing would render the right shape with the wrong content.
        assert!(matches!(
            r("{{ messages | upper }}").unwrap_err(),
            Error::Unsupported(_)
        ));
        assert!(matches!(
            r("{{ lipsum() }}").unwrap_err(),
            Error::Unsupported(_)
        ));
        // Even a filter we DO know is refused when given arguments, because
        // ignoring the argument changes the output rather than failing.
        assert!(matches!(
            r("{{ messages | join(', ') }}").unwrap_err(),
            Error::Unsupported(_)
        ));
    }

    #[test]
    fn string_concatenation_works_and_mixed_types_refuse() {
        assert_eq!(r("{{ 'a' + 'b' }}").unwrap(), "ab");
        assert!(matches!(
            r("{{ 'a' + 1 }}").unwrap_err(),
            Error::Unsupported(_)
        ));
    }
}
