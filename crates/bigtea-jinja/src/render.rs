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
                let saved_pair: Vec<(String, Option<crate::Value>)> = var
                    .split(',')
                    .map(|n| n.trim().to_string())
                    .map(|n| {
                        let old = env.vars.get(&n).cloned();
                        (n, old)
                    })
                    .collect();
                let saved_loop = env.vars.get("loop").cloned();
                for (idx, item) in items.into_iter().enumerate() {
                    let mut lp = HashMap::new();
                    lp.insert("index0".to_string(), Value::Int(idx as i64));
                    lp.insert("index".to_string(), Value::Int(idx as i64 + 1));
                    lp.insert("first".to_string(), Value::Bool(idx == 0));
                    lp.insert("last".to_string(), Value::Bool(idx + 1 == n));
                    lp.insert("length".to_string(), Value::Int(n as i64));
                    env.set("loop", Value::Map(lp));
                    // `for k, v in pairs` binds two names from a two-element
                    // item. A pair that is not two elements is a template bug,
                    // and binding `none` would render a prompt with holes, so
                    // it refuses.
                    if let Some((a, b)) = var.split_once(',') {
                        let Value::List(pair) = &item else {
                            return Err(Error::Unsupported(format!(
                                "unpacking `{var}` from a non-pair"
                            )));
                        };
                        if pair.len() != 2 {
                            return Err(Error::Unsupported(format!(
                                "unpacking `{var}` from {} elements",
                                pair.len()
                            )));
                        }
                        env.set(a.trim(), pair[0].clone());
                        env.set(b.trim(), pair[1].clone());
                    } else {
                        env.set(var, item);
                    }
                    exec(body, env, out)?;
                }
                match saved_var {
                    Some(v) => env.set(var, v),
                    None => {
                        env.vars.remove(var);
                        &mut *env
                    }
                };
                // ...and each unpacked name, so `for k, v in` does not leak
                // either. Jinja's loop scope covers both.
                for (name, old) in saved_pair {
                    match old {
                        Some(v) => {
                            env.set(&name, v);
                        }
                        None => {
                            env.vars.remove(&name);
                        }
                    }
                }
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
        last_name: None,
    };
    p.ws();
    let v = p.ternary()?;
    p.ws();
    if p.i < p.s.len() {
        return Err(Error::Unsupported(format!(
            "trailing `{}` in expression `{src}`",
            src.get(p.i..).unwrap_or("")
        )));
    }
    Ok(v)
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
    src: &'a str,
    env: &'a Env,
    /// The last bare name read, so `is defined` can ask about the NAME rather
    /// than about the value it evaluated to. Without it a missing variable and
    /// a built-in function are indistinguishable -- both are `None`.
    last_name: Option<String>,
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
        // `get`, not `[..]`. Indexing panics when the range ends inside a
        // multi-byte character, and DeepSeek's template is full of U+FF5C
        // (`｜`) -- so a real container CRASHED this parser rather than being
        // refused by it. A panic is strictly worse than a refusal here: the
        // caller can fall back from a refusal.
        if self.src.get(self.i..end) == Some(word)
            && (end == self.s.len() || !is_name_byte(self.s[end]))
        {
            self.i = end;
            return true;
        }
        false
    }

    /// Whether the next non-space byte is `b`, without consuming it.
    fn peek(&mut self, b: u8) -> bool {
        self.ws();
        self.s.get(self.i) == Some(&b)
    }

    fn sym(&mut self, sym: &str) -> bool {
        self.ws();
        let end = self.i + sym.len();
        // As in `word`: multi-byte characters make byte indexing a panic.
        if self.src.get(self.i..end) == Some(sym) {
            self.i = end;
            return true;
        }
        false
    }

    /// `a if cond else b`. Jinja's inline conditional, and the only place the
    /// `if` keyword appears inside an expression rather than starting a block.
    ///
    /// **Both branches are evaluated** before one is chosen, which differs from
    /// Jinja's laziness. Harmless for chat templates -- the branches are string
    /// literals and variable reads with no side effects -- and noted rather
    /// than hidden, because a template whose unchosen branch raised would
    /// behave differently here.
    fn ternary(&mut self) -> Result<Value> {
        let first = self.or()?;
        let save = self.i;
        if !self.word("if") {
            return Ok(first);
        }
        let cond = self.or()?;
        if !self.word("else") {
            // `x if y` with no `else` is not something a chat template writes,
            // and guessing `none` for the missing branch would render a prompt
            // with a hole in it.
            self.i = save;
            return Err(Error::Unsupported(format!(
                "`if` without `else` in expression `{}`",
                self.src
            )));
        }
        let otherwise = self.ternary()?;
        Ok(if cond.truthy() { first } else { otherwise })
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
                //
                // **A built-in FUNCTION is defined too.** Llama-3's template
                // guards with `if strftime_now is defined`, and answering
                // `false` sent it down a fallback branch that hardcodes
                // `26 Jul 2024` -- so every Llama-3 prompt carried a date two
                // years stale, four tokens different from llama.cpp --jinja.
                "defined" => {
                    !matches!(left, Value::None)
                        || self.last_name.as_deref().is_some_and(is_builtin)
                }
                "none" => matches!(left, Value::None),
                "string" => matches!(left, Value::Str(_)),
                // Qwen3 writes `is false` where a plain `not` would do. NOT the
                // same as falsy: `is false` asks whether the value IS the
                // boolean false, so an empty string must not satisfy it.
                "false" => matches!(left, Value::Bool(false)),
                "true" => matches!(left, Value::Bool(true)),
                "number" | "integer" => matches!(left, Value::Int(_)),
                "sequence" => matches!(left, Value::List(_)),
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
            // `-` and `%` are here because the acceptance test found them, not
            // because the census predicted them: Qwen3 writes
            // `messages|length - 1` and Gemma-3 writes `loop.index0 % 2 == 0`.
            // The census counted statement tags and saw neither.
            let op = if self.sym("+") {
                '+'
            } else if self.sym("-") {
                '-'
            } else if self.sym("%") {
                '%'
            } else {
                return Ok(left);
            };
            {
                let right = self.unary()?;
                left = match (op, &left, &right) {
                    ('-', Value::Int(a), Value::Int(b)) => Value::Int(a - b),
                    ('%', Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a % b),
                    ('+', Value::Str(a), Value::Str(b)) => Value::Str(format!("{a}{b}")),
                    ('+', Value::Int(a), Value::Int(b)) => Value::Int(a + b),
                    ('+', Value::List(a), Value::List(b)) => {
                        let mut v = a.clone();
                        v.extend(b.clone());
                        Value::List(v)
                    }
                    // Jinja would coerce; coercing silently is how a template
                    // that meant to concatenate ends up printing `None`.
                    _ => {
                        return Err(Error::Unsupported(format!(
                            "`{op}` between {left:?} and {right:?}"
                        )))
                    }
                };
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
                // A slice, `messages[1:]`, which four templates on disk use to
                // drop the system turn. The census that scoped this crate
                // counted STATEMENT TAGS and missed every expression form --
                // this was the largest of the misses.
                if self.sym(":") {
                    let end = if self.peek(b']') {
                        None
                    } else {
                        Some(self.or()?)
                    };
                    if !self.sym("]") {
                        return Err(Error::Syntax(format!("unclosed `[` in `{}`", self.src)));
                    }
                    v = slice(&v, None, end.as_ref());
                    continue;
                }
                let idx = self.or()?;
                if self.sym(":") {
                    let end = if self.peek(b']') {
                        None
                    } else {
                        Some(self.or()?)
                    };
                    if !self.sym("]") {
                        return Err(Error::Syntax(format!("unclosed `[` in `{}`", self.src)));
                    }
                    v = slice(&v, Some(&idx), end.as_ref());
                    continue;
                }
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
            let v = self.ternary()?;
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
        // A negative literal. Not folded into `add` as a binary minus, because
        // `range(n, -1, -1)` has no left operand there -- Qwen3 walks backwards
        // over prior turns with exactly that, and without this the call failed
        // to parse at the comma.
        if c == b'-' && self.s.get(self.i + 1).is_some_and(|d| d.is_ascii_digit()) {
            self.i += 1;
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            return Ok(Value::Int(
                -self.src[start..self.i].parse::<i64>().unwrap_or(0),
            ));
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
            let mut kwargs: HashMap<String, Value> = HashMap::new();
            loop {
                self.ws();
                if self.sym(")") {
                    break;
                }
                // `namespace(multi_step_tool=true, last_query_index=...)` is
                // how Qwen3 opens its namespace. Detected by looking ahead for
                // a `name =` that is not `==`, because `a == b` is an ordinary
                // positional argument and misreading it would silently bind a
                // comparison as a field.
                let save = self.i;
                let kw = self.name().filter(|_| {
                    self.ws();
                    self.s.get(self.i) == Some(&b'=') && self.s.get(self.i + 1) != Some(&b'=')
                });
                match kw {
                    Some(k) => {
                        self.i += 1;
                        kwargs.insert(k, self.or()?);
                    }
                    None => {
                        self.i = save;
                        args.push(self.or()?);
                    }
                }
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
                // The keyword arguments ARE the namespace's initial fields.
                "namespace" => Ok(Value::Map(kwargs)),
                // Llama-3's template stamps today's date into the system turn.
                // llama.cpp --jinja emits the real date; without this we took
                // the template's hardcoded 2024 fallback and produced a prompt
                // that differed from the reference by four tokens.
                //
                // **This makes the render non-reproducible**, which is a real
                // cost: two runs a day apart produce different prompts, so a
                // byte-comparison against a captured fixture will fail for a
                // reason that is not a bug. Recorded rather than avoided,
                // because freezing a fake date would make every Llama-3 prompt
                // wrong in a way nothing would ever notice.
                "strftime_now" => {
                    let fmt = args.first().map(|a| a.render()).unwrap_or_default();
                    Ok(Value::Str(strftime_now(&fmt)))
                }
                // Python's semantics, including a negative step -- Qwen3 walks
                // backwards over prior turns with `range(n, -1, -1)`, and a
                // forward-only range would silently produce an empty loop and
                // drop every turn it was meant to inspect.
                "range" => {
                    let n = |i: usize| match args.get(i) {
                        Some(Value::Int(v)) => Some(*v),
                        _ => None,
                    };
                    let (start, stop, step) = match args.len() {
                        1 => (0, n(0).unwrap_or(0), 1),
                        2 => (n(0).unwrap_or(0), n(1).unwrap_or(0), 1),
                        3 => (n(0).unwrap_or(0), n(1).unwrap_or(0), n(2).unwrap_or(1)),
                        _ => return Err(Error::Unsupported("range() with 0 or >3 args".into())),
                    };
                    if step == 0 {
                        return Err(Error::Unsupported("range() with step 0".into()));
                    }
                    let mut out = Vec::new();
                    let mut i = start;
                    while (step > 0 && i < stop) || (step < 0 && i > stop) {
                        out.push(Value::Int(i));
                        i += step;
                    }
                    Ok(Value::List(out))
                }
                other => Err(Error::Unsupported(format!("call to `{other}()`"))),
            };
        }
        // A bare name. Missing is `none` so `is defined` can ask about it --
        // every other use of a missing name surfaces as a render that is
        // visibly wrong rather than silently short.
        let v = self.env.get(&name).cloned().unwrap_or(Value::None);
        self.last_name = Some(name);
        Ok(v)
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

/// Names the evaluator provides as functions rather than variables.
///
/// Kept next to the call site that implements them; a name here with no
/// implementation would report itself defined and then fail to call, which is a
/// worse failure than reporting it undefined.
fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        "strftime_now" | "namespace" | "raise_exception" | "range"
    )
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

/// Python's slice semantics, which Jinja inherits: negative indices count from
/// the end and out-of-range bounds clamp rather than panic.
fn slice(v: &Value, start: Option<&Value>, end: Option<&Value>) -> Value {
    let Value::List(items) = v else {
        // A slice of anything else is a template bug. `None` renders visibly
        // wrong rather than silently dropping turns.
        return Value::None;
    };
    let n = items.len() as i64;
    let norm = |x: i64| -> usize {
        let x = if x < 0 { n + x } else { x };
        x.clamp(0, n) as usize
    };
    let lo = match start {
        Some(Value::Int(i)) => norm(*i),
        _ => 0,
    };
    let hi = match end {
        Some(Value::Int(i)) => norm(*i),
        _ => n as usize,
    };
    Value::List(if lo >= hi {
        Vec::new()
    } else {
        items[lo..hi].to_vec()
    })
}

/// The handful of `strftime` fields Llama-3's template uses, from the system
/// clock.
///
/// Written out rather than pulled from `chrono`: this crate has no dependencies
/// by design, and the conversion is Howard Hinnant's `civil_from_days`, which
/// is exact for every date in the range and about ten lines.
fn strftime_now(fmt: &str) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut out = String::new();
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('Y') => out.push_str(&y.to_string()),
            Some('b') => out.push_str(MONTHS[(m as usize).clamp(1, 12) - 1]),
            Some('e') => out.push_str(&format!("{d:2}")),
            // An unknown field is emitted literally rather than dropped: a
            // silently missing date component is a prompt that looks right.
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// Days since the Unix epoch to `(year, month, day)`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
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
    fn a_builtin_function_is_defined() {
        // Llama-3 guards with `if strftime_now is defined` and takes a
        // fallback branch that hardcodes 26 Jul 2024 when the answer is false.
        assert_eq!(r("{{ strftime_now is defined }}").unwrap(), "True");
        assert_eq!(r("{{ namespace is defined }}").unwrap(), "True");
        assert_eq!(r("{{ nope is defined }}").unwrap(), "False");
    }

    #[test]
    fn strftime_formats_the_fields_llama3_uses() {
        // Not asserting the value -- it is the wall clock. Asserting the SHAPE,
        // which is what a wrong conversion would break: `26 Jul 2024`.
        let s = strftime_now("%d %b %Y");
        let parts: Vec<&str> = s.split(' ').collect();
        assert_eq!(parts.len(), 3, "{s:?}");
        assert_eq!(parts[0].len(), 2, "{s:?}");
        assert_eq!(parts[1].len(), 3, "{s:?}");
        assert_eq!(parts[2].len(), 4, "{s:?}");
        assert!(parts[2].parse::<i64>().unwrap() >= 2024, "{s:?}");
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        // Exact anchors, because an off-by-one here is a prompt that differs
        // from the reference by one token and nothing else would catch it.
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        // A leap day, which is where naive conversions break.
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    }

    #[test]
    fn multibyte_characters_do_not_panic() {
        // DeepSeek's template is full of U+FF5C (`｜`), and byte-indexed
        // keyword matching PANICKED on it against a real container:
        // `end byte index 3 is not a char boundary`. A panic is strictly
        // worse than a refusal, because the caller can fall back from a
        // refusal and cannot fall back from a crash.
        for src in ["'<｜User｜>'", "a｜b", "｜", "messages[0]['｜']"] {
            let _ = eval(src, &env());
        }
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
