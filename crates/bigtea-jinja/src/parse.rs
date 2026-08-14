//! Turning the token stream into a tree of blocks.
//!
//! Expressions are **not** parsed here — they are kept as source text and
//! evaluated in `render`. That is deliberate: the expression language is the
//! part most likely to meet something outside the subset, so keeping it in one
//! place means there is exactly one function that decides whether to refuse.

use crate::lex::{lex, Token, TokenKind};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Text(String),
    /// `{{ expr }}`
    Output(String),
    /// `{% set name = expr %}`, including the `ns.field = expr` form that
    /// `namespace()` exists for.
    Set {
        target: String,
        expr: String,
    },
    If {
        /// `(condition, body)` for the `if` and each `elif`, in order.
        arms: Vec<(String, Vec<Node>)>,
        otherwise: Vec<Node>,
    },
    For {
        var: String,
        iter: String,
        body: Vec<Node>,
    },
}

/// Parse a template. Refuses any statement outside the censused subset.
pub fn parse(src: &str) -> Result<Vec<Node>> {
    let tokens = lex(src)?;
    let mut i = 0usize;
    let nodes = parse_block(&tokens, &mut i, &[])?;
    if let Some(TokenKind::Stmt(s)) = tokens.get(i).map(|t| t.kind.clone()) {
        return Err(Error::Syntax(format!("unexpected `{{% {s} %}}`")));
    }
    Ok(nodes)
}

/// Parse until one of `stop` (or the end). `i` is left ON the stopping token.
fn parse_block(tokens: &[Token], i: &mut usize, stop: &[&str]) -> Result<Vec<Node>> {
    let mut out = Vec::new();
    while *i < tokens.len() {
        match tokens[*i].kind.clone() {
            TokenKind::Text(t) => {
                out.push(Node::Text(t));
                *i += 1;
            }
            TokenKind::Output(e) => {
                out.push(Node::Output(e));
                *i += 1;
            }
            TokenKind::Stmt(s) => {
                let head = s.split_whitespace().next().unwrap_or("").to_string();
                if stop.contains(&head.as_str()) {
                    return Ok(out);
                }
                match head.as_str() {
                    "set" => {
                        let rest = s[3..].trim();
                        let Some((target, expr)) = rest.split_once('=') else {
                            return Err(Error::Syntax(format!("`set` without `=`: {s}")));
                        };
                        out.push(Node::Set {
                            target: target.trim().to_string(),
                            expr: expr.trim().to_string(),
                        });
                        *i += 1;
                    }
                    "if" => {
                        *i += 1;
                        let first = parse_block(tokens, i, &["elif", "else", "endif"])?;
                        let mut arms = vec![(s[2..].trim().to_string(), first)];
                        let mut otherwise = Vec::new();
                        loop {
                            let Some(TokenKind::Stmt(t)) = tokens.get(*i).map(|t| t.kind.clone())
                            else {
                                return Err(Error::Syntax("unterminated `if`".into()));
                            };
                            match t.split_whitespace().next().unwrap_or("") {
                                "elif" => {
                                    *i += 1;
                                    let body = parse_block(tokens, i, &["elif", "else", "endif"])?;
                                    arms.push((t[4..].trim().to_string(), body));
                                }
                                "else" => {
                                    *i += 1;
                                    otherwise = parse_block(tokens, i, &["endif"])?;
                                }
                                "endif" => {
                                    *i += 1;
                                    break;
                                }
                                _ => {
                                    return Err(Error::Syntax(format!("unexpected `{t}` in `if`")))
                                }
                            }
                        }
                        out.push(Node::If { arms, otherwise });
                    }
                    "for" => {
                        let rest = s[3..].trim();
                        let Some((var, iter)) = rest.split_once(" in ") else {
                            return Err(Error::Syntax(format!("`for` without ` in `: {s}")));
                        };
                        // `for k, v in ...` -- DeepSeek-V4-Flash uses it. The
                        // variable names are kept comma-joined and split in the
                        // renderer, which is where the value to unpack lives.
                        *i += 1;
                        let body = parse_block(tokens, i, &["endfor"])?;
                        match tokens.get(*i).map(|t| t.kind.clone()) {
                            Some(TokenKind::Stmt(ref t)) if t.starts_with("endfor") => *i += 1,
                            _ => return Err(Error::Syntax("unterminated `for`".into())),
                        }
                        out.push(Node::For {
                            var: var.trim().to_string(),
                            iter: iter.trim().to_string(),
                            body,
                        });
                    }
                    // Everything else Jinja has: macro, import, include,
                    // extends, block, filter, call, with, raw. None appear in
                    // any chat template on disk, and each needs real semantics.
                    other => return Err(Error::Unsupported(format!("`{{% {other} %}}`"))),
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_and_output_round_trip() {
        assert_eq!(
            parse("a{{ x }}").unwrap(),
            vec![Node::Text("a".into()), Node::Output("x".into())]
        );
    }

    #[test]
    fn set_splits_on_the_first_equals() {
        assert_eq!(
            parse("{% set x = a == b %}").unwrap(),
            vec![Node::Set {
                target: "x".into(),
                expr: "a == b".into()
            }]
        );
    }

    #[test]
    fn if_elif_else_keeps_arm_order() {
        let n = parse("{% if a %}1{% elif b %}2{% else %}3{% endif %}").unwrap();
        let Node::If { arms, otherwise } = &n[0] else {
            panic!("{n:?}");
        };
        assert_eq!(arms.len(), 2);
        assert_eq!(arms[0].0, "a");
        assert_eq!(arms[1].0, "b");
        assert_eq!(otherwise.len(), 1);
    }

    #[test]
    fn nested_blocks_close_against_the_right_opener() {
        let n = parse("{% for m in ms %}{% if m %}x{% endif %}{% endfor %}").unwrap();
        let Node::For { body, .. } = &n[0] else {
            panic!("{n:?}");
        };
        assert!(matches!(body[0], Node::If { .. }));
    }

    #[test]
    fn an_unknown_statement_is_refused_by_name() {
        // The safety property. A construct outside the subset must send the
        // caller back to the family matcher, not render something plausible.
        let e = parse("{% macro f() %}{% endmacro %}").unwrap_err();
        assert!(
            matches!(e, Error::Unsupported(ref w) if w.contains("macro")),
            "{e:?}"
        );
    }

    #[test]
    fn tuple_unpacking_keeps_both_names() {
        // DeepSeek-V4-Flash uses this. The names stay comma-joined here and
        // are split in the renderer, where the value to unpack is.
        let n = parse("{% for k, v in m %}{% endfor %}").unwrap();
        let Node::For { var, .. } = &n[0] else {
            panic!("{n:?}");
        };
        assert_eq!(var, "k, v");
    }

    #[test]
    fn an_unclosed_block_is_an_error() {
        assert!(parse("{% if a %}x").is_err());
        assert!(parse("{% for m in ms %}x").is_err());
    }
}
