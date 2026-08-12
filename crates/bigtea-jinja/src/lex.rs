//! Splitting a template into literal text, `{{ … }}` and `{% … %}`.
//!
//! # Whitespace control is not cosmetic here
//!
//! Jinja's `{%-` and `-%}` strip surrounding whitespace, and chat templates use
//! them constantly — Llama-3's is written almost entirely in the `{{-` form. A
//! stripped newline that should have survived, or a surviving one that should
//! have been stripped, is a prompt the model was not trained on. It does not
//! error; it shifts every following token.
//!
//! So the trimming is done here, at the token boundary, rather than left to the
//! renderer where it would have to reason about what came before.

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// Literal text between tags.
    Text(String),
    /// `{{ expr }}` — the inside, untrimmed of its own spaces.
    Output(String),
    /// `{% stmt %}` — likewise.
    Stmt(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    /// Byte offset in the source, for error messages that can be located.
    pub at: usize,
}

/// Split `src` into tokens, applying `-` whitespace control.
pub fn lex(src: &str) -> Result<Vec<Token>> {
    let b = src.as_bytes();
    let mut out: Vec<Token> = Vec::new();
    let mut i = 0usize;
    let mut text_start = 0usize;
    // Set when the previous tag ended with `-%}` or `-}}`: the *next* run of
    // literal text loses its leading whitespace.
    let mut trim_next_leading = false;

    while i < b.len() {
        if b[i] == b'{'
            && i + 1 < b.len()
            && (b[i + 1] == b'{' || b[i + 1] == b'%' || b[i + 1] == b'#')
        {
            let is_output = b[i + 1] == b'{';
            let is_comment = b[i + 1] == b'#';
            // Literal text before this tag.
            let mut text = &src[text_start..i];
            // `{{-` / `{%-` strips the whitespace *before* the tag.
            let strip_before = i + 2 < b.len() && b[i + 2] == b'-';
            if strip_before {
                text = text.trim_end();
            }
            let mut owned = text.to_string();
            if trim_next_leading {
                owned = owned.trim_start().to_string();
            }
            trim_next_leading = false;
            if !owned.is_empty() {
                out.push(Token {
                    kind: TokenKind::Text(owned),
                    at: text_start,
                });
            }

            let close: &[u8] = if is_output {
                b"}}"
            } else if is_comment {
                b"#}"
            } else {
                b"%}"
            };
            let body_start = i + if strip_before { 3 } else { 2 };
            let Some(rel) = find(&b[body_start..], close) else {
                return Err(Error::Syntax(format!(
                    "unterminated {} at byte {i}",
                    if is_output { "{{" } else { "{%" }
                )));
            };
            let mut body_end = body_start + rel;
            // `-%}` / `-}}` strips the whitespace *after* the tag.
            if body_end > body_start && b[body_end - 1] == b'-' {
                body_end -= 1;
                trim_next_leading = true;
            }
            let body = src[body_start..body_end].trim().to_string();
            if !is_comment {
                out.push(Token {
                    kind: if is_output {
                        TokenKind::Output(body)
                    } else {
                        TokenKind::Stmt(body)
                    },
                    at: i,
                });
            }
            i = body_start + rel + 2;
            text_start = i;
        } else {
            i += 1;
        }
    }

    let mut tail = src[text_start..].to_string();
    if trim_next_leading {
        tail = tail.trim_start().to_string();
    }
    if !tail.is_empty() {
        out.push(Token {
            kind: TokenKind::Text(tail),
            at: text_start,
        });
    }
    Ok(out)
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn plain_text_is_one_token() {
        assert_eq!(kinds("hello"), vec![TokenKind::Text("hello".into())]);
    }

    #[test]
    fn output_and_statement_are_separated_from_text() {
        assert_eq!(
            kinds("a{{ x }}b{% if y %}c"),
            vec![
                TokenKind::Text("a".into()),
                TokenKind::Output("x".into()),
                TokenKind::Text("b".into()),
                TokenKind::Stmt("if y".into()),
                TokenKind::Text("c".into()),
            ]
        );
    }

    #[test]
    fn a_leading_dash_strips_whitespace_before_the_tag() {
        // Llama-3's template is written almost entirely in this form, and a
        // newline that survives when it should not shifts every later token.
        assert_eq!(
            kinds("a  \n{{- x }}"),
            vec![TokenKind::Text("a".into()), TokenKind::Output("x".into())]
        );
    }

    #[test]
    fn a_trailing_dash_strips_whitespace_after_the_tag() {
        assert_eq!(
            kinds("{{ x -}}\n  b"),
            vec![TokenKind::Output("x".into()), TokenKind::Text("b".into())]
        );
    }

    #[test]
    fn whitespace_without_a_dash_is_kept_exactly() {
        // The inverse mistake: stripping when not asked. Gemma's template
        // depends on the newline after its `<start_of_turn>` line surviving.
        assert_eq!(
            kinds("a\n{{ x }}\nb"),
            vec![
                TokenKind::Text("a\n".into()),
                TokenKind::Output("x".into()),
                TokenKind::Text("\nb".into()),
            ]
        );
    }

    #[test]
    fn comments_vanish_entirely() {
        assert_eq!(
            kinds("a{# note #}b"),
            vec![TokenKind::Text("a".into()), TokenKind::Text("b".into())]
        );
    }

    #[test]
    fn an_unterminated_tag_is_an_error_not_a_silent_truncation() {
        let e = lex("a{{ x").unwrap_err();
        assert!(matches!(e, Error::Syntax(_)), "{e:?}");
    }
}
