//! Splitting text before BPE runs, chosen by `tokenizer.ggml.pre`.
//!
//! BPE never merges across a split boundary, so this decides which merges are
//! even *possible*. Get it wrong and the ids differ from what the model was
//! trained on; the model then predicts a fluent continuation of the wrong tokens,
//! which looks like a broken forward pass rather than a broken splitter.
//!
//! # The variants are not interchangeable, and the container says which
//!
//! Measured with `llama-tokenize` on the models in this repository:
//!
//! ```text
//!              "4567"            "12345678"             "don't"
//! qwen2        4 5 6 7           1 2 3 4 5 6 7 8        don 't
//! llama-bpe    456 7             123 456 78             don 't
//! ```
//!
//! One digit at a time against groups of three. Every number in a prompt
//! tokenizes differently, and every boundary after it shifts. `tokenizer.ggml.pre`
//! was previously **ignored**, so a Qwen container was split with Llama's rule
//! and a contraction was cut into three pieces (`don`, `'`, `t`) where both
//! reference implementations produce two.
//!
//! # Why this is hand-written
//!
//! The patterns need negative lookahead (`\s+(?!\S)`) and case-insensitive
//! alternation, and the workspace has no external dependencies. Each variant is
//! therefore an ordered list of rules tried at each position — which is what an
//! alternation *is*, so the structure mirrors the regex rather than
//! reinterpreting it.

use std::fmt;

/// Which splitting rule a container asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizer {
    /// `llama-bpe` / `llama3`. GPT-4 style: contractions, `\p{N}{1,3}`.
    LlamaBpe,
    /// `qwen2`. As above but **one digit at a time**.
    Qwen2,
    /// `joyai-llm`, DeepSeek-V4-Flash. Adds a CJK rule and has no contraction
    /// rule; verified against llama.cpp on that model.
    JoyaiLlm,
}

/// A `tokenizer.ggml.pre` this build has not verified against a real container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownPreTokenizer(pub String);

impl fmt::Display for UnknownPreTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tokenizer.ggml.pre = {:?} is not implemented (verified here: \
             \"llama-bpe\"/\"llama3\", \"qwen2\", \"joyai-llm\"). \
             The pre-tokenizer decides where BPE may merge, so guessing one \
             shifts every token boundary and the model answers fluently and \
             wrongly rather than failing — which is why this refuses instead. \
             Adding it needs the variant's rules and a container to check them \
             against.",
            self.0
        )
    }
}

impl std::error::Error for UnknownPreTokenizer {}

impl PreTokenizer {
    /// Resolve the metadata string, refusing anything untested.
    pub fn from_name(name: &str) -> Result<Self, UnknownPreTokenizer> {
        match name {
            "llama-bpe" | "llama3" => Ok(PreTokenizer::LlamaBpe),
            "qwen2" => Ok(PreTokenizer::Qwen2),
            "joyai-llm" => Ok(PreTokenizer::JoyaiLlm),
            other => Err(UnknownPreTokenizer(other.to_string())),
        }
    }

    /// How many digits may group into one piece.
    fn max_digits(self) -> usize {
        match self {
            PreTokenizer::Qwen2 => 1,
            PreTokenizer::LlamaBpe | PreTokenizer::JoyaiLlm => 3,
        }
    }
}

/// Split `text` into the pieces BPE will be applied to, in order.
///
/// Concatenating the result always reproduces the input exactly.
pub fn pre_tokenize(text: &str, pre: PreTokenizer) -> Vec<String> {
    match pre {
        PreTokenizer::JoyaiLlm => joyai(text),
        PreTokenizer::LlamaBpe | PreTokenizer::Qwen2 => gpt4_style(text, pre.max_digits()),
    }
}

/// The contractions both GPT-4-style variants match, case-insensitively.
const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

/// `(?i:'s|'t|…)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,n}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
///
/// Rules are tried in that order at every position, which is what makes the
/// alternation's order meaningful: the contraction rule must win over the
/// punctuation rule or `'t` becomes `'` then `t`.
fn gpt4_style(text: &str, max_digits: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        // 1. A contraction, longest first so `'re` is not cut to `'r`.
        if let Some(len) = contraction_at(&chars, i) {
            out.push(chars[i..i + len].iter().collect());
            i += len;
            continue;
        }

        // 2. `[^\r\n\p{L}\p{N}]?\p{L}+` — one optional non-letter, non-digit,
        //    non-newline character, then letters. This is what keeps the leading
        //    space on a word and produces the `Ġword` tokens the vocabulary is
        //    built from.
        let lead =
            usize::from(!is_newline(chars[i]) && !chars[i].is_alphabetic() && !is_digit(chars[i]));
        if i + lead < chars.len() && chars[i + lead].is_alphabetic() {
            let mut j = i + lead;
            while j < chars.len() && chars[j].is_alphabetic() {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        // 3. Digits, at most `max_digits`. **This is the qwen2 difference.**
        if is_digit(chars[i]) {
            let mut j = i;
            while j < chars.len() && is_digit(chars[j]) && j - i < max_digits {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*` — optional space, punctuation or
        //    symbols, then any newlines.
        let lead = usize::from(chars[i] == ' ' && i + 1 < chars.len() && is_other(chars[i + 1]));
        if i + lead < chars.len() && is_other(chars[i + lead]) {
            let mut j = i + lead;
            while j < chars.len() && is_other(chars[j]) {
                j += 1;
            }
            while j < chars.len() && is_newline(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        // 5. `\s*[\r\n]+` — whitespace run ending in newlines.
        if chars[i].is_whitespace() {
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() && !is_newline(chars[j]) {
                j += 1;
            }
            if j < chars.len() && is_newline(chars[j]) {
                while j < chars.len() && is_newline(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }

            // 6. `\s+(?!\S)` — a whitespace run with nothing after it.
            if j >= chars.len() {
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }

            // 7. `\s+`, but the last space belongs to whatever follows, which is
            //    rule 2's optional lead. Emitting the whole run here would strip
            //    every leading space and lose almost every merge.
            let split = j - 1;
            if split > i {
                out.push(chars[i..split].iter().collect());
                i = split;
                continue;
            }
            // A single space with a non-letter after it: it stands alone.
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        // Unclassified: stand alone rather than be dropped.
        out.push(chars[i..i + 1].iter().collect());
        i += 1;
    }
    out
}

/// Length in `chars` of a contraction starting at `i`, if any.
fn contraction_at(chars: &[char], i: usize) -> Option<usize> {
    if chars[i] != '\'' {
        return None;
    }
    // Longest first: `'re` must not be taken as `'r`... and there is no `'r`,
    // but `'ll` versus `'l` is the same shape and the ordering costs nothing.
    let mut best: Option<usize> = None;
    for c in CONTRACTIONS {
        let n = c.chars().count();
        if i + n > chars.len() {
            continue;
        }
        let matches = c
            .chars()
            .zip(&chars[i..i + n])
            .all(|(a, b)| a.eq_ignore_ascii_case(b));
        if matches && best.is_none_or(|b| n > b) {
            best = Some(n);
        }
    }
    best
}

/// DeepSeek-V4-Flash's `joyai-llm`, unchanged and still verified against it.
///
/// ```text
/// \p{N}{1,3}
/// [一-龥぀-ゟ゠-ヿ]+
/// [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
/// ```
fn joyai(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let start = i;

        if is_digit(chars[i]) {
            let mut n = 0;
            while i < chars.len() && is_digit(chars[i]) && n < 3 {
                i += 1;
                n += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        if is_cjk(chars[i]) {
            while i < chars.len() && is_cjk(chars[i]) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        if chars[i].is_whitespace() {
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() && !is_newline(chars[j]) {
                j += 1;
            }
            if j < chars.len() && is_newline(chars[j]) {
                while j < chars.len() && is_newline(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }

            let ws_end = j;
            if ws_end >= chars.len() {
                out.push(chars[i..ws_end].iter().collect());
                i = ws_end;
                continue;
            }
            let split = ws_end - 1;
            if split > i {
                out.push(chars[i..split].iter().collect());
                i = split;
            }
            let lead = i;
            i += 1;
            if i < chars.len() && (is_letter(chars[i]) || is_mark(chars[i])) {
                while i < chars.len() && (is_letter(chars[i]) || is_mark(chars[i])) {
                    i += 1;
                }
            } else if i < chars.len() && is_other(chars[i]) {
                while i < chars.len() && is_other(chars[i]) {
                    i += 1;
                }
                while i < chars.len() && is_newline(chars[i]) {
                    i += 1;
                }
            }
            out.push(chars[lead..i].iter().collect());
            continue;
        }

        if is_other(chars[i]) {
            while i < chars.len() && is_other(chars[i]) {
                i += 1;
            }
            while i < chars.len() && is_newline(chars[i]) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        if is_letter(chars[i]) || is_mark(chars[i]) {
            while i < chars.len() && (is_letter(chars[i]) || is_mark(chars[i])) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        i += 1;
        out.push(chars[start..i].iter().collect());
    }
    out
}

fn is_digit(c: char) -> bool {
    c.is_ascii_digit() || matches!(c, '\u{0660}'..='\u{0669}' | '\u{06F0}'..='\u{06F9}')
}

fn is_newline(c: char) -> bool {
    c == '\n' || c == '\r'
}

/// CJK ideographs and the Japanese kana blocks `joyai-llm` names.
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FA5}'
        | '\u{3040}'..='\u{309F}'
        | '\u{30A0}'..='\u{30FF}'
    )
}

/// Approximates `\p{L}`. Exact for ASCII; beyond it uses Rust's Unicode tables.
fn is_letter(c: char) -> bool {
    c.is_alphabetic() && !is_cjk(c)
}

/// Approximates `\p{M}` (combining marks).
fn is_mark(c: char) -> bool {
    matches!(c,
        '\u{0300}'..='\u{036F}'
        | '\u{1AB0}'..='\u{1AFF}'
        | '\u{20D0}'..='\u{20FF}'
        | '\u{FE20}'..='\u{FE2F}'
    )
}

/// `[^\s\p{L}\p{N}]`: printable, and neither letter, digit nor whitespace.
fn is_other(c: char) -> bool {
    !c.is_alphanumeric() && !c.is_whitespace() && !c.is_control()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_lossless(text: &str, pre: PreTokenizer) {
        let joined: String = pre_tokenize(text, pre).concat();
        assert_eq!(joined, text, "pre-tokenizing changed the text ({pre:?})");
    }

    /// The invariant that matters most: splitting never loses or alters input.
    #[test]
    fn splitting_is_lossless_for_every_variant() {
        for pre in [
            PreTokenizer::LlamaBpe,
            PreTokenizer::Qwen2,
            PreTokenizer::JoyaiLlm,
        ] {
            for text in [
                "The capital of France is",
                "hello, world!",
                "  leading and trailing  ",
                "line one\nline two\n\nline four",
                "tabs\there",
                "unicode: héllo — naïve café",
                "日本語のテキスト",
                "mixed 123 numbers 4567 here",
                "don't It's they're we've I'm you'll he'd",
                "",
                " ",
                "!!!",
                "a\n\n\nb",
                "(parens) [brackets] {braces}",
            ] {
                assert_lossless(text, pre);
            }
        }
    }

    /// **The qwen2 difference**, and the reason `pre` cannot be ignored.
    #[test]
    fn digit_grouping_differs_by_variant() {
        assert_eq!(pre_tokenize("4567", PreTokenizer::LlamaBpe), ["456", "7"]);
        assert_eq!(
            pre_tokenize("4567", PreTokenizer::Qwen2),
            ["4", "5", "6", "7"]
        );
        assert_eq!(
            pre_tokenize("12345678", PreTokenizer::LlamaBpe),
            ["123", "456", "78"]
        );
    }

    /// A contraction is one piece, not `'` then a letter.
    #[test]
    fn contractions_stay_whole() {
        for pre in [PreTokenizer::LlamaBpe, PreTokenizer::Qwen2] {
            assert_eq!(pre_tokenize("don't", pre), ["don", "'t"]);
            assert_eq!(pre_tokenize("It's", pre), ["It", "'s"]);
            assert_eq!(pre_tokenize("they're", pre), ["they", "'re"]);
            assert_eq!(pre_tokenize("we've", pre), ["we", "'ve"]);
            assert_eq!(pre_tokenize("you'll", pre), ["you", "'ll"]);
        }
    }

    /// llama.cpp matches these case-insensitively.
    #[test]
    fn contractions_are_case_insensitive() {
        assert_eq!(pre_tokenize("DON'T", PreTokenizer::LlamaBpe), ["DON", "'T"]);
    }

    #[test]
    fn a_leading_space_stays_with_its_word() {
        for pre in [
            PreTokenizer::LlamaBpe,
            PreTokenizer::Qwen2,
            PreTokenizer::JoyaiLlm,
        ] {
            assert_eq!(
                pre_tokenize("the capital of", pre),
                ["the", " capital", " of"],
                "{pre:?}"
            );
        }
    }

    #[test]
    fn punctuation_separates_from_words() {
        assert_eq!(
            pre_tokenize("hello, world!", PreTokenizer::LlamaBpe),
            ["hello", ",", " world", "!"]
        );
    }

    #[test]
    fn newlines_group_together() {
        assert_eq!(
            pre_tokenize("a\n\nb", PreTokenizer::LlamaBpe),
            ["a", "\n\n", "b"]
        );
    }

    #[test]
    fn cjk_runs_are_their_own_piece_under_joyai() {
        let parts = pre_tokenize("hi 日本語 ok", PreTokenizer::JoyaiLlm);
        assert!(parts.iter().any(|p| p == "日本語"), "got {parts:?}");
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert!(pre_tokenize("", PreTokenizer::LlamaBpe).is_empty());
    }

    #[test]
    fn names_resolve_and_unknown_ones_are_refused() {
        assert_eq!(
            PreTokenizer::from_name("llama-bpe"),
            Ok(PreTokenizer::LlamaBpe)
        );
        assert_eq!(
            PreTokenizer::from_name("llama3"),
            Ok(PreTokenizer::LlamaBpe)
        );
        assert_eq!(PreTokenizer::from_name("qwen2"), Ok(PreTokenizer::Qwen2));
        assert_eq!(
            PreTokenizer::from_name("joyai-llm"),
            Ok(PreTokenizer::JoyaiLlm)
        );
        // Real llama.cpp variants this build has no container to check against.
        for unknown in ["deepseek-llm", "falcon", "default", "gpt-2", "smaug-bpe"] {
            let err = PreTokenizer::from_name(unknown).expect_err("must refuse");
            assert_eq!(err.0, unknown);
            let text = err.to_string();
            assert!(text.contains(unknown), "the message must name it: {text}");
            assert!(
                text.contains("not implemented"),
                "the message must say so: {text}"
            );
        }
    }
}
