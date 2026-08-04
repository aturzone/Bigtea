//! Splitting text before BPE runs.
//!
//! BPE never merges across a split boundary, so this decides which merges are
//! even possible. Get it wrong and the token ids differ from what the model
//! was trained on — the model then predicts a fluent continuation of the wrong
//! tokens, which looks like a broken forward pass rather than a broken
//! splitter.
//!
//! This implements the three patterns DeepSeek-V4-Flash declares
//! (`tokenizer.ggml.pre = "joyai-llm"`), hand-written rather than fed to a
//! regex engine because the third pattern needs negative lookahead
//! (`\s+(?!\S)`), which the usual crate does not support:
//!
//! ```text
//! \p{N}{1,3}
//! [一-龥぀-ゟ゠-ヿ]+
//! [!-~ punctuation][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//!   | ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//! ```
//!
//! Character classes are resolved from Unicode ranges directly. ASCII — which
//! dominates English prompts — is exact; the wider Unicode classes cover the
//! common blocks and are documented as approximate where they are.

/// Split `text` into the pieces BPE will be applied to, in order.
///
/// Concatenating the result always reproduces the input exactly.
pub fn pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let start = i;

        // 1. Runs of digits, at most three -- keeps numbers from merging into
        //    arbitrarily long tokens.
        if is_digit(chars[i]) {
            let mut n = 0;
            while i < chars.len() && is_digit(chars[i]) && n < 3 {
                i += 1;
                n += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        // 2. CJK runs.
        if is_cjk(chars[i]) {
            while i < chars.len() && is_cjk(chars[i]) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        // 3a. Line breaks, with any leading whitespace.
        if is_space(chars[i]) {
            let mut j = i;
            while j < chars.len() && is_space(chars[j]) && !is_newline(chars[j]) {
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

            // 3b. `\s+(?!\S)` -- whitespace with no following non-space, i.e.
            //     a trailing run. Otherwise a run of spaces attaches to the
            //     word that follows it, which is what produces 'Ġword' tokens.
            let ws_end = j;
            if ws_end >= chars.len() {
                out.push(chars[i..ws_end].iter().collect());
                i = ws_end;
                continue;
            }
            // Leave exactly one space to lead the next piece.
            let split = ws_end - 1;
            if split > i {
                out.push(chars[i..split].iter().collect());
                i = split;
            }
            // Fall through: the single space joins the following token.
            let lead = i;
            i += 1;
            if i < chars.len() && (is_letter(chars[i]) || is_mark(chars[i])) {
                while i < chars.len() && (is_letter(chars[i]) || is_mark(chars[i])) {
                    i += 1;
                }
            } else if i < chars.len() && is_punct_or_symbol(chars[i]) {
                while i < chars.len() && is_punct_or_symbol(chars[i]) {
                    i += 1;
                }
                while i < chars.len() && is_newline(chars[i]) {
                    i += 1;
                }
            }
            out.push(chars[lead..i].iter().collect());
            continue;
        }

        // 4. Punctuation or symbols, with any trailing newlines.
        if is_punct_or_symbol(chars[i]) {
            while i < chars.len() && is_punct_or_symbol(chars[i]) {
                i += 1;
            }
            while i < chars.len() && is_newline(chars[i]) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        // 5. Letters and combining marks.
        if is_letter(chars[i]) || is_mark(chars[i]) {
            while i < chars.len() && (is_letter(chars[i]) || is_mark(chars[i])) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        // Anything unclassified stands alone rather than being dropped.
        i += 1;
        out.push(chars[start..i].iter().collect());
    }
    out
}

fn is_digit(c: char) -> bool {
    c.is_ascii_digit() || matches!(c, '\u{0660}'..='\u{0669}' | '\u{06F0}'..='\u{06F9}')
}

fn is_space(c: char) -> bool {
    c.is_whitespace()
}

fn is_newline(c: char) -> bool {
    c == '\n' || c == '\r'
}

/// CJK ideographs and the Japanese kana blocks named in the pattern.
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FA5}'   // 一-龥
        | '\u{3040}'..='\u{309F}' // ぀-ゟ hiragana
        | '\u{30A0}'..='\u{30FF}' // ゠-ヿ katakana
    )
}

/// Approximates `\p{L}`. Exact for ASCII; covers the common alphabetic blocks
/// beyond it via Rust's own Unicode tables.
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

/// Approximates `\p{P}` and `\p{S}` together: anything printable that is
/// neither a letter, a digit, nor whitespace.
fn is_punct_or_symbol(c: char) -> bool {
    !c.is_alphanumeric() && !c.is_whitespace() && !c.is_control()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The invariant that matters most: splitting must never lose or alter
    /// input. A tokenizer that drops a character produces subtly wrong ids.
    fn assert_lossless(text: &str) {
        let joined: String = pre_tokenize(text).concat();
        assert_eq!(joined, text, "pre-tokenizing changed the text");
    }

    #[test]
    fn splitting_is_lossless() {
        for text in [
            "The capital of France is",
            "hello, world!",
            "  leading and trailing  ",
            "line one\nline two\n\nline four",
            "tabs\there",
            "unicode: héllo — naïve café",
            "日本語のテキスト",
            "mixed 123 numbers 4567 here",
            "",
            " ",
            "!!!",
        ] {
            assert_lossless(text);
        }
    }

    #[test]
    fn a_leading_space_stays_with_its_word() {
        // This is what produces the 'Ġword' tokens a GPT-2 vocabulary is built
        // from. Splitting the space off instead would miss almost every merge.
        let parts = pre_tokenize("the capital of");
        assert_eq!(parts, vec!["the", " capital", " of"]);
    }

    #[test]
    fn digits_split_into_groups_of_at_most_three() {
        assert_eq!(pre_tokenize("4567"), vec!["456", "7"]);
        assert_eq!(pre_tokenize("12"), vec!["12"]);
    }

    #[test]
    fn punctuation_separates_from_words() {
        assert_eq!(pre_tokenize("hello, world!"), vec!["hello", ",", " world", "!"]);
    }

    #[test]
    fn newlines_group_together() {
        let parts = pre_tokenize("a\n\nb");
        assert_eq!(parts, vec!["a", "\n\n", "b"]);
    }

    #[test]
    fn cjk_runs_are_their_own_piece() {
        let parts = pre_tokenize("hi 日本語 ok");
        assert!(parts.iter().any(|p| p == "日本語"), "got {parts:?}");
        assert_lossless("hi 日本語 ok");
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert!(pre_tokenize("").is_empty());
    }
}
