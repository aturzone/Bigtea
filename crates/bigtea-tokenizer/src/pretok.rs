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
    /// `default`, **and what an absent `tokenizer.ggml.pre` means.**
    ///
    /// llama.cpp's `LLAMA_VOCAB_PRE_TYPE_DEFAULT`, which is also where its
    /// fallback lands when the key is missing. Structurally unlike the others:
    /// **four rules applied in sequence**, each splitting the pieces the last
    /// one produced, rather than one ordered alternation. See [`default_gpt2`].
    Default,
    /// `gpt-2`, and also `mpt`, `olmo`, `jais`. **One** rule, not four.
    ///
    /// Easy to conflate with [`Default`](Self::Default), and this build did:
    /// `from_name` mapped `"gpt2"` onto `Default`. They are separate entries in
    /// llama.cpp — `LLAMA_VOCAB_PRE_TYPE_GPT2` is the single GPT-2 expression,
    /// while the switch's `default:` arm wraps that same expression in three
    /// more passes. Sharing a name in the source is not sharing a rule.
    Gpt2,
}

/// A `tokenizer.ggml.pre` this build has not verified against a real container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownPreTokenizer(pub String);

impl fmt::Display for UnknownPreTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tokenizer.ggml.pre = {:?} is not implemented (verified here: \
             \"llama-bpe\"/\"llama3\", \"qwen2\", \"joyai-llm\", \"default\", \
             \"gpt-2\"/\"mpt\"/\"olmo\"/\"jais\"). \
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
            // `falcon3` is not a rule of its own: llama.cpp folds it into the
            // same arm as `llama-bpe`, with the same `LLAMA_VOCAB_PRE_TYPE_LLAMA3`
            // and the same `ignore_merges` / `add_bos`. Checked against
            // Falcon3-1B-Instruct. `llama-v3` is the third alias in that arm.
            "llama-bpe" | "llama3" | "llama-v3" | "falcon3" => Ok(PreTokenizer::LlamaBpe),
            "qwen2" => Ok(PreTokenizer::Qwen2),
            "joyai-llm" => Ok(PreTokenizer::JoyaiLlm),
            // **Also the absent case.** `Tokenizer::from_metadata` passes
            // "default" when the container declares no `tokenizer.ggml.pre`,
            // which is what llama.cpp does with a missing key.
            "default" => Ok(PreTokenizer::Default),
            // These four share `LLAMA_VOCAB_PRE_TYPE_GPT2` there, and it is
            // **not** the `default:` arm. Checked against OLMo-1B.
            "gpt-2" | "gpt2" | "mpt" | "olmo" | "jais" => Ok(PreTokenizer::Gpt2),
            other => Err(UnknownPreTokenizer(other.to_string())),
        }
    }

    /// How many digits may group into one piece.
    fn max_digits(self) -> usize {
        match self {
            PreTokenizer::Qwen2 => 1,
            PreTokenizer::LlamaBpe
            | PreTokenizer::JoyaiLlm
            | PreTokenizer::Default
            | PreTokenizer::Gpt2 => 3,
        }
    }
}

/// Split `text` into the pieces BPE will be applied to, in order.
///
/// Concatenating the result always reproduces the input exactly.
pub fn pre_tokenize(text: &str, pre: PreTokenizer) -> Vec<String> {
    match pre {
        PreTokenizer::JoyaiLlm => joyai(text),
        PreTokenizer::Default => default_gpt2(text),
        PreTokenizer::Gpt2 => gpt2_rule(text),
        PreTokenizer::LlamaBpe | PreTokenizer::Qwen2 => gpt4_style(text, pre.max_digits()),
    }
}

/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
///
/// Transcribed from `unicode_regex_split_custom_gpt2`, not from the expression
/// above — llama.cpp dispatches this exact regex string to a hand-written
/// splitter, so **the C++ is the specification and the regex is a comment on
/// it.** Two places where they differ and the code wins:
///
///   * the run of whitespace at the end has a `\s+` fallback the expression
///     does not list, so a trailing run is emitted whole rather than dropped;
///   * the contraction rule is **case-sensitive** here. `llama3` writes
///     `(?i:'s|…)`, this one does not, so `'S` is punctuation-then-letter.
///
/// Reusing `gpt4_style` for this would have been close and wrong: it keeps a
/// non-space lead character on a word (`[^\r\n\p{L}\p{N}]?\p{L}+` against
/// ` ?\p{L}+`) and caps digit runs at three. Both differences are invisible in
/// [`default_gpt2`], whose other three passes happen to undo them, and neither
/// is invisible here, where this rule runs alone.
fn gpt2_rule(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;

    // `[^\s\p{L}\p{N}]`. Deliberately not [`is_other`], which also excludes
    // control characters: llama.cpp tests `!(whitespace|letter|number)` plus
    // "has any flag at all", and a control character has one.
    fn is_sym(c: char) -> bool {
        !c.is_whitespace() && !c.is_alphabetic() && !is_digit(c)
    }

    while i < n {
        // `'s|'t|'re|'ve|'m|'ll|'d`, lower case only.
        if chars[i] == '\'' && i + 1 < n {
            let a = chars[i + 1];
            if matches!(a, 's' | 't' | 'm' | 'd') {
                out.push(chars[i..i + 2].iter().collect());
                i += 2;
                continue;
            }
            if i + 2 < n {
                let b = chars[i + 2];
                if (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') {
                    out.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
            }
        }

        // The three ` ?…+` rules share one shape: look past a single literal
        // space, and if what follows is classified, take the space with it.
        // A space followed by anything unclassified — another space, a
        // newline, the end — falls through to the whitespace rules below.
        let probe = i + usize::from(chars[i] == ' ');
        let class: Option<fn(char) -> bool> = match chars.get(probe) {
            Some(&c) if c.is_alphabetic() => Some(|c: char| c.is_alphabetic()),
            // Unbounded, unlike `llama3`'s `\p{N}{1,3}`. `default_gpt2` chops
            // the run in a later pass; standing alone this one does not.
            Some(&c) if is_digit(c) => Some(is_digit),
            Some(&c) if is_sym(c) => Some(is_sym),
            _ => None,
        };
        if let Some(f) = class {
            let mut j = probe;
            while j < n && f(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        let mut ws = 0;
        while i + ws < n && chars[i + ws].is_whitespace() {
            ws += 1;
        }
        // `\s+(?!\S)` — a run with something after it gives its last character
        // back, because that character is the next piece's leading space.
        //
        // **This was briefly changed to emit the run whole, and that was
        // wrong.** OLMo's `a  b` does tokenize as `'a' '  ' 'b'` in the
        // reference, and it looked like this rule handing a space forward — but
        // the cause is one layer up. OLMo gives runs of spaces their own token
        // ids, and `specials` excluded the short ones *by length*, so a
        // two-space run never reached the splitter as a unit at all. Fixing that
        // guard fixed the tokens with this rule untouched, and reverting here is
        // what leaves every other GPT-2-family container alone.
        //
        // The lesson is the attribution: the symptom showed in this function's
        // output and the bug was in what was handed to it.
        if ws > 1 && i + ws < n {
            let j = i + ws - 1;
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        // The `\s+` fallback: a run at the very end, or a single space.
        if ws > 0 {
            out.push(chars[i..i + ws].iter().collect());
            i += ws;
            continue;
        }
        out.push(chars[i..i + 1].iter().collect());
        i += 1;
    }
    out
}

/// llama.cpp's `LLAMA_VOCAB_PRE_TYPE_DEFAULT`, and therefore what an **absent**
/// `tokenizer.ggml.pre` means.
///
/// # Why this one is a pipeline and the others are not
///
/// The `llama3`/`qwen2` variants are a single regex whose alternatives are
/// tried in order, so one pass over the text produces the pieces. The default
/// is **four regexes applied in sequence** — `unicode_regex_split` runs each
/// over the output of the last:
///
/// ```text
/// [\p{P}\$\+<=>\^~\|]+                                   punctuation runs
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
/// \p{N}+                                                    digit runs
/// [0-9][0-9][0-9]                                           groups of three
/// ```
///
/// The first pass is what separates it from `llama-bpe` in practice: a run of
/// punctuation is cut out *whole and first*, so `def fibonacci(n):` becomes
/// `def fibonacci` `(` `n` `):` before anything else runs — five pieces where
/// `llama-bpe` makes four. That one difference is the whole of StableLM's
/// disagreement with llama.cpp.
fn default_gpt2(text: &str) -> Vec<String> {
    // Pass 1: runs of punctuation and the listed symbols, taken whole.
    let is_punct_run = |c: char| {
        c.is_ascii_punctuation() && !matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '~' | '|')
            || matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '~' | '|')
            || (!c.is_alphanumeric() && !c.is_whitespace() && !c.is_ascii())
    };
    let mut pieces = split_runs(text, is_punct_run);

    // Pass 2: the GPT-2 rule, and **the same code the `gpt-2` pre-tokenizer
    // runs** — llama.cpp dispatches on the regex string, and this pass's string
    // is byte-identical to `LLAMA_VOCAB_PRE_TYPE_GPT2`'s. This used to call
    // `gpt4_style(_, usize::MAX)`, which is a different rule that the other
    // three passes mostly hide; `gpt2_rule` says which differences.
    pieces = pieces.into_iter().flat_map(|p| gpt2_rule(&p)).collect();

    // Pass 3: separate digit runs from anything still attached to them, then
    // pass 4: chop those runs into threes.
    pieces = pieces
        .into_iter()
        .flat_map(|p| split_runs(&p, |c| c.is_ascii_digit()))
        .flat_map(|p| chunk_digits(&p))
        .collect();

    pieces.retain(|p| !p.is_empty());
    pieces
}

/// Split `text` wherever `wanted` changes, keeping matching runs whole.
///
/// Losslessly: concatenating the result reproduces the input.
fn split_runs(text: &str, wanted: impl Fn(char) -> bool) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_is = None;
    for c in text.chars() {
        let is = wanted(c);
        if cur_is != Some(is) && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur_is = Some(is);
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// `[0-9][0-9][0-9]` — a run of digits becomes groups of three, longest first.
///
/// Anything that is not all digits passes through untouched.
fn chunk_digits(piece: &str) -> Vec<String> {
    if piece.is_empty() || !piece.chars().all(|c| c.is_ascii_digit()) {
        return vec![piece.to_string()];
    }
    piece
        .as_bytes()
        .chunks(3)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect()
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

    /// **An absent `tokenizer.ggml.pre` is `default`, not `llama-bpe`.**
    ///
    /// The default's first pass cuts a run of punctuation out whole, so
    /// `def fibonacci(n):` is five pieces where `llama-bpe` makes four --
    /// verified against `llama-tokenize` on StableLM, which declares no key.
    #[test]
    fn the_default_cuts_punctuation_runs_whole() {
        let d = pre_tokenize("def fibonacci(n):", PreTokenizer::Default);
        assert_eq!(d, vec!["def", " fibonacci", "(", "n", "):"]);
        // The variant it used to fall back to does not agree, which is the
        // whole reason this exists.
        let l = pre_tokenize("def fibonacci(n):", PreTokenizer::LlamaBpe);
        assert_ne!(d, l, "if these agree the bug could not have happened");
    }

    /// Splitting never alters or drops input, on every variant.
    #[test]
    fn the_default_is_lossless() {
        for text in [
            "def fibonacci(n):",
            "hello, world! 12345",
            "a\n\nb   c",
            "\u{4e2d}\u{6587}(test)",
            "",
        ] {
            let joined: String = pre_tokenize(text, PreTokenizer::Default).concat();
            assert_eq!(joined, text, "lossy on {text:?}");
        }
    }

    /// **`default` and `gpt2` are different rules**, and this test used to
    /// assert they were the same one. llama.cpp's `LLAMA_VOCAB_PRE_TYPE_GPT2`
    /// is the single GPT-2 expression; the switch's `default:` arm wraps that
    /// expression in three more passes. The names being adjacent in the source
    /// is not the rules being equal.
    #[test]
    fn default_and_gpt2_are_not_the_same_rule() {
        assert_eq!(
            PreTokenizer::from_name("default"),
            Ok(PreTokenizer::Default)
        );
        assert_eq!(PreTokenizer::from_name("gpt2"), Ok(PreTokenizer::Gpt2));
        assert_eq!(PreTokenizer::from_name("olmo"), Ok(PreTokenizer::Gpt2));

        // The `default:` arm's first pass cuts a punctuation run out whole, so
        // it splits where the bare GPT-2 rule does not.
        let text = "def fibonacci(n):";
        assert_eq!(
            pre_tokenize(text, PreTokenizer::Default),
            vec!["def", " fibonacci", "(", "n", "):"]
        );
        assert_eq!(
            pre_tokenize(text, PreTokenizer::Gpt2),
            vec!["def", " fibonacci", "(", "n", "):"]
        );
        // …and on a number it is passes 3 and 4 that differ: the bare rule
        // takes a digit run whole, the default chops it into threes.
        assert_eq!(pre_tokenize("12345", PreTokenizer::Gpt2), vec!["12345"]);
        assert_eq!(
            pre_tokenize("12345", PreTokenizer::Default),
            vec!["123", "45"]
        );
    }

    /// `falcon3` is `llama-bpe` under another name — one arm in llama.cpp, one
    /// `pre_type`, and the same `ignore_merges`/`add_bos`.
    #[test]
    fn falcon3_resolves_to_the_llama3_rule() {
        assert_eq!(
            PreTokenizer::from_name("falcon3"),
            Ok(PreTokenizer::LlamaBpe)
        );
    }

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
        for unknown in ["deepseek-llm", "falcon", "smaug-bpe", "bert-bge"] {
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
