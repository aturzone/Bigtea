//! WordPiece (`tokenizer.ggml.model = "bert"`), the BERT/embedding family.
//!
//! Two things make this a different algorithm rather than a variant of the
//! others, and both are easy to get subtly wrong:
//!
//! 1. **It never merges.** BPE builds tokens up from characters by rank and
//!    SentencePiece by score; WordPiece cuts a word down from the left, taking
//!    the longest prefix in the vocabulary, then the longest prefix of what
//!    remains, marked so it is known to continue a word.
//! 2. **It has no byte fallback.** A word it cannot cover becomes one `[UNK]`,
//!    whole. There is no per-byte escape hatch, so the vocabulary decides what
//!    is representable and everything else is flattened to a single id.
//!
//! # The spelling trap
//!
//! HuggingFace writes WordPiece vocabularies as `capital` and `##ization`.
//! **GGUF converters do not.** They rewrite the vocabulary into SentencePiece
//! spelling: `▁capital` starts a word and a bare `ization` continues it. In
//! `all-MiniLM-L6-v2` the strings `capital` and `##ization` are *not in the
//! vocabulary at all*.
//!
//! A textbook `##` implementation therefore matches nothing and every ordinary
//! word becomes `[UNK]` — `"The capital of France is Paris."` tokenizes to
//! `the [UNK] of [UNK] is [UNK] .`. No error, every id valid, output fluent
//! nonsense. That is exactly the failure this crate's own header warns about,
//! and it is why the spelling is *detected from the vocabulary* rather than
//! assumed: a container built directly from a `vocab.txt` really does use `##`.
//!
//! # The preprocessing is not incidental
//!
//! WordPiece is defined over an already-normalised string, and llama.cpp's rule
//! is specific enough that guessing it produces a tokenizer that works on
//! `"hello"` and diverges on real text. Verified against `llama-tokenize` on
//! `all-MiniLM-L6-v2`:
//!
//! ```text
//! "The capital of France is Paris."  ->  the capital of france is paris .
//! "café naïve"                       ->  cafe naive        (accents dropped)
//! "tokenization"                     ->  ▁token ization
//! "北京大学"                          ->  one token per CJK character
//! "hello 🦄 world"                    ->  hello [UNK] world
//! "Ω≈ç√"                             ->  ω ≈ c √
//! ```
//!
//! That last line pins the splitting rule. A codepoint starts a new word if it
//! is punctuation, **or an ASCII symbol**, or CJK — but a *non-ASCII* symbol
//! does not. `≈` and `√` stay glued to the preceding word and come out as
//! continuation pieces. A rule that split on every symbol would look entirely
//! reasonable and be wrong.

use std::collections::HashMap;

/// How a vocabulary spells "this piece continues the previous one".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spelling {
    /// `▁capital` starts a word, a bare `ization` continues it. What GGUF
    /// converters emit, and what llama.cpp matches against.
    Marked,
    /// `capital` starts a word, `##ization` continues it. HuggingFace's
    /// `vocab.txt`, and what a container built directly from one carries.
    Hash,
}

/// The word-start marker, `U+2581 LOWER ONE EIGHTH BLOCK`.
const MARK: char = '\u{2581}';

/// Which spelling a vocabulary uses. Decided once, from the vocabulary itself.
pub fn detect_spelling<S: AsRef<str>>(tokens: &[S]) -> Spelling {
    match tokens.iter().any(|t| t.as_ref().starts_with(MARK)) {
        true => Spelling::Marked,
        false => Spelling::Hash,
    }
}

/// Cut text into WordPiece words, as llama.cpp does before matching.
///
/// Lowercased, accents removed, control characters dropped; punctuation, ASCII
/// symbols and CJK characters each become a word of their own.
pub fn preprocess(text: &str) -> Vec<String> {
    let mut words: Vec<String> = vec![String::new()];
    for ch in text.chars() {
        let cp = ch as u32;
        if ch.is_whitespace() {
            if !words.last().expect("never empty").is_empty() {
                words.push(String::new());
            }
            continue;
        }
        // Controls, NUL and the replacement character carry no text. Keeping
        // one would make its word unmatchable and turn it into `[UNK]`.
        if cp == 0 || cp == 0xFFFD || ch.is_control() || is_combining_mark(cp) {
            continue;
        }
        let base = strip_accent(ch);
        if is_punctuation(cp) || (cp < 0x7F && is_ascii_symbol(cp)) || is_cjk(cp) {
            if !words.last().expect("never empty").is_empty() {
                words.push(String::new());
            }
            words
                .last_mut()
                .expect("never empty")
                .extend(base.to_lowercase());
            words.push(String::new());
        } else {
            words
                .last_mut()
                .expect("never empty")
                .extend(base.to_lowercase());
        }
    }
    words.retain(|w| !w.is_empty());
    words
}

/// Greedy longest-match-first over each word.
///
/// `unk` is emitted for a word no split can cover — for the **whole** word, not
/// for the character that failed, which is what llama.cpp does and what makes
/// `"hello 🦄 world"` three tokens rather than a run of fragments.
pub fn encode(
    text: &str,
    ids: &HashMap<String, u32>,
    unk: Option<u32>,
    spelling: Spelling,
) -> Vec<u32> {
    let mut out = Vec::new();
    for word in preprocess(text) {
        let chars: Vec<char> = word.chars().collect();
        let mut pieces = Vec::new();
        let mut i = 0;
        let mut matched = true;
        while i < chars.len() {
            // Longest first: the vocabulary holds both `▁to` and `▁token`, and
            // taking the shorter one changes every piece that follows.
            let mut end = chars.len();
            let found = loop {
                if end <= i {
                    break None;
                }
                let piece: String = chars[i..end].iter().collect();
                let key = match (spelling, i) {
                    (Spelling::Marked, 0) => format!("{MARK}{piece}"),
                    (Spelling::Hash, 0) => piece,
                    (Spelling::Marked, _) => piece,
                    (Spelling::Hash, _) => format!("##{piece}"),
                };
                if let Some(&id) = ids.get(&key) {
                    break Some((id, end));
                }
                end -= 1;
            };
            match found {
                Some((id, next)) => {
                    pieces.push(id);
                    i = next;
                }
                None => {
                    matched = false;
                    break;
                }
            }
        }
        match matched {
            true => out.extend(pieces),
            false => out.extend(unk),
        }
    }
    out
}

/// Join pieces back into text, undoing whichever marker is in use.
///
/// Lossy by construction: case and accents were destroyed at encode time and no
/// detokenizer can restore them. A round-trip test demanding equality here is
/// testing the wrong thing.
pub fn decode(pieces: &[&str], spelling: Spelling) -> String {
    let mut out = String::new();
    for piece in pieces {
        let (starts_word, text) = match spelling {
            Spelling::Marked => (piece.starts_with(MARK), piece.trim_start_matches(MARK)),
            Spelling::Hash => match piece.strip_prefix("##") {
                Some(rest) => (false, rest),
                None => (true, *piece),
            },
        };
        if starts_word && !out.is_empty() {
            out.push(' ');
        }
        out.push_str(text);
    }
    out
}

/// Combining marks, dropped so `e` + U+0301 and `é` reach the same word.
///
/// The precomposed form is handled by [`strip_accent`]; this covers text that
/// arrives already decomposed, and the marks NFD would have produced.
fn is_combining_mark(cp: u32) -> bool {
    matches!(cp,
        0x0300..=0x036F      // combining diacritical marks
        | 0x1AB0..=0x1AFF    // extended
        | 0x1DC0..=0x1DFF    // supplement
        | 0x20D0..=0x20FF    // for symbols
        | 0xFE20..=0xFE2F    // half marks
    )
}

/// Precomposed Latin letters to their unaccented base.
///
/// The decomposition half of NFD, restricted to the range that matters: Latin-1
/// Supplement and Latin Extended-A, which covers every accented character in
/// European text. **Scripts whose marks are not here — Arabic harakat,
/// Devanagari matras — are left alone**, where llama.cpp would strip them. Those
/// characters are almost never in a WordPiece vocabulary and become `[UNK]`
/// either way, but it is a real difference and is written down rather than
/// assumed away.
fn strip_accent(ch: char) -> char {
    let cp = ch as u32;
    // Latin-1 Supplement, U+00C0..U+00FF, in codepoint order.
    const LATIN1: &str = "AAAAAAÆCEEEEIIIIÐNOOOOO×ØUUUUYÞßaaaaaaæceeeeiiiiðnooooo÷øuuuuypy";
    if (0x00C0..=0x00FF).contains(&cp) {
        if let Some(c) = LATIN1.chars().nth((cp - 0x00C0) as usize) {
            return c;
        }
    }
    // Latin Extended-A, U+0100..U+017F: upper/lower pairs over a small set of
    // bases, so a table indexed by codepoint is shorter and less error-prone
    // than 128 match arms.
    const EXT_A: &str = concat!(
        "AaAaAaCcCcCcCcDdDdEeEeEeEeEeGgGgGgGgHhHhIiIiIiIiIi",
        "JjKkkLlLlLlLlLlNnNnNnnNnOoOoOoRrRrRrSsSsSsSsTtTtTt",
        "UuUuUuUuUuUuWwYyYZzZzZzs"
    );
    if (0x0100..=0x017F).contains(&cp) {
        if let Some(c) = EXT_A.chars().nth((cp - 0x0100) as usize) {
            return c;
        }
    }
    ch
}

/// ASCII symbols — Unicode `S*` inside ASCII. These split; their non-ASCII kin
/// do not.
fn is_ascii_symbol(cp: u32) -> bool {
    matches!(
        cp as u8,
        b'$' | b'+' | b'<' | b'=' | b'>' | b'^' | b'`' | b'|' | b'~'
    )
}

/// Punctuation, ASCII and the main Unicode blocks.
///
/// Range-based rather than a full category table, which the workspace's
/// no-dependency rule rules out. Covers ASCII, Latin-1 punctuation, General
/// Punctuation, CJK punctuation and the fullwidth forms — everything a
/// WordPiece vocabulary is likely to hold a token for.
fn is_punctuation(cp: u32) -> bool {
    if cp < 0x80 {
        let b = cp as u8;
        return b.is_ascii_punctuation() && !is_ascii_symbol(cp);
    }
    matches!(cp,
        0x00A1 | 0x00A7 | 0x00AB | 0x00B6 | 0x00B7 | 0x00BB | 0x00BF
        | 0x2010..=0x2027
        | 0x2030..=0x205E
        | 0x3001..=0x3003
        | 0x300C..=0x300F
        | 0x301D..=0x301F
        | 0xFF01..=0xFF0F
        | 0xFF1A..=0xFF20
    )
}

/// The CJK ranges llama.cpp treats as one-character words.
fn is_cjk(cp: u32) -> bool {
    matches!(cp,
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0xF900..=0xFAFF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0x2B920..=0x2CEAF
        | 0x2F800..=0x2FA1F
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(words: &[&str]) -> HashMap<String, u32> {
        words
            .iter()
            .enumerate()
            .map(|(i, w)| ((*w).to_string(), i as u32))
            .collect()
    }

    #[test]
    fn splits_off_punctuation_and_lowercases() {
        assert_eq!(preprocess("Hello, World!"), ["hello", ",", "world", "!"]);
    }

    #[test]
    fn collapses_runs_of_whitespace() {
        assert_eq!(preprocess("  spaced   out  "), ["spaced", "out"]);
    }

    #[test]
    fn gives_each_cjk_character_its_own_word() {
        assert_eq!(preprocess("北京大学"), ["北", "京", "大", "学"]);
    }

    #[test]
    fn drops_accents_precomposed_and_decomposed() {
        assert_eq!(preprocess("café"), ["cafe"]);
        assert_eq!(preprocess("cafe\u{0301}"), ["cafe"]);
        assert_eq!(preprocess("naïve"), ["naive"]);
    }

    /// The rule `Ω≈ç√` pinned: non-ASCII symbols stay inside the word.
    #[test]
    fn ascii_symbols_split_but_non_ascii_symbols_do_not() {
        assert_eq!(preprocess("a=b"), ["a", "=", "b"]);
        assert_eq!(preprocess("a≈b"), ["a≈b"]);
    }

    /// The spelling a GGUF converter actually emits.
    #[test]
    fn marked_spelling_starts_words_with_the_marker() {
        let v = vocab(&["\u{2581}token", "ization", "[UNK]"]);
        assert_eq!(detect_spelling(&["\u{2581}token"]), Spelling::Marked);
        assert_eq!(
            encode("tokenization", &v, Some(2), Spelling::Marked),
            [0, 1]
        );
    }

    /// ...and the one a container built from `vocab.txt` carries.
    #[test]
    fn hash_spelling_marks_continuations_instead() {
        let v = vocab(&["token", "##ization", "[UNK]"]);
        assert_eq!(detect_spelling(&["token", "##ization"]), Spelling::Hash);
        assert_eq!(encode("tokenization", &v, Some(2), Spelling::Hash), [0, 1]);
    }

    /// The bug this cost an hour: `##` matching against a `▁` vocabulary finds
    /// nothing, so every ordinary word becomes UNK — silently.
    #[test]
    fn the_wrong_spelling_unks_everything_rather_than_failing() {
        let v = vocab(&["\u{2581}token", "ization", "[UNK]"]);
        assert_eq!(encode("tokenization", &v, Some(2), Spelling::Hash), [2]);
    }

    #[test]
    fn prefers_the_longest_piece() {
        // `▁to` also matches; taking it would change everything after.
        let v = vocab(&["\u{2581}to", "\u{2581}token", "ization", "s", "[UNK]"]);
        assert_eq!(
            encode("tokenization", &v, Some(4), Spelling::Marked),
            [1, 2]
        );
    }

    /// An uncoverable word is one `[UNK]`, not a run of fragments — and the
    /// words around it are unaffected.
    #[test]
    fn an_uncoverable_word_becomes_one_unk() {
        let v = vocab(&["\u{2581}hello", "\u{2581}world", "[UNK]"]);
        assert_eq!(
            encode("hello 🦄 world", &v, Some(2), Spelling::Marked),
            [0, 2, 1]
        );
    }

    #[test]
    fn a_partially_matchable_word_is_still_one_unk() {
        // `▁un` matches, nothing covers the rest; the whole word is UNK.
        let v = vocab(&["\u{2581}un", "[UNK]"]);
        assert_eq!(encode("unobtainium", &v, Some(1), Spelling::Marked), [1]);
    }

    #[test]
    fn without_an_unk_id_an_uncoverable_word_vanishes_rather_than_panicking() {
        let v = vocab(&["\u{2581}hello"]);
        assert_eq!(encode("hello 🦄", &v, None, Spelling::Marked), [0]);
    }

    #[test]
    fn decode_undoes_both_spellings() {
        assert_eq!(
            decode(&["token", "##ization", "is", "fun"], Spelling::Hash),
            "tokenization is fun"
        );
        assert_eq!(
            decode(
                &["\u{2581}token", "ization", "\u{2581}is", "\u{2581}fun"],
                Spelling::Marked
            ),
            "tokenization is fun"
        );
    }

    #[test]
    fn empty_text_encodes_to_nothing() {
        let v = vocab(&["a", "[UNK]"]);
        assert!(encode("", &v, Some(1), Spelling::Marked).is_empty());
        assert!(encode("   ", &v, Some(1), Spelling::Marked).is_empty());
    }
}
