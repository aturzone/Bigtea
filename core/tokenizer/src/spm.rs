//! SentencePiece, as GGUF stores it (`tokenizer.ggml.model = "llama"`).
//!
//! # Why this is a separate algorithm and not a variant of BPE
//!
//! Byte-level BPE merges the adjacent pair with the **lowest rank**, where rank
//! comes from an explicit ordered merge table. SentencePiece has no merge table:
//! it merges the adjacent pair whose **concatenation scores highest** in the
//! vocabulary itself. Same shape of loop, entirely different decision, and a
//! vocabulary that carries scores usually carries no merges at all.
//!
//! Three more differences, each of which silently changes the token stream:
//!
//! * **Space is `▁` (U+2581)**, not `Ġ`, and it is substituted before anything
//!   else happens.
//! * **A dummy space is prepended** to the text. Without it the first word
//!   tokenizes differently from the same word mid-sentence, which is a small
//!   difference that compounds over a prompt.
//! * **Unknown text falls back to byte tokens** spelled `<0xF0>`, not to
//!   single characters. A vocabulary of 32000 pieces does not contain every
//!   character, so this path is reached in normal use, not only on junk.
//!
//! Getting any of these wrong produces a valid-looking token stream and fluent
//! nonsense out of the model, so each is tested on its own below.

use std::collections::HashMap;

/// The SentencePiece space marker, U+2581 LOWER ONE EIGHTH BLOCK.
pub const SPACE: char = '\u{2581}';

/// One symbol in the working list: a slice of the text plus its neighbours.
///
/// A doubly-linked list held in a `Vec` rather than with pointers: merging is
/// "extend left, unlink right", which is O(1) here, and indices survive the
/// merges that would invalidate references.
#[derive(Clone, Copy)]
struct Symbol {
    prev: i32,
    next: i32,
    /// Byte offset into the normalised text.
    start: usize,
    len: usize,
}

/// A candidate merge, ordered by score then by position.
#[derive(PartialEq)]
struct Bigram {
    left: i32,
    right: i32,
    score: f32,
    /// Length at the time it was queued; a stale entry is discarded on pop.
    size: usize,
}

impl Eq for Bigram {}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher score wins. Ties go to the **leftmost** pair, so tokenization
        // is deterministic — a max-heap would otherwise pick arbitrarily
        // between equal scores and two runs of the same text could differ.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Replace spaces with `▁` and prepend one, the way SentencePiece expects.
pub fn escape_whitespace(text: &str, add_dummy_prefix: bool) -> String {
    let mut out = String::with_capacity(text.len() + 3);
    if add_dummy_prefix {
        out.push(SPACE);
    }
    for ch in text.chars() {
        if ch == ' ' {
            out.push(SPACE);
        } else {
            out.push(ch);
        }
    }
    out
}

/// Undo [`escape_whitespace`] on decoded text.
pub fn unescape_whitespace(text: &str) -> String {
    text.replace(SPACE, " ")
}

/// The byte-fallback token for one byte, e.g. `<0x0A>`.
///
/// Upper-case hex, two digits, always — `<0xa>` and `<0xA>` are both absent
/// from every vocabulary that uses this convention.
pub fn byte_token(b: u8) -> String {
    format!("<0x{b:02X}>")
}

/// Encode with the SentencePiece merge rule.
///
/// `ids` maps piece text to id; `scores` is indexed by id. Returns ids, with
/// byte fallback for anything the vocabulary cannot spell.
pub fn encode(
    text: &str,
    ids: &HashMap<String, u32>,
    scores: &[f32],
    add_dummy_prefix: bool,
) -> Vec<u32> {
    let text = escape_whitespace(text, add_dummy_prefix);
    if text.is_empty() {
        return Vec::new();
    }

    // One symbol per UTF-8 character. Splitting by byte would merge across
    // character boundaries and produce pieces no vocabulary contains.
    let mut symbols: Vec<Symbol> = Vec::new();
    for (start, ch) in text.char_indices() {
        let i = symbols.len() as i32;
        symbols.push(Symbol {
            prev: i - 1,
            next: i + 1,
            start,
            len: ch.len_utf8(),
        });
    }
    if let Some(last) = symbols.last_mut() {
        last.next = -1;
    }

    let piece = |s: &Symbol| &text[s.start..s.start + s.len];
    let mut queue = std::collections::BinaryHeap::new();
    let push = |queue: &mut std::collections::BinaryHeap<Bigram>,
                symbols: &[Symbol],
                left: i32,
                right: i32| {
        if left == -1 || right == -1 {
            return;
        }
        let (l, r) = (&symbols[left as usize], &symbols[right as usize]);
        // The two are adjacent in the text, so the merged piece is one slice
        // rather than a concatenation — no allocation on the hot path.
        let merged = &text[l.start..r.start + r.len];
        if let Some(&id) = ids.get(merged) {
            queue.push(Bigram {
                left,
                right,
                score: scores.get(id as usize).copied().unwrap_or(0.0),
                size: merged.len(),
            });
        }
    };

    for i in 1..symbols.len() as i32 {
        push(&mut queue, &symbols, i - 1, i);
    }

    while let Some(b) = queue.pop() {
        let (li, ri) = (b.left as usize, b.right as usize);
        let (left, right) = (symbols[li], symbols[ri]);
        // Stale: one side was already merged into something else. The queue is
        // never cleaned up, so every pop has to re-validate.
        if left.len == 0 || right.len == 0 || left.len + right.len != b.size {
            continue;
        }

        symbols[li].len = left.len + right.len;
        symbols[ri].len = 0;
        symbols[li].next = right.next;
        if right.next != -1 {
            symbols[right.next as usize].prev = b.left;
        }
        push(&mut queue, &symbols, symbols[li].prev, b.left);
        push(&mut queue, &symbols, b.left, symbols[li].next);
    }

    let mut out = Vec::new();
    let mut i = 0i32;
    while i != -1 {
        let s = symbols[i as usize];
        if s.len > 0 {
            match ids.get(piece(&s)) {
                Some(&id) => out.push(id),
                // Not in the vocabulary: spell it one byte at a time. A piece
                // this reaches is normal, not junk — no 32k vocabulary covers
                // every character.
                None => {
                    for &byte in piece(&s).as_bytes() {
                        if let Some(&id) = ids.get(&byte_token(byte)) {
                            out.push(id);
                        }
                    }
                }
            }
        }
        i = s.next;
    }
    out
}

/// Turn one token's stored text into the bytes it represents.
///
/// `<0x0A>` is a *byte*, not the six characters that spell it, and treating it
/// literally is a common way to produce output full of angle brackets.
pub fn piece_bytes(text: &str) -> Vec<u8> {
    if text.len() == 6 && text.starts_with("<0x") && text.ends_with('>') {
        if let Ok(b) = u8::from_str_radix(&text[3..5], 16) {
            return vec![b];
        }
    }
    unescape_whitespace(text).into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A miniature SentencePiece vocabulary, scored so the intended merges win.
    fn vocab() -> (HashMap<String, u32>, Vec<f32>) {
        let pieces = [
            "<unk>",
            "<s>",
            "</s>",
            "\u{2581}",
            "h",
            "e",
            "l",
            "o",
            "\u{2581}h",
            "he",
            "ll",
            "hell",
            "hello",
            "\u{2581}hello",
            "<0x0A>",
            "<0xC3>",
            "<0xA9>",
        ];
        let scores = [
            0.0, 0.0, 0.0, -1.0, -5.0, -5.0, -5.0, -5.0, -4.0, -3.0, -3.0, -2.0, -1.5, -1.0, -9.0,
            -9.0, -9.0,
        ];
        let ids = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.to_string(), i as u32))
            .collect();
        (ids, scores.to_vec())
    }

    fn text_of(ids_out: &[u32]) -> String {
        let pieces = [
            "<unk>",
            "<s>",
            "</s>",
            "\u{2581}",
            "h",
            "e",
            "l",
            "o",
            "\u{2581}h",
            "he",
            "ll",
            "hell",
            "hello",
            "\u{2581}hello",
            "<0x0A>",
            "<0xC3>",
            "<0xA9>",
        ];
        ids_out
            .iter()
            .map(|&i| pieces[i as usize].to_string())
            .collect::<Vec<_>>()
            .join("|")
    }

    #[test]
    fn the_highest_scoring_merge_wins_not_the_earliest() {
        // This is the whole difference from BPE. "▁hello" scores -1.0, better
        // than any of its parts, so the greedy-by-score rule must reach it.
        let (ids, scores) = vocab();
        let out = encode("hello", &ids, &scores, true);
        assert_eq!(text_of(&out), "▁hello", "got {:?}", text_of(&out));
    }

    #[test]
    fn a_leading_dummy_space_is_added_and_changes_the_result() {
        // Without the dummy prefix the same word tokenizes differently. Both
        // are "valid"; only one matches what the model was trained on.
        let (ids, scores) = vocab();
        let with = encode("hello", &ids, &scores, true);
        let without = encode("hello", &ids, &scores, false);
        assert_ne!(
            text_of(&with),
            text_of(&without),
            "the dummy prefix must matter, or it is not being applied"
        );
        assert_eq!(text_of(&without), "hello");
    }

    #[test]
    fn spaces_become_the_sentencepiece_marker() {
        let escaped = escape_whitespace("a b", false);
        assert_eq!(escaped, "a\u{2581}b");
        assert_eq!(unescape_whitespace(&escaped), "a b");
    }

    #[test]
    fn text_the_vocabulary_cannot_spell_falls_back_to_bytes() {
        // "é" is not a piece, but its two UTF-8 bytes are. Falling back to
        // characters instead — the byte-level BPE behaviour — would drop it.
        let (ids, scores) = vocab();
        let out = encode("é", &ids, &scores, false);
        assert_eq!(text_of(&out), "<0xC3>|<0xA9>");
    }

    #[test]
    fn byte_tokens_decode_to_bytes_not_to_their_spelling() {
        assert_eq!(piece_bytes("<0x0A>"), vec![b'\n']);
        assert_eq!(piece_bytes("<0xC3>"), vec![0xC3]);
        // Upper-case only, and a lookalike that is not a byte token stays text.
        assert_eq!(piece_bytes("<0xzz>"), b"<0xzz>".to_vec());
        assert_eq!(piece_bytes("\u{2581}hi"), b" hi".to_vec());
    }

    #[test]
    fn tokenization_is_deterministic_when_scores_tie() {
        // Equal scores must resolve the same way every run, or two identical
        // prompts can produce different tokens and the bug looks like the model.
        let mut ids: HashMap<String, u32> = HashMap::new();
        for (i, p) in ["a", "b", "ab", "ba"].iter().enumerate() {
            ids.insert(p.to_string(), i as u32);
        }
        let scores = vec![-1.0, -1.0, -2.0, -2.0];
        let first = encode("abab", &ids, &scores, false);
        for _ in 0..20 {
            assert_eq!(encode("abab", &ids, &scores, false), first);
        }
    }

    #[test]
    fn an_empty_string_encodes_to_nothing_without_a_prefix() {
        let (ids, scores) = vocab();
        assert!(encode("", &ids, &scores, false).is_empty());
    }
}
