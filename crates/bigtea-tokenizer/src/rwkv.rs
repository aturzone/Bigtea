//! RWKV: greedy longest-match over a trie of raw byte strings.
//!
//! # Why it is not just "BPE with a different table"
//!
//! There is no merge table and no ranking. Every vocabulary entry is a byte
//! sequence, and tokenizing is: at each position, walk the trie as far as it
//! goes, emit the **longest entry seen on the way**, and continue from just
//! after it. No scores, no merges, no pre-tokenizer splitting the text first.
//!
//! Two details make it easy to get subtly wrong, and neither raises:
//!
//! * **The vocabulary is stored escaped.** `\n`, `\t`, `\r`, `\\` and `\xNN`
//!   appear as literal backslash sequences in `tokenizer.ggml.tokens`, so a
//!   loader that takes the strings as-is builds a trie keyed on the *text of
//!   the escape* rather than the byte it denotes. `\n` then never matches a
//!   real newline, and every line break becomes an unknown token.
//! * **Longest match is over the whole traversal, not the deepest node.** The
//!   trie can descend past the last entry that actually exists — `ab` and
//!   `abcd` present, `abc` absent — and stopping at the deepest node reached
//!   would emit nothing for `abc`. The last node *with a value* is the answer,
//!   which is why the walk records as it goes rather than at the end.
//!
//! Both failures produce a tokenization that is plausible and different, which
//! is the shape of every expensive bug in this project.

use std::collections::HashMap;

/// One node of the byte trie.
#[derive(Default, Debug)]
struct Node {
    /// Token id, when a vocabulary entry ends exactly here.
    value: Option<u32>,
    next: HashMap<u8, usize>,
}

/// A vocabulary of raw byte strings, ready for greedy matching.
#[derive(Debug, Default)]
pub struct Trie {
    nodes: Vec<Node>,
}

impl Trie {
    pub fn new() -> Self {
        Trie {
            nodes: vec![Node::default()],
        }
    }

    /// Insert `bytes` as token `id`.
    ///
    /// A later duplicate wins, matching llama.cpp's insertion order — a
    /// vocabulary with a repeated entry is malformed, and silently keeping the
    /// first would disagree with the reference on exactly that vocabulary.
    pub fn insert(&mut self, bytes: &[u8], id: u32) {
        let mut cur = 0usize;
        for &b in bytes {
            cur = match self.nodes[cur].next.get(&b) {
                Some(&n) => n,
                None => {
                    self.nodes.push(Node::default());
                    let n = self.nodes.len() - 1;
                    self.nodes[cur].next.insert(b, n);
                    n
                }
            };
        }
        self.nodes[cur].value = Some(id);
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    /// The longest entry that is a prefix of `text`, as `(id, length)`.
    fn longest(&self, text: &[u8]) -> Option<(u32, usize)> {
        let mut cur = 0usize;
        let mut best: Option<(u32, usize)> = None;
        for (i, &b) in text.iter().enumerate() {
            let Some(&n) = self.nodes[cur].next.get(&b) else {
                break;
            };
            cur = n;
            // Recorded as we go, NOT at the end: `ab` and `abcd` present with
            // `abc` absent means the deepest node reached for "abc" has no
            // value, and taking it would emit nothing.
            if let Some(id) = self.nodes[cur].value {
                best = Some((id, i + 1));
            }
        }
        best
    }
}

/// Undo the escaping RWKV vocabularies are stored with.
///
/// `\n`, `\t`, `\r`, `\xNN`, and `\<anything else>` meaning that character.
/// **Not a UTF-8 operation** — the result is bytes, because `\xNN` can denote a
/// byte that is not valid UTF-8 on its own and a multi-byte character arrives
/// as several of them.
pub fn unescape(escaped: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(escaped.len());
    let mut escaping = false;
    let mut hex_remaining = 0u8;
    let mut hex_acc = 0u8;

    for &c in escaped.as_bytes() {
        if hex_remaining != 0 {
            let v = match c {
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => c.wrapping_sub(b'0'),
            };
            hex_acc = (hex_acc << 4) + v;
            hex_remaining -= 1;
            if hex_remaining == 0 {
                out.push(hex_acc);
                hex_acc = 0;
            }
            continue;
        }
        if escaping {
            match c {
                b't' => out.push(b'\t'),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b'x' => hex_remaining = 2,
                // `\\` and `\"` and anything else: the character itself.
                other => out.push(other),
            }
            escaping = false;
            continue;
        }
        if c == b'\\' {
            escaping = true;
            continue;
        }
        out.push(c);
    }
    out
}

/// Build the trie from a vocabulary whose entries are still escaped.
pub fn build(tokens: &[String]) -> Trie {
    let mut trie = Trie::new();
    for (id, text) in tokens.iter().enumerate() {
        let bytes = unescape(text);
        if bytes.is_empty() {
            // An empty entry would match at every position with length 0 and
            // spin the loop forever. Skipped rather than trusted.
            continue;
        }
        trie.insert(&bytes, id as u32);
    }
    trie
}

/// Tokenize `text`, emitting `unk` for any byte the trie cannot start from.
pub fn encode(trie: &Trie, text: &str, unk: Option<u32>) -> Vec<u32> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        match trie.longest(&bytes[pos..]) {
            Some((id, len)) => {
                out.push(id);
                // `len` is always >= 1 because empty entries are skipped at
                // build time; without that guarantee this loop would not
                // terminate.
                pos += len;
            }
            None => {
                if let Some(u) = unk {
                    out.push(u);
                }
                pos += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(entries: &[&str]) -> Trie {
        build(&entries.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn escapes_become_the_bytes_they_denote() {
        // A loader that takes the strings as-is keys the trie on the TEXT of
        // the escape, so `\n` never matches a real newline and every line
        // break becomes an unknown token.
        assert_eq!(unescape("\\n"), b"\n");
        assert_eq!(unescape("\\t"), b"\t");
        assert_eq!(unescape("\\r"), b"\r");
        assert_eq!(unescape("\\\\"), b"\\");
        assert_eq!(unescape("a\\nb"), b"a\nb");
    }

    #[test]
    fn hex_escapes_can_denote_bytes_that_are_not_utf8() {
        // Which is why this returns bytes rather than a String -- `\xff` alone
        // is not valid UTF-8 and a String-based unescape would have to lose it.
        assert_eq!(unescape("\\xff"), vec![0xff]);
        assert_eq!(unescape("\\x41\\x42"), b"AB");
        // Upper-case hex digits too; llama.cpp's own reader handles only
        // lower-case, and a vocabulary using upper would silently differ.
        assert_eq!(unescape("\\xFF"), vec![0xff]);
    }

    #[test]
    fn greedy_match_takes_the_longest_entry() {
        let t = vocab(&["a", "ab", "abc"]);
        assert_eq!(encode(&t, "abc", None), vec![2]);
        assert_eq!(encode(&t, "ab", None), vec![1]);
    }

    #[test]
    fn the_longest_match_is_the_last_node_with_a_value() {
        // `ab` and `abcd` present, `abc` absent. The trie descends past `ab`
        // to a node with no value; stopping at the deepest node reached would
        // emit nothing for "abc".
        let t = vocab(&["ab", "abcd"]);
        // `ab` is emitted and the unmatched `c` falls through to unk. If the
        // walk took the DEEPEST node instead of the last one with a value,
        // "abc" would emit nothing at all and the `c` would be consumed too.
        assert_eq!(encode(&t, "abc", Some(9)), vec![0, 9]);
        assert_eq!(encode(&t, "abcd", None), vec![1]);
    }

    #[test]
    fn an_unmatched_byte_becomes_unk_and_the_position_still_advances() {
        let t = vocab(&["a"]);
        assert_eq!(encode(&t, "aXa", Some(9)), vec![0, 9, 0]);
        // Without an unk id the byte is dropped, but the loop must still move.
        assert_eq!(encode(&t, "aXa", None), vec![0, 0]);
    }

    #[test]
    fn an_empty_vocabulary_entry_cannot_hang_the_loop() {
        // An empty entry matches at every position with length 0. Skipped at
        // build time, because the alternative is an infinite loop on real
        // input rather than a wrong answer.
        let t = vocab(&["", "a"]);
        assert_eq!(encode(&t, "aa", Some(7)), vec![1, 1]);
    }

    #[test]
    fn a_later_duplicate_wins_as_it_does_in_llamacpp() {
        // A vocabulary with a repeated entry is malformed; keeping the first
        // silently would disagree with the reference on exactly that file.
        let t = vocab(&["dup", "dup"]);
        assert_eq!(encode(&t, "dup", None), vec![1]);
    }

    #[test]
    fn multibyte_text_matches_on_bytes_not_characters() {
        // The trie is byte-keyed, so a multi-byte character is several edges.
        let t = vocab(&["\u{4f60}\u{597d}", "\u{4f60}"]);
        assert_eq!(encode(&t, "\u{4f60}\u{597d}", None), vec![0]);
        assert_eq!(encode(&t, "\u{4f60}", None), vec![1]);
    }

    #[test]
    fn an_empty_trie_reports_itself_rather_than_matching_nothing_silently() {
        assert!(Trie::new().is_empty());
        assert!(!vocab(&["a"]).is_empty());
    }
}
