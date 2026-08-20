//! Unigram (`tokenizer.ggml.model = "t5"`), the T5/mT5 family.
//!
//! # Why this is not the SentencePiece already in [`crate::spm`]
//!
//! Both come from SentencePiece and both spell a word boundary `▁`, which makes
//! them look like the same thing. They choose tokens by opposite methods:
//!
//! - **BPE/SPM** is *constructive and greedy*. Start from characters, repeatedly
//!   merge the best-scoring adjacent pair. Each decision is local and final.
//! - **Unigram** is *selective and global*. The vocabulary is a probability
//!   model; every way of cutting the string is a candidate, and the tokenization
//!   is the one whose token scores sum highest. That is a shortest-path problem
//!   over a lattice, solved with Viterbi — a locally worse split often wins
//!   because of what it makes possible later.
//!
//! Running SPM's merge loop on a Unigram vocabulary produces valid ids and a
//! different segmentation. No error, no crash, just a stream the model was not
//! trained on.
//!
//! # Scores
//!
//! Vocabulary scores are log probabilities, so all negative and summed along a
//! path. Two details decide whether the result matches the reference exactly:
//!
//! - **`USER_DEFINED` tokens score 0**, not their stored score. Zero beats every
//!   real token's negative score, which is how added tokens win against any
//!   ordinary segmentation covering the same span.
//! - **Sums accumulate in `f64`.** llama.cpp is explicit that this is what makes
//!   its output identical to the HuggingFace tokenizer; `f32` accumulation
//!   drifts and flips near-ties on long inputs.
//!
//! # The unknown penalty
//!
//! A codepoint no token covers is charged `min_score - 10.0` — worse than the
//! worst real token, so the lattice uses `<unk>` only when nothing else reaches.

use std::collections::HashMap;

/// The word-boundary marker SentencePiece writes for a space.
const MARK: char = '\u{2581}';

/// How much worse than the worst real token an unknown codepoint is.
///
/// llama.cpp's constant. It only has to be large enough that no path prefers
/// `<unk>` while a real covering exists.
const UNKNOWN_PENALTY: f64 = 10.0;

/// Apply SentencePiece's normalization: spaces become `▁`, runs collapse, and a
/// boundary is prepended so the first word tokenizes as it would mid-sentence.
///
/// **The precompiled charsmap is not applied.** T5 containers ship
/// `tokenizer.ggml.precompiled_charsmap` — a serialised NFKC automaton, 237 KB
/// in `flan-t5-small` — and llama.cpp walks it before this step. For text that
/// is already in normal form, which is all ASCII and nearly all ordinary prose,
/// it is the identity. Input that is *not* normal (fullwidth forms `Ｈｅｌｌｏ`,
/// ligatures `ﬁ`, compatibility digits `②`) will tokenize differently here than
/// in llama.cpp. That is a real gap and is stated rather than hidden; see the
/// research node.
pub fn normalize(text: &str, add_space_prefix: bool, remove_extra_whitespaces: bool) -> String {
    let mut out = String::with_capacity(text.len() + 3);
    if add_space_prefix {
        out.push(MARK);
    }
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space {
            // Collapsing is what turns "spaced   out" into two words rather than
            // one word and two empty ones.
            if !remove_extra_whitespaces || !out.ends_with(MARK) {
                out.push(MARK);
            }
            pending_space = false;
        }
        out.push(ch);
    }
    // Trailing whitespace produces no boundary: llama.cpp drops it, and keeping
    // it would add a stray `▁` token at the end of every input.
    out
}

/// The longest token in a vocabulary, in bytes.
///
/// Bounds the prefix search: at each position only spans up to this length can
/// possibly be a token, which keeps the lattice build linear in practice.
pub fn max_token_len<S: AsRef<str>>(tokens: &[S]) -> usize {
    tokens.iter().map(|t| t.as_ref().len()).max().unwrap_or(0)
}

/// One cell of the Viterbi lattice: the best way to arrive at this byte offset.
#[derive(Clone, Copy)]
struct Best {
    token: Option<u32>,
    from: usize,
    score: f64,
}

/// Tokenize normalized text by highest-scoring path through the lattice.
///
/// `user_defined` is indexed by token id and zeroes that token's score, which is
/// how an added token outranks any ordinary segmentation of the same span.
pub fn encode(
    normalized: &str,
    ids: &HashMap<String, u32>,
    scores: &[f32],
    user_defined: &[bool],
    unk: Option<u32>,
    max_len: usize,
) -> Vec<u32> {
    let bytes = normalized.as_bytes();
    let n = bytes.len();
    if n == 0 {
        return Vec::new();
    }

    let min_score = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let unk_score = f64::from(min_score) - UNKNOWN_PENALTY;

    let mut best = vec![
        Best {
            token: None,
            from: 0,
            score: f64::NEG_INFINITY,
        };
        n + 1
    ];
    best[0].score = 0.0;

    let mut at = 0usize;
    while at < n {
        // Unreachable only if a previous step left a gap, which cannot happen
        // because the unknown fallback always advances one whole codepoint.
        if best[at].score == f64::NEG_INFINITY {
            at += 1;
            continue;
        }
        let here = best[at].score;
        let cp_len = utf8_len(bytes[at]).min(n - at);
        let mut covered_whole_codepoint = false;

        let limit = (at + max_len).min(n);
        for end in (at + 1)..=limit {
            // A span that splits a character is never a token, and slicing a
            // `str` there would panic — so only consider char boundaries.
            if !normalized.is_char_boundary(end) {
                continue;
            }
            let Some(&id) = ids.get(&normalized[at..end]) else {
                continue;
            };
            if end - at == cp_len {
                covered_whole_codepoint = true;
            }
            // USER_DEFINED scores 0; everything else uses its log probability.
            let score = match user_defined.get(id as usize) {
                Some(true) => 0.0,
                _ => f64::from(scores.get(id as usize).copied().unwrap_or(0.0)),
            };
            let candidate = here + score;
            if candidate > best[end].score {
                best[end] = Best {
                    token: Some(id),
                    from: at,
                    score: candidate,
                };
            }
        }

        // Nothing covered this codepoint on its own, so it can only be reached
        // as `<unk>` — charged the penalty so a real covering always wins.
        if !covered_whole_codepoint {
            let end = at + cp_len;
            let candidate = here + unk_score;
            if end <= n && candidate > best[end].score {
                best[end] = Best {
                    token: unk,
                    from: at,
                    score: candidate,
                };
            }
        }
        at += cp_len;
    }

    // Walk the back-pointers from the end and reverse.
    let mut out = Vec::new();
    let mut pos = n;
    while pos > 0 {
        let cell = best[pos];
        if cell.score == f64::NEG_INFINITY {
            // No path reached the end. Cannot happen with the unknown fallback,
            // but returning what was found beats looping forever.
            break;
        }
        out.extend(cell.token);
        if cell.from >= pos {
            break;
        }
        pos = cell.from;
    }
    out.reverse();
    out
}

/// Bytes in the UTF-8 sequence a lead byte starts.
fn utf8_len(lead: u8) -> usize {
    match lead {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        // A continuation byte here means the input was not valid UTF-8; treat it
        // as one byte so the loop still advances.
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(entries: &[(&str, f32)]) -> (HashMap<String, u32>, Vec<f32>, Vec<bool>) {
        let ids = entries
            .iter()
            .enumerate()
            .map(|(i, (t, _))| ((*t).to_string(), i as u32))
            .collect();
        let scores = entries.iter().map(|(_, s)| *s).collect();
        let flags = vec![false; entries.len()];
        (ids, scores, flags)
    }

    #[test]
    fn prefixes_a_boundary_and_collapses_runs() {
        assert_eq!(normalize("spaced   out", true, true), "▁spaced▁out");
        assert_eq!(normalize("  spaced   out  ", true, true), "▁spaced▁out");
    }

    #[test]
    fn honours_a_container_that_declines_the_prefix() {
        assert_eq!(normalize("hello world", false, true), "hello▁world");
    }

    /// The property that separates Unigram from greedy merging: a locally worse
    /// split wins because the whole path scores higher.
    ///
    /// Greedy-longest would take `ab` (-1) then be forced onto `c` (-9) for -10.
    /// The lattice prefers `a` + `bc`, total -4.
    #[test]
    fn picks_the_best_path_not_the_best_first_step() {
        let (ids, scores, ud) = build(&[("ab", -1.0), ("a", -2.0), ("bc", -2.0), ("c", -9.0)]);
        assert_eq!(encode("abc", &ids, &scores, &ud, None, 2), [1, 2]);
    }

    #[test]
    fn a_user_defined_token_outscores_any_ordinary_split() {
        let (ids, scores, mut ud) = build(&[
            ("<extra>", -50.0),
            ("<", -1.0),
            ("extra", -1.0),
            (">", -1.0),
        ]);
        // Scored as stored, the three-piece split (-3) beats the single (-50).
        assert_eq!(encode("<extra>", &ids, &scores, &ud, None, 7), [1, 2, 3]);
        // Marked USER_DEFINED it scores 0 and wins.
        ud[0] = true;
        assert_eq!(encode("<extra>", &ids, &scores, &ud, None, 7), [0]);
    }

    #[test]
    fn an_uncovered_codepoint_becomes_unk() {
        let (ids, scores, ud) = build(&[("<unk>", -100.0), ("a", -1.0)]);
        assert_eq!(encode("aqa", &ids, &scores, &ud, Some(0), 1), [1, 0, 1]);
    }

    /// One `<unk>` per codepoint, not per byte — a 4-byte emoji is one token.
    #[test]
    fn unk_consumes_a_whole_codepoint() {
        let (ids, scores, ud) = build(&[("<unk>", -100.0), ("a", -1.0)]);
        assert_eq!(encode("a🦄a", &ids, &scores, &ud, Some(0), 1), [1, 0, 1]);
    }

    #[test]
    fn empty_input_encodes_to_nothing() {
        let (ids, scores, ud) = build(&[("<unk>", -100.0), ("a", -1.0)]);
        assert!(encode("", &ids, &scores, &ud, Some(0), 1).is_empty());
    }

    #[test]
    fn max_token_len_bounds_the_search() {
        assert_eq!(max_token_len(&["a", "abc", "ab"]), 3);
        assert_eq!(max_token_len::<&str>(&[]), 0);
    }
}
