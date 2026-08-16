//! The stack machine that decides which code points may come next.
//!
//! # The shape of the state
//!
//! A grammar position is a stack of element positions: the top is what must
//! match next, and everything below is what to return to when the current rule
//! finishes. Because a rule can have alternatives, there is not one stack but a
//! *set* of them — every way the grammar could currently be being satisfied.
//!
//! An **empty stack means the grammar is complete**, which is why
//! [`Matcher::is_complete`] is a search for an empty stack rather than a flag.
//!
//! # Why matching is per code point and not per byte
//!
//! GBNF terminals are Unicode code points, so `[^\n]` is one element that
//! matches a three-byte character. Tokens, on the other hand, are byte strings
//! and a single token may end **half way through** a character — an emoji is
//! four byte-fallback tokens under SentencePiece. So the token filter decodes
//! as far as it can and asks [`Matcher::could_still_match`] about the trailing
//! partial character, exactly as llama.cpp's `match_partial_char` does. Judging
//! a partial character as a failure would mask every byte-fallback token and
//! make non-ASCII output impossible under any grammar.

use crate::gbnf::{Element, ElementType, Parsed};

/// A position in the rule table: which rule, and which element of it.
type Pos = (usize, usize);

/// One way the grammar could currently be satisfied. The top of the stack is
/// the element that must match next.
type Stack = Vec<Pos>;

#[derive(Clone, Debug)]
pub struct Matcher<'g> {
    grammar: &'g Parsed,
    stacks: Vec<Stack>,
}

impl<'g> Matcher<'g> {
    pub fn new(grammar: &'g Parsed) -> Self {
        let mut stacks = Vec::new();
        // **Every alternative of the root rule is a starting point.** Entering
        // a rule through a `RuleRef` fans out over its alternatives, but the
        // root is entered without one, so starting at element 0 would explore
        // only its first alternative — `root ::= "cat" | "car"` would accept
        // `cat` and reject `car`, with no error anywhere.
        for start in alternative_starts(&grammar.rules[grammar.root]) {
            advance(grammar, vec![(grammar.root, start)], &mut stacks);
        }
        Matcher { grammar, stacks }
    }

    /// True once the whole grammar has been satisfied — an empty stack.
    ///
    /// Not the same as "nothing more may follow": a grammar can be complete and
    /// still able to continue, which is exactly what `("a")+` is after one `a`.
    pub fn is_complete(&self) -> bool {
        self.stacks.iter().any(|s| s.is_empty())
    }

    /// True when no continuation is possible at all. A matcher in this state
    /// masks every token, so a caller that ignores it sees generation stop with
    /// no explanation.
    pub fn is_stuck(&self) -> bool {
        self.stacks.is_empty()
    }

    /// Advance over one code point. Returns false and leaves the matcher
    /// unchanged if the character is not accepted.
    pub fn accept_char(&mut self, chr: u32) -> bool {
        let mut next = Vec::new();
        for stack in &self.stacks {
            let Some(&(rule, at)) = stack.last() else {
                // A complete parse cannot consume more input, but a *different*
                // stack still might, so this is not a failure.
                continue;
            };
            let elements = &self.grammar.rules[rule];
            let (matched, after) = match_char(elements, at, chr);
            if !matched {
                continue;
            }
            let mut s = stack.clone();
            s.pop();
            if !is_end_of_sequence(elements, after) {
                s.push((rule, after));
            }
            advance(self.grammar, s, &mut next);
        }
        if next.is_empty() {
            return false;
        }
        self.stacks = next;
        true
    }

    /// Accept a whole UTF-8 string, stopping at the first rejected character.
    pub fn accept_str(&mut self, text: &str) -> bool {
        for c in text.chars() {
            if !self.accept_char(c as u32) {
                return false;
            }
        }
        true
    }

    /// Could `bytes` — an **incomplete** UTF-8 sequence — still become a
    /// character this grammar accepts?
    ///
    /// A token may end mid-character, and rejecting that would mask every
    /// byte-fallback token and make non-ASCII output impossible under any
    /// grammar. So a partial sequence is judged on the code points it could
    /// still become.
    pub fn could_still_match(&self, bytes: &[u8]) -> bool {
        let Some((low, high)) = partial_range(bytes) else {
            return false;
        };
        self.stacks.iter().any(|stack| {
            let Some(&(rule, at)) = stack.last() else {
                return false;
            };
            match_partial(&self.grammar.rules[rule], at, low, high)
        })
    }
}

/// The element index each alternative of `rule` begins at.
///
/// A rule is a flat sequence with `Alt` separating alternatives, so this walks
/// it once and notes every position just past an `Alt`.
fn alternative_starts(rule: &[Element]) -> Vec<usize> {
    let mut starts = vec![0];
    for (i, e) in rule.iter().enumerate() {
        if e.ty == ElementType::Alt {
            starts.push(i + 1);
        }
    }
    starts
}

/// True if the element at `at` ends the current alternative.
fn is_end_of_sequence(elements: &[Element], at: usize) -> bool {
    match elements.get(at) {
        None => true,
        Some(e) => matches!(e.ty, ElementType::End | ElementType::Alt),
    }
}

/// Expand every rule reference on top of `stack` until each resulting stack has
/// a character element on top (or is empty, meaning complete).
///
/// This is where alternatives multiply: one stack in, many out.
fn advance(grammar: &Parsed, stack: Stack, out: &mut Vec<Stack>) {
    let mut todo = vec![stack];
    // Stacks recur through recursive rules -- `S' ::= S S' |` is the shape
    // every `*` desugars to -- so without this the expansion does not
    // terminate on a grammar that is perfectly valid.
    let mut seen: Vec<Stack> = Vec::new();

    while let Some(stack) = todo.pop() {
        if seen.contains(&stack) {
            continue;
        }
        seen.push(stack.clone());

        let Some(&(rule, at)) = stack.last() else {
            if !out.contains(&stack) {
                out.push(stack);
            }
            continue;
        };
        let element = grammar.rules[rule][at];
        match element.ty {
            ElementType::RuleRef => {
                let referenced = element.value as usize;
                let sub = &grammar.rules[referenced];
                let mut alt_start = 0usize;
                loop {
                    let mut next = stack.clone();
                    next.pop();
                    // What to come back to once the referenced rule finishes.
                    if !is_end_of_sequence(&grammar.rules[rule], at + 1) {
                        next.push((rule, at + 1));
                    }
                    // An empty alternative contributes nothing to push.
                    if !is_end_of_sequence(sub, alt_start) {
                        next.push((referenced, alt_start));
                    }
                    todo.push(next);

                    // Scan to the end of this alternative.
                    let mut scan = alt_start;
                    while !is_end_of_sequence(sub, scan) {
                        scan += 1;
                    }
                    match sub.get(scan).map(|e| e.ty) {
                        Some(ElementType::Alt) => alt_start = scan + 1,
                        _ => break,
                    }
                }
            }
            // A character element is a place the machine waits for input.
            _ => {
                if !out.contains(&stack) {
                    out.push(stack);
                }
            }
        }
    }
}

/// Does `chr` satisfy the character set starting at `at`? Returns the match and
/// the index just past the set.
///
/// The encoding is llama.cpp's: a `Char`/`CharNot` optionally followed by a
/// `CharRngUpper` to make it a range, then any number of `CharAlt` (each itself
/// optionally followed by a `CharRngUpper`) adding more alternatives. The
/// polarity of the *first* element decides whether the whole set is positive or
/// negated.
fn match_char(elements: &[Element], at: usize, chr: u32) -> (bool, usize) {
    let positive = matches!(elements[at].ty, ElementType::Char | ElementType::CharAny);
    let mut found = false;
    let mut i = at;
    loop {
        if elements[i].ty == ElementType::CharAny {
            found = true;
            i += 1;
        } else if elements.get(i + 1).map(|e| e.ty) == Some(ElementType::CharRngUpper) {
            found = found || (elements[i].value <= chr && chr <= elements[i + 1].value);
            i += 2;
        } else {
            found = found || elements[i].value == chr;
            i += 1;
        }
        if elements.get(i).map(|e| e.ty) != Some(ElementType::CharAlt) {
            break;
        }
    }
    (found == positive, i)
}

/// Could any code point in `low..=high` satisfy the set at `at`?
fn match_partial(elements: &[Element], at: usize, low: u32, high: u32) -> bool {
    let Some(first) = elements.get(at) else {
        return false;
    };
    let positive = matches!(first.ty, ElementType::Char | ElementType::CharAny);
    if !matches!(
        first.ty,
        ElementType::Char | ElementType::CharAny | ElementType::CharNot
    ) {
        return false;
    }
    let mut i = at;
    loop {
        if elements[i].ty == ElementType::CharAny {
            // Any character at all: every candidate qualifies, and for a
            // negated set that means none do.
            return positive;
        }
        let (lo, hi) = if elements.get(i + 1).map(|e| e.ty) == Some(ElementType::CharRngUpper) {
            let r = (elements[i].value, elements[i + 1].value);
            i += 2;
            r
        } else {
            let v = elements[i].value;
            i += 1;
            (v, v)
        };
        // Do the two ranges intersect?
        if lo <= high && low <= hi {
            return positive;
        }
        if elements.get(i).map(|e| e.ty) != Some(ElementType::CharAlt) {
            break;
        }
    }
    // Nothing in the set overlaps. For a negated set that is exactly what makes
    // the candidates acceptable.
    !positive
}

/// The inclusive range of code points an incomplete UTF-8 sequence could become.
///
/// `None` if the bytes are not a valid prefix of any encoding — a continuation
/// byte with no lead, an over-long lead, or a sequence already longer than its
/// lead byte allows.
fn partial_range(bytes: &[u8]) -> Option<(u32, u32)> {
    let (&lead, rest) = bytes.split_first()?;
    let (width, mut value) = match lead {
        0x00..=0x7F => (1usize, lead as u32),
        0xC0..=0xDF => (2, (lead & 0x1F) as u32),
        0xE0..=0xEF => (3, (lead & 0x0F) as u32),
        0xF0..=0xF7 => (4, (lead & 0x07) as u32),
        // A continuation byte with nothing in front, or an invalid lead.
        _ => return None,
    };
    if rest.len() + 1 > width {
        return None;
    }
    for &b in rest {
        if b & 0xC0 != 0x80 {
            return None;
        }
        value = (value << 6) | (b & 0x3F) as u32;
    }
    // Each byte still to come contributes six unconstrained bits.
    let remaining = width - bytes.len();
    let shift = 6 * remaining as u32;
    let low = value << shift;
    let high = low | ((1u32 << shift) - 1);
    Some((low, high))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gbnf;

    fn matcher_for(src: &str) -> Parsed {
        gbnf::parse(src).expect("parses")
    }

    #[test]
    fn a_literal_is_accepted_exactly() {
        let g = matcher_for(r#"root ::= "abc""#);
        let mut m = Matcher::new(&g);
        assert!(m.accept_str("abc"));
        assert!(m.is_complete());

        let mut m = Matcher::new(&g);
        assert!(!m.accept_str("abd"));
    }

    #[test]
    fn alternatives_are_explored_in_parallel() {
        let g = matcher_for(r#"root ::= "cat" | "car""#);
        for (text, ok) in [("cat", true), ("car", true), ("cab", false)] {
            let mut m = Matcher::new(&g);
            assert_eq!(m.accept_str(text) && m.is_complete(), ok, "{text}");
        }
    }

    #[test]
    fn star_accepts_zero_or_more_and_is_complete_at_zero() {
        let g = matcher_for(r#"root ::= "a"*"#);
        let mut m = Matcher::new(&g);
        assert!(m.is_complete(), "zero repetitions already satisfies it");
        assert!(m.accept_str("aaaa"));
        assert!(m.is_complete());
    }

    #[test]
    fn plus_needs_at_least_one() {
        let g = matcher_for(r#"root ::= "a"+"#);
        let mut m = Matcher::new(&g);
        assert!(!m.is_complete());
        assert!(m.accept_str("a"));
        assert!(m.is_complete());
        assert!(m.accept_str("aa"));
    }

    #[test]
    fn question_mark_is_zero_or_one() {
        let g = matcher_for(r#"root ::= "a"? "b""#);
        for text in ["b", "ab"] {
            let mut m = Matcher::new(&g);
            assert!(m.accept_str(text) && m.is_complete(), "{text}");
        }
        let mut m = Matcher::new(&g);
        assert!(!m.accept_str("aab"));
    }

    #[test]
    fn bounded_repetition_counts_exactly() {
        let g = matcher_for(r#"root ::= "a"{2,3}"#);
        for (text, ok) in [("a", false), ("aa", true), ("aaa", true)] {
            let mut m = Matcher::new(&g);
            let got = m.accept_str(text) && m.is_complete();
            assert_eq!(got, ok, "{text}");
        }
        let mut m = Matcher::new(&g);
        assert!(!m.accept_str("aaaa"));
    }

    #[test]
    fn exact_repetition_counts_exactly() {
        let g = matcher_for(r#"root ::= "a"{3}"#);
        for (text, ok) in [("aa", false), ("aaa", true)] {
            let mut m = Matcher::new(&g);
            assert_eq!(m.accept_str(text) && m.is_complete(), ok, "{text}");
        }
    }

    #[test]
    fn a_negated_class_rejects_only_what_it_lists() {
        let g = matcher_for(r#"root ::= [^ab]+"#);
        let mut m = Matcher::new(&g);
        assert!(m.accept_str("xyz"));
        let mut m = Matcher::new(&g);
        assert!(!m.accept_str("a"));
    }

    #[test]
    fn dot_matches_any_code_point() {
        let g = matcher_for("root ::= .");
        for text in ["a", "\u{4e2d}", "\u{1f600}"] {
            let mut m = Matcher::new(&g);
            assert!(m.accept_str(text) && m.is_complete(), "{text:?}");
        }
    }

    /// The shape every `*` desugars to is recursive, so an expansion that did
    /// not remember where it had been would not terminate here.
    #[test]
    fn a_recursive_rule_terminates() {
        let g = matcher_for("root ::= item\nitem ::= \"a\" item | \"b\"\n");
        let mut m = Matcher::new(&g);
        assert!(m.accept_str("aaab"));
        assert!(m.is_complete());
    }

    #[test]
    fn nested_groups_and_alternation() {
        let g = matcher_for(r#"root ::= ("a" | "b") ("c" | "d")"#);
        for text in ["ac", "ad", "bc", "bd"] {
            let mut m = Matcher::new(&g);
            assert!(m.accept_str(text) && m.is_complete(), "{text}");
        }
        let mut m = Matcher::new(&g);
        assert!(!m.accept_str("ab"));
    }

    #[test]
    fn a_grammar_that_cannot_continue_reports_itself_stuck() {
        let g = matcher_for(r#"root ::= "a""#);
        let mut m = Matcher::new(&g);
        assert!(m.accept_str("a"));
        assert!(!m.accept_char('b' as u32));
        // The rejected character must leave the matcher usable, not poisoned.
        assert!(m.is_complete());
        assert!(!m.is_stuck());
    }

    #[test]
    fn partial_utf8_is_judged_on_what_it_could_become() {
        // A three-byte lead for U+4E2D is 0xE4; the first continuation narrows
        // it, and neither prefix is yet a character.
        let g = matcher_for("root ::= [\u{4e00}-\u{9fff}]");
        let m = Matcher::new(&g);
        assert!(m.could_still_match(&[0xE4]));
        assert!(m.could_still_match(&[0xE4, 0xB8]));
        // An ASCII-only grammar cannot be satisfied by any three-byte sequence.
        let g2 = matcher_for("root ::= [a-z]");
        let m2 = Matcher::new(&g2);
        assert!(!m2.could_still_match(&[0xE4]));
    }

    #[test]
    fn a_continuation_byte_with_no_lead_is_not_a_prefix_of_anything() {
        assert_eq!(partial_range(&[0x80]), None);
        assert_eq!(partial_range(&[0xFF]), None);
        assert_eq!(partial_range(&[0xE4, 0x41]), None);
    }
}
