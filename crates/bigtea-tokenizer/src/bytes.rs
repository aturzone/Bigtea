//! GPT-2's byte-to-unicode mapping.
//!
//! Byte-level BPE has to represent arbitrary bytes as characters, including
//! bytes that are not printable and not valid UTF-8 on their own. GPT-2 solved
//! this by mapping the 256 possible bytes onto 256 *printable* codepoints:
//! printable ASCII and Latin-1 map to themselves, and everything else is
//! shifted up into an unused block starting at U+0100.
//!
//! This is why a vocabulary contains tokens like `Ġthe` — `Ġ` (U+0120) is how
//! a leading space is written. Getting this mapping wrong means every token
//! containing a space is looked up under the wrong key, so nothing matches and
//! the tokenizer silently falls back to single bytes.

/// Byte value -> the character that represents it.
pub fn byte_to_char(b: u8) -> char {
    // The three ranges GPT-2 leaves alone: '!'..='~', '¡'..='¬', '®'..='ÿ'.
    let keep =
        (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    if keep {
        return b as char;
    }
    // Everything else is assigned a slot above U+0100, in byte order.
    let rank = (0u8..=255)
        .filter(|c| {
            !((0x21..=0x7E).contains(c) || (0xA1..=0xAC).contains(c) || (0xAE..=0xFF).contains(c))
        })
        .position(|c| c == b)
        .expect("every byte is either kept or shifted");
    char::from_u32(0x100 + rank as u32).expect("shifted range is valid")
}

/// The inverse of [`byte_to_char`], or `None` for a character that never
/// appears in a byte-level vocabulary.
pub fn char_to_byte(c: char) -> Option<u8> {
    let code = c as u32;
    if (0x21..=0x7E).contains(&code)
        || (0xA1..=0xAC).contains(&code)
        || (0xAE..=0xFF).contains(&code)
    {
        return Some(code as u8);
    }
    if (0x100..0x100 + 68).contains(&code) {
        let rank = (code - 0x100) as usize;
        return (0u8..=255)
            .filter(|c| {
                !((0x21..=0x7E).contains(c)
                    || (0xA1..=0xAC).contains(c)
                    || (0xAE..=0xFF).contains(c))
            })
            .nth(rank);
    }
    None
}

/// Encode raw bytes as the printable string a byte-level vocabulary uses.
pub fn encode(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| byte_to_char(b)).collect()
}

/// Decode such a string back to the original bytes.
///
/// Characters outside the mapping are skipped rather than guessed at — they
/// cannot have come from this encoding.
pub fn decode(s: &str) -> Vec<u8> {
    s.chars().filter_map(char_to_byte).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_maps_to_the_marker_seen_in_vocabularies() {
        // 'Ġ' is U+0120 and is how a leading space appears in GPT-2-family
        // vocabularies -- confirmed by this model's own merges, whose first
        // entry is "Ġ t".
        assert_eq!(byte_to_char(b' '), '\u{0120}');
        assert_eq!(char_to_byte('\u{0120}'), Some(b' '));
    }

    #[test]
    fn printable_ascii_maps_to_itself() {
        for b in 0x21u8..=0x7E {
            assert_eq!(byte_to_char(b), b as char);
            assert_eq!(char_to_byte(b as char), Some(b));
        }
    }

    #[test]
    fn every_byte_round_trips() {
        // The mapping must be a bijection over all 256 values; a collision
        // would make some byte sequences unrepresentable.
        let mut seen = std::collections::HashSet::new();
        for b in 0u8..=255 {
            let c = byte_to_char(b);
            assert!(seen.insert(c), "byte {b} collided on {c:?}");
            assert_eq!(char_to_byte(c), Some(b), "byte {b} failed to round-trip");
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn strings_round_trip_including_non_ascii() {
        for original in [
            "hello world",
            "  leading",
            "tabs\tand\nnewlines",
            "héllo — ünïcode",
        ] {
            let encoded = encode(original.as_bytes());
            assert_eq!(
                decode(&encoded),
                original.as_bytes(),
                "failed on {original:?}"
            );
        }
    }
}
