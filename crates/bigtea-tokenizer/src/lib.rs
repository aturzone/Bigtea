//! Byte-level BPE, driven entirely by the container's own vocabulary.
//!
//! Nothing here is hard-coded to a model: the vocabulary, the merge table and
//! the special token ids all come from the GGUF metadata, so a different model
//! tokenizes correctly without a code change.
//!
//! # Why a wrong tokenizer is the worst kind of bug
//!
//! It does not crash. It produces *different* tokens, the model dutifully
//! predicts a continuation of those tokens, and the output is fluent nonsense
//! — indistinguishable at a glance from a broken forward pass. That makes it
//! expensive to diagnose later, so the pieces here are tested individually
//! (byte mapping, splitting, merging) rather than only end to end.

mod bytes;
pub mod chat;
mod pretok;
pub mod spm;

pub use bytes::{decode as bytes_decode, encode as bytes_encode};
pub use chat::{ChatFormat, Message};
pub use pretok::pre_tokenize;

/// Which tokenization rule a container asks for.
///
/// GGUF names these in `tokenizer.ggml.model`. They are not variants of one
/// algorithm — see [`spm`] for why the merge decision itself differs — so the
/// choice is made once, at load, and never re-examined per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `"gpt2"` — byte-level BPE over an explicit ranked merge table.
    Bpe,
    /// `"llama"` — SentencePiece: merge by vocabulary score, `▁` for space,
    /// `<0xXX>` byte fallback.
    Spm,
}

use std::collections::HashMap;
use std::fmt;

use bigtea_gguf::Value;

#[derive(Debug)]
pub enum TokenizerError {
    /// The container declares no vocabulary.
    MissingVocab,
    /// The container declares a tokenizer model we do not implement.
    UnsupportedModel(String),
    /// A merge rule was not two space-separated pieces.
    BadMerge(String),
    /// SentencePiece needs one score per token and the container disagrees.
    MissingScores { have: usize, want: usize },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::MissingVocab => {
                f.write_str("container has no tokenizer.ggml.tokens array")
            }
            TokenizerError::UnsupportedModel(m) => {
                write!(
                    f,
                    "unsupported tokenizer model {m:?} \
                     (implemented: \"gpt2\" byte-level BPE, \"llama\" SentencePiece)"
                )
            }
            TokenizerError::BadMerge(m) => write!(f, "malformed merge rule {m:?}"),
            TokenizerError::MissingScores { have, want } => write!(
                f,
                "SentencePiece needs one score per token, but the container has \
                 {have} scores for {want} tokens; without them every merge would \
                 score equally and tokenize wrongly without failing"
            ),
        }
    }
}

impl std::error::Error for TokenizerError {}

pub struct Tokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    /// Merge pair -> rank. Lower rank merges first. Empty for SentencePiece.
    merges: HashMap<(String, String), u32>,
    kind: Kind,
    /// Per-token score, indexed by id. Empty for byte-level BPE.
    scores: Vec<f32>,
    add_dummy_prefix: bool,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
    /// The raw Jinja template, kept so the chat format can be identified and
    /// so a caller can print it when the format is not recognised.
    chat_template: Option<String>,
    /// Control tokens, longest first, with their ids.
    ///
    /// These must be matched **literally in the text and mapped to one id**.
    /// Running `<|start_header_id|>` through BPE splits it into `<`, `|`,
    /// `start`, … — pieces the model has never seen in that position — so a
    /// chat template applies without error and the model answers as though it
    /// had been given raw text. That is precisely what happened here: framing
    /// Llama-3.2 changed nothing until this existed.
    ///
    /// Longest first so `<|eot_id|>` cannot be shadowed by a shorter prefix.
    specials: Vec<(String, u32)>,
}

impl Tokenizer {
    /// Build from a container's metadata map.
    pub fn from_metadata(
        meta: &std::collections::BTreeMap<String, Value>,
    ) -> Result<Self, TokenizerError> {
        let model = meta
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .unwrap_or("gpt2");
        let kind = match model {
            "gpt2" => Kind::Bpe,
            "llama" => Kind::Spm,
            other => return Err(TokenizerError::UnsupportedModel(other.to_string())),
        };

        let tokens: Vec<String> = meta
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or(TokenizerError::MissingVocab)?
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect();
        if tokens.is_empty() {
            return Err(TokenizerError::MissingVocab);
        }

        let ids: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let mut merges = HashMap::new();
        if let Some(list) = meta.get("tokenizer.ggml.merges").and_then(Value::as_array) {
            for (rank, entry) in list.iter().enumerate() {
                let Some(text) = entry.as_str() else { continue };
                // Merge rules are "left right"; the pieces themselves never
                // contain a space, because spaces are encoded as 'Ġ'.
                let Some((a, b)) = text.split_once(' ') else {
                    return Err(TokenizerError::BadMerge(text.to_string()));
                };
                merges.insert((a.to_string(), b.to_string()), rank as u32);
            }
        }

        // GGUF token types: 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED,
        // 5 UNUSED, 6 BYTE. CONTROL and USER_DEFINED are the ones that must be
        // matched literally rather than merged.
        let mut specials: Vec<(String, u32)> = Vec::new();
        if let Some(types) = meta
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
        {
            for (i, t) in types.iter().enumerate() {
                let ty = t.as_u64().unwrap_or(1);
                if (ty == 3 || ty == 4) && i < tokens.len() {
                    let text = &tokens[i];
                    // A single character marked USER_DEFINED is not a marker
                    // worth partitioning on, and matching one would slice
                    // ordinary text apart.
                    if text.len() > 2 {
                        specials.push((text.clone(), i as u32));
                    }
                }
            }
        }
        specials.sort_by_key(|(text, _)| std::cmp::Reverse(text.len()));

        let id_of = |key: &str| meta.get(key).and_then(Value::as_u64).map(|v| v as u32);
        let flag = |key: &str| matches!(meta.get(key), Some(Value::Bool(true)));

        // SentencePiece merges by score, so the array is not optional there —
        // an absent one would make every merge score 0.0 and reduce the
        // algorithm to "merge whatever is leftmost", which tokenizes without
        // complaint and produces the wrong stream.
        let scores: Vec<f32> = meta
            .get("tokenizer.ggml.scores")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_f32).collect())
            .unwrap_or_default();
        if kind == Kind::Spm && scores.len() != tokens.len() {
            return Err(TokenizerError::MissingScores {
                have: scores.len(),
                want: tokens.len(),
            });
        }

        Ok(Tokenizer {
            tokens,
            ids,
            merges,
            kind,
            scores,
            // SentencePiece prepends a space so the first word tokenizes as it
            // would mid-sentence. Containers may say otherwise; the default is
            // on, which is what every Llama-family model uses.
            add_dummy_prefix: match meta.get("tokenizer.ggml.add_space_prefix") {
                Some(Value::Bool(v)) => *v,
                _ => kind == Kind::Spm,
            },
            bos: id_of("tokenizer.ggml.bos_token_id"),
            eos: id_of("tokenizer.ggml.eos_token_id"),
            // Llama-family containers frequently omit the flag and still expect
            // BOS. Defaulting it on for SPM matches llama.cpp; for BPE the
            // absent flag genuinely means "no".
            add_bos: match meta.get("tokenizer.ggml.add_bos_token") {
                Some(Value::Bool(v)) => *v,
                _ => kind == Kind::Spm,
            },
            add_eos: flag("tokenizer.ggml.add_eos_token"),
            chat_template: meta
                .get("tokenizer.chat_template")
                .and_then(Value::as_str)
                .map(str::to_string),
            specials,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn id_of(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Which chat framing this container asks for.
    pub fn chat_format(&self) -> ChatFormat {
        ChatFormat::detect(self.chat_template())
    }

    /// Render messages into the prompt string this model was trained on.
    ///
    /// The end-of-sequence *text* comes from the vocabulary rather than being
    /// assumed to be `</s>`: Zephyr and Llama-2 embed it between turns, and a
    /// wrong one there is a token the model has never seen in that position.
    pub fn apply_chat_template(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        let eos = self
            .eos
            .and_then(|id| self.token_text(id))
            .unwrap_or("</s>");
        self.chat_format()
            .apply(messages, eos, add_generation_prompt)
    }

    /// Encode text to token ids, honouring the container's add_bos/add_eos.
    ///
    /// Control tokens in the text (`<|im_start|>`, `<|eot_id|>`, …) are mapped
    /// to their own ids rather than merged — see [`Self::specials`].
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos {
            out.extend(self.bos);
        }
        // Split on control tokens first, then run the ordinary algorithm on the
        // text between them. Without this a chat template is just characters.
        for (piece, special) in self.partition_specials(text) {
            match special {
                Some(id) => out.push(id),
                None => self.encode_plain(&piece, &mut out),
            }
        }
        if self.add_eos {
            out.extend(self.eos);
        }
        out
    }

    /// Cut `text` into ordinary spans and control tokens, in order.
    fn partition_specials(&self, text: &str) -> Vec<(String, Option<u32>)> {
        if self.specials.is_empty() || text.is_empty() {
            return vec![(text.to_string(), None)];
        }
        let mut out = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            // Earliest match wins; ties break to the longest, which is why
            // `specials` is sorted longest-first.
            let mut best: Option<(usize, &String, u32)> = None;
            for (tok, id) in &self.specials {
                if let Some(at) = rest.find(tok.as_str()) {
                    if best.is_none_or(|(b, _, _)| at < b) {
                        best = Some((at, tok, *id));
                    }
                }
            }
            match best {
                Some((at, tok, id)) => {
                    if at > 0 {
                        out.push((rest[..at].to_string(), None));
                    }
                    out.push((tok.clone(), Some(id)));
                    rest = &rest[at + tok.len()..];
                }
                None => {
                    out.push((rest.to_string(), None));
                    break 'outer;
                }
            }
        }
        out
    }

    /// The ordinary path, with no special-token handling.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        if self.kind == Kind::Spm {
            // No pre-tokenizer: SentencePiece works on the whole string, and
            // splitting it first would prevent merges across the boundaries the
            // splitter chose.
            out.extend(spm::encode(
                text,
                &self.ids,
                &self.scores,
                self.add_dummy_prefix,
            ));
            return;
        }
        for piece in pre_tokenize(text) {
            let encoded = bytes::encode(piece.as_bytes());
            for token in self.bpe(&encoded) {
                match self.ids.get(&token) {
                    Some(&id) => out.push(id),
                    // An unmergeable piece falls back to its individual
                    // characters, which a byte-level vocabulary always covers.
                    None => {
                        for ch in token.chars() {
                            if let Some(&id) = self.ids.get(&ch.to_string()) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Decode ids back to text.
    ///
    /// Lossy on invalid UTF-8, because a partial multi-byte character is
    /// normal when streaming one token at a time.
    /// Decode to **bytes**, without deciding they are valid UTF-8.
    ///
    /// # Why a streaming caller must use this
    ///
    /// One character is often several tokens. `😀` is four byte-fallback tokens,
    /// and a Persian or Chinese character is typically two or three; decoding
    /// each to a `String` on its own converts every incomplete fragment to `�`,
    /// permanently. The text is then unrecoverable no matter what the caller
    /// does downstream.
    ///
    /// So generation appends these bytes to a buffer and converts only at a
    /// valid UTF-8 boundary. [`Self::decode`] is for whole sequences, where the
    /// bytes are all present and the conversion is safe.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        if self.kind == Kind::Spm {
            let mut bytes = Vec::new();
            for &id in ids {
                if Some(id) == self.bos || Some(id) == self.eos {
                    continue;
                }
                if let Some(text) = self.token_text(id) {
                    bytes.extend(spm::piece_bytes(text));
                }
            }
            return bytes;
        }
        let joined: String = ids.iter().filter_map(|&id| self.token_text(id)).collect();
        bytes::decode(&joined)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        if self.kind == Kind::Spm {
            let mut bytes = Vec::new();
            for &id in ids {
                // BOS/EOS have no text of their own; emitting their spelling
                // ("<s>") would put markup in the user's output.
                if Some(id) == self.bos || Some(id) == self.eos {
                    continue;
                }
                if let Some(text) = self.token_text(id) {
                    bytes.extend(spm::piece_bytes(text));
                }
            }
            let text = String::from_utf8_lossy(&bytes).into_owned();
            // Undo the dummy prefix `encode` added — but **only** when this is a
            // whole sequence, which BOS in first position is the evidence for.
            //
            // Generation decodes one token at a time, and `▁The` must stay
            // " The" there. Stripping unconditionally ran every word together
            // ("Thecapital") — output that looks like a broken forward pass and
            // is really a detokenizer applying a whole-sequence rule per piece.
            let whole_sequence = self.bos.is_some() && ids.first().copied() == self.bos;
            return match whole_sequence && self.add_dummy_prefix {
                true => text.strip_prefix(' ').unwrap_or(&text).to_string(),
                false => text,
            };
        }
        let joined: String = ids.iter().filter_map(|&id| self.token_text(id)).collect();
        String::from_utf8_lossy(&bytes::decode(&joined)).into_owned()
    }

    /// Greedy BPE: repeatedly merge the adjacent pair with the lowest rank.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if parts.len() < 2 {
            return parts;
        }

        loop {
            // Find the best-ranked adjacent pair present in the merge table.
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len() - 1 {
                let key = (parts[i].clone(), parts[i + 1].clone());
                if let Some(&rank) = self.merges.get(&key) {
                    if best.is_none_or(|(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((at, _)) = best else { break };

            let merged = format!("{}{}", parts[at], parts[at + 1]);
            parts.splice(at..=at + 1, [merged]);
            if parts.len() < 2 {
                break;
            }
        }
        parts
    }
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab", &self.tokens.len())
            .field("merges", &self.merges.len())
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .finish()
    }
}
