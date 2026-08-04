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
mod pretok;

pub use bytes::{decode as bytes_decode, encode as bytes_encode};
pub use pretok::pre_tokenize;

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
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::MissingVocab => {
                f.write_str("container has no tokenizer.ggml.tokens array")
            }
            TokenizerError::UnsupportedModel(m) => {
                write!(f, "unsupported tokenizer model {m:?} (only byte-level BPE is implemented)")
            }
            TokenizerError::BadMerge(m) => write!(f, "malformed merge rule {m:?}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

pub struct Tokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    /// Merge pair -> rank. Lower rank merges first.
    merges: HashMap<(String, String), u32>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
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
        if model != "gpt2" {
            return Err(TokenizerError::UnsupportedModel(model.to_string()));
        }

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

        let id_of = |key: &str| meta.get(key).and_then(Value::as_u64).map(|v| v as u32);
        let flag = |key: &str| {
            matches!(meta.get(key), Some(Value::Bool(true)))
        };

        Ok(Tokenizer {
            tokens,
            ids,
            merges,
            bos: id_of("tokenizer.ggml.bos_token_id"),
            eos: id_of("tokenizer.ggml.eos_token_id"),
            add_bos: flag("tokenizer.ggml.add_bos_token"),
            add_eos: flag("tokenizer.ggml.add_eos_token"),
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

    /// Encode text to token ids, honouring the container's add_bos/add_eos.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos {
            out.extend(self.bos);
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
        if self.add_eos {
            out.extend(self.eos);
        }
        out
    }

    /// Decode ids back to text.
    ///
    /// Lossy on invalid UTF-8, because a partial multi-byte character is
    /// normal when streaming one token at a time.
    pub fn decode(&self, ids: &[u32]) -> String {
        let joined: String = ids
            .iter()
            .filter_map(|&id| self.token_text(id))
            .collect();
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
