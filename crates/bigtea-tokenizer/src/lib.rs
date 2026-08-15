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
pub mod rwkv;
pub mod spm;
pub mod ugm;
pub mod wpm;

pub use bytes::{decode as bytes_decode, encode as bytes_encode};
pub use chat::{ChatFormat, Message};
pub use pretok::{pre_tokenize, PreTokenizer, UnknownPreTokenizer};

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
    /// `"bert"` — WordPiece: longest-prefix match, `##` continuations, and one
    /// `[UNK]` for any word the vocabulary cannot cover. No byte fallback.
    Wpm,
    /// `"t5"` — Unigram: the highest-scoring path through a lattice of every
    /// possible segmentation, not a greedy merge. See [`ugm`].
    Ugm,
    /// `"rwkv"` — greedy longest match over a trie of raw byte strings. No
    /// merge table, no scores, and **the vocabulary is stored escaped**. See
    /// [`rwkv`].
    Rwkv,
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
    /// The container asks for a pre-tokenizer this build has not verified.
    UnsupportedPreTokenizer(UnknownPreTokenizer),
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
                     (implemented: \"gpt2\" byte-level BPE, \"llama\" SentencePiece,                       \"bert\" WordPiece, \"t5\" Unigram)"
                )
            }
            TokenizerError::BadMerge(m) => write!(f, "malformed merge rule {m:?}"),
            TokenizerError::UnsupportedPreTokenizer(e) => write!(f, "{e}"),
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
    /// Set by `--chat-template`, and it wins over detection.
    chat_override: Option<ChatFormat>,
    /// The id for text the vocabulary cannot represent. WordPiece has no byte
    /// fallback, so without this an unknown word would silently disappear.
    unk: Option<u32>,
    /// Which continuation spelling this WordPiece vocabulary uses. Read off
    /// the vocabulary rather than assumed -- see [`wpm`], where guessing it
    /// costs every ordinary word without producing an error.
    wpm_spelling: wpm::Spelling,
    /// Which pre-tokenizer this container asked for. Only byte-level BPE uses
    /// one; SentencePiece, WordPiece and Unigram do their own splitting.
    pre: PreTokenizer,
    /// Whether each token is `USER_DEFINED`, indexed by id. Unigram scores those
    /// 0 so an added token beats any ordinary segmentation of the same span.
    user_defined: Vec<bool>,
    /// Longest token in bytes, which bounds Unigram's prefix search.
    max_token_len: usize,
    /// SentencePiece's `remove_extra_whitespaces`.
    remove_extra_whitespaces: bool,
    /// Built once, on first use. A 65k-entry trie is not worth constructing for
    /// a tokenizer that will never take the RWKV path.
    rwkv_trie: std::sync::OnceLock<rwkv::Trie>,
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
            "rwkv" => Kind::Rwkv,
            "bert" => Kind::Wpm,
            "t5" => Kind::Ugm,
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

        // USER_DEFINED (4), indexed by id. Unigram gives these a score of 0.
        let mut user_defined = vec![false; tokens.len()];
        if let Some(types) = meta
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
        {
            for (i, t) in types.iter().enumerate() {
                if t.as_u64().unwrap_or(1) == 4 && i < user_defined.len() {
                    user_defined[i] = true;
                }
            }
        }

        let id_of = |key: &str| meta.get(key).and_then(Value::as_u64).map(|v| v as u32);

        // SentencePiece merges by score, so the array is not optional there —
        // an absent one would make every merge score 0.0 and reduce the
        // algorithm to "merge whatever is leftmost", which tokenizes without
        // complaint and produces the wrong stream.
        let scores: Vec<f32> = meta
            .get("tokenizer.ggml.scores")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_f32).collect())
            .unwrap_or_default();
        if matches!(kind, Kind::Spm | Kind::Ugm) && scores.len() != tokens.len() {
            return Err(TokenizerError::MissingScores {
                have: scores.len(),
                want: tokens.len(),
            });
        }

        // Before the struct takes ownership of `tokens`.
        let wpm_spelling = wpm::detect_spelling(&tokens);
        let max_token_len = ugm::max_token_len(&tokens);

        // `tokenizer.ggml.pre` selects the splitting rule, and it was previously
        // read by nobody: a Qwen container was split with Llama's rule, which
        // groups three digits where Qwen takes one, so every number and every
        // boundary after it moved. Only byte-level BPE consults it -- the other
        // three do their own splitting -- so an unfamiliar name on those is not
        // a reason to refuse the model.
        let pre = match kind {
            Kind::Bpe => {
                let name = meta
                    .get("tokenizer.ggml.pre")
                    .and_then(Value::as_str)
                    // **An absent key is not "llama-bpe".** llama.cpp falls
                    // back to its DEFAULT GPT-2 rule, whose first pass cuts a
                    // run of punctuation out whole -- so `def fibonacci(n):`
                    // is five pieces there and was four here. A6c refused every
                    // unknown `pre` by name and then guessed this case, which
                    // is the same mistake one layer down.
                    .unwrap_or("default");
                PreTokenizer::from_name(name).map_err(TokenizerError::UnsupportedPreTokenizer)?
            }
            _ => PreTokenizer::LlamaBpe,
        };

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
                _ => matches!(kind, Kind::Spm | Kind::Ugm),
            },
            // Detection until `--chat-template` says otherwise.
            chat_override: None,
            unk: id_of("tokenizer.ggml.unknown_token_id"),
            wpm_spelling,
            pre,
            user_defined,
            max_token_len,
            rwkv_trie: std::sync::OnceLock::new(),
            remove_extra_whitespaces: match meta.get("tokenizer.ggml.remove_extra_whitespaces") {
                Some(Value::Bool(v)) => *v,
                _ => true,
            },
            // 11 for a BPE vocabulary that names no BOS. That is not a guess:
            // llama.cpp's `tokenizer_model == "gpt2"` branch sets
            // `special_bos_id = special_eos_id = 11` before the container's own
            // keys are read. **Falcon3 declares an EOS and no BOS**, so the
            // default is what it runs on, and `add_bos` below is what makes it
            // matter.
            bos: id_of("tokenizer.ggml.bos_token_id").or(match kind {
                Kind::Bpe => Some(11),
                _ => None,
            }),
            eos: id_of("tokenizer.ggml.eos_token_id"),
            // Llama-family containers frequently omit the flag and still expect
            // BOS. Defaulting it on for SPM matches llama.cpp; for BPE it
            // depends on the **pre-tokenizer**, not on the vocabulary kind.
            add_bos: match meta.get("tokenizer.ggml.add_bos_token") {
                Some(Value::Bool(v)) => *v,
                // BERT containers declare neither flag and still expect the
                // sequence wrapped in [CLS] .. [SEP] -- llama.cpp adds both
                // unconditionally for WordPiece, and a missing [CLS] shifts
                // every position by one.
                //
                // The `llama3`/`llama-bpe`/`falcon3` arm sets `add_bos = true`
                // there too, and Falcon3 is the container that declares neither
                // the flag nor a BOS id. Without both halves of this it saw a
                // sequence one token shorter than the reference did, and then
                // disagreed from the first generated token — on two prompts,
                // while five more sat close enough to read as near-ties.
                //
                // `kind == Bpe` is load-bearing: `pre` is set to `LlamaBpe` as
                // a placeholder for every non-BPE vocabulary, so testing it
                // alone would switch BOS on for T5, which has none.
                _ => {
                    matches!(kind, Kind::Spm | Kind::Wpm)
                        || (kind == Kind::Bpe && pre == PreTokenizer::LlamaBpe)
                }
            },
            add_eos: match meta.get("tokenizer.ggml.add_eos_token") {
                Some(Value::Bool(v)) => *v,
                _ => kind == Kind::Wpm,
            },
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

    /// Whether `text` already begins with the BOS token written out.
    ///
    /// Only the *literal* spelling counts. A prompt that merely starts with the
    /// same characters as some other token is untouched, and a model with no
    /// BOS, or one whose BOS has empty text, always answers `false` rather than
    /// matching at every position.
    fn opens_with_bos(&self, text: &str) -> bool {
        let Some(bos) = self.bos else { return false };
        let Some(spelling) = self.token_text(bos) else {
            return false;
        };
        !spelling.is_empty() && text.starts_with(spelling)
    }

    /// Whether `id` is a control token — `<|im_end|>`, `<s>`, `[CLS]`.
    ///
    /// These are framing, not output. `decode` deliberately drops most of them,
    /// so a caller that wants to *show* them (llama.cpp's `--special`) has to
    /// ask which ones they are rather than inferring it from the text.
    pub fn is_control(&self, id: u32) -> bool {
        if Some(id) == self.bos || Some(id) == self.eos {
            return true;
        }
        self.specials.iter().any(|(_, sid)| *sid == id)
    }

    /// A control token's literal spelling, for `--special`.
    ///
    /// Returns the raw vocabulary entry rather than routing through `decode`,
    /// which is exactly the code that hides these.
    pub fn control_text(&self, id: u32) -> String {
        self.token_text(id).unwrap_or_default().to_string()
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Which chat framing this container asks for.
    pub fn chat_format(&self) -> ChatFormat {
        self.chat_override
            .unwrap_or_else(|| ChatFormat::detect(self.chat_template()))
    }

    /// Force a chat format, ignoring what the container declares.
    ///
    /// `--chat-template`. Two cases make this necessary rather than a
    /// curiosity: a container with no template at all (many base-model
    /// conversions), and one whose template this build does not recognise, both
    /// of which otherwise fall back to a plain framing the model was never
    /// trained on.
    pub fn set_chat_format(&mut self, format: ChatFormat) {
        self.chat_override = Some(format);
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
        // **Not when the text already opens with it.** A chat template
        // evaluated by `--jinja` often emits the BOS token itself — Gemma's
        // begins with a literal `<bos>`, Llama-3's with `<|begin_of_text|>` —
        // and `partition_specials` below correctly maps that to its own id. Add
        // one here as well and the model is prefilled a token LONG:
        //
        //   bigtea --jinja : [2, 2, 105, 2364, ...]
        //   llama.cpp      : [2,    105, 2364, ...]
        //
        // Measured on gemma-3, Llama-3.2, internlm2 and Phi-3. It is the mirror
        // of the Falcon3 bug, which was prefilled a token SHORT, and it is just
        // as quiet: the model answers fluently from a position nobody trained.
        // The hardcoded family renderers were unaffected because they do not
        // emit the BOS text, which is why this only appeared once a real Jinja
        // engine started evaluating the container's own template.
        if self.add_bos && !self.opens_with_bos(text) {
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
        if self.kind == Kind::Rwkv {
            // No pre-tokenizer at all: RWKV walks the raw bytes and takes the
            // longest vocabulary entry at each position. Splitting the text
            // first would forbid entries that span a split point, which is
            // most of the interesting ones.
            out.extend(rwkv::encode(
                self.rwkv_trie.get_or_init(|| rwkv::build(&self.tokens)),
                text,
                self.unk,
            ));
            return;
        }
        if self.kind == Kind::Wpm {
            // No pre-tokenizer and no byte fallback: WordPiece does its own
            // splitting, and anything it cannot cover becomes one [UNK].
            out.extend(wpm::encode(text, &self.ids, self.unk, self.wpm_spelling));
            return;
        }
        if self.kind == Kind::Ugm {
            // Unigram scores whole segmentations, so pre-splitting would forbid
            // the very paths the lattice exists to compare.
            let normalized =
                ugm::normalize(text, self.add_dummy_prefix, self.remove_extra_whitespaces);
            out.extend(ugm::encode(
                &normalized,
                &self.ids,
                &self.scores,
                &self.user_defined,
                self.unk,
                self.max_token_len,
            ));
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
        for piece in pre_tokenize(text, self.pre) {
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
        if self.kind == Kind::Rwkv {
            // The vocabulary is stored ESCAPED, so decoding has to unescape --
            // emitting the stored text would put a literal `\n` in the output
            // where the model produced a newline.
            let mut bytes = Vec::new();
            for &id in ids {
                if let Some(text) = self.token_text(id) {
                    bytes.extend(rwkv::unescape(text));
                }
            }
            return bytes;
        }
        if self.kind == Kind::Wpm {
            return self.decode(ids).into_bytes();
        }
        if matches!(self.kind, Kind::Spm | Kind::Ugm) {
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
        self.bpe_decode_bytes(ids)
    }

    /// The BPE arm of llama.cpp's `token_to_piece`: **a USER_DEFINED token's
    /// text is copied verbatim; only a NORMAL token goes through the byte
    /// alphabet.**
    ///
    /// Falcon3 is why this is not one `bytes::decode` over the joined text.
    /// Its newline is id 12, marked USER_DEFINED, and holds a *raw* `\n` rather
    /// than the `Ċ` a GPT-2-family vocabulary usually spells it with.
    /// `bytes::decode` drops every character outside that alphabet, so the
    /// newline vanished and generation arrived as one run-on line —
    /// `Paris.Q: What is…` where the reference had a paragraph. Nothing failed:
    /// the ids were right and only the rendering was wrong, so it surfaced as a
    /// parity mismatch rather than an error.
    ///
    /// Runs of NORMAL tokens are still decoded **together**, because one
    /// character is often several byte tokens and decoding each alone would
    /// drop the continuation bytes.
    fn bpe_decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut run = String::new();
        for &id in ids {
            let Some(text) = self.token_text(id) else {
                continue;
            };
            if self.user_defined.get(id as usize).copied().unwrap_or(false) {
                out.extend(bytes::decode(&std::mem::take(&mut run)));
                out.extend_from_slice(text.as_bytes());
            } else {
                run.push_str(text);
            }
        }
        out.extend(bytes::decode(&run));
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        if self.kind == Kind::Wpm {
            // [CLS]/[SEP] carry no text; printing their spelling would put
            // markup in the user's output.
            let pieces: Vec<&str> = ids
                .iter()
                .filter(|id| Some(**id) != self.bos && Some(**id) != self.eos)
                .filter_map(|&id| self.token_text(id))
                .collect();
            return wpm::decode(&pieces, self.wpm_spelling);
        }
        if matches!(self.kind, Kind::Spm | Kind::Ugm) {
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
            // whole sequence.
            //
            // Generation decodes one token at a time, and `▁The` must stay
            // " The" there. Stripping unconditionally ran every word together
            // ("Thecapital") — output that looks like a broken forward pass and
            // is really a detokenizer applying a whole-sequence rule per piece.
            //
            // BOS in first position is the evidence for the Llama family. **T5
            // has no BOS at all** (`add_bos_token = false`), so that test never
            // fires there and every decoded T5 sequence kept a leading space.
            // A terminating EOS is the same evidence for a family that brackets
            // the other way; a lone generated token is neither.
            let whole_sequence = (self.bos.is_some() && ids.first().copied() == self.bos)
                || (self.eos.is_some() && ids.len() > 1 && ids.last().copied() == self.eos);
            return match whole_sequence && self.add_dummy_prefix {
                true => text.strip_prefix(' ').unwrap_or(&text).to_string(),
                false => text,
            };
        }
        String::from_utf8_lossy(&self.bpe_decode_bytes(ids)).into_owned()
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
