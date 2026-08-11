//! Turning a list of messages into the exact string a chat model expects.
//!
//! # Why this is not optional
//!
//! An instruct model is trained on one specific framing — `<|im_start|>user`,
//! or `[INST]`, or `<|start_header_id|>user<|end_header_id|>`. Hand it the raw
//! text instead and it does not fail: it continues the text, comments on the
//! question, or answers as though it were still writing the prompt. That is
//! exactly what Bigtea did before this module — asked to "Write one sentence
//! about the sea", Llama-3.2 replied *"The sentence should be concise and
//! evocative"*, because without the framing it was completing an instruction
//! rather than following one.
//!
//! So `/v1/chat/completions` cannot be honest without this, and neither can any
//! quality comparison against llama.cpp.
//!
//! # Detection, not evaluation
//!
//! GGUF stores `tokenizer.chat_template` as a **Jinja2 template**. Evaluating
//! Jinja properly means a whole expression language — Llama-3's template alone
//! uses `set`, `if defined`, loops, `strftime_now` and tool-call branches — and
//! a half-implemented one silently produces the wrong framing, which is the
//! failure mode this project is most expensive at.
//!
//! llama.cpp does not evaluate them either. It matches the template against
//! known families by substring and applies a hardcoded formatter, and that is
//! what happens here. It is honest about its limits: an unrecognised template
//! reports itself as [`ChatFormat::Generic`] so the caller can say so rather
//! than pretend.

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Message {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

/// The families this build knows how to frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    /// `<|im_start|>role\ncontent<|im_end|>` — Qwen, Yi, many finetunes.
    ChatMl,
    /// `<|start_header_id|>role<|end_header_id|>` — Llama 3.x.
    Llama3,
    /// `[INST] ... [/INST]`, with `<<SYS>>` for the system turn — Llama 2.
    Llama2,
    /// `[INST] ... [/INST]` with no system role — Mistral.
    Mistral,
    /// `<|user|>\ncontent</s>` — Zephyr, TinyLlama-Chat.
    Zephyr,
    /// `<|user|>\ncontent<|end|>` — Phi-3.
    Phi3,
    /// `<start_of_turn>user ... <end_of_turn>` — Gemma. No system role.
    Gemma,
    /// `USER: ... ASSISTANT:` — Vicuna.
    Vicuna,
    /// `### Instruction:` / `### Response:` — Alpaca and DeepSeek-Coder.
    Alpaca,
    /// Nothing matched. Framed plainly, and the caller should say it is a guess.
    /// DeepSeek v1: `### Instruction:` / `### Response:` with its own markers.
    DeepSeek,
    /// DeepSeek v2: `User: ` / `Assistant: ` with an EOS between turns.
    DeepSeek2,
    /// DeepSeek v3: `<｜User｜>` / `<｜Assistant｜>` — full-width bars, and
    /// they are not the ASCII `|`. A near-miss here is a token the model has
    /// never seen in that position.
    DeepSeek3,
    /// Command-R: `<|START_OF_TURN_TOKEN|><|USER_TOKEN|>`.
    CommandR,
    /// ChatGLM 3: `<|user|>` without the newline Zephyr uses.
    ChatGlm3,
    /// ChatGLM 4 / GLM-Edge: `[gMASK]<sop>` preamble then `<|role|>`.
    ChatGlm4,
    /// Mistral v7: `[SYSTEM_PROMPT]` and a space after `[INST]`.
    MistralV7,
    /// Falcon 3 and similar: plain `User:`/`Assistant:` with double newlines.
    Falcon3,
    /// OpenChat: `GPT4 Correct User:` … `<|end_of_turn|>`.
    OpenChat,
    /// Orion: `Human: ` … `Assistant: ` with EOS after the assistant turn.
    Orion,
    /// MiniCPM: `<用户>` … `<AI>`.
    MiniCpm,
    /// Granite: `<|start_of_role|>user<|end_of_role|>`.
    Granite,
    /// EXAONE 3: `[|user|]` … `[|assistant|]`.
    Exaone3,
    /// Phi-4: `<|im_sep|>` rather than a newline after the role.
    Phi4,
    /// RWKV-World: `User: ` … `Assistant:` with double newlines.
    RwkvWorld,
    /// Monarch/Bailing: `<role>HUMAN</role>` … `<role>ASSISTANT</role>`.
    Monarch,
    Generic,
}

impl ChatFormat {
    /// Identify the family from the raw Jinja template.
    ///
    /// Order matters: several templates contain more than one marker. Phi-3 and
    /// Zephyr both use `<|user|>`, so Phi-3's `<|end|>` has to be checked first;
    /// Llama-2's `[INST]` appears in Mistral's too, so `<<SYS>>` discriminates.
    pub fn detect(template: Option<&str>) -> Self {
        let Some(t) = template else {
            return ChatFormat::Generic;
        };
        // Order matters and is most-specific-first: several of these share a
        // marker with another, and a looser rule placed earlier silently wins.
        // `<|im_sep|>` before `<|im_start|>` is the clearest case — Phi-4 has
        // both, and matching ChatML first renders it with a newline where the
        // model was trained on a separator token.
        if t.contains("<|im_sep|>") {
            ChatFormat::Phi4
        } else if t.contains("<\u{ff5c}User\u{ff5c}>") || t.contains("<\u{ff5c}Assistant\u{ff5c}>")
        {
            ChatFormat::DeepSeek3
        } else if t.contains("<|START_OF_TURN_TOKEN|>") {
            ChatFormat::CommandR
        } else if t.contains("[gMASK]") {
            ChatFormat::ChatGlm4
        } else if t.contains("<|start_of_role|>") {
            ChatFormat::Granite
        } else if t.contains("[|assistant|]") {
            ChatFormat::Exaone3
        } else if t.contains("<\u{7528}\u{6237}>") {
            ChatFormat::MiniCpm
        } else if t.contains("GPT4 Correct") {
            ChatFormat::OpenChat
        } else if t.contains("<role>HUMAN</role>") {
            ChatFormat::Monarch
        } else if t.contains("[SYSTEM_PROMPT]") {
            ChatFormat::MistralV7
        } else if t.contains("<|im_start|>") {
            ChatFormat::ChatMl
        } else if t.contains("<|start_header_id|>") {
            ChatFormat::Llama3
        } else if t.contains("<start_of_turn>") {
            ChatFormat::Gemma
        } else if t.contains("[INST]") {
            // `<<SYS>>` is what separates Llama 2 from Mistral, which shares the
            // brackets but has no system turn.
            if t.contains("<<SYS>>") {
                ChatFormat::Llama2
            } else {
                ChatFormat::Mistral
            }
        } else if t.contains("<|end|>") && t.contains("<|assistant|>") {
            ChatFormat::Phi3
        } else if t.contains("<|user|>") || t.contains("<|assistant|>") {
            ChatFormat::Zephyr
        } else if t.contains("### Instruction") {
            ChatFormat::Alpaca
        } else if t.contains("ASSISTANT:") || t.contains("USER:") {
            ChatFormat::Vicuna
        } else if t.contains("### Response:") {
            ChatFormat::DeepSeek
        } else if t.contains("Assistant: ") && t.contains("User: ") {
            // DeepSeek 2 and RWKV-World differ only in the blank line between
            // turns, which is the sort of thing that is invisible when reading
            // and decisive when tokenised.
            if t.contains("\n\n") {
                ChatFormat::RwkvWorld
            } else {
                ChatFormat::DeepSeek2
            }
        } else if t.contains("Human: ") {
            ChatFormat::Orion
        } else if t.contains("User:") && t.contains("Falcon") {
            ChatFormat::Falcon3
        } else {
            ChatFormat::Generic
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ChatFormat::ChatMl => "chatml",
            ChatFormat::Llama3 => "llama3",
            ChatFormat::Llama2 => "llama2",
            ChatFormat::Mistral => "mistral",
            ChatFormat::Zephyr => "zephyr",
            ChatFormat::Phi3 => "phi3",
            ChatFormat::Gemma => "gemma",
            ChatFormat::Vicuna => "vicuna",
            ChatFormat::Alpaca => "alpaca",
            ChatFormat::DeepSeek => "deepseek",
            ChatFormat::DeepSeek2 => "deepseek2",
            ChatFormat::DeepSeek3 => "deepseek3",
            ChatFormat::CommandR => "command-r",
            ChatFormat::ChatGlm3 => "chatglm3",
            ChatFormat::ChatGlm4 => "chatglm4",
            ChatFormat::MistralV7 => "mistral-v7",
            ChatFormat::Falcon3 => "falcon3",
            ChatFormat::OpenChat => "openchat",
            ChatFormat::Orion => "orion",
            ChatFormat::MiniCpm => "minicpm",
            ChatFormat::Granite => "granite",
            ChatFormat::Exaone3 => "exaone3",
            ChatFormat::Phi4 => "phi4",
            ChatFormat::RwkvWorld => "rwkv-world",
            ChatFormat::Monarch => "monarch",
            ChatFormat::Generic => "generic",
        }
    }

    /// Look a format up by the name [`Self::name`] prints.
    ///
    /// The inverse of `name`, for `--chat-template`. `None` for anything
    /// unrecognised so the caller can refuse rather than quietly falling back
    /// to the generic framing, which is how a model ends up answering the
    /// wrong question fluently.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "chatml" => ChatFormat::ChatMl,
            "llama3" => ChatFormat::Llama3,
            "llama2" => ChatFormat::Llama2,
            "mistral" => ChatFormat::Mistral,
            "zephyr" => ChatFormat::Zephyr,
            "phi3" => ChatFormat::Phi3,
            "gemma" => ChatFormat::Gemma,
            "vicuna" => ChatFormat::Vicuna,
            "alpaca" => ChatFormat::Alpaca,
            "deepseek" => ChatFormat::DeepSeek,
            "deepseek2" => ChatFormat::DeepSeek2,
            "deepseek3" => ChatFormat::DeepSeek3,
            "command-r" => ChatFormat::CommandR,
            "chatglm3" => ChatFormat::ChatGlm3,
            "chatglm4" | "glmedge" => ChatFormat::ChatGlm4,
            "mistral-v7" => ChatFormat::MistralV7,
            "falcon3" => ChatFormat::Falcon3,
            "openchat" => ChatFormat::OpenChat,
            "orion" => ChatFormat::Orion,
            "minicpm" => ChatFormat::MiniCpm,
            "granite" => ChatFormat::Granite,
            "exaone3" => ChatFormat::Exaone3,
            "phi4" => ChatFormat::Phi4,
            "rwkv-world" => ChatFormat::RwkvWorld,
            "monarch" | "bailing" => ChatFormat::Monarch,
            _ => return None,
        })
    }

    /// Every name `from_name` accepts, for an error message that lists them.
    pub fn known_names() -> &'static [&'static str] {
        &[
            "chatml",
            "llama3",
            "llama2",
            "mistral",
            "mistral-v7",
            "zephyr",
            "phi3",
            "phi4",
            "gemma",
            "vicuna",
            "alpaca",
            "deepseek",
            "deepseek2",
            "deepseek3",
            "command-r",
            "chatglm3",
            "chatglm4",
            "falcon3",
            "openchat",
            "orion",
            "minicpm",
            "granite",
            "exaone3",
            "rwkv-world",
            "monarch",
        ]
    }

    /// Whether this build actually recognised the template.
    pub fn is_known(&self) -> bool {
        !matches!(self, ChatFormat::Generic)
    }

    /// Render `messages` into the prompt string this family expects.
    ///
    /// `eos` is the container's end-of-sequence text, which several families
    /// embed between turns. `add_generation_prompt` opens the assistant's turn
    /// and leaves it open — without it the model is being asked to continue the
    /// *user's* message, which is a common and confusing mistake.
    pub fn apply(&self, messages: &[Message], eos: &str, add_generation_prompt: bool) -> String {
        let mut out = String::new();
        match self {
            ChatFormat::ChatMl => {
                for m in messages {
                    out.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n",
                        m.role, m.content
                    ));
                }
                if add_generation_prompt {
                    out.push_str("<|im_start|>assistant\n");
                }
            }
            ChatFormat::Llama3 => {
                for m in messages {
                    out.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        m.role, m.content
                    ));
                }
                if add_generation_prompt {
                    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
            }
            ChatFormat::Zephyr => {
                for m in messages {
                    out.push_str(&format!("<|{}|>\n{}{eos}\n", m.role, m.content));
                }
                if add_generation_prompt {
                    out.push_str("<|assistant|>\n");
                }
            }
            ChatFormat::Phi3 => {
                for m in messages {
                    out.push_str(&format!("<|{}|>\n{}<|end|>\n", m.role, m.content));
                }
                if add_generation_prompt {
                    out.push_str("<|assistant|>\n");
                }
            }
            ChatFormat::Gemma => {
                // Gemma has no system role and calls the assistant "model".
                // A system message is folded into the first user turn rather
                // than dropped, which would silently lose the instruction.
                let mut pending_system = String::new();
                for m in messages {
                    match m.role.as_str() {
                        "system" => {
                            pending_system.push_str(&m.content);
                            pending_system.push_str("\n\n");
                        }
                        role => {
                            let who = if role == "assistant" { "model" } else { "user" };
                            let body = if who == "user" && !pending_system.is_empty() {
                                let joined = format!("{pending_system}{}", m.content);
                                pending_system.clear();
                                joined
                            } else {
                                m.content.clone()
                            };
                            out.push_str(&format!("<start_of_turn>{who}\n{body}<end_of_turn>\n"));
                        }
                    }
                }
                if add_generation_prompt {
                    out.push_str("<start_of_turn>model\n");
                }
            }
            ChatFormat::Llama2 | ChatFormat::Mistral => {
                let with_sys = matches!(self, ChatFormat::Llama2);
                let mut system = String::new();
                let mut first = true;
                for m in messages {
                    match m.role.as_str() {
                        "system" => system = m.content.clone(),
                        "user" => {
                            let body = if first && with_sys && !system.is_empty() {
                                format!("<<SYS>>\n{system}\n<</SYS>>\n\n{}", m.content)
                            } else if first && !system.is_empty() {
                                // Mistral has no system slot, so it is prepended
                                // rather than dropped.
                                format!("{system}\n\n{}", m.content)
                            } else {
                                m.content.clone()
                            };
                            out.push_str(&format!("[INST] {body} [/INST]"));
                            first = false;
                        }
                        _ => out.push_str(&format!(" {}{eos}", m.content)),
                    }
                }
            }
            ChatFormat::Vicuna => {
                for m in messages {
                    match m.role.as_str() {
                        "system" => out.push_str(&format!("{}\n\n", m.content)),
                        "user" => out.push_str(&format!("USER: {}\n", m.content)),
                        _ => out.push_str(&format!("ASSISTANT: {}{eos}\n", m.content)),
                    }
                }
                if add_generation_prompt {
                    out.push_str("ASSISTANT:");
                }
            }
            ChatFormat::Alpaca => {
                for m in messages {
                    match m.role.as_str() {
                        "system" => out.push_str(&format!("{}\n\n", m.content)),
                        "user" => out.push_str(&format!("### Instruction:\n{}\n\n", m.content)),
                        _ => out.push_str(&format!("### Response:\n{}\n\n", m.content)),
                    }
                }
                if add_generation_prompt {
                    out.push_str("### Response:\n");
                }
            }
            ChatFormat::DeepSeek => {
                for m in messages {
                    match m.role.as_str() {
                        "system" => out.push_str(&format!("{}\n\n", m.content)),
                        "user" => out.push_str(&format!("### Instruction:\n{}\n\n", m.content)),
                        _ => out.push_str(&format!("### Response:\n{}\n<|EOT|>\n", m.content)),
                    }
                }
                if add_generation_prompt {
                    out.push_str("### Response:\n");
                }
            }
            ChatFormat::DeepSeek2 => {
                for m in messages {
                    match m.role.as_str() {
                        "system" => out.push_str(&format!("{}\n\n", m.content)),
                        "user" => out.push_str(&format!("User: {}\n\n", m.content)),
                        _ => out.push_str(&format!("Assistant: {}{eos}", m.content)),
                    }
                }
                if add_generation_prompt {
                    out.push_str("Assistant:");
                }
            }
            ChatFormat::DeepSeek3 => {
                // Full-width bars U+FF5C, not ASCII `|`. Getting this wrong is
                // a token the model has never seen in that position.
                for m in messages {
                    match m.role.as_str() {
                        "system" => out.push_str(&format!("{}\n\n", m.content)),
                        "user" => out.push_str(&format!("<\u{ff5c}User\u{ff5c}>{}", m.content)),
                        _ => out.push_str(&format!(
                            "<\u{ff5c}Assistant\u{ff5c}>{}<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
                            m.content
                        )),
                    }
                }
                if add_generation_prompt {
                    out.push_str("<\u{ff5c}Assistant\u{ff5c}>");
                }
            }
            ChatFormat::CommandR => {
                for m in messages {
                    let tag = match m.role.as_str() {
                        "system" => "SYSTEM_TOKEN",
                        "user" => "USER_TOKEN",
                        _ => "CHATBOT_TOKEN",
                    };
                    out.push_str(&format!(
                        "<|START_OF_TURN_TOKEN|><|{tag}|>{}<|END_OF_TURN_TOKEN|>",
                        m.content.trim()
                    ));
                }
                if add_generation_prompt {
                    out.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
                }
            }
            ChatFormat::ChatGlm3 => {
                for m in messages {
                    out.push_str(&format!("<|{}|>\n{}", m.role, m.content));
                }
                if add_generation_prompt {
                    out.push_str("<|assistant|>");
                }
            }
            ChatFormat::ChatGlm4 => {
                out.push_str("[gMASK]<sop>");
                for m in messages {
                    out.push_str(&format!("<|{}|>\n{}", m.role, m.content));
                }
                if add_generation_prompt {
                    out.push_str("<|assistant|>");
                }
            }
            ChatFormat::MistralV7 => {
                // A space after [INST] and before [/INST]; Mistral v7 is
                // whitespace-sensitive where v1 is not.
                for m in messages {
                    match m.role.as_str() {
                        "system" => {
                            out.push_str(&format!("[SYSTEM_PROMPT] {}[/SYSTEM_PROMPT]", m.content))
                        }
                        "user" => out.push_str(&format!("[INST] {}[/INST]", m.content)),
                        _ => out.push_str(&format!(" {}{eos}", m.content)),
                    }
                }
            }
            ChatFormat::Falcon3 | ChatFormat::RwkvWorld => {
                for m in messages {
                    match m.role.as_str() {
                        "system" => out.push_str(&format!("System: {}\n\n", m.content)),
                        "user" => out.push_str(&format!("User: {}\n\n", m.content)),
                        _ => out.push_str(&format!("Assistant: {}\n\n", m.content)),
                    }
                }
                if add_generation_prompt {
                    out.push_str("Assistant:");
                }
            }
            ChatFormat::OpenChat => {
                for m in messages {
                    match m.role.as_str() {
                        "system" => out.push_str(&format!("{}<|end_of_turn|>", m.content)),
                        "user" => out
                            .push_str(&format!("GPT4 Correct User: {}<|end_of_turn|>", m.content)),
                        _ => out.push_str(&format!(
                            "GPT4 Correct Assistant: {}<|end_of_turn|>",
                            m.content
                        )),
                    }
                }
                if add_generation_prompt {
                    out.push_str("GPT4 Correct Assistant:");
                }
            }
            ChatFormat::Orion => {
                for m in messages {
                    match m.role.as_str() {
                        "system" => out.push_str(&m.content),
                        "user" => out.push_str(&format!("Human: {}\n\nAssistant: ", m.content)),
                        _ => out.push_str(&format!("{}{eos}", m.content)),
                    }
                }
            }
            ChatFormat::MiniCpm => {
                for m in messages {
                    match m.role.as_str() {
                        "user" => out.push_str(&format!("<\u{7528}\u{6237}>{}", m.content.trim())),
                        _ => out.push_str(&format!("<AI>{}", m.content.trim())),
                    }
                }
                if add_generation_prompt {
                    out.push_str("<AI>");
                }
            }
            ChatFormat::Granite => {
                for m in messages {
                    out.push_str(&format!(
                        "<|start_of_role|>{}<|end_of_role|>\n{}<|end_of_text|>\n",
                        m.role, m.content
                    ));
                }
                if add_generation_prompt {
                    out.push_str("<|start_of_role|>assistant<|end_of_role|>\n");
                }
            }
            ChatFormat::Exaone3 => {
                for m in messages {
                    match m.role.as_str() {
                        "system" => {
                            out.push_str(&format!("[|system|]{}[|endofturn|]\n", m.content))
                        }
                        "user" => out.push_str(&format!("[|user|]{}\n", m.content)),
                        _ => out.push_str(&format!("[|assistant|]{}[|endofturn|]\n", m.content)),
                    }
                }
                if add_generation_prompt {
                    out.push_str("[|assistant|]");
                }
            }
            ChatFormat::Phi4 => {
                for m in messages {
                    out.push_str(&format!(
                        "<|im_start|>{}<|im_sep|>{}<|im_end|>",
                        m.role, m.content
                    ));
                }
                if add_generation_prompt {
                    out.push_str("<|im_start|>assistant<|im_sep|>");
                }
            }
            ChatFormat::Monarch => {
                for m in messages {
                    let role = match m.role.as_str() {
                        "user" => "HUMAN",
                        "system" => "SYSTEM",
                        _ => "ASSISTANT",
                    };
                    out.push_str(&format!("<role>{role}</role>{}", m.content));
                }
                if add_generation_prompt {
                    out.push_str("<role>ASSISTANT</role>");
                }
            }
            ChatFormat::Generic => {
                // Plain and readable. Not a guess at a family — a deliberate
                // neutral framing, so a caller can report that the template was
                // not recognised instead of quietly using someone else's.
                for m in messages {
                    out.push_str(&format!("{}: {}\n", m.role, m.content));
                }
                if add_generation_prompt {
                    out.push_str("assistant:");
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_known_name_round_trips() {
        // `from_name` and `name` must agree, or `--chat-template granite`
        // reports a different template than it applied.
        for n in ChatFormat::known_names() {
            let f = ChatFormat::from_name(n)
                .unwrap_or_else(|| panic!("known_names lists {n:?} but from_name rejects it"));
            // glmedge and bailing are accepted aliases, so the canonical name
            // may differ -- but it must itself round-trip.
            let canon = f.name();
            assert_eq!(
                ChatFormat::from_name(canon),
                Some(f),
                "{n:?} -> {canon:?} does not round-trip"
            );
            assert!(f.is_known(), "{n:?} resolved to Generic");
        }
    }

    #[test]
    fn every_format_renders_something_with_the_role_in_it() {
        // A variant added to the enum but forgotten in `apply` would fall
        // through to a catch-all and silently render the wrong framing.
        for n in ChatFormat::known_names() {
            let f = ChatFormat::from_name(n).expect("known");
            let out = f.apply(&[Message::new("user", "PING")], "</s>", true);
            assert!(out.contains("PING"), "{n}: content dropped: {out:?}");
            assert!(!out.is_empty(), "{n}: rendered nothing");
        }
    }

    fn convo() -> Vec<Message> {
        vec![
            Message::new("system", "You are terse."),
            Message::new("user", "Hi."),
        ]
    }

    // The three below are the real templates from the containers on this
    // machine, trimmed to the part detection reads. Testing against invented
    // strings would prove only that the matcher matches itself.

    #[test]
    fn tinyllama_is_detected_as_zephyr() {
        let real = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n\
                    {{ '<|user|>\n' + message['content'] + eos_token }}\n\
                    {% elif message['role'] == 'system' %}\n\
                    {{ '<|system|>\n' + message['content'] + eos_token }}\n\
                    {% endif %}\n{% if loop.last and add_generation_prompt %}\n\
                    {{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}";
        let f = ChatFormat::detect(Some(real));
        assert_eq!(f, ChatFormat::Zephyr);
        assert_eq!(
            f.apply(&convo(), "</s>", true),
            "<|system|>\nYou are terse.</s>\n<|user|>\nHi.</s>\n<|assistant|>\n"
        );
    }

    #[test]
    fn llama32_is_detected_as_llama3() {
        let real = "{{- bos_token }}\n{%- if custom_tools is defined %}\n\
                    {%- set tools = custom_tools %}\n{%- endif %}\n\
                    <|start_header_id|>system<|end_header_id|>";
        let f = ChatFormat::detect(Some(real));
        assert_eq!(f, ChatFormat::Llama3);
        let got = f.apply(&convo(), "", true);
        assert!(got
            .starts_with("<|start_header_id|>system<|end_header_id|>\n\nYou are terse.<|eot_id|>"));
        assert!(got.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn qwen3_is_detected_as_chatml() {
        let real = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n\
                    {%- if messages[0].role == 'system' %}";
        let f = ChatFormat::detect(Some(real));
        assert_eq!(f, ChatFormat::ChatMl);
        assert_eq!(
            f.apply(&convo(), "", true),
            "<|im_start|>system\nYou are terse.<|im_end|>\n\
             <|im_start|>user\nHi.<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn phi3_wins_over_zephyr_because_both_use_the_user_marker() {
        // Ordering trap: Phi-3 and Zephyr share `<|user|>`, so a matcher that
        // checked Zephyr first would frame every Phi-3 model wrongly.
        let phi = "{% for message in messages %}{{'<|user|>' + message['content'] + '<|end|>'}}\
                   {% endfor %}{{ '<|assistant|>' }}";
        assert_eq!(ChatFormat::detect(Some(phi)), ChatFormat::Phi3);
    }

    #[test]
    fn llama2_and_mistral_are_separated_by_the_system_marker() {
        let llama2 =
            "{% if system %}[INST] <<SYS>>{{system}}<</SYS>> {{prompt}} [/INST]{% endif %}";
        let mistral = "{% for m in messages %}[INST] {{ m['content'] }} [/INST]{% endfor %}";
        assert_eq!(ChatFormat::detect(Some(llama2)), ChatFormat::Llama2);
        assert_eq!(ChatFormat::detect(Some(mistral)), ChatFormat::Mistral);

        // Mistral has no system slot; the instruction must be folded in, not
        // dropped, or the model silently loses it.
        let got = ChatFormat::Mistral.apply(&convo(), "</s>", true);
        assert!(got.contains("You are terse."), "system was lost: {got:?}");
    }

    #[test]
    fn gemma_renames_the_assistant_and_folds_the_system_turn() {
        // Gemma has no system role at all and calls the assistant "model".
        let got = ChatFormat::Gemma.apply(&convo(), "", true);
        assert!(!got.contains("system"), "gemma has no system role: {got:?}");
        assert!(got.contains("You are terse.\n\nHi."), "folded: {got:?}");
        assert!(got.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn an_unknown_template_reports_itself_unknown() {
        // The important part is `is_known` being false, so the caller can say
        // so rather than silently applying someone else's framing.
        let f = ChatFormat::detect(Some("{{ some_custom_thing }}"));
        assert_eq!(f, ChatFormat::Generic);
        assert!(!f.is_known());
        assert!(ChatFormat::detect(None) == ChatFormat::Generic);
        assert!(ChatFormat::ChatMl.is_known());
    }

    #[test]
    fn the_generation_prompt_opens_the_assistant_turn_and_leaves_it_open() {
        // Without it the model continues the user's message instead of
        // answering, which reads as the model ignoring the question.
        for f in [
            ChatFormat::ChatMl,
            ChatFormat::Llama3,
            ChatFormat::Zephyr,
            ChatFormat::Phi3,
            ChatFormat::Gemma,
            ChatFormat::Vicuna,
            ChatFormat::Alpaca,
            ChatFormat::Generic,
        ] {
            let with = f.apply(&convo(), "</s>", true);
            let without = f.apply(&convo(), "</s>", false);
            assert!(
                with.len() > without.len(),
                "{} must append a generation prompt",
                f.name()
            );
            assert!(
                with.starts_with(&without),
                "{} changed the history",
                f.name()
            );
        }
    }

    #[test]
    fn a_multi_turn_conversation_keeps_every_turn() {
        let msgs = vec![
            Message::new("user", "one"),
            Message::new("assistant", "two"),
            Message::new("user", "three"),
        ];
        for f in [ChatFormat::ChatMl, ChatFormat::Llama3, ChatFormat::Zephyr] {
            let got = f.apply(&msgs, "</s>", true);
            for needle in ["one", "two", "three"] {
                assert!(
                    got.contains(needle),
                    "{} dropped {needle}: {got:?}",
                    f.name()
                );
            }
        }
    }
}
