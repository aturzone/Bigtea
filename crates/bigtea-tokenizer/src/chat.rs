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
        if t.contains("<|im_start|>") {
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
            ChatFormat::Generic => "generic",
        }
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
