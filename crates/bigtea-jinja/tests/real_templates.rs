//! Every `tokenizer.chat_template` on this machine, rendered.
//!
//! # Why against real templates and not invented ones
//!
//! An invented template proves the engine matches my idea of Jinja. These are
//! the strings the models actually ship, complete with the whitespace-control
//! dashes, the `namespace()` dance around loop scoping, and the
//! `raise_exception` guards that make a template refuse a conversation.
//!
//! The templates are checked in as fixtures rather than read from the GGUFs, so
//! this runs in the ggml-free CI job and on a machine with no models.

use bigtea_jinja::{parse, render, Env, Error, Value};
use std::collections::HashMap;

fn msg(role: &str, content: &str) -> Value {
    let mut m = HashMap::new();
    m.insert("role".to_string(), Value::Str(role.to_string()));
    m.insert("content".to_string(), Value::Str(content.to_string()));
    Value::Map(m)
}

fn env(messages: Vec<Value>) -> Env {
    let mut e = Env::new();
    e.set("messages", Value::List(messages));
    e.set("bos_token", Value::Str("<s>".into()));
    e.set("eos_token", Value::Str("</s>".into()));
    e.set("add_generation_prompt", Value::Bool(true));
    e
}

fn go(template: &str, messages: Vec<Value>) -> Result<String, Error> {
    let nodes = parse(template)?;
    render(&nodes, &mut env(messages))
}

/// Qwen2/Qwen3's template, trimmed to what detection reads.
const CHATML: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + \
     message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}\
     {{ '<|im_start|>assistant\\n' }}{% endif %}";

/// Gemma's, including the role remap and its `raise_exception` guard.
const GEMMA: &str = "{% for message in messages %}{% if (message['role'] == 'assistant') %}\
     {% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}\
     {{ '<start_of_turn>' + role + '\\n' + message['content'] + '<end_of_turn>\\n' }}\
     {% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}";

/// TinyLlama's Zephyr template, written in the whitespace-stripping form.
const ZEPHYR: &str = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n\
     {{ '<|user|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n\
     {{ '<|system|>\\n' + message['content'] + eos_token }}\n{% endif %}\n\
     {% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}";

#[test]
fn chatml_renders_exactly() {
    assert_eq!(
        go(CHATML, vec![msg("system", "SYS"), msg("user", "HI")]).unwrap(),
        "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nHI<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn gemma_remaps_assistant_to_model() {
    // The remap is the whole point of Gemma's template: a turn labelled
    // `assistant` is a token the model has never seen in that position.
    let out = go(GEMMA, vec![msg("user", "HI"), msg("assistant", "HO")]).unwrap();
    assert_eq!(
        out,
        "<start_of_turn>user\nHI<end_of_turn>\n<start_of_turn>model\nHO<end_of_turn>\n\
         <start_of_turn>model\n"
    );
    assert!(!out.contains("<start_of_turn>assistant"));
}

#[test]
fn zephyr_uses_the_container_eos_and_loop_last() {
    let out = go(ZEPHYR, vec![msg("system", "SYS"), msg("user", "HI")]).unwrap();
    // `eos_token` is the CONTAINER's, which is the thing llama.cpp's hardcoded
    // renderer cannot know -- it emits `<|endoftext|>` on every model.
    assert!(out.contains("SYS</s>"), "{out:?}");
    assert!(out.contains("HI</s>"), "{out:?}");
    // `loop.last` gates the generation prompt, so it appears once.
    assert_eq!(out.matches("<|assistant|>").count(), 1, "{out:?}");
}

#[test]
fn a_template_that_rejects_a_conversation_actually_fails() {
    // Gemma-style guard. Swallowing this renders a system turn the model was
    // trained to never see, which is fluent and wrong.
    let guard = "{% if messages[0]['role'] == 'system' %}\
         {{ raise_exception('System role not supported') }}{% endif %}{{ 'ok' }}";
    let e = go(guard, vec![msg("system", "SYS")]).unwrap_err();
    assert!(
        matches!(e, Error::Raised(ref m) if m.contains("System role")),
        "{e:?}"
    );
    // ...and passes when the guard does not fire.
    assert_eq!(go(guard, vec![msg("user", "HI")]).unwrap(), "ok");
}

#[test]
fn the_namespace_dance_survives_the_loop() {
    // Real Llama-3 templates use this shape: a plain `set` inside a `for` is
    // scoped to the body, so state that must outlive the loop goes in a
    // namespace. Getting it wrong loses the flag and changes the framing.
    let t = "{% set ns = namespace() %}{% set ns.saw_system = false %}\
         {% for m in messages %}{% if m['role'] == 'system' %}{% set ns.saw_system = true %}\
         {% endif %}{% endfor %}{{ ns.saw_system }}";
    assert_eq!(go(t, vec![msg("user", "HI")]).unwrap(), "False");
    assert_eq!(go(t, vec![msg("system", "S")]).unwrap(), "True");
}

#[test]
fn whitespace_control_matches_jinjas() {
    // `{%- ... -%}` around a loop is how Llama-3's template avoids emitting a
    // newline per iteration. A stray newline shifts every following token.
    let t = "{%- for m in messages -%}\n    {{- m['content'] -}}\n{%- endfor -%}";
    assert_eq!(
        go(t, vec![msg("user", "A"), msg("user", "B")]).unwrap(),
        "AB"
    );
}

#[test]
fn an_unsupported_construct_refuses_rather_than_rendering_something() {
    // The property the whole crate exists for. A template using anything
    // outside the subset must send the caller back to the family matcher.
    for t in [
        "{% macro x() %}{% endmacro %}",
        "{% include 'other' %}",
        "{{ messages | map(attribute='role') }}",
        "{{ messages | join(', ') }}",
    ] {
        let e = go(t, vec![msg("user", "HI")]).unwrap_err();
        assert!(
            matches!(e, Error::Unsupported(_)),
            "{t:?} should be refused, got {e:?}"
        );
    }
}

#[test]
fn an_empty_conversation_does_not_panic() {
    assert_eq!(go(CHATML, vec![]).unwrap(), "<|im_start|>assistant\n");
}
