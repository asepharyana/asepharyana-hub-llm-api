//! Chat completion use cases.
//!
//! Orchestrates prompt building using the model's baked-in Jinja template
//! via the `minijinja` crate, sampler construction, and output parsing.

use std::collections::HashMap;

use llama_cpp_2::sampling::LlamaSampler;
use minijinja::{Environment, Value};

use crate::domain::entity::{ChatMessage, ChatRequest, ToolCall, ToolCallFunction};

/// Build a prompt from messages using the project's Jinja chat template.
///
/// Renders the template (see `templates/chat_template.jinja`) via minijinja,
/// passing the message history, optional tool definitions, and the
/// generation-prompt switches. The template owns the `<s>` BOS token, so
/// tokenization must NOT add another one (`AddBos::Never`).
pub fn build_prompt(
    messages: &[ChatMessage],
    tools: &Option<Vec<crate::domain::entity::ToolDef>>,
) -> Result<String, String> {
    let template_str = include_str!("templates/chat_template.jinja");

    let mut env = Environment::new();
    env.add_template("chat", template_str)
        .map_err(|e| format!("Template add error: {e}"))?;

    // Register tojson filter (safe: Rust serde_json defaults to ensure_ascii=false)
    env.add_filter("tojson", |value: &Value| -> String {
        serde_json::to_string(value).unwrap_or_default()
    });

    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("Template get error: {e}"))?;

    // Build messages as serde_json::Value for minijinja.
    // Assistant tool calls from history are embedded directly into the content
    // as XML (in the exact format the model is told to emit) — the template
    // cannot accumulate `set` variables across a loop, so this is built here.
    let mut msgs_val: Vec<Value> = Vec::new();
    for msg in messages {
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("role".into(), Value::from(msg.role.clone()));

        let mut content = msg.content.clone().unwrap_or_default();

        if msg.role == "assistant" {
            if let Some(tcs) = &msg.tool_calls {
                for tc in tcs {
                    if tc.call_type == "function" {
                        content
                            .push_str(&format!("\n<tool_call>\n<function={}>\n", tc.function.name));
                        let args: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                        if let Some(obj) = args.as_object() {
                            for (k, v) in obj {
                                // Strings stay raw (no JSON quotes) so they round-trip
                                // through parse_tool_calls unchanged.
                                let rendered = match v {
                                    serde_json::Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                };
                                content
                                    .push_str(&format!("<parameter={k}>{rendered}</parameter>\n"));
                            }
                        }
                        content.push_str("</function>\n</tool_call>");
                    }
                }
            }
        }

        // NOTE: the template wraps tool-role content in <tool_response>; do not
        // wrap here or it would be double-wrapped.
        m.insert("content".into(), Value::from(content));

        msgs_val.push(Value::from(m));
    }

    // BOS token for sentencepiece / unigram models (template-owned).
    let bos_token: &str = "<s>";

    // Build context
    let mut ctx: HashMap<String, Value> = HashMap::new();
    ctx.insert("bos_token".into(), Value::from(bos_token));
    ctx.insert("messages".into(), Value::from(msgs_val));
    ctx.insert("add_generation_prompt".into(), Value::from(true));
    ctx.insert("enable_thinking".into(), Value::from(true));

    // Tool definitions (optional) — previously dead, now actually rendered.
    if let Some(tools) = tools {
        if !tools.is_empty() {
            let tools_val: Vec<Value> = tools.iter().map(Value::from_serialize).collect();
            ctx.insert("tools".into(), Value::from(tools_val));
        }
    }

    // Render
    let result = tmpl
        .render(&ctx)
        .map_err(|e| format!("Template render error: {e}"))?;

    Ok(result)
}

/// Validate that the requested model matches the single served model.
pub fn validate_model(model: &str) -> Result<(), String> {
    if model != crate::config::MODEL_ID {
        return Err(format!(
            "Unknown model '{model}'. Available: {}",
            crate::config::MODEL_ID
        ));
    }
    Ok(())
}

/// Parameters for building a [`LlamaSampler`] chain.
pub struct SamplerParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u32>,
}

impl SamplerParams {
    pub fn from_request(req: &ChatRequest) -> Self {
        Self {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            repeat_penalty: req.repeat_penalty,
            frequency_penalty: req.frequency_penalty,
            presence_penalty: req.presence_penalty,
            seed: req.seed,
        }
    }
}

/// Build a [`LlamaSampler`] chain from [`SamplerParams`].
pub fn build_sampler(params: &SamplerParams) -> LlamaSampler {
    use llama_cpp_2::sampling::LlamaSampler as LS;

    let temperature = params.temperature;
    let top_p = params.top_p;
    let top_k = params.top_k;
    let min_p = params.min_p;
    let repeat_penalty = params.repeat_penalty;
    let frequency_penalty = params.frequency_penalty;
    let presence_penalty = params.presence_penalty;
    let seed = params.seed;
    let mut samplers: Vec<LlamaSampler> = Vec::new();

    let repeat = repeat_penalty.unwrap_or(1.0);
    let freq = frequency_penalty.unwrap_or(0.0);
    let present = presence_penalty.unwrap_or(0.0);
    if (repeat - 1.0).abs() > 1e-6 || freq > 0.0 || present > 0.0 {
        samplers.push(LS::penalties(64, repeat, freq, present));
    }

    if let Some(k) = top_k {
        samplers.push(LS::top_k(k as i32));
    }

    if let Some(p) = top_p {
        samplers.push(LS::top_p(p, 1));
    }

    if let Some(p) = min_p {
        samplers.push(LS::min_p(p, 1));
    }

    let temp = temperature.unwrap_or(0.0);
    if temp <= 0.0 {
        samplers.push(LS::greedy());
    } else {
        if (temp - 1.0).abs() > 1e-6 {
            samplers.push(LS::temp(temp));
        }
        let s = seed.unwrap_or(0);
        samplers.push(LS::dist(s));
    }

    LlamaSampler::chain_simple(samplers)
}

// ═══════════════════════════════════════════════════════════════
//  TEXT PROCESSING
// ═══════════════════════════════════════════════════════════════

/// Remove special tokens from generated text and separate reasoning.
///
/// For thinking models, returns (reasoning, cleaned_answer).
pub fn clean_text(text: &str) -> (String, String) {
    let text = text.replace("<|im_end|>", "").replace("<|im_start|>", "");

    // Separate reasoning (between <think>/</think>) from answer
    let text = text.trim();
    let (reasoning, answer) = if let Some(close_idx) = text.find("</think>") {
        let reasoning = text[..close_idx]
            .trim()
            .trim_start_matches("<think>")
            .trim()
            .to_string();
        let answer = text[close_idx + 8..].trim().to_string();
        (reasoning, answer)
    } else if text.contains("<think>") {
        // Still thinking — everything is reasoning
        let reasoning = text.trim_start_matches("<think>").trim().to_string();
        (reasoning, String::new())
    } else {
        (String::new(), text.to_string())
    };

    let answer = answer.replace("<think>", "").replace("</think>", "");
    (reasoning, answer.trim().to_string())
}

/// Parse tool calls from generated text in the format:
///
/// ```xml
/// <tool_call>
/// <function=name>
/// <parameter=key>value</parameter>
/// </function>
/// </tool_call>
/// ```
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut clean = text.to_string();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    let mut idx = 0;
    loop {
        let start_tag = "<tool_call>";
        let end_tag = "</tool_call>";

        let start = match clean[idx..].find(start_tag) {
            Some(s) => idx + s,
            None => break,
        };

        let end = match clean[start..].find(end_tag) {
            Some(e) => start + e + end_tag.len(),
            None => break,
        };

        let block = &clean[start + start_tag.len()..end - end_tag.len()];
        let trimmed = block.trim();

        // Parse function name
        let func_name = trimmed
            .lines()
            .next()
            .and_then(|l| {
                let l = l.trim();
                l.strip_prefix("<function=")
                    .and_then(|s| s.strip_suffix('>'))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_default();

        // Parse parameters
        let mut args_map = serde_json::Map::new();
        let lines = trimmed.lines();
        let mut current_param: Option<String> = None;
        let mut current_value = String::new();
        let mut in_param = false;

        for line in lines {
            let line = line.trim();
            if let Some(param) = line
                .strip_prefix("<parameter=")
                .and_then(|s| s.strip_suffix('>'))
            {
                if let Some(p) = current_param.take() {
                    args_map.insert(
                        p,
                        serde_json::Value::String(current_value.trim().to_string()),
                    );
                    current_value = String::new();
                }
                current_param = Some(param.to_string());
                in_param = true;
            } else if line == "</parameter>" {
                in_param = false;
            } else if in_param {
                if !current_value.is_empty() {
                    current_value.push('\n');
                }
                current_value.push_str(line);
            } else if line.starts_with("<function=") || line.starts_with("</function>") {
                continue;
            }
        }
        if let Some(p) = current_param.take() {
            args_map.insert(
                p,
                serde_json::Value::String(current_value.trim().to_string()),
            );
        }

        let args_json = serde_json::Value::Object(args_map).to_string();

        tool_calls.push(ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4().to_string().replace('-', "")),
            call_type: "function".into(),
            function: ToolCallFunction {
                name: func_name,
                arguments: args_json,
            },
        });

        idx = end;
    }

    clean = clean.replace("<tool_call>", "").replace("</tool_call>", "");
    clean = clean.trim().to_string();
    let (_reasoning, cleaned) = clean_text(&clean);

    (cleaned, tool_calls)
}

// ═══════════════════════════════════════════════════════════════
//  STREAMING CHUNK SPLITTING
// ═══════════════════════════════════════════════════════════════

/// Remove special tokens and tool-call XML markup from a text fragment.
///
/// Handles both fixed tags (`<|im_end|>`, `<think>`, `<tool_call>`, …) and the
/// attribute-bearing openers used by this model's tool format (`<function=…>`,
/// `<parameter=…>`), even when a tag straddles a token boundary.
pub fn strip_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while !rest.is_empty() {
        let Some(idx) = rest.find('<') else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..idx]);
        let tail = &rest[idx..];

        // Fixed tags (no attribute content).
        let fixed = [
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "</function>",
            "</parameter>",
        ];
        if let Some(tag) = fixed.iter().find(|t| tail.starts_with(**t)) {
            rest = &tail[tag.len()..];
            continue;
        }

        // Attribute-bearing openers: <function=…> / <parameter=…>.
        if let Some(attr) = tail
            .strip_prefix("<function=")
            .or_else(|| tail.strip_prefix("<parameter="))
        {
            if let Some(end) = attr.find('>') {
                rest = &attr[end + 1..];
                continue;
            }
        }

        // Not a known tag — keep this char and advance one UTF-8 char.
        let ch = tail.chars().next().expect("non-empty tail");
        out.push(ch);
        rest = &tail[ch.len_utf8()..];
    }

    out
}

/// Compute `(reasoning, content)` deltas for an incremental streamed fragment.
///
/// * `new_text` — the not-yet-emitted fragment (`text_buf[sent_len..]`).
/// * `sent_len` — byte offset in the full buffer where `new_text` begins.
/// * `content_start` — byte offset in the full buffer where the content phase
///   begins (immediately after the first `</think>`); `None` while still
///   thinking.
///
/// The boundary is a position in the *full* buffer, not a string search in the
/// fragment — this stays correct even when `</think>` is split across tokens.
/// Whitespace inside a fragment is preserved; only the edges around the
/// boundary are trimmed. A stray second `</think>` is stripped, not re-split.
pub fn split_stream_chunk(
    new_text: &str,
    sent_len: usize,
    content_start: Option<usize>,
) -> (Option<String>, String) {
    match content_start {
        // Still thinking — everything is reasoning.
        None => {
            let cleaned = strip_markup(new_text);
            let reasoning = if cleaned.is_empty() {
                None
            } else {
                Some(cleaned)
            };
            (reasoning, String::new())
        }
        // Boundary already emitted — everything is content.
        Some(cs) if cs <= sent_len => (None, strip_markup(new_text)),
        // Boundary falls inside this fragment (or beyond it, defensively).
        Some(cs) => {
            let rel = (cs - sent_len).min(new_text.len());
            let (before, after) = new_text.split_at(rel);
            let mut before = strip_markup(before);
            let mut after = strip_markup(after);

            while before.ends_with(['\n', ' ', '\t']) {
                before.pop();
            }
            while after.starts_with(['\n', ' ', '\t']) {
                after.remove(0);
            }

            let reasoning = if before.is_empty() {
                None
            } else {
                Some(before)
            };
            (reasoning, after)
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entity::{
        ChatMessage, ToolCallFunction, ToolCallResponse, ToolDef, ToolFunction,
    };

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn tool_def() -> ToolDef {
        ToolDef {
            tool_type: "function".into(),
            function: ToolFunction {
                name: "get_weather".into(),
                description: "Get weather for a city".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                }),
            },
        }
    }

    // ── split_stream_chunk ──

    #[test]
    fn split_chunk_before_think_is_reasoning() {
        let (reasoning, content) = split_stream_chunk("Hello ", 0, None);
        assert_eq!(reasoning.as_deref(), Some("Hello "));
        assert_eq!(content, "");
    }

    #[test]
    fn split_chunk_preserves_internal_spaces() {
        // Regression: trimming every fragment used to eat inter-word spaces.
        let (r1, _) = split_stream_chunk("Hello", 0, None);
        let (r2, _) = split_stream_chunk(" world", 5, None);
        assert_eq!(format!("{}{}", r1.unwrap(), r2.unwrap()), "Hello world");
    }

    #[test]
    fn split_chunk_boundary_trims_only_edges() {
        // text_buf = "question\n</think>\n\nAnswer "; boundary right after the
        // tag at byte 17.
        let (reasoning, content) = split_stream_chunk("question\n</think>\n\nAnswer ", 0, Some(17));
        assert_eq!(reasoning.as_deref(), Some("question"));
        assert_eq!(content, "Answer ");
    }

    #[test]
    fn split_chunk_after_think_is_content() {
        let (reasoning, content) = split_stream_chunk(" answer", 0, Some(0));
        assert_eq!(reasoning, None);
        assert_eq!(content, " answer");
    }

    #[test]
    fn split_chunk_stray_think_tag_is_stripped_not_split() {
        // A second </think> (already past the boundary) must not restart
        // reasoning classification.
        let (reasoning, content) = split_stream_chunk("...</think>more", 0, Some(0));
        assert_eq!(reasoning, None);
        assert_eq!(content, "...more");
    }

    #[test]
    fn split_chunk_boundary_straddling_tokens() {
        // `</think>` split as "</think" + ">" across two fragments: the boundary
        // is detected on the full buffer, so the answer still becomes content.
        let (r1, _) = split_stream_chunk("reasoning...</think", 0, None);
        assert!(r1.is_some());
        let (r2, c2) = split_stream_chunk("\n\n2 + 2 = 4.", 18, Some(18));
        assert_eq!(r2, None);
        assert_eq!(c2, "\n\n2 + 2 = 4.");
    }

    #[test]
    fn split_chunk_strips_special_and_markup() {
        // boundary at byte 30 (right after "</think>").
        let (reasoning, content) = split_stream_chunk(
            "<|im_end|><think>Hello</think>\n<tool_call><function=get_weather>",
            0,
            Some(30),
        );
        assert_eq!(reasoning.as_deref(), Some("Hello"));
        assert_eq!(content, "");
    }

    // ── clean_text ──

    #[test]
    fn clean_text_splits_reasoning_and_answer() {
        let (reasoning, answer) =
            clean_text("<think>Let me think\nabout it</think>\nThe answer is 42.");
        assert_eq!(reasoning, "Let me think\nabout it");
        assert_eq!(answer, "The answer is 42.");
    }

    #[test]
    fn clean_text_no_think_returns_answer() {
        let (reasoning, answer) = clean_text("Just an answer");
        assert_eq!(reasoning, "");
        assert_eq!(answer, "Just an answer");
    }

    // ── parse_tool_calls ──

    #[test]
    fn parse_single_tool_call() {
        let text = "I'll look that up.\n<tool_call>\n<function=get_weather>\n<parameter=city>Jakarta</parameter>\n</function>\n</tool_call>";
        let (cleaned, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Jakarta"));
        assert!(!cleaned.contains("<tool_call>"));
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let text = "<tool_call>\n<function=a>\n<parameter=x>1</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=y>2</parameter>\n</function>\n</tool_call>";
        let (_cleaned, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
    }

    #[test]
    fn parse_malformed_tool_call_returns_empty() {
        let (cleaned, calls) = parse_tool_calls("no calls here");
        assert!(calls.is_empty());
        assert_eq!(cleaned, "no calls here");
    }

    // ── validate_model ──

    #[test]
    fn validate_model_accepts_served_id() {
        assert!(validate_model(crate::config::MODEL_ID).is_ok());
    }

    #[test]
    fn validate_model_rejects_unknown_id() {
        assert!(validate_model("minicpm-v-4.6").is_err());
    }

    // ── build_sampler (smoke — no model required) ──

    #[test]
    fn build_sampler_constructs_for_common_params() {
        let params = SamplerParams {
            temperature: Some(0.8),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            repeat_penalty: Some(1.1),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            seed: Some(42),
        };
        let _ = build_sampler(&params);

        let greedy = SamplerParams {
            temperature: Some(0.0),
            ..params
        };
        let _ = build_sampler(&greedy);
    }

    // ── build_prompt ──

    #[test]
    fn build_prompt_renders_messages() {
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hi!")];
        let prompt = build_prompt(&messages, &None).unwrap();
        assert!(prompt.starts_with("<s>"));
        assert!(prompt.contains("<|im_start|>system\nYou are helpful."));
        assert!(prompt.contains("<|im_start|>user\nHi!"));
        assert!(prompt.contains("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn build_prompt_includes_tool_definitions() {
        let messages = vec![msg("user", "What's the weather?")];
        let prompt = build_prompt(&messages, &Some(vec![tool_def()])).unwrap();
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("Tool usage guidelines"));
    }

    #[test]
    fn build_prompt_wraps_tool_response_once() {
        let messages = vec![msg("user", "Weather?"), msg("tool", "Sunny")];
        let prompt = build_prompt(&messages, &None).unwrap();
        assert_eq!(prompt.matches("<tool_response>").count(), 1);
        assert_eq!(prompt.matches("</tool_response>").count(), 1);
    }

    #[test]
    fn build_prompt_renders_assistant_tool_calls_history() {
        let assistant = ChatMessage {
            role: "assistant".into(),
            content: Some("".into()),
            tool_calls: Some(vec![ToolCallResponse {
                id: "call_1".into(),
                call_type: "function".into(),
                function: ToolCallFunction {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Jakarta"}"#.into(),
                },
            }]),
            tool_call_id: None,
            name: None,
        };
        let prompt = build_prompt(&[assistant], &None).unwrap();
        assert!(prompt.contains("<function=get_weather>"));
        assert!(prompt.contains("<parameter=city>Jakarta</parameter>"));
    }
}
