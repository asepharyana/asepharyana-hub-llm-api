//! Chat completion use cases.
//!
//! Orchestrates prompt building using the model's baked-in Jinja template
//! via the `minijinja` crate, sampler construction, and output parsing.

use std::collections::HashMap;

use minijinja::{Environment, Value};
use llama_cpp_2::sampling::LlamaSampler;

use crate::domain::entity::{ChatMessage, ChatRequest, ToolCall, ToolCallFunction};

/// Build a prompt from messages using the GGUF's Jinja chat template.
///
/// Renders the model's baked-in template via minijinja, passing the message
/// history, optional tool definitions, and generation-prompt switches.
pub fn build_prompt(
    model: &llama_cpp_2::model::LlamaModel,
    messages: &[ChatMessage],
    _tools: &Option<Vec<crate::domain::entity::ToolDef>>,
) -> Result<String, String> {
    // Load the GGUF's chat template (embedded at crate build time)
    let template_str = include_str!("templates/chat_template.jinja");

    let mut env = Environment::new();
    env.add_template("chat", template_str)
        .map_err(|e| format!("Template add error: {e}"))?;

    // Register tojson filter (safe: Rust serde_json defaults to ensure_ascii=false)
    env.add_filter("tojson", |value: &Value| -> String {
        serde_json::to_string(value).unwrap_or_default()
    });

    let tmpl = env.get_template("chat")
        .map_err(|e| format!("Template get error: {e}"))?;

    // Build messages as serde_json::Value for minijinja
    let mut msgs_val: Vec<Value> = Vec::new();
    for msg in messages {
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("role".into(), Value::from(msg.role.clone()));

        let content = msg.content.clone().unwrap_or_default();

        // For assistant messages, check if there are tool_calls
        if msg.role == "assistant" {
            if let Some(tcs) = &msg.tool_calls {
                // Serialise tool calls per the template's expected format
                let tcs_val: Vec<Value> = tcs.iter().map(|tc| {
                    let args: serde_json::Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                    Value::from_serialize(&serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": args,
                        }
                    }))
                }).collect();
                m.insert("tool_calls".into(), Value::from(tcs_val));
            }
        }

        // Handle tool role messages
        if msg.role == "tool" {
            // Wrap in tool_response as the template expects
            let wrapped = format!("<tool_response>\n{}\n</tool_response>", content);
            m.insert("content".into(), Value::from(wrapped));
        } else {
            m.insert("content".into(), Value::from(content));
        }

        msgs_val.push(Value::from(m));
    }

    // BOS token for sentencepiece / unigram models
    let bos_token: &str = "<s>";

    // Build context
    let mut ctx: HashMap<String, Value> = HashMap::new();
    ctx.insert("bos_token".into(), Value::from(bos_token));
    ctx.insert("messages".into(), Value::from(msgs_val));
    ctx.insert("add_generation_prompt".into(), Value::from(true));
    ctx.insert("enable_thinking".into(), Value::from(true));

    // Render
    let result = tmpl
        .render(&ctx)
        .map_err(|e| format!("Template render error: {e}"))?;

    Ok(result)
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
    let text = text.replace("<|im_end|>", "")
        .replace("<|im_start|>", "");

    // Separate reasoning (between <think>/</think>) from answer
    let text = text.trim();
    let (reasoning, answer) = if let Some(close_idx) = text.find("</think>") {
        let reasoning = text[..close_idx].trim()
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
            if let Some(param) =
                line.strip_prefix("<parameter=")
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
            args_map.insert(p, serde_json::Value::String(current_value.trim().to_string()));
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
