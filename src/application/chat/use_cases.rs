//! Chat completion use cases.
//!
//! Orchestrates prompt building, sampler construction, and output parsing.
//! These are pure functions with no framework dependencies.

use llama_cpp_2::sampling::LlamaSampler;

use crate::domain::entity::{ChatMessage, ChatRequest, ToolCall, ToolCallFunction, ToolDef};

/// Build a prompt string from conversation messages and optional tool definitions.
///
/// Uses ChatML format with `<|im_start|>` / `<|im_end|>` delimiters. Tool
/// definitions are injected into the first system message.
pub fn build_prompt(messages: &[ChatMessage], tools: &Option<Vec<ToolDef>>) -> String {
    let mut prompt = String::new();

    for (i, msg) in messages.iter().enumerate() {
        match msg.role.as_str() {
            "system" => {
                let mut content = msg.content.clone().unwrap_or_default();
                // Inject tools into the system message (first occurrence)
                if i == 0 {
                    if let Some(tools_list) = tools {
                        if !tools_list.is_empty() {
                            let mut tools_text = String::from(
                                "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>",
                            );
                            for tool in tools_list {
                                tools_text.push('\n');
                                tools_text.push_str(
                                    &serde_json::to_string(tool).unwrap_or_default(),
                                );
                            }
                            tools_text.push_str(
                                "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>",
                            );
                            content.push_str(&tools_text);
                        }
                    }
                }
                prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", content));
            }
            "user" => {
                let content = msg.content.as_deref().unwrap_or("");
                if msg.tool_call_id.is_some() || msg.name.is_some() {
                    prompt.push_str(&format!(
                        "<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n",
                        content
                    ));
                } else {
                    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", content));
                }
            }
            "assistant" => {
                let content = msg.content.as_deref().unwrap_or("");
                if let Some(tcs) = &msg.tool_calls {
                    let mut asst = format!("<|im_start|>assistant\n{}", content);
                    for tc in tcs {
                        let args: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                        asst.push_str(&format!(
                            "<tool_call>\n<function={}>\n",
                            tc.function.name
                        ));
                        if let Some(obj) = args.as_object() {
                            for (k, v) in obj {
                                let val = match v {
                                    serde_json::Value::String(s) => s.clone(),
                                    other => serde_json::to_string(other).unwrap_or_default(),
                                };
                                asst.push_str(&format!("<parameter={}>\n{}\n</parameter>\n", k, val));
                            }
                        }
                        asst.push_str("</function>\n</tool_call>");
                    }
                    asst.push_str("<|im_end|>\n");
                    prompt.push_str(&asst);
                } else {
                    prompt.push_str(&format!(
                        "<|im_start|>assistant\n{}<|im_end|>\n",
                        content
                    ));
                }
            }
            _ => {
                let content = msg.content.as_deref().unwrap_or("");
                prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", content));
            }
        }
    }

    // Generation prompt: thinking mode (MiniCPM5 native)
    prompt.push_str("<|im_start|>assistant\n/think\n\n");
    prompt
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

    // Repetition/frequency/presence penalties
    let repeat = repeat_penalty.unwrap_or(1.0);
    let freq = frequency_penalty.unwrap_or(0.0);
    let present = presence_penalty.unwrap_or(0.0);
    if (repeat - 1.0).abs() > 1e-6 || freq > 0.0 || present > 0.0 {
        samplers.push(LS::penalties(64, repeat, freq, present));
    }

    // top_k
    if let Some(k) = top_k {
        samplers.push(LS::top_k(k as i32));
    }

    // top_p
    if let Some(p) = top_p {
        samplers.push(LS::top_p(p, 1));
    }

    // min_p
    if let Some(p) = min_p {
        samplers.push(LS::min_p(p, 1));
    }

    // Temperature + final selector
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

/// Remove special tokens from generated text.
pub fn clean_text(text: &str) -> String {
    text.replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .replace("<|thought_begin|>", "")
        .replace("<|thought_end|>", "")
        .replace("<|tool_call|>", "")
        .replace("<|execute_start|>", "")
        .replace("<|execute_end|>", "")
        .replace("/think", "")
        .replace("/no_think", "")
        .trim()
        .to_string()
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
        // Save last param
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

    // Remove tool_call blocks from the text
    clean = clean.replace("<tool_call>", "").replace("</tool_call>", "");
    // XML-like tags are already fully parsed; remaining text is the content
    clean = clean.trim().to_string();
    // Strip remaining XML tags that aren't part of clean
    let cleaned = clean_text(&clean);

    (cleaned, tool_calls)
}
