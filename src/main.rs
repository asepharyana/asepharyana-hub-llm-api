use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
    TokenToStringError,
};
use serde::{Deserialize, Serialize};
use std::{num::NonZeroU32, sync::Arc};
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use tracing::info;

// ── Thread-safe wrapper ──
struct CtxInner {
    context: llama_cpp_2::context::LlamaContext<'static>,
}

unsafe impl Send for CtxInner {}
unsafe impl Sync for CtxInner {}

// LlamaSampler raw pointer is safe to Send (llama.cpp is thread-safe)
struct SendSampler(LlamaSampler);
unsafe impl Send for SendSampler {}
unsafe impl Sync for SendSampler {}

impl std::ops::Deref for SendSampler {
    type Target = LlamaSampler;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl std::ops::DerefMut for SendSampler {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl CtxInner {
    fn clear(&mut self) {
        self.context.clear_kv_cache();
    }

    fn prefill(&mut self, tokens: &[LlamaToken]) -> Result<(), String> {
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, &token) in tokens.iter().enumerate() {
            batch
                .add(token, i as i32, &[0], i == tokens.len() - 1)
                .map_err(|e| e.to_string())?;
        }
        self.context.decode(&mut batch).map_err(|e| e.to_string())
    }

    fn sample(&mut self, sampler: &mut LlamaSampler) -> LlamaToken {
        sampler.sample(&self.context, -1)
    }

    fn decode(&mut self, token: LlamaToken, pos: i32) -> Result<(), String> {
        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(token, pos, &[0], true)
            .map_err(|e| e.to_string())?;
        self.context.decode(&mut batch).map_err(|e| e.to_string())
    }
}

struct AppState {
    model: LlamaModel,
    ctx: Mutex<CtxInner>,
}

// ── OpenAI Chat Request ──
#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    stop: Option<Vec<String>>,
    #[serde(default)]
    tools: Option<Vec<ToolDef>>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: Option<String>, // null for tool calls
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallResponse>>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize, Serialize, Clone)]
struct ToolDef {
    #[serde(rename = "type")]
    tool_type: String,
    function: ToolFunction,
}

#[derive(Deserialize, Serialize, Clone)]
struct ToolFunction {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    parameters: serde_json::Value,
}

// ── OpenAI Chat Response ──
#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// ── Tool Call Types ──
#[derive(Serialize, Deserialize, Clone)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone)]
struct ToolCallFunction {
    name: String,
    arguments: String, // JSON string
}

// For parsing tool calls from history
#[derive(Deserialize, Clone)]
struct ToolCallResponse {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ToolCallFunction,
}

// ── SSE Chunk Types ──
#[derive(Serialize)]
struct SseChunk {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<SseChoice>,
}

#[derive(Serialize)]
struct SseChoice {
    index: u32,
    delta: SseDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct SseDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model: String,
}

const DEFAULT_MODEL_PATH: &str = "/models/MiniCPM-V-4.6-Q4_K_M.gguf";
const MODEL_ID: &str = "minicpm-v-4.6";

// ═══════════════════════════════════════════
//  SAMPLER BUILDER
// ═══════════════════════════════════════════

fn build_sampler(req: &ChatRequest) -> LlamaSampler {
    let mut samplers: Vec<LlamaSampler> = Vec::new();

    // Repetition/frequency/presence penalties
    let repeat = req.repeat_penalty.unwrap_or(1.0);
    let freq = req.frequency_penalty.unwrap_or(0.0);
    let present = req.presence_penalty.unwrap_or(0.0);
    if (repeat - 1.0).abs() > 1e-6 || freq > 0.0 || present > 0.0 {
        samplers.push(LlamaSampler::penalties(64, repeat, freq, present));
    }

    // top_k
    if let Some(k) = req.top_k {
        samplers.push(LlamaSampler::top_k(k as i32));
    }

    // top_p
    if let Some(p) = req.top_p {
        samplers.push(LlamaSampler::top_p(p, 1));
    }

    // min_p
    if let Some(p) = req.min_p {
        samplers.push(LlamaSampler::min_p(p, 1));
    }

    // Temperature + final selector
    let temp = req.temperature.unwrap_or(0.0);
    if temp <= 0.0 {
        samplers.push(LlamaSampler::greedy());
    } else {
        if (temp - 1.0).abs() > 1e-6 {
            samplers.push(LlamaSampler::temp(temp));
        }
        let seed = req.seed.unwrap_or(0);
        samplers.push(LlamaSampler::dist(seed));
    }

    LlamaSampler::chain_simple(samplers)
}

fn build_sampler_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u32>,
) -> LlamaSampler {
    let mut samplers: Vec<LlamaSampler> = Vec::new();

    let repeat = repeat_penalty.unwrap_or(1.0);
    let freq = frequency_penalty.unwrap_or(0.0);
    let present = presence_penalty.unwrap_or(0.0);
    if (repeat - 1.0).abs() > 1e-6 || freq > 0.0 || present > 0.0 {
        samplers.push(LlamaSampler::penalties(64, repeat, freq, present));
    }

    if let Some(k) = top_k {
        samplers.push(LlamaSampler::top_k(k as i32));
    }

    if let Some(p) = top_p {
        samplers.push(LlamaSampler::top_p(p, 1));
    }

    if let Some(p) = min_p {
        samplers.push(LlamaSampler::min_p(p, 1));
    }

    let temp = temperature.unwrap_or(0.0);
    if temp <= 0.0 {
        samplers.push(LlamaSampler::greedy());
    } else {
        if (temp - 1.0).abs() > 1e-6 {
            samplers.push(LlamaSampler::temp(temp));
        }
        let s = seed.unwrap_or(0);
        samplers.push(LlamaSampler::dist(s));
    }

    LlamaSampler::chain_simple(samplers)
}

// ═══════════════════════════════════════════
//  PROMPT / TOOL BUILDERS
// ═══════════════════════════════════════════

fn build_prompt(messages: &[ChatMessage], tools: &Option<Vec<ToolDef>>) -> String {
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
                // Handle tool results
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
                    // Assistant message with tool calls
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

    // Generation prompt: non-thinking mode
    prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    prompt
}

// ═══════════════════════════════════════════
//  TOOL CALL PARSER
// ═══════════════════════════════════════════

fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut clean = text.to_string();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    // Find all <tool_call>...</tool_call> blocks
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
            if let Some(param) = line.strip_prefix("<parameter=").and_then(|s| s.strip_suffix('>'))
            {
                // Save previous param
                if let Some(p) = current_param.take() {
                    args_map.insert(p, serde_json::Value::String(current_value.trim().to_string()));
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
    // Also remove leftover function/parameter XML
    clean = clean.replace(r#"<function=\S+>"#, "");
    clean = clean.replace("</function>", "");
    clean = clean.replace(r#"<parameter=\S+>"#, "");
    clean = clean.replace("</parameter>", "");

    (clean_text(&clean), tool_calls)
}

// ═══════════════════════════════════════════
//  TOKEN DECODING
// ═══════════════════════════════════════════

fn decode_token_piece(model: &LlamaModel, token: LlamaToken) -> String {
    let bytes = match model.token_to_piece_bytes(token, 32, true, None) {
        Ok(b) => b,
        Err(TokenToStringError::InsufficientBufferSpace(neg)) => {
            let size = (-neg).max(0).try_into().unwrap_or(256);
            model
                .token_to_piece_bytes(token, size, true, None)
                .unwrap_or_default()
        }
        _ => return String::new(),
    };
    String::from_utf8(bytes).unwrap_or_default()
}

fn clean_text(text: &str) -> String {
    text.replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .replace("<think>", "")
        .replace("</think>", "")
        .trim()
        .to_string()
}

fn decode_tokens(model: &LlamaModel, tokens: &[LlamaToken]) -> String {
    let mut out = String::with_capacity(tokens.len() * 4);
    for &token in tokens {
        out.push_str(&decode_token_piece(model, token));
    }
    clean_text(&out)
}

// ═══════════════════════════════════════════
//  AUTH
// ═══════════════════════════════════════════

fn check_auth(headers: &HeaderMap) -> Result<(), (StatusCode, String)> {
    let api_key = std::env::var("API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        return Ok(());
    }
    let header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let expected = format!("Bearer {api_key}");
    if header == expected || header == api_key {
        return Ok(());
    }
    Err((
        StatusCode::UNAUTHORIZED,
        "{\"error\":\"unauthorized\",\"message\":\"Invalid API key\"}".into(),
    ))
}

// ═══════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    info!("Initializing backend...");
    let backend = LlamaBackend::init().expect("Backend init failed");

    info!("Loading model...");
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    info!("  Model: {model_path}");
    let model = LlamaModel::load_from_file(&backend, &model_path, &LlamaModelParams::default())
        .expect("Failed to load model");
    info!("  Vocab: {}", model.n_vocab());
    info!("  Params: {}", model.n_params());
    info!("  Layers: {}", model.n_layer());

    info!("Creating context...");
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_n_batch(512)
        .with_n_threads(4)
        .with_n_threads_batch(4);

    let context = model
        .new_context(&backend, ctx_params)
        .expect("Failed to create context");
    let context: llama_cpp_2::context::LlamaContext<'static> =
        unsafe { std::mem::transmute(context) };

    let state = Arc::new(AppState {
        model,
        ctx: Mutex::new(CtxInner { context }),
    });

    info!("Server ready on :8080");

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .expect("Failed to bind");

    axum::serve(listener, app).await.expect("Server failed");
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        model: format!("{MODEL_ID}-q4_k_m"),
    })
}

async fn list_models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".into(),
        data: vec![ModelInfo {
            id: MODEL_ID.into(),
            object: "model".into(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "asepharyana".into(),
        }],
    })
}

// ═══════════════════════════════════════════
//  GENERATE TOKENS (shared by stream & non-stream)
// ═══════════════════════════════════════════

fn generate_tokens(
    state: &AppState,
    inner: &mut CtxInner,
    input_tokens: &[LlamaToken],
    sampler: &mut LlamaSampler,
    max_tokens: u32,
    stop: &[String],
) -> (Vec<LlamaToken>, String, Vec<ToolCall>) {
    let mut output: Vec<LlamaToken> = Vec::new();
    let mut text_buf = String::new();
    let mut stop_now = false;

    let mut current = inner.sample(sampler);

    for _ in 0..max_tokens {
        if state.model.is_eog_token(current) {
            break;
        }

        let pos = input_tokens.len() as i32 + output.len() as i32;
        output.push(current);

        if let Err(e) = inner.decode(current, pos) {
            info!("  Decode error: {e}");
            break;
        }

        // Decode this token for stop checking
        let piece = decode_token_piece(&state.model, current);
        text_buf.push_str(&piece);

        // Check stop sequences
        for s in stop {
            if text_buf.contains(s) {
                stop_now = true;
                break;
            }
        }
        if stop_now {
            break;
        }

        // Check for tool_call start
        if text_buf.contains("<tool_call>") {
            // Keep generating until tool_call block is closed
            let close_tag = "</tool_call>";
            let close_count = text_buf.matches(close_tag).count();
            let open_count = text_buf.matches("<tool_call>").count();
            if open_count > 0 && close_count >= open_count {
                // All tool call blocks are closed
                break;
            }
        }

        current = inner.sample(sampler);
    }

    let (clean, tool_calls) = parse_tool_calls(&clean_text(&text_buf));
    (output, clean, tool_calls)
}

// ═══════════════════════════════════════════
//  CHAT COMPLETIONS HANDLER
// ═══════════════════════════════════════════

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    check_auth(&headers)?;

    let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let max_tokens = req.max_tokens.unwrap_or(256).min(1024);
    let stop = req.stop.clone().unwrap_or_default();
    let prompt = build_prompt(&req.messages, &req.tools);
    let has_tools = req
        .tools
        .as_ref()
        .is_some_and(|t| !t.is_empty());

    // Tokenize
    let input_tokens = state
        .model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let prompt_tokens = input_tokens.len() as u32;
    info!(
        "  Chat: {} prompt tokens, max_tokens={}, tools={}",
        prompt_tokens,
        max_tokens,
        has_tools
    );

    if req.stream.unwrap_or(false) {
        // ── STREAMING ──
        let state = state.clone();
        let model_name = req.model.clone();
        let stop_clone = stop.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);

        // Extract sampling params for the spawned task
        let temp = req.temperature;
        let top_p = req.top_p;
        let top_k = req.top_k;
        let min_p = req.min_p;
        let freq_penalty = req.frequency_penalty;
        let pres_penalty = req.presence_penalty;
        let rep_penalty = req.repeat_penalty;
        let seed = req.seed;
        let _max_tokens_s = max_tokens;

        tokio::spawn(async move {
            // Role chunk
            let role_chunk = serde_json::to_string(&SseChunk {
                id: chat_id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![SseChoice {
                    index: 0,
                    delta: SseDelta {
                        role: Some("assistant".into()),
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
            })
            .unwrap();
            if tx.send(Ok(Event::default().data(role_chunk))).await.is_err() {
                return;
            }

            // Lock context
            let mut inner = state.ctx.lock().await;
            inner.clear();
            if let Err(e) = inner.prefill(&input_tokens) {
                info!("  Prefill error: {e}");
                return;
            }

            let mut sampler = SendSampler(build_sampler_params(
                temp, top_p, top_k, min_p,
                rep_penalty, freq_penalty, pres_penalty, seed,
            ));
            let mut count = 0u32;
            let mut text_buf = String::new();
            let mut current = inner.sample(&mut sampler);

            loop {
                if count >= max_tokens {
                    let chunk = serde_json::to_string(&SseChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![SseChoice {
                            index: 0,
                            delta: SseDelta {
                                role: None,
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some("length".into()),
                        }],
                    })
                    .unwrap();
                    let _ = tx.send(Ok(Event::default().data(chunk))).await;
                    break;
                }

                if state.model.is_eog_token(current) {
                    let reason = if has_tools && text_buf.contains("<tool_call>") {
                        "tool_calls"
                    } else {
                        "stop"
                    };
                    let chunk = serde_json::to_string(&SseChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![SseChoice {
                            index: 0,
                            delta: SseDelta {
                                role: None,
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some(reason.into()),
                        }],
                    })
                    .unwrap();
                    let _ = tx.send(Ok(Event::default().data(chunk))).await;
                    break;
                }

                let piece = decode_token_piece(&state.model, current);
                let content = clean_text(&piece);

                if !content.is_empty() {
                    let chunk = serde_json::to_string(&SseChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![SseChoice {
                            index: 0,
                            delta: SseDelta {
                                role: None,
                                content: Some(content.clone()),
                                tool_calls: None,
                            },
                            finish_reason: None,
                        }],
                    })
                    .unwrap();
                    if tx.send(Ok(Event::default().data(chunk))).await.is_err() {
                        break;
                    }
                }

                text_buf.push_str(&piece);

                // Check stop
                let mut stop_now = false;
                for s in &stop_clone {
                    if text_buf.contains(s) {
                        stop_now = true;
                        break;
                    }
                }
                if stop_now {
                    let chunk = serde_json::to_string(&SseChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![SseChoice {
                            index: 0,
                            delta: SseDelta {
                                role: None,
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some("stop".into()),
                        }],
                    })
                    .unwrap();
                    let _ = tx.send(Ok(Event::default().data(chunk))).await;
                    break;
                }

                // Check tool call completeness
                if has_tools && text_buf.contains("<tool_call>") {
                    let open = text_buf.matches("<tool_call>").count();
                    let close = text_buf.matches("</tool_call>").count();
                    if close >= open {
                        let chunk = serde_json::to_string(&SseChunk {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![SseChoice {
                                index: 0,
                                delta: SseDelta {
                                    role: None,
                                    content: None,
                                    tool_calls: None,
                                },
                                finish_reason: Some("tool_calls".into()),
                            }],
                        })
                        .unwrap();
                        let _ = tx.send(Ok(Event::default().data(chunk))).await;
                        break;
                    }
                }

                let pos = input_tokens.len() as i32 + count as i32;
                if let Err(e) = inner.decode(current, pos) {
                    info!("  Decode error: {e}");
                    break;
                }
                count += 1;
                current = inner.sample(&mut sampler);
            }
        });

        let stream = ReceiverStream::new(rx);
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        return Ok(sse.into_response());
    }

    // ── NON-STREAMING ──
    let mut inner = state.ctx.lock().await;
    inner.clear();
    inner.prefill(&input_tokens).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Prefill: {e}"))
    })?;

    let mut sampler = build_sampler(&req);
    let (output_tokens, output_text, tool_calls) =
        generate_tokens(&state, &mut inner, &input_tokens, &mut sampler, max_tokens, &stop);

    let completion_tokens = output_tokens.len() as u32;
    info!("  {} generated tokens", completion_tokens);

    let finish_reason = if has_tools && !tool_calls.is_empty() {
        "tool_calls"
    } else if completion_tokens < max_tokens {
        "stop"
    } else {
        "length"
    };

    Ok(Json(ChatResponse {
        id: chat_id,
        object: "chat.completion".into(),
        created,
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".into(),
                content: if tool_calls.is_empty() {
                    Some(output_text)
                } else {
                    Some(output_text)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            },
            finish_reason: finish_reason.into(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}
