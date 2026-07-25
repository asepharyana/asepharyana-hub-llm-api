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
use std::{
    num::NonZeroU32,
    sync::Arc,
};
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use tracing::info;

// ── Thread-safe wrapper ──
struct CtxInner {
    context: llama_cpp_2::context::LlamaContext<'static>,
    sampler: LlamaSampler,
}

unsafe impl Send for CtxInner {}
unsafe impl Sync for CtxInner {}

impl CtxInner {
    fn clear(&mut self) {
        self.context.clear_kv_cache();
    }

    fn prefill(&mut self, tokens: &[LlamaToken]) -> Result<(), String> {
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, &token) in tokens.iter().enumerate() {
            batch.add(token, i as i32, &[0], i == tokens.len() - 1)
                .map_err(|e| e.to_string())?;
        }
        self.context.decode(&mut batch).map_err(|e| e.to_string())
    }

    fn sample_token(&mut self) -> LlamaToken {
        let ctx_ptr: *const llama_cpp_2::context::LlamaContext = &self.context;
        let ctx_ref = unsafe { &*ctx_ptr };
        self.sampler.sample(ctx_ref, -1)
    }

    fn decode_token(&mut self, token: LlamaToken, pos: i32) -> Result<(), String> {
        let mut batch = LlamaBatch::new(1, 1);
        batch.add(token, pos, &[0], true)
            .map_err(|e| e.to_string())?;
        self.context.decode(&mut batch).map_err(|e| e.to_string())
    }
}

struct AppState {
    model: LlamaModel,
    ctx: Mutex<CtxInner>,
}

// ── OpenAI Types ──

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

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
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
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
}

const DEFAULT_MODEL_PATH: &str = "/models/MiniCPM-V-4.6-Q4_K_M.gguf";

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
    let model = LlamaModel::load_from_file(
        &backend,
        &model_path,
        &LlamaModelParams::default(),
    )
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

    let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    let state = Arc::new(AppState {
        model,
        ctx: Mutex::new(CtxInner { context, sampler }),
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
        model: "minicpm-v-4.6-q4_k_m".into(),
    })
}

async fn list_models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".into(),
        data: vec![ModelInfo {
            id: "minicpm-v-4.6".into(),
            object: "model".into(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "asepharyana".into(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    check_auth(&headers)?;

    let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let max_tokens = req.max_tokens.unwrap_or(256).min(1024);
    let prompt = build_prompt(&req.messages);

    // Tokenize
    let input_tokens = state
        .model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let prompt_tokens = input_tokens.len() as u32;
    info!("  Chat: {} prompt tokens, max_tokens={}", prompt_tokens, max_tokens);

    if req.stream.unwrap_or(false) {
        // ── Streaming mode: spawn generator, pipe via mpsc channel ──
        let state = state.clone();
        let model_name = req.model.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);

        tokio::spawn(async move {
            // Send role chunk
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
                    },
                    finish_reason: None,
                }],
            }).unwrap();
            let _ = tx.send(Ok(Event::default().data(role_chunk))).await;

            // Lock model context
            let mut inner = state.ctx.lock().await;
            inner.clear();
            if let Err(e) = inner.prefill(&input_tokens) {
                info!("  Prefill error: {e}");
                return;
            }

            let mut count = 0u32;
            loop {
                if count >= max_tokens {
                    let chunk = serde_json::to_string(&SseChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![SseChoice {
                            index: 0,
                            delta: SseDelta { role: None, content: None },
                            finish_reason: Some("length".into()),
                        }],
                    }).unwrap();
                    let _ = tx.send(Ok(Event::default().data(chunk))).await;
                    break;
                }

                let token = inner.sample_token();

                if state.model.is_eog_token(token) {
                    let chunk = serde_json::to_string(&SseChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![SseChoice {
                            index: 0,
                            delta: SseDelta { role: None, content: None },
                            finish_reason: Some("stop".into()),
                        }],
                    }).unwrap();
                    let _ = tx.send(Ok(Event::default().data(chunk))).await;
                    break;
                }

                let piece = decode_token_piece(&state.model, token);
                let content = clean_text(&piece);
                if !content.is_empty() {
                    let chunk = serde_json::to_string(&SseChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![SseChoice {
                            index: 0,
                            delta: SseDelta { role: None, content: Some(content) },
                            finish_reason: None,
                        }],
                    }).unwrap();
                    if tx.send(Ok(Event::default().data(chunk))).await.is_err() {
                        break; // Client disconnected
                    }
                }

                let pos = input_tokens.len() as i32 + count as i32;
                if let Err(e) = inner.decode_token(token, pos) {
                    info!("  Decode error: {e}");
                    break;
                }
                count += 1;
            }
        });

        let stream = ReceiverStream::new(rx);
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        Ok(sse.into_response())
    } else {
        // ── Non-streaming mode ──
        let mut inner = state.ctx.lock().await;
        inner.clear();
        inner.prefill(&input_tokens).map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Prefill: {e}"))
        })?;

        let mut output_tokens: Vec<LlamaToken> = Vec::new();
        let mut current = inner.sample_token();

        for _ in 0..max_tokens {
            if state.model.is_eog_token(current) {
                break;
            }
            let pos = input_tokens.len() as i32 + output_tokens.len() as i32;
            output_tokens.push(current);
            inner.decode_token(current, pos)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Decode: {e}")))?;
            current = inner.sample_token();
        }

        let output_text = decode_tokens(&state.model, &output_tokens);
        let completion_tokens = output_tokens.len() as u32;

        info!("  {} generated tokens", completion_tokens);

        Ok(Json(ChatResponse {
            id: chat_id,
            object: "chat.completion".into(),
            created,
            model: req.model,
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".into(),
                    content: output_text,
                },
                finish_reason: if completion_tokens < max_tokens { "stop" } else { "length" }.into(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
        .into_response())
    }
}

// ── Helper Functions ──

fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for (i, msg) in messages.iter().enumerate() {
        let role = match msg.role.as_str() {
            "system" => "system",
            "user" => "user",
            "assistant" => "assistant",
            _ => "user",
        };
        if i == 0 && role == "system" {
            prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content));
        } else {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, msg.content));
        }
    }
    prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    prompt
}

fn decode_token_piece(model: &LlamaModel, token: LlamaToken) -> String {
    let bytes = match model.token_to_piece_bytes(token, 32, true, None) {
        Ok(b) => b,
        Err(TokenToStringError::InsufficientBufferSpace(neg)) => {
            let size = (-neg).max(0).try_into().unwrap_or(256);
            model.token_to_piece_bytes(token, size, true, None).unwrap_or_default()
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
