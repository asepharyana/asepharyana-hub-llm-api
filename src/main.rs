use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::info;

// ── Thread-safe wrapper ──
struct CtxInner {
    context: llama_cpp_2::context::LlamaContext<'static>,
    sampler: LlamaSampler,
}

// SAFETY: llama.cpp contexts are accessed from a single thread via the Mutex
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

    // Use raw pointer to avoid borrow checker limitations with llama-cpp-2 API
    fn sample_token(&mut self) -> LlamaToken {
        let ctx_ptr: *const llama_cpp_2::context::LlamaContext = &self.context;
        // SAFETY: sampler is the sole owner of the context reference during this call
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

const DEFAULT_MODEL_PATH: &str = "/root/models/gguf/MiniCPM-V-4.6-Q4_K_M.gguf";
const EOS_TOKEN: i32 = 248044;

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

    // Extend lifetime: model outlives context
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
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let prompt = build_prompt(&req.messages);
    let max_tokens = req.max_tokens.unwrap_or(256).min(1024);

    info!("Chat: {} chars, max_tokens={}", prompt.len(), max_tokens);

    // Tokenize
    let input_tokens = state
        .model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let prompt_tokens = input_tokens.len() as u32;
    info!("  {} prompt tokens", prompt_tokens);

    // Lock context
    let mut inner = state.ctx.lock().await;

    inner.clear();
    inner.prefill(&input_tokens).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Prefill: {e}"))
    })?;

    // Generate
    let mut output_tokens: Vec<LlamaToken> = Vec::new();

    // First sample from prefill
    let mut current = inner.sample_token();

    for _ in 0..max_tokens {
        if current.0 == EOS_TOKEN {
            break;
        }
        let pos = input_tokens.len() as i32 + output_tokens.len() as i32;
        output_tokens.push(current);

        // Decode the last token
        inner.decode_token(current, pos)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Decode: {e}")))?;

        // Sample next token
        current = inner.sample_token();
    }

    let output_text = state
        .model
        .tokens_to_str(&output_tokens, Special::Tokenize)
        .unwrap_or_else(|_| "<decode error>".to_string());

    let completion_tokens = output_tokens.len() as u32;

    info!("  {} generated tokens", completion_tokens);

    Ok(Json(ChatResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        created: chrono::Utc::now().timestamp(),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".into(),
                content: output_text,
            },
            finish_reason: if completion_tokens < max_tokens {
                "stop"
            } else {
                "length"
            }
            .into(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => prompt.push_str(&format!("System: {}\n", msg.content)),
            "user" => prompt.push_str(&format!("User: {}\n", msg.content)),
            "assistant" => prompt.push_str(&format!("Assistant: {}\n", msg.content)),
            _ => prompt.push_str(&format!("{}\n", msg.content)),
        }
    }
    prompt.push_str("Assistant: ");
    prompt
}
