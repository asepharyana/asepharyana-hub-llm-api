//! LlamaEngine — safe wrapper around llama-cpp-2 inference.
//!
//! Encapsulates model loading, context management (with the unavoidable
//! lifetime transmute), tokenization, and generation.

use std::num::NonZeroU32;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::TokenToStringError;
use tokio::sync::Mutex;
use tracing::info;

use crate::config::CONFIG;
use crate::domain::LlmError;

// ── Thread-safe wrapper for raw llama.cpp context ──

/// Wrapper around [`LlamaContext`] that makes it Send + Sync.
///
/// # Safety
///
/// The contained context has its lifetime transmuted to `'static` because it is
/// owned by [`LlamaEngine`] which lives for the entire program lifetime (held in
/// an `Arc`). The engine is only dropped at process shutdown, so no dangling
/// reference can be created.
/// Wrapper around LlamaContext with `'static` lifetime for sharing.
pub struct CtxInner {
    /// Invariant: this context is dropped only when the engine is destroyed.
    context: LlamaContext<'static>,
}

unsafe impl Send for CtxInner {}
unsafe impl Sync for CtxInner {}

/// Wrapper for [`LlamaSampler`] to make it Send + Sync.
///
/// # Safety
///
/// `llama-cpp-2`'s `LlamaSampler` is a C opaque pointer. The underlying
/// `llama.cpp` sampling API is reentrant for distinct contexts and thread-safe
/// when used with a single context from one thread at a time (which we enforce
/// via `Mutex<CtxInner>`).
pub struct SendSampler(pub LlamaSampler);

unsafe impl Send for SendSampler {}
unsafe impl Sync for SendSampler {}

impl std::ops::Deref for SendSampler {
    type Target = LlamaSampler;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SendSampler {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl CtxInner {
    pub(crate) fn clear(&mut self) {
        self.context.clear_kv_cache();
    }

    pub(crate) fn prefill(&mut self, tokens: &[LlamaToken]) -> Result<(), String> {
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, &token) in tokens.iter().enumerate() {
            batch
                .add(token, i as i32, &[0], i == tokens.len() - 1)
                .map_err(|e| e.to_string())?;
        }
        self.context.decode(&mut batch).map_err(|e| e.to_string())
    }

    pub(crate) fn sample(&mut self, sampler: &mut LlamaSampler) -> LlamaToken {
        sampler.sample(&self.context, -1)
    }

    pub(crate) fn decode(&mut self, token: LlamaToken, pos: i32) -> Result<(), String> {
        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(token, pos, &[0], true)
            .map_err(|e| e.to_string())?;
        self.context.decode(&mut batch).map_err(|e| e.to_string())
    }
}

// ── LlamaEngine ──

/// Safe interface to a llama.cpp model and inference context.
///
/// All access to the underlying context is serialized through a `Mutex`,
/// so only one generation can happen at a time. This is intentional —
/// the model is designed for sequential inference.
pub struct LlamaEngine {
    /// The loaded model (read-only after load, safe to share).
    pub model: LlamaModel,

    /// The inference context (single-threaded access via Mutex).
    pub ctx: Mutex<CtxInner>,
}

impl LlamaEngine {
    /// Load a model from disk and create an inference context.
    ///
    /// # Errors
    ///
    /// Returns `LlmError::Model` if the model cannot be loaded or the context
    /// cannot be created.
    pub fn load() -> Result<Self, LlmError> {
        info!("Initializing llama backend...");
        let backend = LlamaBackend::init().map_err(|e| {
            LlmError::Model(format!("Backend init failed: {e}"))
        })?;

        // Backend must outlive model and context. We leak it to achieve 'static
        // lifetime since the engine lives for the program lifetime.
        let backend: &'static LlamaBackend = Box::leak(Box::new(backend));

        info!("Loading model: {}", CONFIG.model_path);
        let model = LlamaModel::load_from_file(backend, &CONFIG.model_path, &LlamaModelParams::default())
            .map_err(|e| LlmError::Model(format!("Failed to load model: {e}")))?;
        info!("  Vocab: {}", model.n_vocab());
        info!("  Params: {}", model.n_params());
        info!("  Layers: {}", model.n_layer());

        info!("Creating context...");
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONFIG.n_ctx))
            .with_n_batch(CONFIG.n_batch)
            .with_n_threads(CONFIG.n_threads)
            .with_n_threads_batch(CONFIG.n_threads);

        let context = model
            .new_context(backend, ctx_params)
            .map_err(|e| LlmError::Model(format!("Failed to create context: {e}")))?;

        // SAFETY: `context` is tied to `backend`'s lifetime, which we leaked
        // above to achieve `'static`. The engine owns both and lives for the
        // program duration (held in a global Arc). When the engine is dropped
        // at process shutdown, the leaked backend is cleaned up by the OS.
        let context: LlamaContext<'static> = unsafe { std::mem::transmute(context) };

        Ok(Self {
            model,
            ctx: Mutex::new(CtxInner { context }),
        })
    }

    /// Tokenize a prompt string into tokens.
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<LlamaToken>, LlmError> {
        self.model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| LlmError::Model(format!("Tokenization failed: {e}")))
    }

    /// Decode a single token to its string representation.
    pub fn decode_token(&self, token: LlamaToken) -> String {
        let bytes = match self.model.token_to_piece_bytes(token, 32, true, None) {
            Ok(b) => b,
            Err(TokenToStringError::InsufficientBufferSpace(neg)) => {
                let size = (-neg).max(0).try_into().unwrap_or(256);
                self.model
                    .token_to_piece_bytes(token, size, true, None)
                    .unwrap_or_default()
            }
            _ => return String::new(),
        };
        String::from_utf8(bytes).unwrap_or_default()
    }

    /// Decode multiple tokens to a single string.
    pub fn decode_tokens(&self, tokens: &[LlamaToken]) -> String {
        let mut out = String::with_capacity(tokens.len() * 4);
        for &token in tokens {
            out.push_str(&self.decode_token(token));
        }
        out
    }

    /// Check if a token is an end-of-generation token.
    pub fn is_eog(&self, token: LlamaToken) -> bool {
        self.model.is_eog_token(token)
    }

    /// Return a reference to the context mutex for advanced operations.
    pub fn ctx(&self) -> &Mutex<CtxInner> {
        &self.ctx
    }

    /// Generate tokens (non-streaming) and return output tokens, cleaned text, and tool calls.
    ///
    /// Locks the context mutex, prefill the prompt, then iterates sampling + decoding
    /// until EOG, max_tokens, stop sequence, or tool call completion.
    pub async fn generate(
        &self,
        input_tokens: &[LlamaToken],
        sampler: &mut SendSampler,
        max_tokens: u32,
        stop: &[String],
    ) -> Result<(Vec<LlamaToken>, String), LlmError> {
        let mut inner = self.ctx.lock().await;
        inner.clear();
        inner
            .prefill(input_tokens)
            .map_err(|e| LlmError::Model(format!("Prefill: {e}")))?;

        let mut output: Vec<LlamaToken> = Vec::new();
        let mut text_buf = String::new();
        let mut stop_now = false;

        let mut current = inner.sample(sampler);

        // Skip leading EOS tokens (like <|im_end|> as first token)
        while output.is_empty() && self.model.is_eog_token(current) {
            let pos = input_tokens.len() as i32 + output.len() as i32;
            if let Err(e) = inner.decode(current, pos) {
                tracing::info!("  Decode error: {e}");
                break;
            }
            current = inner.sample(sampler);
        }

        for _ in 0..max_tokens {
            if self.model.is_eog_token(current) {
                break;
            }

            let pos = input_tokens.len() as i32 + output.len() as i32;
            output.push(current);

            let piece = self.decode_token(current);
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

            // Check for tool_call block completion
            if text_buf.contains("<tool_call>") {
                let close_count = text_buf.matches("</tool_call>").count();
                let open_count = text_buf.matches("<tool_call>").count();
                if open_count > 0 && close_count >= open_count {
                    break;
                }
            }

            if let Err(e) = inner.decode(current, pos) {
                tracing::info!("  Decode error: {e}");
                break;
            }

            current = inner.sample(sampler);
        }

        Ok((output, text_buf))
    }
}
