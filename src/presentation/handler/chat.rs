//! Chat completions endpoint — streaming and non-streaming.
//!
//! Both paths share the same synchronous generation core
//! ([`LlamaEngine::generate`]), which runs on the tokio blocking pool via
//! `spawn_blocking` so worker threads are not hogged by CPU-bound inference.
//! The streaming path forwards each generated token into an SSE channel.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chrono::Utc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;

use crate::application::chat;
use crate::config::CONFIG;
use crate::domain::entity::{
    ChatRequest, ChatResponse, Choice, FinishReason, ResponseMessage, SseChunk, SseDelta, Usage,
};
use crate::infrastructure::llama::SendSampler;
use crate::presentation::error::AppError;
use crate::presentation::handler::metrics;
use crate::presentation::state::AppState;

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, AppError> {
    // Strict model validation — reject unknown model ids up front.
    chat::validate_model(&req.model).map_err(AppError::BadRequest)?;

    // Cap max_tokens at the configured hard limit (0 = unlimited).
    let max_tokens = req.max_tokens.unwrap_or(256).min(CONFIG.max_tokens.max(1));
    let stop = req.stop.clone().unwrap_or_default();
    let prompt = chat::build_prompt(&req.messages, &req.tools).map_err(AppError::LlmError)?;

    // Tokenize (fast — keep on the async thread).
    let input_tokens = state
        .engine
        .tokenize(&prompt)
        .map_err(|e| AppError::LlmError(e.to_string()))?;

    let prompt_tokens = input_tokens.len() as u32;
    info!(
        "Chat: {} prompt tokens, max_tokens={}, tools={}",
        prompt_tokens,
        max_tokens,
        req.tools.as_ref().is_some_and(|t| !t.is_empty())
    );

    metrics::count_request(req.stream.unwrap_or(false));

    let response = if req.stream.unwrap_or(false) {
        handle_streaming(state.clone(), req, max_tokens, stop, input_tokens).await?
    } else {
        handle_non_streaming(state.clone(), req, max_tokens, stop, input_tokens).await?
    };
    Ok(response)
}

// ── Non-streaming path ──

async fn handle_non_streaming(
    state: Arc<AppState>,
    req: ChatRequest,
    max_tokens: u32,
    stop: Vec<String>,
    input_tokens: Vec<llama_cpp_2::token::LlamaToken>,
) -> Result<Response, AppError> {
    let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = Utc::now().timestamp();
    let prompt_tokens = input_tokens.len() as u32;
    let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());
    let params = chat::SamplerParams::from_request(&req);

    let engine = state.engine.clone();
    let gen_start = std::time::Instant::now();
    let outcome = tokio::task::spawn_blocking(move || {
        let mut sampler = SendSampler(chat::build_sampler(&params));
        engine.generate(
            &input_tokens,
            &mut sampler,
            max_tokens,
            &stop,
            has_tools,
            &mut |_token, _piece| true,
        )
    })
    .await
    .map_err(|e| AppError::Internal(format!("Generation task panicked: {e}")))?
    .map_err(AppError::from)?;
    let duration_ms = gen_start.elapsed().as_millis() as u64;

    let completion_tokens = outcome.tokens.len() as u32;
    info!(
        "  {} generated tokens in {}ms",
        completion_tokens, duration_ms
    );

    metrics::record_tokens(prompt_tokens, completion_tokens, duration_ms);
    if outcome.finish == FinishReason::Aborted {
        metrics::count_aborted();
    }

    let (reasoning, cleaned) = chat::clean_text(&outcome.text);
    let (output_text, tool_calls) = chat::parse_tool_calls(&cleaned);

    let finish_reason = match (outcome.finish, tool_calls.is_empty()) {
        (FinishReason::ToolCalls, false) => "tool_calls",
        // Model opened a <tool_call> but never completed it — don't claim a call.
        (FinishReason::ToolCalls, true) => "stop",
        (f, _) => f.as_str(),
    };

    let reasoning_opt = if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    };

    let tok_per_s = if duration_ms > 0 {
        Some(completion_tokens as f64 / (duration_ms as f64 / 1000.0))
    } else {
        None
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
                content: Some(output_text),
                reasoning_content: reasoning_opt,
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
            duration_ms: Some(duration_ms),
            tokens_per_second: tok_per_s,
        },
    })
    .into_response())
}

// ── Streaming path ──

async fn handle_streaming(
    state: Arc<AppState>,
    req: ChatRequest,
    max_tokens: u32,
    stop: Vec<String>,
    input_tokens: Vec<llama_cpp_2::token::LlamaToken>,
) -> Result<Response, AppError> {
    let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = Utc::now().timestamp();
    let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());
    let model_name = req.model.clone();
    let prompt_tokens = input_tokens.len() as u32;
    let params = chat::SamplerParams::from_request(&req);

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    // First chunk: announce the assistant role.
    let role_chunk = SseChunk::delta(
        chat_id.clone(),
        created,
        model_name.clone(),
        SseDelta {
            role: Some("assistant".into()),
            content: None,
            tool_calls: None,
            reasoning_content: None,
        },
    );
    let role_event = serde_json::to_string(&role_chunk).unwrap();
    let _ = tx.send(Ok(Event::default().data(role_event))).await;

    let engine = state.engine.clone();
    tokio::task::spawn_blocking(move || {
        let gen_start = std::time::Instant::now();
        let mut sampler = SendSampler(chat::build_sampler(&params));
        let mut text_buf = String::new();
        let mut sent_len: usize = 0;
        // Byte offset in text_buf where the content phase begins (right after
        // the first `</think>`); None while still thinking.
        let mut content_start: Option<usize> = None;

        let outcome = engine.generate(
            &input_tokens,
            &mut sampler,
            max_tokens,
            &stop,
            has_tools,
            &mut |_token, piece| {
                text_buf.push_str(piece);

                // Robust boundary detection on the *full* buffer — a `</think>`
                // tag may be split across tokens, which would defeat a search
                // over the incremental fragment only.
                if content_start.is_none() {
                    if let Some(pos) = text_buf.find("</think>") {
                        content_start = Some(pos + 8);
                    }
                }

                let new_text = &text_buf[sent_len..];
                if new_text.is_empty() {
                    return true;
                }

                let (reasoning, content) =
                    chat::split_stream_chunk(new_text, sent_len, content_start);

                if let Some(reasoning) = reasoning {
                    let chunk = SseChunk::delta(
                        chat_id.clone(),
                        created,
                        model_name.clone(),
                        SseDelta {
                            role: None,
                            content: None,
                            tool_calls: None,
                            reasoning_content: Some(reasoning),
                        },
                    );
                    let event = serde_json::to_string(&chunk).unwrap();
                    if tx.blocking_send(Ok(Event::default().data(event))).is_err() {
                        return false;
                    }
                }

                if !content.is_empty() {
                    let chunk = SseChunk::delta(
                        chat_id.clone(),
                        created,
                        model_name.clone(),
                        SseDelta {
                            role: None,
                            content: Some(content),
                            tool_calls: None,
                            reasoning_content: None,
                        },
                    );
                    let event = serde_json::to_string(&chunk).unwrap();
                    if tx.blocking_send(Ok(Event::default().data(event))).is_err() {
                        return false;
                    }
                }

                sent_len = text_buf.len();
                true
            },
        );

        match outcome {
            Ok(outcome) => {
                let duration_ms = gen_start.elapsed().as_millis() as u64;
                let completion_tokens = outcome.tokens.len() as u32;
                let usage = Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    duration_ms: Some(duration_ms),
                    tokens_per_second: if duration_ms > 0 {
                        Some(completion_tokens as f64 / (duration_ms as f64 / 1000.0))
                    } else {
                        None
                    },
                };
                info!(
                    "  stream: {} generated tokens in {}ms",
                    completion_tokens, duration_ms
                );
                metrics::record_tokens(prompt_tokens, completion_tokens, duration_ms);
                if outcome.finish == FinishReason::Aborted {
                    metrics::count_aborted();
                }

                // If the model never emitted `</think>`, everything was streamed
                // as reasoning_content. Flush it as content so the client always
                // receives the response text.
                if content_start.is_none() && !text_buf.is_empty() {
                    let cleaned = chat::strip_markup(&text_buf);
                    if !cleaned.is_empty() {
                        let chunk = SseChunk::delta(
                            chat_id.clone(),
                            created,
                            model_name.clone(),
                            SseDelta {
                                role: None,
                                content: Some(cleaned),
                                tool_calls: None,
                                reasoning_content: None,
                            },
                        );
                        let event = serde_json::to_string(&chunk).unwrap();
                        let _ = tx.blocking_send(Ok(Event::default().data(event)));
                    }
                }

                // Single-shot tool-calls delta (this model emits whole blocks).
                let mut sent_tool_calls = false;
                if outcome.finish == FinishReason::ToolCalls {
                    let (_cleaned, calls) = chat::parse_tool_calls(&outcome.text);
                    if !calls.is_empty() {
                        sent_tool_calls = true;
                        let chunk = SseChunk::delta(
                            chat_id.clone(),
                            created,
                            model_name.clone(),
                            SseDelta {
                                role: None,
                                content: None,
                                tool_calls: Some(calls),
                                reasoning_content: None,
                            },
                        );
                        let event = serde_json::to_string(&chunk).unwrap();
                        let _ = tx.blocking_send(Ok(Event::default().data(event)));
                    }
                }

                let finish_reason = match (outcome.finish, sent_tool_calls) {
                    (FinishReason::ToolCalls, true) => "tool_calls",
                    (FinishReason::ToolCalls, false) => "stop",
                    (f, _) => f.as_str(),
                };

                let finish_chunk =
                    SseChunk::finish(chat_id.clone(), created, model_name.clone(), finish_reason);
                let event = serde_json::to_string(&finish_chunk).unwrap();
                let _ = tx.blocking_send(Ok(Event::default().data(event)));

                let usage_chunk = SseChunk::usage(chat_id, created, model_name, usage);
                let event = serde_json::to_string(&usage_chunk).unwrap();
                let _ = tx.blocking_send(Ok(Event::default().data(event)));
            }
            Err(e) => {
                // Surface the error instead of silently truncating the stream.
                metrics::count_error();
                let error_body = serde_json::json!({
                    "error": {
                        "message": e.to_string(),
                        "type": "server_error",
                    }
                });
                let _ = tx.blocking_send(Ok(Event::default().data(error_body.to_string())));
            }
        }

        // OpenAI-compatible terminator.
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    let stream = ReceiverStream::new(rx);
    let mut headers = HeaderMap::new();
    headers.insert("X-Accel-Buffering", "no".parse().unwrap());
    headers.insert("Cache-Control", "no-cache".parse().unwrap());
    headers.insert("Connection", "keep-alive".parse().unwrap());
    let sse = Sse::new(stream).keep_alive(KeepAlive::default());
    Ok((headers, sse).into_response())
}
