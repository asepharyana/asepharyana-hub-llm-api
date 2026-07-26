//! Chat completions endpoint — streaming and non-streaming.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::{Json, response::{IntoResponse, Response}};
use chrono::Utc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;

use crate::application::chat;
use crate::domain::entity::{
    ChatRequest, ChatResponse, Choice, ResponseMessage, SseChunk, SseChoice, SseDelta, Usage,
};
use crate::infrastructure::llama::SendSampler;
use crate::presentation::error::AppError;
use crate::presentation::state::AppState;

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, AppError> {
    let max_tokens = req.max_tokens.unwrap_or(256).min(1024);
    let stop = req.stop.clone().unwrap_or_default();
    let prompt = chat::build_prompt(&state.engine.model, &req.messages, &req.tools)
        .map_err(|e| AppError::LlmError(e))?;

    // Tokenize
    let input_tokens = state
        .engine
        .tokenize(&prompt)
        .map_err(|e| AppError::LlmError(e.to_string()))?;

    let prompt_tokens = input_tokens.len() as u32;
    info!(
        "  Chat: {} prompt tokens, max_tokens={}, tools={}",
        prompt_tokens,
        max_tokens,
        req.tools.as_ref().is_some_and(|t| !t.is_empty())
    );

    let response = if req.stream.unwrap_or(false) {
        handle_streaming(state.clone(), req, max_tokens, stop, input_tokens).await?
    } else {
        handle_non_streaming(state.clone(), req, max_tokens, stop, input_tokens).await?
    };
    Ok(response.into_response())
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
    let mut sampler = SendSampler(chat::build_sampler(&params));

    let (output_tokens, raw_text) = state
        .engine
        .generate(&input_tokens, &mut sampler, max_tokens, &stop)
        .await?;

    let (reasoning, cleaned) = chat::clean_text(&raw_text);
    let (output_text, tool_calls) = chat::parse_tool_calls(&cleaned);

    let completion_tokens = output_tokens.len() as u32;
    info!("  {} generated tokens", completion_tokens);

    let finish_reason = if has_tools && !tool_calls.is_empty() {
        "tool_calls"
    } else if completion_tokens < max_tokens {
        "stop"
    } else {
        "length"
    };

    let reasoning_opt = if reasoning.is_empty() { None } else { Some(reasoning) };

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
    let params = chat::SamplerParams::from_request(&req);
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

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
                    reasoning_content: None,
                },
                finish_reason: None,
            }],
        })
        .unwrap();
        if tx.send(Ok(Event::default().data(role_chunk))).await.is_err() {
            return;
        }

        // Build sampler
        let mut sampler = SendSampler(chat::build_sampler(&params));

        // Lock context
        let mut inner = state.engine.ctx().lock().await;
        inner.clear();
        if let Err(e) = inner.prefill(&input_tokens) {
            info!("  Prefill error: {e}");
            return;
        }

        let mut count = 0u32;
        let mut text_buf = String::new();
        let mut sent_len: usize = 0;     // how many chars of text_buf have been sent
        let mut think_done: bool = false; // true once </think> seen
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
                            reasoning_content: None,
                        },
                        finish_reason: Some("length".into()),
                    }],
                })
                .unwrap();
                let _ = tx.send(Ok(Event::default().data(chunk))).await;
                break;
            }

            // Skip leading EOS tokens (like <|im_end|> at start of generation)
            if state.engine.is_eog(current) && text_buf.is_empty() {
                // safety bound: don't skip more than 10
                if count >= max_tokens || count > 10 {
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
                                reasoning_content: None,
                            },
                            finish_reason: Some("stop".into()),
                        }],
                    })
                    .unwrap();
                    let _ = tx.send(Ok(Event::default().data(chunk))).await;
                    break;
                }
                count += 1;
                let pos = input_tokens.len() as i32 + count as i32;
                if let Err(e) = inner.decode(current, pos) {
                    info!("  Decode error: {e}");
                    break;
                }
                current = inner.sample(&mut sampler);
                continue;
            }

            if state.engine.is_eog(current) {
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
                            reasoning_content: None,
                        },
                        finish_reason: Some(reason.into()),
                    }],
                })
                .unwrap();
                let _ = tx.send(Ok(Event::default().data(chunk))).await;
                break;
            }

            let piece = state.engine.decode_token(current);

            // Push into buffer
            text_buf.push_str(&piece);

            // Detect </think> transition
            if !think_done && text_buf.contains("</think>") {
                think_done = true;
            }

            // Find new text since last send
            let new_text = &text_buf[sent_len..]; // everything not yet streamed
            if new_text.is_empty() {
                // Nothing new to send; skip straight to decode
                let pos = input_tokens.len() as i32 + count as i32;
                if let Err(e) = inner.decode(current, pos) {
                    info!("  Decode error: {e}");
                    break;
                }
                count += 1;
                current = inner.sample(&mut sampler);
                continue;
            }

            // Strip special tokens from the NEW text chunk
            // Split at </think> if present — before goes to reasoning, after to content
            let (reasoning_part, content_part) = if let Some(pos) = new_text.find("</think>") {
                let before = new_text[..pos]
                    .replace("<|im_end|>", "")
                    .replace("<|im_start|>", "")
                    .replace("<think>", "")
                    .trim()
                    .to_string();
                let after = new_text[pos + 8..]
                    .replace("<|im_end|>", "")
                    .replace("<|im_start|>", "")
                    .replace("<think>", "")
                    .trim()
                    .to_string();
                think_done = true;
                (Some(before), after)
            } else if think_done {
                let cleaned = new_text
                    .replace("<|im_end|>", "")
                    .replace("<|im_start|>", "")
                    .replace("<think>", "")
                    .trim()
                    .to_string();
                (None, cleaned)
            } else {
                let cleaned = new_text
                    .replace("<|im_end|>", "")
                    .replace("<|im_start|>", "")
                    .replace("<think>", "")
                    .trim()
                    .to_string();
                (Some(cleaned), String::new())
            };

            // Send reasoning part (before </think>, or entire text if still thinking)
            if let Some(ref r) = reasoning_part {
                if !r.is_empty() {
                    let delta = SseDelta {
                        role: None,
                        content: None,
                        reasoning_content: Some(r.clone()),
                        tool_calls: None,
                    };
                    let chunk = serde_json::to_string(&SseChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![SseChoice {
                            index: 0,
                            delta,
                            finish_reason: None,
                        }],
                    })
                    .unwrap();
                    let _ = tx.send(Ok(Event::default().data(chunk))).await;
                }
            }

            // Send content part (after </think>, or never if model doesn't think)
            if !content_part.is_empty() {
                let delta = SseDelta {
                    role: None,
                    content: Some(content_part),
                    reasoning_content: None,
                    tool_calls: None,
                };
                let chunk = serde_json::to_string(&SseChunk {
                    id: chat_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model_name.clone(),
                    choices: vec![SseChoice {
                        index: 0,
                        delta,
                        finish_reason: None,
                    }],
                })
                .unwrap();
                if tx.send(Ok(Event::default().data(chunk))).await.is_err() {
                    break;
                }
            }

            sent_len = text_buf.len();

            // Check stop sequences
            let mut stop_now = false;
            for s in &stop {
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
                            reasoning_content: None,
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
                                reasoning_content: None,
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
    Ok(sse.into_response())
}
