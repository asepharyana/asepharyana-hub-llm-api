//! Domain entities for the LLM inference API.
//!
//! Pure data structs with no framework dependencies beyond serde.
//! These represent the OpenAI-compatible API shapes used across all layers.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
//  REQUEST TYPES
// ═══════════════════════════════════════════════════════════════

/// OpenAI-compatible chat completion request body.
#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDef>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

/// A single message in the chat conversation.
#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallResponse>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

/// Tool/function definition for function calling.
#[derive(Deserialize, Serialize, Clone)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

// ═══════════════════════════════════════════════════════════════
//  RESPONSE TYPES
// ═══════════════════════════════════════════════════════════════

/// OpenAI-compatible chat completion response (non-streaming).
#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ═══════════════════════════════════════════════════════════════
//  TOOL CALL TYPES
// ═══════════════════════════════════════════════════════════════

/// A tool call in responses or streaming deltas.
#[derive(Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// For parsing tool calls from message history (has extra fields).
#[derive(Deserialize, Clone)]
pub struct ToolCallResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

// ═══════════════════════════════════════════════════════════════
//  SSE (STREAMING) TYPES
// ═══════════════════════════════════════════════════════════════

/// Server-Sent Event chunk for streaming responses.
#[derive(Serialize)]
pub struct SseChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<SseChoice>,
}

#[derive(Serialize)]
pub struct SseChoice {
    pub index: u32,
    pub delta: SseDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct SseDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ═══════════════════════════════════════════════════════════════
//  MODELS & HEALTH TYPES
// ═══════════════════════════════════════════════════════════════

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
}
