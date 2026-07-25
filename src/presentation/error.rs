//! Application-level HTTP error handling.
//!
//! Maps domain errors into HTTP responses.

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;
use thiserror::Error;

use crate::domain::LlmError;

/// Top-level HTTP error returned by all API handlers.
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Authentication failed")]
    Unauthorized,

    #[error("Model error: {0}")]
    LlmError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

// ── From impls — convert domain errors to AppError ──

impl From<LlmError> for AppError {
    fn from(err: LlmError) -> Self {
        match err {
            LlmError::InvalidRequest(msg) => AppError::BadRequest(msg),
            LlmError::Model(msg) => AppError::LlmError(msg),
            LlmError::Unauthorized => AppError::Unauthorized,
            LlmError::Internal(msg) => AppError::Internal(msg),
        }
    }
}

impl From<String> for AppError {
    fn from(s: String) -> Self {
        AppError::Internal(s)
    }
}

impl From<&str> for AppError {
    fn from(s: &str) -> Self {
        AppError::Internal(s.to_string())
    }
}

// ── Error Response DTO ──

#[derive(Serialize)]
struct ErrorBody {
    error: String,
    message: String,
}

// ── IntoResponse — render AppError as HTTP response ──

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_msg, detail_msg) = match &self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg.as_str()),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, "not_found", msg.as_str()),
            AppError::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized", "Invalid API key"),
            AppError::LlmError(_msg) => {
                tracing::error!(%self, "LLM error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "Internal server error",
                )
            }
            AppError::Internal(_) => {
                tracing::error!(%self, "Internal error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "Internal server error",
                )
            }
        };

        let body = Json(ErrorBody {
            error: error_msg.to_string(),
            message: detail_msg.to_string(),
        });

        (status, body).into_response()
    }
}
