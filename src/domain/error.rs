//! Domain-level error types.
//!
//! Framework-agnostic errors that can be mapped to HTTP errors
//! at the presentation layer.

use thiserror::Error;

/// Errors originating from LLM inference operations.
#[derive(Error, Debug)]
pub enum LlmError {
    /// Invalid request parameters
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Model or inference error
    #[error("Model error: {0}")]
    Model(String),

    /// Authentication failure
    #[error("Authentication failed")]
    Unauthorized,

    /// Internal/unexpected error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<String> for LlmError {
    fn from(s: String) -> Self {
        LlmError::Internal(s)
    }
}

impl From<&str> for LlmError {
    fn from(s: &str) -> Self {
        LlmError::Internal(s.to_string())
    }
}
