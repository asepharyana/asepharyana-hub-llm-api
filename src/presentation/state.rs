//! Application state shared across all handlers.

use std::sync::Arc;

use crate::infrastructure::llama::LlamaEngine;

/// Shared application state injected into every handler via Axum State.
///
/// Contains the infrastructure dependencies that handlers need.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<LlamaEngine>,
}
