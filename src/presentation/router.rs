//! Axum router assembly.

use std::sync::Arc;

use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;

use super::handler::{chat, chat_ui, health, metrics, models};
use crate::presentation::middleware::auth::auth_middleware;
use crate::presentation::state::AppState;

/// Build the main application router with all routes and middleware.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Public routes (no auth)
        .route("/", get(chat_ui::chat_ui))
        .route("/health", get(health::health_check))
        .route("/metrics", get(metrics::metrics))
        .route("/v1/models", get(models::list_models))
        // Chat completions (auth-protected)
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route_layer(middleware::from_fn(auth_middleware))
        // Global middleware
        .layer(CorsLayer::permissive())
        .with_state(state)
}
