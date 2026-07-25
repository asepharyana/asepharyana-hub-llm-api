//! Authentication middleware.
//!
//! Checks for a valid Bearer token in the Authorization header.
//! Only applied to routes that require authentication.

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;

use crate::config::CONFIG;

/// Middleware that validates the Bearer token in the Authorization header.
///
/// If `API_KEY` is not set (empty), authentication is disabled and
/// all requests pass through. If set, the middleware rejects requests
/// without a matching token.
pub async fn auth_middleware(request: Request, next: Next) -> Result<Response, (StatusCode, String)> {
    let api_key = &CONFIG.api_key;
    if api_key.is_empty() {
        return Ok(next.run(request).await);
    }

    let header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let expected = format!("Bearer {api_key}");
    if header == expected || header == api_key {
        return Ok(next.run(request).await);
    }

    Err((
        StatusCode::UNAUTHORIZED,
        "{\"error\":\"unauthorized\",\"message\":\"Invalid API key\"}".into(),
    ))
}
