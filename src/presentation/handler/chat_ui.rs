//! Chat UI — simple web interface for interacting with the LLM.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

const HTML: &str = include_str!("chat-ui/index.html");

/// GET / — serve the chat UI page.
pub async fn chat_ui() -> Response {
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html; charset=utf-8"),
        )],
        HTML,
    )
        .into_response()
}
