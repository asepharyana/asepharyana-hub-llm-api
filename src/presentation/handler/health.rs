//! Health check endpoint.

use axum::Json;

use crate::config::MODEL_ID;
use crate::domain::entity::HealthResponse;

pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        // Report the exact model id served by /v1/models (no extra suffix).
        model: MODEL_ID.into(),
    })
}
