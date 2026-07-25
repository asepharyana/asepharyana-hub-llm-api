//! Health check endpoint.

use axum::Json;

use crate::config::MODEL_ID;
use crate::domain::entity::HealthResponse;

pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        model: format!("{MODEL_ID}-q4_k_m"),
    })
}
