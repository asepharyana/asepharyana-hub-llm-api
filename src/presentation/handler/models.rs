//! Models list endpoint.

use axum::Json;
use chrono::Utc;

use crate::config::MODEL_ID;
use crate::domain::entity::{ModelInfo, ModelsResponse};

pub async fn list_models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".into(),
        data: vec![ModelInfo {
            id: MODEL_ID.into(),
            object: "model".into(),
            created: Utc::now().timestamp(),
            owned_by: "asepharyana".into(),
        }],
    })
}
