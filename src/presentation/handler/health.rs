//! Health check endpoint.

use std::sync::atomic::AtomicU64;
use std::sync::LazyLock;
use std::time::Instant;

use axum::Json;

use crate::config::{CONFIG, MODEL_ID};
use crate::domain::entity::HealthResponse;

/// Process start instant — used to compute uptime for /health and /metrics.
pub static START_INSTANT: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Process start timestamp (unix seconds) — exported as a Prometheus gauge.
pub static START_TIMESTAMP: LazyLock<AtomicU64> = LazyLock::new(|| {
    AtomicU64::new(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    )
});

/// Seconds since process start.
pub fn uptime_secs() -> u64 {
    START_INSTANT.elapsed().as_secs()
}

pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        // Report the exact model id served by /v1/models (no extra suffix).
        model: MODEL_ID.into(),
        uptime_s: Some(uptime_secs()),
        n_ctx: Some(CONFIG.n_ctx),
        version: Some(env!("CARGO_PKG_VERSION").into()),
    })
}
