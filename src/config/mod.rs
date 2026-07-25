//! Type-safe application configuration.
//!
//! Loads configuration from environment variables at startup with fail-fast behavior.

use std::sync::LazyLock;

const DEFAULT_MODEL_PATH: &str = "/models/MiniCPM-V-4.6-Q4_K_M.gguf";
pub const MODEL_ID: &str = "minicpm-v-4.6";

/// Application configuration loaded at startup from environment variables.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Path to the GGUF model file
    pub model_path: String,

    /// API key for authentication (empty = disabled)
    pub api_key: String,

    /// Server port to bind to
    pub server_port: u16,

    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,

    /// LLM context size (n_ctx)
    pub n_ctx: u32,

    /// LLM batch size (n_batch)
    pub n_batch: u32,

    /// Number of CPU threads for inference
    pub n_threads: i32,
}

impl AppConfig {
    /// Load configuration from environment variables.
    pub fn load() -> Self {
        Self {
            model_path: std::env::var("MODEL_PATH")
                .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string()),
            api_key: std::env::var("API_KEY").unwrap_or_default(),
            server_port: std::env::var("SERVER_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8080),
            log_level: std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            n_ctx: 2048,
            n_batch: 512,
            n_threads: 4,
        }
    }
}

/// Global configuration instance, loaded once at startup.
pub static CONFIG: LazyLock<AppConfig> = LazyLock::new(|| {
    let config = AppConfig::load();
    tracing::info!("Configuration loaded: model={:?}", config.model_path);
    config
});
