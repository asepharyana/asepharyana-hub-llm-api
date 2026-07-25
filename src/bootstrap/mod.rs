//! Application initialization and lifecycle management.

use std::sync::Arc;

use axum::Router;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

use crate::config::CONFIG;
use crate::infrastructure::llama::LlamaEngine;
use crate::presentation::router::build_router;
use crate::presentation::state::AppState;

/// The running application.
///
/// Encapsulates the router, listener, and port so that everything can be
/// created in `build()` and then served in `run()`, enabling testability.
pub struct Application {
    pub port: u16,
    router: Router,
    listener: TcpListener,
}

impl Application {
    /// Initialize all dependencies and build the application.
    ///
    /// 1. Init tracing subscriber
    /// 2. Load config (triggers LazyLock — fails fast on missing vars)
    /// 3. Load model and create LlamaEngine
    /// 4. Build router with all routes and middleware
    /// 5. Bind TCP listener
    pub async fn build() -> anyhow::Result<Self> {
        // Initialize tracing
        let env_filter = EnvFilter::new(&CONFIG.log_level);
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .init();
        tracing::info!("🚀 LLM API starting up...");

        // Load model (fail-fast)
        let engine = LlamaEngine::load().map_err(|e| {
            anyhow::anyhow!("Failed to initialize LLM engine: {e}")
        })?;
        let engine = Arc::new(engine);

        let state = Arc::new(AppState { engine });

        // Build router
        let router = build_router(state);

        // Bind listener
        let addr = format!("0.0.0.0:{}", CONFIG.server_port);
        let listener = TcpListener::bind(&addr).await?;
        tracing::info!(
            "Server listening on {}",
            listener.local_addr()?
        );

        Ok(Self {
            port: CONFIG.server_port,
            router,
            listener,
        })
    }

    /// Start serving requests.
    pub async fn run(self) -> std::io::Result<()> {
        axum::serve(self.listener, self.router.into_make_service()).await
    }
}
