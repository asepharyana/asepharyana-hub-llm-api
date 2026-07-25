//! Entry point for the LLM inference API server.
//!
//! Delegates all initialization to [`llm_api::bootstrap::Application`].

use llm_api::bootstrap::Application;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let app = Application::build().await?;
    app.run().await?;

    Ok(())
}
