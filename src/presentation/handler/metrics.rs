//! Prometheus metrics endpoint.
//!
//! Exports process-level metrics (CPU, RSS) plus request counters, token
//! usage and generation latencies. Implemented with `std` only — no external
//! metrics dependency — so the deploy stays dependency-free.
//!
//! Scrape config (VPS): `prometheus.yml` file_sd targets llm-api at
//! `/metrics` with default `__metrics_path__`.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

use super::health::{uptime_secs, START_INSTANT, START_TIMESTAMP};

// ── Atomic counters (updated by the chat handlers) ──

/// Total /v1/chat/completions requests received.
pub static REQUESTS_TOTAL: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(0));
/// Requests that ended in an error (any 4xx/5xx).
pub static ERRORS_TOTAL: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(0));
/// Requests that streamed (`stream: true`).
pub static STREAMING_TOTAL: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(0));
/// Prompt tokens accepted across all requests.
pub static PROMPT_TOKENS_TOTAL: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(0));
/// Completion tokens generated across all requests.
pub static COMPLETION_TOKENS_TOTAL: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(0));
/// Generation time spent across all requests, in milliseconds.
pub static GENERATION_MS_TOTAL: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(0));
/// Requests whose generation was aborted early (client disconnect).
pub static ABORTED_TOTAL: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(0));

// ── Public helpers used by handlers ──

pub fn count_request(streaming: bool) {
    REQUESTS_TOTAL.fetch_add(1, Ordering::Relaxed);
    if streaming {
        STREAMING_TOTAL.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn count_error() {
    ERRORS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn count_aborted() {
    ABORTED_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_tokens(prompt_tokens: u32, completion_tokens: u32, duration_ms: u64) {
    PROMPT_TOKENS_TOTAL.fetch_add(prompt_tokens as u64, Ordering::Relaxed);
    COMPLETION_TOKENS_TOTAL.fetch_add(completion_tokens as u64, Ordering::Relaxed);
    GENERATION_MS_TOTAL.fetch_add(duration_ms, Ordering::Relaxed);
}

fn f(field: &mut String, name: &str, value: impl std::fmt::Display) {
    let _ = writeln!(field, "{name} {value}");
}

/// GET /metrics — Prometheus text exposition format.
pub async fn metrics() -> Response {
    let uptime = uptime_secs();
    let start_ts = START_TIMESTAMP.load(Ordering::Relaxed);

    // Per-second rates over the process lifetime.
    let total = REQUESTS_TOTAL.load(Ordering::Relaxed);
    let errors = ERRORS_TOTAL.load(Ordering::Relaxed);
    let streaming = STREAMING_TOTAL.load(Ordering::Relaxed);
    let aborted = ABORTED_TOTAL.load(Ordering::Relaxed);
    let prompt_tokens = PROMPT_TOKENS_TOTAL.load(Ordering::Relaxed);
    let completion_tokens = COMPLETION_TOKENS_TOTAL.load(Ordering::Relaxed);
    let gen_ms = GENERATION_MS_TOTAL.load(Ordering::Relaxed);

    let rps = if uptime > 0 {
        total as f64 / uptime as f64
    } else {
        0.0
    };
    let tok_per_s = if gen_ms > 0 {
        completion_tokens as f64 / (gen_ms as f64 / 1000.0)
    } else {
        0.0
    };
    let avg_ms = if total > 0 {
        gen_ms as f64 / total as f64
    } else {
        0.0
    };

    let mut body = String::with_capacity(2048);
    body.push_str("# HELP llm_api_requests_total Total /v1/chat/completions requests received.\n");
    body.push_str("# TYPE llm_api_requests_total counter\n");
    f(&mut body, "llm_api_requests_total", total);
    body.push_str("# HELP llm_api_errors_total Requests that ended in an error.\n");
    body.push_str("# TYPE llm_api_errors_total counter\n");
    f(&mut body, "llm_api_errors_total", errors);
    body.push_str("# HELP llm_api_streaming_requests_total Requests that streamed.\n");
    body.push_str("# TYPE llm_api_streaming_requests_total counter\n");
    f(&mut body, "llm_api_streaming_requests_total", streaming);
    body.push_str(
        "# HELP llm_api_aborted_requests_total Generations aborted early (client disconnect).\n",
    );
    body.push_str("# TYPE llm_api_aborted_requests_total counter\n");
    f(&mut body, "llm_api_aborted_requests_total", aborted);
    body.push_str("# HELP llm_api_prompt_tokens_total Prompt tokens accepted.\n");
    body.push_str("# TYPE llm_api_prompt_tokens_total counter\n");
    f(&mut body, "llm_api_prompt_tokens_total", prompt_tokens);
    body.push_str("# HELP llm_api_completion_tokens_total Completion tokens generated.\n");
    body.push_str("# TYPE llm_api_completion_tokens_total counter\n");
    f(
        &mut body,
        "llm_api_completion_tokens_total",
        completion_tokens,
    );
    body.push_str("# HELP llm_api_generation_ms_total Generation time in milliseconds.\n");
    body.push_str("# TYPE llm_api_generation_ms_total counter\n");
    f(&mut body, "llm_api_generation_ms_total", gen_ms);
    body.push_str("# HELP llm_api_requests_per_second Lifetime request rate.\n");
    body.push_str("# TYPE llm_api_requests_per_second gauge\n");
    f(
        &mut body,
        "llm_api_requests_per_second",
        format!("{rps:.3}"),
    );
    body.push_str("# HELP llm_api_tokens_per_second Lifetime generation throughput.\n");
    body.push_str("# TYPE llm_api_tokens_per_second gauge\n");
    f(
        &mut body,
        "llm_api_tokens_per_second",
        format!("{tok_per_s:.3}"),
    );
    body.push_str(
        "# HELP llm_api_average_generation_ms Average generation duration per request.\n",
    );
    body.push_str("# TYPE llm_api_average_generation_ms gauge\n");
    f(
        &mut body,
        "llm_api_average_generation_ms",
        format!("{avg_ms:.1}"),
    );
    body.push_str("# HELP llm_api_uptime_seconds Server process uptime.\n");
    body.push_str("# TYPE llm_api_uptime_seconds gauge\n");
    f(&mut body, "llm_api_uptime_seconds", uptime);
    body.push_str("# HELP llm_api_start_time_seconds Process start time (unix).\n");
    body.push_str("# TYPE llm_api_start_time_seconds gauge\n");
    f(&mut body, "llm_api_start_time_seconds", start_ts);

    // Engine identity (helpful when multiple model servers exist).
    body.push_str("# HELP llm_api_build_info Build information.\n");
    body.push_str("# TYPE llm_api_build_info gauge\n");
    let _ = writeln!(
        body,
        "llm_api_build_info{{version=\"{}\",model=\"{}\"}} 1",
        env!("CARGO_PKG_VERSION"),
        crate::config::MODEL_ID
    );

    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        )],
        body,
    )
        .into_response()
}

/// Snapshot helper used by tests to inspect counters.
pub fn snapshot() -> (u64, u64, u64) {
    (
        REQUESTS_TOTAL.load(Ordering::Relaxed),
        COMPLETION_TOKENS_TOTAL.load(Ordering::Relaxed),
        GENERATION_MS_TOTAL.load(Ordering::Relaxed),
    )
}

/// Used by tests to verify the monotonic clock source is live.
pub fn start_instant() -> &'static Instant {
    &START_INSTANT
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset() {
        for c in [
            &REQUESTS_TOTAL,
            &ERRORS_TOTAL,
            &STREAMING_TOTAL,
            &PROMPT_TOKENS_TOTAL,
            &COMPLETION_TOKENS_TOTAL,
            &GENERATION_MS_TOTAL,
            &ABORTED_TOTAL,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }

    #[test]
    fn counters_accumulate() {
        reset();
        count_request(true);
        count_request(false);
        count_error();
        count_aborted();
        record_tokens(100, 250, 5000);

        assert_eq!(REQUESTS_TOTAL.load(Ordering::Relaxed), 2);
        assert_eq!(STREAMING_TOTAL.load(Ordering::Relaxed), 1);
        assert_eq!(ERRORS_TOTAL.load(Ordering::Relaxed), 1);
        assert_eq!(ABORTED_TOTAL.load(Ordering::Relaxed), 1);
        assert_eq!(PROMPT_TOKENS_TOTAL.load(Ordering::Relaxed), 100);
        assert_eq!(COMPLETION_TOKENS_TOTAL.load(Ordering::Relaxed), 250);
        assert_eq!(GENERATION_MS_TOTAL.load(Ordering::Relaxed), 5000);
    }

    #[test]
    fn metrics_body_has_prometheus_shape() {
        reset();
        count_request(true);
        record_tokens(10, 20, 1000);

        let response = futures::executor::block_on(metrics());
        let bytes =
            futures::executor::block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
                .expect("read body");
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(text.contains("# TYPE llm_api_requests_total counter"));
        assert!(text.contains("llm_api_requests_total 1"));
        assert!(text.contains("llm_api_completion_tokens_total 20"));
        assert!(text.contains("llm_api_build_info{version="));
        assert!(text.contains("llm_api_uptime_seconds "));
    }

    #[test]
    fn uptime_is_monotonic() {
        let a = uptime_secs();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let b = uptime_secs();
        assert!(b >= a);
    }
}
