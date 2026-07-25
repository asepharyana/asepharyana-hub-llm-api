# ── Build stage: cargo-chef ──
FROM lukemathwalker/cargo-chef:latest-rust-1.89.0 AS chef
RUN apt-get update && apt-get install -y --no-install-recommends libclang-dev cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo chef cook --release --recipe-path recipe.json

COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release && \
    cp target/release/llm-api /app/llm-api

# ── Runtime image ──
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/sh appuser

WORKDIR /app
COPY --from=builder /app/llm-api /app/llm-api

# Model will be mounted at runtime
RUN mkdir -p /root/models/gguf && chown -R appuser:appgroup /root/models/gguf

USER appuser

EXPOSE 8080
CMD ["./llm-api"]
