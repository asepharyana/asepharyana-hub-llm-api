# llm-api

OpenAI-compatible LLM inference server using `llama-cpp-2` (Rust).

**Model:** MiniCPM-V-4.6 Q4_K_M (505 MB)  
**Engine:** llama.cpp via `llama-cpp-2` crate  
**Domain:** [ai.asepharyana.my.id](https://ai.asepharyana.my.id)

## API

### `GET /health`
```json
{"status": "ok", "model": "minicpm-v-4.6-q4_k_m"}
```

### `GET /v1/models`
OpenAI-compatible model listing.

### `POST /v1/chat/completions`
OpenAI-compatible chat completions.

```bash
curl https://ai.asepharyana.my.id/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm-v-4.6",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Development

```bash
# Build
cargo build --release

# Run (with model path env var)
MODEL_PATH=/path/to/model.gguf ./target/release/llm-api

# Or use default path
./target/release/llm-api
```

## Docker

```bash
docker compose -f docker-compose.yml up -d
```

## Benchmark

| Framework | Model Size | tok/s | vs PyTorch |
|-----------|-----------|-------|------------|
| PyTorch BF16 | 2.48 GB | 0.97 | 1.0x |
| **llama.cpp Q4_K_M** | **505 MB** | **39.1** | **40.3x** 🏆 |

## Infrastructure

- Traefik router: `ai.asepharyana.my.id` → `127.0.0.1:4010`
- Network: `app-shared-net`
- Docker Compose: see `llm-api.yml`
