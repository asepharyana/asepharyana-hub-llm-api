# llm-api

OpenAI-compatible LLM inference server using `llama-cpp-2` (Rust).

**Model:** MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking (Q8_0)  
**Engine:** llama.cpp via `llama-cpp-2` crate  
**Domain:** [ai.asepharyana.my.id](https://ai.asepharyana.my.id)

## API

### `GET /health`
```json
{"status": "ok", "model": "minicpm5-1b-fable5-v2-thinking"}
```

### `GET /v1/models`
OpenAI-compatible model listing.

### `POST /v1/chat/completions`
OpenAI-compatible chat completions.

The server serves a single model and rejects unknown model ids with `400`:

```bash
curl https://ai.asepharyana.my.id/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm5-1b-fable5-v2-thinking",
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

### Environment variables

| Var | Default | Description |
|-----|---------|-------------|
| `MODEL_PATH` | `/models/MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-Q8_0.gguf` | GGUF model file |
| `API_KEY` | *(empty = auth off)* | Bearer token required on `/v1/chat/completions` |
| `SERVER_PORT` | `4010` | Listen port |
| `RUST_LOG` | `info` | Log level |
| `N_CTX` / `N_BATCH` / `N_THREADS` | `8192` / `512` / `4` | llama.cpp context/batch/threads |

### Smoke test (setelah deploy)

```bash
./scripts/smoke-test.sh http://127.0.0.1:4010        # tanpa auth
./scripts/smoke-test.sh https://ai.asepharyana.my.id "$API_KEY"
```

## Deploy (Nix + systemd)

```bash
nix build .#default --impure --option sandbox false
# GitHub Actions: nix copy ssh://vps → systemctl restart llm-api
```

> **Legacy (2026-08-02):** Docker compose dihapus dari produksi. Deploy sekarang Nix+systemd.

## Benchmark

> *Historic* (MiniCPM-V-4.6). Kept for reference; numbers predate the current
> MiniCPM5-1B Thinking model.

| Framework | Model Size | tok/s | vs PyTorch |
|-----------|-----------|-------|------------|
| PyTorch BF16 | 2.48 GB | 0.97 | 1.0x |
| **llama.cpp Q4_K_M** | **505 MB** | **39.1** | **40.3x** 🏆 |

## Infrastructure

- Caddy reverse proxy: `ai.asepharyana.my.id` → `127.0.0.1:4010`
- systemd unit `llm-api`, deploy Nix via GitHub Actions
