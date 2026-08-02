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

## Deploy (Nix + systemd)

```bash
nix build .#default --impure --option sandbox false
# GitHub Actions: nix copy ssh://vps → systemctl restart llm-api
```

> **Legacy (2026-08-02):** Docker compose dihapus dari produksi. Deploy sekarang Nix+systemd.

## Benchmark

| Framework | Model Size | tok/s | vs PyTorch |
|-----------|-----------|-------|------------|
| PyTorch BF16 | 2.48 GB | 0.97 | 1.0x |
| **llama.cpp Q4_K_M** | **505 MB** | **39.1** | **40.3x** 🏆 |

## Infrastructure

- Caddy reverse proxy: `ai.asepharyana.my.id` → `127.0.0.1:4010`
- systemd unit `llm-api`, deploy Nix via GitHub Actions
