# llm-api

OpenAI-compatible LLM inference server using `llama-cpp-2` (Rust).

**Model:** MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking (Q8_0)  
**Engine:** llama.cpp via `llama-cpp-2` crate  
**Domain:** [ai.asepharyana.my.id](https://ai.asepharyana.my.id)

## API

### `GET /health`
```json
{"status": "ok", "model": "minicpm5-1b-fable5-v2-thinking", "uptime_s": 1234, "n_ctx": 8192, "version": "0.1.0"}
```

### `GET /metrics`
Prometheus text exposition (no auth) — request counters, token usage, generation
latency/throughput, process uptime:

```
llm_api_requests_total            # total /v1/chat/completions
llm_api_errors_total              # errored requests
llm_api_streaming_requests_total  # stream: true requests
llm_api_aborted_requests_total    # aborted generations (client disconnect)
llm_api_prompt_tokens_total       # prompt tokens accepted
llm_api_completion_tokens_total   # tokens generated
llm_api_generation_ms_total       # generation time (ms)
llm_api_tokens_per_second         # lifetime throughput gauge
llm_api_build_info{version,model} # identity
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
| `MAX_TOKENS` | `2048` | Hard cap untuk `max_tokens` request (0 = unlimited) |

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
