#!/usr/bin/env bash
# Smoke test untuk llm-api — jalankan di VPS setelah deploy.
#
# Memverifikasi alur inti:
#   /health, /v1/models, chat non-streaming, chat streaming (SSE + [DONE]),
#   penolakan model tidak dikenal (400), dan request dengan tools.
#
# Usage:
#   ./scripts/smoke-test.sh [BASE_URL] [API_KEY]
#   BASE_URL default: http://127.0.0.1:4010
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:4010}"
API_KEY="${2:-}"
MODEL_ID="minicpm5-1b-fable5-v2-thinking"

AUTH=()
if [[ -n "$API_KEY" ]]; then
  AUTH=(-H "Authorization: Bearer $API_KEY")
fi

pass=0
fail=0
check() {
  local name="$1" ok="$2"
  if [[ "$ok" == "0" ]]; then
    echo "  PASS: $name"
    pass=$((pass + 1))
  else
    echo "  FAIL: $name"
    fail=$((fail + 1))
  fi
}

echo "== 1. Health =="
health=$(curl -sf "$BASE_URL/health") || { echo "FAIL: /health unreachable"; exit 1; }
echo "$health" | jq -e ".status == \"ok\" and .model == \"$MODEL_ID\"" >/dev/null
check "health melaporkan $MODEL_ID" $?

echo "== 2. Models =="
models=$(curl -sf "$BASE_URL/v1/models")
echo "$models" | jq -e ".data[0].id == \"$MODEL_ID\"" >/dev/null
check "models mencantumkan $MODEL_ID" $?

echo "== 3. Chat non-streaming =="
resp=$(curl -sf "${AUTH[@]}" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":64}" \
  "$BASE_URL/v1/chat/completions")
echo "$resp" | jq -e '.choices[0].message.content | type == "string"' >/dev/null
check "non-streaming mengembalikan content" $?
echo "$resp" | jq -e '.usage.total_tokens > 0' >/dev/null
check "non-streaming berisi usage" $?

echo "== 4. Chat streaming =="
stream=$(curl -sfN "${AUTH[@]}" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 5\"}],\"stream\":true,\"max_tokens\":64}" \
  "$BASE_URL/v1/chat/completions")

echo "$stream" | grep -q '\[DONE\]'
check "stream diakhiri [DONE]" $?

echo "$stream" | grep -q '"finish_reason":"stop"\|"finish_reason":"length"'
check "stream punya finish_reason" $?

# Gabungkan semua delta content untuk memastikan output tidak kosong
# dan tidak bocor markup (mis. <tool_call> / <|im_end|>).
joined=$(echo "$stream" | grep '^data: ' | sed 's/^data: //' \
  | grep -v '\[DONE\]' \
  | jq -r 'select((.choices? // []) | length > 0) | .choices[0].delta.content // empty' 2>/dev/null \
  | tr -d '\n')
if [[ -z "$joined" ]]; then
  # Sebagian output mungkin semua ber-label reasoning_content; cek keduanya.
  joined=$(echo "$stream" | grep '^data: ' | sed 's/^data: //' \
    | grep -v '\[DONE\]' \
    | jq -r 'select((.choices? // []) | length > 0) | (.choices[0].delta.content // .choices[0].delta.reasoning_content) // empty' 2>/dev/null \
    | tr -d '\n')
fi
[[ -n "$joined" ]]
check "stream menghasilkan teks" $?
if [[ -n "$joined" ]]; then
  if [[ "$joined" == *"<"* ]]; then
    check "tidak ada markup bocor di stream" 1
  else
    check "tidak ada markup bocor di stream" 0
  fi
fi

echo "== 5. Model tidak dikenal ditolak =="
status=$(curl -s -o /dev/null -w "%{http_code}" "${AUTH[@]}" -H "Content-Type: application/json" \
  -d '{"model":"minicpm-v-4.6","messages":[{"role":"user","content":"hi"}]}' \
  "$BASE_URL/v1/chat/completions")
[[ "$status" == "400" ]]
check "unknown model -> 400" $?

echo "== 6. Request dengan tools diterima =="
status2=$(curl -s -o /dev/null -w "%{http_code}" "${AUTH[@]}" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"What's the weather in Jakarta?\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}],\"max_tokens\":128}" \
  "$BASE_URL/v1/chat/completions")
[[ "$status2" == "200" ]]
check "tools request sukses (200)" $?

echo
echo "Result: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
