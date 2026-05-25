# Google AI OpenAI-Compatible Proxy

A small Go proxy that lets OpenAI-compatible AI agents talk to Google Gemini through OpenAI-style endpoints.

It sits between your agent and Google AI Studio / Gemini API, translates requests and responses, and rotates Google API keys per model after a configured number of requests.

## Features

- OpenAI-compatible routes:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- Non-streaming and streaming chat completions.
- Per-model OpenAI alias to Google model mapping.
- Per-model Google API key pools.
- Key rotation after `requests_per_api_key` requests.
- Retry on quota/transient upstream failures (`429`, `500`, `502`, `503`, `504`) by rotating to the next key.
- On selected upstream `403 PERMISSION_DENIED` / "denied access" errors, the proxy removes the bad key from the active in-memory pool and continues with the remaining keys.
- Filters Gemini/Gemma `thought` parts from returned assistant text so OpenAI-compatible clients receive cleaner content.
- Optional proxy authentication via OpenAI-style `Authorization: Bearer ...`.
- Stdlib-only Go implementation; no runtime dependencies.

## Quick start

```bash
# From this directory
export GOOGLE_API_KEYS="google-key-1,google-key-2"
export REQUESTS_PER_API_KEY=500
export PROXY_API_KEYS="local-proxy-key"
export MODEL_ALIASES="gpt-4o-mini=gemini-1.5-flash,gpt-4o=gemini-1.5-pro"

go run .
```

Point any OpenAI-compatible client/agent at:

```text
Base URL: http://localhost:8080/v1
API key:  local-proxy-key
Model:    gpt-4o-mini
```

Example curl:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Authorization: Bearer local-proxy-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "Say hello from Gemini through an OpenAI-compatible proxy."}
    ]
  }'
```

## JSON config

For production, use `CONFIG_FILE` instead of long env vars:

```bash
cp config.example.json config.json
# edit keys and limits
CONFIG_FILE=./config.json go run .
```

Config shape:

```json
{
  "listen_addr": ":8080",
  "google_base_url": "https://generativelanguage.googleapis.com/v1beta/models",
  "upstream_timeout": "120s",
  "proxy_api_keys": ["replace-with-your-openai-compatible-proxy-key"],
  "default": {
    "google_model": "gemini-1.5-flash",
    "api_keys": ["google-key-1", "google-key-2"],
    "requests_per_api_key": 1000
  },
  "models": {
    "gpt-4o-mini": {
      "google_model": "gemini-1.5-flash",
      "api_keys": ["flash-key-1", "flash-key-2"],
      "requests_per_api_key": 500
    }
  }
}
```

### Rotation semantics

For each configured OpenAI model alias, the proxy maintains an independent rotation counter.

Example:

```json
"gpt-4o-mini": {
  "google_model": "gemini-1.5-flash",
  "api_keys": ["k1", "k2"],
  "requests_per_api_key": 3
}
```

Requests use keys like this:

```text
k1, k1, k1, k2, k2, k2, k1, ...
```

- If Google returns a retryable quota/transient error, the proxy immediately rotates away from that key and retries with the next key, up to the number of configured keys for that model.
- If Google returns `403 PERMISSION_DENIED` / "denied access" for a key, the proxy removes that key from the active pool for the running process and continues with the remaining keys.

## Environment variables

- `LISTEN_ADDR`: server bind address. Default: `:8080`.
- `GOOGLE_API_BASE`: Google Gemini models base URL. Default: `https://generativelanguage.googleapis.com/v1beta/models`.
- `GOOGLE_API_KEYS`: comma-separated default Google keys.
- `REQUESTS_PER_API_KEY`: default request count per key before rotation. Default: `1000`.
- `MODEL_ALIASES`: comma-separated model mapping, e.g. `gpt-4o-mini=gemini-1.5-flash,gpt-4o=gemini-1.5-pro`.
- `PROXY_API_KEYS`: comma-separated accepted client API keys. If unset, proxy auth is disabled.
- `UPSTREAM_TIMEOUT`: HTTP timeout for Google calls, e.g. `120s`.
- `CONFIG_FILE`: optional JSON config path. Env vars can still add/override proxy keys and aliases.

## Docker

```bash
docker build -t google-ai-openai-proxy .
docker run --rm -p 8080:8080 \
  -e GOOGLE_API_KEYS="google-key-1,google-key-2" \
  -e REQUESTS_PER_API_KEY=500 \
  -e PROXY_API_KEYS="local-proxy-key" \
  -e MODEL_ALIASES="gpt-4o-mini=gemini-1.5-flash" \
  google-ai-openai-proxy
```

With config file:

```bash
docker run --rm -p 8080:8080 \
  -v "$PWD/config.json:/config.json:ro" \
  -e CONFIG_FILE=/config.json \
  google-ai-openai-proxy
```

## Supported request features

Currently supported well:

- Text chat messages.
- `system`, `developer`, `user`, `assistant`, and `tool` roles.
- `temperature`, `top_p`, and `max_tokens` mapping to Gemini generation config.
- OpenAI text content arrays (`[{"type":"text","text":"..."}]`).
- Basic OpenAI tool declaration translation into Gemini function declarations.
- Streaming text deltas as OpenAI-compatible SSE chunks.

Important limitation:

- This is focused on OpenAI-compatible chat agents. It does not yet implement OpenAI embeddings, images, audio, or the newer `/v1/responses` API.
- Tool/function call response parsing from Gemini back into OpenAI `tool_calls` is intentionally minimal in this first version. Text chat agents work best.

## Development

```bash
gofmt -w .
go test ./...
go build -o google-ai-openai-proxy .
```

## Health check

```bash
curl http://localhost:8080/healthz
```
