# IIT KGP Capstone — Track B (Qwen3-8B)

## Setup
1. Put certificates in `secrets/` (gitignored): your group's chat cert (e.g. `vm04-ca.crt`) and `vm02-ca.crt` for embeddings.
2. `cp .env.example .env` and fill in your group URL, key and cert paths.
3. `python3 check_connection.py` — confirms the chat model and embeddings both answer.

## Service limits (from the Track B guide)
- Chat: 1 active request per group key, 60/min, 8192 tokens in+out, `max_tokens` ≤ 1024. 429 = key busy; retry with backoff 2/4/8s.
- Embeddings: `bge-small-en-v1.5`, 384 dims, ≤16 texts per request, ≤512 tokens per chunk.
