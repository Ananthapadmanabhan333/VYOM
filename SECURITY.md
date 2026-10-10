# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 2.0.x   | yes       |
| < 2.0   | no        |

## Threat model and defaults

Vyom is designed to run **on your own machine**.

* The server binds to `127.0.0.1:8000` by default (`VYOM_HOST`, `VYOM_PORT`). Do not expose
  it to a network without enabling authentication and putting it behind TLS.
* **Authentication.** If `VYOM_API_TOKENS="tokenA:alice,tokenB:bob"` is set, every `/api/*`
  route except `/api/health` requires `Authorization: Bearer <token>`, and the user identity
  comes from the token. Without it, the server is in single-user local mode: the
  `X-Vyom-User` header only separates memory namespaces and is **not** authentication.
* **CORS** is restricted to `http://127.0.0.1:8000` and `http://localhost:8000`
  (`VYOM_CORS_ORIGINS`); credentials are not allowed.
* **Input limits**: message ≤ 4,000 chars, history ≤ 40 turns, `max_new_tokens` ≤ 256,
  image ≤ ~6 MB (base64 ≤ 8 MB) and ≤ 40 MP (PIL decompression-bomb guard), memory content
  ≤ 2,000 chars; numeric parameters are range-checked (HTTP 422 otherwise).
* **Rate limiting**: chat endpoints allow `VYOM_RATE_LIMIT_PER_MIN` (default 30) requests per
  user+IP per minute (in-memory, single process).
* **No downloads at runtime**: Hugging Face is forced offline unless
  `VYOM_ALLOW_DOWNLOAD=1`.
* **No tool execution**: the model cannot run code, shell commands, or network requests. The
  planner acts only inside the synthetic GridWorld.
* **Data**: conversation history stays in the browser's `localStorage`. Memories are stored in
  `data/vyom.db` only when the user explicitly saves them; `data/` is git-ignored. Users can
  delete individual memories or wipe all of them.
* **Secrets**: no secrets are required. Put tokens in environment variables, never in the
  repository.

## Reporting a vulnerability

Please open a private GitHub security advisory for the repository (or contact the
maintainers privately) with a description, reproduction steps and impact. Do not open public
issues for vulnerabilities. We aim to acknowledge reports within 7 days.
