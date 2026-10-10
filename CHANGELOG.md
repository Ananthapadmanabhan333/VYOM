# Changelog

## 2.0.0 — Vyom upgrade

This release turns the prototype into a working, honest local application and adds a
reproducible evaluation suite. Several problems fixed here were introduced in earlier
iterations of this project (including by AI-assisted edits) and are listed so users of older
versions know what was wrong.

### Removed (fabrication / misleading behaviour)
- `src/neurograph/gemini_chat.py`: keyword-matched templates replaced the model's output with
  canned text, and an 8-line "thought trace" was invented for every reply. Replies are now the
  base model's actual output; a few accurate self-descriptions are returned only for
  "who are you / what is Vyom / what can you do" and are labelled `curated`.
- Fake image "analysis" (only file metadata was read; brightness was computed as
  `(min+max)/2`). Images now produce measured properties only, with an explicit
  "no vision model is loaded" notice.
- Fake "▶ Run" code button in the UI.
- Model modes "Flash/Pro/Ultra/Ensemble" that were labels only. Replaced by Fast /
  Cognitive / Adaptive, each a distinct computation path.
- `/api/memory/query` used a random vector as the query. Replaced by real semantic search.
- `/api/plan` silently substituted A* when the model planner failed while the UI claimed a
  model-based plan. Planner output now states `planner_used`; BFS is shown separately as an
  oracle reference.
- Hard-coded 0.15 ms BFS latency and unsupported README claims (SmolLM2 base model,
  "zero-lint conformance"). CI lint is now enforced.
- Server silently loaded/ignored an incompatible checkpoint. Compatibility is now checked
  tensor-by-tensor and reported in `/api/status`.

### Added
- `GatedCrossAttentionAdapter` (default): zero-initialised, exact identity at init, composes
  with KV caching. The legacy prefix adapter (untrained: perplexity 121 → ~2,800) is kept as
  `adapter_mode="prefix"`.
- Workspace reads contextual LM hidden states (`context_source="hidden"`).
- Per-sample adaptive halting in the workspace; `steps_executed` and `step_deltas` telemetry.
- `NeuroGraphConfig.validate()`.
- KV-cached streaming generation with cancellation, wall-clock budget, repetition penalty,
  tokenizer EOS, measured stats.
- `engine.py` (ChatEngine), SSE endpoint `/api/chat/stream`, `/api/chat/cancel/{id}`,
  `/api/health`, `/api/status`, `/api/modes`.
- `memory_store.py`: SQLite, per-user, explicit-only memory with provenance, versioned
  conflict handling, edit/delete/wipe/disable, MiniLM or hashed embeddings.
- `world_model_training.py` (real transitions, held-out maps) and `WorldModelSearchPlanner`
  (imagined BFS + replanning, no access to the obstacle map).
- Security: optional bearer-token auth, localhost CORS, request limits, rate limiting,
  offline-by-default model loading, graceful cancellation on shutdown.
- `benchmarks/run_all.py` with raw JSON results in `benchmarks/results/`.
- Docs: rewritten README, `docs/architecture.md`, `docs/BENCHMARKS.md`, `MODEL_CARD.md`,
  `SECURITY.md`, `THIRD_PARTY_NOTICES.md`, `CONTRIBUTING.md`; `requirements.lock.txt`.
- Tests: API (streaming, validation, memory isolation/conflicts/toggle, honesty labels),
  memory store, cancellation, time budget, halting, checkpoint compatibility, planner.

### Changed
- Project renamed to `vyom` 2.0.0 in `pyproject.toml`; missing runtime deps declared
  (fastapi, uvicorn, pydantic, pillow, psutil).
- Condition D now requires a model built with static connectivity (previously it silently
  ran Condition C).

## 0.1.0
- Initial NeuroGraph-LLM research prototype.
