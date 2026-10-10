"""Vyom FastAPI server.

Run:  PYTHONPATH="src;." python -m neurograph.server   (binds 127.0.0.1:8000)

Security model (see SECURITY.md):
* Binds to localhost by default; CORS is restricted to local origins.
* If ``VYOM_API_TOKENS`` is set ("token:user,token2:user2"), every /api route except
  /api/health requires ``Authorization: Bearer <token>`` and the user id comes from the
  token. Otherwise the server runs in single-machine mode: the user id is taken from the
  ``X-Vyom-User`` header (default "local"), which is NOT an authentication mechanism.
* Request sizes and parameter ranges are validated; chat endpoints are rate limited.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from neurograph.engine import MODES, ChatEngine, ChatRequestData, default_world_model_path
from neurograph.memory_store import MAX_CONTENT_CHARS, MAX_KEY_CHARS, MemoryStore

log = logging.getLogger("vyom")
logging.basicConfig(level=os.environ.get("VYOM_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

WEB_DIR = Path(__file__).parent / "web"
MAX_IMAGE_B64_CHARS = 8 * 1024 * 1024  # ~6 MB decoded
RATE_LIMIT_PER_MIN = int(os.environ.get("VYOM_RATE_LIMIT_PER_MIN", "30"))


class State:
    engine: Optional[ChatEngine] = None
    memory: Optional[MemoryStore] = None
    world_model: Any = None
    world_model_report: Dict[str, Any] = {}
    cancel_events: Dict[str, threading.Event] = {}
    rate: Dict[str, Deque[float]] = defaultdict(deque)
    started_at: float = time.time()


S = State()


def _parse_tokens() -> Dict[str, str]:
    raw = os.environ.get("VYOM_API_TOKENS", "").strip()
    out: Dict[str, str] = {}
    for part in filter(None, (p.strip() for p in raw.split(","))):
        tok, _, user = part.partition(":")
        if tok and user:
            out[tok] = user
    return out


TOKENS = _parse_tokens()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    S.engine = ChatEngine()
    S.memory = MemoryStore(os.environ.get("VYOM_DB", "data/vyom.db"))
    wm_path = default_world_model_path()
    if wm_path.exists():
        from neurograph.world_model_training import load_world_model
        S.world_model, S.world_model_report = load_world_model(wm_path)
    log.info("Vyom ready: model=%s backend=%s embedder=%s world_model=%s",
             S.engine.model_name, S.engine.backend, S.memory.embedder_name, bool(S.world_model))
    yield
    for ev in S.cancel_events.values():  # graceful shutdown: stop in-flight generations
        ev.set()
    S.memory.close()


app = FastAPI(title="Vyom — Toward Machine Cognition", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get(
        "VYOM_CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",") if o],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Vyom-User"],
)


# ------------------------------------------------------------------ dependencies
def current_user(request: Request) -> str:
    if TOKENS:
        auth = request.headers.get("authorization", "")
        tok = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        user = TOKENS.get(tok)
        if not user:
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
        return user
    user = request.headers.get("x-vyom-user", "local").strip() or "local"
    if len(user) > 64 or not all(c.isalnum() or c in "-_." for c in user):
        raise HTTPException(status_code=400, detail="Invalid X-Vyom-User header")
    return user


def rate_limited(request: Request, user: str = Depends(current_user)) -> str:
    key = f"{user}|{request.client.host if request.client else '-'}"
    now = time.time()
    q = S.rate[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded; try again shortly")
    q.append(now)
    return user


# ---------------------------------------------------------------------- schemas
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field("", max_length=4000)
    history: List[Turn] = Field(default_factory=list, max_length=40)
    mode: Literal["fast", "cognitive", "adaptive"] = "fast"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    max_new_tokens: int = Field(64, ge=1, le=256)
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0)
    do_sample: bool = True
    use_memory: bool = True
    image_base64: Optional[str] = Field(None, max_length=MAX_IMAGE_B64_CHARS)
    request_id: Optional[str] = Field(None, max_length=64)
    seed: Optional[int] = None
    prompt: Optional[str] = Field(None, max_length=4000, description="Deprecated alias of `message`.")

    def model_post_init(self, _ctx: Any) -> None:
        if not self.message and self.prompt:
            self.message = self.prompt


class MemoryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    key: Optional[str] = Field(None, max_length=MAX_KEY_CHARS)
    source: Literal["user_explicit", "chat_save"] = "user_explicit"
    conversation_id: Optional[str] = Field(None, max_length=64)


class MemoryUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    key: Optional[str] = Field(None, max_length=MAX_KEY_CHARS)


class MemorySearch(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(3, ge=1, le=20)


class MemorySettings(BaseModel):
    enabled: bool


class PlanRequest(BaseModel):
    max_steps: int = Field(30, ge=1, le=100)
    obstacle_density: float = Field(0.2, ge=0.0, le=0.4)
    seed: int = Field(42, ge=0, le=10_000_000)


# ------------------------------------------------------------------------ pages
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/vyom_logo.png")
async def logo():
    path = WEB_DIR / "vyom_logo.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/health")
async def health():
    return {"status": "ok" if S.engine is not None else "starting",
            "uptime_s": round(time.time() - S.started_at, 1)}


@app.get("/api/status")
async def status(user: str = Depends(current_user)):
    import psutil

    st = S.engine.status()
    st.update(
        memory_store={"embedder": S.memory.embedder_name, "enabled_for_user": S.memory.is_enabled(user)},
        world_model={
            "loaded": S.world_model is not None,
            "path": str(default_world_model_path()),
            "held_out_position_accuracy": (S.world_model_report.get("test") or {}).get("position_accuracy"),
        },
        auth="bearer_tokens" if TOKENS else "local_single_user (X-Vyom-User header, not authenticated)",
        user=user,
        rss_mb=round(psutil.Process().memory_info().rss / 1e6, 1),
    )
    return st


# ------------------------------------------------------------------------- chat
def _decode_image(b64: Optional[str]) -> Optional[bytes]:
    if not b64:
        return None
    raw = b64.split(",", 1)[1] if b64.startswith("data:") else b64
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64")


def _to_engine_request(req: ChatRequest, user: str) -> ChatRequestData:
    memories = []
    if req.use_memory and req.message.strip():
        memories = S.memory.search(user, req.message, top_k=3, min_score=0.35)
    return ChatRequestData(
        message=req.message, history=[t.model_dump() for t in req.history], mode=req.mode,
        temperature=req.temperature, top_p=req.top_p, max_new_tokens=req.max_new_tokens,
        repetition_penalty=req.repetition_penalty, do_sample=req.do_sample,
        memories=[{"id": m["id"], "content": m["content"], "score": m["score"]} for m in memories],
        image_bytes=_decode_image(req.image_base64), seed=req.seed,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, user: str = Depends(rate_limited)):
    if not req.message.strip() and not req.image_base64:
        raise HTTPException(status_code=400, detail="message or image required")
    data = _to_engine_request(req, user)
    rid = req.request_id or uuid.uuid4().hex
    cancel = threading.Event()
    S.cancel_events[rid] = cancel
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            for ev in S.engine.stream_chat(data, cancel):
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def events():
        yield f"data: {json.dumps({'type': 'start', 'request_id': rid})}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    cancel.set()
                ev = await queue.get()
                if ev is None:
                    break
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            cancel.set()
            S.cancel_events.pop(rid, None)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Request-Id": rid})


@app.post("/api/chat/cancel/{request_id}")
async def chat_cancel(request_id: str, _user: str = Depends(current_user)):
    ev = S.cancel_events.get(request_id)
    if ev is None:
        return {"cancelled": False, "reason": "no active request with that id"}
    ev.set()
    return {"cancelled": True}


@app.post("/api/generate")
async def generate(req: ChatRequest, user: str = Depends(rate_limited)):
    """Non-streaming chat (kept for API compatibility)."""
    if not req.message.strip() and not req.image_base64:
        raise HTTPException(status_code=400, detail="message or image required")
    data = _to_engine_request(req, user)
    return await asyncio.to_thread(S.engine.chat, data)


@app.get("/api/modes")
async def modes():
    return {k: {"label": v["label"], "description": v["description"], "condition": v["condition"]}
            for k, v in MODES.items()}


# ----------------------------------------------------------------------- memory
# NOTE: /api/memory/settings must be registered before /api/memory/{memory_id}.
@app.get("/api/memory/settings")
async def memory_settings_get(user: str = Depends(current_user)):
    return {"enabled": S.memory.is_enabled(user)}


@app.put("/api/memory/settings")
async def memory_settings_put(body: MemorySettings, user: str = Depends(current_user)):
    S.memory.set_enabled(user, body.enabled)
    return {"enabled": body.enabled}


@app.get("/api/memory")
async def memory_list(include_superseded: bool = False, user: str = Depends(current_user)):
    recs = S.memory.list(user, include_superseded=include_superseded)
    return {"enabled": S.memory.is_enabled(user), "embedder": S.memory.embedder_name,
            "total": len(recs), "records": recs}


@app.post("/api/memory")
async def memory_create(body: MemoryCreate, user: str = Depends(current_user)):
    try:
        return S.memory.add(user, body.content, key=body.key, source=body.source,
                            conversation_id=body.conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/memory/{memory_id}")
async def memory_update(memory_id: str, body: MemoryUpdate, user: str = Depends(current_user)):
    try:
        rec = S.memory.update(user, memory_id, body.content, key=body.key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if rec is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return rec


@app.delete("/api/memory/{memory_id}")
async def memory_delete(memory_id: str, user: str = Depends(current_user)):
    if not S.memory.delete(user, memory_id):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": memory_id}


@app.delete("/api/memory")
async def memory_clear(user: str = Depends(current_user)):
    return {"deleted": S.memory.clear(user)}


@app.post("/api/memory/search")
async def memory_search(body: MemorySearch, user: str = Depends(current_user)):
    return {"query": body.query, "enabled": S.memory.is_enabled(user), "embedder": S.memory.embedder_name,
            "results": S.memory.search(user, body.query, top_k=body.top_k, respect_disabled=False)}


# ---------------------------------------------------------------------- planner
@app.post("/api/plan")
async def plan(req: PlanRequest, _user: str = Depends(current_user)):
    from environments.gridworld import GridWorld
    from neurograph.planner import SearchBasedPlanner, WorldModelSearchPlanner

    def run() -> Dict[str, Any]:
        state_dim = S.world_model.state_dim if S.world_model is not None else 32
        env = GridWorld.generate_random_map(6, 6, req.obstacle_density, seed=req.seed, state_dim=state_dim)
        env.max_steps = req.max_steps
        layout = [["S" if (r, c) == env.start_pos else "G" if (r, c) == env.goal_pos
                   else "X" if (r, c) in env.obstacles else "." for c in range(env.width)]
                  for r in range(env.height)]
        t0 = time.perf_counter()
        bfs = SearchBasedPlanner(GridWorld.ACTION_MAP).plan_bfs(
            env.start_pos, env.goal_pos, env.is_valid_coord, max_depth=req.max_steps)
        bfs_ms = (time.perf_counter() - t0) * 1000
        out: Dict[str, Any] = {
            "grid_layout": layout, "start": env.start_pos, "goal": env.goal_pos,
            "reference_bfs": {"steps": len(bfs), "found": bool(bfs), "latency_ms": round(bfs_ms, 3),
                              "note": "BFS reads the full obstacle map (oracle reference)."},
        }
        if S.world_model is None:
            out.update(planner_used="none", success=False, steps=0, trajectory=[env.start_pos],
                       actions=[], error="No trained world model found; run "
                       "`python benchmarks/run_all.py --only world_model` to create one.")
            return out
        t0 = time.perf_counter()
        res = WorldModelSearchPlanner(S.world_model, env.height, env.width).execute_plan(env, req.max_steps)
        out.update(
            planner_used="world_model_search",
            success=res["success"], steps=res["steps"], invalid_actions=res["invalid_actions"],
            trajectory=res["trajectory"], actions=[GridWorld.ACTION_NAMES[a] for a in res["actions"]],
            prediction_accuracy=res["prediction_accuracy"], world_model_calls=res["world_model_calls"],
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return out

    return await asyncio.to_thread(run)


def start_server(host: Optional[str] = None, port: Optional[int] = None):
    import uvicorn

    uvicorn.run(app, host=host or os.environ.get("VYOM_HOST", "127.0.0.1"),
                port=port or int(os.environ.get("VYOM_PORT", "8000")), reload=False)


if __name__ == "__main__":
    start_server()
