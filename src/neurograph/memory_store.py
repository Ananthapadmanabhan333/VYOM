"""User-controlled persistent text memory for Vyom.

Design rules (see docs/ARCHITECTURE.md, "Persistent memory"):

* Memories are created **only** by explicit user action (the "Remember" form or the
  "Save to memory" action on a message). Conversations are never stored automatically.
* Every record carries provenance (source, optional conversation id) and timestamps.
* All queries are scoped by ``user_id``; there is no API that crosses users.
* Users can list, edit, delete, wipe, and disable memory. When disabled, nothing is
  retrieved or injected into chats.
* Conflicts: writing a memory whose ``key`` matches an active memory supersedes the old
  one (kept for audit, versioned). Un-keyed memories that are highly similar to an
  existing one are stored but flagged with ``possible_conflict_with`` for the user.

Embeddings: ``sentence-transformers/all-MiniLM-L6-v2`` (Apache-2.0) when it is already
present in the local Hugging Face cache, otherwise a deterministic hashed
bag-of-words embedder. The backend in use is reported by ``embedder_name``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
MAX_CONTENT_CHARS = 2000
MAX_KEY_CHARS = 120


class HashingEmbedder:
    """Deterministic lexical fallback: hashed unigrams + bigrams, L2-normalised."""

    name = "hashed-bow-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = _TOKEN_RE.findall(t.lower())
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class MiniLMEmbedder:
    """Mean-pooled all-MiniLM-L6-v2 loaded strictly from the local HF cache."""

    name = "all-MiniLM-L6-v2"
    repo = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.repo, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.repo, local_files_only=True).eval()

    def encode(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            enc = self.tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            hid = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hid * m).sum(1) / m.sum(1).clamp_min(1e-9)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.numpy().astype(np.float32)


def load_embedder(prefer: str = "auto"):
    """Return the best locally available embedder (never downloads)."""
    if prefer in ("auto", "minilm"):
        try:
            return MiniLMEmbedder()
        except Exception:
            if prefer == "minilm":
                raise
    return HashingEmbedder()


class MemoryStore:
    def __init__(self, db_path: str | Path = "data/vyom.db", embedder: Any = None):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder if embedder is not None else load_embedder()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    @property
    def embedder_name(self) -> str:
        return self.embedder.name

    # ----------------------------------------------------------------- schema
    def _migrate(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self._conn.execute("SELECT version FROM schema_version").fetchone()
            version = row[0] if row else 0
            if version < 1:
                self._conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        key TEXT,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        conversation_id TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1,
                        superseded_by TEXT,
                        possible_conflict_with TEXT,
                        embedding BLOB NOT NULL,
                        embed_model TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);
                    CREATE TABLE IF NOT EXISTS user_settings (
                        user_id TEXT PRIMARY KEY,
                        memory_enabled INTEGER NOT NULL DEFAULT 1
                    );
                    """
                )
                self._conn.execute("DELETE FROM schema_version")
                self._conn.execute("INSERT INTO schema_version (version) VALUES (1)")

    # ---------------------------------------------------------------- helpers
    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode([text])[0]

    @staticmethod
    def _row(r: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": r["id"],
            "key": r["key"],
            "content": r["content"],
            "source": r["source"],
            "conversation_id": r["conversation_id"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "version": r["version"],
            "superseded_by": r["superseded_by"],
            "possible_conflict_with": json.loads(r["possible_conflict_with"] or "[]"),
            "embed_model": r["embed_model"],
        }

    @staticmethod
    def _clean(content: str, key: Optional[str]) -> tuple[str, Optional[str]]:
        content = (content or "").strip()
        if not content:
            raise ValueError("Memory content must not be empty")
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError(f"Memory content exceeds {MAX_CONTENT_CHARS} characters")
        key = (key or "").strip() or None
        if key is not None and len(key) > MAX_KEY_CHARS:
            raise ValueError(f"Memory key exceeds {MAX_KEY_CHARS} characters")
        return content, key

    def _active_rows(self, user_id: str) -> List[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM memories WHERE user_id=? AND superseded_by IS NULL", (user_id,)
        ).fetchall()

    def _vec(self, r: sqlite3.Row) -> np.ndarray:
        if r["embed_model"] != self.embedder_name:
            return self._embed(r["content"])  # embedder changed: re-embed on the fly
        return np.frombuffer(r["embedding"], dtype=np.float32)

    # ---------------------------------------------------------------- settings
    def is_enabled(self, user_id: str) -> bool:
        r = self._conn.execute("SELECT memory_enabled FROM user_settings WHERE user_id=?", (user_id,)).fetchone()
        return True if r is None else bool(r[0])

    def set_enabled(self, user_id: str, enabled: bool) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO user_settings(user_id, memory_enabled) VALUES(?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET memory_enabled=excluded.memory_enabled",
                (user_id, int(enabled)),
            )

    # -------------------------------------------------------------------- CRUD
    def add(
        self,
        user_id: str,
        content: str,
        key: Optional[str] = None,
        source: str = "user_explicit",
        conversation_id: Optional[str] = None,
        conflict_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        content, key = self._clean(content, key)
        vec = self._embed(content if key is None else f"{key}: {content}")
        now = time.time()
        new_id = uuid.uuid4().hex
        with self._lock, self._conn:
            version = 1
            superseded: List[str] = []
            conflicts: List[str] = []
            for r in self._active_rows(user_id):
                if key is not None and r["key"] is not None and r["key"].lower() == key.lower():
                    superseded.append(r["id"])
                    version = max(version, r["version"] + 1)
                elif key is None:
                    sim = float(np.dot(vec, self._vec(r)))
                    if sim >= conflict_threshold:
                        conflicts.append(r["id"])
            for sid in superseded:
                self._conn.execute("UPDATE memories SET superseded_by=? WHERE id=?", (new_id, sid))
            self._conn.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_id, user_id, key, content, source, conversation_id, now, now, version, None,
                 json.dumps(conflicts), vec.astype(np.float32).tobytes(), self.embedder_name),
            )
        rec = self.get(user_id, new_id)
        assert rec is not None
        rec["superseded"] = superseded
        return rec

    def get(self, user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        r = self._conn.execute("SELECT * FROM memories WHERE id=? AND user_id=?", (memory_id, user_id)).fetchone()
        return self._row(r) if r else None

    def list(self, user_id: str, include_superseded: bool = False) -> List[Dict[str, Any]]:
        q = "SELECT * FROM memories WHERE user_id=?"
        if not include_superseded:
            q += " AND superseded_by IS NULL"
        q += " ORDER BY updated_at DESC"
        return [self._row(r) for r in self._conn.execute(q, (user_id,)).fetchall()]

    def update(self, user_id: str, memory_id: str, content: str, key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        content, key = self._clean(content, key)
        vec = self._embed(content if key is None else f"{key}: {content}")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE memories SET content=?, key=?, embedding=?, embed_model=?, updated_at=?, "
                "version=version+1 WHERE id=? AND user_id=?",
                (content, key, vec.tobytes(), self.embedder_name, time.time(), memory_id, user_id),
            )
        return self.get(user_id, memory_id) if cur.rowcount else None

    def delete(self, user_id: str, memory_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM memories WHERE id=? AND user_id=?", (memory_id, user_id))
        return cur.rowcount > 0

    def clear(self, user_id: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        return cur.rowcount

    # --------------------------------------------------------------- retrieval
    def search(self, user_id: str, query: str, top_k: int = 3, min_score: float = 0.0,
               respect_disabled: bool = True) -> List[Dict[str, Any]]:
        if respect_disabled and not self.is_enabled(user_id):
            return []
        query = (query or "").strip()
        if not query:
            return []
        rows = self._active_rows(user_id)
        if not rows:
            return []
        q = self._embed(query)
        scored = []
        for r in rows:
            s = float(np.dot(q, self._vec(r)))
            if s >= min_score and not math.isnan(s):
                rec = self._row(r)
                rec["score"] = round(s, 4)
                scored.append(rec)
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        self._conn.close()
