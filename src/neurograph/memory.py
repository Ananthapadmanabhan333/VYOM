"""Persistent Episodic Memory Subsystem for NeuroGraph-LLM."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import uuid
import time
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from neurograph.config import MemoryConfig


@dataclass
class MemoryRecord:
    """An individual episodic memory record."""
    id: str
    key: str
    content: torch.Tensor
    task_context: str
    timestamp: float
    episode_index: int
    source: str = "direct_observation"
    confidence: float = 1.0
    version: int = 1
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "content": self.content.detach().cpu().tolist(),
            "task_context": self.task_context,
            "timestamp": self.timestamp,
            "episode_index": self.episode_index,
            "source": self.source,
            "confidence": self.confidence,
            "version": self.version,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device = torch.device("cpu")) -> "MemoryRecord":
        return cls(
            id=data["id"],
            key=data["key"],
            content=torch.tensor(data["content"], dtype=torch.float32, device=device),
            task_context=data.get("task_context", "default"),
            timestamp=data.get("timestamp", time.time()),
            episode_index=data.get("episode_index", 0),
            source=data.get("source", "direct_observation"),
            confidence=data.get("confidence", 1.0),
            version=data.get("version", 1),
            provenance=data.get("provenance", []),
            metadata=data.get("metadata", {}),
        )


class EpisodicMemory:
    """Persistent Episodic Memory subsystem.

    Distinct from the dynamic cognitive workspace: operates as an addressable episodic
    store with explicit provenance, conflict tracking, and similarity-based retrieval.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.capacity = config.capacity
        self.memory_dim = config.memory_dim
        self.similarity_metric = config.similarity_metric
        self.records: List[MemoryRecord] = []
        self._key_index: Dict[str, List[int]] = {}  # key -> list of record indices

    def __len__(self) -> int:
        return len(self.records)

    def clear(self) -> None:
        self.records.clear()
        self._key_index.clear()

    def insert(
        self,
        key: str,
        content: torch.Tensor,
        task_context: str = "default",
        episode_index: int = 0,
        source: str = "direct_observation",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Insert or version a memory record.

        Avoids silently overwriting conflicting memories:
        If a memory with the same key exists, provenance is preserved and
        versioning is incremented, logging the previous confidence.
        """
        assert content.ndim == 1, f"Expected 1D content vector, got shape {content.shape}"
        assert content.shape[0] == self.memory_dim, f"Expected dim {self.memory_dim}, got {content.shape[0]}"

        rec_id = str(uuid.uuid4())[:8]
        timestamp = time.time()
        meta = metadata or {}

        existing_indices = self._key_index.get(key, [])
        version = 1
        provenance: List[Dict[str, Any]] = []

        if existing_indices:
            # Conflict handling: do not overwrite silently!
            prev_record = self.records[existing_indices[-1]]
            version = prev_record.version + 1
            provenance = list(prev_record.provenance)
            provenance.append({
                "action": "conflict_update",
                "prev_record_id": prev_record.id,
                "prev_confidence": prev_record.confidence,
                "prev_timestamp": prev_record.timestamp,
                "prev_source": prev_record.source,
                "reason": "key_collision_preserved_history",
            })

        # Evict oldest if capacity reached
        if len(self.records) >= self.capacity:
            self.records.pop(0)
            self._rebuild_index()

        record = MemoryRecord(
            id=rec_id,
            key=key,
            content=content.detach().clone(),
            task_context=task_context,
            timestamp=timestamp,
            episode_index=episode_index,
            source=source,
            confidence=confidence,
            version=version,
            provenance=provenance,
            metadata=meta,
        )

        idx = len(self.records)
        self.records.append(record)
        self._key_index.setdefault(key, []).append(idx)
        return record

    def _rebuild_index(self) -> None:
        self._key_index.clear()
        for i, rec in enumerate(self.records):
            self._key_index.setdefault(rec.key, []).append(i)

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None,
        task_context: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Retrieve top-k relevant memory records given a query embedding.

        Args:
            query (torch.Tensor): Query vector [D] or [1, D].
            top_k (Optional[int]): Number of records to return.
            task_context (Optional[str]): If provided, filters by task context.
            min_confidence (float): Minimum confidence threshold.

        Returns:
            List of (MemoryRecord, relevance_score) sorted descending by score.
        """
        if not self.records:
            return []

        q = query.squeeze()
        assert q.shape == (self.memory_dim,), f"Expected query shape ({self.memory_dim},), got {q.shape}"

        candidates = []
        for rec in self.records:
            if rec.confidence < min_confidence:
                continue
            if task_context is not None and rec.task_context != task_context:
                continue
            candidates.append(rec)

        if not candidates:
            return []

        # Stack candidate vectors [M, D]
        cand_vectors = torch.stack([r.content for r in candidates]).to(q.device)

        if self.similarity_metric == "cosine":
            q_norm = F.normalize(q.unsqueeze(0), p=2, dim=-1)
            c_norm = F.normalize(cand_vectors, p=2, dim=-1)
            scores = torch.mm(q_norm, c_norm.t()).squeeze(0)  # [M]
        else:
            scores = torch.mv(cand_vectors, q)  # [M]

        k = top_k or self.config.retrieval_top_k
        k = min(k, len(candidates))

        top_scores, top_indices = torch.topk(scores, k=k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((candidates[idx], float(score)))

        return results

    def differentiable_retrieval(
        self,
        query_batch: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable soft memory retrieval for training.

        Args:
            query_batch (torch.Tensor): [B, D]

        Returns:
            retrieved_embeddings (torch.Tensor): [B, D]
            retrieval_weights (torch.Tensor): [B, M]
        """
        B, D = query_batch.shape
        if not self.records:
            return torch.zeros_like(query_batch), torch.zeros(B, 0, device=query_batch.device)

        cand_vectors = torch.stack([r.content for r in self.records]).to(query_batch.device)  # [M, D]
        # Cosine similarity
        q_norm = F.normalize(query_batch, p=2, dim=-1)
        c_norm = F.normalize(cand_vectors, p=2, dim=-1)
        sim = torch.matmul(q_norm, c_norm.t())  # [B, M]

        weights = F.softmax(sim * 5.0, dim=-1)  # [B, M]
        retrieved = torch.matmul(weights, cand_vectors)  # [B, D]
        return retrieved, weights

    def update(self, record_id: str, new_content: Optional[torch.Tensor] = None, new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Explicitly update a record's content or metadata."""
        for rec in self.records:
            if rec.id == record_id:
                if new_content is not None:
                    rec.content = new_content.detach().clone()
                if new_metadata is not None:
                    rec.metadata.update(new_metadata)
                rec.version += 1
                rec.timestamp = time.time()
                rec.provenance.append({
                    "action": "explicit_update",
                    "timestamp": rec.timestamp,
                })
                return True
        return False

    def delete(self, record_id: str) -> bool:
        """Explicitly delete a record by ID."""
        for i, rec in enumerate(self.records):
            if rec.id == record_id:
                self.records.pop(i)
                self._rebuild_index()
                return True
        return False

    def save_json(self, path: Path | str) -> None:
        """Save memory records to local JSON storage."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self.records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: Path | str, device: torch.device = torch.device("cpu")) -> None:
        """Load memory records from local JSON storage."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.clear()
        for item in data:
            self.records.append(MemoryRecord.from_dict(item, device=device))
        self._rebuild_index()
