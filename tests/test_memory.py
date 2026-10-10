"""Tests for Persistent Episodic Memory subsystem."""

import tempfile
from pathlib import Path
import torch
from neurograph.config import MemoryConfig
from neurograph.memory import EpisodicMemory


def test_memory_insertion_and_retrieval():
    cfg = MemoryConfig(capacity=50, memory_dim=16, similarity_metric="cosine")
    mem = EpisodicMemory(cfg)

    vec1 = torch.zeros(16)
    vec1[0] = 1.0  # Unit vector along dim 0

    vec2 = torch.zeros(16)
    vec2[1] = 1.0  # Unit vector along dim 1

    mem.insert(key="north_route", content=vec1, task_context="task_A")
    mem.insert(key="east_route", content=vec2, task_context="task_A")

    # Query with vec1
    res = mem.retrieve(query=vec1, top_k=2)
    assert len(res) == 2
    assert res[0][0].key == "north_route"
    assert abs(res[0][1] - 1.0) < 1e-4


def test_memory_conflict_handling_preserves_provenance():
    """Verify that conflicting keys are NOT silently overwritten, but preserve provenance."""
    cfg = MemoryConfig(capacity=50, memory_dim=16)
    mem = EpisodicMemory(cfg)

    vec1 = torch.randn(16)
    vec2 = torch.randn(16)

    rec1 = mem.insert(key="door_key_location", content=vec1, confidence=0.8)
    assert rec1.version == 1

    rec2 = mem.insert(key="door_key_location", content=vec2, confidence=0.95)
    assert rec2.version == 2
    assert len(rec2.provenance) == 1
    assert rec2.provenance[0]["action"] == "conflict_update"
    assert rec2.provenance[0]["prev_confidence"] == 0.8
    assert rec2.provenance[0]["prev_record_id"] == rec1.id


def test_memory_update_and_delete():
    cfg = MemoryConfig(capacity=50, memory_dim=16)
    mem = EpisodicMemory(cfg)

    vec = torch.randn(16)
    rec = mem.insert(key="temporary_record", content=vec)
    assert len(mem) == 1

    # Update
    new_vec = torch.randn(16)
    ok_up = mem.update(rec.id, new_content=new_vec, new_metadata={"status": "updated"})
    assert ok_up is True
    assert mem.records[0].version == 2

    # Delete
    ok_del = mem.delete(rec.id)
    assert ok_del is True
    assert len(mem) == 0


def test_memory_serialization():
    cfg = MemoryConfig(capacity=50, memory_dim=16)
    mem = EpisodicMemory(cfg)

    vec = torch.randn(16)
    mem.insert(key="test_item", content=vec, task_context="eval")

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "memory.json"
        mem.save_json(json_path)

        mem2 = EpisodicMemory(cfg)
        mem2.load_json(json_path)

        assert len(mem2) == 1
        assert mem2.records[0].key == "test_item"
        assert torch.allclose(mem2.records[0].content, vec, atol=1e-5)
