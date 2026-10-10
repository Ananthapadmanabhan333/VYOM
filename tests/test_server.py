"""Tests for FastAPI server endpoints."""

from fastapi.testclient import TestClient
from neurograph.server import app


def test_server_index():
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "NeuroGraph Studio" in res.text


def test_server_generate_endpoint():
    client = TestClient(app)
    res = client.post("/api/generate", json={
        "prompt": "Test query for cognitive workspace",
        "condition": "C",
        "recurrent_steps": 2,
    })
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert "adj_matrices" in data
    assert len(data["adj_matrices"]) == 2


def test_server_plan_endpoint():
    client = TestClient(app)
    res = client.post("/api/plan", json={
        "max_steps": 10,
        "obstacle_density": 0.1,
        "seed": 42,
    })
    assert res.status_code == 200
    data = res.json()
    assert "grid_layout" in data
    assert "actions" in data
