"""API tests for the Vyom server (mock LM backend, temporary database)."""

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("VYOM_MODEL", "mock")
    mp.setenv("VYOM_DB", str(tmp_path_factory.mktemp("db") / "test.db"))
    mp.setenv("VYOM_RATE_LIMIT_PER_MIN", "1000")
    from neurograph import server

    with TestClient(server.app) as c:
        yield c
    mp.undo()


def _png() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (20, 10), (200, 100, 50)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _sse(text: str):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_index_is_vyom(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "VYOM" in res.text.upper()
    assert "gemini" not in res.text.lower()


def test_health_and_status(client):
    assert client.get("/api/health").json()["status"] == "ok"
    st = client.get("/api/status").json()
    assert st["backend"] == "mock"
    assert st["workspace"]["adapter_gate"] == 0.0
    assert set(st["modes"]) == {"fast", "cognitive", "adaptive"}


def test_generate_returns_model_output_and_real_telemetry(client):
    res = client.post("/api/generate", json={"message": "graph reasoning test", "mode": "cognitive",
                                            "max_new_tokens": 5, "do_sample": False})
    assert res.status_code == 200
    data = res.json()
    assert data["sources"] == ["model"]
    assert data["stats"]["new_tokens"] <= 5
    tel = data["telemetry"]
    assert tel["active"] is True and tel["steps_executed"] == 3
    assert len(tel["step_deltas"]) == 3
    n = len(tel["adjacency"])
    assert all(abs(sum(row) - 1.0) < 1e-3 for row in tel["adjacency"]) and n > 0


def test_legacy_prompt_field_and_fast_mode(client):
    data = client.post("/api/generate", json={"prompt": "hello", "mode": "fast", "max_new_tokens": 3}).json()
    assert data["condition"] == "A"
    assert data["telemetry"]["active"] is False


def test_adaptive_mode_halts_within_budget(client):
    data = client.post("/api/generate", json={"message": "adaptive", "mode": "adaptive",
                                              "max_new_tokens": 2}).json()
    assert 1 <= data["telemetry"]["steps_executed"] <= 6


def test_curated_answer_is_labelled(client):
    data = client.post("/api/generate", json={"message": "Who are you?"}).json()
    assert data["sources"] == ["curated"]
    assert "Vyom" in data["response"]


def test_image_gives_measured_metadata_only(client):
    data = client.post("/api/generate", json={"message": "", "image_base64": _png()}).json()
    assert data["sources"] == ["image_metadata"]
    assert data["image_info"]["width"] == 20 and data["image_info"]["height"] == 10
    assert "no vision model" in data["response"].lower()


def test_invalid_inputs_rejected(client):
    assert client.post("/api/generate", json={"message": ""}).status_code == 400
    assert client.post("/api/generate", json={"message": "x", "mode": "ultra"}).status_code == 422
    assert client.post("/api/generate", json={"message": "x", "temperature": 9}).status_code == 422
    assert client.post("/api/generate", json={"message": "x" * 5000}).status_code == 422
    assert client.post("/api/generate", json={"message": "x", "image_base64": "@@notb64"}).status_code == 400
    bad_img = base64.b64encode(b"not an image").decode()
    data = client.post("/api/generate", json={"message": "x", "image_base64": bad_img}).json()
    assert "Could not decode image" in data.get("error", "")


def test_stream_emits_start_tokens_done(client):
    with client.stream("POST", "/api/chat/stream", json={"message": "stream me", "max_new_tokens": 4,
                                                          "request_id": "abc123"}) as res:
        body = "".join(res.iter_text())
    events = _sse(body)
    types = [e["type"] for e in events]
    assert types[0] == "start" and events[0]["request_id"] == "abc123"
    assert "meta" in types and types[-1] == "done"
    assert sum(1 for t in types if t == "token") >= 1


def test_cancel_unknown_request(client):
    assert client.post("/api/chat/cancel/nope").json()["cancelled"] is False


def test_memory_crud_isolation_and_toggle(client):
    alice = {"X-Vyom-User": "alice"}
    bob = {"X-Vyom-User": "bob"}
    rec = client.post("/api/memory", json={"content": "My favourite colour is teal", "key": "colour"},
                      headers=alice).json()
    assert rec["version"] == 1 and rec["source"] == "user_explicit"
    assert client.get("/api/memory", headers=bob).json()["total"] == 0
    assert client.delete(f"/api/memory/{rec['id']}", headers=bob).status_code == 404

    rec2 = client.post("/api/memory", json={"content": "My favourite colour is amber", "key": "colour"},
                       headers=alice).json()
    assert rec2["version"] == 2 and rec2["superseded"] == [rec["id"]]
    listed = client.get("/api/memory", headers=alice).json()
    assert [r["id"] for r in listed["records"]] == [rec2["id"]]

    hits = client.post("/api/memory/search", json={"query": "favourite colour"}, headers=alice).json()
    assert hits["results"][0]["id"] == rec2["id"]

    upd = client.put(f"/api/memory/{rec2['id']}", json={"content": "Favourite colour: gold"}, headers=alice).json()
    assert upd["content"] == "Favourite colour: gold" and upd["version"] == 3

    client.put("/api/memory/settings", json={"enabled": False}, headers=alice)
    data = client.post("/api/generate", json={"message": "what is my favourite colour", "max_new_tokens": 2},
                       headers=alice).json()
    assert data["memories"] == []
    client.put("/api/memory/settings", json={"enabled": True}, headers=alice)
    data = client.post("/api/generate", json={"message": "what is my favourite colour", "max_new_tokens": 2},
                       headers=alice).json()
    assert data["memories"] and data["memories"][0]["id"] == rec2["id"]

    assert client.delete("/api/memory", headers=alice).json()["deleted"] >= 1
    assert client.get("/api/memory", headers=alice).json()["total"] == 0


def test_bad_user_header_rejected(client):
    assert client.get("/api/memory", headers={"X-Vyom-User": "a b/c"}).status_code == 400


def test_plan_reports_planner_used(client):
    data = client.post("/api/plan", json={"max_steps": 20, "obstacle_density": 0.1, "seed": 7}).json()
    assert data["planner_used"] in {"world_model_search", "none"}
    assert "grid_layout" in data and "reference_bfs" in data
