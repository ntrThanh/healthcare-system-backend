"""
Integration tests for the AI Server.

Run with:
    pytest tests/test_api.py -v

The tests mock LLM + RAG so no actual model is needed.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ── Patch heavy services before importing app ─────────────────────────────────
@pytest.fixture(autouse=True, scope="session")
def mock_services():
    with (
        patch("app.services.llm_service.LLMService.load"),
        patch("app.services.rag_service.RAGService.load"),
        patch(
            "app.services.llm_service.LLMService.generate",
            return_value=("Mocked LLM answer.", 42.0),
        ),
        patch(
            "app.services.llm_service.LLMService.is_loaded",
            new_callable=lambda: property(lambda self: True),
        ),
        patch(
            "app.services.rag_service.RAGService.build_context",
            return_value=("### Context:\nDoc 1 text", ["doc-1"]),
        ),
        patch(
            "app.services.rag_service.RAGService.is_loaded",
            new_callable=lambda: property(lambda self: True),
        ),
        patch("app.services.summarizer.SummarizerService.maybe_summarize"),
    ):
        yield


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


VALID_KEY = "key-webapp-abc123"
HEADERS = {"X-API-Key": VALID_KEY}
USER_ID = "test-user-001"


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["llm_loaded"] is True
    assert data["rag_loaded"] is True


# ── Auth & Rate limiting ──────────────────────────────────────────────────────

def test_missing_api_key_returns_401(client):
    r = client.post("/api/v1/chat", json={"user_id": USER_ID, "message": "hello"})
    assert r.status_code == 401


def test_invalid_api_key_returns_401(client):
    r = client.post(
        "/api/v1/chat",
        json={"user_id": USER_ID, "message": "hello"},
        headers={"X-API-Key": "bad-key"},
    )
    assert r.status_code == 401


# ── Chat REST ─────────────────────────────────────────────────────────────────

def test_chat_creates_session(client):
    r = client.post(
        "/api/v1/chat",
        json={"user_id": USER_ID, "message": "What is RAG?"},
        headers=HEADERS,
    )
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert "answer" in data
    assert data["answer"] == "Mocked LLM answer."
    assert data["rag_sources"] == ["doc-1"]


def test_chat_reuses_session(client):
    # First message to get a session
    r1 = client.post(
        "/api/v1/chat",
        json={"user_id": USER_ID, "message": "First message"},
        headers=HEADERS,
    )
    session_id = r1.json()["session_id"]

    # Second message reuses that session
    r2 = client.post(
        "/api/v1/chat",
        json={"user_id": USER_ID, "session_id": session_id, "message": "Follow-up"},
        headers=HEADERS,
    )
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id


def test_chat_with_context(client):
    r = client.post(
        "/api/v1/chat",
        json={
            "user_id": USER_ID,
            "message": "Hello",
            "context": "You are a helpful Vietnamese assistant.",
        },
        headers=HEADERS,
    )
    assert r.status_code == 200


# ── Sessions ──────────────────────────────────────────────────────────────────

def test_create_session(client):
    r = client.post(
        "/api/v1/sessions",
        json={"user_id": USER_ID, "title": "My session", "context": "Be concise."},
        headers=HEADERS,
    )
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert data["is_active"] is True


def test_list_sessions(client):
    r = client.get(f"/api/v1/sessions?user_id={USER_ID}", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "sessions" in data
    assert data["total"] >= 1


def test_get_session(client):
    # Create one first
    r1 = client.post(
        "/api/v1/sessions",
        json={"user_id": USER_ID},
        headers=HEADERS,
    )
    sid = r1.json()["id"]

    r2 = client.get(f"/api/v1/sessions/{sid}?user_id={USER_ID}", headers=HEADERS)
    assert r2.status_code == 200
    assert r2.json()["id"] == sid


def test_reset_session(client):
    # Create session + send a message
    r1 = client.post(
        "/api/v1/chat",
        json={"user_id": USER_ID, "message": "Remember this."},
        headers=HEADERS,
    )
    sid = r1.json()["session_id"]

    r2 = client.post(
        f"/api/v1/sessions/{sid}/reset",
        json={"keep_context": True},
        headers=HEADERS,
    )
    assert r2.status_code == 200
    assert r2.json()["message_count"] == 0


def test_delete_session(client):
    r1 = client.post(
        "/api/v1/sessions",
        json={"user_id": USER_ID},
        headers=HEADERS,
    )
    sid = r1.json()["id"]

    r2 = client.delete(f"/api/v1/sessions/{sid}", headers=HEADERS)
    assert r2.status_code == 204


def test_get_nonexistent_session_returns_404(client):
    r = client.get(f"/api/v1/sessions/does-not-exist?user_id={USER_ID}", headers=HEADERS)
    assert r.status_code == 404


# ── WebSocket ─────────────────────────────────────────────────────────────────

def test_websocket_auth_rejected_without_key(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/chat") as ws:
            pass


def test_websocket_ping_pong(client):
    with client.websocket_connect(f"/ws/chat?api_key={VALID_KEY}") as ws:
        ws.send_text(json.dumps({"type": "ping", "user_id": USER_ID, "message": ""}))
        data = json.loads(ws.receive_text())
        assert data["type"] == "pong"


def test_websocket_chat_stream(client):
    with client.websocket_connect(f"/ws/chat?api_key={VALID_KEY}") as ws:
        ws.send_text(json.dumps({
            "type": "chat",
            "user_id": USER_ID,
            "message": "Hello via WebSocket",
        }))
        messages = []
        for _ in range(20):  # collect frames
            msg = json.loads(ws.receive_text())
            messages.append(msg)
            if msg["type"] in ("done", "error"):
                break

        types = [m["type"] for m in messages]
        assert "done" in types
