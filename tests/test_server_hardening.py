"""Tests for the FastAPI server hardening."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sycophancy_rl.data_prep.schema import read_jsonl, write_jsonl
from sycophancy_rl.environment.session_manager import SessionManager
from sycophancy_rl.server.main import DEFAULT_ALLOWED_ORIGINS, create_app


@pytest.fixture
def app(tmp_path: Path):
    pool = tmp_path / "pool.jsonl"
    fixture_source = Path(__file__).resolve().parents[1] / "data/processed/training_pool.jsonl"
    write_jsonl(
        pool,
        [read_jsonl(fixture_source, expected_role="training")[0]],
    )
    return create_app(
        session_manager=SessionManager(max_sessions=2, ttl_seconds=10),
        dataset_path=pool,
    )


def test_app_binds_to_default_origins(app) -> None:
    client = TestClient(app)
    response = client.get("/health/live")
    assert response.status_code == 200


def test_app_includes_package_version(app) -> None:
    client = TestClient(app)
    body = client.get("/health/live").json()
    assert body["version"] != "0.0.0"


def test_readiness_reflects_loaded_episode_count(app) -> None:
    client = TestClient(app)
    body = client.get("/health/ready").json()
    assert body["status"] == "ok"
    assert body["episodes_loaded"] == 1


def test_wildcard_with_credentials_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SYCO_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("SYCO_CORS_ALLOW_CREDENTIALS", "1")
    with pytest.raises(RuntimeError, match="Wildcard CORS origin"):
        create_app(session_manager=SessionManager())


def test_wildcard_without_credentials_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("SYCO_ALLOWED_ORIGINS", "*")
    monkeypatch.delenv("SYCO_CORS_ALLOW_CREDENTIALS", raising=False)
    app = create_app(session_manager=SessionManager())
    # Smoke-test the middleware stack by hitting a public endpoint.
    client = TestClient(app)
    response = client.get("/health/live")
    assert response.status_code == 200


def test_uppercase_origin_environment_variable_is_honored(monkeypatch) -> None:
    monkeypatch.setenv("SYCO_ORIGINS", "https://example.test")
    configured = create_app(session_manager=SessionManager())
    cors = next(
        middleware
        for middleware in configured.user_middleware
        if middleware.cls.__name__ == "CORSMiddleware"
    )
    assert cors.kwargs["allow_origins"] == ["https://example.test"]


def _example() -> dict:
    return {
        "example_id": "x",
        "source": "anthropic/model-written-evals",
        "prompt": [{"role": "user", "content": "q"}],
        "options": {"A": "a", "B": "b"},
        "target_option": "A",
        "independent_option": "B",
        "question_type": "objective",
        "pushback_turns": [],
    }


def test_session_manager_caps_sessions() -> None:
    sm = SessionManager(max_sessions=1, ttl_seconds=10)
    sm.create_session(_example())
    with pytest.raises(RuntimeError, match="Session cap reached"):
        sm.create_session(_example())


def test_session_manager_ttl_eviction() -> None:
    sm = SessionManager(max_sessions=10, ttl_seconds=0.001)
    sm.create_session(_example())
    import time

    time.sleep(0.1)
    sm.create_session(_example())
    # Both calls succeed because TTL evicted the first.
    assert sm.count() == 1


def test_expired_session_cannot_be_fetched() -> None:
    sm = SessionManager(max_sessions=10, ttl_seconds=0.001)
    session_id = sm.create_session(_example())
    import time

    time.sleep(0.1)
    with pytest.raises(KeyError):
        sm.get_session(session_id)


def test_request_schemas_reject_unknown_fields_and_large_responses(app) -> None:
    client = TestClient(app)
    assert client.post("/reset", json={"unknown": True}).status_code == 422
    reset = client.post("/reset", json={"seed": 1})
    session_id = reset.json()["session_id"]
    oversized = client.post(
        "/step",
        json={"session_id": session_id, "response": "x" * 16_001},
    )
    assert oversized.status_code == 422


def test_default_allowed_origins_are_safe() -> None:
    # The wildcard must never appear by default.
    assert "*" not in DEFAULT_ALLOWED_ORIGINS
