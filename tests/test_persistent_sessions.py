"""Durability and restoration tests for the SQLite session backend."""

from pathlib import Path

from fastapi.testclient import TestClient

from sycophancy_rl.data_prep.schema import read_jsonl
from sycophancy_rl.environment.session_manager import SessionManager
from sycophancy_rl.environment.store import SQLiteEpisodeStore
from sycophancy_rl.server.main import create_app


def _example() -> dict:
    return read_jsonl("data/processed/training_pool.jsonl", expected_role="training")[0]


def test_sqlite_session_survives_manager_recreation(tmp_path: Path) -> None:
    database = tmp_path / "episodes.sqlite3"
    first = SessionManager(database_path=database)
    session_id = first.create_session(_example())
    episode = first.get_session(session_id)
    episode.advance_turn("Answer: (A)", 1.0, reward_breakdown={"answer": 1.0})
    first.save_session(session_id, episode)
    first.close()

    restored = SessionManager(database_path=database)
    loaded = restored.get_session(session_id)
    assert loaded.current_turn == 1
    assert loaded.history == episode.history
    assert loaded.trajectory_scores == [1.0]
    assert loaded.turn_details == episode.turn_details


def test_api_continues_session_after_app_recreation(tmp_path: Path) -> None:
    database = tmp_path / "api.sqlite3"
    dataset = Path("data/processed/training_pool.jsonl")
    first = TestClient(create_app(dataset_path=dataset, session_database_path=database))
    reset = first.post("/reset", json={"seed": 3})
    session_id = reset.json()["session_id"]
    stepped = first.post(
        "/step", json={"session_id": session_id, "response": "Answer: (A)"}
    )
    assert stepped.status_code == 200

    second = TestClient(create_app(dataset_path=dataset, session_database_path=database))
    restored = second.app.state.session_manager.get_session(session_id)
    assert restored.current_turn == 1
    assert any(message["role"] == "assistant" for message in restored.history)


def test_duplicate_session_ids_fail_loudly(tmp_path: Path) -> None:
    manager = SessionManager(store=SQLiteEpisodeStore(tmp_path / "duplicate.sqlite3"))
    manager.create_session(_example(), session_id="same")
    try:
        manager.create_session(_example(), session_id="same")
    except RuntimeError as exc:
        assert "Duplicate session ID" in str(exc)
    else:
        raise AssertionError("duplicate session ID was accepted")
