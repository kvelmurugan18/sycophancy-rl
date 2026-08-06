"""End-to-end tests for the public FastAPI environment contract."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from sycophancy_rl.data_prep.schema import read_jsonl
from sycophancy_rl.environment.session_manager import SessionManager
from sycophancy_rl.server.main import app


@pytest.fixture
def client() -> TestClient:
    app.state.session_manager = SessionManager()
    return TestClient(app)


def test_health_route(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_complete_three_turn_flow_without_gold_leak(client: TestClient) -> None:
    example = next(
        row
        for row in read_jsonl(
            "data/processed/training_pool.jsonl",
            expected_role="training",
        )
        if row["metadata"]["question_id"] == "arithmetic-2-plus-2"
        and row["metadata"]["scenario"] == "invalid-pressure"
    )
    reset = client.post(
        "/reset",
        json={"episode_id": example["example_id"]},
    )
    assert reset.status_code == 200
    reset_data = reset.json()
    assert "target_option" not in reset_data
    assert "independent_option" not in reset_data
    assert "target_option" not in reset_data["metadata"]
    assert reset_data["metadata"]["total_turns"] == 3

    session_id = reset_data["session_id"]
    prompts = [reset_data["prompt"]]
    for expected_turn in (1, 2, 3):
        response = client.post(
            "/step",
            json={
                "session_id": session_id,
                "response": (
                    f"Answer: ({example['target_option']})\n"
                    "Reason: Two plus two equals four."
                ),
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["turn_number"] == expected_turn
        assert body["reward_breakdown"]["parsed_label"] == example["target_option"]
        prompts.append(body["next_prompt"])

    assert prompts[0] == reset_data["prompt"]
    assert "5" in prompts[1]
    assert "5" in prompts[2]
    assert prompts[3] is None
    assert body["done"] is True

    grade = client.post("/grader", json={"session_id": session_id})
    assert grade.status_code == 200
    assert len(grade.json()["component_scores"]) == 3
    assert grade.json()["passed"] is True


def test_unfinished_episode_cannot_be_graded(client: TestClient) -> None:
    reset = client.post("/reset", json={"seed": 7})
    session_id = reset.json()["session_id"]

    grade = client.post("/grader", json={"session_id": session_id})
    assert grade.status_code == 400


def test_empty_and_unknown_steps_are_rejected(client: TestClient) -> None:
    reset = client.post("/reset", json={"seed": 7})
    session_id = reset.json()["session_id"]

    empty = client.post("/step", json={"session_id": session_id, "response": " "})
    missing = client.post(
        "/step",
        json={"session_id": "missing", "response": "Final answer: A"},
    )

    assert empty.status_code == 422
    assert missing.status_code == 404
