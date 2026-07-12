"""
Integration tests for the FastAPI server routes.

This suite drives the environment server end-to-end through FastAPI's
``TestClient``, which is a thin wrapper around ``httpx`` that hits the
app *in-process* (no real socket, no port binding) while still going
through the full ASGI middleware stack — CORS, exception handlers,
Pydantic request/response validation, and the route handlers
themselves. That makes the suite the right tool to verify the
*contract* between the four route modules (``/health`` / ``/reset`` /
``/step`` / ``/grader``) and the rest of the system, end to end.

What this suite covers:

- **Liveness probe** (``/health``): a single GET request asserts the
  server is up, the JSON contract is what the rest of the project
  expects (``status == "ok"``), and the FastAPI app loaded without
  crashing during import.
- **Episode start + one step** (``/reset`` then ``/step``): the
  happy-path RL flow — pick an episode, register it with the
  ``SessionManager``, score a model reply, return the next pushback.
  Verifies that the ``session_id`` minted by ``/reset`` is the same
  opaque handle the subsequent ``/step`` echoes back, and that the
  step response carries the two fields the demo Space (and any
  external RL client) reads: ``reward`` and ``next_prompt``.

The suite is intentionally **minimal** — one assertion per behavior,
no deep path coverage of the four-pillar rubric or the
``SessionManager`` internals (those live in their own dedicated test
files). The goal here is just "the wiring is correct: a request into
the FastAPI app produces the response shape the rest of the project
consumes."
"""

import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI ``TestClient`` bound to the live app.

    A fresh client is created per test (pytest's default per-test
    fixture instantiation), so each test gets its own in-process
    request session and its own copy of the ``app.state`` registries.
    The ``app`` itself is shared across tests (it's a module-level
    singleton in :mod:`server.main`), but because the
    ``SessionManager`` is also a module-level singleton on
    ``app.state``, tests can interact without trampling each other as
    long as they use distinct session ids — which the
    ``test_reset_and_step`` test does naturally, since the server
    mints a fresh UUID on every ``/reset`` call.
    """
    return TestClient(app)


def test_health_route(client: TestClient) -> None:
    """``GET /health`` returns 200 and ``status == "ok"``.

    The liveness probe is the smallest possible smoke test: if the
    app imported cleanly enough for ``TestClient(app)`` to
    construct, the four route modules are all wired up and the
    ``SessionManager`` is attached to ``app.state``. A 200 response
    from ``/health`` is the confirmation.

    Asserted:

    - HTTP 200 (the route returns 200 for a healthy server).
    - ``response.json()["status"] == "ok"`` — matches the literal
      string the :class:`HealthResponse` schema documents, so any
      future rename in the route would break this test loudly.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_reset_and_step(client: TestClient) -> None:
    """Happy-path ``/reset`` + ``/step`` flow with a dummy model reply.

    Drives the full episode start + one-step sequence:

    1. ``POST /reset`` with no category filter — the server picks a
       random episode, registers it in the ``SessionManager`` under a
       fresh UUID, and returns the opening prompt.
    2. The test extracts the ``session_id`` from the response.
    3. ``POST /step`` echoes that ``session_id`` back along with a
       stub model reply, and the server scores the reply with
       :func:`composite_reward_func` and returns the next pushback.

    Asserted:

    - ``/reset`` returns 200 (the dataset loaded successfully and an
      episode was registered).
    - ``/step`` returns 200 (the ``session_id`` was recognized, the
      reply was graded, and the next pushback was returned).
    - The step response JSON contains both ``"reward"`` and
      ``"next_prompt"`` keys — the two fields the demo Space
      (:mod:`deploy.demo_space.app`) and any external RL client
      actually read. Their presence is a contract test: a future
      refactor of :class:`StepResponse` that drops or renames either
      field would break the demo's wiring, and this test catches it
      at the integration level rather than in production.

    Note: this test depends on ``data/processed/merged_episodes.jsonl``
    existing on disk. If the file is missing, ``/reset`` will 500 and
    the test will fail loudly with a clear traceback pointing at
    ``_load_episodes`` — which is the right behavior (a missing
    dataset is a real failure, not something to silently skip).
    """
    # 1. Start a fresh episode.
    reset_response = client.post("/reset", json={"category": None})
    assert reset_response.status_code == 200

    reset_data = reset_response.json()
    session_id = reset_data["session_id"]

    # 2. Submit a stub model reply to the first turn.
    step_response = client.post(
        "/step",
        json={"session_id": session_id, "response": "Test answer"},
    )
    assert step_response.status_code == 200

    # 3. Verify the step response carries the contract fields.
    step_data = step_response.json()
    assert "reward" in step_data
    assert "next_prompt" in step_data
