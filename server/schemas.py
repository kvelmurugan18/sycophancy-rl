"""Pydantic request/response models for the Sycophancy RL environment server.

These schemas define the public HTTP contract for the FastAPI app exposed
under `server/`. The server runs a multi-turn RL environment: clients reset
to start an episode, step the model forward turn-by-turn, and optionally
hand the full trajectory to a separate grader endpoint for fine-grained
reward breakdown.

The models are intentionally thin — they only carry data, not behavior. All
Field descriptions show up in the auto-generated OpenAPI docs at `/docs`.
"""

from pydantic import BaseModel, Field


# --- Requests -----------------------------------------------------------------


class ResetRequest(BaseModel):
    """Body for `POST /reset`. Starts (or resumes) an evaluation episode."""

    episode_id: str | None = Field(
        default=None,
        description=(
            "Optional ID of a specific episode to resume. If omitted, the "
            "environment picks the next episode in the queue."
        ),
    )
    category: str | None = Field(
        default=None,
        description=(
            "Optional question category filter (e.g. 'truthfulqa', "
            "'sycophancy_eval', 'custom_pushback'). When set, only episodes "
            "matching this category are considered."
        ),
    )


class StepRequest(BaseModel):
    """Body for `POST /step`. Submits the model's reply for the current turn."""

    session_id: str = Field(
        ...,
        description=(
            "Opaque session handle returned by a prior `/reset` call. "
            "Identifies which episode and which turn number is being advanced."
        ),
    )
    response: str = Field(
        ...,
        description=(
            "The model's answer to the current prompt. The environment will "
            "score it and return the next prompt (or signal episode end)."
        ),
    )


class GraderRequest(BaseModel):
    """Body for `POST /grader`. Re-scores a full trajectory end-to-end."""

    session_id: str = Field(
        ...,
        description=(
            "Session handle for the episode whose trajectory is being graded. "
            "Used to look up ground-truth answers and metadata."
        ),
    )
    trajectory: list[dict] = Field(
        ...,
        description=(
            "Full conversation history as an ordered list of turns. Each "
            "turn is a dict with 'role' (one of 'user' / 'assistant' / "
            "'system') and 'content' (the message text)."
        ),
    )


# --- Responses ----------------------------------------------------------------


class ResetResponse(BaseModel):
    """Reply from `POST /reset`. Carries the first prompt of the episode."""

    session_id: str = Field(
        ...,
        description=(
            "Opaque handle the client must echo back in subsequent `/step` "
            "and `/grader` calls for this episode."
        ),
    )
    prompt: str = Field(
        ...,
        description=(
            "The first user-role prompt of the episode. The model should "
            "respond to this in the next `/step` call."
        ),
    )
    turn_number: int = Field(
        ...,
        description=(
            "Index of the prompt the client just received, 0-indexed. The "
            "first prompt of any episode is turn 0."
        ),
    )
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Free-form context about the episode: question category, source "
            "dataset, gold answer, difficulty tag, etc. Used by the client "
            "for logging and analysis."
        ),
    )


class StepResponse(BaseModel):
    """Reply from `POST /step`. Carries the next prompt and turn reward."""

    session_id: str = Field(
        ...,
        description=(
            "Session handle. Echoed back unchanged so the client can keep "
            "threading it through subsequent calls."
        ),
    )
    next_prompt: str | None = Field(
        ...,
        description=(
            "The next user-role prompt, or null if the episode is finished. "
            "Clients should check `done` before reading this field."
        ),
    )
    reward: float = Field(
        ...,
        description=(
            "Scalar reward for the response the client just submitted. Summed "
            "across components (correctness, calibration, format, ...). Range "
            "and sign depend on the active reward configuration."
        ),
    )
    done: bool = Field(
        ...,
        description=(
            "True if the episode has ended (no more prompts). When done is "
            "True, `next_prompt` is null and further `/step` calls with the "
            "same session will error."
        ),
    )
    turn_number: int = Field(
        ...,
        description=(
            "Index of the turn the model just answered. The next prompt, if "
            "any, will be turn_number + 1."
        ),
    )
    reward_breakdown: dict = Field(
        default_factory=dict,
        description=(
            "Per-component contribution to `reward`, keyed by component "
            "name (e.g. 'correctness', 'calibration', 'format_guard'). "
            "Useful for diagnostics and ablations."
        ),
    )


class GraderResponse(BaseModel):
    """Reply from `POST /grader`. Aggregate scoring of a full trajectory."""

    session_id: str = Field(
        ...,
        description=(
            "Session handle the trajectory was graded for. Echoed back so "
            "the client can correlate results with the originating episode."
        ),
    )
    total_reward: float = Field(
        ...,
        description=(
            "Sum of the per-turn rewards across the whole trajectory. This "
            "is the figure used for training-time advantage estimation."
        ),
    )
    component_scores: dict = Field(
        ...,
        description=(
            "Aggregate score per reward component, averaged (or summed, "
            "depending on the component) over the trajectory. Keys match "
            "the components in `StepResponse.reward_breakdown`."
        ),
    )
    passed: bool = Field(
        ...,
        description=(
            "True if the trajectory met the episode's passing threshold "
            "(e.g. correct final answer with well-calibrated confidence). "
            "A coarse, human-readable summary of success."
        ),
    )


class HealthResponse(BaseModel):
    """Reply from `GET /health`. Liveness + readiness probe."""

    status: str = Field(
        ...,
        description=(
            "Service status string. 'ok' when the server is ready to accept "
            "requests, 'degraded' or 'error' otherwise."
        ),
    )
    version: str = Field(
        ...,
        description=(
            "Server version (typically the project's package version, e.g. "
            "'0.1.0'). Bumped on every release."
        ),
    )
    episodes_loaded: int = Field(
        ...,
        description=(
            "Number of episodes currently available in the in-memory queue. "
            "If this is 0 the server is up but has nothing to serve."
        ),
    )
