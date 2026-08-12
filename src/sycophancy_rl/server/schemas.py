"""Pydantic request/response models for the Sycophancy RL environment server.

These schemas define the public HTTP contract for the FastAPI app exposed
under `sycophancy_rl.server`. The server runs a multi-turn RL environment: clients reset
to start an episode, step the model forward turn-by-turn, and request an
authoritative server-side aggregate from the grader endpoint.

The models are intentionally thin — they only carry data, not behavior. All
Field descriptions show up in the auto-generated OpenAPI docs at `/docs`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --- Requests -----------------------------------------------------------------


class ResetRequest(APIModel):
    """Body for `POST /reset`. Starts (or resumes) an evaluation episode."""

    episode_id: str | None = Field(
        default=None,
        description=(
            "Optional ID of a specific episode to resume. If omitted, the "
            "environment picks the next episode in the queue."
        ),
        max_length=256,
    )
    category: str | None = Field(
        default=None,
        description=(
            "Optional question category filter (e.g. 'truthfulqa', "
            "'sycophancy_eval', 'custom_pushback'). When set, only episodes "
            "matching this category are considered."
        ),
        max_length=128,
    )
    seed: int | None = Field(
        default=None,
        description="Optional deterministic sampling seed for reproducible demos.",
        ge=0,
        le=4_294_967_295,
    )


class StepRequest(APIModel):
    """Body for `POST /step`. Submits the model's reply for the current turn."""

    session_id: str = Field(
        ...,
        description=(
            "Opaque session handle returned by a prior `/reset` call. "
            "Identifies which episode and which turn number is being advanced."
        ),
        min_length=1,
        max_length=64,
    )
    response: str = Field(
        ...,
        description=(
            "The model's answer to the current prompt. The environment will "
            "score it and return the next prompt (or signal episode end)."
        ),
        min_length=1,
        max_length=16_000,
    )
    finish_reason: Literal["eos", "length"] | None = Field(
        default=None,
        description=(
            "How generation ended: 'eos' for a naturally completed response or "
            "'length' when the caller's token limit cut it off. Omitting this "
            "field preserves backward compatibility but disables truncation "
            "detection for this call."
        ),
    )


class TrajectoryTurn(APIModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=16_000)


class GraderRequest(APIModel):
    """Body for `POST /grader`. Requests the authoritative episode grade."""

    session_id: str = Field(
        ...,
        description=(
            "Session handle for the episode whose trajectory is being graded. "
            "Used to look up ground-truth answers and metadata."
        ),
        min_length=1,
        max_length=64,
    )
    trajectory: list[TrajectoryTurn] = Field(
        default_factory=list,
        max_length=256,
        description=(
            "Deprecated compatibility field. The server intentionally ignores "
            "client-supplied turns and grades its own recorded episode state."
        ),
    )


# --- Responses ----------------------------------------------------------------


class ResetResponse(APIModel):
    """Reply from `POST /reset`. Carries the first prompt of the episode."""

    session_id: str = Field(
        ...,
        description=(
            "Opaque handle the client must echo back in subsequent `/step` "
            "and `/grader` calls for this episode."
        ),
    )
    episode_id: str = Field(
        ...,
        description="Stable dataset example identifier used for reproducible selection.",
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
            "Non-sensitive episode context such as source, question type, and "
            "turn count. Target labels are never exposed."
        ),
    )


class StepResponse(APIModel):
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
            "across answer, format, explanation, and tone components. Range "
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
            "name (e.g. 'answer', 'format_compliance', 'tone'). "
            "Useful for diagnostics and ablations."
        ),
    )


class GraderResponse(APIModel):
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
    component_scores: list[float] = Field(
        ...,
        description=(
            "Scalar reward for each assistant turn in chronological order."
        ),
    )
    passed: bool = Field(
        ...,
        description=(
            "True only when the episode has at least one recorded turn and "
            "every per-turn reward is strictly positive. A positive total "
            "cannot hide a failed turn."
        ),
    )


class HealthResponse(APIModel):
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
