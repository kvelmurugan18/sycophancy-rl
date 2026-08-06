"""State machine for one initial question followed by zero or more pushbacks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from sycophancy_rl.data_prep.schema import normalize_option_label, validate_messages


class Episode(BaseModel):
    """Conversation state shared by the reset, step, and grader endpoints."""

    episode_id: str
    source: str
    prompt: list[dict[str, str]]
    options: dict[str, str] = Field(default_factory=dict)
    target_option: str
    independent_option: str
    sycophantic_option: str | None = None
    user_claim_valid: bool | None = None
    question_type: str = "objective"
    pushback_turns: list[dict[str, Any]] = Field(default_factory=list)
    current_turn: int = 0
    history: list[dict[str, str]] = Field(default_factory=list)
    trajectory_scores: list[float] = Field(default_factory=list)
    turn_details: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, value: object) -> list[dict[str, str]]:
        return validate_messages(value)

    @field_validator("target_option", "independent_option")
    @classmethod
    def _validate_required_option(cls, value: str) -> str:
        return str(normalize_option_label(value))

    @field_validator("sycophantic_option")
    @classmethod
    def _validate_optional_option(cls, value: str | None) -> str | None:
        return normalize_option_label(value, allow_none=True)

    @field_validator("pushback_turns")
    @classmethod
    def _validate_pushbacks(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, pushback in enumerate(value):
            if not isinstance(pushback, dict):
                raise ValueError(f"pushback_turns[{index}] must be an object.")
            text = pushback.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"pushback_turns[{index}].text must be non-empty.")
            normalized.append({**pushback, "text": text.strip()})
        return normalized

    @property
    def total_turns(self) -> int:
        """Number of assistant responses: initial answer plus all pushbacks."""

        return 1 + len(self.pushback_turns)

    def is_done(self) -> bool:
        return self.current_turn >= self.total_turns

    def get_current_user_prompt(self) -> str | None:
        """Return the user text the next assistant action must answer."""

        if self.is_done():
            return None
        if self.current_turn == 0:
            return self.prompt[-1]["content"]
        return str(self.pushback_turns[self.current_turn - 1]["text"])

    def get_current_context(self) -> dict[str, Any]:
        """Return grading metadata aligned with the current visible prompt."""

        if self.is_done():
            raise RuntimeError("A completed episode has no current grading context.")
        if self.current_turn == 0:
            return {
                "target_option": self.target_option,
                "independent_option": self.independent_option,
                "sycophantic_option": self.sycophantic_option,
                "user_claim_valid": self.user_claim_valid,
                "evidence_strength": "",
            }
        pushback = self.pushback_turns[self.current_turn - 1]
        return {
            "target_option": pushback.get("target_option", self.target_option),
            "independent_option": pushback.get(
                "independent_option",
                self.independent_option,
            ),
            "sycophantic_option": pushback.get(
                "sycophantic_option",
                self.sycophantic_option,
            ),
            "user_claim_valid": pushback.get(
                "user_claim_valid",
                self.user_claim_valid,
            ),
            "evidence_strength": pushback.get("evidence_strength", ""),
        }

    def begin(self) -> None:
        """Seed history with the initial system/user messages once."""

        if not self.history:
            self.history.extend(dict(message) for message in self.prompt)

    def advance_turn(
        self,
        assistant_response: str,
        reward: float,
        *,
        reward_breakdown: dict[str, Any] | None = None,
    ) -> None:
        """Record an assistant action and enqueue the next user pushback."""

        if self.is_done():
            raise RuntimeError("Cannot advance an episode that is already complete.")
        current_prompt = self.get_current_user_prompt()
        self.history.append({"role": "assistant", "content": assistant_response})
        self.trajectory_scores.append(float(reward))
        self.turn_details.append(
            {
                "turn_number": self.current_turn,
                "user_prompt": current_prompt,
                "assistant_response": assistant_response,
                "reward": float(reward),
                "reward_breakdown": reward_breakdown or {},
            }
        )
        self.current_turn += 1
        next_prompt = self.get_current_user_prompt()
        if next_prompt is not None:
            self.history.append({"role": "user", "content": next_prompt})
