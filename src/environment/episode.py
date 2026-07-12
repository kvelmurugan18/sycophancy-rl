"""Episode state machine.

This file defines the state machine for a single RL episode, tracking the
conversation history and reward trajectory across multiple turns. Each
:class:`Episode` instance is the unit of state that the FastAPI server's
``reset`` / ``step`` / ``grader`` endpoints operate on: the environment hands
the model a pushback prompt, the model responds, the reward function scores
that response, and the episode advances one turn. ``current_turn`` is the
cursor into ``pushback_turns``; ``history`` and ``trajectory_scores`` are the
per-turn logs used downstream for cross-turn math (e.g. discounted returns
and the per-episode consistency penalty).
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Episode(BaseModel):
    """State container for one multi-turn sycophancy-RL episode.

    Attributes:
        episode_id: Stable identifier for the episode (e.g. the row index in
            ``data/processed/merged_episodes.jsonl``).
        prompt: The initial question asked of the assistant.
        correct_answer: The ground-truth answer the assistant should defend
            under sycophantic pressure.
        wrong_answer: The known-bad answer the dataset's pushback is trying
            to lure the assistant toward.
        source: Provenance tag for the episode (e.g. ``"truthfulqa"``,
            ``"sycophancy_eval"``).
        pushback_turns: Ordered list of pushback dictionaries drawn from the
            dataset. Each dict is opaque to this class but typically carries
            the user-side message plus per-turn flags like
            ``is_valid_correction`` and ``evidence_strength``.
        current_turn: Cursor into ``pushback_turns``. ``0`` means the episode
            has not produced any assistant turn yet.
        history: Conversation log of ``{"role": ..., "content": ...}`` dicts
            (system / user / assistant) suitable to feed straight back into
            the chat template.
        trajectory_scores: Per-turn reward emitted by
            :func:`src.reward.reward_fn.composite_reward_func`, in turn order.
            Length always matches the number of assistant turns produced so
            far (i.e. ``len(history) // 2`` for a pure user/assistant log).
    """

    episode_id: str
    prompt: str
    correct_answer: str
    wrong_answer: str
    source: str
    pushback_turns: List[Dict[str, Any]]
    current_turn: int = 0
    history: List[Dict[str, str]] = Field(default_factory=list)
    trajectory_scores: List[float] = Field(default_factory=list)

    def is_done(self) -> bool:
        """Return ``True`` once every pushback turn has been consumed.

        A finished episode has ``current_turn`` pointing one past the end of
        ``pushback_turns``; the grader and session manager use this guard to
        decide when to stop calling the model.
        """
        return self.current_turn >= len(self.pushback_turns)

    def get_current_pushback(self) -> Dict[str, Any] | None:
        """Return the pushback dict for the current turn, or ``None`` if done.

        This is the payload the environment serializes into the next user
        message: the step route calls it, formats it as a chat turn, and
        appends it to ``history`` before requesting the assistant's reply.
        """
        if self.is_done():
            return None
        return self.pushback_turns[self.current_turn]

    def advance_turn(self, assistant_response: str, reward: float) -> None:
        """Record one assistant turn and move the cursor forward.

        Appends the assistant's response to ``history`` as
        ``{"role": "assistant", "content": assistant_response}``, appends the
        per-turn ``reward`` to ``trajectory_scores``, and increments
        ``current_turn`` by one. The matching user turn is expected to have
        already been appended to ``history`` by the caller (the step route
        appends the pushback before invoking the model).
        """
        self.history.append({"role": "assistant", "content": assistant_response})
        self.trajectory_scores.append(reward)
        self.current_turn += 1
