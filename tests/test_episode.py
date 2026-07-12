"""
Unit tests for the :class:`Episode` state container.

This test suite verifies the state transitions of the core RL episode
memory container — the per-turn ledger that the FastAPI server's
``/reset`` / ``/step`` / ``/grader`` endpoints all operate on. The
:class:`Episode` is small but load-bearing: it owns the conversation
history, the per-turn reward trajectory, and the cursor into the
pushback sequence, so a regression in any of its three state
transitions (``__init__`` defaults, ``is_done()``, ``advance_turn``)
would silently corrupt every downstream aggregation (flip-rate,
ablation report, GRPO training signal).

The tests are intentionally *behavior-driven* rather than
implementation-driven: they assert what the public contract says
should happen, not how the class stores it. That keeps the suite
robust to internal refactors of the Pydantic model (e.g. switching
the per-turn log from a plain list to a deque, or moving the cursor
into a separate sub-object) — the tests will only fail when a real
caller-visible behavior changes.
"""

import pytest

from src.environment.episode import Episode


@pytest.fixture
def sample_episode() -> Episode:
    """Build a fully-populated :class:`Episode` for use across the test suite.

    The fixture returns a *fresh* :class:`Episode` per test (pytest
    instantiates fixtures per-test by default), so no test can mutate
    the shared state of another test. The mock data uses values that
    would never appear in the real pipeline (``episode_id="test_1"``,
    ``source="test"``) so any accidental cross-contamination from a
    real ``merged_episodes.jsonl`` would be immediately obvious in a
    failure message.

    The fixture installs exactly **two** pushback dicts in
    ``pushback_turns``. This is the smallest non-trivial number: zero
    would make ``is_done()`` trivially true at construction (so the
    ``test_is_done`` start-state assertion would have to special-case
    it), and one would make the "current_turn == 2" done-state
    assertion exercise only the boundary case. Two is the minimum
    that exercises both the in-progress (``current_turn < 2``) and
    finished (``current_turn == 2``) regimes.
    """
    return Episode(
        episode_id="test_1",
        prompt="Hello",
        correct_answer="A",
        wrong_answer="B",
        source="test",
        pushback_turns=[
            {"text": "Are you sure?", "is_valid_correction": False, "evidence_strength": "weak"},
            {"text": "I really think it's B.", "is_valid_correction": False, "evidence_strength": "strong"},
        ],
    )


def test_episode_initialization(sample_episode: Episode) -> None:
    """A freshly-constructed :class:`Episode` has empty per-turn logs and a zero cursor.

    Asserts the three ``Field(default_factory=...)``-style invariants
    the rest of the system relies on:

    - ``current_turn == 0`` (no assistant turn has been produced yet;
      ``get_current_pushback()`` should now return the first pushback
      in the list).
    - ``history == []`` (the chat-template-ready ``{"role",
      "content"}`` log starts empty; the model hasn't spoken yet).
    - ``trajectory_scores == []`` (no per-turn rewards have been
      emitted by :func:`composite_reward_func` yet).

    The construction-time values for the non-default fields
    (``episode_id``, ``prompt``, ``correct_answer``, ``wrong_answer``,
    ``source``, ``pushback_turns``) are not re-asserted here — the
    fixture's contract is the source of truth for those.
    """
    assert sample_episode.current_turn == 0
    assert sample_episode.history == []
    assert sample_episode.trajectory_scores == []


def test_is_done(sample_episode: Episode) -> None:
    """``is_done()`` flips from False to True once the cursor reaches the pushback count.

    Exercises both regimes:

    - **In-progress**: right after construction
      (``current_turn == 0``, ``len(pushback_turns) == 2``),
      ``is_done()`` must return ``False`` — the model still has
      pushback to respond to.
    - **Finished**: after the cursor is manually bumped to
      ``len(pushback_turns)`` (``2``), ``is_done()`` must return
      ``True``. The ``current_turn`` is mutated directly here (not
      through ``advance_turn``) so the test isolates the
      ``is_done()`` predicate from the ``advance_turn()`` mutation
      — a failure in one should not mask a failure in the other.
    """
    # In-progress: cursor is at the start, two pushbacks remain.
    assert sample_episode.is_done() is False

    # Finished: cursor has consumed every pushback.
    sample_episode.current_turn = 2
    assert sample_episode.is_done() is True


def test_advance_turn(sample_episode: Episode) -> None:
    """``advance_turn(response, reward)`` appends to all three per-turn logs in lockstep.

    Asserts the three side effects of :meth:`Episode.advance_turn`,
    in the order the method performs them:

    1. ``history`` gets a new ``{"role": "assistant", "content":
       response}`` entry — the chat-template-ready log the trainer
       can re-feed into the model for context.
    2. ``trajectory_scores`` gets the scalar ``reward`` appended —
       the per-turn log :func:`grade_episode` sums into
       ``total_reward``.
    3. ``current_turn`` is incremented by 1 — the cursor into
       ``pushback_turns`` advances to the next user message.

    The assertion that *all three* updates land on the same call is
    the test that protects against a partial-update bug (e.g. a
    future refactor that bumps the cursor but forgets to append the
    reward). The test also asserts the *exact* values, not just the
    lengths, so a wrong-but-same-length append (e.g. ``reward=0.0``
    instead of ``0.5``) would fail loudly.
    """
    sample_episode.advance_turn("My answer", 0.5)

    assert sample_episode.history == [
        {"role": "assistant", "content": "My answer"},
    ]
    assert sample_episode.trajectory_scores == [0.5]
    assert sample_episode.current_turn == 1
