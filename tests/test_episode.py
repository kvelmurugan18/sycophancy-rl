"""State-transition tests for the real initial-question + pushback flow."""

import pytest

from sycophancy_rl.environment.episode import Episode


@pytest.fixture
def sample_episode() -> Episode:
    return Episode(
        episode_id="fixture-1",
        source="test",
        prompt=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Question one"},
        ],
        options={"A": "first", "B": "second"},
        target_option="A",
        independent_option="A",
        sycophantic_option="B",
        pushback_turns=[
            {"text": "First pushback", "evidence_strength": "weak"},
            {"text": "Second pushback", "evidence_strength": "weak"},
        ],
    )


def test_initial_question_counts_as_a_turn(sample_episode: Episode) -> None:
    assert sample_episode.total_turns == 3
    assert sample_episode.current_turn == 0
    assert sample_episode.get_current_user_prompt() == "Question one"
    assert sample_episode.is_done() is False


def test_history_and_prompt_are_aligned(sample_episode: Episode) -> None:
    sample_episode.begin()
    assert sample_episode.history[-1] == {"role": "user", "content": "Question one"}

    sample_episode.advance_turn("Final answer: A", 1.0)
    assert sample_episode.current_turn == 1
    assert sample_episode.get_current_user_prompt() == "First pushback"
    assert sample_episode.history[-1] == {"role": "user", "content": "First pushback"}

    sample_episode.advance_turn("Final answer: A", 1.0)
    assert sample_episode.current_turn == 2
    assert sample_episode.get_current_user_prompt() == "Second pushback"

    sample_episode.advance_turn("Final answer: A", 1.0)
    assert sample_episode.current_turn == 3
    assert sample_episode.is_done() is True
    assert sample_episode.get_current_user_prompt() is None


def test_completed_episode_rejects_extra_step(sample_episode: Episode) -> None:
    for _ in range(sample_episode.total_turns):
        sample_episode.advance_turn("Final answer: A", 1.0)

    with pytest.raises(RuntimeError, match="already complete"):
        sample_episode.advance_turn("Final answer: A", 1.0)


def test_turn_details_capture_visible_user_prompt(sample_episode: Episode) -> None:
    sample_episode.begin()
    sample_episode.advance_turn(
        "Final answer: A",
        1.05,
        reward_breakdown={"answer": 1.0},
    )

    detail = sample_episode.turn_details[0]
    assert detail["user_prompt"] == "Question one"
    assert detail["assistant_response"] == "Final answer: A"
    assert detail["reward_breakdown"] == {"answer": 1.0}
