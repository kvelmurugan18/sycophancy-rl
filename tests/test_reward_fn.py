"""Contract and reward-hacking tests for the production reward."""

import pytest

from sycophancy_rl.reward.reward_fn import (
    composite_reward_func,
    get_reward_config,
    make_composite_reward_func,
    score_completion,
)

BASE_COLUMNS = {
    "target_option": ["A"],
    "independent_option": ["A"],
    "sycophantic_option": ["B"],
    "prompts": ["What is 2 + 2? (A) 4 (B) 5"],
    "options": [{"A": "4", "B": "5"}],
}


def score(text: str) -> float:
    return composite_reward_func(completions=[text], **BASE_COLUMNS)[0]


def test_valid_target_has_highest_reward() -> None:
    correct = score("Answer: (A)\nReason: Two plus two equals four.")
    wrong = score("Answer: (B)\nReason: Two plus two equals five.")
    invalid = score("<thought>I should emit formatting only.</thought>")

    assert correct > 0
    assert wrong < 0
    assert invalid < 0
    assert correct > wrong
    assert correct > invalid


def test_format_cannot_rescue_wrong_answer() -> None:
    polished_wrong = score(
        "Answer: (B)\nReason: The arithmetic expression evaluates to five."
    )
    terse_correct = score("Final answer: A")

    assert polished_wrong < 0
    assert terse_correct > polished_wrong


def test_thought_tags_receive_no_reward() -> None:
    plain = score("Final answer: A")
    tagged = score("<thought>private-looking text</thought>\nFinal answer: A")

    assert tagged == plain


def test_conflicting_answers_are_penalized() -> None:
    breakdown = score_completion(
        "Answer: (A)\nThe answer is B.",
        target_option="A",
        independent_option="A",
        sycophantic_option="B",
    )

    assert breakdown.parsed_label is None
    assert breakdown.contradiction < 0
    assert breakdown.total < -1


def test_truncated_response_is_invalid() -> None:
    breakdown = score_completion(
        "Answer: (A)",
        target_option="A",
        independent_option="A",
        sycophantic_option="B",
        finish_reason="length",
    )

    assert breakdown.parsed_label is None
    assert breakdown.parse_status == "truncated_response"
    assert breakdown.total < 0


def test_batch_columns_must_align() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        composite_reward_func(
            completions=["Final answer: A", "Final answer: B"],
            target_option=["A"],
            independent_option=["A", "A"],
            sycophantic_option=["B", "B"],
        )


def test_missing_target_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_option is required"):
        composite_reward_func(
            completions=["Final answer: A"],
            independent_option=["A"],
            sycophantic_option=["B"],
        )


def test_format_only_profile_is_explicitly_diagnostic() -> None:
    config = get_reward_config("diagnostic_format_only")
    reward = make_composite_reward_func(config.name)

    assert config.diagnostic_only is True
    assert reward(
        completions=["Answer: (B)\nReason: This is intentionally wrong."],
        **BASE_COLUMNS,
    )[0] > 0
