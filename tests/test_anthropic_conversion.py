"""Offline conversion tests for the evaluation-only Anthropic benchmark."""

from pathlib import Path

import pytest

from sycophancy_rl.data_prep.prepare_anthropic_benchmark import (
    _assert_benchmark_output,
    normalize_anthropic_row,
)


def test_matching_behavior_is_not_renamed_correct() -> None:
    converted = normalize_anthropic_row(
        {
            "question": "Which do you prefer?\n(A) Cats\n(B) Dogs",
            "answer_matching_behavior": "(A)",
            "answer_not_matching_behavior": "(B)",
        },
        source_file="sycophancy/sycophancy_on_nlp_survey.jsonl",
        source_revision="deadbeef",
        row_index=0,
    )

    assert converted["data_role"] == "benchmark"
    assert converted["sycophantic_option"] == "A"
    assert converted["independent_option"] == "B"
    assert converted["target_option"] == "B"
    assert converted["question_type"] == "subjective"
    assert converted["metadata"]["benchmark_only"] is True
    assert "correct_answer" not in converted
    assert "wrong_answer" not in converted


def test_benchmark_cannot_be_written_into_training_paths() -> None:
    with pytest.raises(ValueError, match="must not be written"):
        _assert_benchmark_output(Path("data/splits/anthropic.jsonl"))
    with pytest.raises(ValueError, match="must not be written"):
        _assert_benchmark_output(Path("data/training/anthropic.jsonl"))


def test_matching_and_independent_options_must_differ() -> None:
    with pytest.raises(ValueError, match="must differ"):
        normalize_anthropic_row(
            {
                "question": "Question\n(A) One\n(B) Two",
                "answer_matching_behavior": "(A)",
                "answer_not_matching_behavior": "(A)",
            },
            source_file="sycophancy/example.jsonl",
            source_revision="deadbeef",
            row_index=0,
        )
