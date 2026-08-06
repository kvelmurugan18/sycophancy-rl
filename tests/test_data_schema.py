"""Validation tests for the canonical role-aware dataset schema."""

import copy

import pytest

from sycophancy_rl.data_prep.schema import read_jsonl, validate_example


def test_saved_splits_are_valid_and_role_separated() -> None:
    train = read_jsonl("data/splits/train.jsonl", expected_role="training")
    validation = read_jsonl(
        "data/splits/validation.jsonl",
        expected_role="validation",
    )
    test = read_jsonl("data/splits/test.jsonl", expected_role="test")

    assert {row["example_id"] for row in train}.isdisjoint(
        {row["example_id"] for row in validation + test}
    )
    assert {row["example_id"] for row in validation}.isdisjoint(
        {row["example_id"] for row in test}
    )


def test_subjective_behavior_labels_are_not_called_truth() -> None:
    row = read_jsonl("data/splits/train.jsonl")[0]

    assert "correct_answer" not in row
    assert "wrong_answer" not in row
    assert "truth" not in row
    assert "independent_option" in row
    assert "sycophantic_option" in row


def test_invalid_or_equal_behavioral_labels_are_rejected() -> None:
    row = read_jsonl("data/splits/train.jsonl")[0]
    invalid = copy.deepcopy(row)
    invalid["sycophantic_option"] = invalid["independent_option"]

    with pytest.raises(ValueError, match="must differ"):
        validate_example(invalid)
