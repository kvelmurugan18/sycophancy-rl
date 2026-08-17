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


def test_invalid_pressure_contract_rejects_a_valid_claim() -> None:
    row = copy.deepcopy(read_jsonl("data/splits/train.jsonl")[0])
    row.update(
        {
            "behavior_target": "resist_invalid_pressure",
            "user_claim_valid": True,
            "user_preferred_option": row["sycophantic_option"],
        }
    )

    with pytest.raises(ValueError, match="requires user_claim_valid=false"):
        validate_example(row)


def test_valid_correction_contract_requires_user_preference_to_match_target() -> None:
    row = copy.deepcopy(read_jsonl("data/splits/train.jsonl")[0])
    other = "B" if row["target_option"] == "A" else "A"
    row.update(
        {
            "behavior_target": "accept_valid_correction",
            "user_claim_valid": True,
            "user_preferred_option": other,
            "sycophantic_option": None,
            "pushback_turns": [],
        }
    )

    with pytest.raises(ValueError, match="must equal target_option"):
        validate_example(row)


def test_independent_option_cannot_conflict_with_correctness_target() -> None:
    row = copy.deepcopy(read_jsonl("data/splits/train.jsonl")[0])
    row["independent_option"] = row["sycophantic_option"]

    with pytest.raises(ValueError, match="independent_option must equal target_option"):
        validate_example(row)
