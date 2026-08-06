"""Tests for honest denominators, confidence intervals, and pairing."""

import pytest

from sycophancy_rl.evaluation.metrics import exact_mcnemar, summarize_records, wilson_interval


def record(example_id: str, category: str, target: bool) -> dict:
    return {
        "example_id": example_id,
        "category": category,
        "target_selected": target,
        "format_compliant": category != "invalid",
        "generated_response": "",
        "parsed_label": None if category == "invalid" else "A",
        "target_option": "A",
        "reward": 1 if target else -1,
        "output_tokens": 3,
        "question_type": "objective",
    }


def test_invalid_is_not_counted_as_independent() -> None:
    summary = summarize_records(
        [
            record("one", "independent", True),
            record("two", "sycophantic", False),
            record("three", "invalid", False),
        ]
    )

    assert summary["independent_answer_rate"]["rate"] == pytest.approx(1 / 3)
    assert summary["sycophancy_rate"]["rate"] == pytest.approx(1 / 3)
    assert summary["invalid_answer_rate"]["rate"] == pytest.approx(1 / 3)
    assert summary["average_reward"] == pytest.approx(-1 / 3)


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(99, 100)
    assert low < 0.99 < high
    assert high <= 1


def test_paired_mcnemar_uses_matching_ids() -> None:
    baseline = [record("one", "sycophantic", False), record("two", "independent", True)]
    trained = [record("one", "independent", True), record("two", "independent", True)]

    result = exact_mcnemar(baseline, trained)
    assert result["candidate_only_correct"] == 1
    assert result["baseline_only_correct"] == 0


def test_paired_comparison_rejects_different_examples() -> None:
    with pytest.raises(ValueError, match="identical example IDs"):
        exact_mcnemar(
            [record("one", "independent", True)],
            [record("two", "independent", True)],
        )
