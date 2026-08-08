"""Regression tests for fair, paired before/after comparisons."""

from __future__ import annotations

from copy import deepcopy

import pytest

from sycophancy_rl.evaluation.compare_runs import compare_runs


def _record(example_id: str = "example-1") -> dict:
    return {
        "example_id": example_id,
        "source_example_id": example_id.split("::", 1)[0],
        "source": "unit-dataset",
        "source_revision": "dataset-v1",
        "benchmark_sha256": "b" * 64,
        "seed": 42,
        "system_prompt_condition": "neutral",
        "prompt_variant": "original",
        "turn_number": 0,
        "model_id": "owner/model",
        "model_revision": "a" * 40,
        "target_option": "A",
        "independent_option": "A",
        "sycophantic_option": "B",
        "user_preferred_option": "B",
        "user_claim_valid": False,
        "question_type": "objective",
        "behavior_target": "resist_invalid_pressure",
        "prompt_sha256": "c" * 64,
        "generation_settings": {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 64,
            "repetition_penalty": 1.0,
        },
        "target_selected": True,
        "category": "independent",
        "format_compliant": True,
        "generated_response": "Answer: (A)",
        "parsed_label": "A",
        "reward": 1.0,
        "output_tokens": 3,
    }


def test_comparison_accepts_only_the_adapter_outcome_changing() -> None:
    baseline = [_record()]
    candidate = deepcopy(baseline)
    candidate[0].update(
        target_selected=False,
        category="sycophantic",
        parsed_label="B",
        generated_response="Answer: (B)",
        reward=-1.0,
    )

    assert compare_runs(baseline, candidate)["comparison_valid"] is True


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (
        ("example_id", "different-example"),
        ("prompt_sha256", "d" * 64),
        ("benchmark_sha256", "e" * 64),
        ("seed", 7),
        ("model_revision", "f" * 40),
        ("target_option", "B"),
        ("source_revision", "dataset-v2"),
        ("system_prompt_condition", "anti_sycophancy"),
    ),
)
def test_comparison_rejects_provenance_mismatch(
    field: str,
    changed_value: object,
) -> None:
    baseline = [_record()]
    candidate = deepcopy(baseline)
    candidate[0][field] = changed_value

    with pytest.raises(ValueError, match="Unfair comparison"):
        compare_runs(baseline, candidate)


def test_comparison_rejects_generation_configuration_mismatch() -> None:
    baseline = [_record()]
    candidate = deepcopy(baseline)
    candidate[0]["generation_settings"]["temperature"] = 0.5

    with pytest.raises(ValueError, match="generation settings"):
        compare_runs(baseline, candidate)


def test_generation_settings_are_matched_per_example_not_as_a_set() -> None:
    first = _record("one")
    second = _record("two")
    second["prompt_sha256"] = "d" * 64
    second["generation_settings"]["max_new_tokens"] = 32
    baseline = [first, second]
    candidate = deepcopy(baseline)
    candidate[0]["generation_settings"], candidate[1]["generation_settings"] = (
        candidate[1]["generation_settings"],
        candidate[0]["generation_settings"],
    )

    with pytest.raises(ValueError, match="generation settings"):
        compare_runs(baseline, candidate)
