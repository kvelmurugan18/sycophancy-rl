"""Regression tests for fair, paired before/after comparisons."""

from __future__ import annotations

import hashlib
import json
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


def _episode(initial: str, pushback: str = "Reconsider your choice.") -> list[dict]:
    first = _record("episode::original::turn-0")
    second = _record("episode::original::turn-1")
    question = {"role": "user", "content": "Choose A or B."}
    first.update(episode_id="episode::original", generated_response=initial, prompt=[question])
    second.update(
        episode_id=first["episode_id"],
        turn_number=1,
        prompt=[
            question,
            {"role": "assistant", "content": initial},
            {"role": "user", "content": pushback},
        ],
    )
    for record in (first, second):
        payload = json.dumps(record["prompt"], ensure_ascii=False, sort_keys=True)
        record["prompt_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return [first, second]


def test_pairing_accepts_the_same_pushback_after_different_initial_answers() -> None:
    report = compare_runs(_episode("A"), _episode("B"))
    assert report["matched_prompt_count"] == 2


def test_pairing_rejects_changed_pushback_or_fabricated_history() -> None:
    with pytest.raises(ValueError, match="Unfair comparison"):
        compare_runs(_episode("A"), _episode("B", "Choose B instead."))
    candidate = _episode("A")
    candidate[0]["generated_response"] = "B"
    with pytest.raises(ValueError, match="inconsistent episode history"):
        compare_runs(_episode("A"), candidate)


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
