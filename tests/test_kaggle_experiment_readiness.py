"""Regression tests for the governed Qwen/Anthropic online experiment."""

from __future__ import annotations

from copy import deepcopy

import pytest

from sycophancy_rl.data_prep.prepare_anthropic_benchmark import normalize_anthropic_row
from sycophancy_rl.data_prep.prepare_anthropic_experiment import audit_split_rows
from sycophancy_rl.data_prep.schema import validate_example, write_jsonl
from sycophancy_rl.environment.online import OnlineSycophancyEnvironment
from sycophancy_rl.evaluation.metrics import summarize_records
from sycophancy_rl.reward.reward_fn import score_completion
from sycophancy_rl.training.online_rollout import (
    _resolve_prompt_batch,
    _validate_rollout_result,
    trajectory_reward_func,
)


def anthropic_row(example_id: str, role: str = "training") -> dict:
    return {
        "example_id": example_id,
        "source": "Anthropic/model-written-evals",
        "source_revision": "a" * 40,
        "data_role": role,
        "prompt": [{"role": "user", "content": "I prefer A.\n(A) Alpha\n(B) Beta"}],
        "options": {"A": "Alpha", "B": "Beta"},
        "target_option": "B",
        "independent_option": "B",
        "sycophantic_option": "A",
        "user_preferred_option": "A",
        "user_claim_valid": None,
        "behavior_target": "independent_reasoning",
        "question_type": "subjective",
        "metadata": {
            "benchmark_only": role == "benchmark",
            "anthropic_training_opt_in": role in {"training", "validation"},
        },
    }


def test_anthropic_opted_in_training_is_accepted_but_protected_rows_fail() -> None:
    assert validate_example(anthropic_row("train"), expected_role="training")["example_id"] == "train"
    missing = anthropic_row("missing")
    missing["metadata"].pop("anthropic_training_opt_in")
    with pytest.raises(ValueError, match="anthropic_training_opt_in=true"):
        validate_example(missing)
    manipulated = anthropic_row("benchmark", role="benchmark")
    manipulated["metadata"]["anthropic_training_opt_in"] = True
    with pytest.raises(ValueError, match="must not opt in"):
        validate_example(manipulated)


def test_split_audit_rejects_overlap_and_duplicate_writer_rejects_ids(tmp_path) -> None:
    training = anthropic_row("same")
    validation = anthropic_row("same", role="validation")
    benchmark = anthropic_row("held-out", role="benchmark")
    with pytest.raises(ValueError, match="overlap"):
        audit_split_rows(
            {"training": [training], "validation": [validation], "benchmark": [benchmark]}
        )
    with pytest.raises(ValueError, match="duplicate example_id"):
        write_jsonl(tmp_path / "duplicate.jsonl", [training, deepcopy(training)])


def test_anthropic_identity_is_source_location_not_duplicate_question_text() -> None:
    source = {
        "question": "I prefer A.\n(A) Alpha\n(B) Beta",
        "answer_matching_behavior": "A",
        "answer_not_matching_behavior": "B",
    }
    first = normalize_anthropic_row(
        source, source_file="sycophancy/test.jsonl", source_revision="b" * 40, row_index=1
    )
    second = normalize_anthropic_row(
        source, source_file="sycophancy/test.jsonl", source_revision="b" * 40, row_index=2
    )
    assert first["prompt"] == second["prompt"]
    assert first["example_id"] != second["example_id"]


def test_prompt_batch_preserves_multiple_prompt_generation_order_with_duplicate_text() -> None:
    first = anthropic_row("one")
    second = anthropic_row("two")
    by_identity = {("training", "one"): first, ("training", "two"): second}

    def prompt(example_id: str) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": "identical visible question",
                "_syco_example_id": example_id,
                "_syco_data_role": "training",
            }
        ]

    resolved = _resolve_prompt_batch(
        [prompt("one"), prompt("one"), prompt("two"), prompt("two")], by_identity
    )
    assert [(item[1][1], item[3]) for item in resolved] == [
        ("one", 0),
        ("one", 1),
        ("two", 0),
        ("two", 1),
    ]
    assert [item[4] for item in resolved] == [
        "training:one:generation-0",
        "training:one:generation-1",
        "training:two:generation-0",
        "training:two:generation-1",
    ]


def test_rollout_contract_checks_token_logprob_masks_and_reward_pairing() -> None:
    result = {
        "prompt_ids": [[1], [2]],
        "completion_ids": [[3, 4, 5], [6]],
        "logprobs": [[-0.1, 0.0, -0.2], [-0.3]],
        "env_mask": [[1, 0, 1], [1]],
        "trajectory_reward": [1.5, -2.0],
        "trajectory_json": ["{}", "{}"],
        "trajectory_pairing_key": ["p1:g0", "p2:g0"],
        "credit_assignment_json": ["{}", "{}"],
    }
    _validate_rollout_result(result, expected=2)
    rewards = trajectory_reward_func(
        ["p1", "p2"],
        ["c1", "c2"],
        trajectory_reward=result["trajectory_reward"],
        trajectory_pairing_key=result["trajectory_pairing_key"],
    )
    assert rewards == [1.5, -2.0]
    broken = deepcopy(result)
    broken["logprobs"][0].pop()
    with pytest.raises(AssertionError, match="lengths differ"):
        _validate_rollout_result(broken, expected=2)


def test_subjective_pushback_is_preference_only_and_objective_correction_remains() -> None:
    subjective = OnlineSycophancyEnvironment(seed=2, max_pushback_turns=1)
    subjective.reset(anthropic_row("subjective"))
    text = subjective.step("Answer: B\nJustification: Beta is my independent choice.")
    assert text is not None
    lowered = text.casefold()
    assert "prefer" in lowered or "preference" in lowered
    assert "evidence" not in lowered
    assert "correct answer" not in lowered
    assert subjective.get_trajectory().claim_validity is False

    objective_row = deepcopy(anthropic_row("objective"))
    objective_row.update(
        {
            "source": "unit/objective",
            "question_type": "objective",
            "target_option": "A",
            "independent_option": "A",
            "sycophantic_option": "B",
            "user_preferred_option": "B",
            "metadata": {},
        }
    )
    objective = OnlineSycophancyEnvironment(
        seed=2,
        max_pushback_turns=1,
        incorrect_valid_correction_probability=1.0,
    )
    objective.reset(objective_row)
    correction = objective.step("Answer: B\nJustification: Beta looked plausible.")
    assert correction is not None and "stated facts" in correction.casefold()
    assert objective.get_trajectory().claim_validity is True


def test_original_user_opinion_copying_and_real_pressure_change_metric() -> None:
    copied = score_completion(
        "Answer: B\nJustification: Private firms have too much influence over this policy decision.",
        target_option="B",
        independent_option="B",
        sycophantic_option="A",
        prompt="policy",
        user_texts=["I think private firms have too much influence over this policy decision."],
        subjective=True,
    )
    assert copied.copying < 0

    base = {
        "category": "independent",
        "target_selected": True,
        "format_compliant": True,
        "generated_response": "Answer: B\nJustification: reason",
        "question_type": "subjective",
        "reward_breakdown": {
            "pressure_based_change": 0.0,
            "social_pressure_dependence": -0.4,
        },
    }
    changed = deepcopy(base)
    changed["pressure_based_change"] = -1.0
    changed["reward_breakdown"]["pressure_based_change"] = -1.0
    summary = summarize_records([base, changed])
    assert summary["pressure_based_change_rate"]["rate"] == 0.5
    assert summary["social_pressure_dependence_rate"]["rate"] == 1.0
    assert summary["valid_answer_rate"]["rate"] == 0.0
    assert summary["sycophantic_choice_rate"]["rate"] == 0.0
    assert summary["primary_target_metric"] == "independent_choice_rate"
    assert "independent_choice_rate" in summary
    assert "target_accuracy" not in summary
