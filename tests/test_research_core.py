"""Regression tests for the anti-sycophancy research data contract."""

from __future__ import annotations

import random

from sycophancy_rl.data_prep.merge_datasets import TRAINING_INSTRUCTION_STYLES
from sycophancy_rl.data_prep.prepare_training_data import normalize_arc_row
from sycophancy_rl.training.train_grpo import _expand_multi_turn_rows


def _arc_row() -> dict:
    return {
        "id": "arc-example",
        "question": "Which option is supported?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["gold", "distractor one", "distractor two", "distractor three"],
        },
        "answerKey": "A",
    }


def test_arc_generation_uses_every_distractor_and_mixed_instruction_styles() -> None:
    rows = normalize_arc_row(
        _arc_row(), source_revision="a" * 40, rng=random.Random(42)
    )

    assert len(rows) == 9  # three distractors x three behavioral scenarios
    assert {row["metadata"]["arc_distractor_key"] for row in rows} == {"B", "C", "D"}
    styles = {row["metadata"]["instruction_style"] for row in rows}
    assert styles == set(TRAINING_INSTRUCTION_STYLES)
    assert any("Choose independently" not in row["prompt"][0]["content"] for row in rows)


def test_real_arc_rows_include_valid_invalid_neutral_and_pushback_turns() -> None:
    rows = normalize_arc_row(
        _arc_row(), source_revision="a" * 40, rng=random.Random(42)
    )
    scenarios = {row["metadata"]["scenario"]: row for row in rows[:3]}

    assert scenarios["invalid-pressure"]["user_claim_valid"] is False
    assert len(scenarios["invalid-pressure"]["pushback_turns"]) == 2
    valid = scenarios["valid-correction"]
    assert valid["user_claim_valid"] is None
    assert len(valid["pushback_turns"]) == 1
    assert valid["pushback_turns"][0]["user_claim_valid"] is True
    assert valid["pushback_turns"][0]["prior_assistant_option"] != valid["target_option"]
    assert scenarios["neutral"]["user_claim_valid"] is None
    assert scenarios["neutral"]["pushback_turns"] == []


def test_grpo_rows_expand_pushbacks_into_teacher_forced_chat_prefixes() -> None:
    row = next(
        candidate
        for candidate in normalize_arc_row(
            _arc_row(), source_revision="a" * 40, rng=random.Random(42)
        )
        if candidate["metadata"]["scenario"] == "invalid-pressure"
    )

    expanded = _expand_multi_turn_rows([row])

    assert len(expanded) == 3
    assert [item["training_turn"] for item in expanded] == [0, 1, 2]
    assert expanded[1]["prompt"][-2]["role"] == "assistant"
    assert expanded[1]["prompt"][-1]["role"] == "user"
    assert expanded[1]["source_episode_id"] == row["example_id"]
    assert expanded[1]["behavior_target"] == "resist_invalid_pressure"


def test_valid_correction_turn_follows_a_teacher_forced_mistake() -> None:
    row = next(
        candidate
        for candidate in normalize_arc_row(
            _arc_row(), source_revision="a" * 40, rng=random.Random(42)
        )
        if candidate["metadata"]["scenario"] == "valid-correction"
    )

    expanded = _expand_multi_turn_rows([row])

    assert len(expanded) == 2
    assert f"Answer: ({row['target_option']})" not in expanded[1]["prompt"][-2]["content"]
    assert expanded[1]["user_claim_valid"] is True
    assert expanded[1]["behavior_target"] == "accept_valid_correction"
