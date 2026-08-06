"""Leakage and permanence tests for group-aware dataset splitting."""

import copy
import json
from pathlib import Path

import pytest

from sycophancy_rl.data_prep.schema import read_jsonl
from sycophancy_rl.data_prep.split_data import (
    find_near_duplicate_groups,
    split_examples,
    write_splits,
)


def test_question_variants_never_cross_splits() -> None:
    rows = read_jsonl("data/processed/training_pool.jsonl", expected_role="training")
    splits = split_examples(rows, seed=42)
    group_to_split: dict[str, str] = {}

    for split_name, split_rows in splits.items():
        for row in split_rows:
            group = row["metadata"]["split_group"]
            assert group_to_split.setdefault(group, split_name) == split_name


def test_duplicate_ids_and_prompts_are_rejected() -> None:
    rows = read_jsonl("data/processed/training_pool.jsonl", expected_role="training")
    duplicate_id = copy.deepcopy(rows[0])
    duplicate_prompt = copy.deepcopy(rows[0])
    duplicate_prompt["example_id"] = "different-id"

    with pytest.raises(ValueError, match="Duplicate example_id"):
        split_examples(rows + [duplicate_id])
    with pytest.raises(ValueError, match="Exact duplicate prompts"):
        split_examples(rows + [duplicate_prompt])


def test_saved_ids_match_saved_rows(tmp_path) -> None:
    source = tmp_path / "pool.jsonl"
    source.write_bytes(Path("data/processed/training_pool.jsonl").read_bytes())
    rows = read_jsonl(source, expected_role="training")
    splits = split_examples(rows, seed=7)
    output = tmp_path / "splits"

    write_splits(splits, output, seed=7, source_path=source)

    for role, stem in (
        ("training", "train"),
        ("validation", "validation"),
        ("test", "test"),
    ):
        ids = json.loads((output / f"{stem}_ids.json").read_text(encoding="utf-8"))
        saved = read_jsonl(output / f"{stem}.jsonl", expected_role=role)
        assert ids == [row["example_id"] for row in saved]


def test_near_duplicate_questions_are_detected() -> None:
    rows = read_jsonl("data/processed/training_pool.jsonl", expected_role="training")
    near_duplicate = copy.deepcopy(rows[0])
    near_duplicate["example_id"] = "near-duplicate-id"
    near_duplicate["metadata"]["question_id"] = "near-duplicate-question"
    near_duplicate["metadata"]["question_text"] = "What is the capital of France ?"
    near_duplicate["prompt"][0]["content"] = (
        near_duplicate["prompt"][0]["content"].replace(
            "What is the capital of France?",
            "What is the capital of France ?",
        )
    )

    matches = find_near_duplicate_groups(rows + [near_duplicate])
    assert any(match["right_group"] == "near-duplicate-question" for match in matches)
    with pytest.raises(ValueError, match="Near-duplicate question groups"):
        split_examples(rows + [near_duplicate])
