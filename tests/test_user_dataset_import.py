"""Offline tests for user-owned dataset normalization and governance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sycophancy_rl.data_prep.import_user_dataset import import_user_dataset
from sycophancy_rl.data_prep.schema import read_jsonl


def test_simple_choice_csv_import_creates_immutable_splits(tmp_path: Path) -> None:
    source = tmp_path / "choices.csv"
    questions = (
        "Which planet is known as the red planet?",
        "What city is the capital of Japan?",
        "Which gas is absorbed by green plants?",
        "What number results from two multiplied by six?",
        "Which common material is attracted to magnets?",
        "Which ocean has the greatest surface area?",
    )
    source.write_text(
        "id,question,option_a,option_b,target_option,user_preferred_option\n"
        + "\n".join(
            f"q{index},{question},Correct {index},Wrong {index},A,B"
            for index, question in enumerate(questions, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    destination = import_user_dataset(
        input_path=source,
        dataset_name="customer-one",
        output_root=tmp_path / "generated",
        seed=7,
    )

    train = read_jsonl(destination / "splits" / "train.jsonl", expected_role="training")
    validation = read_jsonl(
        destination / "splits" / "validation.jsonl", expected_role="validation"
    )
    test = read_jsonl(destination / "splits" / "test.jsonl", expected_role="test")
    assert len(train) + len(validation) + len(test) == 6
    assert all(row["source"] == "user/customer-one" for row in train + validation + test)
    manifest = json.loads(
        (destination / "import_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["anthropic_benchmark_used_for_training"] is False
    assert manifest["input_format"] == "simple-choice"

    with pytest.raises(FileExistsError, match="already exists"):
        import_user_dataset(
            input_path=source,
            dataset_name="customer-one",
            output_root=tmp_path / "generated",
        )


def test_canonical_import_rejects_anthropic_benchmark_rows(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.jsonl"
    source.write_text(
        json.dumps(
            {
                "example_id": "unsafe-1",
                "source": "Anthropic/model-written-evals",
                "data_role": "training",
                "prompt": [{"role": "user", "content": "Question?"}],
                "target_option": "A",
                "independent_option": "A",
                "sycophantic_option": "B",
                "metadata": {"benchmark_only": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Evaluation-only benchmark"):
        import_user_dataset(
            input_path=source,
            dataset_name="unsafe",
            output_root=tmp_path / "generated",
            input_format="canonical",
        )


def test_import_requires_safe_dataset_name(tmp_path: Path) -> None:
    source = tmp_path / "empty.csv"
    source.write_text("question,option_a,option_b,target_option\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_name"):
        import_user_dataset(
            input_path=source,
            dataset_name="../escape",
            output_root=tmp_path,
        )
