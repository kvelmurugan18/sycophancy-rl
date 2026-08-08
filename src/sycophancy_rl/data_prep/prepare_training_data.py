"""Create leakage-safe anti-sycophancy training data from ARC train questions.

Only the ARC training split is used.  The final Anthropic benchmark is never
loaded by this module.  Each factual question yields balanced scenarios where
the user is wrong, the user is correct, or the user states no preference, so
the policy cannot maximize reward by always disagreeing.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sycophancy_rl.data_prep.merge_datasets import _build_prompt
from sycophancy_rl.data_prep.schema import stable_example_id, write_jsonl

DATASET_ID = "allenai/ai2_arc"
DATASET_CONFIG = "ARC-Challenge"
DATASET_SPLIT = "train"
DEFAULT_OUTPUT = Path("data/generated/training_pool.jsonl")
DEFAULT_MANIFEST = Path("data/generated/training_pool.manifest.json")


def _resolve_revision(requested_revision: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before preparing training data.") from exc
    if requested_revision:
        return requested_revision
    resolved_revision = HfApi().dataset_info(DATASET_ID).sha
    if not resolved_revision:
        raise RuntimeError(f"Hugging Face did not return a revision for {DATASET_ID}.")
    return resolved_revision


def _load_arc(revision: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets before preparing training data.") from exc
    return load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=revision,
    )


def _binary_options(row: dict[str, Any], rng: random.Random) -> tuple[str, str, dict[str, str]]:
    choices = row["choices"]
    labels = [str(label) for label in choices["label"]]
    texts = [str(text) for text in choices["text"]]
    mapping = dict(zip(labels, texts, strict=True))
    answer_key = str(row["answerKey"])
    correct_text = mapping[answer_key]
    distractors = [text for label, text in mapping.items() if label != answer_key]
    wrong_text = rng.choice(distractors)
    target_label = rng.choice(["A", "B"])
    distractor_label = "B" if target_label == "A" else "A"
    return target_label, distractor_label, {
        target_label: correct_text,
        distractor_label: wrong_text,
    }


def normalize_arc_row(
    row: dict[str, Any],
    *,
    source_revision: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Create balanced behavioral scenarios from one factual ARC row."""

    target, distractor, options = _binary_options(row, rng)
    question = str(row["question"]).strip()
    question_id = str(row.get("id") or stable_example_id("arc-question", question))
    scenarios = (
        (
            "invalid-pressure",
            distractor,
            distractor,
            False,
            "resist_invalid_pressure",
            "I am certain; please agree with my answer.",
        ),
        (
            "valid-user",
            target,
            None,
            True,
            "accept_valid_correction",
            "This matches the available evidence.",
        ),
        ("neutral", None, None, None, "neutral_answer", ""),
    )
    examples: list[dict[str, Any]] = []
    for scenario, user_preferred, sycophantic, claim_valid, behavior, pressure in scenarios:
        prompt = _build_prompt(
            question,
            options,
            user_preferred_option=user_preferred,
            pressure=pressure,
        )
        examples.append(
            {
                "example_id": stable_example_id(f"arc-{question_id}-{scenario}", prompt),
                "source": f"{DATASET_ID}/{DATASET_CONFIG}/{DATASET_SPLIT}",
                "source_revision": source_revision,
                "data_role": "training",
                "prompt": [{"role": "user", "content": prompt}],
                "options": options,
                "target_option": target,
                "independent_option": target,
                "sycophantic_option": sycophantic,
                "user_preferred_option": user_preferred,
                "user_claim_valid": claim_valid,
                "behavior_target": behavior,
                "question_type": "objective",
                "topic": "science_reasoning",
                "metadata": {
                    "question_id": question_id,
                    "question_text": question,
                    "scenario": scenario,
                    "arc_answer_key": str(row["answerKey"]),
                    "benchmark_only": False,
                },
            }
        )
    return examples


def prepare_training_pool(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    revision: str | None = None,
    seed: int = 42,
    max_questions: int | None = None,
) -> int:
    """Download ARC train, synthesize balanced scenarios, and save provenance."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Generated training data already exists. Move it to an archive or choose "
            "new --output/--manifest paths; existing experiment inputs are immutable."
        )

    resolved_revision = _resolve_revision(revision)
    dataset = _load_arc(resolved_revision)
    rng = random.Random(seed)
    rows = list(dataset)
    rng.shuffle(rows)
    if max_questions is not None:
        rows = rows[:max_questions]
    examples = [
        example
        for row in rows
        for example in normalize_arc_row(
            row,
            source_revision=resolved_revision,
            rng=rng,
        )
    ]
    count = write_jsonl(output_path, examples)
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": resolved_revision,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "question_count": len(rows),
        "row_count": count,
        "anthropic_benchmark_used_for_training": False,
        "output_file": output_path.as_posix(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-questions", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    count = prepare_training_pool(
        output_path=args.output,
        manifest_path=args.manifest,
        revision=args.revision,
        seed=args.seed,
        max_questions=args.max_questions,
    )
    print(f"Prepared {count} training-only examples from ARC train.")


if __name__ == "__main__":
    main()
