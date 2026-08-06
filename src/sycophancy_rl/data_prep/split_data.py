"""Create permanent, group-aware train/validation/test splits.

All variants derived from the same underlying question stay in the same split.
The split IDs and a provenance manifest are saved so future runs cannot
silently reshuffle the final test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sycophancy_rl.data_prep.schema import (
    normalized_prompt_fingerprint,
    prompt_text,
    read_jsonl,
    write_jsonl,
)

DEFAULT_INPUT = Path("data/generated/training_pool.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/generated/splits")


def _question_group(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    return str(metadata.get("question_id") or normalized_prompt_fingerprint(prompt_text(row)))


def _question_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    value = metadata.get("question_text")
    if isinstance(value, str) and value.strip():
        return " ".join(value.casefold().split())
    # Canonical generated prompts put the question before the first option.
    return " ".join(prompt_text(row).split("\n(A)", maxsplit=1)[0].casefold().split())


def find_near_duplicate_groups(
    rows: list[dict[str, Any]],
    *,
    threshold: float = 0.94,
) -> list[dict[str, Any]]:
    """Return highly similar question groups before they can cross a split.

    This deterministic text-similarity gate complements exact prompt hashes.
    It deliberately works on the underlying question, not the shared answer
    formatting and pressure template, which would otherwise inflate scores.
    """

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    representatives: dict[str, str] = {}
    for row in rows:
        representatives.setdefault(_question_group(row), _question_text(row))
    groups = sorted(representatives)
    matches: list[dict[str, Any]] = []
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            similarity = SequenceMatcher(
                None,
                representatives[left],
                representatives[right],
                autojunk=False,
            ).ratio()
            if similarity >= threshold:
                matches.append(
                    {
                        "left_group": left,
                        "right_group": right,
                        "similarity": similarity,
                    }
                )
    return matches


def _allocate_groups(
    groups: list[str],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    if len(groups) < 3:
        raise ValueError("At least three distinct question groups are required.")
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)

    n_groups = len(shuffled)
    n_validation = max(1, round(n_groups * validation_fraction))
    n_test = max(1, round(n_groups * test_fraction))
    if n_validation + n_test >= n_groups:
        n_validation = 1
        n_test = 1

    assignments: dict[str, str] = {}
    for group in shuffled[:n_test]:
        assignments[group] = "test"
    for group in shuffled[n_test : n_test + n_validation]:
        assignments[group] = "validation"
    for group in shuffled[n_test + n_validation :]:
        assignments[group] = "training"
    return assignments


def split_examples(
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """Return group-aware, leakage-checked splits."""

    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation plus test fraction must be less than 1.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_fingerprints: dict[str, str] = {}
    for row in rows:
        example_id = row["example_id"]
        if example_id in seen_ids:
            raise ValueError(f"Duplicate example_id: {example_id}")
        seen_ids.add(example_id)
        fingerprint = normalized_prompt_fingerprint(prompt_text(row))
        previous = seen_fingerprints.get(fingerprint)
        if previous is not None:
            raise ValueError(f"Exact duplicate prompts: {previous} and {example_id}")
        seen_fingerprints[fingerprint] = example_id
        grouped[_question_group(row)].append(row)

    near_duplicates = find_near_duplicate_groups(rows)
    if near_duplicates:
        first = near_duplicates[0]
        raise ValueError(
            "Near-duplicate question groups detected before splitting: "
            f"{first['left_group']} and {first['right_group']} "
            f"(similarity={first['similarity']:.3f}). Deduplicate or assign "
            "them a shared question_id."
        )

    assignments = _allocate_groups(
        sorted(grouped),
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    result = {"training": [], "validation": [], "test": []}
    for group, group_rows in grouped.items():
        split_name = assignments[group]
        for row in group_rows:
            copy = dict(row)
            copy["data_role"] = split_name
            copy.setdefault("metadata", {})
            copy["metadata"] = {**copy["metadata"], "split_group": group}
            result[split_name].append(copy)

    split_fingerprints: dict[str, set[str]] = {
        name: {
            normalized_prompt_fingerprint(prompt_text(row)) for row in split_rows
        }
        for name, split_rows in result.items()
    }
    names = list(split_fingerprints)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_fingerprints[left] & split_fingerprints[right]
            if overlap:
                raise AssertionError(f"Leakage detected between {left} and {right}.")
    return result


def write_splits(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    *,
    seed: int,
    source_path: Path,
) -> None:
    """Persist split rows, IDs, and a versioned manifest."""

    expected_outputs = {
        output_dir / name
        for name in (
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "train_ids.json",
            "validation_ids.json",
            "test_ids.json",
            "split_manifest.json",
        )
    }
    existing = sorted(path for path in expected_outputs if path.exists())
    if existing:
        raise FileExistsError(
            "Refusing to overwrite immutable split artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "source_path": source_path.as_posix(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "splits": {},
    }
    file_names = {
        "training": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }
    id_file_names = {
        "training": "train_ids.json",
        "validation": "validation_ids.json",
        "test": "test_ids.json",
    }
    for split_name, rows in splits.items():
        rows = sorted(rows, key=lambda row: row["example_id"])
        count = write_jsonl(output_dir / file_names[split_name], rows)
        ids = [row["example_id"] for row in rows]
        (output_dir / id_file_names[split_name]).write_text(
            json.dumps(ids, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["splits"][split_name] = {
            "count": count,
            "ids_file": id_file_names[split_name],
            "data_file": file_names[split_name],
        }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    rows = read_jsonl(args.input, expected_role="training")
    splits = split_examples(
        rows,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    write_splits(splits, args.output_dir, seed=args.seed, source_path=args.input)
    counts = ", ".join(f"{name}={len(rows)}" for name, rows in splits.items())
    print(f"Permanent split complete: {counts}.")


if __name__ == "__main__":
    main()
