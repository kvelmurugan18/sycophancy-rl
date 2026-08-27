"""Import a user-owned choice dataset into governed immutable experiment splits.

Supported inputs are JSONL, a JSON array, and CSV. Two row layouts are accepted:

* ``canonical``: the project's complete role-aware schema;
* ``simple-choice``: question/prompt, option_a, option_b, and target_option,
  with optional user-preference and metadata columns.

The importer never accepts protected Anthropic benchmark rows as training data.
Canonical Anthropic development rows require explicit training opt-in. It normalizes
the input, performs the same duplicate/leakage checks as the built-in ARC preparation
path, and writes an immutable pool, splits, IDs, and manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sycophancy_rl.data_prep.schema import (
    normalize_option_label,
    stable_example_id,
    validate_example,
    write_jsonl,
)
from sycophancy_rl.data_prep.split_data import split_examples, write_splits

SUPPORTED_FORMATS = ("auto", "canonical", "simple-choice")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"User dataset does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    elif suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: each row must be an object.")
                rows.append(value)
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"{path} must contain a JSON array of objects.")
        rows = [dict(row) for row in value]
    else:
        raise ValueError("User datasets must use .csv, .json, or .jsonl.")
    if not rows:
        raise ValueError(f"User dataset contains no rows: {path}")
    return rows


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: object) -> bool | None:
    text = _optional_text(value)
    if text is None:
        return None
    lowered = text.casefold()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}.")


def _simple_choice_row(
    row: dict[str, Any],
    *,
    dataset_name: str,
    source_revision: str,
    row_index: int,
) -> dict[str, Any]:
    question = _optional_text(row.get("question")) or _optional_text(row.get("prompt"))
    option_a = _optional_text(row.get("option_a"))
    option_b = _optional_text(row.get("option_b"))
    if not question or not option_a or not option_b:
        raise ValueError(
            f"Row {row_index + 1} requires question/prompt, option_a, and option_b."
        )
    target = normalize_option_label(row.get("target_option", row.get("answer")))
    independent = normalize_option_label(_optional_text(row.get("independent_option")) or target)
    preferred = normalize_option_label(
        _optional_text(row.get("user_preferred_option")), allow_none=True
    )
    sycophantic = normalize_option_label(
        _optional_text(row.get("sycophantic_option")), allow_none=True
    )
    if sycophantic is None and preferred is not None and preferred != independent:
        sycophantic = preferred
    if sycophantic == independent:
        sycophantic = None
    claim_valid = _optional_bool(row.get("user_claim_valid"))
    if claim_valid is None and preferred is not None:
        claim_valid = preferred == target
    if preferred is None:
        behavior = "neutral_answer"
    elif claim_valid:
        behavior = "accept_valid_correction"
    else:
        behavior = "resist_invalid_pressure"
    pressure = _optional_text(row.get("pressure"))
    prompt_parts = [
        question,
        f"(A) {option_a}",
        f"(B) {option_b}",
    ]
    if preferred is not None:
        prompt_parts.append(f"The user believes option ({preferred}) is correct.")
    if pressure:
        prompt_parts.append(pressure)
    prompt_parts.append("Answer independently using: Answer: (A) or Answer: (B), then Reason: ...")
    prompt_text = "\n".join(prompt_parts)
    source = f"user/{dataset_name}"
    source_id = _optional_text(row.get("id")) or _optional_text(row.get("example_id"))
    example_id = source_id or stable_example_id(source, prompt_text)
    question_id = (
        _optional_text(row.get("group_id"))
        or _optional_text(row.get("question_id"))
        or example_id
    )
    return validate_example(
        {
            "example_id": str(example_id),
            "source": source,
            "source_revision": source_revision,
            "data_role": "training",
            "prompt": [{"role": "user", "content": prompt_text}],
            "options": {"A": option_a, "B": option_b},
            "target_option": target,
            "independent_option": independent,
            "sycophantic_option": sycophantic,
            "user_preferred_option": preferred,
            "user_claim_valid": claim_valid,
            "behavior_target": behavior,
            "question_type": _optional_text(row.get("question_type")) or "objective",
            "topic": _optional_text(row.get("topic")) or "user_dataset",
            "metadata": {
                "question_id": question_id,
                "question_text": question,
                "source_row_index": row_index,
                "benchmark_only": False,
            },
        },
        expected_role="training",
    )


def _canonical_row(
    row: dict[str, Any],
    *,
    dataset_name: str,
    source_revision: str,
    row_index: int,
) -> dict[str, Any]:
    normalized = dict(row)
    role = normalized.get("data_role", "training")
    if role != "training":
        raise ValueError(
            f"Canonical row {row_index + 1} has data_role={role!r}; imports are training-only."
        )
    source = str(normalized.get("source", f"user/{dataset_name}"))
    metadata = normalized.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Canonical row {row_index + 1} metadata must be an object.")
    if metadata.get("benchmark_only"):
        raise ValueError("Evaluation-only benchmark rows cannot be imported for training.")
    if (
        source.casefold() == "anthropic/model-written-evals"
        and metadata.get("anthropic_training_opt_in") is not True
    ):
        raise ValueError(
            "Anthropic rows require metadata.anthropic_training_opt_in=true for training."
        )
    normalized["data_role"] = "training"
    normalized.setdefault("source", f"user/{dataset_name}")
    normalized.setdefault("source_revision", source_revision)
    normalized.setdefault("metadata", {})
    normalized["metadata"] = {
        **normalized["metadata"],
        "source_row_index": row_index,
        "benchmark_only": False,
    }
    normalized["metadata"].setdefault("question_id", normalized.get("example_id"))
    return validate_example(normalized, expected_role="training")


def import_user_dataset(
    *,
    input_path: Path,
    dataset_name: str,
    output_root: Path = Path("data/generated/user"),
    input_format: str = "auto",
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    license_id: str = "other",
    source_url: str | None = None,
) -> Path:
    """Normalize one user dataset and return its immutable output directory."""

    if not _SAFE_NAME.fullmatch(dataset_name):
        raise ValueError(
            "dataset_name must be 1-64 path-safe characters and start with a letter or digit."
        )
    if input_format not in SUPPORTED_FORMATS:
        raise ValueError(f"input_format must be one of {SUPPORTED_FORMATS}.")
    if not license_id.strip():
        raise ValueError("license_id must be non-empty.")
    destination = (output_root / dataset_name).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Imported dataset already exists: {destination}. Choose a new dataset name."
        )
    rows = _read_rows(input_path)
    detected = input_format
    if detected == "auto":
        detected = "canonical" if isinstance(rows[0].get("prompt"), list) else "simple-choice"
    source_revision = _sha256(input_path)
    normalizer = _canonical_row if detected == "canonical" else _simple_choice_row
    normalized = [
        normalizer(
            row,
            dataset_name=dataset_name,
            source_revision=source_revision,
            row_index=index,
        )
        for index, row in enumerate(rows)
    ]
    splits = split_examples(
        normalized,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    pool_path = destination / "training_pool.jsonl"
    try:
        write_jsonl(pool_path, normalized)
        write_splits(
            splits,
            destination / "splits",
            seed=seed,
            source_path=pool_path,
        )
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_name": dataset_name,
            "input_path": input_path.resolve().as_posix(),
            "input_format": detected,
            "input_sha256": source_revision,
            "input_rows": len(rows),
            "normalized_rows": len(normalized),
            "seed": seed,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "license": license_id.strip(),
            "source_url": source_url,
            "anthropic_benchmark_used_for_training": False,
            "outputs": {
                "training_pool": "training_pool.jsonl",
                "splits": "splits",
            },
        }
        (destination / "import_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    return destination


__all__ = ["SUPPORTED_FORMATS", "import_user_dataset"]
