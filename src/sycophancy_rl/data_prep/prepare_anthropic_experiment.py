"""Prepare the governed 80/10/10 Anthropic experiment split.

The source contains 30,168 examples.  A deterministic seeded permutation is
partitioned into 24,134 training, 3,017 validation, and 3,017 held-out
benchmark rows.  Training and validation are explicitly opted in; benchmark
rows remain protected.  IDs, counts, hashes, and zero-overlap checks are
recorded in one manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sycophancy_rl.data_prep.prepare_anthropic_benchmark import (
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    download_snapshot,
    normalize_snapshot,
)
from sycophancy_rl.data_prep.schema import validate_example, write_jsonl

TOTAL_COUNT = 30_168
TRAIN_COUNT = 24_134
VALIDATION_COUNT = 3_017
BENCHMARK_COUNT = 3_017
DEFAULT_OUTPUT_DIR = Path("data/anthropic_experiment")


def _ids_sha256(ids: set[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split_rows(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Reject duplicate/overlapping IDs and return auditable split identity."""

    expected_roles = {"training", "validation", "benchmark"}
    if set(splits) != expected_roles:
        raise ValueError(f"Expected split roles {sorted(expected_roles)}; got {sorted(splits)}.")
    ids_by_role: dict[str, set[str]] = {}
    for role, rows in splits.items():
        ids = [str(row["example_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
            raise ValueError(f"Duplicate example_id in {role}: {duplicates[:5]}")
        for row in rows:
            validate_example(row, expected_role=role)
        ids_by_role[role] = set(ids)
    overlap = {
        "training_validation": sorted(ids_by_role["training"] & ids_by_role["validation"]),
        "training_benchmark": sorted(ids_by_role["training"] & ids_by_role["benchmark"]),
        "validation_benchmark": sorted(ids_by_role["validation"] & ids_by_role["benchmark"]),
    }
    if any(overlap.values()):
        samples = {key: value[:5] for key, value in overlap.items() if value}
        raise ValueError(f"Anthropic split overlap detected: {samples}")
    return {
        "counts": {role: len(rows) for role, rows in splits.items()},
        "example_id_sha256": {
            role: _ids_sha256(ids) for role, ids in ids_by_role.items()
        },
        "overlap_counts": {key: len(value) for key, value in overlap.items()},
    }


def split_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Return the exact governed deterministic split without writing files."""

    if len(rows) != TOTAL_COUNT:
        raise ValueError(
            f"Expected exactly {TOTAL_COUNT:,} Anthropic rows; found {len(rows):,}. "
            "Refusing to silently truncate or pad the experiment."
        )
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Normalized Anthropic source contains duplicate example_id values.")
    ordered = sorted(rows, key=lambda row: str(row["example_id"]))
    random.Random(seed).shuffle(ordered)
    raw_splits = {
        "training": ordered[:TRAIN_COUNT],
        "validation": ordered[TRAIN_COUNT : TRAIN_COUNT + VALIDATION_COUNT],
        "benchmark": ordered[TRAIN_COUNT + VALIDATION_COUNT :],
    }
    governed: dict[str, list[dict[str, Any]]] = {}
    for role, role_rows in raw_splits.items():
        governed[role] = []
        for source_row in role_rows:
            row = dict(source_row)
            row["data_role"] = role
            metadata = dict(row.get("metadata", {}))
            metadata.update(
                {
                    "benchmark_only": role == "benchmark",
                    "anthropic_training_opt_in": role in {"training", "validation"},
                    "split_seed": seed,
                }
            )
            row["metadata"] = metadata
            governed[role].append(validate_example(row, expected_role=role))
    audit = audit_split_rows(governed)
    if audit["counts"] != {
        "training": TRAIN_COUNT,
        "validation": VALIDATION_COUNT,
        "benchmark": BENCHMARK_COUNT,
    }:
        raise AssertionError(f"Unexpected split counts: {audit['counts']}")
    return governed


def prepare_experiment(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_dir: Path = Path("data/cache/anthropic_model_written_evals"),
    revision: str | None = None,
    snapshot_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Normalize, split, write, and audit the governed Anthropic experiment."""

    if snapshot_dir is None:
        snapshot_dir, resolved_revision = download_snapshot(cache_dir, revision)
    else:
        if not revision:
            raise ValueError("A local --snapshot-dir requires its exact --revision commit SHA.")
        resolved_revision = revision
    rows, source_files = normalize_snapshot(snapshot_dir, source_revision=resolved_revision)
    splits = split_rows(rows, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "training": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "benchmark": output_dir / "benchmark.jsonl",
    }
    for role, path in paths.items():
        write_jsonl(path, splits[role])
    audit = audit_split_rows(splits)
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_revision": resolved_revision,
        "default_dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "source_row_count": len(rows),
        "source_files": source_files,
        **audit,
        "files": {role: path.as_posix() for role, path in paths.items()},
        "file_sha256": {role: _file_sha256(path) for role, path in paths.items()},
        "governance": {
            "development_requires_anthropic_training_opt_in": True,
            "benchmark_only": True,
            "held_out_benchmark_used_for_training": False,
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/anthropic_model_written_evals"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = prepare_experiment(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        revision=args.revision,
        snapshot_dir=args.snapshot_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
