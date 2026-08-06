"""Non-destructive helpers for the setup scripts.

The setup scripts must:

* never overwrite an existing ``data/processed/training_pool.jsonl``;
* never silently overwrite existing train/validation/test splits;
* still generate the smoke fixture pool on a fresh clone so tests and the
  smoke preflight have canonical data;
* fail with a clear, recoverable message when the existing data state is
  incomplete or inconsistent.  Error messages recommend moving or backing
  up the bad artifacts, never blindly deleting them.

The helpers in this module are intentionally small, return plain dicts, and
do not perform any file IO besides reading the existing files.  ``inspect_pool``
is the single shared entry point that schema-validates a pool, classifies
it as fixture-only, real-only, mixed or unrecognised, and returns both the
validated rows and the classification.  Every setup path runs through it
so the behaviour stays identical across platforms.

``is_fixture_row`` is the single source of truth for fixture classification
and is shared with the trainer guard so a missing ``metadata.is_fixture``
flag never silently reclassifies a known smoke-fixture source as real data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from sycophancy_rl.data_prep.schema import read_jsonl

TRAINING_POOL_PATH = Path("data/processed/training_pool.jsonl")
SPLIT_DIR = Path("data/splits")
SPLIT_LABELS: tuple[str, ...] = ("training", "validation", "test")

# Schema version of ``split_manifest.json`` written by
# ``split_data.write_splits``.  Bumping this constant forces every consumer
# to make a deliberate compatibility decision.
SUPPORTED_MANIFEST_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})

# Canonical source identifiers that always indicate a local smoke fixture,
# even when ``metadata.is_fixture`` is missing or stripped.
FIXTURE_SOURCES: frozenset[str] = frozenset({"local-smoke-fixture"})

# Manifest fields written by ``split_data.write_splits``.
MANIFEST_PATH: Path = SPLIT_DIR / "split_manifest.json"


class PoolKind(str, Enum):
    """Classification of a training pool's contents."""

    FIXTURE_ONLY = "fixture_only"
    REAL_ONLY = "real_only"
    MIXED = "mixed"
    EMPTY = "empty"


@dataclass(frozen=True)
class PoolInspection:
    """Result of inspecting a training pool.

    ``rows`` are schema-validated.  ``kind`` classifies the pool so the
    caller can branch on it without re-scanning rows.  ``example_ids`` is
    a frozen copy of every example's id, used to cross-check splits.
    """

    rows: tuple[dict[str, Any], ...]
    kind: PoolKind
    example_ids: frozenset[str]


def _split_files(split_dir: Path) -> tuple[tuple[str, Path], ...]:
    return tuple(
        (label, split_dir / f"{name}.jsonl")
        for label, name in zip(
            SPLIT_LABELS, ("train", "validation", "test"), strict=True
        )
    )


def _split_id_files(split_dir: Path) -> tuple[Path, ...]:
    return tuple(split_dir / f"{name}_ids.json" for name in ("train", "validation", "test"))


SPLIT_FILES: tuple[tuple[str, Path], ...] = _split_files(SPLIT_DIR)
SPLIT_ID_FILES: tuple[Path, ...] = _split_id_files(SPLIT_DIR)


def _load_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"{path} must contain a JSON list of example ids; got {type(raw).__name__}."
        )
    for index, value in enumerate(raw):
        if not isinstance(value, str):
            raise ValueError(
                f"{path} position {index} must be a string example id; "
                f"got {type(value).__name__}."
            )
    return raw


def is_fixture_row(row: dict[str, Any]) -> bool:
    """Return ``True`` when a row is a local smoke fixture.

    A row is a fixture when either its ``metadata.is_fixture`` flag is set
    OR its canonical ``source`` identifies it as local smoke-fixture data.
    The check is fail-closed: a missing metadata flag cannot convert a
    known fixture source into real data.
    """

    if not isinstance(row, dict):
        return False
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("is_fixture"):
        return True
    source = row.get("source", "")
    if isinstance(source, str) and source.casefold() in FIXTURE_SOURCES:
        return True
    return False


def is_real_row(row: dict[str, Any]) -> bool:
    """Return ``True`` for rows that are not local smoke fixtures."""

    return not is_fixture_row(row)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_pool_rows_or_raise(pool_path: Path) -> list[dict[str, Any]]:
    """Read and schema-validate a pool, raising with a clear, recoverable
    error message when the file is empty, malformed, or schema-invalid.
    """

    if not pool_path.exists():
        raise FileNotFoundError(
            f"Training pool is missing at {pool_path}. Move any existing "
            f"files out of the way (for example "
            f"`mv {pool_path} {pool_path}.bak-$(date +%s)`) and run "
            "`python -m sycophancy_rl.data_prep.merge_datasets` (smoke fixtures) or "
            "`python -m sycophancy_rl.data_prep.prepare_training_data` (real ARC "
            "data) to regenerate it."
        )
    if pool_path.stat().st_size == 0:
        raise ValueError(
            f"Training pool at {pool_path} is empty. Back up the file "
            f"(`mv {pool_path} {pool_path}.bak-$(date +%s)`) and run "
            "`python -m sycophancy_rl.data_prep.merge_datasets` (smoke fixtures) "
            "or `python -m sycophancy_rl.data_prep.prepare_training_data` (real "
            "ARC data) to regenerate it."
        )

    try:
        rows = read_jsonl(pool_path, expected_role="training")
    except ValueError as exc:
        raise ValueError(
            f"Training pool at {pool_path} is malformed or schema-invalid: "
            f"{exc}. Move the file aside (`mv {pool_path} "
            f"{pool_path}.bak-$(date +%s)`) and regenerate it with "
            "`python -m sycophancy_rl.data_prep.merge_datasets` or "
            "`python -m sycophancy_rl.data_prep.prepare_training_data`."
        ) from exc
    return rows


def inspect_pool(pool_path: Path = TRAINING_POOL_PATH) -> PoolInspection:
    """Read, validate, and classify a training pool.

    Raises a recoverable ``FileNotFoundError`` or ``ValueError`` when the
    pool is missing, empty, malformed, or schema-invalid.  When the pool
    is readable, returns a :class:`PoolInspection` whose ``kind`` is one
    of :class:`PoolKind` so the caller can branch without re-scanning.
    """

    rows = _read_pool_rows_or_raise(pool_path)
    fixture_ids = [row["example_id"] for row in rows if is_fixture_row(row)]
    real_ids = [row["example_id"] for row in rows if is_real_row(row)]
    if fixture_ids and real_ids:
        kind = PoolKind.MIXED
    elif fixture_ids:
        kind = PoolKind.FIXTURE_ONLY
    elif real_ids:
        kind = PoolKind.REAL_ONLY
    else:
        kind = PoolKind.EMPTY
    return PoolInspection(
        rows=tuple(rows),
        kind=kind,
        example_ids=frozenset(row["example_id"] for row in rows),
    )


def _recoverable_move_hint(path: Path) -> str:
    return (
        f"Move the existing artifacts aside (for example "
        f"`mv {path} {path}.bak-$(date +%s)`) before rerunning the setup "
        f"or rebuild them with the appropriate data-prep command."
    )


def assert_training_pool_is_real(pool_path: Path = TRAINING_POOL_PATH) -> None:
    """Fail loudly when the existing pool is missing, empty, malformed,
    schema-invalid, only contains fixtures, or mixes fixtures with real
    data.

    The function never modifies the file.  ``ensure_fixture_pool`` is the
    only entry point that may write a fixture pool, and it never overwrites
    an existing file.
    """

    inspection = inspect_pool(pool_path)
    if inspection.kind is PoolKind.FIXTURE_ONLY:
        sample = ", ".join(
            row["example_id"] for row in inspection.rows[:5]
        )
        raise ValueError(
            f"Training pool at {pool_path} contains only smoke fixtures "
            f"({sample}). Smoke fixtures are for tests and the smoke "
            "preflight only and cannot be used for real training. Run "
            "`python -m sycophancy_rl.data_prep.prepare_training_data` to build a "
            "real training pool."
        )
    if inspection.kind is PoolKind.MIXED:
        fixture_sample = ", ".join(
            row["example_id"] for row in inspection.rows if is_fixture_row(row)
        )[:0] or ", ".join(
            row["example_id"] for row in inspection.rows[:5]
            if is_fixture_row(row)
        )
        raise ValueError(
            f"Training pool at {pool_path} mixes smoke fixtures with real "
            f"data (fixture ids include: {fixture_sample}). Setup refuses "
            "to mix local smoke fixtures with real training rows; rebuild "
            "the pool with `python -m sycophancy_rl.data_prep.prepare_training_data` "
            "for real data only, or with "
            "`python -m sycophancy_rl.data_prep.merge_datasets` for smoke-only "
            "tests."
        )
    if inspection.kind is PoolKind.EMPTY:
        raise ValueError(
            f"Training pool at {pool_path} contains no recognisable rows. "
            f"{_recoverable_move_hint(pool_path)}"
        )


def _split_id_path_for_label(
    split_id_files: tuple[Path, ...], label: str
) -> Path:
    return split_id_files[SPLIT_LABELS.index(label)]


def _split_data_path_for_label(
    split_files: tuple[tuple[str, Path], ...], label: str
) -> Path:
    for entry_label, entry_path in split_files:
        if entry_label == label:
            return entry_path
    raise KeyError(label)


def assert_splits_are_complete_and_consistent(
    split_dir: Path = SPLIT_DIR,
    *,
    pool_path: Path | None = TRAINING_POOL_PATH,
    pool_example_ids: frozenset[str] | None = None,
) -> None:
    """Fail when a split file is missing, malformed, or disagrees with its
    companion IDs file, the split manifest, or the current training pool.

    When ``pool_path`` is supplied (and exists), the function also requires
    ``split_manifest.json`` and verifies its ``source_sha256`` against the
    current pool's SHA-256.  When ``pool_example_ids`` is supplied, the
    function verifies that the union of the split IDs exactly equals the
    pool's example IDs.  When the pool is supplied as
    ``TRAINING_POOL_PATH`` (the default) and that file does not exist, the
    manifest cross-check is skipped so the helper still works in the
    fresh-clone smoke-test case where the manifest has not yet been
    written.
    """

    split_files = _split_files(split_dir)
    split_id_files = _split_id_files(split_dir)
    missing_files: list[str] = []
    for _label, path in split_files:
        if not path.exists():
            missing_files.append(path.as_posix())
            continue
        if path.stat().st_size == 0:
            missing_files.append(path.as_posix())
    for ids_path in split_id_files:
        if not ids_path.exists():
            missing_files.append(ids_path.as_posix())
            continue
        if ids_path.stat().st_size == 0:
            missing_files.append(ids_path.as_posix())
    if missing_files:
        raise FileNotFoundError(
            "Existing data/splits state is incomplete; missing: "
            + ", ".join(missing_files)
            + f". {_recoverable_move_hint(split_dir)}"
        )

    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Existing splits lack {manifest_path}. "
            f"{_recoverable_move_hint(manifest_path)}"
        )
    try:
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{manifest_path} is malformed JSON: {exc}. "
            f"{_recoverable_move_hint(manifest_path)}"
        ) from exc
    if not isinstance(manifest_raw, dict):
        raise ValueError(
            f"{manifest_path} must be a JSON object; got "
            f"{type(manifest_raw).__name__}. "
            f"{_recoverable_move_hint(manifest_path)}"
        )

    schema_version = manifest_raw.get("schema_version")
    if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(
            f"{manifest_path} has unsupported schema_version={schema_version!r}; "
            "this version of sycophancy-rl supports "
            f"schema_version={sorted(SUPPORTED_MANIFEST_SCHEMA_VERSIONS)}. "
            f"{_recoverable_move_hint(manifest_path)}"
        )

    manifest_source = manifest_raw.get("source_path")
    manifest_sha = manifest_raw.get("source_sha256")
    if not isinstance(manifest_source, str):
        raise ValueError(
            f"{manifest_path} is missing the 'source_path' field. "
            f"{_recoverable_move_hint(manifest_path)}"
        )
    if not isinstance(manifest_sha, str):
        raise ValueError(
            f"{manifest_path} is missing the 'source_sha256' field. "
            f"{_recoverable_move_hint(manifest_path)}"
        )

    if pool_path is not None and pool_path.exists():
        current_sha = _file_sha256(pool_path)
        if current_sha != manifest_sha:
            raise ValueError(
                f"Existing splits were produced from a different training "
                f"pool than the one at {pool_path}. Manifest "
                f"source_sha256={manifest_sha[:12]}... does not match "
                f"current pool sha256={current_sha[:12]}.... "
                f"{_recoverable_move_hint(manifest_path)}"
            )

    combined_ids: set[str] = set()
    loaded_split_counts: dict[str, int] = {}
    for (label, path), ids_path in zip(split_files, split_id_files, strict=True):
        try:
            rows = read_jsonl(path, expected_role=label)
        except ValueError as exc:
            raise ValueError(
                f"Split file {path} is malformed or schema-invalid: "
                f"{exc}. {_recoverable_move_hint(path)}"
            ) from exc
        ids = _load_ids(ids_path)
        if ids != [row["example_id"] for row in rows]:
            raise ValueError(
                f"Split file {path} disagrees with its companion IDs file. "
                f"{_recoverable_move_hint(path)}"
            )
        overlap = combined_ids.intersection(row["example_id"] for row in rows)
        if overlap:
            raise ValueError(
                f"Example ids appear in multiple splits: {sorted(overlap)[:5]}. "
                f"{_recoverable_move_hint(split_dir)}"
            )
        combined_ids.update(row["example_id"] for row in rows)
        loaded_split_counts[label] = len(rows)

    manifest_splits = manifest_raw.get("splits", {})
    if not isinstance(manifest_splits, dict):
        raise ValueError(
            f"{manifest_path} has invalid 'splits' field. "
            f"{_recoverable_move_hint(manifest_path)}"
        )
    for label in SPLIT_LABELS:
        info = manifest_splits.get(label)
        if not isinstance(info, dict):
            raise ValueError(
                f"{manifest_path} is missing the '{label}' split info. "
                f"{_recoverable_move_hint(manifest_path)}"
            )
        declared_count = info.get("count")
        declared_data_file = info.get("data_file")
        declared_ids_file = info.get("ids_file")
        if declared_count != loaded_split_counts[label]:
            raise ValueError(
                f"{manifest_path} declares '{label}' count={declared_count} "
                f"but {loaded_split_counts[label]} rows are present. "
                f"{_recoverable_move_hint(manifest_path)}"
            )
        expected_data_path = _split_data_path_for_label(split_files, label)
        expected_ids_path = _split_id_path_for_label(split_id_files, label)
        if not isinstance(declared_data_file, str) or declared_data_file != expected_data_path.name:
            raise ValueError(
                f"{manifest_path} points the '{label}' data file at "
                f"{declared_data_file!r}, but the expected file lives at "
                f"{expected_data_path.name}. {_recoverable_move_hint(manifest_path)}"
            )
        if not isinstance(declared_ids_file, str) or declared_ids_file != expected_ids_path.name:
            raise ValueError(
                f"{manifest_path} points the '{label}' ids file at "
                f"{declared_ids_file!r}, but the expected file lives at "
                f"{expected_ids_path.name}. {_recoverable_move_hint(manifest_path)}"
            )

    if pool_example_ids is not None:
        missing_from_splits = pool_example_ids - combined_ids
        extra_in_splits = combined_ids - pool_example_ids
        if missing_from_splits or extra_in_splits:
            sample_missing = sorted(missing_from_splits)[:3]
            sample_extra = sorted(extra_in_splits)[:3]
            raise ValueError(
                f"Existing splits do not exactly match the training pool. "
                f"Missing from splits: {sample_missing}; extra in splits: "
                f"{sample_extra}. {_recoverable_move_hint(split_dir)}"
            )


def ensure_fixture_pool(
    pool_path: Path = TRAINING_POOL_PATH,
) -> str:
    """Write the smoke fixture pool only if no pool exists.

    When a pool already exists, the function runs :func:`inspect_pool`
    first to validate it before reporting "preserved", so a broken or
    schema-invalid pool is surfaced instead of silently carried forward.
    Returns a human-readable status string for the setup script to print.
    Never overwrites an existing pool.
    """

    if pool_path.exists():
        try:
            inspection = inspect_pool(pool_path)
        except (FileNotFoundError, ValueError) as exc:
            return (
                f"Existing training pool at {pool_path} failed inspection: "
                f"{exc}. {_recoverable_move_hint(pool_path)}"
            )
        kind_label = {
            PoolKind.FIXTURE_ONLY: "smoke fixture",
            PoolKind.REAL_ONLY: "real",
            PoolKind.MIXED: "mixed fixture+real",
            PoolKind.EMPTY: "empty",
        }[inspection.kind]
        return (
            f"Existing training pool preserved at {pool_path} "
            f"({kind_label}, {len(inspection.rows)} rows)."
        )
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    from sycophancy_rl.data_prep.merge_datasets import build_fixture_examples
    from sycophancy_rl.data_prep.schema import write_jsonl

    count = write_jsonl(pool_path, build_fixture_examples())
    return (
        f"Generated {count} smoke fixture examples at {pool_path} "
        "for tests and the smoke preflight."
    )


def ensure_splits(
    *,
    split_dir: Path = SPLIT_DIR,
    pool_path: Path = TRAINING_POOL_PATH,
    seed: int = 42,
) -> str:
    """Generate splits only when they are missing; otherwise preserve them.

    Always validates the training pool via :func:`inspect_pool` first so
    mixed or malformed pools are surfaced before any split work.  Returns
    a human-readable status string.  Raises ``FileNotFoundError`` if the
    training pool is missing or only contains fixtures, raises
    ``FileExistsError`` if individual split files are missing while others
    exist (i.e. the existing state is inconsistent and must be cleaned up
    rather than silently completed), and raises ``ValueError`` if the
    existing splits were produced from a different training pool than the
    one currently on disk.
    """

    inspection = inspect_pool(pool_path)
    if inspection.kind is PoolKind.EMPTY:
        raise ValueError(
            f"Training pool at {pool_path} contains no recognisable rows. "
            f"{_recoverable_move_hint(pool_path)}"
        )
    if inspection.kind is PoolKind.MIXED:
        raise ValueError(
            f"Training pool at {pool_path} mixes smoke fixtures with real "
            "data. Setup refuses to mix the two kinds; rebuild the pool "
            "with `python -m sycophancy_rl.data_prep.prepare_training_data` (real "
            "data) or `python -m sycophancy_rl.data_prep.merge_datasets` (smoke "
            "fixtures) and rerun."
        )

    rows = list(inspection.rows)

    if not split_dir.exists():
        from sycophancy_rl.data_prep.split_data import split_examples, write_splits

        splits = split_examples(rows, seed=seed)
        write_splits(splits, split_dir, seed=seed, source_path=pool_path)
        return (
            "Generated train/validation/test splits under "
            f"{split_dir} from {pool_path}."
        )

    split_files = _split_files(split_dir)
    existing = [path for _, path in split_files if path.exists()]
    if existing and len(existing) != len(split_files):
        missing = [
            path.as_posix()
            for _, path in split_files
            if not path.exists()
        ]
        raise FileExistsError(
            "Some split files already exist but others are missing: "
            + ", ".join(missing)
            + f". {_recoverable_move_hint(split_dir)}"
        )

    if all(path.exists() for _, path in split_files):
        assert_splits_are_complete_and_consistent(
            split_dir,
            pool_path=pool_path,
            pool_example_ids=inspection.example_ids,
        )
        return (
            "Existing train/validation/test splits preserved at "
            f"{split_dir} (not regenerated)."
        )

    from sycophancy_rl.data_prep.split_data import split_examples, write_splits

    splits = split_examples(rows, seed=seed)
    write_splits(splits, split_dir, seed=seed, source_path=pool_path)
    return f"Generated splits under {split_dir} from {pool_path}."


def main() -> None:
    """Subcommand entry point used by scripts/setup.{ps1,sh}."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "ensure-fixtures",
        help="Write the smoke fixture pool only if no pool exists yet.",
    )
    sub.add_parser(
        "ensure-splits",
        help=(
            "Generate train/validation/test splits only when they are missing; "
            "verify and preserve an existing complete split directory."
        ),
    )
    args = parser.parse_args()
    if args.command == "ensure-fixtures":
        print(ensure_fixture_pool())
        return
    if args.command == "ensure-splits":
        print(ensure_splits())
        return
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
