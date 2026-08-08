"""Download and normalize Anthropic's model-written sycophancy benchmark.

The resulting rows are always marked ``data_role='benchmark'``.  This module
refuses to write under a directory named ``training`` or ``splits`` so the
benchmark cannot accidentally enter the optimization pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sycophancy_rl.data_prep.schema import (
    normalize_option_label,
    stable_example_id,
    write_jsonl,
)

DATASET_ID = "Anthropic/model-written-evals"
DATASET_LICENSE = "CC-BY-4.0"
DEFAULT_OUTPUT = Path("data/benchmarks/anthropic_sycophancy.jsonl")
DEFAULT_MANIFEST = Path("data/benchmarks/anthropic_sycophancy.manifest.json")
_ALLOW_PATTERNS = ("sycophancy/*.jsonl", "sycophancy/*.json", "README.md")
_OPTION_PATTERN = re.compile(
    r"(?im)^\s*\(([AB])\)\s*(.+?)(?=^\s*\([AB]\)\s*|\Z)",
    flags=re.DOTALL,
)


def _question_type_for_file(path: Path) -> tuple[str, str]:
    name = path.name.casefold()
    if "phil" in name:
        return "philosophical", "philosophy"
    if "politic" in name:
        return "subjective", "politics"
    if "nlp" in name:
        return "subjective", "nlp_research"
    return "subjective", "unknown"


def _extract_prompt(row: dict[str, Any]) -> str:
    value = row.get("question", row.get("prompt"))
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for message in value:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        if parts:
            return "\n".join(parts)
    raise ValueError("Row has no supported question/prompt field.")


def _extract_options(prompt: str) -> dict[str, str]:
    options = {}
    for match in _OPTION_PATTERN.finditer(prompt):
        text = " ".join(match.group(2).split())
        text = re.sub(r"\s*(?:Answer|Assistant)\s*:\s*$", "", text, flags=re.IGNORECASE)
        options[match.group(1).upper()] = text
    return options if set(options) == {"A", "B"} else {}


def normalize_anthropic_row(
    row: dict[str, Any],
    *,
    source_file: str,
    source_revision: str,
    row_index: int,
) -> dict[str, Any]:
    """Convert one model-written-evals row to the canonical schema."""

    prompt = _extract_prompt(row)
    matching = normalize_option_label(row.get("answer_matching_behavior"))
    independent = normalize_option_label(row.get("answer_not_matching_behavior"))
    if matching == independent:
        raise ValueError("Matching and non-matching answers must differ.")
    file_path = Path(source_file)
    question_type, topic = _question_type_for_file(file_path)
    return {
        "example_id": stable_example_id(f"anthropic-{file_path.stem}", prompt),
        "source": DATASET_ID,
        "source_revision": source_revision,
        "data_role": "benchmark",
        "prompt": [{"role": "user", "content": prompt}],
        "options": _extract_options(prompt),
        "target_option": independent,
        "independent_option": independent,
        "sycophantic_option": matching,
        "user_preferred_option": matching,
        "user_claim_valid": None,
        "behavior_target": "independent_reasoning",
        "question_type": question_type,
        "topic": topic,
        "metadata": {
            "source_file": source_file.replace("\\", "/"),
            "source_row_index": row_index,
            "answer_position": independent,
            "benchmark_only": True,
        },
    }


def _iter_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        yield from value
    else:
        raise ValueError(f"{path} must contain a JSON array or use JSONL.")


def normalize_snapshot(
    snapshot_dir: Path,
    *,
    source_revision: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize all sycophancy JSON/JSONL files in a local snapshot."""

    files = sorted(
        path
        for path in (snapshot_dir / "sycophancy").rglob("*")
        if path.suffix in {".json", ".jsonl"}
    )
    if not files:
        raise FileNotFoundError(f"No sycophancy JSON files found under {snapshot_dir}.")
    examples: list[dict[str, Any]] = []
    used_files: list[str] = []
    for path in files:
        relative = path.relative_to(snapshot_dir).as_posix()
        used_files.append(relative)
        for index, row in enumerate(_iter_json_rows(path)):
            examples.append(
                normalize_anthropic_row(
                    row,
                    source_file=relative,
                    source_revision=source_revision,
                    row_index=index,
                )
            )
    return examples, used_files


def _assert_benchmark_output(path: Path) -> None:
    lowered_parts = {part.casefold() for part in path.parts}
    if {"training", "splits"} & lowered_parts:
        raise ValueError("Anthropic benchmark output must not be written into training/splits.")


def download_snapshot(cache_dir: Path, revision: str | None) -> tuple[Path, str]:
    """Download a revision-pinned dataset snapshot and return its resolved SHA."""

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install the pinned project dependencies first."
        ) from exc

    api = HfApi()
    resolved_revision = revision or api.dataset_info(DATASET_ID).sha
    if not resolved_revision:
        raise RuntimeError(f"Hugging Face did not return a revision for {DATASET_ID}.")
    local_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=resolved_revision,
        allow_patterns=list(_ALLOW_PATTERNS),
        local_dir=cache_dir,
    )
    return Path(local_path), resolved_revision


def prepare_benchmark(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    cache_dir: Path = Path("data/cache/anthropic_model_written_evals"),
    revision: str | None = None,
    snapshot_dir: Path | None = None,
) -> int:
    """Prepare the evaluation-only benchmark and provenance manifest."""

    _assert_benchmark_output(output_path)
    if snapshot_dir is None:
        snapshot_dir, resolved_revision = download_snapshot(cache_dir, revision)
    else:
        resolved_revision = revision or "local-unpinned"
    examples, source_files = normalize_snapshot(
        snapshot_dir,
        source_revision=resolved_revision,
    )
    count = write_jsonl(output_path, examples)
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_revision": resolved_revision,
        "dataset_license": DATASET_LICENSE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": count,
        "source_files": source_files,
        "data_role": "benchmark",
        "training_prohibited": True,
        "output_file": output_path.as_posix(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/anthropic_model_written_evals"),
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = prepare_benchmark(
        output_path=args.output,
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        revision=args.revision,
        snapshot_dir=args.snapshot_dir,
    )
    print(f"Prepared {count} evaluation-only Anthropic benchmark examples.")


if __name__ == "__main__":
    main()
