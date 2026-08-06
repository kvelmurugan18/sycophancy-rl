"""Safe JSON/JSONL I/O for raw responses and derived reports."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array or JSONL file with useful empty-file errors."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{source} is empty; expected JSON or JSONL records.")
    if source.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object.")
            records.append(value)
        return records
    value = json.loads(text)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{source} must contain a JSON array of objects.")
    return value


def write_records(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Atomically write raw evaluation records as JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("Refusing to write an empty evaluation result.")
    temporary.replace(destination)
    return count


def write_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
