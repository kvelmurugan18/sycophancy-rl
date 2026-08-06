"""Canonical schemas and validation helpers for training and evaluation data.

The project uses two distinct data roles:

* ``training`` examples may be optimized against.
* ``benchmark`` examples are evaluation-only and must never enter a trainer.

The Anthropic sycophancy datasets describe answers that match or do not match
the user's stated view.  They do not generally establish objective truth, so
the canonical names are ``sycophantic_option`` and ``independent_option``.
Factual training examples additionally carry ``target_option`` and
``user_claim_valid`` so agreeing with a correct user can be rewarded.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

OPTION_LABELS = ("A", "B")
DATA_ROLES = ("training", "validation", "test", "benchmark")
QUESTION_TYPES = ("objective", "subjective", "moral", "philosophical", "ambiguous")
BEHAVIOR_TARGETS = (
    "independent_reasoning",
    "resist_invalid_pressure",
    "accept_valid_correction",
    "neutral_answer",
)


def normalize_option_label(value: object, *, allow_none: bool = False) -> str | None:
    """Return ``A`` or ``B`` from common label representations.

    Values such as ``"(A)"``, ``"a"`` and ``"Answer: B"`` are accepted.
    Arbitrary text containing a label is rejected to avoid silently converting
    a full generated response into a gold label.
    """

    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Option label must be a string, got {type(value).__name__}.")

    match = re.fullmatch(
        r"\s*(?:(?:final\s+)?answer\s*:\s*)?\(?\s*([AB])\s*\)?\s*[.!]?\s*",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unsupported option label: {value!r}; expected A or B.")
    return match.group(1).upper()


def stable_example_id(source: str, prompt: str, *, namespace: str = "sycophancy-rl") -> str:
    """Create a stable, content-derived example identifier."""

    normalized = " ".join(prompt.split()).casefold()
    digest = hashlib.sha256(f"{namespace}\0{source}\0{normalized}".encode()).hexdigest()
    return f"{source.replace('/', '-')}-{digest[:16]}"


def normalized_prompt_fingerprint(prompt: str) -> str:
    """Return a hash used for exact leakage and duplicate checks."""

    normalized = re.sub(r"\s+", " ", prompt).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_messages(messages: object) -> list[dict[str, str]]:
    """Validate and normalize an OpenAI-style chat message list."""

    if not isinstance(messages, list) or not messages:
        raise ValueError("'prompt' must be a non-empty list of chat messages.")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"prompt[{index}] must be an object.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"prompt[{index}].role has unsupported value {role!r}.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"prompt[{index}].content must be non-empty text.")
        normalized.append({"role": role, "content": content.strip()})
    if normalized[-1]["role"] != "user":
        raise ValueError("The last prompt message must have role='user'.")
    return normalized


def prompt_text(example: dict[str, Any]) -> str:
    """Return the user-visible text used for duplicate and leakage checks."""

    messages = validate_messages(example.get("prompt"))
    return "\n".join(message["content"] for message in messages if message["role"] == "user")


def validate_example(example: dict[str, Any], *, expected_role: str | None = None) -> dict[str, Any]:
    """Validate one canonical example and return a normalized copy."""

    if not isinstance(example, dict):
        raise ValueError("Each example must be a JSON object.")

    required_text = ("example_id", "source")
    for key in required_text:
        if not isinstance(example.get(key), str) or not example[key].strip():
            raise ValueError(f"'{key}' must be non-empty text.")

    role = example.get("data_role")
    if role not in DATA_ROLES:
        raise ValueError(f"'data_role' must be one of {DATA_ROLES}, got {role!r}.")
    if expected_role is not None and role != expected_role:
        raise ValueError(
            f"Expected data_role={expected_role!r}, got {role!r} for {example['example_id']}."
        )

    question_type = example.get("question_type", "objective")
    if question_type not in QUESTION_TYPES:
        raise ValueError(
            f"'question_type' must be one of {QUESTION_TYPES}, got {question_type!r}."
        )

    normalized = dict(example)
    normalized["prompt"] = validate_messages(example.get("prompt"))
    normalized["question_type"] = question_type

    for key in ("target_option", "independent_option", "sycophantic_option", "user_preferred_option"):
        if key in example:
            normalized[key] = normalize_option_label(example[key], allow_none=True)

    if normalized.get("target_option") is None:
        raise ValueError("'target_option' is required and must be A or B.")
    if normalized.get("independent_option") is None:
        raise ValueError("'independent_option' is required and must be A or B.")

    sycophantic = normalized.get("sycophantic_option")
    independent = normalized["independent_option"]
    if sycophantic is not None and sycophantic == independent:
        raise ValueError("'sycophantic_option' and 'independent_option' must differ.")

    behavior_target = example.get("behavior_target", "independent_reasoning")
    if behavior_target not in BEHAVIOR_TARGETS:
        raise ValueError(
            f"'behavior_target' must be one of {BEHAVIOR_TARGETS}, got {behavior_target!r}."
        )
    normalized["behavior_target"] = behavior_target

    options = example.get("options", {})
    if options:
        if not isinstance(options, dict):
            raise ValueError("'options' must be an object keyed by A and B.")
        normalized_options = {
            normalize_option_label(label): str(text).strip() for label, text in options.items()
        }
        if set(normalized_options) != set(OPTION_LABELS):
            raise ValueError("'options' must contain exactly A and B.")
        if not all(normalized_options.values()):
            raise ValueError("Option text must not be empty.")
        normalized["options"] = normalized_options

    metadata = example.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("'metadata' must be an object.")
    normalized["metadata"] = metadata
    return normalized


def read_jsonl(path: str | Path, *, expected_role: str | None = None) -> list[dict[str, Any]]:
    """Read and validate canonical JSONL examples."""

    source_path = Path(path)
    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                rows.append(validate_example(value, expected_role=expected_role))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{source_path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{source_path} contains no examples.")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write canonical JSONL atomically and return the number of rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            normalized = validate_example(row)
            handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    if count == 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("Refusing to write an empty dataset.")
    temp_path.replace(output_path)
    return count
