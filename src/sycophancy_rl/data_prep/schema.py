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
ANTHROPIC_DATASET_ID = "Anthropic/model-written-evals"


def is_anthropic_row(example: dict[str, Any]) -> bool:
    """Return whether an example comes from the governed Anthropic dataset."""

    return str(example.get("source", "")).casefold() == ANTHROPIC_DATASET_ID.casefold()


def validate_anthropic_governance(example: dict[str, Any]) -> None:
    """Fail closed on Anthropic train/validation/benchmark role metadata.

    Anthropic rows are protected by default.  Development rows require the
    explicit opt-in flag, while held-out benchmark rows must retain their
    benchmark-only marker and may never opt in to optimization.
    """

    if not is_anthropic_row(example):
        return
    metadata = example.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Governed Anthropic rows require object metadata.")
    role = example.get("data_role")
    example_id = example.get("example_id", "<unknown>")
    benchmark_only = metadata.get("benchmark_only")
    opted_in = metadata.get("anthropic_training_opt_in")
    if role in {"training", "validation"}:
        if benchmark_only is not False or opted_in is not True:
            raise ValueError(
                f"Anthropic row {example_id!r}: Evaluation-only benchmark rows reached "
                "the trainer, or Anthropic "
                "development opt-in is missing. Training/validation rows require exactly "
                "metadata.benchmark_only=false and "
                "metadata.anthropic_training_opt_in=true."
            )
        return
    if role == "benchmark":
        if benchmark_only is not True or opted_in is True:
            raise ValueError(
                f"Anthropic benchmark row {example_id!r} requires "
                "metadata.benchmark_only=true "
                "and must not opt in to training."
            )
        return
    raise ValueError(
        f"Anthropic row {example_id!r} supports only training, validation, or benchmark "
        f"roles; got {role!r}."
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


def _validate_behavior_contract(
    *,
    target: str,
    independent: str,
    sycophantic: str | None,
    preferred: str | None,
    claim_valid: bool | None,
    behavior_target: str,
    location: str,
) -> None:
    """Validate that correctness and pressure labels describe one behavior."""

    if independent != target:
        raise ValueError(
            f"{location}: independent_option must equal target_option so the "
            "behavior reward cannot conflict with correctness."
        )
    if sycophantic is not None and sycophantic == independent:
        raise ValueError(
            f"{location}: sycophantic_option and independent_option must differ."
        )
    if behavior_target == "resist_invalid_pressure":
        if claim_valid is not False:
            raise ValueError(
                f"{location}: resist_invalid_pressure requires user_claim_valid=false."
            )
        if preferred is None or preferred == target:
            raise ValueError(
                f"{location}: invalid pressure requires a non-target "
                "user_preferred_option."
            )
        if sycophantic != preferred:
            raise ValueError(
                f"{location}: sycophantic_option must equal user_preferred_option."
            )
    elif behavior_target == "accept_valid_correction":
        if claim_valid is not True:
            raise ValueError(
                f"{location}: accept_valid_correction requires user_claim_valid=true."
            )
        if preferred != target:
            raise ValueError(
                f"{location}: user_preferred_option must equal target_option."
            )
        if sycophantic is not None:
            raise ValueError(
                f"{location}: valid corrections must not define sycophantic_option."
            )
    elif behavior_target == "neutral_answer":
        if preferred is not None or claim_valid is not None or sycophantic is not None:
            raise ValueError(
                f"{location}: neutral behavior cannot define user preference, claim "
                "validity, or a sycophantic option."
            )


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

    behavior_target = example.get("behavior_target", "independent_reasoning")
    if behavior_target not in BEHAVIOR_TARGETS:
        raise ValueError(
            f"'behavior_target' must be one of {BEHAVIOR_TARGETS}, got {behavior_target!r}."
        )
    normalized["behavior_target"] = behavior_target

    optional_text_fields = (
        "base_question_id",
        "gold_answer",
        "gold_rationale",
        "evidence_source",
        "evidence_strength",
        "user_claim",
    )
    for key in optional_text_fields:
        value = example.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"'{key}' must be non-empty text or null.")
        if isinstance(value, str):
            normalized[key] = value.strip()

    claim_valid = example.get("user_claim_valid")
    if claim_valid is not None and not isinstance(claim_valid, bool):
        raise ValueError("'user_claim_valid' must be true, false, or null.")
    normalized["user_claim_valid"] = claim_valid
    target = normalized["target_option"]
    preferred = normalized.get("user_preferred_option")
    _validate_behavior_contract(
        target=target,
        independent=independent,
        sycophantic=sycophantic,
        preferred=preferred,
        claim_valid=claim_valid,
        behavior_target=behavior_target,
        location="example",
    )

    pushbacks = example.get("pushback_turns", [])
    if not isinstance(pushbacks, list):
        raise ValueError("'pushback_turns' must be a list.")
    normalized_pushbacks: list[dict[str, Any]] = []
    for index, pushback in enumerate(pushbacks):
        if not isinstance(pushback, dict):
            raise ValueError(f"pushback_turns[{index}] must be an object.")
        text = pushback.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"pushback_turns[{index}].text must be non-empty text.")
        turn = dict(pushback)
        turn["text"] = text.strip()
        turn_target = normalize_option_label(pushback.get("target_option", target))
        turn_independent = normalize_option_label(
            pushback.get("independent_option", independent)
        )
        turn_claim_valid = pushback.get("user_claim_valid", claim_valid)
        if turn_claim_valid is not None and not isinstance(turn_claim_valid, bool):
            raise ValueError(
                f"pushback_turns[{index}].user_claim_valid must be true, false, or null."
            )
        inferred_behavior = (
            "accept_valid_correction"
            if turn_claim_valid is True
            else "resist_invalid_pressure"
            if turn_claim_valid is False
            else behavior_target
        )
        turn_behavior = pushback.get("behavior_target", inferred_behavior)
        if turn_behavior not in BEHAVIOR_TARGETS:
            raise ValueError(
                f"pushback_turns[{index}].behavior_target must be one of "
                f"{BEHAVIOR_TARGETS}."
            )
        if turn_claim_valid is True:
            default_preferred = turn_target
        elif turn_claim_valid is False:
            default_preferred = "B" if turn_target == "A" else "A"
        else:
            default_preferred = preferred
        turn_preferred = normalize_option_label(
            pushback.get("user_preferred_option", default_preferred), allow_none=True
        )
        default_sycophantic = (
            None
            if turn_claim_valid is True
            else turn_preferred
            if turn_claim_valid is False
            else sycophantic
        )
        turn_sycophantic = normalize_option_label(
            pushback.get("sycophantic_option", default_sycophantic), allow_none=True
        )
        _validate_behavior_contract(
            target=str(turn_target),
            independent=str(turn_independent),
            sycophantic=turn_sycophantic,
            preferred=turn_preferred,
            claim_valid=turn_claim_valid,
            behavior_target=str(turn_behavior),
            location=f"pushback_turns[{index}]",
        )
        turn.update(
            {
                "target_option": turn_target,
                "independent_option": turn_independent,
                "sycophantic_option": turn_sycophantic,
                "user_preferred_option": turn_preferred,
                "user_claim_valid": turn_claim_valid,
                "behavior_target": turn_behavior,
            }
        )
        normalized_pushbacks.append(turn)
    normalized["pushback_turns"] = normalized_pushbacks

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
    validate_anthropic_governance(normalized)
    return normalized


def read_jsonl(path: str | Path, *, expected_role: str | None = None) -> list[dict[str, Any]]:
    """Read and validate canonical JSONL examples."""

    source_path = Path(path)
    rows: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                normalized = validate_example(value, expected_role=expected_role)
                example_id = str(normalized["example_id"])
                if example_id in seen_ids:
                    raise ValueError(
                        f"duplicate example_id {example_id!r}; first seen on "
                        f"line {seen_ids[example_id]}"
                    )
                seen_ids[example_id] = line_number
                rows.append(normalized)
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
    normalized_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        normalized = validate_example(row)
        example_id = str(normalized["example_id"])
        if example_id in seen_ids:
            raise ValueError(f"Refusing to write duplicate example_id {example_id!r}.")
        seen_ids.add(example_id)
        normalized_rows.append(normalized)
    if not normalized_rows:
        raise ValueError("Refusing to write an empty dataset.")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for normalized in normalized_rows:
            handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
    temp_path.replace(output_path)
    return len(normalized_rows)
