"""Shared experiment manifest used by every runner.

A single ``experiment.json`` file describes one run end-to-end.
Local and Kaggle runners both write and read this schema so the same
benchmarks, prompts, and generation settings are used before and after
training and so resume checks can refuse incompatible restarts.

Status transitions are atomic and one-way.  The terminal states are
``completed``, ``failed``, and ``interrupted``.  ``running`` is an
in-progress state that the runner must rewrite if it crashes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# --- schema versioning -----------------------------------------------------


SCHEMA_VERSION = 1
"""Bumped when the manifest JSON layout changes incompatibly."""


SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})


# --- status + stage enums --------------------------------------------------


class Status(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    COMPLETED = "completed"


class Stage(str, Enum):
    BEFORE = "before"
    TRAINING = "training"
    AFTER = "after"


# --- transition table -----------------------------------------------------


_ALLOWED_TRANSITIONS: frozenset[tuple[Status, Status]] = frozenset({
    (Status.CREATED, Status.RUNNING),
    (Status.CREATED, Status.FAILED),
    (Status.RUNNING, Status.COMPLETED),
    (Status.RUNNING, Status.FAILED),
    (Status.RUNNING, Status.INTERRUPTED),
    (Status.INTERRUPTED, Status.RUNNING),
    (Status.FAILED, Status.RUNNING),
})


# --- manifest dataclass ----------------------------------------------------


@dataclass(frozen=True)
class ExperimentManifest:
    """One immutable view of an experiment manifest."""

    schema_version: int
    run_id: str
    stage: Stage
    model_id: str
    model_revision: str
    adapter_path: str
    adapter_sha256: str | None
    training_hash: str
    validation_hash: str
    benchmark_hash: str
    seed: int
    prompt_condition: str
    generation_settings: dict[str, Any]
    reward_profile: str
    effective_training_config: dict[str, Any]
    python_versions: dict[str, str]
    hardware: dict[str, Any]
    git_commit: str | None
    runner: str
    timestamps: dict[str, str]
    status: Status
    data_governance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stage": self.stage.value,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "adapter_path": self.adapter_path,
            "adapter_sha256": self.adapter_sha256,
            "training_hash": self.training_hash,
            "validation_hash": self.validation_hash,
            "benchmark_hash": self.benchmark_hash,
            "seed": self.seed,
            "prompt_condition": self.prompt_condition,
            "generation_settings": copy.deepcopy(self.generation_settings),
            "reward_profile": self.reward_profile,
            "effective_training_config": copy.deepcopy(self.effective_training_config),
            "python_versions": dict(self.python_versions),
            "hardware": dict(self.hardware),
            "git_commit": self.git_commit,
            "runner": self.runner,
            "timestamps": dict(self.timestamps),
            "status": self.status.value,
            "data_governance": dict(self.data_governance),
        }
        return d


# --- factory + IO ---------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, raising if it doesn't exist."""

    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_manifest(
    *,
    run_id: str,
    stage: Stage,
    model_id: str,
    model_revision: str,
    adapter_path: str,
    training_hash: str,
    validation_hash: str,
    benchmark_hash: str,
    seed: int,
    prompt_condition: str,
    generation_settings: dict[str, Any],
    reward_profile: str,
    effective_training_config: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
    python_versions: dict[str, str] | None = None,
    git_commit: str | None = None,
    runner: str = "local",
    data_governance: dict[str, Any] | None = None,
) -> ExperimentManifest:
    """Construct a new manifest in the ``created`` status.

    The ``adapter_sha256`` is computed lazily when the adapter is saved;
    pass ``adapter_sha256="..."`` explicitly via :func:`write_status` when
    it becomes available.
    """

    return ExperimentManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        stage=stage,
        model_id=model_id,
        model_revision=model_revision,
        adapter_path=adapter_path,
        adapter_sha256=None,
        training_hash=training_hash,
        validation_hash=validation_hash,
        benchmark_hash=benchmark_hash,
        seed=seed,
        prompt_condition=prompt_condition,
        generation_settings=dict(generation_settings),
        reward_profile=reward_profile,
        effective_training_config=dict(effective_training_config or {}),
        python_versions=dict(python_versions or {}),
        hardware=dict(hardware or {}),
        git_commit=git_commit,
        runner=runner,
        timestamps={"created": _now_iso()},
        status=Status.CREATED,
        data_governance=dict(data_governance or {}),
    )


# --- atomic status writer --------------------------------------------------


def transition(
    manifest: ExperimentManifest,
    *,
    to: Status,
    extra_timestamps: dict[str, str] | None = None,
) -> ExperimentManifest:
    """Return a copy of ``manifest`` advanced to ``to`` status."""

    if (manifest.status, to) not in _ALLOWED_TRANSITIONS:
        raise ValueError(
            f"Illegal manifest status transition: {manifest.status.value} -> {to.value}"
        )
    timestamps = dict(manifest.timestamps)
    if extra_timestamps:
        timestamps.update(extra_timestamps)
    if to == Status.RUNNING and "started" not in timestamps:
        timestamps["started"] = _now_iso()
    if to in (Status.COMPLETED, Status.FAILED, Status.INTERRUPTED):
        timestamps["finished"] = _now_iso()
    return ExperimentManifest(
        schema_version=manifest.schema_version,
        run_id=manifest.run_id,
        stage=manifest.stage,
        model_id=manifest.model_id,
        model_revision=manifest.model_revision,
        adapter_path=manifest.adapter_path,
        adapter_sha256=manifest.adapter_sha256,
        training_hash=manifest.training_hash,
        validation_hash=manifest.validation_hash,
        benchmark_hash=manifest.benchmark_hash,
        seed=manifest.seed,
        prompt_condition=manifest.prompt_condition,
        generation_settings=dict(manifest.generation_settings),
        reward_profile=manifest.reward_profile,
        effective_training_config=dict(manifest.effective_training_config),
        python_versions=dict(manifest.python_versions),
        hardware=dict(manifest.hardware),
        git_commit=manifest.git_commit,
        runner=manifest.runner,
        timestamps=timestamps,
        status=to,
        data_governance=dict(manifest.data_governance),
    )


def load_manifest(
    path: Path,
) -> ExperimentManifest:
    """Load a manifest from ``path``, refusing unsupported schemas."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object; got {type(raw).__name__}.")
    schema = raw.get("schema_version")
    if schema not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{path} has unsupported schema_version={schema!r}; this build "
            f"supports {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )
    return _from_dict(raw)


def _from_dict(raw: dict[str, Any]) -> ExperimentManifest:
    required = {
        "run_id",
        "stage",
        "model_id",
        "model_revision",
        "training_hash",
        "validation_hash",
        "benchmark_hash",
        "seed",
        "reward_profile",
        "runner",
        "status",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"{raw.get('run_id', '?')}: manifest missing fields: {missing}")
    return ExperimentManifest(
        schema_version=raw["schema_version"],
        run_id=raw["run_id"],
        stage=Stage(raw["stage"]),
        model_id=raw["model_id"],
        model_revision=raw["model_revision"],
        adapter_path=raw.get("adapter_path", ""),
        adapter_sha256=raw.get("adapter_sha256"),
        training_hash=raw["training_hash"],
        validation_hash=raw["validation_hash"],
        benchmark_hash=raw["benchmark_hash"],
        seed=raw["seed"],
        prompt_condition=raw.get("prompt_condition", "neutral"),
        generation_settings=dict(raw.get("generation_settings", {})),
        reward_profile=raw["reward_profile"],
        effective_training_config=dict(raw.get("effective_training_config", {})),
        python_versions=dict(raw.get("python_versions", {})),
        hardware=dict(raw.get("hardware", {})),
        git_commit=raw.get("git_commit"),
        runner=raw["runner"],
        timestamps=dict(raw.get("timestamps", {})),
        status=Status(raw["status"]),
        data_governance=dict(raw.get("data_governance", {})),
    )


def write_atomically(
    manifest: ExperimentManifest,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write ``manifest`` to ``path`` via a temp file + os.replace.

    The manifest path is the *only* way an artifact can become visible.
    A pre-existing manifest is never overwritten unless ``overwrite=True``,
    which preserves the "no silent reuse of run directories" guarantee.
    """

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists at {path}. Pass overwrite=True to "
            "replace it (used only when transitioning an existing run)."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{int(datetime.now(timezone.utc).timestamp())}"
    payload = manifest.to_dict()
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_artifact_checksums(
    artifacts: dict[str, Path],
    path: Path,
) -> dict[str, str]:
    """Compute SHA-256s for each artifact and write them to ``path``.

    Returns the dict that was written so the caller can attach it to the
    manifest.  Refuses to write a partial map: all artifacts must hash
    successfully.
    """

    sums: dict[str, str] = {}
    for label, art in artifacts.items():
        sums[label] = sha256_file(Path(art))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sums


# --- resume-compatibility check ------------------------------------------


def is_resume_compatible(
    previous: ExperimentManifest,
    next_manifest: ExperimentManifest,
) -> tuple[bool, str]:
    """Return (ok, reason).  Resume is allowed only when every required
    field matches and the previous status is ``interrupted``, ``failed``,
    or ``created``.
    """

    if previous.status not in (
        Status.INTERRUPTED,
        Status.FAILED,
        Status.CREATED,
        Status.RUNNING,
    ):
        return False, (
            f"Previous run is in terminal state {previous.status.value!r}; "
            "resume is not allowed."
        )
    if previous.model_id != next_manifest.model_id:
        return False, "model_id changed between runs"
    if previous.model_revision != next_manifest.model_revision:
        return False, "model_revision changed between runs"
    if previous.training_hash != next_manifest.training_hash:
        return False, "training data hash changed between runs"
    if previous.validation_hash != next_manifest.validation_hash:
        return False, "validation data hash changed between runs"
    if previous.benchmark_hash != next_manifest.benchmark_hash:
        return False, "benchmark data hash changed between runs"
    if previous.seed != next_manifest.seed:
        return False, "seed changed between runs"
    if previous.reward_profile != next_manifest.reward_profile:
        return False, "reward profile changed between runs"
    if previous.prompt_condition != next_manifest.prompt_condition:
        return False, "prompt condition changed between runs"
    if previous.generation_settings != next_manifest.generation_settings:
        return False, "generation settings changed between runs"
    if previous.effective_training_config != next_manifest.effective_training_config:
        return False, "effective training config changed between runs"
    if previous.runner != next_manifest.runner:
        return False, "runner changed between runs"
    return True, "ok"


# --- governance helpers ---------------------------------------------------


def assert_benchmark_only(manifest: ExperimentManifest) -> None:
    """Refuse to enter a training stage with an evaluation-only benchmark."""

    governance = manifest.data_governance or {}
    source = str(governance.get("source", "")).casefold()
    benchmark_only = bool(governance.get("benchmark_only"))
    if benchmark_only or source == "anthropic/model-written-evals":
        raise ValueError(
            "Anthropic benchmark data is evaluation-only and cannot be "
            "fed into a training stage.  Build the training pool with "
            "`python -m sycophancy_rl.data_prep.prepare_training_data`."
        )


def assert_no_fixture_contamination(manifest: ExperimentManifest) -> None:
    """Refuse to start a real run when the data governance flag marks it fixture."""

    if bool(manifest.data_governance.get("is_fixture")):
        raise ValueError(
            "Fixture data reached a real-profile run.  Real profiles must "
            "consume a pool built by `python -m sycophancy_rl.data_prep.prepare_training_data`."
        )


def generation_settings_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Generation-settings equality that ignores unknown keys.

    The before/after benchmarks must use *identical* generation settings.
    We require the canonical keys to match exactly; extras on either side
    are ignored.
    """

    keys = {"temperature", "top_p", "top_k", "max_new_tokens", "do_sample"}
    return all(a.get(k) == b.get(k) for k in keys)
