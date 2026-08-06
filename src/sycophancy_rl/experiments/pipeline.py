"""Shared, dependency-injected experiment pipeline.

This module is the single orchestration entry point shared by the local
trainer and the Kaggle runner.  It models a complete experiment as a
fixed, ordered list of stages:

    1. validate_environment
    2. validate_data_and_leakage
    3. benchmark_before
    4. train
    5. benchmark_after
    6. compare_before_after
    7. finalize_artifacts

The plan is a small dataclass :class:`ExperimentPlan`; each stage is a
callable that takes the plan and a :class:`StageContext` and returns
the updated context.  Tests inject fake callables so the orchestration
can be exercised without loading a model, downloading data, or starting
a training run.

Three execution modes are supported:

* ``plan``   — validate the plan and print the intended stage list,
                never invoke any stage callable.
* ``dry-run`` — print the exact commands, paths, hashes, and artifacts
                the pipeline *would* produce without invoking any stage.
* ``execute`` — actually invoke the stage callables in order.

Real execution is rejected unless the caller passes
``mode=EXECUTE_MODE`` explicitly.  ``dry-run`` never silently becomes
``execute``.

Experiment integrity guards (in :class:`Pipeline.run`) refuse to
start a real run when the data is fixture-only, when benchmark rows
slipped into training, when a publishable run lacks a pinned model
revision, or when the before/after benchmark plans drift apart.
"""

from __future__ import annotations

import abc
import dataclasses
import gc
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# --- execution modes -------------------------------------------------------


class ExecutionMode(str, Enum):
    PLAN = "plan"
    DRY_RUN = "dry-run"
    EXECUTE = "execute"


# --- stage names ----------------------------------------------------------


class StageName(str, Enum):
    VALIDATE_ENVIRONMENT = "validate_environment"
    VALIDATE_DATA_AND_LEAKAGE = "validate_data_and_leakage"
    BENCHMARK_BEFORE = "benchmark_before"
    TRAIN = "train"
    BENCHMARK_AFTER = "benchmark_after"
    COMPARE_BEFORE_AFTER = "compare_before_after"
    FINALIZE_ARTIFACTS = "finalize_artifacts"


# Default order used by every run.
DEFAULT_STAGE_ORDER: tuple[StageName, ...] = (
    StageName.VALIDATE_ENVIRONMENT,
    StageName.VALIDATE_DATA_AND_LEAKAGE,
    StageName.BENCHMARK_BEFORE,
    StageName.TRAIN,
    StageName.BENCHMARK_AFTER,
    StageName.COMPARE_BEFORE_AFTER,
    StageName.FINALIZE_ARTIFACTS,
)


# --- data plan -------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentPlan:
    """Immutable description of one end-to-end experiment.

    Every field is documented so a saved ``experiment.json`` can be
    audited by humans without re-running the code.  The plan is the
    single source of truth for both the local trainer and the Kaggle
    runner.
    """

    run_id: str
    model_id: str
    model_revision: str
    training_path: Path
    validation_path: Path
    benchmark_path: Path
    seed: int
    training_profile: str
    reward_profile: str
    prompt_condition: str
    prompt_variants: tuple[str, ...]
    generation_settings: Mapping[str, Any]
    output_root: Path
    checkpoint_dir: Path
    runner: str
    publishable: bool
    max_examples: int | None = None
    batch_size: int = 4
    load_in_4bit: bool = True
    allow_cpu: bool = False
    allow_unpinned_model: bool = False
    custom_supports_4bit: bool = False
    custom_lora_targets: tuple[str, ...] = ("all-linear",)
    custom_context_length: int = 2048
    custom_min_vram_gib: float = 0.0
    resume: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def frozen_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the plan.

        The plan itself is frozen; the returned dict is a deep copy so
        the caller can mutate without affecting the immutable plan.
        """

        payload = dataclasses.asdict(self)
        payload["training_path"] = str(self.training_path)
        payload["validation_path"] = str(self.validation_path)
        payload["benchmark_path"] = str(self.benchmark_path)
        payload["output_root"] = str(self.output_root)
        payload["checkpoint_dir"] = str(self.checkpoint_dir)
        payload["prompt_variants"] = list(self.prompt_variants)
        payload["custom_lora_targets"] = list(self.custom_lora_targets)
        payload["generation_settings"] = dict(self.generation_settings)
        payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentPlan:
        """Load a plan written by :meth:`frozen_dict` with no implicit defaults."""

        required = {
            item.name
            for item in dataclasses.fields(cls)
            if item.default is dataclasses.MISSING
            and item.default_factory is dataclasses.MISSING
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Experiment plan is missing fields: {missing}")
        values = dict(payload)
        for key in (
            "training_path",
            "validation_path",
            "benchmark_path",
            "output_root",
            "checkpoint_dir",
        ):
            values[key] = Path(values[key])
        values["prompt_variants"] = tuple(values["prompt_variants"])
        values["custom_lora_targets"] = tuple(
            values.get("custom_lora_targets", ("all-linear",))
        )
        values["generation_settings"] = dict(values["generation_settings"])
        values["extra"] = dict(values.get("extra", {}))
        return cls(**values)

    def as_manifest_dict(self) -> dict[str, Any]:
        """Return a manifest-shaped dict with hashes resolved at runtime."""

        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "training_path": str(self.training_path),
            "validation_path": str(self.validation_path),
            "benchmark_path": str(self.benchmark_path),
            "seed": self.seed,
            "training_profile": self.training_profile,
            "reward_profile": self.reward_profile,
            "prompt_condition": self.prompt_condition,
            "prompt_variants": list(self.prompt_variants),
            "generation_settings": dict(self.generation_settings),
            "output_root": str(self.output_root),
            "checkpoint_dir": str(self.checkpoint_dir),
            "runner": self.runner,
            "publishable": self.publishable,
            "max_examples": self.max_examples,
            "batch_size": self.batch_size,
            "load_in_4bit": self.load_in_4bit,
            "allow_cpu": self.allow_cpu,
            "allow_unpinned_model": self.allow_unpinned_model,
            "custom_supports_4bit": self.custom_supports_4bit,
            "custom_lora_targets": list(self.custom_lora_targets),
            "custom_context_length": self.custom_context_length,
            "custom_min_vram_gib": self.custom_min_vram_gib,
            "resume": self.resume,
        }

    @property
    def plan_sha256(self) -> str:
        frozen = self.frozen_dict()
        # Resume is an execution control, not an experiment parameter.
        frozen.pop("resume", None)
        payload = json.dumps(frozen, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- runtime context ------------------------------------------------------


@dataclass
class StageContext:
    """Mutable context shared by every stage.

    Stages must not raise to signal a recoverable failure — they should
    populate ``errors`` and let the orchestrator decide.  Critical
    failures (e.g. data guard refused) raise :class:`PipelineError` so
    the orchestrator records a clean ``failed`` status and the
    exception type stays informative.
    """

    plan: ExperimentPlan
    mode: ExecutionMode
    manifest: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    training_hash: str = ""
    validation_hash: str = ""
    benchmark_hash: str = ""
    benchmark_ids: tuple[str, ...] = ()
    baseline_records_path: Path | None = None
    candidate_records_path: Path | None = None
    comparison_path: Path | None = None
    adapter_path: Path | None = None
    errors: list[str] = field(default_factory=list)
    completed_stages: list[str] = field(default_factory=list)
    stage_statuses: dict[str, str] = field(
        default_factory=lambda: {stage.value: "pending" for stage in DEFAULT_STAGE_ORDER}
    )
    current_stage: str | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str = ""

    def fail(self, message: str) -> None:
        self.errors.append(message)


class PipelineError(RuntimeError):
    """Raised by a stage when the run must abort immediately."""


# --- stage-callable protocol ----------------------------------------------


class StageCallable(abc.ABC):
    """Abstract base for stage callables.

    Tests can subclass this to record invocations.  The default
    :class:`StageFunc` wraps a plain function.
    """

    name: StageName

    @abc.abstractmethod
    def __call__(self, ctx: StageContext) -> StageContext:
        ...


@dataclass
class StageFunc(StageCallable):
    """Adapter that turns a plain function into a stage."""

    func: Callable[[StageContext], StageContext]
    name: StageName = StageName.VALIDATE_ENVIRONMENT

    def __call__(self, ctx: StageContext) -> StageContext:
        return self.func(ctx)


StageMap = Mapping[StageName, StageCallable]


# --- integrity guards ------------------------------------------------------


class ExperimentIntegrityError(PipelineError):
    """Raised when the plan fails an experiment-integrity check."""


def _canonical_generation_keys() -> set[str]:
    return {
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "do_sample",
        "repetition_penalty",
    }


def assert_publishable_pins_revision(plan: ExperimentPlan) -> None:
    """A publishable run must pin the model revision exactly."""

    if plan.publishable and not re.fullmatch(r"[0-9a-fA-F]{40,64}", plan.model_revision):
        raise ExperimentIntegrityError(
            "Publishable experiments must pin --model-revision to a "
            "40-64 character hexadecimal commit SHA so before/after benchmarks stay "
            "byte-identical."
        )


def assert_training_profile_is_real(plan: ExperimentPlan) -> None:
    """Real Kaggle training must not use the smoke profile."""

    if plan.runner == "kaggle" and plan.training_profile == "smoke":
        raise ExperimentIntegrityError(
            "Refusing to run the smoke profile on Kaggle. The smoke "
            "profile exists for offline preflight checks and must not "
            "be promoted to a real Kaggle experiment."
        )


def assert_pinned_revision_for_real_profile(plan: ExperimentPlan) -> None:
    """Every real training profile must pin the revision before model loading."""

    if plan.training_profile != "smoke" and not re.fullmatch(
        r"[0-9a-fA-F]{40,64}", plan.model_revision
    ):
        raise ExperimentIntegrityError(
            "Real training profiles must pin "
            "--model-revision so the same baseline can be reproduced."
        )


def assert_no_duplicate_paths(plan: ExperimentPlan) -> None:
    """Output paths must not overlap inputs."""

    inputs = tuple(
        input_path.resolve()
        for input_path in (
            plan.training_path,
            plan.validation_path,
            plan.benchmark_path,
        )
    )
    if len(set(inputs)) != len(inputs):
        raise ExperimentIntegrityError(
            "Training, validation, and benchmark paths must be distinct."
        )
    for writable in (plan.output_root.resolve(), plan.checkpoint_dir.resolve()):
        for resolved in inputs:
            try:
                resolved.relative_to(writable)
                raise ExperimentIntegrityError(
                    f"Writable path {writable} overlaps input {resolved}; "
                    "refusing to write artifacts on top of experiment data."
                )
            except ValueError:
                pass


# --- plan-level checks (the orchestrator runs them before any stage) -----


def run_plan_checks(plan: ExperimentPlan) -> None:
    """Validate the plan before any stage runs."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", plan.run_id):
        raise ExperimentIntegrityError(
            "run_id must be 1-128 path-safe characters and start with a letter or digit."
        )
    if not plan.model_id:
        raise ExperimentIntegrityError("model_id must be non-empty.")
    if plan.runner not in ("local", "kaggle"):
        raise ExperimentIntegrityError(
            f"runner must be 'local' or 'kaggle'; got {plan.runner!r}."
        )
    for required in (
        plan.training_path,
        plan.validation_path,
        plan.benchmark_path,
    ):
        if not str(required):
            raise ExperimentIntegrityError(
                f"Plan is missing a required path: {required!r}."
            )
    canonical = _canonical_generation_keys()
    extra = set(plan.generation_settings) - canonical
    missing = canonical - set(plan.generation_settings)
    if missing:
        raise ExperimentIntegrityError(
            f"generation_settings is missing required keys: {sorted(missing)}."
        )
    if extra:
        # We never silently drop extras; surface them so the caller
        # notices a typo before the manifest is written.
        raise ExperimentIntegrityError(
            f"generation_settings has unexpected keys: {sorted(extra)}."
        )
    if not plan.prompt_variants:
        raise ExperimentIntegrityError(
            "prompt_variants must be a non-empty tuple."
        )
    from sycophancy_rl.evaluation.prompts import ADVERSARIAL_PRESSURES, SYSTEM_PROMPTS

    if plan.prompt_condition not in SYSTEM_PROMPTS:
        raise ExperimentIntegrityError(
            f"Unknown prompt condition: {plan.prompt_condition!r}."
        )
    allowed_variants = {"original", "swap_options", "paraphrased_instruction"} | {
        f"adversarial_{name}" for name in ADVERSARIAL_PRESSURES
    }
    unknown_variants = set(plan.prompt_variants) - allowed_variants
    if unknown_variants:
        raise ExperimentIntegrityError(
            f"Unknown prompt variants: {sorted(unknown_variants)}."
        )
    from sycophancy_rl.training.grpo_config import PROFILES as TRAINING_PROFILES

    if plan.training_profile not in TRAINING_PROFILES:
        raise ExperimentIntegrityError(
            f"Unknown training profile: {plan.training_profile!r}."
        )
    if plan.reward_profile not in {
        "combined",
        "answer_only",
        "diagnostic_format_only",
    }:
        raise ExperimentIntegrityError(f"Unknown reward profile: {plan.reward_profile!r}.")
    if plan.batch_size < 1:
        raise ExperimentIntegrityError("batch_size must be at least 1.")
    if plan.seed < 0:
        raise ExperimentIntegrityError("seed must be non-negative.")
    if plan.max_examples is not None and plan.max_examples < 1:
        raise ExperimentIntegrityError("max_examples must be at least 1 when supplied.")
    if not isinstance(plan.generation_settings["do_sample"], bool):
        raise ExperimentIntegrityError("generation_settings.do_sample must be boolean.")
    numeric = {
        key: plan.generation_settings[key]
        for key in ("temperature", "top_p", "repetition_penalty")
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric.values()):
        raise ExperimentIntegrityError("Float generation settings must be numeric, not boolean.")
    if numeric["temperature"] <= 0 or not 0 < numeric["top_p"] <= 1:
        raise ExperimentIntegrityError("temperature must be > 0 and top_p must be in (0, 1].")
    if numeric["repetition_penalty"] <= 0:
        raise ExperimentIntegrityError("repetition_penalty must be > 0.")
    top_k = plan.generation_settings["top_k"]
    max_new_tokens = plan.generation_settings["max_new_tokens"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ExperimentIntegrityError("top_k must be a non-negative integer.")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ExperimentIntegrityError("max_new_tokens must be a positive integer.")
    sensitive = ("token", "secret", "password", "kaggle_key")
    if any(any(part in str(key).casefold() for part in sensitive) for key in plan.extra):
        raise ExperimentIntegrityError("Secrets must not be stored in ExperimentPlan.extra.")
    if plan.training_profile == "qlora_7b_16gb":
        if not plan.load_in_4bit:
            raise ExperimentIntegrityError("qlora_7b_16gb requires 4-bit loading.")
        if plan.batch_size > 2:
            raise ExperimentIntegrityError(
                "qlora_7b_16gb benchmark batch_size must be 1 or 2 on a 16 GiB GPU."
            )
    assert_publishable_pins_revision(plan)
    assert_training_profile_is_real(plan)
    assert_pinned_revision_for_real_profile(plan)
    assert_no_duplicate_paths(plan)
    from sycophancy_rl.training.model_registry import resolve_profile

    try:
        model_profile = resolve_profile(
            plan.model_id,
            model_revision=plan.model_revision,
            allow_unpinned=plan.allow_unpinned_model,
            custom_supports_4bit=plan.custom_supports_4bit,
            custom_lora_targets=plan.custom_lora_targets,
            custom_context_length=plan.custom_context_length,
            custom_min_vram_gib=plan.custom_min_vram_gib,
        )
    except ValueError as exc:
        raise ExperimentIntegrityError(f"Invalid model contract: {exc}") from exc
    if plan.load_in_4bit and not model_profile.supports_4bit:
        raise ExperimentIntegrityError(
            f"Model {plan.model_id!r} is not configured for 4-bit loading."
        )


# --- manifest hashing ------------------------------------------------------


def hash_file(path: Path) -> str:
    """Return the SHA-256 of ``path``; raise ``FileNotFoundError`` if missing."""

    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_data_hashes(plan: ExperimentPlan) -> tuple[str, str, str]:
    """SHA-256 the training, validation, and benchmark files."""

    return (
        hash_file(plan.training_path) if plan.training_path.exists() else "",
        hash_file(plan.validation_path) if plan.validation_path.exists() else "",
        hash_file(plan.benchmark_path) if plan.benchmark_path.exists() else "",
    )


# --- default stage implementations ----------------------------------------


def _validate_environment(ctx: StageContext) -> StageContext:
    plan = ctx.plan
    missing: list[str] = []
    for label, path in (
        ("training", plan.training_path),
        ("validation", plan.validation_path),
        ("benchmark", plan.benchmark_path),
    ):
        if not path.exists():
            missing.append(f"{label}={path}")
    if missing:
        raise PipelineError(
            "Environment validation failed; missing required input files: "
            + ", ".join(missing)
        )
    ctx.completed_stages.append(StageName.VALIDATE_ENVIRONMENT.value)
    return ctx


def _validate_data_and_leakage(ctx: StageContext) -> StageContext:
    """Re-run the leakage audit and data guards on the plan's files."""

    from sycophancy_rl.data_prep.schema import read_jsonl
    from sycophancy_rl.data_prep.setup_data import is_fixture_row
    plan = ctx.plan
    if not plan.training_path.exists():
        raise PipelineError(
            f"Training data not found at {plan.training_path}."
        )
    rows = read_jsonl(plan.training_path, expected_role="training")
    if plan.training_profile != "smoke" and any(is_fixture_row(row) for row in rows):
        sample = ", ".join(
            row["example_id"] for row in rows if is_fixture_row(row)
        )[:200]
        raise PipelineError(
            "Fixture-only data reached a real training run: "
            f"{sample}. Re-run data prep with the real ARC pool."
        )
    if plan.runner == "kaggle" and plan.training_profile == "smoke":
        raise PipelineError(
            "Refusing to start a Kaggle run with the smoke profile."
        )
    if any(
        str(row.get("source", "")).casefold() == "anthropic/model-written-evals"
        for row in rows
    ):
        raise PipelineError(
            "Anthropic benchmark rows reached the trainer. The Kaggle "
            "runner must never feed benchmark data into training."
        )
    validation_rows = read_jsonl(plan.validation_path, expected_role="validation")
    if plan.training_profile != "smoke" and any(
        is_fixture_row(row) for row in validation_rows
    ):
        raise PipelineError("Fixture validation data reached a real training run.")
    train_ids = {str(row["example_id"]) for row in rows}
    validation_ids = {str(row["example_id"]) for row in validation_rows}
    overlap = train_ids & validation_ids
    if overlap:
        raise PipelineError(
            "Training/validation leakage detected: " + ", ".join(sorted(overlap)[:5])
        )
    if plan.benchmark_path.exists():
        bench_rows = read_jsonl(plan.benchmark_path, expected_role="benchmark")
        # Confirm the manifest guard agrees the benchmark cannot train.
        for row in bench_rows:
            if str(row.get("data_role", "")).casefold() != "benchmark":
                raise PipelineError(
                    f"Benchmark row {row.get('example_id')} has "
                    f"data_role={row.get('data_role')!r}; expected 'benchmark'."
                )
        selected_benchmark_rows = (
            bench_rows[: plan.max_examples]
            if plan.max_examples is not None
            else bench_rows
        )
        ctx.benchmark_ids = tuple(row["example_id"] for row in selected_benchmark_rows)
        benchmark_ids = {str(row["example_id"]) for row in bench_rows}
        overlap = benchmark_ids & (train_ids | validation_ids)
        if overlap:
            raise PipelineError(
                "Benchmark leakage detected: " + ", ".join(sorted(overlap)[:5])
            )
    ctx.training_hash, ctx.validation_hash, ctx.benchmark_hash = _freeze_data_hashes(plan)
    ctx.completed_stages.append(StageName.VALIDATE_DATA_AND_LEAKAGE.value)
    return ctx


def _benchmark_before(ctx: StageContext) -> StageContext:
    """Run the frozen baseline benchmark with no adapter."""

    plan = ctx.plan
    output_dir = plan.output_root / plan.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "benchmark_ids.json"
    ids_path.write_text(
        json.dumps(list(ctx.benchmark_ids), indent=2) + "\n", encoding="utf-8"
    )
    from sycophancy_rl.evaluation.run_benchmark import (
        BenchmarkConfig,
        GenerationSettings,
        run_benchmark_job,
    )

    result = run_benchmark_job(
        BenchmarkConfig(
            run_name="before",
            model_id=plan.model_id,
            model_revision=plan.model_revision,
            benchmark_path=plan.benchmark_path,
            output_dir=output_dir,
            system_prompt_condition=plan.prompt_condition,
            prompt_variants=plan.prompt_variants,
            seed=plan.seed,
            max_examples=plan.max_examples,
            settings=GenerationSettings(**dict(plan.generation_settings)),
            load_in_4bit=plan.load_in_4bit,
            batch_size=plan.batch_size,
            resume=plan.resume,
        )
    )
    ctx.baseline_records_path = Path(result["records_path"])
    ctx.artifacts["benchmark_ids"] = ids_path
    ctx.artifacts["before_records"] = ctx.baseline_records_path
    ctx.artifacts["before_summary"] = Path(result["summary_path"])
    ctx.artifacts["before_manifest"] = Path(result["manifest_path"])
    ctx.completed_stages.append(StageName.BENCHMARK_BEFORE.value)
    return ctx


def _train(ctx: StageContext) -> StageContext:
    """Run the shared trainer with the exact paths frozen in the plan."""

    plan = ctx.plan
    from sycophancy_rl.training import train_grpo

    argv = [
        "--profile",
        plan.training_profile,
        "--model-id",
        plan.model_id,
        "--model-revision",
        plan.model_revision,
        "--train",
        str(plan.training_path),
        "--validation",
        str(plan.validation_path),
        "--benchmark",
        str(plan.benchmark_path),
        "--seed",
        str(plan.seed),
        "--run-name",
        plan.checkpoint_dir.name,
        "--output-root",
        str(plan.checkpoint_dir.parent),
        "--reward-profile",
        plan.reward_profile,
        "--system-prompt-condition",
        plan.prompt_condition,
        "--runner",
        plan.runner,
    ]
    if not plan.load_in_4bit:
        argv.append("--no-4bit")
    if plan.allow_cpu:
        argv.append("--allow-cpu")
    if plan.allow_unpinned_model:
        argv.append("--allow-unpinned-model")
    if plan.custom_supports_4bit:
        argv.append("--custom-supports-4bit")
    argv.extend(["--custom-lora-targets", ",".join(plan.custom_lora_targets)])
    argv.extend(["--custom-context-length", str(plan.custom_context_length)])
    argv.extend(["--custom-min-vram-gib", str(plan.custom_min_vram_gib)])
    if plan.resume:
        argv.append("--resume")
    for flag, key in (
        ("--learning-rate", "learning_rate"),
        ("--beta", "beta"),
        ("--max-steps", "max_steps"),
        ("--num-generations", "num_generations"),
    ):
        value = plan.extra.get(key)
        if value is not None:
            argv.extend([flag, str(value)])
    training_output = train_grpo.run_training(train_grpo._parse_args(argv))
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        pass
    ctx.adapter_path = training_output / "final_adapter"
    ctx.artifacts["training_manifest"] = training_output / "training_manifest.json"
    ctx.artifacts["training_summary"] = training_output / "training_summary.json"
    ctx.artifacts["adapter_checksums"] = training_output / "adapter_checksums.json"
    ctx.completed_stages.append(StageName.TRAIN.value)
    return ctx


def _benchmark_after(ctx: StageContext) -> StageContext:
    """Run the same benchmark settings against the trained adapter."""

    plan = ctx.plan
    output_dir = plan.output_root / plan.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if ctx.adapter_path is None or not ctx.adapter_path.exists():
        raise PipelineError("Training completed without a final adapter artifact.")
    from sycophancy_rl.evaluation.run_benchmark import (
        BenchmarkConfig,
        GenerationSettings,
        run_benchmark_job,
    )

    result = run_benchmark_job(
        BenchmarkConfig(
            run_name="after",
            model_id=plan.model_id,
            model_revision=plan.model_revision,
            adapter_path=ctx.adapter_path,
            benchmark_path=plan.benchmark_path,
            output_dir=output_dir,
            system_prompt_condition=plan.prompt_condition,
            prompt_variants=plan.prompt_variants,
            seed=plan.seed,
            max_examples=plan.max_examples,
            settings=GenerationSettings(**dict(plan.generation_settings)),
            load_in_4bit=plan.load_in_4bit,
            batch_size=plan.batch_size,
            resume=plan.resume,
        )
    )
    ctx.candidate_records_path = Path(result["records_path"])
    ctx.artifacts["after_records"] = ctx.candidate_records_path
    ctx.artifacts["after_summary"] = Path(result["summary_path"])
    ctx.artifacts["after_manifest"] = Path(result["manifest_path"])
    ctx.completed_stages.append(StageName.BENCHMARK_AFTER.value)
    return ctx


def _compare_before_after(ctx: StageContext) -> StageContext:
    """Create a provenance-checked paired comparison report."""

    plan = ctx.plan
    output_dir = plan.output_root / plan.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx.comparison_path = output_dir / "comparison.json"
    if ctx.baseline_records_path is None or ctx.candidate_records_path is None:
        raise PipelineError("Before/after response artifacts are missing.")
    from sycophancy_rl.evaluation.compare_runs import compare_runs_from_paths
    from sycophancy_rl.evaluation.io import write_json

    report = compare_runs_from_paths(
        ctx.baseline_records_path,
        ctx.candidate_records_path,
    )
    write_json(ctx.comparison_path, report)
    ctx.artifacts["comparison"] = ctx.comparison_path
    ctx.completed_stages.append(StageName.COMPARE_BEFORE_AFTER.value)
    return ctx


def _finalize_artifacts(ctx: StageContext) -> StageContext:
    """Hash all file artifacts; the driver writes the canonical manifest."""

    plan = ctx.plan
    output_dir = plan.output_root / plan.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = {
        label: hash_file(path)
        for label, path in sorted(ctx.artifacts.items())
        if path.exists() and path.is_file()
    }
    checksums_path = output_dir / "checksums.json"
    checksums_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ctx.artifacts["checksums"] = checksums_path
    ctx.completed_stages.append(StageName.FINALIZE_ARTIFACTS.value)
    return ctx


# --- default stage map ---------------------------------------------------


def default_stage_map() -> StageMap:
    """The default stage-callable map for every experiment.

    Tests can override any of these by passing a custom ``stage_map``
    to :class:`Pipeline`.
    """

    return {
        StageName.VALIDATE_ENVIRONMENT: StageFunc(
            _validate_environment, name=StageName.VALIDATE_ENVIRONMENT
        ),
        StageName.VALIDATE_DATA_AND_LEAKAGE: StageFunc(
            _validate_data_and_leakage, name=StageName.VALIDATE_DATA_AND_LEAKAGE
        ),
        StageName.BENCHMARK_BEFORE: StageFunc(
            _benchmark_before, name=StageName.BENCHMARK_BEFORE
        ),
        StageName.TRAIN: StageFunc(_train, name=StageName.TRAIN),
        StageName.BENCHMARK_AFTER: StageFunc(
            _benchmark_after, name=StageName.BENCHMARK_AFTER
        ),
        StageName.COMPARE_BEFORE_AFTER: StageFunc(
            _compare_before_after, name=StageName.COMPARE_BEFORE_AFTER
        ),
        StageName.FINALIZE_ARTIFACTS: StageFunc(
            _finalize_artifacts, name=StageName.FINALIZE_ARTIFACTS
        ),
    }


# --- pipeline driver ------------------------------------------------------


@dataclass
class PipelineResult:
    """The result of one pipeline run."""

    mode: ExecutionMode
    completed_stages: tuple[str, ...]
    errors: tuple[str, ...]
    manifest: dict[str, Any]
    artifacts: dict[str, Path]


def _parent_manifest_payload(ctx: StageContext, *, status: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": ctx.plan.run_id,
        "status": status,
        "current_stage": ctx.current_stage,
        "plan": ctx.plan.frozen_dict(),
        "plan_sha256": ctx.plan.plan_sha256,
        "data_hashes": {
            "training": ctx.training_hash,
            "validation": ctx.validation_hash,
            "benchmark": ctx.benchmark_hash,
        },
        "frozen_benchmark_ids": list(ctx.benchmark_ids),
        "stage_statuses": dict(ctx.stage_statuses),
        "completed_stages": list(ctx.completed_stages),
        "artifacts": {
            label: str(path) for label, path in sorted(ctx.artifacts.items())
        },
        "errors": list(ctx.errors),
        "started_at": ctx.started_at,
        "finished_at": ctx.finished_at or None,
        "runner": ctx.plan.runner,
    }


def _write_parent_manifest(ctx: StageContext, *, status: str) -> Path:
    path = ctx.plan.output_root / ctx.plan.run_id / "experiment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _parent_manifest_payload(ctx, status=status)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    ctx.manifest = payload
    return path


def _resume_stage_is_complete(ctx: StageContext, stage: StageName) -> bool:
    if ctx.stage_statuses.get(stage.value) != "completed":
        return False
    required: dict[StageName, tuple[str, ...]] = {
        StageName.BENCHMARK_BEFORE: ("before_records",),
        StageName.TRAIN: ("training_manifest",),
        StageName.BENCHMARK_AFTER: ("after_records",),
        StageName.COMPARE_BEFORE_AFTER: ("comparison",),
        StageName.FINALIZE_ARTIFACTS: ("checksums",),
    }
    if stage in (StageName.VALIDATE_ENVIRONMENT, StageName.VALIDATE_DATA_AND_LEAKAGE):
        return False
    if stage is StageName.TRAIN and not (ctx.plan.checkpoint_dir / "final_adapter").exists():
        return False
    return all(
        label in ctx.artifacts and ctx.artifacts[label].exists()
        for label in required.get(stage, ())
    )


class Pipeline:
    """Run a plan in ``plan``/``dry-run``/``execute`` mode."""

    def __init__(
        self,
        plan: ExperimentPlan,
        *,
        mode: ExecutionMode = ExecutionMode.PLAN,
        stage_map: StageMap | None = None,
        stage_order: Sequence[StageName] = DEFAULT_STAGE_ORDER,
    ) -> None:
        self.plan = plan
        self.mode = mode
        self.stage_map: StageMap = dict(stage_map or default_stage_map())
        self.stage_order: tuple[StageName, ...] = tuple(stage_order)
        self._verify_stage_map()

    def _verify_stage_map(self) -> None:
        for stage in self.stage_order:
            if stage not in self.stage_map:
                raise ExperimentIntegrityError(
                    f"Stage {stage.value!r} is not registered in stage_map."
                )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of what the run would do."""

        return {
            "mode": self.mode.value,
            "plan": self.plan.frozen_dict(),
            "stage_order": [stage.value for stage in self.stage_order],
            "registered_stages": sorted(stage.value for stage in self.stage_map),
        }

    def run(self) -> PipelineResult:
        """Run the pipeline according to ``self.mode``."""

        run_plan_checks(self.plan)
        ctx = StageContext(plan=self.plan, mode=self.mode)

        if self.mode is ExecutionMode.PLAN:
            return PipelineResult(
                mode=ExecutionMode.PLAN,
                completed_stages=(),
                errors=(),
                manifest={"plan": self.plan.frozen_dict()},
                artifacts={},
            )

        if self.mode is ExecutionMode.DRY_RUN:
            # Run only pure, lightweight validation. These stages read and
            # hash JSONL inputs but never import torch or create output paths.
            ctx = _validate_environment(ctx)
            ctx = _validate_data_and_leakage(ctx)
            return PipelineResult(
                mode=ExecutionMode.DRY_RUN,
                completed_stages=(),
                errors=(),
                manifest={
                    "plan": self.plan.frozen_dict(),
                    "training_hash": ctx.training_hash,
                    "validation_hash": ctx.validation_hash,
                    "benchmark_hash": ctx.benchmark_hash,
                    "frozen_benchmark_id_count": len(ctx.benchmark_ids),
                    "frozen_benchmark_ids_preview": list(ctx.benchmark_ids[:5]),
                },
                artifacts={
                    "planned_output_dir": self.plan.output_root / self.plan.run_id,
                    "planned_checkpoint_dir": self.plan.checkpoint_dir,
                },
            )

        # EXECUTE is the only mode that creates artifacts.
        manifest_path = self.plan.output_root / self.plan.run_id / "experiment.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not self.plan.resume:
                raise FileExistsError(
                    f"Experiment already exists: {manifest_path}. Use a new run ID "
                    "or pass --resume with an identical plan."
                )
            if previous.get("plan_sha256") != self.plan.plan_sha256:
                raise ExperimentIntegrityError(
                    "Resume refused because the frozen experiment plan changed."
                )
            if previous.get("status") == "completed":
                raise FileExistsError("The experiment is already completed; resume is unnecessary.")
            ctx.training_hash = str(previous.get("data_hashes", {}).get("training", ""))
            ctx.validation_hash = str(previous.get("data_hashes", {}).get("validation", ""))
            ctx.benchmark_hash = str(previous.get("data_hashes", {}).get("benchmark", ""))
            ctx.benchmark_ids = tuple(previous.get("frozen_benchmark_ids", ()))
            ctx.stage_statuses.update(previous.get("stage_statuses", {}))
            ctx.completed_stages = list(previous.get("completed_stages", ()))
            ctx.artifacts = {
                label: Path(path)
                for label, path in previous.get("artifacts", {}).items()
            }
            ctx.baseline_records_path = ctx.artifacts.get("before_records")
            ctx.candidate_records_path = ctx.artifacts.get("after_records")
            ctx.comparison_path = ctx.artifacts.get("comparison")
            adapter = self.plan.checkpoint_dir / "final_adapter"
            ctx.adapter_path = adapter if adapter.exists() else None
        _write_parent_manifest(ctx, status="running")
        try:
            for stage in self.stage_order:
                if self.plan.resume and _resume_stage_is_complete(ctx, stage):
                    continue
                ctx.current_stage = stage.value
                ctx.stage_statuses[stage.value] = "running"
                _write_parent_manifest(ctx, status="running")
                callable_ = self.stage_map[stage]
                ctx = callable_(ctx)
                ctx.completed_stages = list(dict.fromkeys(ctx.completed_stages))
                ctx.stage_statuses[stage.value] = "completed"
                _write_parent_manifest(ctx, status="running")
        except KeyboardInterrupt:
            if ctx.current_stage:
                ctx.stage_statuses[ctx.current_stage] = "interrupted"
            ctx.errors.append("Experiment interrupted by user or platform signal.")
            ctx.finished_at = datetime.now(timezone.utc).isoformat()
            _write_parent_manifest(ctx, status="interrupted")
            raise
        except Exception as exc:
            if ctx.current_stage:
                ctx.stage_statuses[ctx.current_stage] = "failed"
            ctx.errors.append(str(exc))
            ctx.finished_at = datetime.now(timezone.utc).isoformat()
            _write_parent_manifest(ctx, status="failed")
            return PipelineResult(
                mode=ExecutionMode.EXECUTE,
                completed_stages=tuple(ctx.completed_stages),
                errors=tuple(ctx.errors),
                manifest=ctx.manifest,
                artifacts=ctx.artifacts,
            )
        ctx.finished_at = datetime.now(timezone.utc).isoformat()
        ctx.current_stage = None
        _write_parent_manifest(ctx, status="completed")
        return PipelineResult(
            mode=ExecutionMode.EXECUTE,
            completed_stages=tuple(ctx.completed_stages),
            errors=tuple(ctx.errors),
            manifest=ctx.manifest,
            artifacts=ctx.artifacts,
        )


# --- helpers for the CLI / Kaggle runner ---------------------------------


def build_default_plan(
    *,
    run_id: str,
    model_id: str,
    model_revision: str,
    training_path: Path,
    validation_path: Path,
    benchmark_path: Path,
    seed: int,
    training_profile: str,
    reward_profile: str,
    prompt_condition: str,
    prompt_variants: Iterable[str],
    generation_settings: Mapping[str, Any],
    output_root: Path,
    checkpoint_dir: Path,
    runner: str,
    publishable: bool = False,
    max_examples: int | None = None,
    batch_size: int = 4,
    load_in_4bit: bool = True,
    allow_cpu: bool = False,
    allow_unpinned_model: bool = False,
    custom_supports_4bit: bool = False,
    custom_lora_targets: Iterable[str] = ("all-linear",),
    custom_context_length: int = 2048,
    custom_min_vram_gib: float = 0.0,
    resume: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> ExperimentPlan:
    """Helper to build an :class:`ExperimentPlan` with type-checked fields."""

    return ExperimentPlan(
        run_id=run_id,
        model_id=model_id,
        model_revision=model_revision,
        training_path=Path(training_path),
        validation_path=Path(validation_path),
        benchmark_path=Path(benchmark_path),
        seed=seed,
        training_profile=training_profile,
        reward_profile=reward_profile,
        prompt_condition=prompt_condition,
        prompt_variants=tuple(prompt_variants),
        generation_settings=dict(generation_settings),
        output_root=Path(output_root),
        checkpoint_dir=Path(checkpoint_dir),
        runner=runner,
        publishable=publishable,
        max_examples=max_examples,
        batch_size=batch_size,
        load_in_4bit=load_in_4bit,
        allow_cpu=allow_cpu,
        allow_unpinned_model=allow_unpinned_model,
        custom_supports_4bit=custom_supports_4bit,
        custom_lora_targets=tuple(custom_lora_targets),
        custom_context_length=custom_context_length,
        custom_min_vram_gib=custom_min_vram_gib,
        resume=resume,
        extra=dict(extra or {}),
    )


# --- safety helpers -------------------------------------------------------


def scrub_secrets(payload: Mapping[str, Any] | str) -> str:
    """Return ``payload`` as a JSON string with secret-like keys redacted.

    The Kaggle runner and the CLI never print HF_TOKEN, KAGGLE_KEY, or
    any other environment secret.  The scrubber is intentionally
    conservative: any key that contains ``token`` or ``key`` is
    redacted, regardless of where it appears in a nested dict.
    """

    SENSITIVE = ("hf_token", "kaggle_key", "kaggle_username", "secret", "password")

    def _scrub(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: "***" if any(token in str(key).casefold() for token in SENSITIVE) else _scrub(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [_scrub(item) for item in value]
        return value

    if isinstance(payload, str):
        return payload
    return json.dumps(_scrub(payload), indent=2, sort_keys=True, default=str)


__all__ = [
    "DEFAULT_STAGE_ORDER",
    "ExperimentIntegrityError",
    "ExperimentPlan",
    "ExecutionMode",
    "Pipeline",
    "PipelineError",
    "PipelineResult",
    "StageCallable",
    "StageContext",
    "StageFunc",
    "StageMap",
    "StageName",
    "assert_no_duplicate_paths",
    "assert_pinned_revision_for_real_profile",
    "assert_publishable_pins_revision",
    "assert_training_profile_is_real",
    "build_default_plan",
    "default_stage_map",
    "hash_file",
    "run_plan_checks",
    "scrub_secrets",
]
