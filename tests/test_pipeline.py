"""Offline tests for the canonical experiment pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sycophancy_rl.experiments.pipeline import (
    ExecutionMode,
    ExperimentIntegrityError,
    Pipeline,
    PipelineError,
    StageFunc,
    StageName,
    build_default_plan,
)


def _plan(tmp_path: Path, **overrides):
    paths = {}
    for name in ("train", "validation", "benchmark"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        paths[name] = path
    values = {
        "run_id": "unit-run",
        "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "model_revision": "31b70e2e869a7173562077fd711b654946d38674",
        "training_path": paths["train"],
        "validation_path": paths["validation"],
        "benchmark_path": paths["benchmark"],
        "seed": 42,
        "training_profile": "smoke",
        "reward_profile": "combined",
        "prompt_condition": "neutral",
        "prompt_variants": ("original",),
        "generation_settings": {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 32,
            "repetition_penalty": 1.0,
        },
        "output_root": tmp_path / "outputs",
        "checkpoint_dir": tmp_path / "checkpoints" / "unit-run",
        "runner": "local",
        "allow_cpu": True,
        "load_in_4bit": False,
    }
    values.update(overrides)
    return build_default_plan(**values)


def test_plan_mode_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    result = Pipeline(plan, mode=ExecutionMode.PLAN).run()
    assert result.errors == ()
    assert not plan.output_root.exists()
    assert result.manifest["plan"]["run_id"] == "unit-run"


def test_preference_only_plan_serializes_without_factual_test(tmp_path: Path) -> None:
    plan = _plan(tmp_path, preference_only=True)

    frozen = plan.frozen_dict()
    restored = type(plan).from_dict(frozen)

    assert plan.factual_test_path is None
    assert frozen["factual_test_path"] is None
    assert restored.factual_test_path is None


def test_execute_writes_completed_parent_manifest(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    def complete(ctx):
        ctx.completed_stages.append(StageName.VALIDATE_ENVIRONMENT.value)
        return ctx

    result = Pipeline(
        plan,
        mode=ExecutionMode.EXECUTE,
        stage_order=(StageName.VALIDATE_ENVIRONMENT,),
        stage_map={
            StageName.VALIDATE_ENVIRONMENT: StageFunc(
                complete, name=StageName.VALIDATE_ENVIRONMENT
            )
        },
    ).run()
    manifest = json.loads(
        (plan.output_root / plan.run_id / "experiment.json").read_text(encoding="utf-8")
    )
    assert result.errors == ()
    assert manifest["status"] == "completed"
    assert manifest["stage_statuses"]["validate_environment"] == "completed"
    assert manifest["plan_sha256"] == plan.plan_sha256


def test_execute_records_stage_failure(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    def fail(_ctx):
        raise PipelineError("expected failure")

    result = Pipeline(
        plan,
        mode=ExecutionMode.EXECUTE,
        stage_order=(StageName.VALIDATE_ENVIRONMENT,),
        stage_map={
            StageName.VALIDATE_ENVIRONMENT: StageFunc(
                fail, name=StageName.VALIDATE_ENVIRONMENT
            )
        },
    ).run()
    assert result.errors == ("expected failure",)
    assert result.manifest["status"] == "failed"
    assert result.manifest["stage_statuses"]["validate_environment"] == "failed"


def test_resume_refuses_changed_frozen_plan(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    def complete(ctx):
        ctx.completed_stages.append(StageName.VALIDATE_ENVIRONMENT.value)
        return ctx

    stages = {
        StageName.VALIDATE_ENVIRONMENT: StageFunc(
            complete, name=StageName.VALIDATE_ENVIRONMENT
        )
    }
    Pipeline(
        plan,
        mode=ExecutionMode.EXECUTE,
        stage_order=(StageName.VALIDATE_ENVIRONMENT,),
        stage_map=stages,
    ).run()
    changed = replace(plan, seed=7, resume=True)
    with pytest.raises(PipelineError, match="frozen experiment plan changed"):
        Pipeline(
            changed,
            mode=ExecutionMode.EXECUTE,
            stage_order=(StageName.VALIDATE_ENVIRONMENT,),
            stage_map=stages,
        ).run()


def test_kaggle_plan_rejects_unpinned_model_before_execution(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        runner="kaggle",
        training_profile="local_8gb",
        model_id="example/custom-model",
        model_revision="main",
        allow_unpinned_model=True,
    )
    with pytest.raises(ExperimentIntegrityError, match="must pin"):
        Pipeline(plan, mode=ExecutionMode.PLAN).run()


def test_registered_qwen_7b_kaggle_plan_passes_without_loading_weights(
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path,
        runner="kaggle",
        training_profile="qlora_7b_16gb",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
        batch_size=1,
        load_in_4bit=True,
    )
    result = Pipeline(plan, mode=ExecutionMode.PLAN).run()
    assert result.errors == ()
    assert result.manifest["plan"]["model_id"] == "Qwen/Qwen2.5-7B-Instruct"


def test_7b_profile_requires_4bit_and_safe_benchmark_batch(tmp_path: Path) -> None:
    base = _plan(
        tmp_path,
        runner="kaggle",
        training_profile="qlora_7b_16gb",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
        batch_size=1,
        load_in_4bit=True,
    )
    with pytest.raises(ExperimentIntegrityError, match="requires 4-bit"):
        Pipeline(replace(base, load_in_4bit=False), mode=ExecutionMode.PLAN).run()
    with pytest.raises(ExperimentIntegrityError, match="batch_size"):
        Pipeline(replace(base, batch_size=3), mode=ExecutionMode.PLAN).run()


@pytest.mark.parametrize("profile", ("kaggle_online_smoke", "qwen25_7b_online"))
def test_online_profiles_reject_prepared_rollouts(tmp_path: Path, profile: str) -> None:
    overrides = {
        "training_profile": profile,
        "extra": {"rollout_mode": "prepared"},
    }
    if profile == "qwen25_7b_online":
        overrides.update(
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                "load_in_4bit": True,
            }
        )
    plan = _plan(tmp_path, **overrides)

    with pytest.raises(ExperimentIntegrityError, match="requires online rollout mode"):
        Pipeline(plan, mode=ExecutionMode.PLAN).run()


@pytest.mark.parametrize(
    ("key", "value"),
    (("temperature", True), ("top_p", 0.0), ("top_k", 1.5), ("max_new_tokens", 2.5)),
)
def test_plan_rejects_invalid_generation_types(
    tmp_path: Path, key: str, value: object
) -> None:
    plan = _plan(tmp_path)
    settings = dict(plan.generation_settings)
    settings[key] = value
    with pytest.raises(ExperimentIntegrityError):
        Pipeline(replace(plan, generation_settings=settings), mode=ExecutionMode.PLAN).run()


def test_publishable_plan_rejects_smoke_token_budget(tmp_path: Path) -> None:
    plan = _plan(tmp_path, publishable=True, training_profile="local_8gb")
    with pytest.raises(ExperimentIntegrityError, match="max_new_tokens >= 128"):
        Pipeline(plan, mode=ExecutionMode.PLAN).run()


def test_publishable_plan_rejects_smoke_profile(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        publishable=True,
        generation_settings={
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 192,
            "repetition_penalty": 1.0,
        },
    )
    with pytest.raises(ExperimentIntegrityError, match="cannot use the smoke profile"):
        Pipeline(plan, mode=ExecutionMode.PLAN).run()


def test_publishable_plan_rejects_reduced_profile_steps(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        publishable=True,
        generation_settings={
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 192,
            "repetition_penalty": 1.0,
        },
        training_profile="qlora_7b_16gb",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
        load_in_4bit=True,
        extra={"max_steps": 5},
    )
    with pytest.raises(ExperimentIntegrityError, match="cannot reduce max_steps"):
        Pipeline(plan, mode=ExecutionMode.PLAN).run()
