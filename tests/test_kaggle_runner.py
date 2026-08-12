"""Tests for the Kaggle runner helpers.

These tests never reach the network and never require Kaggle
credentials.  They exercise metadata rendering, validation, dry-run
doctor output, and the CLI surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sycophancy_rl.data_prep.schema import read_jsonl, write_jsonl
from sycophancy_rl.experiments import kaggle as kaggle_mod
from sycophancy_rl.experiments.pipeline import build_default_plan


def _import_main():
    return kaggle_mod.main


def test_build_kernel_metadata_substitutes_username() -> None:
    meta = kaggle_mod.build_kernel_metadata("velmurugan", accelerator="gpu")
    assert meta["id"] == "velmurugan/sycophancy-rl-runner"
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["is_private"] is True
    assert meta["model_sources"] == []
    assert meta["code_file"] == "runner.py"


def test_build_kernel_metadata_rejects_bad_username() -> None:
    with pytest.raises(ValueError):
        kaggle_mod.build_kernel_metadata("")
    with pytest.raises(ValueError):
        kaggle_mod.build_kernel_metadata("evil/path")


def test_write_kernel_metadata_writes_json(tmp_path: Path) -> None:
    path = tmp_path / "kernel.json"
    meta = kaggle_mod.write_kernel_metadata(path, "velmurugan")
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == meta
    assert on_disk["id"].startswith("velmurugan/")


def test_doctor_does_not_require_credentials(monkeypatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    snapshot = kaggle_mod.doctor()
    assert snapshot["kaggle_username"] is None
    assert snapshot["kaggle_key_present"] is False


def test_cmd_init_renders_metadata(tmp_path: Path) -> None:
    out = tmp_path / "kernel.json"
    code = kaggle_mod.cmd_init(
        type("A", (), {
            "username": "velmurugan",
            "output": str(out),
            "public": False,
            "accelerator": "cpu",
        })()
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["enable_gpu"] is False


def test_cmd_validate_ok(tmp_path: Path) -> None:
    path = tmp_path / "kernel.json"
    kaggle_mod.write_kernel_metadata(path, "velmurugan")
    code = kaggle_mod.cmd_validate(
        type("A", (), {"metadata": str(path)})()
    )
    assert code == 0


def test_cmd_validate_missing_file(tmp_path: Path) -> None:
    code = kaggle_mod.cmd_validate(
        type("A", (), {"metadata": str(tmp_path / "nope.json")})()
    )
    assert code == 2


def test_cmd_validate_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "kernel.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    code = kaggle_mod.cmd_validate(type("A", (), {"metadata": str(path)})())
    assert code == 2


def test_cmd_validate_rejects_bad_id(tmp_path: Path) -> None:
    path = tmp_path / "kernel.json"
    path.write_text(
        json.dumps(
            {
                "id": "no-slash",
                "code_file": "x.py",
                "kernel_type": "script",
                "enable_gpu": True,
            }
        ),
        encoding="utf-8",
    )
    code = kaggle_mod.cmd_validate(type("A", (), {"metadata": str(path)})())
    assert code == 2


def test_repository_root_uses_checkout_cwd_for_wheel_install(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "deploy" / "kaggle").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (checkout / "deploy" / "kaggle" / "runner.py").write_text("", encoding="utf-8")
    fake_installed_module = (
        tmp_path / "site-packages" / "sycophancy_rl" / "experiments" / "kaggle.py"
    )

    monkeypatch.chdir(checkout)
    monkeypatch.setattr(kaggle_mod, "__file__", str(fake_installed_module))

    assert kaggle_mod._repository_root() == checkout.resolve()


def test_cmd_push_dry_run_emits_snapshot(capsys) -> None:
    args = type("A", (), {"metadata": "deploy/kaggle/kernel-metadata.json", "dry_run": True})()
    code = kaggle_mod.cmd_push(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["dry_run"] is True


def test_cmd_dry_run_summary() -> None:
    args = type("A", (), {})()
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = kaggle_mod.cmd_dry_run(args)
    payload = json.loads(buf.getvalue())
    assert code == 0
    assert payload["runner"] == "kaggle"
    assert "snapshot" in payload


def test_main_dispatch_init(tmp_path: Path) -> None:
    out = tmp_path / "k.json"
    code = kaggle_mod.main([
        "init",
        "--username", "velmurugan",
        "--output", str(out),
        "--accelerator", "cpu",
    ])
    assert code == 0


def test_main_dispatch_validate_missing() -> None:
    code = kaggle_mod.main(["validate", "--metadata", "nope.json"])
    assert code == 2


def test_stage_kernel_is_self_contained_and_offline(tmp_path: Path) -> None:
    plan = build_default_plan(
        run_id="kaggle-unit",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
        training_path=Path("data/generated/splits/train.jsonl"),
        validation_path=Path("data/generated/splits/validation.jsonl"),
        benchmark_path=Path("data/benchmarks/anthropic_sycophancy.jsonl"),
        seed=42,
        training_profile="qlora_7b_16gb",
        reward_profile="combined",
        prompt_condition="neutral",
        prompt_variants=("original",),
        generation_settings={
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 32,
            "repetition_penalty": 1.0,
        },
        output_root=Path("/kaggle/working/outputs"),
        checkpoint_dir=Path("/kaggle/working/checkpoints/kaggle-unit"),
        runner="kaggle",
        batch_size=1,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.frozen_dict()), encoding="utf-8")
    staged = kaggle_mod.stage_kernel(
        plan_path=plan_path,
        output_dir=tmp_path / "staged",
        username="velmurugan",
        dataset_slug="velmurugan/sycophancy-data",
    )
    for relative in (
        "runner.py",
        "requirements.lock",
        "experiment-plan.json",
        "kernel-metadata.json",
        "pyproject.toml",
        "src/sycophancy_rl/experiments/pipeline.py",
    ):
        assert (staged / relative).exists()


def test_stage_dataset_keeps_benchmark_separate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base = dict(
        read_jsonl(root / "data/processed/training_pool.jsonl", expected_role="training")[0]
    )
    base["source"] = "allenai/ai2_arc/ARC-Challenge/train"
    base["source_revision"] = "abc123"
    base["metadata"] = {**base.get("metadata", {}), "is_fixture": False}
    training = dict(base)
    training["example_id"] = "real-train"
    validation = dict(base)
    validation["example_id"] = "real-validation"
    validation["data_role"] = "validation"
    # Build a valid benchmark row inside the test. The real Anthropic benchmark
    # is intentionally gitignored, so a clean checkout must not depend on a
    # developer's previously downloaded copy.
    benchmark = dict(base)
    benchmark["example_id"] = "benchmark-only"
    benchmark["source"] = "anthropic/model-written-evals"
    benchmark["source_revision"] = "fixture-revision"
    benchmark["data_role"] = "benchmark"
    benchmark["target_option"] = benchmark["independent_option"]
    benchmark["metadata"] = {
        **benchmark.get("metadata", {}),
        "benchmark_only": True,
        "is_fixture": False,
    }
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    benchmark_path = tmp_path / "benchmark.jsonl"
    write_jsonl(train_path, [training])
    write_jsonl(validation_path, [validation])
    write_jsonl(benchmark_path, [benchmark])

    staged = kaggle_mod.stage_dataset(
        training_path=train_path,
        validation_path=validation_path,
        benchmark_path=benchmark_path,
        output_dir=tmp_path / "data-build",
        dataset_slug="velmurugan/sycophancy-data",
    )
    manifest = json.loads((staged / "data-manifest.json").read_text(encoding="utf-8"))
    assert manifest["anthropic_benchmark_used_for_training"] is False
    assert (staged / "splits/train.jsonl").exists()
    assert (staged / "benchmarks/anthropic_sycophancy.jsonl").exists()
