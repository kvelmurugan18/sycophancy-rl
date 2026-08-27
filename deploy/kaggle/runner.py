#!/usr/bin/env python3
"""Execute one frozen before/train/after experiment inside Kaggle."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
PLAN_PATH = Path(__file__).with_name("experiment-plan.json")
LOCK_PATH = Path(__file__).with_name("requirements.lock")
SOURCE_ROOT = Path(__file__).with_name("src")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _load_hf_token() -> str | None:
    names = ("HUGGINGFACE_TOKEN", "HF_TOKEN")
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        client = UserSecretsClient()
        for name in names:
            try:
                token = client.get_secret(name)
            except Exception:
                token = None
            if token:
                return token
    except ImportError:
        pass
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def _export_hf_token() -> bool:
    token = _load_hf_token()
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_TOKEN"] = token
    return True


def _locked_requirements() -> dict[str, str]:
    requirements: dict[str, str] = {}
    for line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, version = stripped.split("==", 1)
        requirements[name] = version
    return requirements


def _bootstrap_dependencies() -> None:
    required = _locked_requirements()
    missing_or_wrong: list[str] = []
    for name, expected in required.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            missing_or_wrong.append(f"{name}=={expected}")
    if missing_or_wrong:
        try:
            torch_before = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "Kaggle's CUDA-matched torch must be preinstalled; this runner will not install it."
            ) from exc
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--requirement",
                str(LOCK_PATH),
            ],
            check=True,
        )
        torch_after = importlib.metadata.version("torch")
        if torch_after != torch_before:
            raise RuntimeError(
                f"Dependency bootstrap changed torch {torch_before} -> {torch_after}; aborting."
            )


def _find_data_root() -> Path:
    configured = os.environ.get("SYCO_DATA_ROOT")
    candidates = [Path(configured)] if configured else []
    if KAGGLE_INPUT.exists():
        for mounted in sorted(path for path in KAGGLE_INPUT.iterdir() if path.is_dir()):
            candidates.extend((mounted, mounted / "data"))
    for candidate in candidates:
        if (
            (candidate / "splits" / "train.jsonl").exists()
            and (candidate / "splits" / "validation.jsonl").exists()
            and (candidate / "benchmarks" / "anthropic_sycophancy.jsonl").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "No mounted Kaggle dataset contains splits/train.jsonl, "
        "splits/validation.jsonl, and benchmarks/anthropic_sycophancy.jsonl. "
        "An optional objective split may also be mounted as splits/test.jsonl."
    )


def _doctor() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "kaggle_input_exists": KAGGLE_INPUT.exists(),
        "kaggle_working_exists": KAGGLE_WORKING.exists(),
        "hf_token_present": bool(_load_hf_token()),
        "plan_exists": PLAN_PATH.exists(),
        "cuda_available": False,
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["torch_cuda_version"] = torch.version.cuda
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snapshot["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        snapshot["torch"] = None
    for package in ("transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            snapshot[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            snapshot[package] = None
    snapshot["readiness_level"] = (
        "LEVEL 2: KAGGLE MODEL LOAD NOT YET VERIFIED"
        if snapshot["cuda_available"]
        else "LEVEL 1: STATIC CODE ONLY; CUDA NOT AVAILABLE"
    )
    return snapshot


def _restore_checkpoint_if_requested(run_id: str, destination: Path) -> None:
    source_root = os.environ.get("SYCO_RESUME_CHECKPOINT_ROOT")
    if not source_root or destination.exists():
        return
    source = Path(source_root) / run_id
    if not source.exists():
        raise FileNotFoundError(f"Requested resume checkpoint is missing: {source}")
    shutil.copytree(source, destination)


def main() -> int:
    print(json.dumps({"doctor": _doctor()}, indent=2))
    if not PLAN_PATH.exists() or not LOCK_PATH.exists():
        raise FileNotFoundError("Staged plan or requirements.lock is missing.")
    _bootstrap_dependencies()
    _export_hf_token()

    from dataclasses import replace

    from sycophancy_rl.experiments.pipeline import ExecutionMode, ExperimentPlan, Pipeline

    raw = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment-plan.json must contain an object.")
    original = ExperimentPlan.from_dict(raw)
    data_root = _find_data_root()
    plan = replace(
        original,
        training_path=data_root / "splits" / "train.jsonl",
        validation_path=data_root / "splits" / "validation.jsonl",
        factual_test_path=(
            data_root / "splits" / "test.jsonl"
            if (data_root / "splits" / "test.jsonl").exists()
            else None
        ),
        benchmark_path=data_root / "benchmarks" / "anthropic_sycophancy.jsonl",
        output_root=KAGGLE_WORKING / "outputs",
        checkpoint_dir=KAGGLE_WORKING / "checkpoints" / original.run_id,
        runner="kaggle",
        allow_cpu=False,
    )
    _restore_checkpoint_if_requested(plan.run_id, plan.checkpoint_dir)
    result = Pipeline(plan, mode=ExecutionMode.EXECUTE).run()
    print(
        json.dumps(
            {
                "completed_stages": result.completed_stages,
                "errors": result.errors,
                "manifest": result.manifest,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0 if not result.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
