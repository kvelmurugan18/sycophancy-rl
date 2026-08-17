"""Kaggle package — read-only ``/kaggle/input`` data, writable
``/kaggle/working`` output, no token logging, dry-run by default.

This module owns all of the Kaggle-specific runner logic so the local
runner doesn't need to special-case the Kaggle environment.  The CLI
subcommands ``syco kaggle init``, ``syco kaggle validate`` and
``syco kaggle push`` all delegate here, and ``syco train --runner
kaggle --dry-run`` runs the same code path without uploading or
executing anything.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_OUTPUTS = KAGGLE_WORKING / "outputs"
KAGGLE_CHECKPOINTS = KAGGLE_WORKING / "checkpoints"


def _repository_root() -> Path:
    """Locate a checkout root for editable and wheel-installed invocations.

    A normal wheel lives under ``site-packages`` and cannot discover
    repository-only deployment assets from ``__file__``. Searching the
    current directory first lets ``syco kaggle stage*`` work after a standard
    install when it is launched from the cloned repository.
    """

    seen: set[Path] = set()
    for search_root in (Path.cwd().resolve(), Path(__file__).resolve()):
        for candidate in (search_root, *search_root.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "deploy" / "kaggle" / "runner.py"
            ).is_file():
                return candidate
    raise RuntimeError(
        "Could not locate the sycophancy-rl repository root. Run this command "
        "from the cloned repository, or use a staged self-contained kernel."
    )


# --- kernel-metadata template ----------------------------------------------


KERNEL_METADATA_TEMPLATE: dict[str, Any] = {
    "id": "REPLACE_WITH_USERNAME/sycophancy-rl-runner",
    "title": "sycophancy-rl training runner",
    "code_file": "runner.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": ["REPLACE_WITH_USERNAME/sycophancy-rl-benchmarks"],
    "competition_sources": [],
    "model_sources": [],
}


def build_kernel_metadata(
    username: str,
    *,
    private: bool = True,
    accelerator: str = "gpu",
) -> dict[str, Any]:
    """Return a kernel-metadata dict with the user's Kaggle username.

    The function never logs the username or the rendered token; it only
    substitutes the username into the template.
    """

    if not username or "/" in username:
        raise ValueError(
            f"Kaggle username {username!r} is empty or contains a path "
            "separator; refusing to render the kernel metadata."
        )
    rendered = json.loads(json.dumps(KERNEL_METADATA_TEMPLATE))
    rendered["id"] = f"{username}/sycophancy-rl-runner"
    rendered["is_private"] = bool(private)
    if accelerator == "gpu":
        rendered["enable_gpu"] = True
        rendered["enable_tpu"] = False
    elif accelerator == "tpu":
        rendered["enable_gpu"] = False
        rendered["enable_tpu"] = True
    else:  # "cpu"
        rendered["enable_gpu"] = False
        rendered["enable_tpu"] = False
    rendered["dataset_sources"] = [f"{username}/sycophancy-rl-benchmarks"]
    rendered["model_sources"] = []
    return rendered


def write_kernel_metadata(
    path: Path,
    username: str,
    *,
    private: bool = True,
    accelerator: str = "gpu",
) -> dict[str, Any]:
    rendered = build_kernel_metadata(
        username, private=private, accelerator=accelerator
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
    return rendered


# --- env + doctor checks --------------------------------------------------


REQUIRED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
)


def doctor() -> dict[str, Any]:
    """Return a snapshot of the Kaggle environment.

    Never raises; surfaces failures as ``None`` fields so the caller can
    decide whether to abort.  Does NOT require Kaggle credentials.
    """

    snapshot: dict[str, Any] = {
        "kaggle_input_exists": KAGGLE_INPUT.exists(),
        "kaggle_working_exists": KAGGLE_WORKING.exists(),
        "kaggle_username": os.environ.get("KAGGLE_USERNAME"),
        "kaggle_key_present": bool(os.environ.get("KAGGLE_KEY")),
    }
    if KAGGLE_INPUT.exists():
        snapshot["input_datasets"] = sorted(
            p.name for p in KAGGLE_INPUT.iterdir() if p.is_dir()
        )
    if KAGGLE_WORKING.exists():
        snapshot["working_writable"] = os.access(KAGGLE_WORKING, os.W_OK)
    else:
        snapshot["working_writable"] = False

    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snapshot["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        snapshot["torch"] = None
        snapshot["cuda_available"] = False

    for pkg in REQUIRED_PACKAGES:
        try:
            snapshot[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            snapshot[pkg] = None

    return snapshot


# --- CLI surface ----------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    rendered = write_kernel_metadata(
        Path(args.output),
        args.username,
        private=not args.public,
        accelerator=args.accelerator,
    )
    print(json.dumps(rendered, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.metadata)
    if not path.exists():
        print(f"missing: {path}")
        return 2
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print("metadata must be a JSON object")
        return 2
    required = {"id", "code_file", "kernel_type", "enable_gpu"}
    missing = sorted(required - raw.keys())
    if missing:
        print(f"missing keys: {missing}")
        return 2
    if "/" not in raw["id"]:
        print("id must be <username>/<slug>")
        return 2
    print(json.dumps({"ok": True, "metadata": raw}, indent=2))
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    """Push the runner to Kaggle.

    Requires the pinned Kaggle CLI; the command refuses if the package isn't
    installed and prints a dry-run summary if ``--dry-run`` is set.
    """

    snapshot = doctor()
    if not args.dry_run:
        try:
            import kaggle  # noqa: F401
        except ImportError:
            print(
                "kaggle package not installed; install with "
                "`pip install kaggle` or pass --dry-run."
            )
            return 3
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(Path(args.metadata).parent),
            ],
            check=True,
        )
    print(json.dumps({"ok": True, "dry_run": args.dry_run, "snapshot": snapshot}, indent=2))
    return 0


def stage_kernel(
    *,
    plan_path: Path,
    output_dir: Path,
    username: str,
    dataset_slug: str,
    force: bool = False,
) -> Path:
    """Build a self-contained Kaggle kernel directory without uploading it."""

    from sycophancy_rl.experiments.pipeline import ExperimentPlan, run_plan_checks

    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment plan must be a JSON object.")
    plan = ExperimentPlan.from_dict(raw)
    run_plan_checks(plan)
    if plan.runner != "kaggle":
        raise ValueError("The staged plan must declare runner='kaggle'.")
    destination = output_dir.resolve()
    repository = _repository_root()
    if destination == repository or destination in repository.parents:
        raise ValueError("Refusing to use the repository root as a staging directory.")
    slug_parts = dataset_slug.split("/")
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if len(slug_parts) != 2 or any(
        not part or any(char not in safe_chars for char in part) for part in slug_parts
    ):
        raise ValueError("dataset_slug must be a safe Kaggle owner/name identifier.")
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"Staging directory already exists: {destination}. Pass --force to replace it."
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(repository / "deploy" / "kaggle" / "runner.py", destination / "runner.py")
    shutil.copy2(repository / "deploy" / "kaggle" / "requirements.lock", destination / "requirements.lock")
    shutil.copytree(
        repository / "src",
        destination / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(repository / "pyproject.toml", destination / "pyproject.toml")
    (destination / "experiment-plan.json").write_text(
        json.dumps(plan.frozen_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = build_kernel_metadata(username)
    metadata["dataset_sources"] = [dataset_slug]
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def cmd_stage(args: argparse.Namespace) -> int:
    path = stage_kernel(
        plan_path=Path(args.plan),
        output_dir=Path(args.output),
        username=args.username,
        dataset_slug=args.dataset,
        force=args.force,
    )
    print(json.dumps({"ok": True, "staging_directory": str(path)}, indent=2))
    return 0


def stage_dataset(
    *,
    training_path: Path,
    validation_path: Path,
    factual_test_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    dataset_slug: str,
    force: bool = False,
) -> Path:
    """Stage only governed inputs for a separate read-only Kaggle dataset."""

    from sycophancy_rl.data_prep.schema import read_jsonl
    from sycophancy_rl.data_prep.setup_data import is_fixture_row
    from sycophancy_rl.experiments.pipeline import hash_file

    training = read_jsonl(training_path, expected_role="training")
    validation = read_jsonl(validation_path, expected_role="validation")
    factual_test = read_jsonl(factual_test_path, expected_role="test")
    benchmark = read_jsonl(benchmark_path, expected_role="benchmark")
    if any(is_fixture_row(row) for row in (*training, *validation, *factual_test)):
        raise ValueError("Real Kaggle datasets cannot contain smoke fixtures.")
    if any(
        str(row.get("source", "")).casefold() == "anthropic/model-written-evals"
        for row in (*training, *validation, *factual_test)
    ):
        raise ValueError("Anthropic benchmark rows cannot be staged as training data.")
    train_ids = {str(row["example_id"]) for row in training}
    validation_ids = {str(row["example_id"]) for row in validation}
    factual_test_ids = {str(row["example_id"]) for row in factual_test}
    benchmark_ids = {str(row["example_id"]) for row in benchmark}
    development_ids = train_ids | validation_ids
    if (
        train_ids & validation_ids
        or factual_test_ids & development_ids
        or benchmark_ids & (development_ids | factual_test_ids)
    ):
        raise ValueError(
            "Dataset staging refused overlapping train/validation/test/benchmark IDs."
        )

    parts = dataset_slug.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError("dataset_slug must be owner/name.")
    destination = output_dir.resolve()
    repository = _repository_root()
    if destination == repository or destination in repository.parents:
        raise ValueError("Refusing a broad dataset staging directory.")
    if destination.exists():
        if not force:
            raise FileExistsError(f"Dataset staging directory exists: {destination}")
        shutil.rmtree(destination)
    (destination / "splits").mkdir(parents=True)
    (destination / "benchmarks").mkdir()
    copies = {
        "splits/train.jsonl": training_path,
        "splits/validation.jsonl": validation_path,
        "splits/test.jsonl": factual_test_path,
        "benchmarks/anthropic_sycophancy.jsonl": benchmark_path,
    }
    provenance_candidates = {
        "governance/split_manifest.json": training_path.parent / "split_manifest.json",
        "governance/import_manifest.json": training_path.parent.parent
        / "import_manifest.json",
        "governance/training_pool.manifest.json": training_path.parent.parent
        / "training_pool.manifest.json",
        "governance/benchmark_manifest.json": benchmark_path.with_name(
            "anthropic_sycophancy.manifest.json"
        ),
    }
    copies.update(
        {
            relative: source
            for relative, source in provenance_candidates.items()
            if source.is_file()
        }
    )
    for relative, source in copies.items():
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / relative)
    (destination / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "sycophancy-rl governed experiment data",
                "id": dataset_slug,
                "licenses": [{"name": "other"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "data-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "counts": {
                    "training": len(training),
                    "validation": len(validation),
                    "factual_test": len(factual_test),
                    "benchmark": len(benchmark),
                },
                "sha256": {
                    relative: hash_file(source) for relative, source in copies.items()
                },
                "anthropic_benchmark_used_for_training": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def cmd_stage_data(args: argparse.Namespace) -> int:
    path = stage_dataset(
        training_path=Path(args.train),
        validation_path=Path(args.validation),
        factual_test_path=Path(args.test),
        benchmark_path=Path(args.benchmark),
        output_dir=Path(args.output),
        dataset_slug=args.dataset,
        force=args.force,
    )
    print(json.dumps({"ok": True, "dataset_directory": str(path)}, indent=2))
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    """The ``syco train --runner kaggle --dry-run`` path.

    Validates the Kaggle environment without uploading anything and
    prints the manifest that would be written.  Useful for CI smoke
    tests that have no Kaggle credentials.
    """

    snapshot = doctor()
    manifest = {
        "ok": True,
        "dry_run": True,
        "runner": "kaggle",
        "snapshot": snapshot,
        "input_datasets_expected": ["sycophancy-rl-benchmarks"],
        "output_paths_expected": [
            str(KAGGLE_OUTPUTS),
            str(KAGGLE_CHECKPOINTS),
        ],
    }
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syco-kaggle",
        description="Kaggle runner helpers (dry-run by default).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Render kernel-metadata.json for the given Kaggle username.")
    init.add_argument("--username", required=True)
    init.add_argument("--output", default="deploy/kaggle/kernel-metadata.json")
    init.add_argument("--public", action="store_true", help="Push as a public kernel.")
    init.add_argument(
        "--accelerator",
        choices=("gpu", "tpu", "cpu"),
        default="gpu",
    )
    init.set_defaults(func=cmd_init)

    validate = sub.add_parser(
        "validate", help="Validate an existing kernel-metadata.json."
    )
    validate.add_argument("--metadata", default="deploy/kaggle/kernel-metadata.json")
    validate.set_defaults(func=cmd_validate)

    push = sub.add_parser(
        "push",
        help=(
            "Push the runner to Kaggle.  Refuses to run without "
            "--dry-run when the kaggle package is unavailable."
        ),
    )
    push.add_argument("--metadata", default="deploy/kaggle/kernel-metadata.json")
    push.add_argument("--dry-run", action="store_true")
    push.set_defaults(func=cmd_push)

    stage = sub.add_parser("stage", help="Build a self-contained Kaggle kernel directory.")
    stage.add_argument("--plan", required=True)
    stage.add_argument("--username", required=True)
    stage.add_argument("--dataset", required=True, help="Kaggle dataset slug: owner/name")
    stage.add_argument("--output", default=".kaggle-build")
    stage.add_argument("--force", action="store_true")
    stage.set_defaults(func=cmd_stage)

    stage_data = sub.add_parser(
        "stage-data", help="Build a governed Kaggle dataset upload directory."
    )
    stage_data.add_argument("--train", default="data/generated/splits/train.jsonl")
    stage_data.add_argument(
        "--validation", default="data/generated/splits/validation.jsonl"
    )
    stage_data.add_argument("--test", default="data/generated/splits/test.jsonl")
    stage_data.add_argument(
        "--benchmark", default="data/benchmarks/anthropic_sycophancy.jsonl"
    )
    stage_data.add_argument("--dataset", required=True, help="Kaggle owner/name")
    stage_data.add_argument("--output", default=".kaggle-data-build")
    stage_data.add_argument("--force", action="store_true")
    stage_data.set_defaults(func=cmd_stage_data)

    sub.add_parser("doctor", help="Print the Kaggle environment snapshot.").set_defaults(
        func=lambda _args: print(json.dumps(doctor(), indent=2)) or 0
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
