"""``syco`` command-line interface entry point.

The CLI is the single user-facing entry point for local training and
benchmarking.  Each subcommand delegates to an existing module that has
its own safety checks:

* ``syco doctor`` — prints hardware / dependency / data state without
  touching anything.
* ``syco setup`` — runs the non-destructive setup helpers (Phase 0).
* ``syco prepare-data`` — downloads a pinned ARC snapshot and prepares splits.
* ``syco import-data`` — normalizes a user-owned choice dataset and prepares splits.
* ``syco train`` — dispatches to the local trainer or the Kaggle
  staging helper. Local training is the default; Kaggle jobs use the
  canonical ``syco run`` plan-and-stage workflow.
* ``syco benchmark`` — dispatches to the real benchmark runner when
  the heavy optional dependencies are available, and refuses to
  silently claim success when they are not.
* ``syco compare`` — runs a paired before/after benchmark comparison.
* ``syco kaggle`` — Kaggle data/kernel staging, validation, push, and doctor helpers.
* ``syco version`` — prints the package version.

The CLI never downloads model weights or datasets by default and never
starts a real training run unless ``syco train`` is invoked without
``--preflight-only``.  Its job is to be the one shell entry point that
always behaves the same way regardless of which machine, container, or
notebook environment it is invoked from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from sycophancy_rl import __version__
from sycophancy_rl.evaluation.run_benchmark import DEFAULT_BENCHMARK_MAX_NEW_TOKENS

__all__ = ["main"]

TRAINING_PROFILE_NAMES = (
    "smoke",
    "local_8gb",
    "local_16gb",
    "qlora_7b_16gb",
    "kaggle_online_smoke",
    "qwen25_05b_online",
    "qwen25_7b_online",
)


def _custom_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    targets = tuple(
        value.strip()
        for value in str(getattr(args, "custom_lora_targets", "all-linear")).split(",")
        if value.strip()
    )
    return {
        "custom_supports_4bit": bool(getattr(args, "custom_supports_4bit", False)),
        "custom_lora_targets": targets,
        "custom_context_length": int(getattr(args, "custom_context_length", 2048)),
        "custom_min_vram_gib": float(getattr(args, "custom_min_vram_gib", 0.0)),
    }


def _add_custom_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--custom-supports-4bit",
        action="store_true",
        help="Declare NF4 compatibility for an unregistered custom model.",
    )
    parser.add_argument(
        "--custom-lora-targets",
        default="all-linear",
        help="Comma-separated PEFT target modules for an unregistered model.",
    )
    parser.add_argument("--custom-context-length", type=int, default=2048)
    parser.add_argument("--custom-min-vram-gib", type=float, default=0.0)


VERSION = __version__


def _train_argv(args: argparse.Namespace) -> list[str]:
    """Translate CLI flags into the trainer's argv.

    The trainer parses its own arguments via ``_parse_args``; the CLI
    must not silently force flags the caller did not request.  In
    particular, ``--allow-cpu``, ``--no-4bit``, and ``--preflight-only``
    are all off by default and only enabled when the caller explicitly
    asks for them.

    The trainer's ``--model-revision`` defaults to the reference
    baseline SHA.  To prevent the trainer's default from silently
    leaking into a custom-model registry lookup we always forward
    ``--model-revision`` (with an explicit empty string when the CLI
    caller did not pin one) so the trainer sees the caller's choice
    verbatim.
    """

    argv: list[str] = ["--profile", args.profile]
    argv += ["--model-id", args.model_id]
    argv += ["--model-revision", args.model_revision or ""]
    argv += ["--seed", str(args.seed)]
    argv += ["--output-root", str(args.output_root)]
    argv += ["--reward-profile", args.reward_profile]
    argv += ["--rollout-mode", getattr(args, "rollout_mode", "online")]
    argv += ["--max-pushback-turns", str(getattr(args, "max_pushback_turns", 1))]
    for flag, attribute in (
        ("--train", "train_path"),
        ("--validation", "validation_path"),
        ("--benchmark", "benchmark_path"),
        ("--run-name", "run_name"),
        ("--system-prompt-condition", "system_prompt_condition"),
        ("--learning-rate", "learning_rate"),
        ("--beta", "beta"),
        ("--max-steps", "max_steps"),
        ("--num-generations", "num_generations"),
        ("--resume-from-checkpoint", "resume_from_checkpoint"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            argv.extend([flag, str(value)])
    argv += ["--runner", getattr(args, "runner", "local")]
    if args.allow_cpu:
        argv.append("--allow-cpu")
    if args.no_4bit:
        argv.append("--no-4bit")
    if args.preflight_only:
        argv.append("--preflight-only")
    if getattr(args, "resume", False):
        argv.append("--resume")
    if getattr(args, "allow_unpinned_model", False):
        argv.append("--allow-unpinned-model")
    custom = _custom_model_kwargs(args)
    if custom["custom_supports_4bit"]:
        argv.append("--custom-supports-4bit")
    argv += ["--custom-lora-targets", ",".join(custom["custom_lora_targets"])]
    argv += ["--custom-context-length", str(custom["custom_context_length"])]
    argv += ["--custom-min-vram-gib", str(custom["custom_min_vram_gib"])]
    return argv


def _experiment_plan_from_args(args: argparse.Namespace):
    from sycophancy_rl.experiments.pipeline import build_default_plan
    from sycophancy_rl.training.model_registry import resolve_profile

    custom = _custom_model_kwargs(args)
    model_profile = resolve_profile(
        args.model_id,
        model_revision=args.model_revision or None,
        allow_unpinned=args.allow_unpinned_model,
        **custom,
    )

    return build_default_plan(
        run_id=args.run_id,
        model_id=args.model_id,
        model_revision=model_profile.revision,
        training_path=args.train_path,
        validation_path=args.validation_path,
        benchmark_path=args.benchmark_path,
        factual_test_path=args.factual_test_path,
        preference_only=bool(getattr(args, "preference_only", False)),
        seed=args.seed,
        training_profile=args.profile,
        reward_profile=args.reward_profile,
        prompt_condition=args.system_prompt_condition,
        prompt_variants=(
            tuple(value.strip() for value in args.prompt_variants.split(",") if value.strip())
        ),
        generation_settings={
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        },
        output_root=args.output_root,
        checkpoint_dir=args.checkpoint_root / args.run_id,
        runner=args.runner,
        publishable=args.publishable,
        capability_tasks=tuple(
            task.strip() for task in args.capability_tasks.split(",") if task.strip()
        ),
        capability_limit=args.capability_limit,
        capability_batch_size=args.capability_batch_size,
        maximum_capability_drop=args.maximum_capability_drop,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        load_in_4bit=not args.no_4bit,
        allow_cpu=args.allow_cpu,
        allow_unpinned_model=args.allow_unpinned_model,
        **custom,
        resume=args.resume,
        extra={
            key: value
            for key, value in {
                "learning_rate": args.learning_rate,
                "beta": args.beta,
                "max_steps": args.max_steps,
                "num_generations": args.num_generations,
                "rollout_mode": args.rollout_mode,
                "max_pushback_turns": args.max_pushback_turns,
            }.items()
            if value is not None
        },
    )


def _cmd_run(args: argparse.Namespace) -> int:
    """Plan, dry-run, or execute the complete before/train/after workflow."""

    from sycophancy_rl.experiments.pipeline import ExecutionMode, Pipeline, scrub_secrets

    mode = ExecutionMode.EXECUTE if args.execute else (
        ExecutionMode.DRY_RUN if args.dry_run else ExecutionMode.PLAN
    )
    if mode is ExecutionMode.EXECUTE and args.runner == "kaggle":
        print(
            "Kaggle execution must run inside deploy/kaggle/runner.py. "
            "Use `syco kaggle stage` and then push the staged kernel.",
            file=sys.stderr,
        )
        return 3
    plan = _experiment_plan_from_args(args)
    if args.plan_output is not None:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(
            json.dumps(plan.frozen_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    result = Pipeline(plan, mode=mode).run()
    print(
        scrub_secrets(
            {
                "mode": result.mode.value,
                "completed_stages": result.completed_stages,
                "errors": result.errors,
                "manifest": result.manifest,
                "artifacts": result.artifacts,
            }
        )
    )
    return 0 if not result.errors else 5


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Print a one-shot diagnostic summary without mutating anything."""

    snapshot: dict[str, Any] = {}
    try:
        import platform as _platform

        snapshot["python"] = _platform.python_version()
        snapshot["platform"] = _platform.platform()
    except Exception:  # pragma: no cover - reporting only
        pass
    snapshot["python_supported"] = (3, 10) <= sys.version_info[:2] < (3, 13)

    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["torch_cuda_version"] = torch.version.cuda
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snapshot["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        snapshot["torch"] = None
        snapshot["cuda_available"] = False

    try:
        for pkg in ("transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"):
            try:
                snapshot[pkg] = importlib_metadata.version(pkg)
            except importlib_metadata.PackageNotFoundError:
                snapshot[pkg] = None
    except Exception:  # pragma: no cover - reporting only
        pass

    pool_path = Path("data/processed/training_pool.jsonl")
    snapshot["training_pool"] = {
        "exists": pool_path.exists(),
        "size_bytes": pool_path.stat().st_size if pool_path.exists() else 0,
    }
    split_dir = Path("data/splits")
    snapshot["splits_dir"] = {
        "exists": split_dir.exists(),
        "files": (
            sorted(p.name for p in split_dir.iterdir() if p.is_file())
            if split_dir.exists()
            else []
        ),
    }
    experiment_dir = Path("data/anthropic_experiment")
    snapshot["anthropic_experiment"] = {
        "exists": experiment_dir.exists(),
        "files": (
            sorted(path.name for path in experiment_dir.iterdir() if path.is_file())
            if experiment_dir.exists()
            else []
        ),
    }
    required_training_packages = ("torch", "transformers", "trl", "peft", "datasets")
    snapshot["cpu_ready"] = all(snapshot.get(pkg) for pkg in required_training_packages)
    snapshot["gpu_training_ready"] = bool(
        snapshot["cpu_ready"]
        and snapshot.get("cuda_available")
        and snapshot.get("bitsandbytes")
    )
    snapshot["gpu_status"] = (
        "ready"
        if snapshot["gpu_training_ready"]
        else "not verified: CUDA and bitsandbytes are required for QLoRA"
    )
    expected_versions = {
        "transformers": "5.0.0",
        "trl": "1.8.0",
        "peft": "0.19.1",
        "datasets": "5.0.0",
        "accelerate": "1.13.0",
        "bitsandbytes": "0.49.2",
    }
    snapshot["expected_versions"] = expected_versions
    snapshot["version_match"] = {
        name: snapshot.get(name) == version for name, version in expected_versions.items()
    }
    snapshot["benchmark_ready"] = bool(
        snapshot["python_supported"]
        and snapshot.get("torch")
        and snapshot.get("transformers")
        and snapshot["version_match"]["transformers"]
    )
    snapshot["training_ready"] = bool(
        snapshot["python_supported"]
        and snapshot["gpu_training_ready"]
        and snapshot.get("accelerate")
        and all(snapshot["version_match"].values())
    )
    snapshot["benchmark_status"] = (
        "BENCHMARK READY" if snapshot["benchmark_ready"] else "BENCHMARK NOT READY"
    )
    snapshot["training_status"] = (
        "TRAINING READY" if snapshot["training_ready"] else "TRAINING NOT READY"
    )
    snapshot["readiness_hierarchy"] = {
        "level_1_static_code_ready": None,
        "level_2_kaggle_model_load_verified": False,
        "level_3_20_real_episodes_verified": False,
        "level_4_optimizer_step_verified": False,
        "level_5_adapter_save_reload_verified": False,
        "level_6_small_before_train_after_verified": False,
    }
    print(json.dumps(snapshot, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_setup(_args: argparse.Namespace) -> int:
    """Run the non-destructive setup helpers (Phase 0)."""

    from sycophancy_rl.data_prep import setup_data

    print(setup_data.ensure_fixture_pool())
    print(setup_data.ensure_splits())
    return 0


def _cmd_prepare_data(args: argparse.Namespace) -> int:
    """Download pinned ARC training data and create immutable real splits."""

    from sycophancy_rl.data_prep import prepare_training_data, split_data
    from sycophancy_rl.data_prep.schema import read_jsonl

    count = prepare_training_data.prepare_training_pool(
        output_path=args.output,
        manifest_path=args.manifest,
        revision=args.revision,
        seed=args.seed,
        max_questions=args.max_questions,
    )
    rows = read_jsonl(args.output, expected_role="training")
    splits = split_data.split_examples(
        rows,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    split_data.write_splits(
        splits,
        args.split_dir,
        seed=args.seed,
        source_path=args.output,
    )
    print(f"Prepared {count} real training rows and immutable splits under {args.split_dir}.")
    return 0


def _cmd_import_data(args: argparse.Namespace) -> int:
    """Normalize a user dataset and create immutable governed splits."""

    from sycophancy_rl.data_prep.import_user_dataset import import_user_dataset

    destination = import_user_dataset(
        input_path=args.input,
        dataset_name=args.dataset_name,
        output_root=args.output_dir,
        input_format=args.format,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        license_id=args.license,
        source_url=args.source_url,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "dataset_directory": str(destination),
                "train": str(destination / "splits" / "train.jsonl"),
                "validation": str(destination / "splits" / "validation.jsonl"),
                "test": str(destination / "splits" / "test.jsonl"),
            },
            indent=2,
        )
    )
    return 0


def _cmd_import_openr1(args: argparse.Namespace) -> int:
    """Stream a pinned OpenR1 sample and create governed immutable splits."""

    from sycophancy_rl.data_prep.import_openr1 import import_openr1

    manifest = import_openr1(
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _cmd_train_local(args: argparse.Namespace) -> int:
    """Dispatch ``syco train --runner local`` to the real trainer.

    Training runs only when ``--preflight-only`` is absent; otherwise
    we delegate to the trainer's preflight path that never loads
    weights.  All relevant arguments (model id, revision, profile, seed,
    output dir, reward profile, ``--allow-cpu``, ``--no-4bit``) are
    forwarded verbatim.
    """

    from sycophancy_rl.training import train_grpo

    argv = _train_argv(args)
    parsed = train_grpo._parse_args(argv)
    if parsed.preflight_only:
        result = train_grpo.run_preflight(parsed)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    output = train_grpo.run_training(parsed)
    print(f"Training completed; artifacts saved under {output}.")
    return 0


def _cmd_train_kaggle(args: argparse.Namespace) -> int:
    """Dispatch ``syco train --runner kaggle`` to the Kaggle runner.

    Direct submission is intentionally unavailable because a complete
    before/train/after experiment requires a frozen plan and governed
    dataset. The canonical ``syco run`` plus ``syco kaggle stage`` flow
    creates both artifacts before a user explicitly pushes the kernel.
    """

    if not args.dry_run:
        print(
            "syco train --runner kaggle uses the staged experiment workflow. "
            "Create a frozen plan with `syco run --runner kaggle --plan-output "
            "experiment-plan.json`, then use `syco kaggle stage-data`, "
            "`syco kaggle stage`, and `syco kaggle push`. Re-run this command "
            "with --dry-run only for its legacy diagnostic snapshot.",
            file=sys.stderr,
        )
        return 3
    from sycophancy_rl.experiments import kaggle as kaggle_mod

    return kaggle_mod.cmd_dry_run(
        argparse.Namespace(
            model_id=args.model_id,
            model_revision=args.model_revision,
            profile=args.profile,
            seed=args.seed,
            output_root=str(args.output_root),
            reward_profile=args.reward_profile,
            allow_cpu=args.allow_cpu,
            no_4bit=args.no_4bit,
        )
    )


def _cmd_train(args: argparse.Namespace) -> int:
    """Route ``syco train`` to the selected runner."""

    if args.runner == "kaggle":
        return _cmd_train_kaggle(args)
    return _cmd_train_local(args)


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare before/after benchmark reports and print a summary."""

    from sycophancy_rl.evaluation import compare_runs

    payload = compare_runs.compare_runs_from_paths(
        Path(args.before),
        Path(args.after),
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the real benchmark via ``sycophancy_rl.evaluation.run_benchmark``.

    The benchmark imports torch / transformers / peft and may require a
    GPU.  When those dependencies are missing the command refuses to
    silently report success; it exits with a clear non-zero code so the
    caller can distinguish "the benchmark ran" from "the benchmark
    could not run".
    """

    try:
        from sycophancy_rl.evaluation import run_benchmark
    except ImportError as exc:  # pragma: no cover - import guard
        print(f"syco benchmark: cannot import benchmark runner: {exc}", file=sys.stderr)
        return 2
    try:
        import torch  # noqa: F401  -- explicit dependency check
        import transformers  # noqa: F401
    except ImportError as exc:
        print(
            f"syco benchmark: torch/transformers not installed ({exc}). "
            "Install the pinned ML dependencies or run "
            "`python -m sycophancy_rl.evaluation.run_benchmark` directly after "
            "installing them.",
            file=sys.stderr,
        )
        return 4
    from sycophancy_rl.training.model_registry import resolve_profile_safe

    try:
        profile = resolve_profile_safe(
            args.model_id,
            model_revision=args.model_revision or None,
            allow_unpinned=args.allow_unpinned_model,
            **_custom_model_kwargs(args),
        )
    except ValueError as exc:
        print(f"syco benchmark: {exc}", file=sys.stderr)
        return 2
    if args.load_in_4bit and not profile.supports_4bit:
        print(
            f"syco benchmark: model {profile.model_id!r} is not configured for 4-bit loading.",
            file=sys.stderr,
        )
        return 2
    benchmark_argv: list[str] = [
        "--run-name",
        args.run_name,
        "--model-id",
        profile.model_id,
        "--model-revision",
        profile.revision or args.model_revision,
        "--seed",
        str(args.seed),
        "--benchmark",
        str(args.benchmark_path),
        "--prompt-variants",
        args.prompt_variants,
        "--system-prompt-condition",
        args.system_prompt_condition,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--batch-size",
        str(args.batch_size),
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(args.shard_index),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.adapter:
        benchmark_argv += ["--adapter", str(args.adapter)]
    if args.load_in_4bit:
        benchmark_argv.append("--load-in-4bit")
    if args.do_sample:
        benchmark_argv.append("--do-sample")
    if args.resume:
        benchmark_argv.append("--resume")
    if args.expected_example_id_sha256:
        benchmark_argv += [
            "--expected-example-id-sha256",
            args.expected_example_id_sha256,
        ]
    original_argv = sys.argv
    try:
        sys.argv = ["run_benchmark", *benchmark_argv]
        run_benchmark.main()
    finally:
        sys.argv = original_argv
    return 0


def _cmd_kaggle(args: argparse.Namespace) -> int:
    """Dispatch the ``syco kaggle`` subcommands."""

    from sycophancy_rl.experiments import kaggle as kaggle_mod

    sub_argv = [args.kaggle_command, *args.kaggle_args]
    return kaggle_mod.main(sub_argv)


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"syco {VERSION}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syco",
        description="sycophancy-rl CLI (local trainer orchestrator).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "doctor",
        help="Print a read-only diagnostic summary of the environment.",
    )
    sub.add_parser(
        "setup",
        help="Run the non-destructive setup helpers (smoke fixtures + splits).",
    )
    prepare = sub.add_parser(
        "prepare-data",
        help="Download ARC train and create generated leakage-safe splits.",
    )
    prepare.add_argument("--output", type=Path, default=Path("data/generated/training_pool.jsonl"))
    prepare.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/training_pool.manifest.json"),
    )
    prepare.add_argument("--split-dir", type=Path, default=Path("data/generated/splits"))
    prepare.add_argument("--revision", default=None)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-questions", type=int, default=None)
    prepare.add_argument("--validation-fraction", type=float, default=0.15)
    prepare.add_argument("--test-fraction", type=float, default=0.15)

    import_data = sub.add_parser(
        "import-data",
        help="Normalize a user JSON/JSONL/CSV choice dataset and create governed splits.",
    )
    import_data.add_argument("--input", type=Path, required=True)
    import_data.add_argument("--dataset-name", required=True)
    import_data.add_argument("--output-dir", type=Path, default=Path("data/generated/user"))
    import_data.add_argument(
        "--format",
        choices=("auto", "canonical", "simple-choice"),
        default="auto",
    )
    import_data.add_argument("--seed", type=int, default=42)
    import_data.add_argument("--validation-fraction", type=float, default=0.15)
    import_data.add_argument("--test-fraction", type=float, default=0.15)
    import_data.add_argument("--license", default="other")
    import_data.add_argument("--source-url", default=None)

    openr1 = sub.add_parser(
        "import-openr1",
        help="Stream a pinned numeric OpenR1-Math sample into governed A/B splits.",
    )
    openr1.add_argument("--output-dir", type=Path, required=True)
    openr1.add_argument("--sample-count", type=int, default=500)
    openr1.add_argument("--seed", type=int, default=42)
    openr1.add_argument("--validation-fraction", type=float, default=0.1)
    openr1.add_argument("--test-fraction", type=float, default=0.1)

    train = sub.add_parser(
        "train",
        help=(
            "Run the trainer. Dispatches to the local runner by default "
            "or to the Kaggle runner with --runner kaggle. Training only "
            "happens when --preflight-only is omitted."
        ),
    )
    train.add_argument(
        "--profile",
        choices=TRAINING_PROFILE_NAMES,
        default="local_8gb",
        help="Which training profile to use (default: local_8gb).",
    )
    train.add_argument(
        "--runner",
        choices=("local", "kaggle"),
        default="local",
        help="Which runner to dispatch to (default: local).",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Kaggle-only: print a snapshot instead of starting a real run.",
    )
    train.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate hardware, data, and config without loading weights.",
    )
    train.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id to train (default: the pinned reference model).",
    )
    train.add_argument(
        "--model-revision",
        default="",
        help="Optional pinned revision. The registry pins the registered models; pass only to override or to use a custom model.",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to the trainer (default: 42).",
    )
    train.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Root directory for run artifacts (default: outputs/checkpoints).",
    )
    train.add_argument(
        "--reward-profile",
        choices=("combined", "answer_only", "diagnostic_format_only"),
        default="combined",
        help="Which reward ablation to use during training (default: combined).",
    )
    train.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Acknowledge that CPU-only training is fine (required for --profile smoke on machines without CUDA).",
    )
    train.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit QLoRA quantization (off by default; the trainer uses 4-bit when a CUDA GPU is detected).",
    )
    train.add_argument("--train-path", type=Path, default=Path("data/anthropic_experiment/train.jsonl"))
    train.add_argument(
        "--validation-path", type=Path, default=Path("data/anthropic_experiment/validation.jsonl")
    )
    train.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/anthropic_experiment/benchmark.jsonl"),
    )
    train.add_argument("--run-name", default=None)
    train.add_argument(
        "--system-prompt-condition",
        choices=("none", "neutral", "anti_sycophancy", "pro_agreement_control"),
        default="neutral",
    )
    train.add_argument("--learning-rate", type=float, default=None)
    train.add_argument("--beta", type=float, default=None)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--num-generations", type=int, default=None)
    train.add_argument("--rollout-mode", choices=("online", "prepared"), default="online")
    train.add_argument("--max-pushback-turns", type=int, default=1)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--resume-from-checkpoint", default=None)
    train.add_argument("--allow-unpinned-model", action="store_true")
    _add_custom_model_arguments(train)

    run = sub.add_parser(
        "run",
        help="Run the complete baseline -> training -> after -> comparison workflow.",
    )
    run.add_argument("--run-id", required=True)
    run.add_argument("--runner", choices=("local", "kaggle"), default="local")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    run.add_argument(
        "--profile",
        choices=TRAINING_PROFILE_NAMES,
        default="local_8gb",
    )
    run.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    run.add_argument(
        "--model-revision",
        default="",
    )
    run.add_argument(
        "--train-path", type=Path, default=Path("data/anthropic_experiment/train.jsonl")
    )
    run.add_argument(
        "--validation-path",
        type=Path,
        default=Path("data/anthropic_experiment/validation.jsonl"),
    )
    run.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/anthropic_experiment/benchmark.jsonl"),
    )
    run.add_argument(
        "--factual-test-path",
        type=Path,
        default=None,
        help="Held-out objective test split used for factual before/after evaluation.",
    )
    run.add_argument(
        "--preference-only",
        action="store_true",
        help="Skip the optional objective factual benchmark.",
    )
    run.add_argument("--output-root", type=Path, default=Path("outputs/experiments"))
    run.add_argument("--checkpoint-root", type=Path, default=Path("outputs/checkpoints"))
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--reward-profile",
        choices=("combined", "answer_only", "diagnostic_format_only"),
        default="combined",
    )
    run.add_argument(
        "--system-prompt-condition",
        choices=("none", "neutral", "anti_sycophancy", "pro_agreement_control"),
        default="neutral",
    )
    run.add_argument("--prompt-variants", default="original,swap_options")
    run.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional benchmark cap for smoke/debug runs; default evaluates all held-out rows.",
    )
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_BENCHMARK_MAX_NEW_TOKENS,
        help=(
            "Maximum tokens for each before/after benchmark response "
            f"(default: {DEFAULT_BENCHMARK_MAX_NEW_TOKENS})."
        ),
    )
    run.add_argument("--do-sample", action="store_true")
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--top-k", type=int, default=0)
    run.add_argument("--repetition-penalty", type=float, default=1.0)
    run.add_argument("--no-4bit", action="store_true")
    run.add_argument("--allow-cpu", action="store_true")
    run.add_argument("--allow-unpinned-model", action="store_true")
    _add_custom_model_arguments(run)
    run.add_argument("--publishable", action="store_true")
    run.add_argument(
        "--capability-tasks",
        default="",
        help="Comma-separated lm-evaluation-harness tasks (required for --publishable).",
    )
    run.add_argument(
        "--capability-limit",
        type=int,
        default=None,
        help="Debug-only per-task example limit; forbidden for --publishable.",
    )
    run.add_argument("--capability-batch-size", type=int, default=1)
    run.add_argument("--maximum-capability-drop", type=float, default=0.02)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--plan-output",
        type=Path,
        default=None,
        help="Write the frozen plan JSON used by a staged Kaggle runner.",
    )
    run.add_argument("--learning-rate", type=float, default=None)
    run.add_argument("--beta", type=float, default=None)
    run.add_argument("--max-steps", type=int, default=None)
    run.add_argument("--num-generations", type=int, default=None)
    run.add_argument("--rollout-mode", choices=("online", "prepared"), default="online")
    run.add_argument("--max-pushback-turns", type=int, default=1)

    benchmark = sub.add_parser(
        "benchmark",
        help="Run the benchmark via sycophancy_rl.evaluation.run_benchmark.",
    )
    benchmark.add_argument(
        "--run-name",
        required=True,
        help="Run directory name under --output-dir.",
    )
    benchmark.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id (default: the pinned reference model).",
    )
    benchmark.add_argument(
        "--model-revision",
        default="",
        help="Optional pinned revision; the registry pins the registered models.",
    )
    benchmark.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional path to a PEFT adapter produced by `syco train`.",
    )
    benchmark.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/anthropic_experiment/benchmark.jsonl"),
    )
    benchmark.add_argument("--prompt-variants", default="original,swap_options")
    benchmark.add_argument("--allow-unpinned-model", action="store_true")
    _add_custom_model_arguments(benchmark)
    benchmark.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the benchmark (default: 42).",
    )
    benchmark.add_argument(
        "--system-prompt-condition",
        choices=("none", "neutral", "anti_sycophancy", "pro_agreement_control"),
        default="neutral",
    )
    benchmark.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_BENCHMARK_MAX_NEW_TOKENS,
        help=(
            "Maximum completion tokens (default: "
            f"{DEFAULT_BENCHMARK_MAX_NEW_TOKENS}; use less than 128 only for "
            "smoke/debug evaluation)."
        ),
    )
    benchmark.add_argument("--temperature", type=float, default=1.0)
    benchmark.add_argument("--top-p", type=float, default=1.0)
    benchmark.add_argument("--top-k", type=int, default=0)
    benchmark.add_argument("--repetition-penalty", type=float, default=1.0)
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--num-shards", type=int, default=1)
    benchmark.add_argument("--shard-index", type=int, default=0)
    benchmark.add_argument("--resume", action="store_true")
    benchmark.add_argument("--expected-example-id-sha256", default=None)
    benchmark.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model with 4-bit quantization.",
    )
    benchmark.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic decoding (off by default; deterministic by default).",
    )
    benchmark.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluations"),
        help="Root directory for benchmark artifacts (default: outputs/evaluations).",
    )

    compare = sub.add_parser(
        "compare",
        help="Compare before/after benchmark reports.",
    )
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)

    kaggle = sub.add_parser(
        "kaggle",
        help="Kaggle staging, validation, push, and doctor helpers.",
    )
    kaggle.add_argument(
        "kaggle_command",
        choices=("init", "validate", "stage", "stage-data", "push", "doctor"),
        help="Kaggle subcommand to run.",
    )
    kaggle.add_argument(
        "kaggle_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the Kaggle subcommand.",
    )

    sub.add_parser(
        "version",
        help="Print the syco CLI version.",
    )

    return parser


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "doctor": _cmd_doctor,
    "setup": _cmd_setup,
    "prepare-data": _cmd_prepare_data,
    "import-data": _cmd_import_data,
    "import-openr1": _cmd_import_openr1,
    "train": _cmd_train,
    "run": _cmd_run,
    "benchmark": _cmd_benchmark,
    "compare": _cmd_compare,
    "kaggle": _cmd_kaggle,
    "version": _cmd_version,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv, dispatch to the matching handler, and return an exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.command]
    try:
        return handler(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        # These are the documented recoverable errors.  Print the message
        # without a traceback so the CLI is friendly to humans.
        print(f"syco {args.command}: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        # RuntimeError is the documented signal for a trainer / runner
        # precondition failure (missing CUDA, missing disk space,
        # unsupported profile, etc.).  Surface it without a traceback
        # and exit non-zero so the caller can detect it.
        print(f"syco {args.command}: {exc}", file=sys.stderr)
        return 5
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
