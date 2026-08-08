"""Leakage-safe, registry-aware local QLoRA + GRPO training entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sycophancy_rl.data_prep.schema import read_jsonl
from sycophancy_rl.data_prep.setup_data import is_fixture_row
from sycophancy_rl.evaluation.prompts import apply_system_prompt
from sycophancy_rl.evaluation.run_benchmark import (
    DEFAULT_MODEL_ID,
    set_reproducible_seed,
)
from sycophancy_rl.experiments.manifest import (
    Stage,
    Status,
    is_resume_compatible,
    load_manifest,
    new_manifest,
    sha256_file,
    transition,
    write_atomically,
)
from sycophancy_rl.reward.reward_fn import REWARD_PROFILES, make_composite_reward_func
from sycophancy_rl.training.grpo_config import PROFILES as TRAINING_PROFILES
from sycophancy_rl.training.grpo_config import get_grpo_config
from sycophancy_rl.training.model_registry import (
    list_supported,
    resolve_profile,
)
from sycophancy_rl.training.reliability import (
    assert_disk_space,
    graceful_shutdown,
    promote_adapter_atomic,
    record_completed,
    record_failed,
    record_interrupted,
    record_running,
)
from sycophancy_rl.training.tracking import JsonlTrackingCallback, RewardEarlyStoppingCallback

DEFAULT_TRAIN = Path("data/splits/train.jsonl")
DEFAULT_VALIDATION = Path("data/splits/validation.jsonl")


SMOKE_PROFILE = "smoke"
REAL_TRAINING_PROFILES = frozenset(set(TRAINING_PROFILES) - {SMOKE_PROFILE})


def _is_benchmark_row(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata", {})
    if metadata.get("benchmark_only"):
        return True
    return str(row.get("source", "")).casefold() == "anthropic/model-written-evals"


def _require_training_rows(
    path: Path,
    role: str,
    *,
    profile: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path, expected_role=role)
    benchmark_rows = [
        row["example_id"] for row in rows if _is_benchmark_row(row)
    ]
    if benchmark_rows:
        raise ValueError(
            "Evaluation-only benchmark rows reached the trainer: "
            + ", ".join(benchmark_rows[:5])
        )
    fixture_rows = [
        row["example_id"] for row in rows if is_fixture_row(row)
    ]
    if fixture_rows and profile != SMOKE_PROFILE:
        sample = ", ".join(fixture_rows[:5])
        raise ValueError(
            f"Smoke fixture rows reached the {profile} profile: {sample}. "
            "Smoke fixtures are intended for tests and the smoke preflight "
            "only; they are not real training data. Prepare real training "
            "data with `python -m sycophancy_rl.data_prep.prepare_training_data` "
            "(produces data/processed/training_pool.jsonl) and rebuild the "
            "splits with `python -m sycophancy_rl.data_prep.split_data` before "
            "running real training."
        )
    return rows


def _load_dataset(
    path: Path,
    role: str,
    system_prompt_condition: str,
    *,
    profile: str,
):
    rows = _require_training_rows(path, role, profile=profile)
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install the pinned datasets dependency before training.") from exc
    prepared: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        copy["prompt"] = apply_system_prompt(
            row["prompt"],
            system_prompt_condition,
        )
        prepared.append(copy)
    return Dataset.from_list(prepared)


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "bitsandbytes"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _resolve_model(args: argparse.Namespace) -> Any:
    """Resolve the model via the supported-model registry.

    The registry is the single source of truth for revisions.  For
    registered models the pinned revision is returned; for an unknown
    model id we use the caller-supplied ``--model-revision`` (which
    may be empty).  A known model id never silently inherits an
    unrelated revision, and an unknown model id never silently
    inherits the reference baseline revision.

    The argument parser uses an empty revision by default. Registered
    models resolve to their registry pin; custom models must supply an
    exact Hub commit SHA and explicit compatibility settings.
    """

    model_id = args.model_id
    explicit_revision = getattr(args, "model_revision", None)
    revision_for_registry = explicit_revision or None
    custom_targets = tuple(
        target.strip()
        for target in str(getattr(args, "custom_lora_targets", "all-linear")).split(",")
        if target.strip()
    )
    profile = resolve_profile(
        model_id,
        model_revision=revision_for_registry,
        allow_unpinned=bool(getattr(args, "allow_unpinned_model", False)),
        custom_supports_4bit=bool(getattr(args, "custom_supports_4bit", False)),
        custom_lora_targets=custom_targets,
        custom_context_length=int(getattr(args, "custom_context_length", 2048)),
        custom_min_vram_gib=float(getattr(args, "custom_min_vram_gib", 0.0)),
    )
    if model_id in {p.model_id for p in list_supported()}:
        # Registered models: trust the pinned revision.  A mismatch
        # with --model-revision would have raised inside the registry.
        return profile
    # Unknown model id: the resolved profile has whatever revision
    # the caller passed (possibly empty); trust the registry and
    # do NOT silently substitute the SmolLM default.
    return profile


def _preflight(
    profile: str,
    allow_cpu: bool,
    *,
    model_min_vram_gib: float = 0.0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cuda_available": False,
        "torch": None,
        "cuda_version": None,
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        cuda = torch.cuda.is_available()
        result["cuda_available"] = cuda
    except ImportError:
        if not allow_cpu or profile != SMOKE_PROFILE:
            raise RuntimeError(
                "PyTorch is not installed. Install torch to train, or pass "
                "--allow-cpu with --profile smoke for a no-torch preflight."
            ) from None
        torch = None  # type: ignore[assignment]
        cuda = False
    if not cuda and not allow_cpu:
        raise RuntimeError(
            "No CUDA GPU detected. GRPO on CPU is a smoke-only path; pass "
            "--allow-cpu explicitly or use a CUDA-capable local/Colab machine."
        )
    try:
        import psutil

        result["system_ram_gib"] = psutil.virtual_memory().total / 1024**3
    except ImportError:
        result["system_ram_gib"] = None
    if cuda and torch is not None:
        properties = torch.cuda.get_device_properties(0)
        memory_gib = properties.total_memory / 1024**3
        result.update({"gpu": properties.name, "gpu_memory_gib": memory_gib})
        profile_minimums = {
            "smoke": 0.0,
            "local_8gb": 7.0,
            "local_16gb": 12.0,
            "qlora_7b_16gb": 14.0,
        }
        recommended = max(profile_minimums.get(profile, 0.0), model_min_vram_gib)
        if memory_gib < recommended:
            raise RuntimeError(
                f"Profile {profile} expects about {recommended:.0f}+ GiB VRAM; "
                f"detected {memory_gib:.1f} GiB. Use a smaller profile."
            )
    return result


def _model_and_tokenizer(
    *,
    model_id: str,
    model_revision: str | None,
    load_in_4bit: bool,
    allow_cpu: bool,
    lora_targets: tuple[str, ...] = ("all-linear",),
):
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch, transformers, peft, and bitsandbytes."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=model_revision,
            use_fast=True,
            trust_remote_code=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not load tokenizer for {model_id!r} at revision "
            f"{model_revision!r}. Check network access, the model ID/revision, "
            "or pre-download the model into the Hugging Face cache."
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # trust_remote_code is explicitly False (no unsafe loading flags).
    # safetensors is preferred; HF falls back to it automatically when
    # the repository does not publish .bin weights.
    kwargs: dict[str, Any] = {
        "revision": model_revision,
        "device_map": "auto" if not allow_cpu else None,
        "torch_dtype": "auto",
        "trust_remote_code": False,
    }
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("The reference QLoRA path requires CUDA.")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load model {model_id!r} at revision {model_revision!r}. "
            "Check access, disk space, and the local Hugging Face cache."
        ) from exc
    model.config.use_cache = False
    target_modules: str | list[str]
    if lora_targets == ("all-linear",):
        target_modules = "all-linear"
    else:
        target_modules = list(lora_targets)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return model, tokenizer, lora


def run_training(args: argparse.Namespace) -> Path:
    """Run one fully tracked experiment and return its output directory."""

    set_reproducible_seed(args.seed)
    run_name = args.run_name or (
        f"grpo-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{args.profile}-{args.reward_profile}-seed{args.seed}"
    )
    output_dir = args.output_root / run_name
    resume_requested = bool(
        getattr(args, "resume", False) or getattr(args, "resume_from_checkpoint", None)
    )
    if output_dir.exists() and not resume_requested:
        raise FileExistsError(
            f"Run directory already exists: {output_dir}. Supply a new --run-name "
            "or pass --resume for a compatible interrupted run."
        )
    if resume_requested and not output_dir.exists():
        raise FileNotFoundError(
            f"Cannot resume because the run directory does not exist: {output_dir}."
        )
    # Data validation must happen before any heavy dependency import so the
    # safety guard short-circuits cleanly even when torch isn't installed.
    train_dataset = _load_dataset(
        args.train,
        "training",
        args.system_prompt_condition,
        profile=args.profile,
    )
    validation_dataset = _load_dataset(
        args.validation,
        "validation",
        args.system_prompt_condition,
        profile=args.profile,
    )
    # Resolve the model only after the data-role guards have passed.
    model_profile = _resolve_model(args)
    if not args.no_4bit and not model_profile.supports_4bit:
        raise ValueError(
            f"Model {model_profile.model_id!r} is not registered for 4-bit loading. "
            "Pass --no-4bit or validate and update its registry profile."
        )
    hardware = _preflight(
        args.profile,
        args.allow_cpu,
        model_min_vram_gib=model_profile.min_vram_gib,
    )
    disk_probe = args.output_root
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    assert_disk_space(disk_probe)
    model, tokenizer, lora_config = _model_and_tokenizer(
        model_id=args.model_id,
        model_revision=model_profile.revision or args.model_revision,
        load_in_4bit=not args.no_4bit,
        allow_cpu=args.allow_cpu,
        lora_targets=model_profile.lora_targets,
    )
    import torch
    from trl import GRPOTrainer

    training_args = get_grpo_config(
        output_dir=str(output_dir),
        run_name=run_name,
        profile_name=args.profile,
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        use_cpu=not torch.cuda.is_available(),
        overrides={
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
        },
    )
    # Do not create a partial run directory when data, dependencies, model
    # access, or the TRL configuration fail preflight.
    if not resume_requested:
        output_dir.mkdir(parents=True, exist_ok=False)
    early_stopping = RewardEarlyStoppingCallback(
        patience=args.early_stopping_patience,
    )
    tracker = JsonlTrackingCallback(output_dir / "training_metrics.jsonl")
    # Shared ExperimentManifest, written atomically so a crash mid-run never
    # leaves a partially-written manifest on disk.
    train_path = Path(args.train)
    val_path = Path(args.validation)
    benchmark_path = Path(args.benchmark) if getattr(args, "benchmark", None) else Path(
        "data/benchmarks/anthropic_sycophancy.jsonl"
    )
    try:
        benchmark_hash = sha256_file(benchmark_path) if benchmark_path.exists() else "missing"
    except FileNotFoundError:
        benchmark_hash = "missing"
    manifest = new_manifest(
        run_id=run_name,
        stage=Stage.TRAINING,
        model_id=args.model_id,
        model_revision=model_profile.revision or args.model_revision or "",
        adapter_path=str(output_dir / "final_adapter"),
        training_hash=hashlib.sha256(train_path.read_bytes()).hexdigest(),
        validation_hash=hashlib.sha256(val_path.read_bytes()).hexdigest(),
        benchmark_hash=benchmark_hash,
        seed=args.seed,
        prompt_condition=args.system_prompt_condition,
        generation_settings={
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 0,
            "max_new_tokens": 128,
            "do_sample": True,
        },
        reward_profile=args.reward_profile,
        effective_training_config=training_args.to_dict(),
        hardware=hardware,
        python_versions=_package_versions(),
        git_commit=None,
        runner=getattr(args, "runner", "local"),
        data_governance={
            "is_fixture": any(is_fixture_row(row) for row in read_jsonl(train_path)),
            "benchmark_only": False,
            "profile": args.profile,
            "method": "QLoRA+GRPO" if not args.no_4bit else "LoRA+GRPO",
        },
    )
    manifest_path = output_dir / "training_manifest.json"
    if resume_requested:
        previous = load_manifest(manifest_path)
        compatible, reason = is_resume_compatible(previous, manifest)
        if not compatible:
            raise ValueError(f"Resume refused: {reason}.")
        if previous.status is Status.RUNNING:
            previous = transition(previous, to=Status.INTERRUPTED)
            write_atomically(previous, manifest_path, overwrite=True)
        manifest = previous
    else:
        write_atomically(manifest, manifest_path)

    resume_checkpoint = args.resume_from_checkpoint
    if resume_requested and not resume_checkpoint:
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1])
            if path.name.rsplit("-", 1)[-1].isdigit()
            else -1,
        )
        resume_checkpoint = str(checkpoints[-1]) if checkpoints else None

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_composite_reward_func(args.reward_profile)],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[tracker, early_stopping],
    )
    started = time.perf_counter()
    try:
        # Atomic CREATED -> RUNNING transition so a crash before training
        # never claims the run was running.
        manifest = record_running(load_manifest(manifest_path), manifest_path)
        with graceful_shutdown(
            on_signal=lambda _sig: record_interrupted(
                load_manifest(manifest_path), manifest_path
            )
        ):
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        try:
            record_interrupted(load_manifest(manifest_path), manifest_path)
        except Exception:
            pass
        raise
    except torch.cuda.OutOfMemoryError as exc:
        try:
            record_failed(load_manifest(manifest_path), manifest_path)
        except Exception:
            pass
        (output_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failure_type": "cuda_out_of_memory",
                    "global_step": trainer.state.global_step,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "CUDA out of memory. Retry with the qlora_7b_16gb profile, "
            "fewer --num-generations, or a shorter completion length."
        ) from exc
    except Exception as exc:
        try:
            record_failed(load_manifest(manifest_path), manifest_path)
        except Exception:
            pass
        (output_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                    "global_step": trainer.state.global_step,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    duration = time.perf_counter() - started
    staged_adapter = output_dir / "final_adapter.staging"
    if staged_adapter.exists():
        raise FileExistsError(
            f"Staged adapter already exists: {staged_adapter}. Remove it only after "
            "confirming no interrupted save is recoverable."
        )
    trainer.save_model(str(staged_adapter))
    tokenizer.save_pretrained(staged_adapter)
    final_adapter = promote_adapter_atomic(staged_adapter, output_dir / "final_adapter")
    summary = {
        "duration_seconds": duration,
        "global_step": trainer.state.global_step,
        "train_metrics": train_result.metrics,
        "best_validation_reward": early_stopping.best_reward,
        "best_validation_step": early_stopping.best_step,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    # Atomic RUNNING -> COMPLETED transition.  The adapter checksum is
    # recorded on the manifest as well so a follow-up reader can verify
    # the artifacts haven't drifted.
    completed = record_completed(
        load_manifest(manifest_path),
        manifest_path,
        adapter_path=final_adapter,
    )
    (output_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "global_step": trainer.state.global_step,
                "final_adapter": "final_adapter",
                "adapter_sha256": completed.adapter_sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate local hardware, split roles, and the live TRL config."""

    # Data validation runs before hardware preflight so safety guards
    # short-circuit cleanly even on machines without torch installed.
    data_state: dict[str, Any] = {}
    if (not args.train.exists()) and (not args.validation.exists()):
        if args.profile != SMOKE_PROFILE:
            raise FileNotFoundError(
                f"Training split is missing at {args.train} and validation "
                f"split is missing at {args.validation}. Run `syco setup` "
                "or `python -m sycophancy_rl.data_prep.merge_datasets` first."
            )
        data_state = {
            "data_state": "not_provisioned",
            "train_examples": 0,
            "validation_examples": 0,
            "data_note": (
                "Smoke preflight runs without a provisioned training pool; "
                "run `syco setup` to generate the fixture pool and splits."
            ),
        }
        train_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []
    else:
        train_rows = _require_training_rows(args.train, "training", profile=args.profile)
        validation_rows = _require_training_rows(
            args.validation, "validation", profile=args.profile
        )
        data_state = {
            "data_state": "provisioned",
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
        }
    # Resolve the model through the registry so the preflight reports the
    # resolved revision and surfaces registry errors before any training
    # attempt.  The registry never silently rewrites a custom model id
    # to a reference revision.
    model_profile = _resolve_model(args)
    if not args.no_4bit and not model_profile.supports_4bit:
        raise ValueError(
            f"Model {model_profile.model_id!r} is not approved for 4-bit loading. "
            "Use a registered 7B model or provide --custom-supports-4bit with "
            "--allow-unpinned-model after verifying compatibility."
        )
    hardware = _preflight(
        args.profile,
        args.allow_cpu,
        model_min_vram_gib=model_profile.min_vram_gib,
    )
    cuda_available = bool(hardware.get("cuda_available"))
    if cuda_available:
        import torch

        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else:
        bf16 = False
    try:
        config = get_grpo_config(
            output_dir=str(args.output_root / "preflight-only"),
            run_name="preflight-only",
            profile_name=args.profile,
            seed=args.seed,
            bf16=bf16,
            fp16=cuda_available and not bf16,
            use_cpu=not cuda_available,
            overrides={
                "learning_rate": args.learning_rate,
                "beta": args.beta,
                "max_steps": args.max_steps,
                "num_generations": args.num_generations,
            },
        )
    except (ImportError, RuntimeError) as exc:
        if (
            "TRL" in str(exc).upper()
            or "trl" in str(exc).lower()
            or "pinned" in str(exc).lower()
            or "torch" in str(exc).lower()
            or "GRPOConfig" in str(exc)
        ) and args.profile == SMOKE_PROFILE:
            config = None
        else:
            raise
    payload = {
        "status": "ok",
        "model_id": model_profile.model_id,
        "model_revision": model_profile.revision,
        "reward_profile": args.reward_profile,
        "profile": args.profile,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "hardware": hardware,
        "packages": _package_versions(),
        "effective_config": config.to_dict() if config is not None else None,
        "note": "Model weights are checked only when training starts.",
        "registry": {
            "architecture": model_profile.architecture.value
            if hasattr(model_profile.architecture, "value")
            else str(model_profile.architecture),
            "min_vram_gib": model_profile.min_vram_gib,
            "supports_4bit": model_profile.supports_4bit,
            "test_status": model_profile.test_status.value
            if hasattr(model_profile.test_status, "value")
            else str(model_profile.test_status),
            "context_length": model_profile.context_length,
        },
    }
    payload.update(data_state)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument(
        "--profile",
        choices=tuple(TRAINING_PROFILES),
        default="local_8gb",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument(
        "--system-prompt-condition",
        choices=("none", "neutral", "anti_sycophancy", "pro_agreement_control"),
        default="neutral",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument(
        "--reward-profile",
        choices=tuple(REWARD_PROFILES),
        default="combined",
        help="Use diagnostic_format_only only for the pre-registered ablation.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the newest compatible checkpoint in --run-name.",
    )
    parser.add_argument(
        "--runner",
        choices=("local", "kaggle"),
        default="local",
    )
    parser.add_argument(
        "--allow-unpinned-model",
        action="store_true",
        help="Acknowledge that a custom model is outside the validated registry.",
    )
    parser.add_argument(
        "--custom-supports-4bit",
        action="store_true",
        help="Declare that an unregistered custom model supports NF4 bitsandbytes loading.",
    )
    parser.add_argument(
        "--custom-lora-targets",
        default="all-linear",
        help="Comma-separated PEFT target modules for an unregistered custom model.",
    )
    parser.add_argument("--custom-context-length", type=int, default=2048)
    parser.add_argument("--custom-min-vram-gib", type=float, default=0.0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate hardware, data roles, dependencies, and TRL config without loading weights.",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the trainer CLI arguments.

    Tests and the ``syco`` CLI pass an explicit ``argv`` so they can
    drive the trainer without touching ``sys.argv``.  The ``__main__``
    entry point in this module calls this with no arguments so it reads
    from ``sys.argv`` as a normal CLI would.
    """

    parser = _build_parser()
    return parser.parse_args(argv) if argv is not None else parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.preflight_only:
        print(json.dumps(run_preflight(args), indent=2, sort_keys=True, default=str))
        return
    output = run_training(args)
    print(f"Training completed; artifacts saved under {output}.")


__all__ = [
    "DEFAULT_TRAIN",
    "DEFAULT_VALIDATION",
    "SMOKE_PROFILE",
    "REAL_TRAINING_PROFILES",
    "run_training",
    "run_preflight",
    "main",
    "_build_parser",
    "_parse_args",
    "_resolve_model",
    "_load_dataset",
    "_require_training_rows",
    "_preflight",
    "_model_and_tokenizer",
]


if __name__ == "__main__":
    main()
