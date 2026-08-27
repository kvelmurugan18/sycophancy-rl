"""Leakage-safe, registry-aware local QLoRA + GRPO training entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
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
from sycophancy_rl.training.online_rollout import (
    make_online_rollout_func,
    trajectory_reward_func,
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

DEFAULT_TRAIN = Path("data/anthropic_experiment/train.jsonl")
DEFAULT_VALIDATION = Path("data/anthropic_experiment/validation.jsonl")


SMOKE_PROFILE = "smoke"
REAL_TRAINING_PROFILES = frozenset(set(TRAINING_PROFILES) - {SMOKE_PROFILE})


def _is_benchmark_row(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata", {})
    return bool(row.get("data_role") == "benchmark" or metadata.get("benchmark_only"))


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
    rollout_mode: str = "prepared",
):
    rows = _require_training_rows(path, role, profile=profile)
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install the pinned datasets dependency before training.") from exc
    prepared: list[dict[str, Any]] = []
    source_rows = _expand_multi_turn_rows(rows) if rollout_mode == "prepared" else rows
    for row in source_rows:
        copy = dict(row)
        copy["prompt"] = apply_system_prompt(
            row["prompt"],
            system_prompt_condition,
        )
        # TRL passes only the structured prompt to a custom rollout function.
        # Reserved non-content keys preserve governed row identity without
        # changing the text rendered by chat templates.
        copy["prompt"][-1]["_syco_example_id"] = str(row["example_id"])
        copy["prompt"][-1]["_syco_data_role"] = role
        prepared.append(copy)
    return Dataset.from_list(prepared)


def _teacher_answer(row: dict[str, Any], option: str | None = None) -> str:
    """Return a short, auditable teacher turn used only as prior chat history."""

    label = option or str(row["target_option"])
    option_text = str(row.get("options", {}).get(label, "the selected option"))
    return f"Answer: ({label})\nReason: The available evidence supports {option_text}."


def _expand_multi_turn_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create teacher-forced chat prefixes for every governed pushback turn.

    TRL's GRPO trainer generates one assistant completion per dataset prompt.
    To train on repeated pressure without pretending that GRPOTrainer is an
    interactive simulator, each episode is expanded into its initial prompt and
    one prompt per later user turn. Prior assistant messages are explicit gold
    scaffolds; evaluation remains free-running and uses the model's own history.
    """

    expanded: list[dict[str, Any]] = []
    for row in rows:
        initial = dict(row)
        initial["training_turn"] = 0
        initial["source_episode_id"] = str(row["example_id"])
        expanded.append(initial)

        history = [dict(message) for message in row["prompt"]]
        for turn_number, pushback in enumerate(row.get("pushback_turns", []), start=1):
            teacher = _teacher_answer(row, pushback.get("prior_assistant_option"))
            history.append({"role": "assistant", "content": teacher})
            history.append({"role": "user", "content": str(pushback["text"])})
            turn = dict(row)
            turn["example_id"] = f"{row['example_id']}::training-turn-{turn_number}"
            turn["source_episode_id"] = str(row["example_id"])
            turn["training_turn"] = turn_number
            turn["prompt"] = [dict(message) for message in history]
            for key in (
                "target_option",
                "independent_option",
                "sycophantic_option",
                "user_preferred_option",
                "user_claim_valid",
                "behavior_target",
                "evidence_strength",
            ):
                if key in pushback:
                    turn[key] = pushback[key]
            turn["pushback_turns"] = []
            expanded.append(turn)
    return expanded


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _example_id_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(row["example_id"]) for row in rows)).encode("utf-8")
    ).hexdigest()


def _validate_split_identity(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate governed split roles, counts, and pairwise ID disjointness."""

    train_rows = _require_training_rows(args.train, "training", profile=args.profile)
    validation_rows = _require_training_rows(
        args.validation, "validation", profile=args.profile
    )
    benchmark_path = (
        Path(args.benchmark)
        if getattr(args, "benchmark", None)
        else Path("data/anthropic_experiment/benchmark.jsonl")
    )
    benchmark_rows = (
        read_jsonl(benchmark_path, expected_role="benchmark")
        if benchmark_path.exists()
        else []
    )
    ids = {
        "training": {str(row["example_id"]) for row in train_rows},
        "validation": {str(row["example_id"]) for row in validation_rows},
        "benchmark": {str(row["example_id"]) for row in benchmark_rows},
    }
    overlaps = {
        "training/validation": ids["training"] & ids["validation"],
        "training/benchmark": ids["training"] & ids["benchmark"],
        "validation/benchmark": ids["validation"] & ids["benchmark"],
    }
    contaminated = {name: sorted(values)[:5] for name, values in overlaps.items() if values}
    if contaminated:
        raise ValueError(f"Example ID overlap across governed splits: {contaminated}")
    if args.profile in {"kaggle_online_smoke", "qwen25_05b_online"}:
        counts = (len(train_rows), len(validation_rows), len(benchmark_rows))
        if counts != (24_134, 3_017, 3_017):
            raise ValueError(
                "The Qwen2.5-0.5B Anthropic experiment requires exact "
                f"24,134/3,017/3,017 train/validation/benchmark counts; got {counts}."
            )
    return train_rows, validation_rows, benchmark_rows


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
    expected_model_class: str = "",
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
    architectures = tuple(getattr(model.config, "architectures", ()) or ())
    if expected_model_class and expected_model_class not in architectures:
        raise RuntimeError(
            f"Model architecture mismatch for {model_id}: expected "
            f"{expected_model_class}, reported {architectures}."
        )
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
    rollout_mode = getattr(args, "rollout_mode", "online")
    if args.profile in REAL_TRAINING_PROFILES and rollout_mode != "online":
        raise ValueError(
            f"{args.profile} requires --rollout-mode online; prepared teacher-forced "
            "rollouts are a smoke-only diagnostic."
        )
    max_pushback_turns = int(getattr(args, "max_pushback_turns", 1))
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
    # All three governed splits are validated before any heavy dependency import.
    train_identity_rows, validation_identity_rows, benchmark_identity_rows = (
        _validate_split_identity(args)
    )
    train_dataset = _load_dataset(
        args.train,
        "training",
        args.system_prompt_condition,
        profile=args.profile,
        rollout_mode=rollout_mode,
    )
    validation_dataset = _load_dataset(
        args.validation,
        "validation",
        args.system_prompt_condition,
        profile=args.profile,
        rollout_mode=rollout_mode,
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
        expected_model_class=model_profile.expected_model_class,
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
        "data/anthropic_experiment/benchmark.jsonl"
    )
    try:
        benchmark_hash = sha256_file(benchmark_path) if benchmark_path.exists() else "missing"
    except FileNotFoundError:
        benchmark_hash = "missing"
    effective_training_config = training_args.to_dict()
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
            "temperature": effective_training_config["temperature"],
            "top_p": effective_training_config["top_p"],
            "top_k": effective_training_config["top_k"],
            "max_new_tokens": effective_training_config["max_completion_length"],
            "repetition_penalty": effective_training_config["repetition_penalty"],
            "do_sample": True,
        },
        reward_profile=args.reward_profile,
        effective_training_config=effective_training_config,
        hardware=hardware,
        python_versions=_package_versions(),
        git_commit=None,
        runner=getattr(args, "runner", "local"),
        data_governance={
            "is_fixture": any(is_fixture_row(row) for row in read_jsonl(train_path)),
            "benchmark_only": False,
            "profile": args.profile,
            "method": "QLoRA+GRPO" if not args.no_4bit else "LoRA+GRPO",
            "rollout_mode": rollout_mode,
            "counts": {
                "training": len(train_identity_rows),
                "validation": len(validation_identity_rows),
                "benchmark": len(benchmark_identity_rows),
            },
            "example_id_sha256": {
                "training": _example_id_sha256(train_identity_rows),
                "validation": _example_id_sha256(validation_identity_rows),
                "benchmark": (
                    _example_id_sha256(benchmark_identity_rows)
                    if benchmark_identity_rows
                    else "missing"
                ),
            },
            "zero_example_id_overlap": True,
            "anthropic_training_opt_in": all(
                row.get("metadata", {}).get("anthropic_training_opt_in") is True
                for row in (*train_identity_rows, *validation_identity_rows)
                if str(row.get("source", "")).casefold()
                == "anthropic/model-written-evals"
            ),
            "held_out_benchmark_used_for_training": False,
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

    trainer_kwargs: dict[str, Any] = {}
    reward_funcs = [make_composite_reward_func(args.reward_profile)]
    if rollout_mode == "online":
        raw_training_rows = train_identity_rows
        raw_validation_rows = validation_identity_rows
        trainer_kwargs["rollout_func"] = make_online_rollout_func(
            raw_training_rows + raw_validation_rows,
            artifact_path=output_dir / "training_trajectories.jsonl",
            seed=args.seed,
            max_pushback_turns=max_pushback_turns,
        )
        reward_funcs = [trajectory_reward_func]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[tracker, early_stopping],
        **trainer_kwargs,
    )
    trainable_parameters = [
        (name, parameter)
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable LoRA parameters were found after trainer setup.")
    trainable_parameter_count = sum(parameter.numel() for _, parameter in trainable_parameters)
    tracked_parameter_name, tracked_parameter = trainable_parameters[0]
    tracked_parameter_before = tracked_parameter.detach().float().cpu().clone()
    global_step_before = int(trainer.state.global_step)
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
    global_step_after = int(trainer.state.global_step)
    tracked_parameter_after = tracked_parameter.detach().float().cpu()
    tracked_parameter_max_abs_delta = float(
        (tracked_parameter_after - tracked_parameter_before).abs().max().item()
    )
    optimizer_update_verified = bool(
        global_step_after > global_step_before and tracked_parameter_max_abs_delta > 0.0
    )
    if args.profile == "kaggle_online_smoke" and not optimizer_update_verified:
        raise RuntimeError(
            "Kaggle smoke did not verify a real LoRA optimizer update: "
            f"global_step {global_step_before}->{global_step_after}, "
            f"tracked delta={tracked_parameter_max_abs_delta}."
        )
    staged_adapter = output_dir / "final_adapter.staging"
    if staged_adapter.exists():
        raise FileExistsError(
            f"Staged adapter already exists: {staged_adapter}. Remove it only after "
            "confirming no interrupted save is recoverable."
        )
    trainer.save_model(str(staged_adapter))
    tokenizer.save_pretrained(staged_adapter)
    final_adapter = promote_adapter_atomic(staged_adapter, output_dir / "final_adapter")
    adapter_reload: dict[str, Any] = {"attempted": False, "verified": False}
    if args.profile == "kaggle_online_smoke":
        adapter_reload["attempted"] = True
        reloaded = trainer.accelerator.unwrap_model(trainer.model)
        if not hasattr(reloaded, "load_adapter") or not hasattr(reloaded, "set_adapter"):
            raise RuntimeError("Saved model does not expose PEFT adapter reload methods.")
        adapter_name = "saved_adapter_reload_verification"
        reloaded.load_adapter(str(final_adapter), adapter_name=adapter_name, is_trainable=False)
        reloaded.set_adapter(adapter_name)
        reload_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply using exactly: Answer: A\nJustification: smoke check."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        reload_inputs = tokenizer(reload_prompt, return_tensors="pt", add_special_tokens=False)
        reload_inputs = {key: value.to(reloaded.device) for key, value in reload_inputs.items()}
        reloaded.eval()
        with torch.no_grad():
            reload_output = reloaded.generate(
                **reload_inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        reload_tokens = reload_output[0, reload_inputs["input_ids"].shape[1] :]
        reload_response = tokenizer.decode(reload_tokens, skip_special_tokens=True)
        if not reload_response.strip():
            raise RuntimeError("Reloaded adapter produced an empty inference response.")
        adapter_reload.update({"verified": True, "response": reload_response})
    summary = {
        "duration_seconds": duration,
        "global_step": trainer.state.global_step,
        "global_step_before": global_step_before,
        "global_step_after": global_step_after,
        "trainable_parameter_count": trainable_parameter_count,
        "tracked_lora_parameter": tracked_parameter_name,
        "tracked_lora_parameter_max_abs_delta": tracked_parameter_max_abs_delta,
        "optimizer_update_verified": optimizer_update_verified,
        "adapter_reload": adapter_reload,
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
        train_rows, validation_rows, benchmark_rows = _validate_split_identity(args)
        data_state = {
            "data_state": "provisioned",
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "benchmark_examples": len(benchmark_rows),
            "zero_example_id_overlap": True,
            "anthropic_training_opt_in_verified": True,
            "held_out_benchmark_protected": bool(benchmark_rows),
        }
    if args.profile != SMOKE_PROFILE:
        import_errors: dict[str, str] = {}
        for package in (
            "torch",
            "transformers",
            "trl",
            "peft",
            "datasets",
            "accelerate",
            "bitsandbytes",
        ):
            try:
                importlib.import_module(package)
            except Exception as exc:  # pragma: no cover - depends on host wheels
                import_errors[package] = f"{type(exc).__name__}: {exc}"
        if import_errors:
            raise RuntimeError(f"Required training package imports failed: {import_errors}")
    output_probe = args.output_root.resolve()
    while not output_probe.exists() and output_probe != output_probe.parent:
        output_probe = output_probe.parent
    if not output_probe.is_dir() or not os.access(output_probe, os.W_OK):
        raise PermissionError(f"Training output parent is not writable: {output_probe}")
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
        "--rollout-mode", choices=("online", "prepared"), default="online",
        help="online uses actual policy responses; prepared uses teacher-forced expanded rows.",
    )
    parser.add_argument("--max-pushback-turns", type=int, default=1)
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
