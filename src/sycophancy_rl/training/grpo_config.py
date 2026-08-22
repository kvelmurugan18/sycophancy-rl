"""Version-current GRPO profiles for reproducible local QLoRA experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingProfile:
    """Hardware-aware settings whose trade-offs are explicit."""

    name: str
    learning_rate: float
    beta: float
    max_steps: int
    num_generations: int
    gradient_accumulation_steps: int
    max_completion_length: int
    logging_steps: int
    eval_steps: int
    save_steps: int


PROFILES: dict[str, TrainingProfile] = {
    "smoke": TrainingProfile(
        name="smoke",
        learning_rate=5e-6,
        beta=0.05,
        max_steps=5,
        num_generations=2,
        gradient_accumulation_steps=2,
        max_completion_length=64,
        logging_steps=1,
        eval_steps=5,
        save_steps=5,
    ),
    "local_8gb": TrainingProfile(
        name="local_8gb",
        learning_rate=5e-6,
        beta=0.05,
        max_steps=200,
        num_generations=2,
        gradient_accumulation_steps=4,
        max_completion_length=96,
        logging_steps=5,
        eval_steps=25,
        save_steps=25,
    ),
    "local_16gb": TrainingProfile(
        name="local_16gb",
        learning_rate=5e-6,
        beta=0.05,
        max_steps=400,
        num_generations=4,
        gradient_accumulation_steps=4,
        max_completion_length=128,
        logging_steps=5,
        eval_steps=25,
        save_steps=25,
    ),
    "qlora_7b_16gb": TrainingProfile(
        name="qlora_7b_16gb",
        learning_rate=2e-6,
        beta=0.04,
        max_steps=150,
        num_generations=4,
        gradient_accumulation_steps=4,
        max_completion_length=64,
        logging_steps=5,
        eval_steps=25,
        save_steps=25,
    ),
    "kaggle_online_smoke": TrainingProfile(
        name="kaggle_online_smoke",
        learning_rate=5e-6,
        beta=0.05,
        max_steps=5,
        num_generations=2,
        gradient_accumulation_steps=2,
        max_completion_length=64,
        logging_steps=1,
        eval_steps=5,
        save_steps=5,
    ),
    "qwen25_7b_online": TrainingProfile(
        name="qwen25_7b_online",
        learning_rate=2e-6,
        beta=0.04,
        max_steps=150,
        num_generations=4,
        gradient_accumulation_steps=4,
        max_completion_length=64,
        logging_steps=5,
        eval_steps=25,
        save_steps=25,
    ),
}


def get_profile(name: str) -> TrainingProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown profile {name!r}; choose from {sorted(PROFILES)}.") from exc


def get_grpo_config(
    *,
    output_dir: str,
    run_name: str,
    profile_name: str,
    seed: int,
    bf16: bool,
    fp16: bool = False,
    use_cpu: bool = False,
    overrides: dict[str, Any] | None = None,
):
    """Return a current :class:`trl.GRPOConfig` for one tracked experiment."""

    try:
        from trl import GRPOConfig
    except ImportError as exc:
        raise RuntimeError("Install the pinned TRL dependencies before training.") from exc

    profile = get_profile(profile_name)
    values: dict[str, Any] = {
        "output_dir": output_dir,
        "run_name": run_name,
        "learning_rate": profile.learning_rate,
        "beta": profile.beta,
        "max_steps": profile.max_steps,
        "max_completion_length": profile.max_completion_length,
        "num_generations": profile.num_generations,
        # Evaluation reports absolute reward and does not estimate a policy
        # gradient, so one completion per validation prompt is sufficient and
        # keeps the global eval batch divisible on a single local device.
        "num_generations_eval": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": bf16,
        "fp16": fp16,
        "use_cpu": use_cpu,
        "logging_steps": profile.logging_steps,
        "logging_first_step": True,
        "eval_strategy": "steps",
        "eval_steps": profile.eval_steps,
        "eval_on_start": True,
        "save_strategy": "steps",
        "save_steps": profile.save_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_reward",
        "greater_is_better": True,
        "seed": seed,
        "data_seed": seed,
        "full_determinism": True,
        "remove_unused_columns": False,
        "report_to": ["tensorboard"],
        "log_completions": True,
        "num_completions_to_print": 2,
        "mask_truncated_completions": True,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 0,
        "repetition_penalty": 1.05,
    }
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})

    effective_batch = (
        values["per_device_train_batch_size"] * values["gradient_accumulation_steps"]
    )
    if effective_batch % values["num_generations"] != 0:
        raise ValueError(
            "per_device_train_batch_size × gradient_accumulation_steps must be "
            "divisible by num_generations."
        )
    return GRPOConfig(**values)


def profile_to_dict(name: str) -> dict[str, Any]:
    return asdict(get_profile(name))
