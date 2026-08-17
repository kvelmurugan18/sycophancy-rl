"""Run a reproducible lm-evaluation-harness capability benchmark.

The dependency is optional because unit tests and pipeline planning must not
download models.  Execute-mode capability stages fail clearly when the
``capability`` extra has not been installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sycophancy_rl.evaluation.io import write_json


def run_capability_benchmark(
    *,
    model_id: str,
    model_revision: str,
    tasks: tuple[str, ...],
    output_path: Path,
    adapter_path: Path | None = None,
    batch_size: int = 1,
    limit: int | None = None,
    load_in_4bit: bool = True,
    seed: int = 42,
) -> Path:
    """Evaluate identical harness tasks for a base model or PEFT adapter."""

    if not tasks:
        raise ValueError("At least one capability task is required.")
    if batch_size < 1:
        raise ValueError("Capability batch_size must be at least 1.")
    if limit is not None and limit < 1:
        raise ValueError("Capability limit must be at least 1 when supplied.")
    try:
        import lm_eval
    except ImportError as exc:
        raise RuntimeError(
            "Capability evaluation requires the optional dependency. Install "
            "with `pip install -e .[capability]` before executing this plan."
        ) from exc

    model_args: dict[str, Any] = {
        "pretrained": model_id,
        "revision": model_revision,
        "trust_remote_code": False,
        "dtype": "auto",
    }
    if load_in_4bit:
        model_args["load_in_4bit"] = True
    if adapter_path is not None:
        if not adapter_path.exists():
            raise FileNotFoundError(f"Capability adapter is missing: {adapter_path}")
        model_args["peft"] = str(adapter_path)

    result = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=list(tasks),
        num_fewshot=0,
        batch_size=batch_size,
        device="cuda:0",
        limit=limit,
        log_samples=False,
        apply_chat_template=True,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )
    if not isinstance(result, dict) or not isinstance(result.get("results"), dict):
        raise RuntimeError("lm-evaluation-harness returned no result mapping.")
    payload = {
        "schema_version": 1,
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "tasks": list(tasks),
        "batch_size": batch_size,
        "limit": limit,
        "seed": seed,
        "results": result["results"],
        "configs": result.get("configs", {}),
        "versions": result.get("versions", {}),
    }
    # Harness payloads may contain NumPy scalars. Round-trip through a strict
    # JSON representation so the retained artifact is portable and auditable.
    serializable = json.loads(json.dumps(payload, default=_json_default))
    write_json(output_path, serializable)
    return output_path


def _json_default(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


__all__ = ["run_capability_benchmark"]
