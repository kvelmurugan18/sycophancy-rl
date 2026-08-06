"""Generate and save reproducible raw responses for one model/condition.

Examples:

    python -m sycophancy_rl.evaluation.run_benchmark --model base
    python -m sycophancy_rl.evaluation.run_benchmark --model trained \
        --adapter outputs/checkpoints/exp01-seed42

Both invocations use the same benchmark, prompt condition, variants, and
generation settings unless the caller explicitly changes them.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sycophancy_rl.data_prep.schema import read_jsonl
from sycophancy_rl.evaluation.io import read_records, write_json, write_records
from sycophancy_rl.evaluation.metrics import simple_choice_baselines, summarize_records
from sycophancy_rl.evaluation.prompts import apply_system_prompt, make_prompt_variant
from sycophancy_rl.reward.reward_fn import score_completion
from sycophancy_rl.utils.answer_parser import classify_answer, parse_final_answer

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DEFAULT_MODEL_REVISION = "31b70e2e869a7173562077fd711b654946d38674"
DEFAULT_BENCHMARK = Path("data/benchmarks/anthropic_sycophancy.jsonl")


@dataclass(frozen=True)
class GenerationSettings:
    """Fully specified generation settings shared by every compared model."""

    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 128
    repetition_penalty: float = 1.0


@dataclass(frozen=True)
class BenchmarkConfig:
    """Serializable inputs for one resumable benchmark shard."""

    run_name: str
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = DEFAULT_MODEL_REVISION
    adapter_path: Path | None = None
    benchmark_path: Path = DEFAULT_BENCHMARK
    output_dir: Path = Path("outputs/evaluations")
    system_prompt_condition: str = "neutral"
    prompt_variants: tuple[str, ...] = ("original", "swap_options")
    seed: int = 42
    max_examples: int | None = None
    settings: GenerationSettings = GenerationSettings()
    load_in_4bit: bool = False
    batch_size: int = 4
    num_shards: int = 1
    shard_index: int = 0
    resume: bool = False


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Warn-only preserves portability when a deterministic CUDA kernel is
        # unavailable, while still recording the setting in the manifest.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hardware_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        result["cuda_available"] = False
    return result


def load_model_and_tokenizer(
    *,
    model_id: str,
    revision: str | None,
    adapter_path: Path | None,
    load_in_4bit: bool,
):
    """Load a base model or PEFT adapter with optional 4-bit quantization."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation requires torch and transformers. Install the pinned ML dependencies."
        ) from exc

    tokenizer_source = str(adapter_path) if adapter_path and (adapter_path / "tokenizer_config.json").exists() else model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if tokenizer_source != model_id else revision,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "device_map": "auto",
        "torch_dtype": "auto",
        "trust_remote_code": False,
    }
    if load_in_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading an adapter requires peft.") from exc
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def _generate_batch(
    model: Any,
    tokenizer: Any,
    conversations: list[list[dict[str, str]]],
    settings: GenerationSettings,
) -> list[tuple[str, str, int, int, float]]:
    """Generate one completion per conversation in a padded batch."""

    import torch

    rendered = [
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        for messages in conversations
    ]
    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            rendered,
            padding=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        padded_input_length = int(inputs["input_ids"].shape[-1])
        input_lengths = [
            int(value) for value in inputs["attention_mask"].sum(dim=1).tolist()
        ]
        generation_kwargs = asdict(settings)
        if not settings.do_sample:
            generation_kwargs.pop("temperature")
            generation_kwargs.pop("top_p")
            generation_kwargs.pop("top_k")
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
        latency = time.perf_counter() - started
    finally:
        tokenizer.padding_side = previous_padding_side
    per_item_latency = latency / max(len(conversations), 1)
    results: list[tuple[str, str, int, int, float]] = []
    for index, output in enumerate(outputs):
        completion_ids = output[padded_input_length:]
        output_length = int(completion_ids.shape[-1])
        response = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        finish_reason = "length" if output_length >= settings.max_new_tokens else "eos"
        results.append(
            (
                response,
                finish_reason,
                input_lengths[index],
                output_length,
                per_item_latency,
            )
        )
    return results


def _generate_one(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    settings: GenerationSettings,
) -> tuple[str, str, int, int, float]:
    """Compatibility wrapper used by narrow unit tests and callers."""

    return _generate_batch(model, tokenizer, [messages], settings)[0]


def evaluate_examples(
    examples: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    model_id: str,
    model_revision: str | None,
    adapter_path: Path | None,
    system_prompt_condition: str,
    prompt_variants: list[str],
    settings: GenerationSettings,
    seed: int,
    batch_size: int = 4,
    existing_records: list[dict[str, Any]] | None = None,
    record_sink: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Generate turn-major batches and resume from existing comparison IDs."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    records: list[dict[str, Any]] = list(existing_records or [])
    existing = {str(row["example_id"]): row for row in records}
    states: list[dict[str, Any]] = []
    for source_example in examples:
        for variant_name in prompt_variants:
            example = make_prompt_variant(source_example, variant_name)
            messages = apply_system_prompt(example["prompt"], system_prompt_condition)
            turns = [None, *example.get("pushback_turns", [])]
            states.append(
                {
                    "example": example,
                    "variant_name": variant_name,
                    "messages": messages,
                    "turns": turns,
                }
            )

    max_turns = max((len(state["turns"]) for state in states), default=0)
    for turn_number in range(max_turns):
        active = [state for state in states if turn_number < len(state["turns"])]
        pending: list[dict[str, Any]] = []
        for state in active:
            example = state["example"]
            variant_name = state["variant_name"]
            messages = state["messages"]
            pushback = state["turns"][turn_number]
            if pushback is not None:
                messages.append({"role": "user", "content": str(pushback["text"])})
            comparison_id = f"{example['example_id']}::{variant_name}::turn-{turn_number}"
            if comparison_id in existing:
                messages.append(
                    {
                        "role": "assistant",
                        "content": str(existing[comparison_id]["generated_response"]),
                    }
                )
            else:
                pending.append(state)

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            generated = _generate_batch(
                model,
                tokenizer,
                [state["messages"] for state in batch],
                settings,
            )
            new_records: list[dict[str, Any]] = []
            for state, generated_item in zip(batch, generated, strict=True):
                example = state["example"]
                variant_name = state["variant_name"]
                messages = state["messages"]
                pushback = state["turns"][turn_number]
                response, finish_reason, input_tokens, output_tokens, latency = generated_item
                parsed = parse_final_answer(response, finish_reason=finish_reason)
                context = pushback or {}
                target = context.get("target_option", example["target_option"])
                independent = context.get(
                    "independent_option",
                    example["independent_option"],
                )
                sycophantic = context.get(
                    "sycophantic_option",
                    example.get("sycophantic_option"),
                )
                breakdown = score_completion(
                    response,
                    target_option=target,
                    independent_option=independent,
                    sycophantic_option=sycophantic,
                    prompt=messages,
                    options=example.get("options"),
                    finish_reason=finish_reason,
                )
                category = classify_answer(
                    parsed,
                    independent_option=independent,
                    sycophantic_option=sycophantic,
                )
                comparison_id = (
                    f"{example['example_id']}::{variant_name}::turn-{turn_number}"
                )
                prompt_snapshot = [dict(message) for message in messages]
                prompt_payload = json.dumps(
                    prompt_snapshot, ensure_ascii=False, sort_keys=True
                )
                record = {
                    "example_id": comparison_id,
                    "source_example_id": example["example_id"],
                    "episode_id": f"{example['example_id']}::{variant_name}",
                    "turn_number": turn_number,
                    "source": example["source"],
                    "source_revision": example.get("source_revision"),
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "adapter_path": str(adapter_path) if adapter_path else None,
                    "seed": seed,
                    "system_prompt_condition": system_prompt_condition,
                    "prompt_variant": variant_name,
                    "prompt": prompt_snapshot,
                    "prompt_sha256": hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest(),
                    "generation_settings": asdict(settings),
                    "generated_response": response,
                    "finish_reason": finish_reason,
                    "parsed_label": parsed.label,
                    "parse_status": parsed.reason,
                    "category": category,
                    "target_option": target,
                    "independent_option": independent,
                    "sycophantic_option": sycophantic,
                    "user_preferred_option": example.get("user_preferred_option"),
                    "user_claim_valid": context.get(
                        "user_claim_valid",
                        example.get("user_claim_valid"),
                    ),
                    "target_selected": parsed.valid and parsed.label == target,
                    "format_compliant": parsed.format_compliant,
                    "contradictory": parsed.contradictory,
                    "truncated": parsed.truncated,
                    "reward": breakdown.total,
                    "reward_breakdown": breakdown.to_dict(),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_seconds": latency,
                    "question_type": example.get("question_type"),
                    "topic": example.get("topic"),
                    "behavior_target": example.get("behavior_target"),
                    "answer_position": example.get("metadata", {}).get(
                        "answer_position",
                        example.get("independent_option"),
                    ),
                }
                records.append(record)
                existing[comparison_id] = record
                new_records.append(record)
                messages.append({"role": "assistant", "content": response})
            if record_sink is not None and new_records:
                record_sink(new_records)
    return sorted(records, key=lambda row: str(row["example_id"]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--system-prompt-condition",
        choices=("none", "neutral", "anti_sycophancy", "pro_agreement_control"),
        default="neutral",
    )
    parser.add_argument(
        "--prompt-variants",
        default="original,swap_options",
        help="Comma-separated robustness variants.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluations"))
    return parser.parse_args()


def run_benchmark_job(config: BenchmarkConfig) -> dict[str, Any]:
    """Execute one benchmark shard and persist restartable partial records."""

    if not config.run_name or Path(config.run_name).name != config.run_name:
        raise ValueError("run_name must be a non-empty path-safe name")
    if config.num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= config.shard_index < config.num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    set_reproducible_seed(config.seed)
    examples = read_jsonl(config.benchmark_path, expected_role="benchmark")
    if config.max_examples is not None:
        examples = examples[: config.max_examples]
    examples = [
        row for index, row in enumerate(examples) if index % config.num_shards == config.shard_index
    ]
    if not examples:
        raise ValueError("The selected benchmark shard contains no examples")

    run_dir = config.output_dir / config.run_name
    final_records_path = run_dir / "responses.jsonl"
    partial_records_path = run_dir / "responses.partial.jsonl"
    if final_records_path.exists() and not config.resume:
        raise FileExistsError(
            f"Benchmark output already exists: {final_records_path}. Pass --resume to reuse it."
        )
    if final_records_path.exists() and config.resume:
        existing_records = read_records(final_records_path)
    elif partial_records_path.exists() and config.resume:
        existing_records = read_records(partial_records_path)
    else:
        existing_records = []

    allowed_source_ids = {str(row["example_id"]) for row in examples}
    expected_adapter = str(config.adapter_path) if config.adapter_path else None
    expected_settings = asdict(config.settings)
    for record in existing_records:
        checks = {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "adapter_path": expected_adapter,
            "seed": config.seed,
            "system_prompt_condition": config.system_prompt_condition,
            "generation_settings": expected_settings,
        }
        if any(record.get(key) != value for key, value in checks.items()):
            raise ValueError(
                "Benchmark resume refused because partial-record provenance changed."
            )
        if str(record.get("source_example_id")) not in allowed_source_ids:
            raise ValueError("Benchmark resume contains an example outside the selected shard.")
        if record.get("prompt_variant") not in config.prompt_variants:
            raise ValueError("Benchmark resume contains an unexpected prompt variant.")

    model, tokenizer = load_model_and_tokenizer(
        model_id=config.model_id,
        revision=config.model_revision,
        adapter_path=config.adapter_path,
        load_in_4bit=config.load_in_4bit,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    def _append_partial(new_records: list[dict[str, Any]]) -> None:
        with partial_records_path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in new_records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    if not config.resume:
        partial_records_path.unlink(missing_ok=True)
    try:
        records = evaluate_examples(
            examples,
            model=model,
            tokenizer=tokenizer,
            model_id=config.model_id,
            model_revision=config.model_revision,
            adapter_path=config.adapter_path,
            system_prompt_condition=config.system_prompt_condition,
            prompt_variants=list(config.prompt_variants),
            settings=config.settings,
            seed=config.seed,
            batch_size=config.batch_size,
            existing_records=existing_records,
            record_sink=_append_partial,
        )
    finally:
        # The canonical Kaggle workflow benchmarks, trains, and benchmarks
        # again in one process. Release the first model before constructing
        # GRPOTrainer so a 7B QLoRA run does not inherit stale CUDA allocations.
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            pass
    write_records(final_records_path, records)
    partial_records_path.unlink(missing_ok=True)
    summary = summarize_records(records)
    summary["choice_baselines"] = simple_choice_baselines(examples, seed=config.seed)
    write_json(run_dir / "summary.json", summary)
    manifest = {
        "run_name": config.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_path": config.benchmark_path.as_posix(),
        "benchmark_sha256": hashlib.sha256(config.benchmark_path.read_bytes()).hexdigest(),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "adapter_path": str(config.adapter_path) if config.adapter_path else None,
        "seed": config.seed,
        "system_prompt_condition": config.system_prompt_condition,
        "prompt_variants": list(config.prompt_variants),
        "generation_settings": asdict(config.settings),
        "batch_size": config.batch_size,
        "shard_index": config.shard_index,
        "num_shards": config.num_shards,
        "record_count": len(records),
        "repository_commit": _git_commit(),
        "packages": _package_versions(),
        "hardware": _hardware_info(),
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Saved {len(records)} raw responses and computed metrics under {run_dir}.")
    return {
        "run_dir": run_dir,
        "records_path": final_records_path,
        "summary_path": run_dir / "summary.json",
        "manifest_path": run_dir / "manifest.json",
        "record_count": len(records),
    }


def main() -> None:
    args = _parse_args()
    settings = GenerationSettings(
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    run_benchmark_job(
        BenchmarkConfig(
            run_name=args.run_name,
            model_id=args.model_id,
            model_revision=args.model_revision,
            adapter_path=args.adapter,
            benchmark_path=args.benchmark,
            output_dir=args.output_dir,
            system_prompt_condition=args.system_prompt_condition,
            prompt_variants=tuple(
                value.strip()
                for value in args.prompt_variants.split(",")
                if value.strip()
            ),
            seed=args.seed,
            max_examples=args.max_examples,
            settings=settings,
            load_in_4bit=args.load_in_4bit,
            batch_size=args.batch_size,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
