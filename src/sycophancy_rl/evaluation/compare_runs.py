"""Create a paired, prompt-controlled before/after comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sycophancy_rl.evaluation.io import read_records, write_json
from sycophancy_rl.evaluation.metrics import exact_mcnemar, summarize_records


def _prompt_map(records: list[dict]) -> dict[str, str]:
    by_turn = {
        (row.get("episode_id"), row.get("turn_number")): row
        for row in records
        if row.get("episode_id") is not None
    }
    mapped: dict[str, str] = {}
    for row in records:
        example_id = str(row["example_id"])
        if example_id in mapped:
            raise ValueError("Comparison inputs contain duplicate example IDs.")
        mapped[example_id] = str(row["prompt_sha256"])
        turn = row.get("turn_number")
        prompt = row.get("prompt")
        if not isinstance(prompt, list):
            continue
        payload = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != mapped[example_id]:
            raise ValueError(f"Comparison input has an invalid prompt hash for {example_id}.")
        if not isinstance(turn, int) or turn < 1:
            continue
        previous = by_turn.get((row.get("episode_id"), turn - 1))
        if previous is None or not isinstance(previous.get("prompt"), list):
            continue  # Partial records still require identical raw prompt hashes.
        # Each earlier scripted turn is paired on its own. The current prompt
        # must continue that exact history with the model's recorded answer
        # and one new scripted user turn.
        history = [
            *previous["prompt"],
            {"role": "assistant", "content": previous["generated_response"]},
        ]
        if (
            len(prompt) != len(history) + 1
            or prompt[:-1] != history
            or not isinstance(prompt[-1], dict)
            or prompt[-1].get("role") != "user"
        ):
            raise ValueError(f"Comparison input has inconsistent episode history for {example_id}.")
        mapped[example_id] = json.dumps(prompt[-1], ensure_ascii=False, sort_keys=True)
    return mapped


def _provenance_map(records: list[dict]) -> dict[str, tuple]:
    fields = (
        "source_example_id",
        "episode_id",
        "source",
        "source_revision",
        "base_question_id",
        "benchmark_sha256",
        "seed",
        "system_prompt_condition",
        "prompt_variant",
        "turn_number",
        "model_id",
        "model_revision",
        "target_option",
        "independent_option",
        "sycophantic_option",
        "user_preferred_option",
        "user_claim_valid",
        "question_type",
        "behavior_target",
    )
    return {
        str(row["example_id"]): tuple(row.get(field) for field in fields)
        for row in records
    }


def compare_runs(
    baseline_records: list[dict],
    candidate_records: list[dict],
) -> dict:
    """Pair the same scripted inputs and settings across each model's own conversation."""

    baseline_prompts = _prompt_map(baseline_records)
    candidate_prompts = _prompt_map(candidate_records)
    if baseline_prompts != candidate_prompts:
        raise ValueError(
            "Unfair comparison: example IDs or prompt hashes differ between runs."
        )
    baseline_settings = {
        str(row["example_id"]): json.dumps(row["generation_settings"], sort_keys=True)
        for row in baseline_records
    }
    candidate_settings = {
        str(row["example_id"]): json.dumps(row["generation_settings"], sort_keys=True)
        for row in candidate_records
    }
    if baseline_settings != candidate_settings:
        raise ValueError("Unfair comparison: generation settings differ between runs.")
    if _provenance_map(baseline_records) != _provenance_map(candidate_records):
        raise ValueError(
            "Unfair comparison: benchmark, seed, model revision, prompt condition, "
            "targets, or source provenance differ between runs."
        )

    baseline_summary = summarize_records(baseline_records)
    candidate_summary = summarize_records(candidate_records)
    target_metric = (
        "independent_choice_rate"
        if "independent_choice_rate" in baseline_summary
        else "target_accuracy"
    )
    return {
        "comparison_valid": True,
        "matched_prompt_count": len(baseline_prompts),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": {
            target_metric: (
                candidate_summary[target_metric]["rate"]
                - baseline_summary[target_metric]["rate"]
            ),
            "sycophancy_rate": (
                candidate_summary["sycophancy_rate"]["rate"]
                - baseline_summary["sycophancy_rate"]["rate"]
            ),
            "invalid_answer_rate": (
                candidate_summary["invalid_answer_rate"]["rate"]
                - baseline_summary["invalid_answer_rate"]["rate"]
            ),
            "average_reward": (
                candidate_summary["average_reward"] - baseline_summary["average_reward"]
                if candidate_summary["average_reward"] is not None
                and baseline_summary["average_reward"] is not None
                else None
            ),
        },
        "paired_significance": exact_mcnemar(baseline_records, candidate_records),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = compare_runs(read_records(args.baseline), read_records(args.candidate))
    write_json(args.output, report)
    print(f"Wrote paired comparison to {args.output}.")


def compare_runs_from_paths(
    baseline_path: Path,
    candidate_path: Path,
) -> dict:
    """Read two record files and run the paired comparison.

    Convenience wrapper that the CLI dispatch uses so the command
    surface never has to import ``read_records`` directly.
    """

    return compare_runs(read_records(baseline_path), read_records(candidate_path))


if __name__ == "__main__":
    main()
