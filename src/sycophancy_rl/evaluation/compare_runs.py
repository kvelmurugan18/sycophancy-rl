"""Create a paired, prompt-controlled before/after comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sycophancy_rl.evaluation.io import read_records, write_json
from sycophancy_rl.evaluation.metrics import exact_mcnemar, summarize_records


def _prompt_map(records: list[dict]) -> dict[str, str]:
    mapped = {str(row["example_id"]): str(row["prompt_sha256"]) for row in records}
    if len(mapped) != len(records):
        raise ValueError("Comparison inputs contain duplicate example IDs.")
    return mapped


def _provenance_map(records: list[dict]) -> dict[str, tuple]:
    fields = (
        "source_example_id",
        "source",
        "source_revision",
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
    """Require identical examples/prompts/settings before comparing models."""

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
    return {
        "comparison_valid": True,
        "matched_prompt_count": len(baseline_prompts),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": {
            "target_accuracy": (
                candidate_summary["target_accuracy"]["rate"]
                - baseline_summary["target_accuracy"]["rate"]
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
