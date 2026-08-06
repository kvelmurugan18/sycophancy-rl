"""Compare completed evaluation conditions without inventing missing results."""

from __future__ import annotations

import argparse
from pathlib import Path

from sycophancy_rl.evaluation.io import read_records, write_json
from sycophancy_rl.evaluation.metrics import summarize_records


def analyze_runs(run_paths: list[Path]) -> dict:
    """Summarize model/prompt/reward ablations from their raw response files."""

    if len(run_paths) < 2:
        raise ValueError("At least two completed runs are required for an ablation table.")
    rows: list[dict] = []
    for path in run_paths:
        records = read_records(path)
        summary = summarize_records(records)
        first = records[0]
        rows.append(
            {
                "run": path.parent.name,
                "model_id": first.get("model_id"),
                "adapter_path": first.get("adapter_path"),
                "system_prompt_condition": first.get("system_prompt_condition"),
                "examples": summary["total_examples"],
                "target_accuracy": summary["target_accuracy"]["rate"],
                "sycophancy_rate": summary["sycophancy_rate"]["rate"],
                "invalid_rate": summary["invalid_answer_rate"]["rate"],
                "format_compliance_rate": summary[
                    "required_format_compliance_rate"
                ]["rate"],
                "average_reward": summary["average_reward"],
            }
        )
    return {"runs": rows}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ablation_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = analyze_runs(args.runs)
    write_json(args.output, result)
    print(
        f"{'Run':<24} {'Prompt':<24} {'Target':>9} {'Sycophancy':>12} "
        f"{'Invalid':>9} {'Reward':>9}"
    )
    print("-" * 92)
    for row in result["runs"]:
        print(
            f"{row['run']:<24} {row['system_prompt_condition']:<24} "
            f"{row['target_accuracy']:>8.2%} {row['sycophancy_rate']:>11.2%} "
            f"{row['invalid_rate']:>8.2%} {row['average_reward']:>9.3f}"
        )


if __name__ == "__main__":
    main()
