"""Summarize saved raw responses using non-misleading sycophancy metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from sycophancy_rl.evaluation.io import read_records, write_json
from sycophancy_rl.evaluation.metrics import summarize_records


def calculate_metrics(results_path: str | Path) -> dict:
    """Return metrics with independent, sycophantic, and invalid categories."""

    return summarize_records(read_records(results_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=Path("outputs/evaluations/baseline/responses.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = calculate_metrics(args.results)
    if args.output:
        write_json(args.output, summary)
    print("=" * 64)
    print("Sycophancy Evaluation Report")
    print("=" * 64)
    print(f"Examples                 : {summary['total_examples']}")
    print(
        "Independent-answer rate  : "
        f"{summary['independent_answer_rate']['rate']:.2%} "
        f"(95% CI {summary['independent_answer_rate']['ci95'][0]:.2%}–"
        f"{summary['independent_answer_rate']['ci95'][1]:.2%})"
    )
    print(f"Sycophancy rate          : {summary['sycophancy_rate']['rate']:.2%}")
    print(f"Invalid-answer rate      : {summary['invalid_answer_rate']['rate']:.2%}")
    target_metric = summary["primary_target_metric"]
    target_label = target_metric.replace("_", " ").title()
    print(f"{target_label:<26}: {summary[target_metric]['rate']:.2%}")
    print(f"Average computed reward  : {summary['average_reward']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
