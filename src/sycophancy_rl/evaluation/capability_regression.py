"""Compare lm-evaluation-harness task scores before and after training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sycophancy_rl.evaluation.io import write_json


def _scores(payload: dict[str, Any]) -> dict[str, float]:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("Expected an lm-eval result object with a 'results' mapping.")
    scores: dict[str, float] = {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        preferred = (
            "acc_norm,none",
            "acc,none",
            "exact_match,strict-match",
            "f1,none",
        )
        metric = next(
            (
                float(metrics[key])
                for key in preferred
                if isinstance(metrics.get(key), (int, float))
            ),
            None,
        )
        if metric is not None:
            scores[str(task)] = metric
    if not scores:
        raise ValueError("No supported numeric task scores were found.")
    return scores


def compare_capabilities(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    maximum_allowed_drop: float,
) -> dict[str, Any]:
    baseline_scores = _scores(baseline)
    candidate_scores = _scores(candidate)
    if set(baseline_scores) != set(candidate_scores):
        raise ValueError("Capability comparison requires identical task names.")
    tasks = {
        task: {
            "baseline": baseline_scores[task],
            "candidate": candidate_scores[task],
            "delta": candidate_scores[task] - baseline_scores[task],
            "passed": candidate_scores[task] - baseline_scores[task]
            >= -maximum_allowed_drop,
        }
        for task in sorted(baseline_scores)
    }
    return {
        "maximum_allowed_drop": maximum_allowed_drop,
        "tasks": tasks,
        "passed": all(result["passed"] for result in tasks.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--maximum-allowed-drop", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = compare_capabilities(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
        maximum_allowed_drop=args.maximum_allowed_drop,
    )
    write_json(args.output, result)
    if not result["passed"]:
        raise SystemExit(f"Capability regression threshold failed; see {args.output}.")
    print(f"Capability regression check passed; wrote {args.output}.")


if __name__ == "__main__":
    main()
