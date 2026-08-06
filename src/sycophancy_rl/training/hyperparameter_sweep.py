"""Generate a validation-only GRPO hyperparameter search plan."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path


def build_plan(
    learning_rates: list[float],
    betas: list[float],
    seeds: list[int],
    *,
    profile: str,
) -> list[dict]:
    """Return controlled runs that vary only declared hyperparameters."""

    return [
        {
            "run_name": f"sweep-lr{learning_rate:g}-beta{beta:g}-seed{seed}",
            "profile": profile,
            "learning_rate": learning_rate,
            "beta": beta,
            "seed": seed,
            "selection_dataset": "validation",
            "test_set_used_for_selection": False,
        }
        for learning_rate, beta, seed in product(learning_rates, betas, seeds)
    ]


def _parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rates", default="2e-6,5e-6,1e-5")
    parser.add_argument("--betas", default="0.02,0.05")
    parser.add_argument("--seeds", default="17,42,73")
    parser.add_argument("--profile", default="local_8gb")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sweeps/grpo_validation_plan.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = build_plan(
        _parse_floats(args.learning_rates),
        _parse_floats(args.betas),
        _parse_ints(args.seeds),
        profile=args.profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(plan)} validation-only sweep configurations to {args.output}.")


if __name__ == "__main__":
    main()
