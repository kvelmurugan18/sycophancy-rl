"""Aggregate completed evaluation summaries across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sycophancy_rl.evaluation.io import write_json
from sycophancy_rl.evaluation.metrics import seed_aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(args.summaries) < 3:
        raise SystemExit("At least three completed seed summaries are required.")
    summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.summaries
    ]
    result = seed_aggregate(summaries)
    result["source_summaries"] = [path.as_posix() for path in args.summaries]
    write_json(args.output, result)
    print(f"Wrote {len(summaries)}-seed aggregate to {args.output}.")


if __name__ == "__main__":
    main()
