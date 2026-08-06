"""Create a blinded human-review sheet and calculate annotator agreement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path
from typing import Any

from sycophancy_rl.evaluation.io import read_records, write_json
from sycophancy_rl.evaluation.metrics import cohen_kappa

LABEL_COLUMNS = (
    "behavior_label",
    "helpfulness_label",
    "respectfulness_label",
    "explanation_quality_label",
)


def create_review_sheet(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    sample_size: int,
    seed: int,
) -> int:
    """Sample and blind responses without exposing model/checkpoint names."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive.")
    rng = random.Random(seed)
    selected = rng.sample(records, min(sample_size, len(records)))
    rng.shuffle(selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "review_id",
        "prompt",
        "response",
        "behavior_label_annotator_1",
        "behavior_label_annotator_2",
        "helpfulness_label_annotator_1",
        "helpfulness_label_annotator_2",
        "respectfulness_label_annotator_1",
        "respectfulness_label_annotator_2",
        "explanation_quality_label_annotator_1",
        "explanation_quality_label_annotator_2",
        "notes_annotator_1",
        "notes_annotator_2",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            key = f"{row['example_id']}\0{row.get('model_id')}\0{row.get('adapter_path')}"
            writer.writerow(
                {
                    "review_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                    "prompt": row["prompt"],
                    "response": row["generated_response"],
                }
            )
    return len(selected)


def analyze_review_sheet(path: Path) -> dict[str, Any]:
    """Calculate agreement for every completed pair of annotation columns."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, Any] = {"review_count": len(rows), "agreement": {}}
    for base in LABEL_COLUMNS:
        left_key = f"{base}_annotator_1"
        right_key = f"{base}_annotator_2"
        paired = [
            (row[left_key].strip(), row[right_key].strip())
            for row in rows
            if row.get(left_key, "").strip() and row.get(right_key, "").strip()
        ]
        result["agreement"][base] = (
            {"completed_pairs": 0}
            if not paired
            else {
                "completed_pairs": len(paired),
                **cohen_kappa(
                    [left for left, _ in paired],
                    [right for _, right in paired],
                ),
            }
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--records", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--sample-size", type=int, default=100)
    create.add_argument("--seed", type=int, default=42)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--sheet", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "create":
        count = create_review_sheet(
            read_records(args.records),
            args.output,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        print(f"Created a blinded {count}-response review sheet at {args.output}.")
    else:
        result = analyze_review_sheet(args.sheet)
        write_json(args.output, result)
        print(f"Wrote inter-annotator agreement report to {args.output}.")


if __name__ == "__main__":
    main()
