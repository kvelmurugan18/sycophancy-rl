"""Generate a deterministic Markdown error-analysis appendix."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from sycophancy_rl.evaluation.io import read_records


def build_error_analysis(
    records: list[dict[str, Any]],
    *,
    examples_per_category: int = 5,
) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        labels = [str(row.get("category", "invalid"))]
        if row.get("contradictory"):
            labels.append("contradictory")
        if row.get("truncated"):
            labels.append("truncated")
        if row.get("format_compliant") is False:
            labels.append("format_failure")
        if row.get("user_claim_valid") is True and not row.get("target_selected"):
            labels.append("unnecessary_disagreement")
        for label in labels:
            groups[label].append(row)

    lines = ["# Error Analysis", ""]
    for category, rows in sorted(groups.items()):
        lines.extend([f"## {category.replace('_', ' ').title()}", ""])
        for row in sorted(rows, key=lambda value: str(value["example_id"]))[
            :examples_per_category
        ]:
            lines.extend(
                [
                    f"### `{row['example_id']}`",
                    "",
                    f"- Parsed label: `{row.get('parsed_label')}`",
                    f"- Parse status: `{row.get('parse_status')}`",
                    f"- Target option: `{row.get('target_option')}`",
                    f"- Reward: `{row.get('reward')}`",
                    "",
                    "**Response**",
                    "",
                    "```text",
                    str(row.get("generated_response", "")),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-category", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_error_analysis(
        read_records(args.records),
        examples_per_category=args.examples_per_category,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote error analysis to {args.output}.")


if __name__ == "__main__":
    main()
