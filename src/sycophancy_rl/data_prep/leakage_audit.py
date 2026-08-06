"""Audit exact and high lexical-similarity leakage across data roles."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from sycophancy_rl.data_prep.schema import (
    normalized_prompt_fingerprint,
    prompt_text,
    read_jsonl,
)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "about", "after", "answer", "based", "before", "choose", "claim",
    "following", "independently", "please", "reason", "reply", "short",
    "sentence", "should", "their", "there", "these", "think", "which",
    "would", "your",
}


def _comparison_text(row: dict[str, Any]) -> str:
    question = row.get("metadata", {}).get("question_text")
    if isinstance(question, str) and question.strip():
        return question
    return prompt_text(row)


def _tokens(row: dict[str, Any]) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(_comparison_text(row).casefold())
        if len(token) >= 4 and token not in _STOP
    }


def leakage_report(
    development_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    *,
    lexical_threshold: float = 0.8,
) -> dict[str, Any]:
    """Return exact prompt overlap and high token-Jaccard candidates."""

    benchmark_fingerprints = {
        normalized_prompt_fingerprint(prompt_text(row)): row["example_id"]
        for row in benchmark_rows
    }
    exact = [
        {
            "development_id": row["example_id"],
            "benchmark_id": benchmark_fingerprints[fingerprint],
        }
        for row in development_rows
        if (
            fingerprint := normalized_prompt_fingerprint(prompt_text(row))
        ) in benchmark_fingerprints
    ]

    benchmark_token_sets = [_tokens(row) for row in benchmark_rows]
    inverted: dict[str, set[int]] = defaultdict(set)
    for index, tokens in enumerate(benchmark_token_sets):
        for token in tokens:
            inverted[token].add(index)
    near: list[dict[str, Any]] = []
    for row in development_rows:
        left = _tokens(row)
        if len(left) < 2:
            continue
        candidate_counts: dict[int, int] = defaultdict(int)
        for token in left:
            for index in inverted.get(token, ()):
                candidate_counts[index] += 1
        for index, shared in candidate_counts.items():
            right = benchmark_token_sets[index]
            if shared < 2 or not right:
                continue
            similarity = shared / len(left | right)
            if similarity >= lexical_threshold:
                near.append(
                    {
                        "development_id": row["example_id"],
                        "benchmark_id": benchmark_rows[index]["example_id"],
                        "token_jaccard": similarity,
                    }
                )
    near.sort(key=lambda item: item["token_jaccard"], reverse=True)
    return {
        "development_examples": len(development_rows),
        "benchmark_examples": len(benchmark_rows),
        "exact_prompt_overlap_count": len(exact),
        "exact_prompt_overlaps": exact,
        "lexical_threshold": lexical_threshold,
        "high_lexical_similarity_count": len(near),
        "high_lexical_similarity_pairs": near,
        "passed": not exact and not near,
        "method": "exact normalized prompt hashes plus token-set Jaccard candidate scan",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        type=Path,
        nargs="+",
        default=[
            Path("data/splits/train.jsonl"),
            Path("data/splits/validation.jsonl"),
            Path("data/splits/test.jsonl"),
        ],
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/benchmarks/anthropic_sycophancy.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/leakage_audit.json"),
    )
    parser.add_argument("--lexical-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    development = [row for path in args.development for row in read_jsonl(path)]
    benchmark = read_jsonl(args.benchmark, expected_role="benchmark")
    report = leakage_report(
        development,
        benchmark,
        lexical_threshold=args.lexical_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(
            "Leakage audit failed; inspect exact/high-similarity pairs in "
            f"{args.output}."
        )
    print(f"Leakage audit passed; wrote {args.output}.")


if __name__ == "__main__":
    main()
