"""Tiered exact, lexical/n-gram, and optional semantic leakage audit."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sycophancy_rl.data_prep.schema import prompt_text, read_jsonl

SemanticSimilarity = Callable[[str, str], float]

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "about", "after", "answer", "based", "before", "choose", "claim",
    "following", "independently", "please", "reason", "reply", "short",
    "sentence", "should", "their", "there", "these", "think", "what", "which",
    "would", "your",
}


def _comparison_text(row: dict[str, Any]) -> str:
    question = row.get("metadata", {}).get("question_text")
    if isinstance(question, str) and question.strip():
        return question
    return prompt_text(row)


def _canonical_prompt(row: dict[str, Any]) -> str:
    """Normalize case, whitespace, and punctuation for exact comparison."""

    return " ".join(_TOKEN.findall(prompt_text(row).casefold()))


def _token_sequence(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        token
        for token in _TOKEN.findall(_comparison_text(row).casefold())
        if len(token) >= 4 and token not in _STOP
    )


def _ngrams(tokens: tuple[str, ...], size: int = 2) -> set[tuple[str, ...]]:
    return {tokens[index:index + size] for index in range(len(tokens) - size + 1)}


def _jaccard(left: set[Any], right: set[Any]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")


def leakage_report(
    development_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    *,
    lexical_threshold: float = 0.8,
    ngram_threshold: float = 0.8,
    semantic_similarity: SemanticSimilarity | None = None,
    semantic_threshold: float = 0.9,
) -> dict[str, Any]:
    """Return auditable overlap tiers without requiring heavy CI dependencies.

    ``semantic_similarity`` is an optional caller-supplied function returning a
    score in ``[0, 1]``. Normal CI runs exact and lightweight lexical tiers;
    production audits may inject a pinned embedding or cross-encoder scorer.
    """

    _validate_threshold("lexical_threshold", lexical_threshold)
    _validate_threshold("ngram_threshold", ngram_threshold)
    _validate_threshold("semantic_threshold", semantic_threshold)

    benchmark_fingerprints: dict[str, list[str]] = defaultdict(list)
    for row in benchmark_rows:
        benchmark_fingerprints[_canonical_prompt(row)].append(row["example_id"])
    exact = [
        {"development_id": row["example_id"], "benchmark_id": benchmark_id}
        for row in development_rows
        for benchmark_id in benchmark_fingerprints.get(_canonical_prompt(row), ())
    ]
    exact_pairs = {
        (item["development_id"], item["benchmark_id"]) for item in exact
    }

    benchmark_sequences = [_token_sequence(row) for row in benchmark_rows]
    benchmark_token_sets = [set(tokens) for tokens in benchmark_sequences]
    benchmark_ngram_sets = [_ngrams(tokens) for tokens in benchmark_sequences]
    inverted: dict[str, set[int]] = defaultdict(set)
    for index, tokens in enumerate(benchmark_token_sets):
        for token in tokens:
            inverted[token].add(index)

    near: list[dict[str, Any]] = []
    for row in development_rows:
        sequence = _token_sequence(row)
        left_tokens = set(sequence)
        if len(left_tokens) < 2:
            continue
        candidate_indexes = {
            index for token in left_tokens for index in inverted.get(token, ())
        }
        left_ngrams = _ngrams(sequence)
        for index in candidate_indexes:
            benchmark_id = benchmark_rows[index]["example_id"]
            if (row["example_id"], benchmark_id) in exact_pairs:
                continue
            token_score = _jaccard(left_tokens, benchmark_token_sets[index])
            ngram_score = _jaccard(left_ngrams, benchmark_ngram_sets[index])
            if token_score >= lexical_threshold or ngram_score >= ngram_threshold:
                near.append(
                    {
                        "development_id": row["example_id"],
                        "benchmark_id": benchmark_id,
                        "token_jaccard": token_score,
                        "bigram_jaccard": ngram_score,
                    }
                )
    near.sort(
        key=lambda item: max(item["token_jaccard"], item["bigram_jaccard"]),
        reverse=True,
    )

    semantic: list[dict[str, Any]] = []
    if semantic_similarity is not None:
        blocked_pairs = exact_pairs | {
            (item["development_id"], item["benchmark_id"]) for item in near
        }
        for development in development_rows:
            for benchmark in benchmark_rows:
                pair = (development["example_id"], benchmark["example_id"])
                if pair in blocked_pairs:
                    continue
                score = float(
                    semantic_similarity(
                        _comparison_text(development),
                        _comparison_text(benchmark),
                    )
                )
                if not 0.0 <= score <= 1.0:
                    raise ValueError(
                        "semantic_similarity must return a score between 0 and 1."
                    )
                if score >= semantic_threshold:
                    semantic.append(
                        {
                            "development_id": development["example_id"],
                            "benchmark_id": benchmark["example_id"],
                            "semantic_similarity": score,
                        }
                    )
        semantic.sort(key=lambda item: item["semantic_similarity"], reverse=True)

    return {
        "development_examples": len(development_rows),
        "benchmark_examples": len(benchmark_rows),
        "exact_prompt_overlap_count": len(exact),
        "exact_prompt_overlaps": exact,
        "lexical_threshold": lexical_threshold,
        "ngram_threshold": ngram_threshold,
        "high_lexical_similarity_count": len(near),
        "high_lexical_similarity_pairs": near,
        "semantic_enabled": semantic_similarity is not None,
        "semantic_threshold": semantic_threshold,
        "high_semantic_similarity_count": len(semantic),
        "high_semantic_similarity_pairs": semantic,
        "passed": not exact and not near and not semantic,
        "method": (
            "canonical exact prompts, token/bigram Jaccard candidate scan, "
            "and optional caller-supplied semantic similarity"
        ),
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
        "--output", type=Path, default=Path("data/splits/leakage_audit.json")
    )
    parser.add_argument("--lexical-threshold", type=float, default=0.8)
    parser.add_argument("--ngram-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    development = [row for path in args.development for row in read_jsonl(path)]
    benchmark = read_jsonl(args.benchmark, expected_role="benchmark")
    report = leakage_report(
        development,
        benchmark,
        lexical_threshold=args.lexical_threshold,
        ngram_threshold=args.ngram_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise SystemExit(
            "Leakage audit failed; inspect exact/high-similarity pairs in "
            f"{args.output}."
        )
    print(f"Leakage audit passed; wrote {args.output}.")


if __name__ == "__main__":
    main()
