"""Tiered cross-role leakage audit tests."""

import pytest

from sycophancy_rl.data_prep.leakage_audit import leakage_report


def row(example_id: str, prompt: str) -> dict:
    return {
        "example_id": example_id,
        "prompt": [{"role": "user", "content": prompt}],
        "metadata": {"question_text": prompt},
    }


def test_exact_overlap_normalizes_case_whitespace_and_punctuation() -> None:
    report = leakage_report(
        [row("dev", "What is the capital of France?")],
        [row("bench", "  WHAT is the capital of France! ")],
    )
    assert report["passed"] is False
    assert report["exact_prompt_overlap_count"] == 1


def test_near_duplicate_is_detected_by_lightweight_lexical_tier() -> None:
    report = leakage_report(
        [row("dev", "Which city is the capital of France?")],
        [row("bench", "What city is France's capital?")],
    )
    assert report["passed"] is False
    assert report["high_lexical_similarity_count"] == 1


def test_optional_semantic_tier_detects_a_paraphrase() -> None:
    def semantic_similarity(left: str, right: str) -> float:
        assert left and right
        return 0.95

    report = leakage_report(
        [row("dev", "Name the French seat of government.")],
        [row("bench", "Which metropolis serves as France's capital?")],
        semantic_similarity=semantic_similarity,
    )
    assert report["semantic_enabled"] is True
    assert report["high_semantic_similarity_count"] == 1
    assert report["passed"] is False


def test_unrelated_questions_pass() -> None:
    report = leakage_report(
        [row("dev", "What is the capital of France?")],
        [row("bench", "Which algorithm parses a dependency tree?")],
    )
    assert report["passed"] is True


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        leakage_report([], [], lexical_threshold=1.1)
