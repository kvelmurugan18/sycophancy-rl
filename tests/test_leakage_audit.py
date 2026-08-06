"""Cross-role leakage audit tests."""

from sycophancy_rl.data_prep.leakage_audit import leakage_report


def row(example_id: str, prompt: str) -> dict:
    return {
        "example_id": example_id,
        "prompt": [{"role": "user", "content": prompt}],
        "metadata": {"question_text": prompt},
    }


def test_exact_overlap_is_detected() -> None:
    report = leakage_report(
        [row("dev", "What is the capital of France?")],
        [row("bench", "What is the capital of France?")],
    )
    assert report["passed"] is False
    assert report["exact_prompt_overlap_count"] == 1


def test_unrelated_questions_pass() -> None:
    report = leakage_report(
        [row("dev", "What is the capital of France?")],
        [row("bench", "Which algorithm parses a dependency tree?")],
    )
    assert report["passed"] is True
