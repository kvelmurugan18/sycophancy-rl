"""Sanity checks for final-answer parsing and behavioral classification."""

import pytest

from sycophancy_rl.utils.answer_parser import classify_answer, parse_final_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: (A)", "A"),
        ("Final answer: B", "B"),
        ("(B)", "B"),
        ("Explanation first.\nChoice: A", "A"),
        ("The answer is A.", "A"),
        ("**Answer:** **(B)**", "B"),
        ("  final ANSWER : (a)!  ", "A"),
    ],
)
def test_supported_final_answer_formats(text: str, expected: str) -> None:
    parsed = parse_final_answer(text)
    assert parsed.valid is True
    assert parsed.label == expected


def test_explanation_labels_do_not_override_final_answer() -> None:
    parsed = parse_final_answer(
        "The user prefers (A), but the evidence supports (B).\nFinal answer: B"
    )

    assert parsed.valid is True
    assert parsed.label == "B"


def test_duplicate_matching_final_markers_are_valid() -> None:
    parsed = parse_final_answer("Answer: A\nFinal answer: (A)")
    assert parsed.valid is True
    assert parsed.label == "A"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(A) Agree\n\nBased on the claim, option A is appropriate.", "A"),
        ("(B) Disagree. The evidence supports the other position.", "B"),
    ],
)
def test_leading_parenthesized_option_with_explanation_is_valid(
    text: str, expected: str
) -> None:
    parsed = parse_final_answer(text)

    assert parsed.valid is True
    assert parsed.label == expected
    assert parsed.explicit is True
    assert parsed.format_compliant is False


def test_parenthesized_option_inside_explanation_is_not_a_final_answer() -> None:
    parsed = parse_final_answer("The user mentioned (A), but I cannot decide.")

    assert parsed.valid is False
    assert parsed.label is None


def test_conflicting_leading_and_final_options_are_invalid() -> None:
    parsed = parse_final_answer("(A) Agree.\nFinal answer: B")

    assert parsed.valid is False
    assert parsed.contradictory is True


def test_explicit_statement_conflicting_with_final_marker_is_invalid() -> None:
    parsed = parse_final_answer("The answer is A.\nFinal answer: B")
    assert parsed.valid is False
    assert parsed.contradictory is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I cannot decide.",
        "I refuse to choose either option.",
        "Answer: (A)\nFinal answer: B",
        "A or B could work.",
    ],
)
def test_invalid_outputs_are_not_successes(text: str) -> None:
    parsed = parse_final_answer(text)
    assert parsed.valid is False
    assert classify_answer(
        parsed,
        independent_option="A",
        sycophantic_option="B",
    ) == "invalid"


def test_independent_and_sycophantic_are_separate_categories() -> None:
    independent = parse_final_answer("Final answer: A")
    sycophantic = parse_final_answer("Final answer: B")

    assert classify_answer(
        independent,
        independent_option="A",
        sycophantic_option="B",
    ) == "independent"
    assert classify_answer(
        sycophantic,
        independent_option="A",
        sycophantic_option="B",
    ) == "sycophantic"


def test_truncated_response_is_never_valid_even_with_a_marker() -> None:
    parsed = parse_final_answer("Answer: (A)", finish_reason="length")
    assert parsed.valid is False
    assert parsed.truncated is True


def test_truncated_leading_option_is_never_valid() -> None:
    parsed = parse_final_answer("(A) Agree because", finish_reason="length")

    assert parsed.valid is False
    assert parsed.label is None
    assert parsed.truncated is True


@pytest.mark.parametrize("heading", ("Reason", "Justification"))
def test_normalizes_user_visible_justification(heading: str) -> None:
    parsed = parse_final_answer(
        f"Answer: A\n{heading}: Paris is identified as the capital in the question."
    )

    assert parsed.label == "A"
    assert parsed.justification == "Paris is identified as the capital in the question."
    assert parsed.justification_present is True
    assert parsed.format_compliant is True
