"""Behavior-diversity and determinism tests for pushback generation."""

import random

import pytest

from sycophancy_rl.data_prep.generate_pushback import (
    PUSHBACK_TEMPLATES,
    generate_multi_turn_pushback,
    generate_pushback,
)


def test_required_pressure_families_are_present() -> None:
    assert {
        "doubt",
        "polite_challenge",
        "informal_challenge",
        "formal_challenge",
        "authority",
        "expert_claim",
        "confidence",
        "social_consensus",
        "direct_disagreement",
        "repeated_insistence",
        "emotional_pressure",
    } <= PUSHBACK_TEMPLATES.keys()


def test_seeded_multi_turn_generation_is_deterministic_and_escalates() -> None:
    first = generate_multi_turn_pushback("A", "B", 6, rng=random.Random(42))
    second = generate_multi_turn_pushback("A", "B", 6, rng=random.Random(42))
    assert first == second
    assert [turn["category"] for turn in first] == [
        "doubt",
        "direct_disagreement",
        "authority",
        "social_consensus",
        "repeated_insistence",
        "emotional_pressure",
    ]
    assert all(turn["is_valid_correction"] is False for turn in first)
    assert all(turn["user_claim_valid"] is False for turn in first)


def test_valid_correction_requires_evidence_and_uses_correct_answer() -> None:
    text = generate_pushback(
        "A",
        "B",
        is_valid_correction=True,
        reason="the cited calculation evaluates to four",
        rng=random.Random(7),
    )
    assert "A" in text
    assert "calculation" in text
    with pytest.raises(ValueError, match="evidence reason"):
        generate_pushback("A", "B", is_valid_correction=True)


def test_invalid_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown pushback category"):
        generate_pushback("A", "B", category="label_leak")
