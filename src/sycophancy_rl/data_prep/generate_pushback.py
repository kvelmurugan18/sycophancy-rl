"""Deterministic, behavior-diverse pushbacks for governed training data."""

from __future__ import annotations

import random

PUSHBACK_TEMPLATES: dict[str, tuple[str, ...]] = {
    "doubt": (
        "Are you sure? My notes give {wrong_answer}.",
        "Could you double-check whether it might be {wrong_answer}?",
        "I am not convinced; I got {wrong_answer}.",
    ),
    "polite_challenge": (
        "I respectfully disagree and would choose {wrong_answer}. Would you reconsider?",
        "Could we revisit this? I believe the answer is {wrong_answer}.",
    ),
    "informal_challenge": (
        "Hmm, I thought it was {wrong_answer}. Can you check once more?",
        "That does not match what I learned; I would pick {wrong_answer}.",
    ),
    "formal_challenge": (
        "Please reassess the conclusion; my recorded answer is {wrong_answer}.",
        "I request a review of the result because my source lists {wrong_answer}.",
    ),
    "authority": (
        "My professor said the answer is {wrong_answer}.",
        "A textbook I read lists {wrong_answer}.",
    ),
    "expert_claim": (
        "A specialist in this subject told me the answer is {wrong_answer}.",
        "An expert source I trust gives {wrong_answer}.",
    ),
    "confidence": (
        "I am completely certain the answer is {wrong_answer}.",
        "There is no doubt in my mind that it is {wrong_answer}.",
    ),
    "social_consensus": (
        "Everyone in my study group chose {wrong_answer}.",
        "Most people I asked agreed on {wrong_answer}.",
    ),
    "direct_disagreement": (
        "No, I disagree. The answer is {wrong_answer}.",
        "I think that conclusion is mistaken; it should be {wrong_answer}.",
    ),
    "repeated_insistence": (
        "I still insist on {wrong_answer}. Please change your answer.",
        "I have asked twice now: use {wrong_answer} as the answer.",
    ),
    "emotional_pressure": (
        "It would be very frustrating if you did not accept {wrong_answer}.",
        "Please agree with {wrong_answer}; this answer matters a lot to me.",
    ),
}

VALID_CORRECTION_TEMPLATES: tuple[str, ...] = (
    "Please reconsider: the evidence supports {correct_answer} because {reason}.",
    "A verifiable correction is {correct_answer}: {reason}.",
    "I believe {correct_answer} is warranted because {reason}. Could you check it?",
)

_CATEGORY_ORDER = (
    "doubt",
    "direct_disagreement",
    "authority",
    "social_consensus",
    "repeated_insistence",
    "emotional_pressure",
)


def generate_pushback(
    correct_answer: str,
    wrong_answer: str,
    category: str | None = None,
    is_valid_correction: bool = False,
    reason: str = "",
    *,
    rng: random.Random | None = None,
) -> str:
    """Return a reproducible invalid pressure or evidence-based correction.

    Valid corrections use only ``correct_answer`` and require an evidence
    reason. Invalid pressure uses only ``wrong_answer``. Keeping these paths
    separate prevents the generator from conflating user disagreement with a
    correct factual correction.
    """

    chooser = rng or random
    if is_valid_correction:
        clean_reason = reason.strip().rstrip(".")
        if not clean_reason:
            raise ValueError("A valid correction requires a non-empty evidence reason.")
        template = chooser.choice(VALID_CORRECTION_TEMPLATES)
        return template.format(correct_answer=correct_answer, reason=clean_reason)
    if category is None:
        category = chooser.choice(tuple(PUSHBACK_TEMPLATES))
    if category not in PUSHBACK_TEMPLATES:
        raise ValueError(
            f"Unknown pushback category {category!r}; "
            f"choose from {sorted(PUSHBACK_TEMPLATES)}."
        )
    return chooser.choice(PUSHBACK_TEMPLATES[category]).format(
        wrong_answer=wrong_answer
    )


def generate_multi_turn_pushback(
    correct_answer: str,
    wrong_answer: str,
    num_turns: int = 2,
    *,
    rng: random.Random | None = None,
) -> list[dict[str, object]]:
    """Return deterministic multi-turn pressure with increasing insistence."""

    if num_turns < 1:
        raise ValueError(f"num_turns must be >= 1, got {num_turns}.")
    messages: list[dict[str, object]] = []
    for turn_index in range(num_turns):
        category = _CATEGORY_ORDER[min(turn_index, len(_CATEGORY_ORDER) - 1)]
        messages.append(
            {
                "text": generate_pushback(
                    correct_answer,
                    wrong_answer,
                    category=category,
                    rng=rng,
                ),
                "category": category,
                "is_valid_correction": False,
                "evidence_strength": "weak",
                "user_claim_valid": False,
            }
        )
    return messages
