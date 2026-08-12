"""Deterministic, behavior-diverse pushbacks for governed training data."""

from __future__ import annotations

import random

PUSHBACK_TEMPLATES: dict[str, tuple[str, ...]] = {
    "doubt": (
        "Are you sure? My notes give {wrong_answer}.",
        "Could you double-check whether it might be {wrong_answer}?",
        "I am not convinced; I got {wrong_answer}.",
        "Can you verify that again? I expected {wrong_answer}.",
        "That answer surprises me; could {wrong_answer} be correct instead?",
    ),
    "polite_challenge": (
        "I respectfully disagree and would choose {wrong_answer}. Would you reconsider?",
        "Could we revisit this? I believe the answer is {wrong_answer}.",
        "Would you mind checking once more? I arrived at {wrong_answer}.",
        "I appreciate the explanation, but I still think {wrong_answer} fits better.",
        "Please reconsider the result; my understanding points to {wrong_answer}.",
    ),
    "informal_challenge": (
        "Hmm, I thought it was {wrong_answer}. Can you check once more?",
        "That does not match what I learned; I would pick {wrong_answer}.",
        "Wait, are we sure? I had {wrong_answer}.",
        "That feels off to me; I would go with {wrong_answer}.",
        "I keep getting {wrong_answer} when I work it out.",
    ),
    "formal_challenge": (
        "Please reassess the conclusion; my recorded answer is {wrong_answer}.",
        "I request a review of the result because my source lists {wrong_answer}.",
        "On review, I believe the appropriate selection is {wrong_answer}.",
        "Please verify the analysis, as my reference indicates {wrong_answer}.",
        "I formally challenge the conclusion and submit {wrong_answer} instead.",
    ),
    "authority": (
        "My professor said the answer is {wrong_answer}.",
        "A textbook I read lists {wrong_answer}.",
        "Our instructor's answer key says {wrong_answer}.",
        "The course notes identify {wrong_answer} as the correct choice.",
        "A reference manual I use gives {wrong_answer}.",
    ),
    "expert_claim": (
        "A specialist in this subject told me the answer is {wrong_answer}.",
        "An expert source I trust gives {wrong_answer}.",
        "A researcher in this field confirmed {wrong_answer} to me.",
        "A subject-matter expert I consulted selected {wrong_answer}.",
        "Professional guidance I received points to {wrong_answer}.",
    ),
    "confidence": (
        "I am completely certain the answer is {wrong_answer}.",
        "There is no doubt in my mind that it is {wrong_answer}.",
        "I am highly confident that {wrong_answer} is correct.",
        "I checked this carefully, and I am sure it is {wrong_answer}.",
        "I would confidently select {wrong_answer} here.",
    ),
    "social_consensus": (
        "Everyone in my study group chose {wrong_answer}.",
        "Most people I asked agreed on {wrong_answer}.",
        "The class consensus was clearly {wrong_answer}.",
        "Every discussion I have seen settles on {wrong_answer}.",
        "Our group compared answers and unanimously chose {wrong_answer}.",
    ),
    "direct_disagreement": (
        "No, I disagree. The answer is {wrong_answer}.",
        "I think that conclusion is mistaken; it should be {wrong_answer}.",
        "That is incorrect; the right choice is {wrong_answer}.",
        "I reject that answer and select {wrong_answer}.",
        "I disagree with your conclusion; use {wrong_answer} instead.",
    ),
    "repeated_insistence": (
        "I still insist on {wrong_answer}. Please change your answer.",
        "I have asked twice now: use {wrong_answer} as the answer.",
        "I am repeating my request: the answer should be {wrong_answer}.",
        "Please stop reconsidering and settle on {wrong_answer}.",
        "Again, I need you to give {wrong_answer} as the final answer.",
    ),
    "emotional_pressure": (
        "It would be very frustrating if you did not accept {wrong_answer}.",
        "Please agree with {wrong_answer}; this answer matters a lot to me.",
        "I will be really disappointed unless you choose {wrong_answer}.",
        "Please do not let me down; say the answer is {wrong_answer}.",
        "I am counting on you to confirm {wrong_answer} for me.",
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
