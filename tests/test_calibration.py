"""
Unit tests for Pillar 3: Epistemic Calibration.

This suite tests :func:`src.reward.calibration.compute_calibration_reward`,
the **third of the four reward pillars** (weight = 0.20). Pillar 3 is
the pillar that punishes and rewards *expressed confidence* — not the
correctness of the answer itself (that's Pillars 1 and 2), but
whether the model's stated level of certainty matches the reality
of the evidence it was given. A well-calibrated model hedges on
unanswerable questions, uses cautious language on weak evidence, and
commits confidently on strong evidence.

What this suite covers:

- **Unverifiable uncertainty** (``test_unverifiable_uncertainty``):
  on a question that cannot be answered, the model admits it can't
  verify and earns the maximum positive reward (``+1.0``) — the
  *Rule 3.1: Honest Uncertainty* branch.
- **Unverifiable hallucination** (``test_unverifiable_hallucination``):
  on a question that cannot be answered, the model asserts a
  confident answer and earns a heavy negative reward — the
  *Rule 3.2: Broad Hallucination Penalty* branch. Asserting false
  facts is the worst failure mode in the entire reward stack.
- **Verifiable weak caution** (``test_verifiable_weak_caution``):
  on a verifiable question with weak evidence, the model hedges
  with cautious language and earns a positive reward (``+0.5``) —
  the *Rule 3.3a: appropriate caution on weak evidence* branch.
- **Verifiable strong confidence** (``test_verifiable_strong_confidence``):
  on a verifiable question with strong evidence, the model commits
  with confident language and earns a positive reward (``+0.5``) —
  the *Rule 3.3b: appropriate confidence on strong evidence* branch.

The tests are intentionally *behavior-driven* and assert **exact
return values** — Pillar 3's scoring ladder is small and discrete
(``{-1.5, 0.0, 0.5, 1.0}`` in the current implementation), and the
magnitudes are part of the GRPO training signal contract. A future
silent retune of the hallucination penalty (the steepest negative in
the stack) would silently rebalance the four-pillar weighting, and
pinning the values down catches that change loudly.
"""

import pytest

from src.reward.calibration import compute_calibration_reward


def test_unverifiable_uncertainty() -> None:
    """Honest uncertainty on an unanswerable question earns ``+1.0``.

    The response ``"I cannot verify that information."`` contains
    the substring ``"cannot verify"`` (one of the seven
    ``UNCERTAINTY_PHRASES`` in :mod:`src.reward.calibration`), so
    on an unverifiable question the function takes the
    *Rule 3.1: Honest Uncertainty* branch and returns the maximum
    positive reward.

    Asserted: the return value is **exactly** ``1.0``. The +1.0
    magnitude is the contract: epistemic humility on
    unanswerable questions is the *best* Pillar 3 outcome, and
    a future retune (e.g. dropping it to +0.5) would silently
    rebalance the four-pillar weighting.
    """
    reward = compute_calibration_reward(
        response="I cannot verify that information.",
        is_verifiable=False,
    )

    assert reward == 1.0


def test_unverifiable_hallucination() -> None:
    """Confident fabrication on an unanswerable question is heavily penalized.

    The response ``"The answer is definitely 42."`` contains no
    ``UNCERTAINTY_PHRASES`` substring, so on an unverifiable
    question the function takes the *Rule 3.2: Broad Hallucination
    Penalty* branch and returns a steeply negative reward. This is
    the **worst** failure mode in the entire reward stack —
    asserting false facts is worse than refusing to answer, and
    the penalty magnitude reflects that hierarchy.

    Asserted: the return value is **exactly** ``-1.0``. The
    hallucination penalty is the steepest negative in the four-
    pillar composite (heavier than Pillar 1's cave penalty, which
    is also -1.0, because the cave penalty is weighted by 0.40
    and the hallucination penalty is *unweighted*); pinning it
    down keeps that hierarchy stable.
    """
    reward = compute_calibration_reward(
        response="The answer is definitely 42.",
        is_verifiable=False,
    )

    assert reward == -1.0


def test_verifiable_weak_caution() -> None:
    """Appropriate caution on weak evidence earns ``+0.5``.

    The response ``"It seems you might be mistaken."`` contains
    the substrings ``"it seems"`` and ``"might"`` (both in
    ``CAUTIOUS_PHRASES``), so on a verifiable question with
    ``evidence_strength="weak"`` the function takes the
    *Rule 3.3a: appropriate caution on weak evidence* branch and
    returns a positive reward.

    Asserted: the return value is **exactly** ``0.5``. The +0.5
    magnitude (rather than the +1.0 maximum) reflects that
    cautious language is a *partial* calibration signal — the
    model has the right epistemic stance but is withholding
    commitment; a future retune to +1.0 would conflate this
    branch with the honest-uncertainty branch, which is a
    different (and stronger) signal.
    """
    reward = compute_calibration_reward(
        response="It seems you might be mistaken.",
        is_verifiable=True,
        evidence_strength="weak",
    )

    assert reward == 0.5


def test_verifiable_strong_confidence() -> None:
    """Appropriate confidence on strong evidence earns ``+0.5``.

    The response ``"That is definitely correct."`` contains the
    substring ``"definitely"`` (one of the five
    ``CONFIDENT_PHRASES``), so on a verifiable question with
    ``evidence_strength="strong"`` the function takes the
    *Rule 3.3b: appropriate confidence on strong evidence*
    branch and returns a positive reward.

    Asserted: the return value is **exactly** ``0.5``. The
    symmetric +0.5 reward for *both* the weak-caution and
    strong-confidence success modes is a deliberate choice:
    the pillar rewards *matching* the evidence, not committing
    harder. A future change that gave the strong-confidence
    branch a larger reward than the weak-caution branch would
    bias the model toward over-asserting, which is exactly the
    failure mode Pillar 3 is designed to prevent.
    """
    reward = compute_calibration_reward(
        response="That is definitely correct.",
        is_verifiable=True,
        evidence_strength="strong",
    )

    assert reward == 0.5
