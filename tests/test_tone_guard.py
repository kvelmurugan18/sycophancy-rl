"""
Unit tests for Pillar 4: Tone Guard.

This suite tests :func:`sycophancy_rl.reward.tone_guard.compute_tone_reward`, the
**fourth of the four reward pillars** (weight = 0.15). Pillar 4 is a
**strictly negative-only guard** — it never returns a positive reward.
Its job is to pull responses away from two extreme tones — groveling
and aggression — and into a polite-but-firm baseline. When no extreme
tone is detected, it abstains (``0.0``) and leaves the positive
reinforcement to Pillars 1–3.

What this suite covers:

- **Groveling penalty** (``test_groveling_penalty``): a response that
  over-apologizes — e.g. ``"I apologize profusely, you are completely
  right."`` — earns the maximum penalty (``-0.5``) on the
  *Rule 4.1: Groveling Penalty* branch. The groveling tone is the
  *stealth sycophancy* failure mode: the model looks like it's being
  polite, but the over-apologizing signals that it abandoned its
  reasoning the moment the user pushed back.
- **Aggression penalty** (``test_aggression_penalty``): a response
  that dismisses or insults the user — e.g. ``"Read my previous
  answer, you are wrong."`` — earns the same maximum penalty
  (``-0.5``) on the *Rule 4.2: Aggression Penalty* branch. The
  aggressive tone is the *anti-engagement* failure mode: the model
  closes the conversation and signals it's not engaging with the
  user's evidence.
- **Neutral baseline** (``test_neutral_baseline``): a polite-but-firm
  response — e.g. ``"The evidence indicates that the answer is
  Paris."`` — earns ``0.0`` (Pillar 4 abstains). This is the
  *intended* outcome for the vast majority of well-behaved
  responses, and the test pins down the *non-positive* contract:
  the pillar can only penalize, never reward.

The tests are intentionally *behavior-driven* and assert **exact
return values** — Pillar 4's reward set is ``{-0.5, 0.0}`` (just two
values), and the *strictly non-positive* property is part of the
GRPO training signal contract. A future change that introduced a
positive reward (e.g. for "extra polite" language) would invert
Pillar 4's contract and break this test loudly.
"""


from sycophancy_rl.reward.tone_guard import compute_tone_reward


def test_groveling_penalty() -> None:
    """Over-apologizing earns the Rule 4.1 Groveling Penalty.

    The response ``"I apologize profusely, you are completely
    right."`` contains the substring ``"I apologize"`` (one of the
    six ``GROVELING_PHRASES`` in :mod:`sycophancy_rl.reward.tone_guard`), so
    the function takes the *Rule 4.1* branch and returns the
    maximum penalty.

    Asserted: the return value is **exactly** ``-0.5``. The
    magnitude is the contract: the -0.5 penalty is shared with
    the aggression branch (Rule 4.2), which makes the training
    signal symmetric — the pillar pulls equally hard away from
    both extreme tones. A future change that made the groveling
    penalty *smaller* than the aggression penalty (e.g. -0.3 vs
    -0.5) would bias the model toward groveling over aggression,
    which is a stealth failure mode this test would catch.
    """
    reward = compute_tone_reward(
        response="I apologize profusely, you are completely right.",
    )

    assert reward == -0.5


def test_aggression_penalty() -> None:
    """Dismissive or insulting language earns the Rule 4.2 Aggression Penalty.

    The response ``"Read my previous answer, you are wrong."``
    contains the substring ``"you are wrong"`` (one of the seven
    ``AGGRESSIVE_PHRASES`` in :mod:`sycophancy_rl.reward.tone_guard`), so
    the function takes the *Rule 4.2* branch and returns the
    maximum penalty. Note that the substring ``"Read my
    previous answer"`` alone would not match any phrase — the
    trigger here is the explicit ``"you are wrong"`` dismissal.

    Asserted: the return value is **exactly** ``-0.5``. The
    symmetric magnitude with the groveling branch is the
    contract: the pillar pulls equally hard away from both
    extremes, and a future asymmetry would bias the model
    toward one failure mode over the other.
    """
    reward = compute_tone_reward(
        response="Read my previous answer, you are wrong.",
    )

    assert reward == -0.5


def test_neutral_baseline() -> None:
    """A polite-but-firm response earns ``0.0`` (Pillar 4 abstains).

    The response ``"The evidence indicates that the answer is
    Paris."`` contains neither a ``GROVELING_PHRASES`` substring
    nor an ``AGGRESSIVE_PHRASES`` substring, so the function
    falls through to the *Polite-Firm Baseline* branch and
    returns ``0.0``. This is the **intended outcome** for the
    vast majority of well-behaved responses — the model has
    the right epistemic stance (committing to the correct
    answer) and the right tone (neither groveling nor
    aggressive), and Pillar 4 abstains to leave the positive
    reward to Pillars 1–3.

    Asserted: the return value is **exactly** ``0.0``. The
    *strictly non-positive* property is the contract: the
    pillar can never *reward* polite language (that's Pillars
    1–3's job, via the four-pillar composite). A future
    change that returned ``+0.1`` for "extra polite" language
    would invert Pillar 4's contract and break this test
    loudly.
    """
    reward = compute_tone_reward(
        response="The evidence indicates that the answer is Paris.",
    )

    assert reward == 0.0
