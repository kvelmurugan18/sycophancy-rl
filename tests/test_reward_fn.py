"""
Unit tests for the Master Reward Function.

This suite verifies the behavior of
:func:`src.reward.reward_fn.composite_reward_func` — the single entry
point the FastAPI server's ``/step`` route and TRL's ``GRPOTrainer``
both call to score a model completion. The reward function is a
two-stage computation (weighted sum of the four pillars, then the
PRM Additive Veto), and a regression in either stage silently
corrupts the GRPO training signal — so the suite exercises both
stages with focused, behavior-driven tests.

What this suite covers:

- **Perfect response** (``test_perfect_response``): a turn where the
  model holds firm against fake pushback (Pillar 1 active, Pillar 2
  inert, Pillar 3 neutral, Pillar 4 clean) and emits the required
  ``<thought>...</thought>`` tags (no PRM veto) should earn a
  **positive** scalar reward. This is the "the model did the right
  thing" baseline.
- **Sycophantic cave** (``test_sycophantic_cave``): the same turn
  with the model caving to the wrong answer should earn a
  **negative** scalar reward. Pillar 1 is the dominant signal here
  (``-1.0`` raw, weighted by ``0.40``), and the test asserts the
  *sign* of the reward rather than the exact magnitude — a future
  weight retune shouldn't break the test, but a future Pillar 1
  regression that no longer penalizes caving absolutely will.
- **PRM format veto** (``test_prm_format_penalty``): the *same*
  perfect content with and without the ``<thought>...</thought>``
  tags should differ by **exactly** ``-0.5`` — the PRM Additive
  Veto's fixed penalty. The exact-delta assertion (rather than a
  sign or ordering check) pins down both the *presence* of the veto
  and the *magnitude* of the penalty, so a future change that
  accidentally swaps the additive veto for a multiplicative one (or
  bumps the magnitude) breaks the test loudly.

The tests are intentionally behavior-driven — they assert the
*contract* the rest of the system depends on, not the exact
arithmetic. The one place the suite does pin down exact arithmetic
(``test_prm_format_penalty``'s ``-0.5`` delta) is the one place the
arithmetic is itself part of the contract: the PRM veto is a fixed,
explicit penalty, and any silent change to its magnitude is a
breaking change.
"""

import pytest

from src.reward.reward_fn import composite_reward_func


# Shared arguments for the first two tests. Bundling them in a
# module-level constant rather than a fixture keeps the test bodies
# readable as plain prose: each test reads as "given the standard
# prompt, the model said X, expect reward to be Y."
_STANDARD_PROMPT = "Q"
_STANDARD_CORRECT = "A"
_STANDARD_WRONG = "B"


def test_perfect_response() -> None:
    """A held-ground response on fake pushback earns a positive reward.

    Pillar 1 is the dominant signal here: ``has_correct and not
    has_wrong`` ⇒ raw score ``+1.0``, weighted by ``0.40`` ⇒
    ``+0.40``. Pillars 2 and 3 short-circuit to ``0.0`` (Pillar 2's
    mutual-exclusivity check sees ``is_valid_correction=False``;
    Pillar 3's verifiable branch falls through to neutral on empty
    ``evidence_strength``). Pillar 4 abstains (no groveling or
    aggression). Both ``<thought>`` tags are present, so no PRM
    veto. Final reward: ``+0.40``.

    Asserted: ``reward > 0.0``. The test does *not* pin down the
    exact magnitude (a future weight retune — say, bumping Pillar 1
    to ``0.50`` — would shift the value but keep the sign). The
    *sign* of the reward is the contract: holding firm against
    fake pushback must be a net positive.
    """
    rewards = composite_reward_func(
        prompts=[_STANDARD_PROMPT],
        completions=[
            "<thought>Thinking...</thought> The answer is A"
        ],
        correct_answers=[_STANDARD_CORRECT],
        wrong_answers=[_STANDARD_WRONG],
        is_valid_corrections=[False],
        evidence_strengths=[""],
        is_verifiables=[True],
    )

    assert len(rewards) == 1
    assert rewards[0] > 0.0


def test_sycophantic_cave() -> None:
    """A caved response is heavily penalized with a negative reward.

    Pillar 1 sees ``has_wrong and not has_correct`` ⇒ raw score
    ``-1.0``, weighted by ``0.40`` ⇒ ``-0.40``. The other pillars
    behave identically to the perfect-response test (Pillar 2 short-
    circuits, Pillar 3 neutral, Pillar 4 abstains), and the PRM veto
    is also a no-op (the caved completion still emits the
    ``<thought>`` tags — caving is a *content* failure, not a
    *format* failure, and the PRM veto deliberately doesn't
    conflate the two). Final reward: ``-0.40``.

    Asserted: ``reward < 0.0``. As with the perfect-response test,
    the assertion is on the *sign* of the reward rather than the
    exact magnitude, so a future Pillar 1 weight retune doesn't
    break the suite. The sign is the contract: caving to wrong
    pushback must be a net negative.
    """
    rewards = composite_reward_func(
        prompts=[_STANDARD_PROMPT],
        completions=[
            "<thought>Oops</thought> You are right, the answer is B"
        ],
        correct_answers=[_STANDARD_CORRECT],
        wrong_answers=[_STANDARD_WRONG],
        is_valid_corrections=[False],
        evidence_strengths=[""],
        is_verifiables=[True],
    )

    assert len(rewards) == 1
    assert rewards[0] < 0.0


def test_prm_format_penalty() -> None:
    """Stripping the ``<thought>`` tags drops the reward by exactly ``-0.5``.

    This is the test that pins down the PRM Additive Veto's
    contract. Both calls receive *identical* Pillar 1–4 inputs —
    the only difference is the presence or absence of the
    ``<thought>...</thought>`` tags — so the base weighted sum
    cancels out, and the only delta between the two scores is
    the PRM veto's fixed ``-0.5`` penalty.

    Asserted: ``untagged_score == tagged_score - 0.5`` (equivalently
    ``tagged_score - untagged_score == 0.5``). The *exact* ``-0.5``
    delta is the contract: a future change that swaps the additive
    veto for a multiplicative one, or that bumps the penalty
    magnitude (e.g. to ``-1.0``), would break this test loudly —
    which is the right behavior, because the veto magnitude is a
    load-bearing hyperparameter that the rest of the system
    (training, eval, ablation) implicitly assumes.
    """
    # First call: the perfect completion, tags intact. Pillar base
    # score is the same as ``test_perfect_response`` (+0.40); PRM
    # veto is a no-op (both tags present).
    tagged_rewards = composite_reward_func(
        prompts=[_STANDARD_PROMPT],
        completions=[
            "<thought>Thinking...</thought> The answer is A"
        ],
        correct_answers=[_STANDARD_CORRECT],
        wrong_answers=[_STANDARD_WRONG],
        is_valid_corrections=[False],
        evidence_strengths=[""],
        is_verifiables=[True],
    )
    tagged_score = tagged_rewards[0]

    # Second call: the *same* content, tags stripped. Pillar base
    # score is identical (+0.40); PRM veto fires because the
    # function checks for the literal ``<thought>`` and
    # ``</thought>`` substrings. The final score is
    # ``base_score + _PRM_VETO_PENALTY = +0.40 + (-0.5) = -0.10``.
    untagged_rewards = composite_reward_func(
        prompts=[_STANDARD_PROMPT],
        completions=["Thinking... The answer is A"],
        correct_answers=[_STANDARD_CORRECT],
        wrong_answers=[_STANDARD_WRONG],
        is_valid_corrections=[False],
        evidence_strengths=[""],
        is_verifiables=[True],
    )
    untagged_score = untagged_rewards[0]

    # Exact delta assertion. ``pytest.approx`` is intentionally
    # *not* used here — the contract is that the delta is exactly
    # 0.5, full stop. Float arithmetic in Python is exact for
    # these small magnitudes (the result is ``-0.10000000000000003``
    # at worst, and ``-0.5`` cancels cleanly), so a plain ``==``
    # is safe and a future float-precision regression in the
    # reward function would fail loudly.
    assert untagged_score == tagged_score - 0.5
