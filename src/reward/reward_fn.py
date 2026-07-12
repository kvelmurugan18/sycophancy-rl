"""Master Reward Function — composite of the four reward pillars.

This module is the single entry point the TRL ``GRPOTrainer`` calls to
score a batch of model completions. It stitches the four pillars into
one reward signal using a two-step recipe:

1. **Weighted Sum (base score).** Each pillar returns a raw score in
   its own bounded set. We multiply by the pillar's weight and sum
   them. The weights are the project defaults:

       Pillar 1 (Correctness / Anti-Evasion)   × 0.40
       Pillar 2 (Legitimate Update)            × 0.25
       Pillar 3 (Epistemic Calibration)        × 0.20
       Pillar 4 (Tone Guard, non-positive)      × 0.15

2. **PRM Additive Veto (format guard).** We *additively* subtract a
   fixed penalty (``-0.5``) from the base score if the completion
   does not contain both ``<thought>`` and ``</thought>`` tags. The
   veto is **additive** (not multiplicative) so it cannot zero out a
   perfectly-correct, well-calibrated response that simply forgot to
   format — it just pulls it below the bar.

The veto is what turns the reward into a Process Reward Model: the
model is forced into a "System 2" style of reasoning that explicitly
deliberates inside a ``<thought>…</thought>`` block before producing
its final answer. Completions that skip the deliberation step are
penalised on top of whatever the four pillars said about their
content.

Note on scope: this function grades a *single turn*. Cross-turn
trajectory math (e.g. accumulating reward across pushback turns in a
multi-turn episode) is handled by the episode-loop wrapper that
calls into this function, not here.
"""

# Pillar 1: truthfulness & resistance to fake pushback.
from .correctness import compute_correctness_reward
# Pillar 2: evidence-based updating under real corrections.
from .legitimate_update import compute_evidence_reward
# Pillar 3: matching expressed confidence to the question's evidence.
from .calibration import compute_calibration_reward
# Pillar 4: tone extremes (strictly non-positive guard).
from .tone_guard import compute_tone_reward

# Pillar weights. Sum to 1.0; each is applied as a scalar multiplier
# to its pillar's raw score before summing.
_WEIGHT_P1_CORRECTNESS: float = 0.40
_WEIGHT_P2_LEGITIMATE_UPDATE: float = 0.25
_WEIGHT_P3_CALIBRATION: float = 0.20
_WEIGHT_P4_TONE_GUARD: float = 0.15

# PRM Additive Veto: a fixed additive penalty applied when the model
# fails to wrap its deliberation in matching <thought>...</thought>
# tags. Additive (not multiplicative) so a single good pillar outcome
# can still partially recover the score.
_PRM_VETO_PENALTY: float = -0.5

# Delimiters the model is required to emit. Stored as module-level
# constants so the substring check is consistent and easy to retarget
# (e.g. switch to ``<reasoning>`` later) without hunting through the
# function body.
_THOUGHT_OPEN: str = "<thought>"
_THOUGHT_CLOSE: str = "</thought>"


def composite_reward_func(
    prompts: list[str],
    completions: list[str],
    correct_answers: list[str],
    wrong_answers: list[str],
    is_valid_corrections: list[bool],
    evidence_strengths: list[str],
    is_verifiables: list[bool],
    **kwargs: object,
) -> list[float]:
    """Score a batch of completions by combining the four reward pillars.

    The signature is fixed by TRL's ``GRPOTrainer``: every named list
    argument is one column of the dataset, aligned positionally across
    the batch, and ``**kwargs`` swallows any extra dataset columns the
    trainer chooses to forward (e.g. ``episode_id``, ``turn_number``).
    All input lists must therefore be the same length; mismatched
    lengths will surface as a ``ValueError`` from the ``zip`` call
    below.

    Per-completion scoring:
        1. Compute the four raw pillar scores.
        2. ``base_score`` = weighted sum of the four raw scores.
        3. If the completion is missing either ``<thought>`` or
           ``</thought>``, apply the PRM Additive Veto
           (``final_score = base_score + PRM_VETO_PENALTY``).
        4. Otherwise ``final_score = base_score``.

    Pillars 1 and 2 are *mutually exclusive* by construction (their
    mutual-exclusivity checks short-circuit to ``0.0`` on the
    off-turn), so the weighted sum never double-counts a single turn
    across the truthfulness and correction regimes. Pillars 3 and 4
    always run, since calibration and tone are properties of the
    response itself, not of the turn type.

    Args:
        prompts: The user-side prompt for each turn (not directly
            graded here, but required by the TRL signature).
        completions: The model's full response text for each turn.
            This is the field the four pillars grade.
        correct_answers: Ground-truth answer for each turn (Pillar 1
            and Pillar 2 input).
        wrong_answers: The foil / pushback answer for each turn
            (Pillar 1 and Pillar 2 input).
        is_valid_corrections: Per-turn flag indicating whether the
            turn is a real correction. Pillar 1 and Pillar 2 use
            this to hand off to each other.
        evidence_strengths: Per-turn latent tag (``"strong"`` /
            ``"weak"`` / ``""``) consumed by Pillar 2 and Pillar 3.
        is_verifiables: Per-turn flag consumed by Pillar 1 (evasion
            penalty) and Pillar 3 (unverifiable branch).
        **kwargs: Absorbs any extra columns the trainer forwards;
            unused inside the function but required for forward
            compatibility with future dataset schemas.

    Returns:
        A list of float rewards, one per completion, in the same
        order as the input lists. Each value is the base weighted
        sum, possibly with the additive PRM veto applied.
    """
    rewards: list[float] = []

    # zip stops at the shortest input, so a length mismatch surfaces
    # as silent truncation. We rely on the caller (the GRPO trainer)
    # to pass aligned columns; the type hints make the contract
    # explicit on the trainer side.
    for (
        _prompt,
        completion,
        correct_answer,
        wrong_answer,
        is_valid_correction,
        evidence_strength,
        is_verifiable,
    ) in zip(
        prompts,
        completions,
        correct_answers,
        wrong_answers,
        is_valid_corrections,
        evidence_strengths,
        is_verifiables,
    ):
        # --- Step 1: raw pillar scores. ---
        p1_score = compute_correctness_reward(
            response=completion,
            correct_answer=correct_answer,
            wrong_answer=wrong_answer,
            is_valid_correction=is_valid_correction,
            is_verifiable=is_verifiable,
        )
        p2_score = compute_evidence_reward(
            response=completion,
            correct_answer=correct_answer,
            wrong_answer=wrong_answer,
            is_valid_correction=is_valid_correction,
            evidence_strength=evidence_strength,
        )
        p3_score = compute_calibration_reward(
            response=completion,
            is_verifiable=is_verifiable,
            evidence_strength=evidence_strength,
        )
        p4_score = compute_tone_reward(response=completion)

        # --- Step 2: weighted base score. ---
        base_score = (
            p1_score * _WEIGHT_P1_CORRECTNESS
            + p2_score * _WEIGHT_P2_LEGITIMATE_UPDATE
            + p3_score * _WEIGHT_P3_CALIBRATION
            + p4_score * _WEIGHT_P4_TONE_GUARD
        )

        # --- Step 3: PRM Additive Veto (System 2 deliberation guard). ---
        if _THOUGHT_OPEN not in completion or _THOUGHT_CLOSE not in completion:
            final_score = base_score + _PRM_VETO_PENALTY
        else:
            final_score = base_score

        rewards.append(final_score)

    return rewards
