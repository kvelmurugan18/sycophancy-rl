"""
Step endpoint for the Sycophancy RL Environment server.

This module exposes ``POST /step``, the **core loop of the RL
environment**. Each call advances the active multi-turn conversation by
exactly one turn:

1. **Fetch the live episode** from the global :class:`SessionManager`
   using the ``session_id`` the client received from ``/reset``.
2. **Resolve the per-turn grading context.** The first turn (``current_turn == 0``)
   is the model's reply to the neutral initial prompt — Pillar 2 is
   inert (no correction was offered) and Pillar 3 has no evidence tag
   to calibrate against. Subsequent turns are replies to pushback, so
   the ``is_valid_correction`` and ``evidence_strength`` flags are
   pulled from the **previous** pushback dict.
3. **Score the response** by calling :func:`composite_reward_func` with
   single-element lists (the function is batch-shaped for TRL's
   ``GRPOTrainer``; we just give it a one-item batch).
4. **Mutate the episode state** via :meth:`Episode.advance_turn`, which
   appends the assistant turn to ``history``, records the per-turn
   reward in ``trajectory_scores``, and bumps the cursor.
5. **Return the next observation** — either the next pushback message
   (formatted as the next user-role prompt) or ``null`` if the
   episode is finished.

The route is the only writer of per-turn ``trajectory_scores``; the
``/grader`` endpoint only reads the accumulated trajectory.
"""

from fastapi import APIRouter, HTTPException, Request

from server.schemas import StepRequest, StepResponse
from src.reward.reward_fn import composite_reward_func

router = APIRouter(tags=["Step"])


@router.post("/step", response_model=StepResponse)
def step_environment(payload: StepRequest, request: Request):
    """Advance the active episode by one turn and return the next observation.

    The handler implements the classic RL ``(state, action) -> (reward,
    next_state, done)`` transition, specialized for a multi-turn text
    environment. The action is ``payload.response`` (the model's reply
    to whichever pushback the episode was on); the next state is the
    next pushback message (or ``None`` if the episode is finished).

    Args:
        payload: :class:`StepRequest` body carrying the
            ``session_id`` returned by the matching ``/reset`` call and
            the model's ``response`` text for this turn.
        request: The incoming FastAPI request, used to reach
            ``request.app.state.session_manager``.

    Returns:
        dict: A mapping FastAPI validates against :class:`StepResponse`,
        with ``session_id`` (echoed back), ``next_prompt`` (the next
        user message or ``None``), ``reward`` (the scalar composite
        reward for the just-scored turn), ``done`` (whether the episode
        has finished), ``turn_number`` (the turn that was just
        answered, after the cursor was advanced), and
        ``reward_breakdown`` (left empty — the four-pillar breakdown
        is not exposed through the HTTP layer).

    Raises:
        HTTPException: ``404`` if the ``session_id`` is unknown
            (``SessionManager.get_session`` raised :class:`KeyError`).
            ``400`` if the episode is already finished
            (``Episode.is_done()`` is ``True``).
    """
    sm = request.app.state.session_manager

    # --- 1. Fetch the live episode. ---
    try:
        active_episode = sm.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown session_id '{payload.session_id}'. Call /reset "
                "to start a new episode."
            ),
        )

    # --- 2. Reject calls to a finished episode. ---
    if active_episode.is_done():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Episode '{active_episode.episode_id}' is already "
                "finished; no further /step calls are accepted."
            ),
        )

    # --- 3. Resolve the per-turn grading context. ---
    # Turn 0 is the model's reply to the neutral initial prompt — no
    # correction was offered and no evidence tag exists, so Pillar 2
    # short-circuits to 0.0 by construction and Pillar 3 falls through
    # to its neutral branch. From turn 1 onward we read the flags off
    # the *previous* pushback (the one the model just replied to).
    if active_episode.current_turn == 0:
        is_valid_correction: bool = False
        evidence_strength: str = ""
    else:
        previous_pushback = active_episode.pushback_turns[
            active_episode.current_turn - 1
        ]
        is_valid_correction = bool(previous_pushback.get("is_valid_correction", False))
        evidence_strength = str(previous_pushback.get("evidence_strength", ""))

    # is_verifiable defaults to True: every prompt in the current
    # dataset is a fact-style question that can be answered with the
    # correct_answer / wrong_answer pair. A future dataset that adds
    # opinion / undecidable questions would override this per-turn.
    is_verifiable: bool = True

    # --- 4. Score the response. ---
    # composite_reward_func is batch-shaped (TRL's GRPOTrainer requires
    # it); we just give it a one-item batch and unwrap the result.
    reward_list = composite_reward_func(
        prompts=[active_episode.prompt],
        completions=[payload.response],
        correct_answers=[active_episode.correct_answer],
        wrong_answers=[active_episode.wrong_answer],
        is_valid_corrections=[is_valid_correction],
        evidence_strengths=[evidence_strength],
        is_verifiables=[is_verifiable],
    )
    reward: float = float(reward_list[0])

    # --- 5. Mutate the episode state. ---
    active_episode.advance_turn(payload.response, reward)

    # --- 6. Build the next observation. ---
    done = active_episode.is_done()
    if done:
        next_prompt: str | None = None
    else:
        next_pushback = active_episode.get_current_pushback()
        # Pushback dicts in the current dataset are plain strings from
        # generate_pushback.py, so the .get("text", .get("message", ...))
        # chain falls through to the "No text provided" fallback. The
        # explicit two-key lookup future-proofs the route for a schema
        # where each pushback is a {text|message, is_valid_correction,
        # evidence_strength} dict.
        next_prompt = next_pushback.get(
            "text",
            next_pushback.get("message", "No text provided"),
        )

    return {
        "session_id": payload.session_id,
        "next_prompt": next_prompt,
        "reward": reward,
        "done": done,
        "turn_number": active_episode.current_turn,
        "reward_breakdown": {},
    }
