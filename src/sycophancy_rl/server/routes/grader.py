"""
Grader endpoint for the Sycophancy RL Environment server.

This module exposes ``POST /grader``, the **terminal endpoint of an RL
episode**. Once the model has answered every pushback turn in the
episode, a client hits this route to receive the final aggregated
grade: total reward, per-turn component scores, and a coarse pass /
fail verdict.

The route is intentionally **stateless on the input side** — it ignores
``payload.trajectory`` entirely and reads the per-turn rewards from the
:class:`Episode` already living in the :class:`SessionManager`. The
trajectory the client submits is never trusted: the server uses its own
authoritative internal memory (``Episode.trajectory_scores``, populated
turn-by-turn by ``/step``) so a malicious or buggy client cannot
inflate the grade by submitting a fabricated conversation log. The
"ground truth" is whatever the grader actually saw during the live
multi-turn exchange, not whatever the client claims it saw.

All cross-turn math — flip-flop penalties, multi-turn bonuses, decay —
is the job of :func:`sycophancy_rl.environment.grader.grade_episode`, which the
route delegates to verbatim. The route itself only handles
session-lookup, done-ness checking, and the response shape.
"""

from fastapi import APIRouter, HTTPException, Request

from sycophancy_rl.environment.grader import grade_episode
from sycophancy_rl.server.schemas import GraderRequest, GraderResponse

router = APIRouter(tags=["Grader"])


@router.post("/grader", response_model=GraderResponse)
def fetch_episode_grade(payload: GraderRequest, request: Request):
    """Return the final aggregated grade for a completed episode.

    Looks up the live :class:`Episode` from the :class:`SessionManager`,
    enforces that the episode is fully consumed (``is_done()`` is
    ``True``), and delegates the actual aggregation to
    :func:`grade_episode`. The ``payload.trajectory`` field is
    intentionally **ignored** — see the module docstring for the
    anti-cheating rationale.

    Args:
        payload: :class:`GraderRequest` body carrying the
            ``session_id`` returned by the matching ``/reset`` call
            (and a ``trajectory`` field, which the route does not
            consult).
        request: The incoming FastAPI request, used to reach
            ``request.app.state.session_manager``.

    Returns:
        dict: A mapping FastAPI validates against :class:`GraderResponse`,
        with ``session_id`` (echoed back), ``total_reward`` (sum of
        ``Episode.trajectory_scores``), ``component_scores`` (shallow
        copy of the per-turn score list, in turn order), and ``passed``
        (true only when every recorded turn has a strictly positive score).

    Raises:
        HTTPException: ``404`` if the ``session_id`` is unknown
            (``SessionManager.get_session`` raised :class:`KeyError`).
            ``400`` if the episode is not yet finished
            (``Episode.is_done()`` is ``False``).
    """
    sm = request.app.state.session_manager

    # --- 1. Fetch the live episode. ---
    # Same KeyError → 404 translation as /step: an unknown session_id
    # means the client is holding a stale or never-was-valid handle.
    try:
        active_episode = sm.get_session(payload.session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown session_id '{payload.session_id}'. Call /reset "
                "to start a new episode."
            ),
        ) from exc

    # --- 2. Reject calls against an unfinished episode. ---
    # Grading a partial trajectory would be misleading (a one-turn
    # "hold firm" reply can look positive before later failures, which
    # isn't what pass / fail is meant to communicate). Enforce the
    # done-ness check at the route layer; grade_episode deliberately
    # does not consult is_done() itself (see its docstring).
    if not active_episode.is_done():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Episode '{active_episode.episode_id}' is not finished "
                f"yet (current_turn={active_episode.current_turn} / "
                f"{len(active_episode.pushback_turns)}). Submit "
                "/step for every remaining turn before calling /grader."
            ),
        )

    # --- 3. Delegate the aggregation. ---
    # payload.trajectory is intentionally NOT passed in — the server's
    # internal Episode.trajectory_scores (written by /step) is the
    # only authoritative source. This prevents a client from
    # manufacturing a "perfect" trajectory to inflate its own grade.
    result = grade_episode(active_episode)

    return {
        "session_id": payload.session_id,
        "total_reward": result["total_reward"],
        "component_scores": result["component_scores"],
        "passed": result["passed"],
    }
