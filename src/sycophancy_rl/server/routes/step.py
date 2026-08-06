"""Advance one correctly aligned turn of an environment episode."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from sycophancy_rl.reward.reward_fn import score_completion
from sycophancy_rl.server.schemas import StepRequest, StepResponse

router = APIRouter(tags=["Step"])


@router.post("/step", response_model=StepResponse)
def step_environment(payload: StepRequest, request: Request):
    session_manager = request.app.state.session_manager
    try:
        episode = session_manager.get_session(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session_id.") from exc
    if episode.is_done():
        raise HTTPException(status_code=400, detail="Episode is already finished.")
    if not payload.response.strip():
        raise HTTPException(status_code=422, detail="response must not be empty.")

    context = episode.get_current_context()
    breakdown = score_completion(
        payload.response,
        target_option=context["target_option"],
        independent_option=context["independent_option"],
        sycophantic_option=context["sycophantic_option"],
        prompt=episode.history,
        options=episode.options,
    )
    episode.advance_turn(
        payload.response,
        breakdown.total,
        reward_breakdown=breakdown.to_dict(),
    )
    return {
        "session_id": payload.session_id,
        "next_prompt": episode.get_current_user_prompt(),
        "reward": breakdown.total,
        "done": episode.is_done(),
        "turn_number": episode.current_turn,
        "reward_breakdown": breakdown.to_dict(),
    }
