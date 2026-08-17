"""Advance one correctly aligned turn of an environment episode."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from sycophancy_rl.environment.store import EpisodeStoreError
from sycophancy_rl.reward.reward_fn import score_completion
from sycophancy_rl.server.schemas import StepRequest, StepResponse

router = APIRouter(tags=["Step"])


@router.post("/step", response_model=StepResponse)
def step_environment(payload: StepRequest, request: Request):
    session_manager = request.app.state.session_manager

    def advance(episode):
        if episode.is_done():
            raise HTTPException(status_code=400, detail="Episode is already finished.")
        context = episode.get_current_context()
        breakdown = score_completion(
            payload.response,
            target_option=context["target_option"],
            independent_option=context["independent_option"],
            sycophantic_option=context["sycophantic_option"],
            user_preferred_option=context["user_preferred_option"],
            user_claim_valid=context["user_claim_valid"],
            behavior_target=context["behavior_target"],
            prompt=episode.history,
            options=episode.options,
            finish_reason=payload.finish_reason,
        )
        episode.advance_turn(
            payload.response,
            breakdown.total,
            reward_breakdown=breakdown.to_dict(),
        )
        return breakdown

    try:
        episode, breakdown = session_manager.update_session(payload.session_id, advance)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session_id.") from exc
    except EpisodeStoreError as exc:
        raise HTTPException(status_code=503, detail="Session storage is unavailable.") from exc
    return {
        "session_id": payload.session_id,
        "next_prompt": episode.get_current_user_prompt(),
        "reward": breakdown.total,
        "done": episode.is_done(),
        "turn_number": episode.current_turn,
        "reward_breakdown": breakdown.to_dict(),
    }
