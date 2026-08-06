"""Start deterministic or randomly sampled environment episodes."""

from __future__ import annotations

import random

from fastapi import APIRouter, HTTPException, Request

from sycophancy_rl.server.schemas import ResetRequest, ResetResponse

router = APIRouter(tags=["Reset"])


def _load_episodes(request: Request) -> list[dict]:
    try:
        return list(request.app.state.episode_store.load())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Invalid episode dataset: {exc}") from exc


@router.post("/reset", response_model=ResetResponse)
def reset_environment(payload: ResetRequest, request: Request):
    examples = _load_episodes(request)
    if payload.category is not None:
        examples = [
            example
            for example in examples
            if example.get("topic") == payload.category
            or example.get("source") == payload.category
        ]
    if payload.episode_id is not None:
        examples = [
            example for example in examples if example["example_id"] == payload.episode_id
        ]
    if not examples:
        raise HTTPException(status_code=404, detail="No matching episode was found.")

    if payload.episode_id is not None:
        example = examples[0]
    elif payload.seed is not None:
        example = random.Random(payload.seed).choice(examples)
    else:
        example = random.choice(examples)

    session_manager = request.app.state.session_manager
    session_id = session_manager.create_session(example)
    episode = session_manager.get_session(session_id)
    return {
        "session_id": session_id,
        "episode_id": episode.episode_id,
        "prompt": episode.get_current_user_prompt(),
        "turn_number": episode.current_turn,
        # Gold/behavior labels are intentionally not exposed to clients.
        "metadata": {
            "source": episode.source,
            "question_type": episode.question_type,
            "total_turns": episode.total_turns,
        },
    }
