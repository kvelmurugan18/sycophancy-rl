"""
Reset endpoint for the Sycophancy RL Environment server.

This module exposes ``POST /reset``, the **starting line for an RL
episode**. A client (an RL trainer, a Gradio demo, or a manual
``curl``) hits this route to spin up a fresh multi-turn conversation:

1. The route reads the processed dataset (``data/processed/merged_episodes.jsonl``
   produced by ``src/data_prep/merge_datasets.py``) and picks one episode.
2. Optionally narrows the pick to a single ``source`` category
   (``"truthfulqa"`` / ``"sycophancy_eval"`` / ``"custom_pushback"``).
3. Hands the episode dict to the global :class:`SessionManager`, which
   generates a fresh ``session_id`` and registers the corresponding
   :class:`Episode` state object in memory.
4. Returns the opening ``prompt`` plus the ``session_id`` the client must
   echo back on every subsequent ``/step`` and ``/grader`` call.

The route is read-only against the dataset file (it loads the whole JSONL
on every call — fine at the demo's data scale) and only mutates the
in-memory session registry, never the file system. No business logic
beyond "pick an episode and check it in" lives here.
"""

import json
import random
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from server.schemas import ResetRequest, ResetResponse

router = APIRouter(tags=["Reset"])


def _load_episodes() -> list[dict]:
    """Load the merged-episode dataset from disk into a list of dicts.

    Reads ``data/processed/merged_episodes.jsonl`` line-by-line, parsing
    each non-blank line as JSON. Returns the list in file order — the
    caller is responsible for any shuffling / sampling.

    Returns:
        list[dict]: One dict per line, in the unified schema produced by
        :mod:`src.data_prep.merge_datasets` (``prompt``,
        ``correct_answer``, ``wrong_answer``, ``source``,
        ``pushback_turns``).

    Raises:
        HTTPException: ``500`` if the file is missing or the working
            directory has not been set to the project root.
    """
    dataset_path = Path("data/processed/merged_episodes.jsonl")
    if not dataset_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Episode dataset not found at {dataset_path}. Run "
                "`python -m src.data_prep.merge_datasets` to generate it."
            ),
        )
    with dataset_path.open("r", encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    return episodes


@router.post("/reset", response_model=ResetResponse)
def reset_environment(payload: ResetRequest, request: Request):
    """Start a new RL episode and return its opening prompt.

    Picks a random episode (optionally filtered by ``payload.category``),
    registers it in the :class:`SessionManager` under a fresh UUID, and
    returns the opening ``prompt`` and the ``session_id`` the client
    needs for subsequent calls.

    Args:
        payload: :class:`ResetRequest` body. ``episode_id`` is currently
            advisory-only (the dataset has no lookup index); ``category``
            filters the candidate pool by the ``source`` field.
        request: The incoming FastAPI request, used to reach
            ``request.app.state.session_manager``.

    Returns:
        dict: A mapping FastAPI validates against :class:`ResetResponse`,
        with ``session_id`` (the new UUID), ``prompt`` (the opening user
        message), ``turn_number`` (always ``0`` for a fresh episode), and
        ``metadata`` carrying ``source`` / ``correct_answer`` /
        ``wrong_answer`` for client-side logging.

    Raises:
        HTTPException: ``500`` if the dataset file is missing
            (raised from :func:`_load_episodes`). ``400`` if a category
            filter was supplied but matched no episodes.
    """
    episodes = _load_episodes()

    # Optional category filter — narrow the candidate pool before sampling
    # so a client asking for "truthfulqa" only ever gets TruthfulQA prompts.
    if payload.category is not None:
        episodes = [e for e in episodes if e.get("source") == payload.category]
        if not episodes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No episodes found for category '{payload.category}'. "
                    "Check the spelling or run merge_datasets with that "
                    "source included."
                ),
            )

    episode = random.choice(episodes)

    sm = request.app.state.session_manager
    session_id = sm.create_session(episode)
    active_episode = sm.get_session(session_id)

    return {
        "session_id": session_id,
        "prompt": active_episode.prompt,
        "turn_number": active_episode.current_turn,
        "metadata": {
            "source": active_episode.source,
            "correct_answer": active_episode.correct_answer,
            "wrong_answer": active_episode.wrong_answer,
        },
    }
