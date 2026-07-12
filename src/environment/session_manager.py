"""In-memory session registry for active RL episodes.

This module acts as the in-memory database for active RL episodes, allowing
the FastAPI server to track multiple concurrent multi-turn conversations
safely. Each :class:`SessionManager` instance is owned by a single server
process; the registry is keyed by a UUID4 ``session_id`` (returned to the
client on ``/reset``) and maps to the :class:`Episode` state object the
``/step`` and ``/grader`` routes mutate. Because the store is a plain
in-process dict, sessions are scoped to one Python process — horizontal
scaling (multiple uvicorn workers) would require swapping this for a shared
backend such as Redis, but for the single-worker FastAPI server that runs
this environment, the dict is sufficient and the simplest correct option.
"""

import uuid
from typing import Dict

from .episode import Episode


class SessionManager:
    """Registry mapping ``session_id`` strings to live :class:`Episode` objects.

    The manager is intentionally minimal: it owns the lifecycle of
    :class:`Episode` instances (create / lookup / delete) and nothing else.
    All per-turn mutation — appending history, advancing ``current_turn``,
    recording ``trajectory_scores`` — happens on the :class:`Episode` itself
    via :meth:`Episode.advance_turn`; the manager just hands the right
    episode to whichever route is asking.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory session registry."""
        self.sessions: Dict[str, Episode] = {}

    def create_session(self, episode_dict: dict) -> str:
        """Create a new :class:`Episode` from ``episode_dict`` and register it.

        The episode dict is expected to carry the unified schema produced by
        ``src/data_prep/merge_datasets.py``: ``prompt``, ``correct_answer``,
        ``wrong_answer``, ``source``, and ``pushback_turns``. The
        ``episode_id`` field on the resulting :class:`Episode` is overwritten
        with a freshly generated UUID4 so the registry key and the episode's
        own identifier stay in sync.

        Args:
            episode_dict: Mapping with the unified episode fields
                (``prompt``, ``correct_answer``, ``wrong_answer``, ``source``,
                ``pushback_turns``). Any additional keys are accepted but
                ignored by the :class:`Episode` constructor.

        Returns:
            The newly generated UUID4 string, which doubles as the
            ``session_id`` the client uses to refer to this episode on
            subsequent ``/step`` and ``/grader`` calls.
        """
        session_id = str(uuid.uuid4())
        episode = Episode(
            episode_id=session_id,
            prompt=episode_dict["prompt"],
            correct_answer=episode_dict["correct_answer"],
            wrong_answer=episode_dict["wrong_answer"],
            source=episode_dict["source"],
            pushback_turns=episode_dict["pushback_turns"],
        )
        self.sessions[session_id] = episode
        return session_id

    def get_session(self, session_id: str) -> Episode:
        """Return the :class:`Episode` registered under ``session_id``.

        Args:
            session_id: The UUID4 string returned by :meth:`create_session`.

        Returns:
            The live :class:`Episode` instance for this session.

        Raises:
            KeyError: If no session is registered under ``session_id`` —
                either the id is wrong or the session was already deleted.
        """
        return self.sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        """Remove ``session_id`` from the registry if it exists.

        Missing keys are silently ignored: the call is idempotent so the
        client can safely retry a ``/reset`` or a server-side cleanup
        without a 404. Any subsequent :meth:`get_session` for the same id
        will of course raise :class:`KeyError`.

        Args:
            session_id: The UUID4 string returned by :meth:`create_session`.
        """
        self.sessions.pop(session_id, None)
