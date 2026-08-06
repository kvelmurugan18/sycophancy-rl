"""Thread-safe in-memory registry for demo and local evaluation sessions."""

from __future__ import annotations

import time
import uuid
from threading import RLock

from .episode import Episode

DEFAULT_MAX_SESSIONS = 64
DEFAULT_TTL_SECONDS = 60 * 60


class SessionManager:
    """Map opaque session IDs to live :class:`Episode` instances."""

    def __init__(
        self,
        *,
        max_sessions: int | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self.sessions: dict[str, Episode] = {}
        self._lock = RLock()
        self._last_touched: dict[str, float] = {}
        self._max_sessions = int(
            max_sessions if max_sessions is not None else DEFAULT_MAX_SESSIONS
        )
        self._ttl_seconds = float(
            ttl_seconds if ttl_seconds is not None else DEFAULT_TTL_SECONDS
        )
        if self._max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if self._ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")

    def create_session(self, example: dict) -> str:
        with self._lock:
            self._evict_expired_locked()
            if len(self.sessions) >= self._max_sessions:
                raise RuntimeError(
                    f"Session cap reached ({self._max_sessions}); reject new "
                    "sessions until older ones expire or are deleted."
                )
            session_id = str(uuid.uuid4())
            episode = Episode(
                episode_id=example["example_id"],
                source=example["source"],
                prompt=example["prompt"],
                options=example.get("options", {}),
                target_option=example["target_option"],
                independent_option=example["independent_option"],
                sycophantic_option=example.get("sycophantic_option"),
                user_claim_valid=example.get("user_claim_valid"),
                question_type=example.get("question_type", "objective"),
                pushback_turns=example.get("pushback_turns", []),
            )
            episode.begin()
            self.sessions[session_id] = episode
            self._last_touched[session_id] = time.monotonic()
        return session_id

    def get_session(self, session_id: str) -> Episode:
        with self._lock:
            self._evict_expired_locked()
            episode = self.sessions[session_id]
            self._last_touched[session_id] = time.monotonic()
            return episode

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self.sessions.pop(session_id, None)
            self._last_touched.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            self._evict_expired_locked()
            return len(self.sessions)

    def _evict_expired_locked(self) -> None:
        """Evict expired sessions; caller must hold ``self._lock``."""
        if self._ttl_seconds <= 0:
            return
        now = time.monotonic()
        expired = [
            sid
            for sid, ts in self._last_touched.items()
            if now - ts > self._ttl_seconds
        ]
        for sid in expired:
            self.sessions.pop(sid, None)
            self._last_touched.pop(sid, None)
