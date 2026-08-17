"""Thread-safe session coordination backed by durable episode storage."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from time import monotonic, time
from typing import Any

from .episode import Episode
from .store import EpisodeStore, SQLiteEpisodeStore

DEFAULT_MAX_SESSIONS = 64
DEFAULT_TTL_SECONDS = 60 * 60


class SessionManager:
    """Coordinate episode operations; persistent storage is authoritative."""

    def __init__(
        self,
        *,
        store: EpisodeStore | None = None,
        database_path: str | Path | None = None,
        max_sessions: int | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        if store is not None and database_path is not None:
            raise ValueError("Pass store or database_path, not both")
        if store is None:
            # create_app supplies its durable application path. A temporary
            # default prevents unrelated standalone managers/tests sharing data.
            if database_path is None:
                runtime_dir = Path.cwd() / ".syco-tmp"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                path = runtime_dir / f"sessions-{uuid.uuid4()}.sqlite3"
            else:
                path = Path(database_path)
            store = SQLiteEpisodeStore(path)
        self.store = store
        self._lock = RLock()
        self._protected_until: dict[str, float] = {}
        self._max_sessions = int(max_sessions or DEFAULT_MAX_SESSIONS)
        self._ttl_seconds = float(
            ttl_seconds if ttl_seconds is not None else DEFAULT_TTL_SECONDS
        )
        if self._max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if self._ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")

    def create_session(self, example: dict[str, Any], *, session_id: str | None = None) -> str:
        with self._lock:
            self._evict_expired_locked()
            if self.store.count() >= self._max_sessions:
                raise RuntimeError(
                    f"Session cap reached ({self._max_sessions}); reject new sessions "
                    "until older ones expire or are deleted."
                )
            identifier = session_id or str(uuid.uuid4())
            episode = Episode(
                episode_id=example["example_id"],
                source=example["source"],
                prompt=example["prompt"],
                options=example.get("options", {}),
                target_option=example["target_option"],
                independent_option=example["independent_option"],
                sycophantic_option=example.get("sycophantic_option"),
                user_preferred_option=example.get("user_preferred_option"),
                user_claim_valid=example.get("user_claim_valid"),
                behavior_target=example.get("behavior_target", "independent_reasoning"),
                question_type=example.get("question_type", "objective"),
                model_identifier=example.get("model_identifier"),
                model_version=example.get("model_version"),
                metadata=example.get("metadata", {}),
                pushback_turns=example.get("pushback_turns", []),
            )
            episode.begin()
            self.store.create(identifier, episode)
            self._protected_until[identifier] = monotonic() + 0.05
            return identifier

    def get_session(self, session_id: str) -> Episode:
        with self._lock:
            self._evict_expired_locked()
            episode, _ = self.store.load(session_id)
            return episode

    def save_session(self, session_id: str, episode: Episode) -> None:
        with self._lock:
            self.store.save(session_id, episode)

    def update_session(
        self,
        session_id: str,
        operation: Callable[[Episode], Any],
    ) -> tuple[Episode, Any]:
        """Load, mutate, and persist one session under the manager lock."""
        with self._lock:
            self._evict_expired_locked()
            episode, _ = self.store.load(session_id)
            result = operation(episode)
            self.store.save(session_id, episode)
            return episode, result

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self.store.delete(session_id)
            self._protected_until.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            self._evict_expired_locked()
            return self.store.count()

    def close(self) -> None:
        self.store.close()

    def _evict_expired_locked(self) -> None:
        if self._ttl_seconds > 0:
            now = monotonic()
            self._protected_until = {
                session_id: deadline
                for session_id, deadline in self._protected_until.items()
                if deadline > now
            }
            self.store.delete_expired(
                time() - self._ttl_seconds,
                tuple(self._protected_until),
            )
