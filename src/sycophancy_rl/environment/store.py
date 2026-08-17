"""Persistent, swappable storage for environment sessions."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from threading import RLock
from time import time

from pydantic import ValidationError

from .episode import Episode


class EpisodeStoreError(RuntimeError):
    """A session could not be read from or written to durable storage."""


class CorruptEpisodeError(EpisodeStoreError):
    """A stored episode failed JSON or schema validation."""


class EpisodeStore(ABC):
    """Storage contract used by :class:`SessionManager`."""

    @abstractmethod
    def create(self, session_id: str, episode: Episode) -> None: ...

    @abstractmethod
    def load(self, session_id: str) -> tuple[Episode, float]: ...

    @abstractmethod
    def save(self, session_id: str, episode: Episode) -> None: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def delete_expired(
        self, cutoff_epoch: float, protected_ids: tuple[str, ...] = ()
    ) -> int: ...

    @abstractmethod
    def close(self) -> None:
        """Release resources; stores without long-lived resources may ignore it."""


class SQLiteEpisodeStore(EpisodeStore):
    """SQLite JSON store with transactions and one connection per operation."""

    def __init__(self, path: str | Path = "data/sessions.sqlite3") -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        try:
            with self._lock, self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS episodes ("
                    "session_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, "
                    "payload TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_episodes_updated_at ON episodes(updated_at)"
                )
        except sqlite3.Error as exc:
            raise EpisodeStoreError(f"Cannot initialize episode store {self.path}: {exc}") from exc

    @staticmethod
    def _payload(episode: Episode) -> str:
        return episode.model_dump_json()

    def create(self, session_id: str, episode: Episode) -> None:
        now = time()
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    "INSERT INTO episodes(session_id, episode_id, payload, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, episode.episode_id, self._payload(episode), now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise EpisodeStoreError(f"Duplicate session ID {session_id!r}") from exc
        except sqlite3.Error as exc:
            raise EpisodeStoreError(f"Cannot persist session {session_id!r}: {exc}") from exc

    def load(self, session_id: str) -> tuple[Episode, float]:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT payload, updated_at FROM episodes WHERE session_id = ?", (session_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise EpisodeStoreError(f"Cannot load session {session_id!r}: {exc}") from exc
        if row is None:
            raise KeyError(session_id)
        try:
            return Episode.model_validate(json.loads(row[0])), float(row[1])
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise CorruptEpisodeError(f"Stored session {session_id!r} is corrupt") from exc

    def save(self, session_id: str, episode: Episode) -> None:
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE episodes SET episode_id = ?, payload = ?, updated_at = ? WHERE session_id = ?",
                    (episode.episode_id, self._payload(episode), time(), session_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(session_id)
        except KeyError:
            raise
        except sqlite3.Error as exc:
            raise EpisodeStoreError(f"Cannot save session {session_id!r}: {exc}") from exc

    def delete(self, session_id: str) -> None:
        try:
            with self._lock, self._connect() as connection:
                connection.execute("DELETE FROM episodes WHERE session_id = ?", (session_id,))
        except sqlite3.Error as exc:
            raise EpisodeStoreError(f"Cannot delete session {session_id!r}: {exc}") from exc

    def count(self) -> int:
        try:
            with self._lock, self._connect() as connection:
                return int(connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
        except sqlite3.Error as exc:
            raise EpisodeStoreError(f"Cannot count sessions: {exc}") from exc

    def delete_expired(
        self, cutoff_epoch: float, protected_ids: tuple[str, ...] = ()
    ) -> int:
        try:
            with self._lock, self._connect() as connection:
                query = "DELETE FROM episodes WHERE updated_at < ?"
                parameters: tuple[object, ...] = (cutoff_epoch,)
                if protected_ids:
                    placeholders = ",".join("?" for _ in protected_ids)
                    query += f" AND session_id NOT IN ({placeholders})"
                    parameters += protected_ids
                cursor = connection.execute(query, parameters)
                return int(cursor.rowcount)
        except sqlite3.Error as exc:
            raise EpisodeStoreError(f"Cannot expire sessions: {exc}") from exc

    def close(self) -> None:
        """Connections are operation-scoped, so there is nothing to release."""
        return None
