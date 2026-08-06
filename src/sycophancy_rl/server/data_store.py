"""Thread-safe, reload-on-change episode dataset cache for the API."""

from __future__ import annotations

from pathlib import Path
from threading import RLock

from sycophancy_rl.data_prep.schema import read_jsonl


class EpisodeStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._signature: tuple[int, int] | None = None
        self._rows: tuple[dict, ...] = ()

    def load(self) -> tuple[dict, ...]:
        with self._lock:
            if not self.path.exists():
                raise FileNotFoundError(self.path)
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != self._signature:
                self._rows = tuple(read_jsonl(self.path))
                self._signature = signature
            return self._rows

    def count(self) -> int:
        return len(self.load())
