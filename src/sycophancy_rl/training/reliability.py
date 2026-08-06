"""Pre-flight, signal-handling and resume helpers for the trainer.

This module owns every reliability concern that is independent of the
TRL training loop: disk-space checks, SIGINT/SIGTERM handling, atomic
checkpoint promotion, OOM messages, and the failure status that the
manifest writes when the trainer crashes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from sycophancy_rl.experiments.manifest import (
    ExperimentManifest,
    Status,
    sha256_file,
    transition,
    write_atomically,
)

MIN_DISK_FREE_GIB: float = 5.0
MIN_VRAM_GIB_BY_PROFILE: dict[str, float] = {
    "smoke": 0.0,
    "local_8gb": 7.0,
    "local_16gb": 12.0,
}


def free_disk_gib(path: Path) -> float:
    """Return free space under ``path`` in GiB; ``0.0`` if not measurable."""

    try:
        total, used, free = shutil.disk_usage(path)
    except OSError:
        return 0.0
    return free / 1024**3


def assert_disk_space(path: Path, min_gib: float = MIN_DISK_FREE_GIB) -> None:
    free = free_disk_gib(path)
    if free < min_gib:
        raise RuntimeError(
            f"Only {free:.1f} GiB free under {path}; need at least "
            f"{min_gib:.1f} GiB. Free up space and rerun."
        )


def assert_vram_compatible(profile: str, available_gib: float) -> None:
    recommended = MIN_VRAM_GIB_BY_PROFILE.get(profile)
    if recommended is None:
        return
    if available_gib < recommended:
        raise RuntimeError(
            f"Profile {profile!r} needs at least {recommended:.1f} GiB VRAM; "
            f"detected {available_gib:.1f} GiB. Use a smaller profile."
        )


def detect_vram_gib() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except ImportError:
        return 0.0


def explain_oom(message: str) -> str:
    """Return a human-friendly explanation for an out-of-memory message."""

    if "CUDA out of memory" not in message and "OutOfMemoryError" not in message:
        return message
    return (
        f"{message}\n\nActionable: reduce `--max-steps`, lower "
        "`max_completion_length`, lower `num_generations`, switch "
        "`--profile local_8gb` -> `smoke`, or enable 4-bit with `--4bit`."
    )


# --- checkpoint handling --------------------------------------------------


def promote_adapter_atomic(src: Path, dest: Path) -> Path:
    """Atomically move ``src`` adapter directory to ``dest``.

    ``dest`` must not already exist; ``src`` is consumed in the process.
    Returns the final ``dest`` path.
    """

    if dest.exists():
        raise FileExistsError(
            f"Adapter destination {dest} already exists; refusing to "
            "overwrite. Remove it manually or supply --run-name."
        )
    if not src.exists():
        raise FileNotFoundError(f"Adapter source {src} does not exist.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".promoting")
    shutil.move(str(src), str(tmp))
    os.replace(str(tmp), str(dest))
    return dest


def sha256_dir(path: Path) -> dict[str, str]:
    """SHA-256 every file under ``path`` recursively."""

    sums: dict[str, str] = {}
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = Path(root) / name
            sums[str(full.relative_to(path)).replace(os.sep, "/")] = sha256_file(full)
    return sums


# --- signal handling ------------------------------------------------------


@contextlib.contextmanager
def graceful_shutdown(
    *,
    on_signal: Callable[[int], None],
) -> Iterator[None]:
    """Install SIGINT/SIGTERM handlers that call ``on_signal`` once.

    On Linux/macOS the handlers are restored on exit.  On Windows only
    SIGINT is supported; SIGTERM is silently ignored.  The original
    handler is restored even when ``on_signal`` raises.
    """

    previous_int = signal.getsignal(signal.SIGINT)
    previous_term: int | signal._HANDLER  # type: ignore[attr-defined]
    if hasattr(signal, "SIGTERM"):
        previous_term = signal.getsignal(signal.SIGTERM)  # type: ignore[attr-defined]
    else:
        previous_term = signal.SIG_DFL  # type: ignore[assignment]

    def _handler(sig, _frame):
        on_signal(sig)
        raise KeyboardInterrupt(f"received signal {sig}")

    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)  # type: ignore[arg-type]

    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_int)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, previous_term)  # type: ignore[arg-type]


# --- manifest status bookkeeping ------------------------------------------


def record_interrupted(manifest: ExperimentManifest, manifest_path: Path) -> None:
    """Best-effort write of the ``interrupted`` status before exit."""

    updated = transition(manifest, to=Status.INTERRUPTED)
    try:
        write_atomically(updated, manifest_path, overwrite=True)
    except OSError:
        pass


def record_failed(manifest: ExperimentManifest, manifest_path: Path) -> None:
    updated = transition(manifest, to=Status.FAILED)
    try:
        write_atomically(updated, manifest_path, overwrite=True)
    except OSError:
        pass


def record_running(manifest: ExperimentManifest, manifest_path: Path) -> ExperimentManifest:
    updated = transition(manifest, to=Status.RUNNING)
    write_atomically(updated, manifest_path, overwrite=True)
    return updated


def record_completed(
    manifest: ExperimentManifest,
    manifest_path: Path,
    *,
    adapter_path: Path,
) -> ExperimentManifest:
    """Transition to ``completed`` and write the adapter checksum."""

    sums = sha256_dir(adapter_path) if adapter_path.exists() else {}
    checksum_payload = json.dumps(sums, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    adapter_sha = hashlib.sha256(checksum_payload).hexdigest() if sums else None
    if sums:
        checksums_path = manifest_path.parent / "adapter_checksums.json"
        temporary = checksums_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(sums, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, checksums_path)
    completed = transition(manifest, to=Status.COMPLETED)
    updated = replace(
        completed,
        adapter_path=str(adapter_path),
        adapter_sha256=adapter_sha,
    )
    write_atomically(updated, manifest_path, overwrite=True)
    return updated


def summarise_hardware() -> dict[str, Any]:
    """Return a small hardware dict for the manifest without importing torch."""

    info: dict[str, Any] = {"vram_gib": detect_vram_gib()}
    try:
        import psutil

        info["system_ram_gib"] = psutil.virtual_memory().total / 1024**3
    except ImportError:
        info["system_ram_gib"] = None
    try:
        import platform

        info["platform"] = platform.platform()
    except Exception:
        info["platform"] = None
    return info
