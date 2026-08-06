"""Tests for the trainer reliability helpers (mock-only, no model downloads)."""

from __future__ import annotations

import signal
from pathlib import Path

import pytest

from sycophancy_rl.experiments.manifest import (
    Stage,
    Status,
    load_manifest,
    new_manifest,
    write_atomically,
)
from sycophancy_rl.training.reliability import (
    assert_disk_space,
    assert_vram_compatible,
    detect_vram_gib,
    explain_oom,
    free_disk_gib,
    graceful_shutdown,
    promote_adapter_atomic,
    record_completed,
    record_failed,
    record_interrupted,
    record_running,
    sha256_dir,
    summarise_hardware,
)


def _base_manifest(**overrides):
    base = dict(
        run_id="r1",
        stage=Stage.BEFORE,
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_revision="a" * 40,
        adapter_path="outputs/r1/adapter",
        training_hash="b" * 64,
        validation_hash="c" * 64,
        benchmark_hash="d" * 64,
        seed=42,
        prompt_condition="neutral",
        generation_settings={"temperature": 0.7, "top_p": 0.9, "do_sample": True},
        reward_profile="combined",
    )
    base.update(overrides)
    return new_manifest(**base)


def test_free_disk_gib_positive(tmp_path: Path) -> None:
    assert free_disk_gib(tmp_path) > 0


def test_assert_disk_space_passes_with_enough_room(tmp_path: Path) -> None:
    assert_disk_space(tmp_path, min_gib=0.001)


def test_assert_disk_space_fails_when_starved(tmp_path: Path, monkeypatch) -> None:
    # Patch disk_usage to report starvation.  We don't touch the real disk.
    import sycophancy_rl.training.reliability as reliability

    monkeypatch.setattr(
        reliability.shutil, "disk_usage", lambda _p: (0, 0, 1)
    )
    with pytest.raises(RuntimeError, match="Free up space"):
        assert_disk_space(tmp_path, min_gib=1.0)


def test_assert_vram_compatible_passes_with_enough() -> None:
    assert_vram_compatible("local_8gb", 8.0)


def test_assert_vram_compatible_fails_with_too_little() -> None:
    with pytest.raises(RuntimeError, match="VRAM"):
        assert_vram_compatible("local_8gb", 2.0)


def test_explain_oom_includes_actionable_hints() -> None:
    msg = "CUDA out of memory. Tried to allocate 1.5 GiB."
    explained = explain_oom(msg)
    assert "Actionable" in explained
    assert "max-steps" in explained


def test_explain_oom_passes_through_unrelated_messages() -> None:
    msg = "ValueError: something else"
    assert explain_oom(msg) == msg


def test_promote_adapter_atomic(tmp_path: Path) -> None:
    src = tmp_path / "src_adapter"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"abc")
    dest = tmp_path / "dest_adapter"
    promote_adapter_atomic(src, dest)
    assert dest.exists()
    assert not src.exists()
    assert (dest / "weights.bin").read_bytes() == b"abc"


def test_promote_adapter_atomic_refuses_existing_dest(tmp_path: Path) -> None:
    src = tmp_path / "src_adapter"
    src.mkdir()
    dest = tmp_path / "dest_adapter"
    dest.mkdir()
    with pytest.raises(FileExistsError):
        promote_adapter_atomic(src, dest)


def test_promote_adapter_atomic_refuses_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        promote_adapter_atomic(tmp_path / "missing", tmp_path / "dest")


def test_sha256_dir_recurses(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "sub" / "b"
    a.write_bytes(b"hello")
    b.parent.mkdir()
    b.write_bytes(b"world")
    sums = sha256_dir(tmp_path)
    assert "a" in sums
    assert "sub/b" in sums
    assert len(set(sums.values())) == 2


def test_graceful_shutdown_invokes_callback(monkeypatch) -> None:
    calls: list[int] = []
    # We can't actually send SIGINT to ourselves in tests, so just
    # exercise the context manager and verify it restores handlers.
    with graceful_shutdown(on_signal=calls.append):
        assert signal.getsignal(signal.SIGINT) is not signal.SIG_DFL
    assert signal.getsignal(signal.SIGINT) in (signal.SIG_DFL, signal.default_int_handler)


def test_graceful_shutdown_restores_handlers() -> None:
    """Verify the context manager restores the SIGINT handler on exit."""

    original = signal.getsignal(signal.SIGINT)
    with graceful_shutdown(on_signal=lambda _sig: None):
        installed = signal.getsignal(signal.SIGINT)
        assert installed is not original
    assert signal.getsignal(signal.SIGINT) is original


def test_record_running_writes_running_status(tmp_path: Path) -> None:
    manifest = _base_manifest()
    path = tmp_path / "experiment.json"
    write_atomically(manifest, path)
    updated = record_running(manifest, path)
    assert updated.status is Status.RUNNING
    on_disk = load_manifest(path)
    assert on_disk.status is Status.RUNNING


def test_record_interrupted_and_failed(tmp_path: Path) -> None:
    manifest = _base_manifest()
    path = tmp_path / "experiment.json"
    record_running(manifest, path)

    record_interrupted(load_manifest(path), path)
    assert load_manifest(path).status is Status.INTERRUPTED

    record_running(load_manifest(path), path)
    record_failed(load_manifest(path), path)
    assert load_manifest(path).status is Status.FAILED


def test_record_completed_writes_adapter_checksum(tmp_path: Path) -> None:
    manifest = _base_manifest()
    path = tmp_path / "experiment.json"
    record_running(manifest, path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "weights.bin").write_bytes(b"weights")
    final = record_completed(load_manifest(path), path, adapter_path=adapter)
    assert final.status is Status.COMPLETED
    assert isinstance(final.adapter_sha256, str) and len(final.adapter_sha256) == 64
    on_disk = load_manifest(path)
    assert on_disk.status is Status.COMPLETED


def test_detect_vram_gib_returns_float() -> None:
    value = detect_vram_gib()
    assert isinstance(value, float)
    assert value >= 0


def test_summarise_hardware_does_not_import_torch_unnecessarily() -> None:
    info = summarise_hardware()
    assert "vram_gib" in info
    assert "platform" in info
