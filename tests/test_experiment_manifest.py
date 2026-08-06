"""Tests for the shared experiment-manifest contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sycophancy_rl.experiments.manifest import (
    SCHEMA_VERSION,
    Stage,
    Status,
    assert_benchmark_only,
    assert_no_fixture_contamination,
    generation_settings_equal,
    is_resume_compatible,
    load_manifest,
    new_manifest,
    sha256_bytes,
    sha256_file,
    transition,
    write_artifact_checksums,
    write_atomically,
)


def _base_kwargs(**overrides):
    base = dict(
        run_id="run-001",
        stage=Stage.BEFORE,
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_revision="31b70e2e869a7173562077fd711b654946d38674",
        adapter_path="outputs/run-001/adapter",
        training_hash="a" * 64,
        validation_hash="b" * 64,
        benchmark_hash="c" * 64,
        seed=42,
        prompt_condition="neutral",
        generation_settings={
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 256,
            "do_sample": True,
        },
        reward_profile="combined",
        effective_training_config={"learning_rate": 1e-5, "beta": 0.04},
        hardware={"cuda_available": False},
        python_versions={"torch": "2.12.1"},
        git_commit="abc1234",
        runner="local",
        data_governance={"benchmark_only": False, "is_fixture": False},
    )
    base.update(overrides)
    return base


def test_new_manifest_starts_in_created_state() -> None:
    manifest = new_manifest(**_base_kwargs())
    assert manifest.status is Status.CREATED
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.adapter_sha256 is None
    assert "created" in manifest.timestamps


def test_status_transitions_are_atomic() -> None:
    manifest = new_manifest(**_base_kwargs())
    running = transition(manifest, to=Status.RUNNING)
    assert running.status is Status.RUNNING
    assert "started" in running.timestamps
    completed = transition(running, to=Status.COMPLETED)
    assert completed.status is Status.COMPLETED
    assert "finished" in completed.timestamps


def test_illegal_status_transition_is_rejected() -> None:
    manifest = new_manifest(**_base_kwargs())
    completed = transition(
        transition(manifest, to=Status.RUNNING), to=Status.COMPLETED
    )
    with pytest.raises(ValueError, match="Illegal manifest status transition"):
        transition(completed, to=Status.RUNNING)


def test_atomic_write_creates_run_directory(tmp_path: Path) -> None:
    manifest = new_manifest(**_base_kwargs())
    path = tmp_path / "experiment.json"
    write_atomically(manifest, path)
    assert path.exists()
    loaded = load_manifest(path)
    assert loaded.run_id == manifest.run_id


def test_atomic_write_refuses_overwrite(tmp_path: Path) -> None:
    manifest = new_manifest(**_base_kwargs())
    path = tmp_path / "experiment.json"
    write_atomically(manifest, path)
    with pytest.raises(FileExistsError):
        write_atomically(manifest, path)


def test_load_rejects_unsupported_schema(tmp_path: Path) -> None:
    bad = {"schema_version": 999, **{k: v for k, v in _base_kwargs().items() if k != "schema_version"}}
    bad["status"] = "created"
    bad["stage"] = "before"
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_manifest(path)


def test_sha256_file_handles_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "nope")


def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    assert sha256_file(p) == sha256_file(p)


def test_write_artifact_checksums(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a" * 100)
    b.write_bytes(b"b" * 50)
    sums = write_artifact_checksums(
        {"a": a, "b": b}, tmp_path / "checksums.json"
    )
    assert len(sums) == 2
    assert all(len(v) == 64 for v in sums.values())


def test_resume_compatible_when_only_status_changed() -> None:
    m1 = new_manifest(**_base_kwargs())
    m2 = new_manifest(**_base_kwargs())
    interrupted = transition(transition(m1, to=Status.RUNNING), to=Status.INTERRUPTED)
    assert is_resume_compatible(interrupted, m2)[0]


def test_resume_incompatible_when_model_revision_changed() -> None:
    m1 = new_manifest(**_base_kwargs())
    m2 = new_manifest(**_base_kwargs(model_revision="0" * 40))
    ok, reason = is_resume_compatible(m1, m2)
    assert not ok
    assert "model_revision" in reason


def test_resume_refuses_when_previous_completed() -> None:
    m1 = new_manifest(**_base_kwargs())
    completed = transition(
        transition(m1, to=Status.RUNNING), to=Status.COMPLETED
    )
    m2 = new_manifest(**_base_kwargs())
    ok, reason = is_resume_compatible(completed, m2)
    assert not ok
    assert "terminal state" in reason


def test_assert_benchmark_only_refuses_anthropic_data() -> None:
    m = new_manifest(
        **_base_kwargs(
            data_governance={
                "source": "anthropic/model-written-evals",
                "benchmark_only": True,
            }
        )
    )
    with pytest.raises(ValueError, match="evaluation-only"):
        assert_benchmark_only(m)


def test_assert_benchmark_only_passes_normal_data() -> None:
    m = new_manifest(**_base_kwargs())
    assert_benchmark_only(m)


def test_assert_no_fixture_contamination_refuses_fixtures() -> None:
    m = new_manifest(
        **_base_kwargs(data_governance={"is_fixture": True})
    )
    with pytest.raises(ValueError, match="Fixture data reached"):
        assert_no_fixture_contamination(m)


def test_generation_settings_equal_strict_match() -> None:
    a = {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "max_new_tokens": 256, "do_sample": True}
    assert generation_settings_equal(a, dict(a))
    b = dict(a)
    b["temperature"] = 0.8
    assert not generation_settings_equal(a, b)


def test_manifest_round_trip_preserves_fields(tmp_path: Path) -> None:
    manifest = new_manifest(**_base_kwargs())
    running = transition(manifest, to=Status.RUNNING)
    path = tmp_path / "experiment.json"
    write_atomically(running, path, overwrite=True)
    loaded = load_manifest(path)
    assert loaded.run_id == running.run_id
    assert loaded.status is Status.RUNNING
    assert loaded.generation_settings == running.generation_settings


def test_sha256_bytes_stable_for_strings() -> None:
    assert sha256_bytes("hello") == sha256_bytes(b"hello")
    assert sha256_bytes("hello") != sha256_bytes("world")
