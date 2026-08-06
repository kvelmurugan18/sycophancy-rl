"""Tests for the ``syco`` command-line interface.

The CLI is a thin orchestrator around existing modules, so the
tests cover:

* argument parsing — every documented subcommand is exposed, every
  argument type is enforced, and unknown args are rejected with exit 2;
* exit behavior — ``syco version`` exits 0, ``syco doctor`` exits 0 and
  prints a JSON snapshot, ``syco setup`` exits 0 with the documented
  messages, ``syco train --profile X --preflight-only`` exercises the
  trainer preflight path without loading a model;
* error surfacing — recoverable errors from delegated helpers are
  printed without a traceback and exit with code 2.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sycophancy_rl.cli.main import __version__, _build_parser, main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _ensure_src_on_pythonpath(monkeypatch):
    """Make ``src`` importable from any cwd the tests chdir to."""

    monkeypatch.setattr(sys, "path", [str(PROJECT_ROOT / "src"), *sys.path])


# --- Argument parsing --------------------------------------------------------


def test_argument_parser_exposes_every_documented_subcommand() -> None:
    parser = _build_parser()
    help_lines = parser.format_help().splitlines()
    joined = "\n".join(help_lines)
    for command in ("doctor", "setup", "prepare-data", "import-data", "train", "version"):
        assert command in joined, f"missing subcommand {command} in help text"


def test_train_subcommand_requires_known_profile() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["train", "--profile", "unknown-profile"])
    assert exc_info.value.code == 2


def test_unknown_subcommand_is_rejected_with_exit_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["not-a-command"])
    assert exc_info.value.code == 2


def test_missing_subcommand_is_rejected_with_exit_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2


def test_version_command_exits_0_and_prints_version() -> None:
    captured: list[str] = []

    def _capture_print(*args, **_kwargs):
        captured.append(" ".join(str(a) for a in args))

    import builtins

    original_print = builtins.print
    builtins.print = _capture_print  # type: ignore[assignment]
    try:
        exit_code = main(["version"])
    finally:
        builtins.print = original_print  # type: ignore[assignment]
    assert exit_code == 0
    assert any("syco" in line and __version__ in line for line in captured)


# --- Exit behavior -----------------------------------------------------------


def test_doctor_command_exits_0_and_prints_json(tmp_path, monkeypatch) -> None:
    # The doctor command reads from the current working directory's data/
    # tree, so point it at an empty tmp_path and run from there.
    monkeypatch.chdir(tmp_path)
    captured: list[str] = []

    def _capture_print(*args, **_kwargs):
        captured.append(" ".join(str(a) for a in args))

    import builtins

    original_print = builtins.print
    builtins.print = _capture_print  # type: ignore[assignment]
    try:
        exit_code = main(["doctor"])
    finally:
        builtins.print = original_print  # type: ignore[assignment]
    assert exit_code == 0
    assert captured, "doctor command printed nothing"
    payload = json.loads("".join(captured))
    assert "python" in payload
    assert "training_pool" in payload
    assert payload["training_pool"]["exists"] is False


def test_setup_command_uses_non_destructive_helpers(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = main(["setup"])
    assert exit_code == 0
    pool = tmp_path / "data" / "processed" / "training_pool.jsonl"
    splits = tmp_path / "data" / "splits"
    assert pool.exists()
    assert (splits / "train.jsonl").exists()
    assert (splits / "validation.jsonl").exists()
    assert (splits / "test.jsonl").exists()
    assert (splits / "split_manifest.json").exists()


def test_setup_command_preserves_existing_real_pool_byte_for_byte(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    pool_dir = tmp_path / "data" / "processed"
    pool_dir.mkdir(parents=True)
    pool = pool_dir / "training_pool.jsonl"
    pool.write_bytes(

            b'{"example_id":"real","source":"allenai/ai2_arc/train",'
            b'"data_role":"training","prompt":[]}\n'

    )
    expected_pool_bytes = pool.read_bytes()

    # `syco setup` cannot validate this minimal real pool (it lacks the
    # training fields required by the schema), so we expect the safety
    # guard to bail out *without* overwriting the existing pool.  The
    # byte-for-byte preservation is the contract under test.
    exit_code = main(["setup"])
    assert exit_code == 2
    assert pool.read_bytes() == expected_pool_bytes


def test_import_data_command_creates_user_splits(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = PROJECT_ROOT / "data" / "examples" / "user_choice_dataset.csv"
    exit_code = main(
        [
            "import-data",
            "--input",
            str(source),
            "--dataset-name",
            "cli-customer",
            "--output-dir",
            str(tmp_path / "generated"),
        ]
    )
    assert exit_code == 0
    destination = tmp_path / "generated" / "cli-customer"
    assert (destination / "splits" / "train.jsonl").exists()
    assert (destination / "splits" / "validation.jsonl").exists()


def test_train_command_runs_preflight_without_loading_model(
    tmp_path, monkeypatch
) -> None:
    """The ``syco train --preflight-only`` subcommand must invoke the preflight path."""

    monkeypatch.chdir(tmp_path)
    captured: list[str] = []

    def _capture_print(*args, **_kwargs):
        captured.append(" ".join(str(a) for a in args))

    import builtins

    original_print = builtins.print
    builtins.print = _capture_print  # type: ignore[assignment]
    try:
        exit_code = main(["train", "--profile", "smoke", "--preflight-only", "--allow-cpu"])
    finally:
        builtins.print = original_print  # type: ignore[assignment]

    assert exit_code == 0
    payload = json.loads("".join(captured))
    assert payload["status"] == "ok"
    assert payload["profile"] == "smoke"


def test_train_command_surfaces_preflight_failure(tmp_path, monkeypatch) -> None:
    """A recoverable error from the preflight must exit 2 without a traceback."""

    monkeypatch.chdir(tmp_path)

    from sycophancy_rl.training import train_grpo

    def _explode(_args):
        raise ValueError("simulated preflight failure")

    monkeypatch.setattr(train_grpo, "run_preflight", _explode)

    captured_stderr: list[str] = []

    def _capture_stderr(*args, **_kwargs):
        captured_stderr.append(" ".join(str(a) for a in args))

    original_stderr_write = sys.stderr.write
    sys.stderr.write = _capture_stderr  # type: ignore[assignment]
    try:
        exit_code = main(["train", "--profile", "smoke", "--preflight-only", "--allow-cpu"])
    finally:
        sys.stderr.write = original_stderr_write  # type: ignore[assignment]

    assert exit_code == 2
    assert any("simulated preflight failure" in line for line in captured_stderr)


# --- Console-script entry point ---------------------------------------------


def _project_python() -> str:
    return sys.executable


def test_console_script_is_registered_in_pyproject() -> None:
    """The ``syco`` entry point must be declared in pyproject.toml."""

    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" in pyproject
    assert "syco = " in pyproject
    assert "sycophancy_rl.cli.main:main" in pyproject


def test_console_script_invocation_via_python_module(tmp_path) -> None:
    """``python -m sycophancy_rl.cli.main version`` must work end-to-end."""

    # Run from the project root so ``src`` is importable; setting
    # ``PYTHONPATH`` makes the test robust to ``pip install`` state.
    project_root = Path(__file__).resolve().parents[1]
    env = {**subprocess.os.environ, "PYTHONPATH": str(project_root / "src")}
    completed = subprocess.run(
        [sys.executable, "-m", "sycophancy_rl.cli.main", "version"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "syco" in completed.stdout
    assert __version__ in completed.stdout
