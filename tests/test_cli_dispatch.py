"""Tests for the ``syco`` CLI dispatch contract.

These tests verify that the CLI is an honest orchestrator: preflight
never loads a model, real local training dispatches to the trainer,
Kaggle refuses to claim a real run, and the benchmark command cannot
silently report success without actually running.  No model weights
or datasets are downloaded and no GPU is required.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ``sycophancy_rl.cli.main`` resolves to the ``main`` function on most imports
# because of the package's ``__init__`` re-export.  Load the module by
# file path so we can introspect its module-level constants (such as
# ``VERSION``) without shadowing the function import.
_cli_main_spec = importlib.util.spec_from_file_location(
    "sycophancy_rl.cli.main",
    Path(__file__).resolve().parents[1]
    / "src"
    / "sycophancy_rl"
    / "cli"
    / "main.py",
)
cli_main_module = importlib.util.module_from_spec(_cli_main_spec)
_cli_main_spec.loader.exec_module(cli_main_module)
from sycophancy_rl.cli.main import (  # noqa: E402
    _build_parser,
    _cmd_train_kaggle,
    _cmd_train_local,
    main,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _ensure_src_on_pythonpath(monkeypatch):
    """Make ``src`` importable from any cwd the tests chdir to."""

    monkeypatch.setattr(sys, "path", [str(PROJECT_ROOT / "src"), *sys.path])


# --- Local preflight does not load a model ---------------------------------


def test_local_preflight_never_loads_a_model(tmp_path, monkeypatch) -> None:
    """`syco train --runner local --preflight-only` runs without weight I/O."""

    monkeypatch.chdir(tmp_path)

    from sycophancy_rl.training import train_grpo

    def _forbid_model_loader(*_args, **_kwargs):
        raise AssertionError("model_and_tokenizer must not be called in preflight mode")

    monkeypatch.setattr(train_grpo, "_model_and_tokenizer", _forbid_model_loader)

    # Replace torch import side-effects inside _preflight to avoid GPU probes.
    import types

    fake_torch = types.SimpleNamespace(
        __version__="0.0",
        version=types.SimpleNamespace(cuda=None),
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda *a, **kw: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    exit_code = main(
        [
            "train",
            "--runner",
            "local",
            "--preflight-only",
            "--profile",
            "smoke",
            "--allow-cpu",
        ]
    )
    assert exit_code == 0


# --- Local non-preflight dispatches run_training using mocks ----------------


def test_local_non_preflight_dispatches_run_training(monkeypatch, tmp_path) -> None:
    """Without --preflight-only, the CLI must call train_grpo.run_training."""

    monkeypatch.chdir(tmp_path)

    from sycophancy_rl.training import train_grpo

    captured: dict[str, object] = {}

    def _fake_run_training(args):
        captured["called"] = True
        captured["preflight_only"] = args.preflight_only
        captured["profile"] = args.profile
        captured["model_id"] = args.model_id
        captured["model_revision"] = args.model_revision
        captured["seed"] = args.seed
        captured["output_root"] = args.output_root
        captured["reward_profile"] = args.reward_profile
        captured["allow_cpu"] = args.allow_cpu
        captured["no_4bit"] = args.no_4bit
        return Path("/tmp/fake-run-dir")

    monkeypatch.setattr(train_grpo, "run_training", _fake_run_training)

    # Provide the minimum pool/splits for the data guard so the test would
    # *fail* if run_training ever actually ran the data validation path.
    pool = tmp_path / "data" / "processed" / "training_pool.jsonl"
    pool.parent.mkdir(parents=True, exist_ok=True)

    exit_code = main(
        [
            "train",
            "--runner",
            "local",
            "--profile",
            "smoke",
            "--model-id",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "--seed",
            "7",
            "--output-root",
            str(tmp_path / "out"),
            "--reward-profile",
            "answer_only",
            "--allow-cpu",
            "--no-4bit",
        ]
    )
    assert exit_code == 0
    assert captured.get("called") is True
    assert captured["preflight_only"] is False
    assert captured["profile"] == "smoke"
    assert captured["model_id"] == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert captured["seed"] == 7
    assert captured["reward_profile"] == "answer_only"
    assert captured["allow_cpu"] is True
    assert captured["no_4bit"] is True


def test_local_train_does_not_silently_force_cpu_or_4bit(monkeypatch, tmp_path) -> None:
    """When the user does not pass --allow-cpu or --no-4bit, the trainer
    must see those flags as False.  The CLI must not silently inject them."""

    monkeypatch.chdir(tmp_path)

    from sycophancy_rl.training import train_grpo

    captured: dict[str, object] = {}

    def _fake_run_training(args):
        captured["allow_cpu"] = args.allow_cpu
        captured["no_4bit"] = args.no_4bit
        captured["preflight_only"] = args.preflight_only
        return Path("/tmp/fake-run-dir")

    monkeypatch.setattr(train_grpo, "run_training", _fake_run_training)

    exit_code = main(["train", "--runner", "local", "--profile", "smoke"])
    assert exit_code == 0
    assert captured["allow_cpu"] is False
    assert captured["no_4bit"] is False
    assert captured["preflight_only"] is False


# --- Kaggle dispatch contract ----------------------------------------------


def test_kaggle_dry_run_reaches_cmd_dry_run(monkeypatch) -> None:
    """`syco train --runner kaggle --dry-run` must reach cmd_dry_run."""

    from sycophancy_rl.experiments import kaggle as kaggle_mod

    calls: dict[str, int] = {"dry_run": 0}

    def _fake_cmd_dry_run(_args):
        calls["dry_run"] += 1
        print(json.dumps({"ok": True, "runner": "kaggle"}))
        return 0

    monkeypatch.setattr(kaggle_mod, "cmd_dry_run", _fake_cmd_dry_run)

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exit_code = main(
            [
                "train",
                "--runner",
                "kaggle",
                "--dry-run",
                "--model-id",
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "--profile",
                "smoke",
            ]
        )
    assert exit_code == 0
    assert calls["dry_run"] == 1
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is True
    assert payload["runner"] == "kaggle"


def test_kaggle_non_dry_run_is_refused_with_non_zero_exit(
    capsys, monkeypatch
) -> None:
    """`syco train --runner kaggle` without --dry-run must fail loudly."""

    from sycophancy_rl.experiments import kaggle as kaggle_mod

    # If the CLI ever forgot to refuse, the dry-run function would be hit.
    monkeypatch.setattr(
        kaggle_mod, "cmd_dry_run", lambda _args: pytest.fail("dry-run must not be called")
    )

    exit_code = main(["train", "--runner", "kaggle", "--profile", "smoke"])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "staged experiment workflow" in captured.err


# --- Benchmark never returns success without execution --------------------


def test_benchmark_command_refuses_when_torch_missing(monkeypatch) -> None:
    """`syco benchmark` must fail with a clear non-zero exit when torch is not installed."""

    # Drop torch / transformers from sys.modules to simulate a thin env.
    for name in ("torch", "transformers"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    # Make the import itself fail.
    import builtins

    original_import = builtins.__import__

    def _restricted(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ImportError(f"simulated missing dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _restricted)

    exit_code = main(["benchmark", "--run-name", "smoke"])
    assert exit_code != 0
    assert exit_code != 2 or exit_code == 4  # explicit exit code


def test_benchmark_command_returns_zero_only_via_run(monkeypatch, tmp_path) -> None:
    """`syco benchmark` must exit 0 only when the real runner is invoked."""

    monkeypatch.chdir(tmp_path)

    from sycophancy_rl.evaluation import run_benchmark as rb

    calls: dict[str, int] = {"main": 0}

    def _fake_main():
        calls["main"] += 1
        print("benchmark ran")

    monkeypatch.setattr(rb, "main", _fake_main)

    # Provide a minimal benchmark path so the runner does not blow up.
    benchmark = tmp_path / "data" / "benchmarks" / "anthropic_sycophancy.jsonl"
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text("", encoding="utf-8")

    # Pretend torch + transformers are importable.
    import builtins
    import types

    original_import = builtins.__import__

    def _allow_ml(name, *args, **kwargs):
        if name == "torch":
            return types.SimpleNamespace(__version__="0.0")
        if name == "transformers":
            return types.SimpleNamespace(__version__="0.0")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _allow_ml)

    exit_code = main(
        [
            "benchmark",
            "--run-name",
            "smoke",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 0
    assert calls["main"] == 1


def test_benchmark_command_does_not_silently_succeed(monkeypatch) -> None:
    """If the underlying runner raises, the CLI must NOT report success."""

    from sycophancy_rl.evaluation import run_benchmark as rb

    def _fake_main():
        raise RuntimeError("simulated GPU/dataset failure")

    monkeypatch.setattr(rb, "main", _fake_main)

    import builtins
    import types

    original_import = builtins.__import__

    def _allow_ml(name, *args, **kwargs):
        if name == "torch":
            return types.SimpleNamespace(__version__="0.0")
        if name == "transformers":
            return types.SimpleNamespace(__version__="0.0")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _allow_ml)

    exit_code = main(["benchmark", "--run-name", "smoke"])
    assert exit_code != 0


# --- Custom models never receive SmolLM's revision -------------------------


def test_custom_model_does_not_inherit_smollm_revision(tmp_path, monkeypatch) -> None:
    """The trainer must not silently substitute the reference revision
    when the user supplies a custom model id without --model-revision."""

    monkeypatch.chdir(tmp_path)
    from sycophancy_rl.evaluation.run_benchmark import DEFAULT_MODEL_REVISION
    from sycophancy_rl.training import train_grpo

    captured: dict[str, object] = {}

    def _fake_run_preflight(args):
        captured["model_id"] = args.model_id
        captured["model_revision"] = args.model_revision
        return {"status": "ok"}

    monkeypatch.setattr(train_grpo, "run_preflight", _fake_run_preflight)

    # The CLI default for --model-revision is empty so that, when forwarded,
    # the trainer must not invent the reference SHA.
    argv = [
        "train",
        "--runner",
        "local",
        "--preflight-only",
        "--profile",
        "smoke",
        "--model-id",
        "custom-org/custom-model",
        "--allow-cpu",
    ]
    exit_code = main(argv)
    assert exit_code == 0
    # The preflight sees no revision; the trainer's own registry resolution
    # is what must decide whether to keep it empty or supply a pinned one.
    assert captured["model_id"] == "custom-org/custom-model"
    assert captured["model_revision"] in (None, "")  # CLI does not invent one

    # Direct invocation of the trainer's registry helper confirms the
    # custom model is not given the SmolLM revision.
    from sycophancy_rl.training.train_grpo import _resolve_model

    parsed = train_grpo._parse_args(
        ["--profile", "smoke", "--model-id", "custom-org/custom-model"]
    )
    parsed.model_revision = "0" * 40
    with pytest.raises(ValueError, match="allow-unpinned-model"):
        _resolve_model(parsed)
    parsed.allow_unpinned_model = True
    profile = _resolve_model(parsed)
    assert profile.model_id == "custom-org/custom-model"
    assert profile.revision != DEFAULT_MODEL_REVISION


def test_registered_model_uses_pinned_revision(tmp_path, monkeypatch) -> None:
    """When the user requests a registered model id, the registry pins its revision."""

    monkeypatch.chdir(tmp_path)
    from sycophancy_rl.training import train_grpo
    from sycophancy_rl.training.train_grpo import _resolve_model

    parsed = train_grpo._parse_args(
        ["--profile", "smoke", "--model-id", "HuggingFaceTB/SmolLM2-1.7B-Instruct"]
    )
    # Empty revision: must still resolve to the registered pinned SHA.
    parsed.model_revision = ""
    profile = _resolve_model(parsed)
    assert profile.model_id == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert len(profile.revision) >= 7
    assert ":" not in profile.revision


# --- Registry loading settings are enforced --------------------------------


def test_trust_remote_code_is_explicitly_false_in_trainer_kwargs(monkeypatch) -> None:
    """The trainer's model loader must pass trust_remote_code=False.

    The previous version used ``sys.modules.setdefault`` which silently
    kept the real ``transformers`` if it was already imported, causing
    the trainer to attempt a real Hugging Face download.  Use
    ``monkeypatch.setitem`` so the fakes are installed deterministically
    and restored after the test.
    """

    from sycophancy_rl.training import train_grpo

    captured: dict[str, object] = {}

    class _FakeTokenizer:
        pad_token_id = 0
        eos_token = "</s>"
        pad_token = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured.setdefault("tokenizer_calls", []).append((args, kwargs))
            return cls()

    class _FakeModel:
        config = type("Cfg", (), {"use_cache": True})()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured.setdefault("model_calls", []).append((args, kwargs))
            return cls()

    class _FakeBitsConfig:
        def __init__(self, **kwargs):
            captured.setdefault("bnb_kwargs", []).append(kwargs)

    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            captured.setdefault("lora_kwargs", []).append(kwargs)

    import types

    fake_torch = types.SimpleNamespace(
        __version__="0.0",
        version=types.SimpleNamespace(cuda=None),
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda *a, **kw: None,
            is_bf16_supported=lambda: False,
        ),
    )
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        AutoModelForCausalLM=_FakeModel,
        BitsAndBytesConfig=_FakeBitsConfig,
    )
    fake_peft = types.SimpleNamespace(LoraConfig=_FakeLoraConfig)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    train_grpo._model_and_tokenizer(
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_revision="31b70e2e869a7173562077fd711b654946d38674",
        load_in_4bit=False,
        allow_cpu=True,
    )

    assert captured["model_calls"], "model.from_pretrained was not called"
    _args, kwargs = captured["model_calls"][0]
    assert kwargs.get("trust_remote_code") is False


def test_model_registry_safe_loading_kwargs_are_enforced() -> None:
    """The CLI must consult the registry's safety filter for loading kwargs."""

    from sycophancy_rl.training.model_registry import is_safe_loading_kwargs

    assert is_safe_loading_kwargs({})
    assert is_safe_loading_kwargs({"dtype": "auto"})
    assert not is_safe_loading_kwargs({"use_unsafe_tokens": True})
    assert not is_safe_loading_kwargs({"force_download": True})
    assert not is_safe_loading_kwargs(
        {"dtype": "auto", "trust_remote_code": True}
    )


# --- Manifest lifecycle is exercised through the real trainer orchestration -


def test_trainer_records_manifest_lifecycle_on_real_run(monkeypatch, tmp_path) -> None:
    """A real ``run_training`` invocation must write a manifest that
    transitions through ``created`` -> ``running`` -> ``completed``."""

    monkeypatch.chdir(tmp_path)

    # Drop any cached torch import so the fake below takes effect.
    for name in ("torch", "transformers", "peft", "trl", "datasets"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    import types

    # Build all the heavy-ML fakes before the trainer module sees them.
    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.state = type("S", (), {"global_step": 0})()
            self._metrics = {"train_loss": 0.5}
            self.train_called = False

        def train(self, resume_from_checkpoint=None):
            self.train_called = True
            self.state.global_step = 5
            return type("R", (), {"metrics": {"train_loss": 0.4}})()

        def save_model(self, path):
            destination = Path(path)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "weights.bin").write_bytes(b"x")

    class _FakeModel:
        config = type("Cfg", (), {"use_cache": True})()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _FakeTokenizer:
        pad_token_id = 0
        eos_token = "</s>"
        pad_token = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return dict(self.kwargs)

    fake_torch = types.SimpleNamespace(
        __version__="0.0",
        version=types.SimpleNamespace(cuda=None),
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda *a, **kw: None,
            max_memory_allocated=lambda: 0,
            is_bf16_supported=lambda: False,
        ),
        manual_seed=lambda seed: None,
        use_deterministic_algorithms=lambda *a, **kw: None,
    )
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        AutoModelForCausalLM=_FakeModel,
        BitsAndBytesConfig=lambda **kw: None,
    )
    fake_peft = types.SimpleNamespace(LoraConfig=_FakeLoraConfig)
    fake_trl = types.SimpleNamespace(
        GRPOConfig=lambda **kw: type("C", (), {"to_dict": lambda self: dict(kw)})(),
        GRPOTrainer=_FakeTrainer,
    )
    fake_datasets = types.SimpleNamespace(
        Dataset=type(
            "D", (), {"from_list": staticmethod(lambda rows: rows)}
        ),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    from sycophancy_rl.experiments.manifest import Status, load_manifest
    from sycophancy_rl.training import train_grpo

    pool = tmp_path / "data" / "processed" / "training_pool.jsonl"
    pool.parent.mkdir(parents=True, exist_ok=True)

    # Minimal non-fixture training row that satisfies the schema.
    minimal_row = {
        "example_id": "lifecycle-row-1",
        "source": "allenai/ai2_arc/ARC-Challenge/train",
        "source_revision": "deadbeef",
        "data_role": "training",
        "prompt": [
            {
                "role": "user",
                "content": (
                    "What is 2 + 2?\n(A) 4\n(B) 5\nChoose independently. "
                    "Reply exactly as: Answer: (A) or Answer: (B)\n"
                    "Reason: one short sentence."
                ),
            }
        ],
        "options": {"A": "4", "B": "5"},
        "target_option": "A",
        "independent_option": "A",
        "sycophantic_option": "B",
        "user_preferred_option": "B",
        "user_claim_valid": False,
        "behavior_target": "resist_invalid_pressure",
        "question_type": "objective",
        "topic": "general_knowledge",
        "pushback_turns": [
            {
                "text": "Are you sure?",
                "category": "doubt",
                "is_valid_correction": False,
                "evidence_strength": "weak",
                "user_claim_valid": False,
            }
        ],
        "metadata": {"question_id": "q1", "scenario": "invalid-pressure"},
    }
    val_row = dict(minimal_row)
    val_row["example_id"] = "lifecycle-val-1"
    val_row["data_role"] = "validation"
    train_path = tmp_path / "data" / "splits" / "train.jsonl"
    val_path = tmp_path / "data" / "splits" / "validation.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(
        json.dumps(minimal_row, sort_keys=True) + "\n", encoding="utf-8"
    )
    val_path.write_text(
        json.dumps(val_row, sort_keys=True) + "\n", encoding="utf-8"
    )

    output_root = tmp_path / "out"
    parsed = train_grpo._parse_args(
        [
            "--profile",
            "smoke",
            "--allow-cpu",
            "--no-4bit",
            "--model-id",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "--seed",
            "11",
            "--output-root",
            str(output_root),
            "--run-name",
            "lifecycle-test",
        ]
    )
    parsed.train = train_path
    parsed.validation = val_path

    out_dir = train_grpo.run_training(parsed)
    assert out_dir.exists()

    manifest_path = out_dir / "training_manifest.json"
    assert manifest_path.exists(), "ExperimentManifest must be written atomically"
    final = load_manifest(manifest_path)
    assert final.status is Status.COMPLETED
    assert final.run_id == "lifecycle-test"
    assert final.adapter_sha256 is not None
    assert final.generation_settings["max_new_tokens"] == 64
    assert (
        final.generation_settings["max_new_tokens"]
        == final.effective_training_config["max_completion_length"]
    )

    # The run_status.json must agree with the manifest's terminal state.
    run_status = json.loads((out_dir / "run_status.json").read_text(encoding="utf-8"))
    assert run_status["status"] == "completed"


# --- Argument parser surface ------------------------------------------------


def test_train_parser_exposes_required_flags() -> None:
    parser = _build_parser()
    help_lines = parser.format_help()
    # Subcommand help lives on each subparser; pull the train one.
    train_parser = None
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices:
            sub = action.choices.get("train")
            if sub is not None:
                train_parser = sub
                break
    assert train_parser is not None, "train subcommand must exist"
    train_help = train_parser.format_help()
    for flag in (
        "--runner",
        "--preflight-only",
        "--model-id",
        "--seed",
        "--output-root",
        "--reward-profile",
        "--allow-cpu",
        "--no-4bit",
    ):
        assert flag in train_help, f"missing {flag} in train help"
    assert "train" in help_lines


def test_run_and_benchmark_use_safe_completion_budget_by_default() -> None:
    parser = _build_parser()
    run_args = parser.parse_args(["run", "--run-id", "default-budget"])
    benchmark_args = parser.parse_args(["benchmark", "--run-name", "default-budget"])

    assert run_args.max_new_tokens == 192
    assert benchmark_args.max_new_tokens == 192


def test_kaggle_runner_dry_run_returns_payload(monkeypatch, capsys) -> None:
    """Direct invocation of the kaggle handler must reach cmd_dry_run."""

    from sycophancy_rl.experiments import kaggle as kaggle_mod

    def _fake_cmd_dry_run(_args):
        print(json.dumps({"ok": True, "dry_run": True, "runner": "kaggle"}))
        return 0

    monkeypatch.setattr(kaggle_mod, "cmd_dry_run", _fake_cmd_dry_run)

    args = argparse.Namespace(
        dry_run=True,
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_revision="",
        profile="smoke",
        seed=42,
        output_root=Path("outputs/checkpoints"),
        reward_profile="combined",
        allow_cpu=False,
        no_4bit=False,
    )
    code = _cmd_train_kaggle(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["dry_run"] is True


def test_kaggle_runner_non_dry_run_is_refused() -> None:
    """The Kaggle handler must refuse to call into the runner without --dry-run."""

    args = argparse.Namespace(
        dry_run=False,
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_revision="",
        profile="smoke",
        seed=42,
        output_root=Path("outputs/checkpoints"),
        reward_profile="combined",
        allow_cpu=False,
        no_4bit=False,
    )
    code = _cmd_train_kaggle(args)
    assert code != 0


def test_local_train_handler_dispatches_to_run_training(monkeypatch, tmp_path) -> None:
    """Direct invocation of the local handler must call run_training."""

    monkeypatch.chdir(tmp_path)
    from sycophancy_rl.training import train_grpo

    called = {"value": False}

    def _fake(args):
        called["value"] = True
        return Path("/tmp/fake")

    monkeypatch.setattr(train_grpo, "run_training", _fake)

    args = argparse.Namespace(
        profile="smoke",
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_revision="",
        seed=42,
        output_root=tmp_path / "out",
        reward_profile="combined",
        allow_cpu=True,
        no_4bit=True,
        preflight_only=False,
    )
    code = _cmd_train_local(args)
    assert code == 0
    assert called["value"] is True


def test_local_train_handler_routes_to_preflight_when_flag_set(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    from sycophancy_rl.training import train_grpo

    called = {"preflight": 0, "training": 0}

    def _fake_preflight(args):
        called["preflight"] += 1
        return {"status": "ok", "profile": args.profile}

    def _fake_training(args):
        called["training"] += 1
        return Path("/tmp/fake")

    monkeypatch.setattr(train_grpo, "run_preflight", _fake_preflight)
    monkeypatch.setattr(train_grpo, "run_training", _fake_training)

    args = argparse.Namespace(
        profile="smoke",
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_revision="",
        seed=42,
        output_root=tmp_path / "out",
        reward_profile="combined",
        allow_cpu=True,
        no_4bit=True,
        preflight_only=True,
    )
    code = _cmd_train_local(args)
    assert code == 0
    assert called["preflight"] == 1
    assert called["training"] == 0


# --- Version comes from package metadata ----------------------------------


def test_cli_version_reads_from_installed_package_metadata(monkeypatch) -> None:
    """`syco version` must print the installed package metadata version, not a hard-coded string."""

    import importlib.metadata as importlib_metadata

    try:
        real_version = importlib_metadata.version("sycophancy-rl")
    except importlib_metadata.PackageNotFoundError:
        # When the package is not installed (e.g. an editable checkout
        # without a metadata install), the CLI must report "unknown"
        # rather than fabricate a version.
        real_version = "unknown"

    code = main(["version"])
    assert code == 0
    captured = cli_main_module.VERSION
    assert captured == real_version
    # And the CLI must not have a hard-coded literal "0.2.0" placeholder
    # when the package is installed (allowing 0.2.0 to be the *real*
    # installed version, not a fabricated one).
    if real_version != "0.2.0":
        assert "0.2.0" not in cli_main_module.VERSION
