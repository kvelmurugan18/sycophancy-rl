"""GRPO training entry point.

This module is the **main entry point** for launching the GRPO
training loop. It is intentionally small — it does the wiring between
three pre-built pieces and then hands off to TRL's ``GRPOTrainer``:

1. **The dataset** — a JSONL file at ``data/processed/merged_episodes.jsonl``
   produced by ``src/data_prep/merge_datasets.py`` and split by
   ``src/data_prep/split_data.py``. Loaded with HuggingFace
   ``datasets`` so TRL can stream it row-by-row.
2. **The four-pillar reward function** — ``composite_reward_func`` from
   ``src/reward/reward_fn.py``. Imported via a ``sys.path`` shim so
   the script works whether it is invoked as ``python -m
   src.training.train_grpo`` (project root on path) or as a bare
   ``python src/training/train_grpo.py``.
3. **The Kaggle-tuned config** — ``get_grpo_config()`` from
   ``src/training/grpo_config.py``. Defaults are sized for a single
   16 GB GPU; see that module's docstring for the rationale.

Run from the project root:

    python -m src.training.train_grpo

…or with overrides for the model and dataset:

    python -m src.training.train_grpo \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --dataset data/splits/train.jsonl
"""

# --- Standard library ---
import argparse
import sys
from pathlib import Path

# --- Third-party ---
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import GRPOTrainer

# --- Local: GRPO config factory (relative import; works when this
# module is loaded as part of the ``src.training`` package). ---
from .grpo_config import get_grpo_config

# --- Local: the four-pillar reward function. The project is laid
# out as a flat package rooted at ``src/`` with an empty
# ``src/__init__.py`` and ``src/reward/__init__.py``, so the absolute
# import ``src.reward.reward_fn`` works as long as the directory
# *containing* ``src/`` (i.e. the project root) is on ``sys.path``.
# We make that guarantee explicitly here so the script can be run
# either as ``python -m src.training.train_grpo`` (which adds CWD to
# ``sys.path`` automatically) or as ``python src/training/train_grpo.py``
# (which does not). The shim is idempotent — re-adding the same
# directory is a no-op. ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.reward.reward_fn import composite_reward_func  # noqa: E402  (sys.path shim above)

# Sensible defaults for the CLI / ``__main__`` block. These match the
# values the rest of the project (data prep scripts, notebooks) was
# built around, so a no-arg run reproduces the documented setup.
_DEFAULT_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
_DEFAULT_DATASET: str = "data/processed/merged_episodes.jsonl"


def run_training(model_id: str, dataset_path: str) -> None:
    """Run the full GRPO training loop end-to-end.

    The function is procedural rather than factored into smaller
    pieces on purpose: it is a linear pipeline (load → instantiate →
    train → save) with no reusable sub-step, and inlining the steps
    keeps the data flow obvious to a reader who has never seen the
    code before.

    Args:
        model_id: HuggingFace model identifier (or local path) of the
            base model to fine-tune. Should be a chat-tuned model
            (e.g. ``Qwen/Qwen2.5-1.5B-Instruct``) so the GRPO loop
            starts from a model that already understands the
            turn-taking format.
        dataset_path: Path to a JSONL file consumable by
            ``datasets.load_dataset``. The file should already be in
            the unified episode schema produced by
            ``src/data_prep/merge_datasets.py`` (columns: ``prompt``,
            ``correct_answer``, ``wrong_answer``, ``source``,
            ``pushback_turns``).
    """
    # --- 1. Tokenizer. ---
    # ``use_fast=True`` picks the Rust tokenizer when available,
    # which is what TRL expects for its dataloader fast paths.
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # --- 2. Base model. ---
    # bfloat16 is the right dtype for a 16 GB card: half the memory
    # of fp32, with the same training stability as fp32 on Ampere+
    # hardware (the T4 / L4 / 3090 / 4090 all support it). For older
    # cards without bf16 support, swap to ``torch_dtype="float16"``
    # here — the rest of the loop is dtype-agnostic.
    # ``device_map="auto"`` lets ``accelerate`` shard the model
    # across whatever devices ``accelerate`` can see; with a single
    # GPU this just means "put it on cuda:0".
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    # --- 3. Dataset. ---
    # ``load_dataset`` with the ``"json"`` builder reads a JSONL file
    # directly into an ``arrow``-backed ``Dataset``. The schema is
    # inferred from the first row; callers are expected to have
    # produced that schema upstream (see merge_datasets.py).
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # --- 4. Config. ---
    # Defaults from ``grpo_config.py`` are tuned for a 16 GB card.
    training_args = get_grpo_config()

    # --- 5. Trainer. ---
    # ``reward_funcs`` is a *list* in TRL's API: you can stack
    # multiple reward functions and TRL will sum them. We pass a
    # single-element list containing our composite function.
    # ``processing_class`` is the new (TRL ≥ 0.12) name for what
    # used to be called ``tokenizer``; the rename is to make it
    # explicit that any processor that can tokenize + post-process
    # is accepted, not just a tokenizer.
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[composite_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # --- 6. Train. ---
    trainer.train()

    # --- 7. Save. ---
    # We save *both* the model and the tokenizer so the
    # ``output_dir`` is a self-contained artifact: anyone can load
    # it back with a single ``from_pretrained`` call without having
    # to fish the tokenizer out of the original base-model repo.
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training run.

    Both arguments are optional: the defaults are the project
    defaults baked into the module (``_DEFAULT_MODEL`` and
    ``_DEFAULT_DATASET``), so a no-arg invocation is the documented
    "just train it" entry point.
    """
    parser = argparse.ArgumentParser(
        description="Launch the GRPO training loop against the four-pillar reward function.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_MODEL,
        help=(
            "HuggingFace model id (or local path) of the base model to "
            "fine-tune. Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=_DEFAULT_DATASET,
        help=(
            "Path to a JSONL file consumable by datasets.load_dataset. "
            "Defaults to %(default)s."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse once, dispatch once. ``run_training`` is the only thing
    # the CLI calls — every other helper above exists to support it.
    _args = _parse_args()
    run_training(model_id=_args.model, dataset_path=_args.dataset)
