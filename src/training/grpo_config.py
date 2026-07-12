"""GRPO training configuration — TRL hyperparameters for consumer GPUs.

This module is the **single source of truth for the GRPO training run's
hyperparameters**. The settings here are tuned for a constrained
hardware budget (a single 16 GB GPU — the upper bound of what
Kaggle / Colab free-tier and most consumer cards can supply) rather
than for a multi-node A100 cluster. Every default below was chosen
with one of three constraints in mind:

1. **Memory ceiling.** ``num_generations = 4`` and
   ``per_device_train_batch_size = 1`` are the two knobs that dominate
   VRAM use during GRPO (each generation is a full forward pass that
   must be kept in memory for the group-relative advantage). Cranking
   either of them up is the fastest way to OOM on a 16 GB card.
2. **Stability under RL fine-tuning.** ``learning_rate = 1e-6`` is
   deliberately tiny — RL on a generative model is famously prone to
   *reward hacking* and catastrophic forgetting if the LR is too
   high, and 1e-6 is the empirical sweet spot for KL-regularised
   GRPO on a 7-9B base. ``beta = 0.05`` is the KL coefficient that
   pins the policy close to the reference model so the optimiser
   cannot drift into gibberish just because gibberish scores well on
   the reward function.
3. **Prompt / completion budget.** ``max_prompt_length = 512`` and
   ``max_completion_length = 512`` together set the per-step token
   budget. The completion side is deliberately generous: the model
   is required (by the PRM Additive Veto in ``reward_fn.py``) to
   emit a ``<thought>...</thought>`` block *and* a final answer, and
   a tight completion cap would force the model to choose between
   the two.

Callers are expected to import :func:`get_grpo_config` and pass the
returned object straight to ``GRPOTrainer``. The factory pattern
keeps the defaults in one place and lets the training script (or a
notebook) override ``output_dir`` without rest of the file.
"""

# TRL's GRPOConfig is the dataclass that backs trl.GRPOTrainer. We
# import it lazily-ish at module top so any import error surfaces
# immediately at config-load time (better than failing mid-training).
from trl import GRPOConfig


def get_grpo_config(output_dir: str = "outputs/grpo_model") -> GRPOConfig:
    """Return a GRPOConfig tuned for a 16 GB single-GPU run.

    The defaults are deliberately conservative. Every value is
    commented in the implementation below with the *reason* it was
    chosen, so future readers can see which knob to turn when they
    have more (or less) VRAM.

    Args:
        output_dir: Filesystem path where TRL will write checkpoints,
            tokenizer config, and trainer state. Defaults to
            ``"outputs/grpo_model"`` — relative to wherever the
            training script is launched from. Override this when
            running multiple experiments so checkpoints don't
            collide.

    Returns:
        A fully populated :class:`trl.GRPOConfig` ready to hand to
        ``GRPOTrainer``.
    """
    return GRPOConfig(
        # --- I/O ---
        # Where checkpoints, tokenizer, and trainer state land. Caller-
        # supplied so multiple experiments don't overwrite each other.
        output_dir=output_dir,

        # --- Optimisation ---
        # Deliberately tiny LR. RL fine-tuning on a generative model
        # is brittle: a "normal" SFT learning rate (1e-4 or higher)
        # almost always either collapses the base model's competence
        # or sends the policy chasing reward-hacks. 1e-6 is the
        # empirical sweet spot for KL-regularised GRPO on a 7-9B base.
        learning_rate=1e-6,

        # KL-divergence coefficient. This is the leash that stops the
        # model from drifting off the reference distribution just
        # because some adversarial completion scored +1.0 on the
        # reward. 0.05 is the project default: small enough that the
        # model can still learn, large enough to keep generations
        # coherent. Bump to 0.1+ if you see the model emitting
        # gibberish that scores well; drop to 0.01 if learning stalls.
        beta=0.05,

        # --- Length budget ---
        # Cap on the prompt side. 512 tokens is enough for the
        # multi-turn pushback prompts in ``data/processed/`` without
        # truncation. Bumping this past ~1024 is a fast path to OOM
        # because the prompt is broadcast across ``num_generations``
        # completions.
        max_prompt_length=512,

        # Cap on the completion side. *Crucial*: the PRM Additive Veto
        # in ``reward_fn.py`` requires a ``<thought>...</thought>``
        # block *plus* a final answer. A 256-token cap would force
        # the model to choose between the two; 512 is the floor that
        # comfortably fits both for the prompts in this project.
        max_completion_length=512,

        # --- Batching / generation ---
        # TRL generates ``num_generations`` completions per prompt and
        # computes the group-relative advantage across them — that
        # is the whole point of GRPO vs. PPO. 8 would give lower-
        # variance advantages, but each generation is a full forward
        # pass held in memory, and 8 generations × 1 batch × 512
        # completion tokens is the cliff edge on 16 GB. 4 is the
        # highest safe default for Kaggle / Colab free-tier cards.
        num_generations=4,

        # Per-device micro batch. 1 is forced by the budget: anything
        # higher multiplies straight into VRAM.
        per_device_train_batch_size=1,

        # Gradient accumulation to recover an effective batch size
        # of 4 (= 1 × 4). The effective batch is what matters for
        # optimiser stability; the per-device size is what matters
        # for memory. Tune the *accumulation* when you want bigger
        # effective batches, not the per-device size.
        gradient_accumulation_steps=4,

        # --- Logging / checkpointing ---
        # Log the running loss / reward every 10 optimiser steps.
        # Cheap (just a print) and frequent enough to catch a
        # diverging run before you burn the rest of the budget.
        logging_steps=10,

        # Snapshot a checkpoint every 100 optimiser steps. 100 is a
        # compromise between disk usage and the ability to roll back
        # after a bad gradient step.
        save_steps=100,
    )
