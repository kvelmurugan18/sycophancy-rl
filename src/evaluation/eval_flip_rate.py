"""
Post-training flip-rate evaluator.

This script parses the post-training evaluation log produced by the
training pipeline (the JSON list at ``outputs/eval_results.json``, where
each entry is one finished episode carrying a ``"passed"`` boolean from
``grade_episode``) and reports the **Flip Rate** — the headline metric
for measuring sycophancy reduction.

A "flip" here is defined as any episode where the assistant **failed to
hold its ground** under pushback, i.e. ``"passed" is False``. Tracking
this number before vs. after GRPO training is how we quantify whether
the policy actually learned to resist sycophantic pressure: a
well-trained model should have a flip rate trending toward zero on the
eval split, and a large delta vs. the pre-training baseline is the
single most important number in the training report.

The script is intentionally minimal — it reads the JSON, counts, prints
— so it can be run as a quick post-training smoke check without
spinning up the FastAPI server or re-running evaluation.
"""

import json
from pathlib import Path


def calculate_flip_rate(results_path: str = "outputs/eval_results.json") -> None:
    """Parse eval results and print a human-readable flip-rate report.

    Reads the JSON list at ``results_path`` (each entry is a finished
    episode with a ``"passed"`` boolean), counts the entries where
    ``"passed" is False`` as flips, and prints a three-line terminal
    summary: total episodes evaluated, total flips (caved to the
    user), and the flip rate as a percentage.

    If the file does not exist, prints a one-line nudge pointing the
    user at the training pipeline and returns without error — so this
    script is safe to wire into a Makefile / CI step that runs before
    evaluation has been performed.

    Args:
        results_path: Path to the evaluation results JSON file.
            Defaults to ``"outputs/eval_results.json"`` (the convention
            used by the rest of the project).
    """
    path = Path(results_path)

    # Pre-read existence check: an evaluation pass hasn't been run yet
    # is an expected state (a fresh checkout, a CI step that runs
    # before training, etc.), so we surface a helpful message and bail
    # cleanly rather than letting ``open()`` raise a FileNotFoundError.
    if not path.exists():
        print(
            f"No evaluation results found yet at '{results_path}'. "
            "Run an evaluation pass first."
        )
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Treat every "passed" == False entry as a flip. This intentionally
    # inverts ``grade_episode``'s ``total_reward > 0.0`` rule: a failed
    # episode means the assistant gave in to pushback on at least one
    # turn, which is exactly the behaviour we want to track down.
    total_episodes = len(data)
    total_flips = sum(1 for episode in data if not episode.get("passed", False))

    # Guard against an empty list (the file existed but no episodes
    # were evaluated) so we don't divide by zero.
    if total_episodes == 0:
        print("Evaluation results file is empty — no episodes to report.")
        return

    flip_rate_percentage = (total_flips / total_episodes) * 100

    # Three-line report. The labels are left-padded to align the
    # colons, which makes the output easy to skim in a CI log.
    print("=" * 50)
    print("  Post-Training Flip Rate Report")
    print("=" * 50)
    print(f"  Total Episodes Evaluated : {total_episodes}")
    print(f"  Total Flips (Caved)      : {total_flips}")
    print(f"  Final Flip Rate          : {flip_rate_percentage:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    calculate_flip_rate()
