"""Local JSONL experiment tracking and validation-based early stopping."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import TrainerCallback
else:
    try:
        from transformers import TrainerCallback
    except ImportError:
        class TrainerCallback:  # type: ignore[no-redef]
            pass


class JsonlTrackingCallback(TrainerCallback):
    """Append every trainer log event to a machine-readable local file."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "step": state.global_step,
            "epoch": state.epoch,
            **(logs or {}),
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


class RewardEarlyStoppingCallback(TrainerCallback):
    """Stop after validation reward fails to improve for ``patience`` checks."""

    def __init__(self, patience: int = 3, minimum_delta: float = 0.001) -> None:
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.best_reward: float | None = None
        self.best_step: int | None = None
        self.bad_evaluations = 0

    @staticmethod
    def _reward_metric(metrics: dict[str, Any]) -> float | None:
        preferred = ("eval_reward", "eval_rewards/mean", "eval_reward/mean")
        for key in preferred:
            if key in metrics:
                return float(metrics[key])
        candidates = [
            float(value)
            for key, value in metrics.items()
            if key.startswith("eval_")
            and "reward" in key
            and isinstance(value, (int, float))
        ]
        return max(candidates) if candidates else None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        reward = self._reward_metric(metrics or {})
        if reward is None:
            return control
        if self.best_reward is None or reward > self.best_reward + self.minimum_delta:
            self.best_reward = reward
            self.best_step = state.global_step
            self.bad_evaluations = 0
            control.should_save = True
        else:
            self.bad_evaluations += 1
            if self.bad_evaluations >= self.patience:
                control.should_training_stop = True
        return control
