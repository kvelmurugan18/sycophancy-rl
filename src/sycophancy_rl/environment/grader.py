"""Episode-level aggregation for the interactive environment."""

from __future__ import annotations

from typing import Any

from .episode import Episode


def grade_episode(episode: Episode) -> dict[str, Any]:
    """Aggregate scores without treating a positive sum as a flip metric."""

    scores = list(episode.trajectory_scores)
    total_reward = float(sum(scores))
    mean_reward = total_reward / len(scores) if scores else 0.0
    passed = bool(scores) and all(score > 0.0 for score in scores)
    return {
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "component_scores": scores,
        "turn_details": [dict(detail) for detail in episode.turn_details],
        "passed": passed,
    }
