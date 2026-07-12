"""Episode-level grader.

This module computes the final episode-level grade by aggregating the
per-turn rewards emitted by :func:`src.reward.reward_fn.composite_reward_func`
into a single result. It is the logical home for any cross-turn trajectory
math — for example, a flip-flop penalty that down-weights episodes where the
assistant's answer changes between turns — before returning the final result
to the API client. The :class:`~src.environment.episode.Episode` itself stays
a pure per-turn ledger; the aggregation lives here so adding a new
trajectory-level signal (consistency, decay, multi-turn bonus) doesn't force
changes to the state container.
"""

from typing import Any, Dict

from .episode import Episode


def grade_episode(episode: Episode) -> Dict[str, Any]:
    """Aggregate an episode's per-turn rewards into a final grade.

    Sums the floats in ``episode.trajectory_scores`` into a single
    ``total_reward``, snapshots the per-turn scores into
    ``component_scores`` (a shallow copy so the caller can mutate the
    returned dict without affecting the episode state), and decides pass /
    fail with the simple ``total_reward > 0.0`` rule. The rule is
    intentionally minimal so a future revision can swap in a richer
    decision (e.g. cross-turn consistency bonus) without changing the
    public contract.

    Args:
        episode: The :class:`Episode` whose per-turn
            ``trajectory_scores`` should be aggregated. The episode's
            ``current_turn`` and ``is_done()`` status are not consulted
            here — callers that want a "grade only after the episode is
            done" check should enforce it at the route layer.

    Returns:
        A dict with three keys:

        - ``total_reward`` (``float``) — sum of every entry in
          ``episode.trajectory_scores``.
        - ``component_scores`` (``list[float]``) — shallow copy of
          ``episode.trajectory_scores`` in turn order.
        - ``passed`` (``bool``) — ``True`` iff ``total_reward > 0.0``.
    """
    component_scores: list = list(episode.trajectory_scores)
    total_reward: float = float(sum(component_scores))
    passed: bool = total_reward > 0.0

    return {
        "total_reward": total_reward,
        "component_scores": component_scores,
        "passed": passed,
    }
