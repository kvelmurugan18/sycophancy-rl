"""
Ablation study analyzer.

This script parses the post-ablation results file (the JSON list at
``outputs/ablation_results.json``) and reports the **Flip Rate per
experiment** so we can compare the relative contribution of each reward
pillar. The "ablation" is the standard ML practice of disabling one
component at a time and re-running evaluation to measure how much that
component was actually contributing.

In this project the ablation studies turn off each of the four reward
pillars — one at a time, plus a "Full Model" baseline — and re-evaluate
the policy under the reduced reward configuration. The expected result,
which the report this script prints is designed to surface at a glance,
is a **monotonic increase in flip rate as pillars are removed**: any
pillar whose removal does *not* increase flips is doing no work and can
be dropped; the configuration with the lowest flip rate is the "Full
Model" baseline. This is how we prove — quantitatively, not just
narratively — that all four pillars are required for optimal
sycophancy reduction.

The script is intentionally minimal: read JSON, group by
``experiment_name``, count flips per group, print a comparison table.
No pandas, no plotting — it should be runnable as a post-training
smoke check without any extra dependencies.
"""

import json
from collections import defaultdict
from pathlib import Path


def analyze_ablation(results_path: str = "outputs/ablation_results.json") -> None:
    """Group ablation results by experiment and print a flip-rate comparison.

    Reads the JSON list at ``results_path`` (each entry is an evaluated
    episode with an ``"experiment_name"`` and a ``"passed"`` boolean),
    groups the entries by experiment name, computes the flip rate for
    each group, and prints a four-column aligned table:

        Experiment Name | Total Episodes | Flips | Flip Rate %

    The table is the headline artifact of the ablation report: the
    reader should be able to glance at it and immediately see which
    pillar removals hurt performance the most (and which hurt the
    least, which is the same as "which pillar is doing the most
    work").

    If the file does not exist, prints a one-line nudge pointing the
    user at the ablation training pass and returns without error — so
    this script is safe to wire into a Makefile / CI step that runs
    before ablation has been performed.

    Args:
        results_path: Path to the ablation results JSON file. Defaults
            to ``"outputs/ablation_results.json"`` (the convention used
            by the rest of the project).
    """
    path = Path(results_path)

    # Pre-read existence check: an ablation pass hasn't been run yet is
    # an expected state (a fresh checkout, a CI step that runs before
    # training, etc.), so we surface a helpful message and bail cleanly
    # rather than letting ``open()`` raise FileNotFoundError.
    if not path.exists():
        print(
            f"No ablation results found yet at '{results_path}'. "
            "Run an ablation training pass first."
        )
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("Ablation results file is empty — no experiments to report.")
        return

    # Group episodes by experiment name. ``defaultdict(list)`` is the
    # right tool here because the per-experiment lists accumulate
    # incrementally and we don't want a separate "is this key already
    # present?" check on every append.
    groups: dict[str, list[dict]] = defaultdict(list)
    for episode in data:
        # Use a defensive .get so a malformed entry doesn't crash the
        # whole report — the offending entry is bucketed under an
        # explicit "unknown" label so the reviewer can see it.
        name = episode.get("experiment_name", "unknown")
        groups[name].append(episode)

    # Per-experiment aggregation. We compute the per-group flip rate
    # eagerly so the printing loop only deals with formatting.
    rows: list[tuple[str, int, int, float]] = []
    for name, episodes in groups.items():
        total = len(episodes)
        # ``.get("passed", False)`` defaults to False (i.e. counts as
        # a flip) for any entry missing the key — failing safe is the
        # right call for a sycophancy metric.
        flips = sum(1 for ep in episodes if not ep.get("passed", False))
        rate = (flips / total) * 100 if total > 0 else 0.0
        rows.append((name, total, flips, rate))

    # Sort by flip rate ascending so the table reads as a "best to
    # worst" leaderboard. The Full Model baseline (which should have
    # the lowest flip rate) ends up at the top — a quick visual sanity
    # check that the ablation did its job.
    rows.sort(key=lambda r: r[3])

    # --- Pretty-print the comparison table. ---
    headers = ("Experiment", "Episodes", "Flips", "Flip Rate %")
    name_w = max(len(headers[0]), max(len(r[0]) for r in rows))
    # Widths for the numeric columns are driven by the header so the
    # table stays tight even with very large episode counts.
    ep_w = max(len(headers[1]), len(str(max(r[1] for r in rows))))
    fl_w = max(len(headers[2]), len(str(max(r[2] for r in rows))))

    sep = "-" * (name_w + ep_w + fl_w + 22)  # 22 = 4 paddings + label chars

    print(sep)
    print(f"  {'Ablation Study — Flip Rate by Experiment':^{len(sep) - 4}}")
    print(sep)
    print(
        f"  {headers[0]:<{name_w}} | "
        f"{headers[1]:>{ep_w}} | "
        f"{headers[2]:>{fl_w}} | {headers[3]:>10}"
    )
    print(sep)
    for name, total, flips, rate in rows:
        print(
            f"  {name:<{name_w}} | "
            f"{total:>{ep_w}} | "
            f"{flips:>{fl_w}} | {rate:>9.2f}%"
        )
    print(sep)


if __name__ == "__main__":
    analyze_ablation()
