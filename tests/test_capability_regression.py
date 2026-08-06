"""Capability-regression comparison tests."""

from sycophancy_rl.evaluation.capability_regression import compare_capabilities


def payload(arc: float, truthful: float) -> dict:
    return {
        "results": {
            "arc_easy": {"acc_norm,none": arc},
            "truthfulqa_mc2": {"acc,none": truthful},
        }
    }


def test_capability_threshold_pass_and_fail() -> None:
    passing = compare_capabilities(
        payload(0.70, 0.50),
        payload(0.69, 0.49),
        maximum_allowed_drop=0.02,
    )
    failing = compare_capabilities(
        payload(0.70, 0.50),
        payload(0.60, 0.49),
        maximum_allowed_drop=0.02,
    )

    assert passing["passed"] is True
    assert failing["passed"] is False
