from sycophancy_rl.data_prep.import_openr1 import (
    convert_openr1_row,
    make_distractor,
    reservoir_sample,
)


def upstream(index: int, answer: str = "42") -> dict:
    return {"uuid": f"q-{index}", "problem": f"Compute value {index}.", "answer": answer,
            "source": "fixture", "problem_type": "arithmetic"}


def test_numeric_distractors_are_different_and_deterministic() -> None:
    assert make_distractor("42") == "43"
    assert make_distractor("-2") == "-3"
    assert make_distractor(r"\frac{2}{3}") == r"\frac{3}{3}"
    assert make_distractor(r"\boxed{42}") == "43"
    assert make_distractor(r"$\dfrac{1}{2}$") == r"\frac{2}{2}"
    assert make_distractor("x+y") is None


def test_conversion_uses_no_teacher_generation() -> None:
    row = upstream(1)
    row["generations"] = ["private generated solution"]
    converted = convert_openr1_row(row, seed=42)
    assert converted is not None
    assert "private generated solution" not in str(converted)
    assert converted["target_option"] in {"A", "B"}
    assert converted["metadata"]["openr1_answer"] == "42"


def test_reservoir_sampling_is_reproducible() -> None:
    rows = [upstream(index) for index in range(20)]
    first, eligible = reservoir_sample(rows, count=5, seed=7)
    second, _ = reservoir_sample(rows, count=5, seed=7)
    assert eligible == 20
    assert [row["example_id"] for row in first] == [row["example_id"] for row in second]
