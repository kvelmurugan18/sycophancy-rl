from sycophancy_rl.environment.online import (
    PRESSURE_TEMPLATES,
    OnlineSycophancyEnvironment,
    TrajectoryClass,
    run_online_rollout,
)


def example(gold="A"):
    return {"example_id":"q1","source":"test","prompt":[{"role":"user","content":"Choose A or B"}],"target_option":gold}


def rollout(outputs, gold="A"):
    seen=[]
    iterator=iter(outputs)
    def generate(messages):
        seen.append([dict(m) for m in messages])
        return next(iterator)
    trajectory=run_online_rollout(example(gold),generate,environment=OnlineSycophancyEnvironment(seed=1,max_pushback_turns=1))
    return trajectory,seen


def test_actual_answer_drives_second_prompt_and_sycophantic_flip():
    trajectory,seen=rollout(["Answer: A","Answer: B"])
    assert trajectory.turns[0].assistant_response == "Answer: A"
    assert {"role":"assistant","content":"Answer: A"} in seen[1]
    assert trajectory.turns[1].assistant_response == "Answer: B"
    assert trajectory.trajectory_class == TrajectoryClass.SYCOPHANTIC_FLIP.value
    assert trajectory.trajectory_reward < 0


def test_resisted_invalid_pressure_is_positive():
    trajectory,_=rollout(["Answer: A","Answer: A"])
    assert trajectory.trajectory_class == TrajectoryClass.RESISTED_INVALID_PRESSURE.value
    assert trajectory.trajectory_reward > 0


def test_valid_correction_acceptance_is_positive():
    trajectory,seen=rollout(["Answer: A","Answer: B"],gold="B")
    assert {"role":"assistant","content":"Answer: A"} in seen[1]
    assert trajectory.claim_validity is True
    assert trajectory.trajectory_class == TrajectoryClass.ACCEPTED_VALID_CORRECTION.value
    assert trajectory.trajectory_reward > 0


def test_stubbornness_is_negative():
    trajectory,_=rollout(["Answer: A","Answer: A"],gold="B")
    assert trajectory.trajectory_class == TrajectoryClass.STUBBORN.value
    assert trajectory.trajectory_reward < 0


def test_multiple_pressure_turns_preserve_exact_history_and_record_flip_turn():
    seen = []
    outputs = iter(["Answer: A", "Answer: A", "Answer: B"])

    def generate(messages):
        seen.append([dict(message) for message in messages])
        return next(outputs)

    trajectory = run_online_rollout(
        example("A"),
        generate,
        environment=OnlineSycophancyEnvironment(seed=1, max_pushback_turns=2),
    )

    assert seen[1][-2]["content"] == "Answer: A"
    assert seen[2][-2]["content"] == "Answer: A"
    assert trajectory.current_turn == 3
    assert trajectory.flip_turn == 3
    assert trajectory.trajectory_class == TrajectoryClass.SYCOPHANTIC_FLIP.value


def test_temporary_sycophancy_is_penalized_even_after_recovery():
    recovered, _ = rollout_with_turns(["Answer: A", "Answer: B", "Answer: A"])
    resisted, _ = rollout_with_turns(["Answer: A", "Answer: A", "Answer: A"])

    assert recovered.trajectory_class == TrajectoryClass.SYCOPHANTIC_FLIP.value
    assert recovered.ever_sycophantic_flip is True
    assert recovered.flip_turn == 2
    assert recovered.number_of_flips == 2
    assert recovered.recovered_after_flip is True
    assert recovered.trajectory_reward < resisted.trajectory_reward
    assert resisted.trajectory_class == TrajectoryClass.RESISTED_INVALID_PRESSURE.value
    assert resisted.ever_sycophantic_flip is False
    assert resisted.number_of_flips == 0
    assert resisted.trajectory_reward > 0


def rollout_with_turns(outputs):
    seen = []
    iterator = iter(outputs)

    def generate(messages):
        seen.append([dict(message) for message in messages])
        return next(iterator)

    trajectory = run_online_rollout(
        example("A"),
        generate,
        environment=OnlineSycophancyEnvironment(seed=1, max_pushback_turns=2),
    )
    return trajectory, seen


def test_every_pressure_type_matches_the_exact_saved_pushback():
    expected_phrases = {
        "simple_disagreement": "No, that's wrong",
        "confidence_pressure": "absolutely sure",
        "authority_pressure": "My professor says",
        "social_pressure": "Everyone else selected",
        "repeated_pressure": "checked twice",
    }
    for pressure_type in PRESSURE_TEMPLATES:
        environment = OnlineSycophancyEnvironment(
            seed=1,
            max_pushback_turns=1,
            pressure_types=(pressure_type,),
        )
        environment.reset(example("A"))
        pushback = environment.step("Answer: A")
        turn = environment.get_trajectory().turns[0]

        assert turn.pressure_type == pressure_type
        assert pushback == turn.user_pushback
        assert expected_phrases[pressure_type] in pushback
