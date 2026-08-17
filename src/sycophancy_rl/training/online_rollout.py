"""TRL ``rollout_func`` adapter for actual multi-turn sycophancy episodes.

The effective GRPO action is the final assistant response conditioned on a
dynamic prompt containing the policy's exact first response and environment
pushback. The trajectory reward is forwarded to a dedicated reward callback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sycophancy_rl.environment.online import (
    OnlineSycophancyEnvironment,
    append_trajectory,
)


def trajectory_reward_func(
    prompts: list[Any],
    completions: list[Any],
    *,
    trajectory_reward: list[float] | None = None,
    **_kwargs: Any,
) -> list[float]:
    """Return rewards emitted by the online environment."""
    del prompts, completions
    if trajectory_reward is None:
        raise ValueError("online rollout did not provide trajectory_reward")
    return [float(value) for value in trajectory_reward]


def make_online_rollout_func(
    rows: list[dict[str, Any]],
    *,
    artifact_path: Path,
    seed: int,
    max_pushback_turns: int,
):
    """Create a version-pinned TRL custom rollout function.

    TRL 1.8's experimental contract requires prompt IDs, completion IDs and
    sampling log-probabilities. Extra per-completion trajectory fields are
    forwarded to reward functions.
    """

    by_question = {
        str(row["prompt"][-1]["content"]): row
        for row in rows
    }

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        import torch

        tokenizer = trainer.processing_class
        model = trainer.accelerator.unwrap_model(trainer.model)
        generation_count = (
            trainer.args.num_generations_eval
            if not model.training
            else trainer.args.num_generations
        )
        result: dict[str, list[Any]] = {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
            "trajectory_reward": [],
            "trajectory_json": [],
        }

        def generate(messages: list[dict[str, str]]) -> tuple[list[int], list[int], list[float], str]:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=trainer.args.max_completion_length,
                    do_sample=True,
                    temperature=trainer.args.temperature,
                    top_p=trainer.args.top_p,
                    top_k=trainer.args.top_k,
                    repetition_penalty=trainer.args.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            completion = generated.sequences[0, input_ids.shape[1]:]
            logprobs = [
                float(torch.log_softmax(scores[0], dim=-1)[token].item())
                for scores, token in zip(generated.scores, completion, strict=True)
            ]
            text = tokenizer.decode(completion, skip_special_tokens=True)
            return input_ids[0].tolist(), completion.tolist(), logprobs, text

        for prompt_index, prompt in enumerate(prompts):
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
            question = str(messages[-1]["content"])
            source = by_question.get(question)
            if source is None:
                raise KeyError(f"No governed row matches online prompt: {question[:80]!r}")
            for generation_index in range(generation_count):
                example = dict(source)
                example["prompt"] = [dict(message) for message in messages]
                env = OnlineSycophancyEnvironment(
                    seed=seed + prompt_index * 10_000 + generation_index,
                    max_pushback_turns=max_pushback_turns,
                )
                current = env.reset(example)
                final_prompt_ids: list[int] = []
                final_completion_ids: list[int] = []
                final_logprobs: list[float] = []
                while not env.is_done():
                    prompt_ids, completion_ids, logprobs, actual = generate(current)
                    env.step(actual)
                    final_prompt_ids = prompt_ids
                    final_completion_ids = completion_ids
                    final_logprobs = logprobs
                    current = env.build_next_prompt()
                trajectory = env.get_trajectory()
                append_trajectory(artifact_path, trajectory)
                result["prompt_ids"].append(final_prompt_ids)
                result["completion_ids"].append(final_completion_ids)
                result["logprobs"].append(final_logprobs)
                result["trajectory_reward"].append(env.calculate_reward())
                result["trajectory_json"].append(json.dumps(trajectory.to_dict(), sort_keys=True))
        return result

    return rollout_func
