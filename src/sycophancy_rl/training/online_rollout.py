"""TRL 1.8 custom rollout adapter for real multi-turn episodes.

The original governed prompt is the TRL prompt. Every later assistant and
environment token is returned as one continuous completion. ``env_mask`` is
1 for sampled assistant tokens and 0 for environment/chat-template suffix
tokens. TRL keeps all tokens in attention but excludes environment tokens
from GRPO loss, so every sampled assistant turn receives the trajectory
advantage and no user/environment token does.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sycophancy_rl.environment.online import OnlineSycophancyEnvironment, append_trajectory

_ID_KEY = "_syco_example_id"
_ROLE_KEY = "_syco_data_role"
_SUPPORTED_TRL_VERSION = "1.8.0"


def trajectory_reward_func(
    prompts: list[Any],
    completions: list[Any],
    *,
    trajectory_reward: list[float] | None = None,
    trajectory_pairing_key: list[str] | None = None,
    **_kwargs: Any,
) -> list[float]:
    """Return trajectory rewards only when all per-sample fields still align."""

    if trajectory_reward is None or trajectory_pairing_key is None:
        raise ValueError("online rollout did not provide reward/pairing fields")
    lengths = {len(prompts), len(completions), len(trajectory_reward), len(trajectory_pairing_key)}
    if len(lengths) != 1:
        raise ValueError(
            "Online reward-to-completion alignment failed: "
            f"prompts={len(prompts)}, completions={len(completions)}, "
            f"rewards={len(trajectory_reward)}, keys={len(trajectory_pairing_key)}."
        )
    if len(set(trajectory_pairing_key)) != len(trajectory_pairing_key):
        raise ValueError("Online rollout produced duplicate trajectory pairing keys.")
    return [float(value) for value in trajectory_reward]


def _clean_prompt_and_identity(prompt: Any) -> tuple[list[dict[str, str]], tuple[str, str]]:
    if not isinstance(prompt, list) or not prompt or not isinstance(prompt[-1], dict):
        raise ValueError("Online rollout requires a non-empty structured chat prompt.")
    final = prompt[-1]
    example_id = final.get(_ID_KEY)
    data_role = final.get(_ROLE_KEY)
    if not isinstance(example_id, str) or not example_id:
        raise ValueError(f"Online prompt is missing reserved identity key {_ID_KEY!r}.")
    if data_role not in {"training", "validation"}:
        raise ValueError(f"Online prompt has invalid governed role {data_role!r}.")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(prompt):
        role = message.get("role") if isinstance(message, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"Online prompt message {index} is malformed.")
        messages.append({"role": role, "content": content})
    return messages, (str(data_role), example_id)


def _render_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def _validate_rollout_result(result: dict[str, list[Any]], expected: int) -> None:
    required = (
        "prompt_ids",
        "completion_ids",
        "logprobs",
        "env_mask",
        "trajectory_reward",
        "trajectory_json",
        "trajectory_pairing_key",
        "credit_assignment_json",
    )
    for key in required:
        if len(result[key]) != expected:
            raise AssertionError(f"{key} has {len(result[key])} rows; expected {expected}.")
    for index, (tokens, logprobs, mask) in enumerate(
        zip(result["completion_ids"], result["logprobs"], result["env_mask"], strict=True)
    ):
        if not (len(tokens) == len(logprobs) == len(mask)):
            raise AssertionError(
                f"rollout {index} token/logprob/mask lengths differ: "
                f"{len(tokens)}/{len(logprobs)}/{len(mask)}"
            )
        if not tokens or not any(mask):
            raise AssertionError(f"rollout {index} has no sampled assistant tokens")
        if any(value not in {0, 1} for value in mask):
            raise AssertionError(f"rollout {index} has a non-binary env_mask")
    keys = result["trajectory_pairing_key"]
    if len(set(keys)) != len(keys):
        raise AssertionError("trajectory pairing keys are not unique")


def _resolve_prompt_batch(
    prompts: list[Any],
    by_identity: dict[tuple[str, str], dict[str, Any]],
) -> list[tuple[list[dict[str, str]], tuple[str, str], dict[str, Any], int, str]]:
    """Resolve repeated TRL prompts in-place without text-keyed lookup."""

    ordinals: Counter[tuple[str, str]] = Counter()
    resolved = []
    for raw_prompt in prompts:
        messages, identity = _clean_prompt_and_identity(raw_prompt)
        source = by_identity.get(identity)
        if source is None:
            raise KeyError(f"No governed row matches online identity {identity!r}.")
        generation_index = ordinals[identity]
        ordinals[identity] += 1
        pairing_key = f"{identity[0]}:{identity[1]}:generation-{generation_index}"
        resolved.append((messages, identity, source, generation_index, pairing_key))
    return resolved


def make_online_rollout_func(
    rows: list[dict[str, Any]],
    *,
    artifact_path: Path,
    seed: int,
    max_pushback_turns: int,
):
    """Create the TRL 1.8 multi-turn rollout function.

    TRL's sampler already repeats each prompt ``num_generations`` times. This
    function emits exactly one rollout for each prompt entry, in the same
    order, and never performs a second accidental repetition.
    """

    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["data_role"]), str(row["example_id"]))
        if key in by_identity:
            raise ValueError(f"Duplicate governed rollout identity: {key!r}")
        by_identity[key] = row

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        import torch

        trl_version = importlib.metadata.version("trl")
        if trl_version != _SUPPORTED_TRL_VERSION:
            raise RuntimeError(
                f"Multi-turn env_mask contract is audited for TRL {_SUPPORTED_TRL_VERSION}; "
                f"found {trl_version}."
            )
        tokenizer = trainer.processing_class
        model = trainer.accelerator.unwrap_model(trainer.model)
        result: dict[str, list[Any]] = {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
            "env_mask": [],
            "trajectory_reward": [],
            "trajectory_json": [],
            "trajectory_pairing_key": [],
            "credit_assignment_json": [],
        }
        def generate(input_ids_list: list[int]) -> tuple[list[int], list[float], str]:
            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=model.device)
            attention_mask = torch.ones_like(input_ids)
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
            completion = generated.sequences[0, input_ids.shape[1] :]
            scores = tuple(generated.scores)
            if completion.numel() != len(scores):
                raise RuntimeError(
                    "Generation token/score mismatch; refusing unaudited sampling logprobs: "
                    f"tokens={completion.numel()}, scores={len(scores)}."
                )
            transition = model.compute_transition_scores(
                generated.sequences, scores, normalize_logits=True
            )[0]
            if transition.numel() != completion.numel():
                raise RuntimeError("compute_transition_scores returned the wrong token count.")
            token_ids = [int(token) for token in completion.tolist()]
            logprobs = [float(value) for value in transition.tolist()]
            if len(token_ids) != len(logprobs):
                raise RuntimeError("Sampled tokens and sampling logprobs are not one-to-one.")
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            return token_ids, logprobs, text

        resolved_prompts = _resolve_prompt_batch(prompts, by_identity)
        for prompt_index, resolved in enumerate(resolved_prompts):
            messages, identity, source, generation_index, pairing_key = resolved

            example = dict(source)
            example["prompt"] = [dict(message) for message in messages]
            env = OnlineSycophancyEnvironment(
                seed=seed + prompt_index * 10_000 + generation_index,
                max_pushback_turns=max_pushback_turns,
            )
            current = env.reset(example)
            prompt_ids = _render_ids(tokenizer, current)
            completion_ids: list[int] = []
            sampling_logprobs: list[float] = []
            env_mask: list[int] = []
            assistant_ranges: list[dict[str, int]] = []
            environment_ranges: list[dict[str, int]] = []

            while not env.is_done():
                current_ids = _render_ids(tokenizer, current)
                if current_ids != prompt_ids + completion_ids:
                    raise RuntimeError(
                        "Chat-template history did not preserve the exact sampled token prefix; "
                        "refusing misaligned multi-turn credit assignment."
                    )
                assistant_ids, assistant_logprobs, actual = generate(current_ids)
                start = len(completion_ids)
                completion_ids.extend(assistant_ids)
                sampling_logprobs.extend(assistant_logprobs)
                env_mask.extend([1] * len(assistant_ids))
                assistant_ranges.append(
                    {"turn": len(assistant_ranges) + 1, "start": start, "end": len(completion_ids)}
                )
                env.step(actual)
                current = env.build_next_prompt()
                if not env.is_done():
                    next_ids = _render_ids(tokenizer, current)
                    sampled_prefix = prompt_ids + completion_ids
                    if next_ids[: len(sampled_prefix)] != sampled_prefix:
                        raise RuntimeError(
                            "Decoded response could not be losslessly re-embedded by the chat template; "
                            "refusing to attach the reward to different tokens."
                        )
                    environment_ids = next_ids[len(sampled_prefix) :]
                    env_start = len(completion_ids)
                    completion_ids.extend(environment_ids)
                    # TRL 1.8 uses explicit zero sentinels for masked external
                    # tokens; env_mask ensures they never enter the loss.
                    sampling_logprobs.extend([0.0] * len(environment_ids))
                    env_mask.extend([0] * len(environment_ids))
                    environment_ranges.append(
                        {
                            "after_turn": len(assistant_ranges),
                            "start": env_start,
                            "end": len(completion_ids),
                        }
                    )

            trajectory = env.get_trajectory()
            completion_sha = hashlib.sha256(
                ",".join(str(token) for token in completion_ids).encode("ascii")
            ).hexdigest()
            credit = {
                "contract": "trl-1.8.0-continuous-completion-env-mask",
                "pairing_key": pairing_key,
                "data_role": identity[0],
                "example_id": identity[1],
                "generation_index": generation_index,
                "prompt_index": prompt_index,
                "completion_ids_sha256": completion_sha,
                "assistant_token_ranges": assistant_ranges,
                "environment_token_ranges": environment_ranges,
                "assistant_token_count": sum(env_mask),
                "environment_token_count": len(env_mask) - sum(env_mask),
                "environment_logprob_sentinel": 0.0,
                "all_assistant_turns_optimized": True,
                "environment_tokens_masked_from_loss": True,
            }
            trajectory.credit_assignment = credit
            reward = env.calculate_reward()
            trajectory_json = json.dumps(trajectory.to_dict(), sort_keys=True)
            append_trajectory(artifact_path, trajectory)

            result["prompt_ids"].append(prompt_ids)
            result["completion_ids"].append(completion_ids)
            result["logprobs"].append(sampling_logprobs)
            result["env_mask"].append(env_mask)
            result["trajectory_reward"].append(reward)
            result["trajectory_json"].append(trajectory_json)
            result["trajectory_pairing_key"].append(pairing_key)
            result["credit_assignment_json"].append(json.dumps(credit, sort_keys=True))

        _validate_rollout_result(result, len(prompts))
        return result

    return rollout_func
