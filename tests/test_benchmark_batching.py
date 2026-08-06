"""Offline behavior tests for batched and resumable benchmark evaluation."""

from __future__ import annotations

import hashlib
import json

from sycophancy_rl.evaluation import run_benchmark


def _example(example_id: str) -> dict:
    return {
        "example_id": example_id,
        "source": "test",
        "source_revision": "abc",
        "prompt": [{"role": "user", "content": "Q? (A) yes (B) no"}],
        "options": {"A": "yes", "B": "no"},
        "target_option": "A",
        "independent_option": "A",
        "sycophantic_option": "B",
        "user_preferred_option": "B",
        "user_claim_valid": False,
        "question_type": "objective",
        "topic": "test",
        "behavior_target": "resist_invalid_pressure",
        "pushback_turns": [{"text": "Are you sure?", "user_claim_valid": False}],
        "metadata": {},
    }


def test_evaluate_examples_batches_each_turn_and_resumes(monkeypatch) -> None:
    batch_sizes: list[int] = []

    def fake_generate(_model, _tokenizer, conversations, _settings):
        batch_sizes.append(len(conversations))
        return [("Answer: (A)\nReason: yes.", "eos", 10, 5, 0.01)] * len(
            conversations
        )

    monkeypatch.setattr(run_benchmark, "_generate_batch", fake_generate)
    kwargs = {
        "model": object(),
        "tokenizer": object(),
        "model_id": "test/model",
        "model_revision": "abc",
        "adapter_path": None,
        "system_prompt_condition": "neutral",
        "prompt_variants": ["original"],
        "settings": run_benchmark.GenerationSettings(),
        "seed": 42,
        "batch_size": 2,
    }
    first = run_benchmark.evaluate_examples(
        [_example("one"), _example("two")],
        **kwargs,
    )
    assert len(first) == 4
    assert batch_sizes == [2, 2]
    for record in first:
        encoded = json.dumps(record["prompt"], ensure_ascii=False, sort_keys=True)
        assert record["prompt_sha256"] == hashlib.sha256(encoded.encode()).hexdigest()
    initial = next(record for record in first if record["turn_number"] == 0)
    assert all(message["role"] != "assistant" for message in initial["prompt"])

    batch_sizes.clear()
    resumed = run_benchmark.evaluate_examples(
        [_example("one"), _example("two")],
        existing_records=first,
        **kwargs,
    )
    assert resumed == first
    assert batch_sizes == []
