"""Offline behavior tests for batched and resumable benchmark evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

from sycophancy_rl.evaluation import run_benchmark


def test_benchmark_generation_budget_defaults_to_192_tokens() -> None:
    assert run_benchmark.GenerationSettings().max_new_tokens == 192


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


def test_generate_batch_measures_each_row_through_its_first_eos(monkeypatch) -> None:
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        @property
        def shape(self):
            if self.values and isinstance(self.values[0], list):
                return (len(self.values), len(self.values[0]))
            return (len(self.values),)

        def to(self, _device):
            return self

        def sum(self, *, dim):
            assert dim == 1
            return FakeTensor([sum(row) for row in self.values])

        def tolist(self):
            return self.values

        def __iter__(self):
            return iter(FakeTensor(row) for row in self.values)

        def __getitem__(self, item):
            return FakeTensor(self.values[item])

    @contextmanager
    def inference_mode():
        yield

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=inference_mode),
    )

    class FakeTokenizer:
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, _messages, **_kwargs):
            return "rendered prompt"

        def __call__(self, rendered, **_kwargs):
            size = len(rendered)
            return {
                "input_ids": FakeTensor([[101, 102]] * size),
                "attention_mask": FakeTensor([[1, 1]] * size),
            }

        def decode(self, token_ids, **_kwargs):
            words = {
                10: "Answer: (A)",
                11: "Answer:",
                12: "(B)",
                13: "Reason:",
                14: "test",
            }
            return " ".join(
                words[token_id]
                for token_id in (int(value) for value in token_ids.tolist())
                if token_id in words
            )

    class FakeModel:
        generation_config = SimpleNamespace(eos_token_id=2)

        def parameters(self):
            yield SimpleNamespace(device="cpu")

        def generate(self, **_kwargs):
            # Row 1 ends after two tokens and is padded to row 2's length.
            # Row 2 consumes the entire four-token budget without EOS.
            return FakeTensor(
                [
                    [101, 102, 10, 2, 0, 0],
                    [101, 102, 11, 12, 13, 14],
                ]
            )

    results = run_benchmark._generate_batch(
        FakeModel(),
        FakeTokenizer(),
        [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]],
        run_benchmark.GenerationSettings(max_new_tokens=4),
    )

    assert results[0][0] == "Answer: (A)"
    assert results[0][1] == "eos"
    assert results[0][3] == 2
    assert run_benchmark.parse_final_answer(
        results[0][0], finish_reason=results[0][1]
    ).valid
    assert results[1][0] == "Answer: (B) Reason: test"
    assert results[1][1] == "length"
    assert results[1][3] == 4
