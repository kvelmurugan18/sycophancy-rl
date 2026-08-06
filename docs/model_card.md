# Model card template

This repository trains adapters; it does not ship a trained model. Complete a
copy of this card from saved run artifacts before publishing any adapter or
result. Do not fill fields from memory or terminal screenshots.

## Identity

- Adapter name: `<run-id>`
- Base model and exact revision: `<from experiment.json>`
- Adapter checksum: `<from completed manifest>`
- Method/profile: `<for example QLoRA + GRPO, qlora_7b_16gb>`
- Seed: `<seed>`
- Code commit: `<git commit or unavailable>`
- Training hardware and duration: `<from manifest>`

## Intended use

The adapter is intended for research on whether a model maintains an
evidence-based binary-choice answer under user pressure. It is not a factuality
system, safety classifier, general assistant guarantee, or production decision
maker.

## Data

- Training dataset name, license, source URL, SHA-256, and row count:
  `<from import/source and split manifests>`
- Validation split SHA-256 and row count: `<from manifests>`
- Evaluation benchmark and exact revision: `<from benchmark manifest>`
- Leakage audit result: `<pass/fail plus artifact path>`

State explicitly that Anthropic model-written sycophancy examples were used for
evaluation only and were rejected by the trainer.

## Evaluation

Record paired values from generated reports:

| Metric | Before | After | Change / interval |
|---|---:|---:|---:|
| Target accuracy | | | |
| Sycophantic-answer rate | | | |
| Independent-answer rate | | | |
| Invalid-answer rate | | | |
| Unnecessary-disagreement rate | | | |
| Capability-regression metrics | | | |

Include the paired McNemar result, Wilson 95% intervals, benchmark row count,
prompt condition, decoding settings, and all invalid outputs in denominators.
Separate smoke subsets from full benchmark results.

## Limitations and risks

- Binary choice behavior does not generalize automatically to open-ended chat.
- Benchmark and training-source artifacts can inflate results.
- GRPO may reduce one failure mode while degrading calibration, helpfulness, or
  unrelated capability.
- Generated explanations are not verified reasoning traces.
- Results from one seed, prompt condition, or hardware run are insufficient for
  a broad effectiveness claim.

## Reproduction

Link the immutable experiment plan, before/after manifests, raw responses,
comparison report, adapter files, package lock, and exact commands. A result is
not reproducible if these artifacts are missing or their checksums differ.
