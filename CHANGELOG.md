# Changelog

This project follows semantic versioning. Changes not yet released are recorded
below.

## Unreleased

### Added

- Governed CSV/JSON/JSONL user-dataset importer with immutable group-aware
  splits, source hashes, licenses, and manifests.
- Commit-pinned Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 registry
  profiles with explicit `configured` status.
- `qlora_7b_16gb` NF4 QLoRA/GRPO profile and strict custom-model capability
  declarations.
- Complete local-container services and scripts for before/train/after/compare.
- Self-contained Kaggle code staging and governed dataset staging with copied
  provenance manifests.
- Tests for the importer, 7B plans, model compatibility, Kaggle staging, CLI
  dispatch, leakage, manifests, rewards, metrics, and server boundaries.
- A post-hoc archive rescoring command that applies the current answer parser
  to saved before/after responses without loading or retraining a model.

### Changed

- Application and API code now use the conventional installable
  `src/sycophancy_rl` package layout with one public CLI entry point.
- Benchmark model memory is explicitly released between pipeline stages.
- Registered revisions are resolved consistently across training, benchmarking,
  local plans, and Kaggle plans.
- Trainer dependencies are pinned in environment-specific lock files.
- Operator documentation now distinguishes offline verification, configured
  models, and real external GPU validation.
- Packaging now declares Apache-2.0, accurate repository URLs, and the
  CI-verified Python 3.10-3.12 range.
- Dependency pins were advanced past known PyTorch, Gradio/Pillow, Starlette,
  and setuptools advisories; Kaggle continues to preserve its preinstalled
  CUDA-compatible PyTorch wheel.
- Tone, answer parsing, pushback diversity, and leakage detection now have
  contextual/tiered behavior with focused regression coverage.
- All model profiles are truthfully marked `configured` until a retained GPU
  validation artifact exists.
- Paired comparisons now bind benchmark hashes, dataset provenance, labels,
  prompts, seeds, revisions, and per-example generation settings.
- The reference dependency set now resolves across Python 3.10-3.12, and CI
  checks the complete source tree, deployment locks, built wheel, and Gradio
  application construction.
- Response-leading `(A)` and `(B)` choices are now accepted by the answer
  parser, with regression coverage for archive rescoring.
- Batched benchmarks now determine completion length and truncation per row at
  its first EOS token instead of using the padded batch width.
- The default benchmark completion budget is now 192 tokens; publishable plans
  reject budgets below 128 and reduced profile step counts.
- All eleven invalid-pressure categories now contain at least five distinct
  phrasings, and Kaggle staging tests no longer depend on gitignored datasets.

### Removed

- Obsolete split baseline/training wrapper scripts superseded by the canonical
  complete-run container scripts.
- Historical audit-count document that could not be verified from artifacts.
- Unused legacy environment/token loader and its unnecessary runtime dependency.
