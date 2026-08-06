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

### Changed

- Application and API code now use the conventional installable
  `src/sycophancy_rl` package layout with one public CLI entry point.
- Benchmark model memory is explicitly released between pipeline stages.
- Registered revisions are resolved consistently across training, benchmarking,
  local plans, and Kaggle plans.
- Trainer dependencies are pinned in environment-specific lock files.
- Operator documentation now distinguishes offline verification, configured
  models, and real external GPU validation.

### Removed

- Obsolete split baseline/training wrapper scripts superseded by the canonical
  complete-run container scripts.
- Historical audit-count document that could not be verified from artifacts.
- Unused legacy environment/token loader and its unnecessary runtime dependency.
