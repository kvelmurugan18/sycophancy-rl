# Generated artifacts

Runtime outputs are intentionally excluded from Git:

- `checkpoints/<run>/`: adapter checkpoints, manifests, metrics, status
- `evaluations/<run>/responses.jsonl`: every raw model response and parse result
- `evaluations/<run>/summary.json`: metrics and confidence intervals
- `evaluations/<run>/manifest.json`: model, prompt, generation, hardware provenance
- `comparisons/`: paired before/after reports and significance tests
- `human_review/`: blinded annotation sheets and agreement reports

Published results must be derived from these files; do not type percentages or
average rewards manually into reports.
