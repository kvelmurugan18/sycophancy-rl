# Evaluation-only benchmarks

Run `python -m sycophancy_rl.data_prep.prepare_anthropic_benchmark` to create the
revision-pinned Anthropic sycophancy benchmark and its provenance manifest.
Generated benchmark files are evaluation-only and must never be copied into
`data/processed` or `data/splits`.
