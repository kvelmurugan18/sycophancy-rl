# Data layout

- `examples/` contains documented user-import examples only.
- `processed/` and `splits/` contain the committed offline smoke fixture and its
  deterministic IDs/manifests.
- `generated/` contains immutable user imports or prepared real data and is
  ignored by Git.
- `benchmarks/` contains instructions plus generated evaluation-only benchmark
  artifacts, which are ignored by Git.
- `cache/` contains recoverable source-download caches and is ignored by Git.

Never copy benchmark rows into training directories. Real data, downloaded
benchmarks, and private sources must not be committed. See
`docs/data_governance.md` for the enforced contract.
