# Data governance

Training, validation, test, and benchmark rows carry an explicit `data_role`.
The loader validates that role at each boundary. Anthropic rows are protected
by default: only explicitly opted-in training/validation rows may reach the
optimizer, and held-out benchmark rows are always rejected.

## Data classes

| Class | Typical path | Allowed use |
|---|---|---|
| Offline smoke fixture | `data/processed/training_pool.jsonl` | tests and `smoke` profile only |
| Imported user data | `data/generated/user/<name>/` | real train/validation/test splits |
| Prepared source data | operator-generated governed directory | real train/validation/test splits |
| Anthropic development | `data/anthropic_experiment/{train,validation}.jsonl` | governed opt-in development |
| Anthropic benchmark | `data/anthropic_experiment/benchmark.jsonl` | held-out before/after evaluation only |

The committed 12-row pool is not suitable for a published training result.

## User-owned dataset import

`syco import-data` accepts CSV, a JSON array, or JSONL. It supports either the
complete canonical schema or a simple binary-choice schema. Required simple
fields are:

- `question` or `prompt`;
- `option_a` and `option_b`;
- `target_option` or `answer` (`A` or `B`).

Optional preference, validity, pressure, topic, and group fields preserve the
behavioral distinction between resisting an invalid preference and accepting a
valid correction. The importer:

1. hashes the original file;
2. normalizes and schema-validates every row;
3. rejects non-training roles, benchmark-only rows, and Anthropic rows without explicit opt-in;
4. groups related rows before deterministic splitting;
5. performs duplicate and leakage checks;
6. writes immutable pool, split, ID, and manifest artifacts;
7. refuses to overwrite an existing dataset version.

The operator must provide an honest license identifier and, when available,
the source URL. Importing data does not grant permission to train on it.

## Mandatory rejection rules

The pipeline refuses to train when any of these is true:

- a row has `data_role="benchmark"` or `metadata.benchmark_only=true`;
- an Anthropic development row lacks `anthropic_training_opt_in=true` or has `benchmark_only` other than `false`;
- a real profile receives a smoke fixture;
- a pool mixes fixtures and real rows;
- example or group IDs overlap across protected splits;
- the current source hash differs from the split/import manifest;
- schema or manifest versions are unsupported;
- Kaggle staging sees overlapping train/validation/benchmark IDs.

## Anthropic experiment preparation

The networked preparation command pins the Hub revision, converts rows to the
canonical benchmark schema, and writes provenance. Its default source commit is
`d533f626cc321c92175a58ee570aa3cdb87238d1`. It must be run explicitly:

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl.data_prep.prepare_anthropic_experiment --seed 42
```

The command requires 30,168 rows and writes immutable 24,134/3,017/3,017
train/validation/benchmark partitions. The manifest records the source pin,
license, counts, split/file hashes, ID hashes, and all pairwise overlap counts.

## Publication requirements

Every reported result must retain:

- input source, revision, license, source URL, and SHA-256;
- immutable train/validation/test identifiers;
- benchmark revision/hash and leakage result;
- row counts and rejected-row explanation;
- exact model revision, prompt condition, seed, and decoding settings;
- raw responses, including invalid/refused/truncated outputs.

Do not move held-out benchmark examples into development, manually move rows
between splits, or report percentages computed outside the saved artifacts.
