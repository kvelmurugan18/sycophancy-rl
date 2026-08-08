# Contributing

Use a short-lived topic branch and a focused pull request. Preserve existing
user data and generated experiment artifacts; never commit secrets, downloaded
model weights, private datasets, or `.env`.

## Required checks

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m pip_audit -r requirements.txt
docker compose -f docker-compose.trainer.yml config --quiet
```

Tests must remain offline: mock model/trainer/network boundaries and use the
committed fixtures. A new model registry entry requires an exact Hub commit SHA,
architecture/quantization contract, truthful validation status, and tests.

## Data and experiment rules

- Never hand-edit committed smoke fixtures or immutable imported datasets.
- Never allow benchmark/evaluation rows into training or validation.
- Preserve group-aware split identity, source hashes, and manifests.
- Any prompt, parser, reward, decoding, or model revision change creates a new
  experiment identity; do not compare it as though only the adapter changed.
- New public metrics must be derived from saved artifacts and include invalid
  outputs in denominators.

## Code style

Ruff is the canonical formatter/linter policy. Public functions should document
their behavioral contract. Add tests at the public boundary and include failure
paths for validation/security-sensitive changes.
