# Operator scripts

- `setup.ps1` / `setup.sh` create the virtual environment, install pinned
  dependencies, preserve existing data, prepare smoke fixtures when absent, and
  run offline checks.
- `run_local_container.ps1` / `run_local_container.sh` execute the complete
  baseline, training, adapter benchmark, and comparison workflow through the
  isolated trainer Compose file.

Scripts are thin entry points. Validation, model selection, training, and
evaluation logic belongs in the Python package so local and Kaggle behavior do
not drift.
