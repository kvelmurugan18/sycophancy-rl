# Architecture

Sycophancy-RL is one installable Python package with explicit boundaries for
data, environment behavior, rewards, training, evaluation, experiment
orchestration, and the diagnostic API.

```text
sycophancy-rl/
|-- src/sycophancy_rl/       installable application package
|   |-- cli/                 public command dispatcher
|   |-- data_prep/           schemas, import, split, provenance, leakage checks
|   |-- environment/         episode and session state
|   |-- reward/              answer/tone reward components
|   |-- training/            model registry, GRPO configuration, reliability
|   |-- evaluation/          inference, metrics, comparisons, error analysis
|   |-- experiments/         frozen plans, manifests, local/Kaggle pipeline
|   |-- server/              local diagnostic FastAPI application
|   `-- utils/               shared parsing utilities
|-- tests/                   offline boundary and contract tests
|-- data/                    fixtures, examples, generated/cache conventions
|-- deploy/                  isolated runtime definitions and pinned locks
|-- scripts/                 setup and complete local-container entry points
|-- docs/                    design, operation, safety, and reporting contracts
`-- outputs/                 ignored runtime artifacts with a tracked guide
```

## Dependency direction

```text
CLI -> data preparation / experiment pipeline / evaluation
experiment pipeline -> training + evaluation + manifests
training -> model registry + reward + governed data
evaluation -> prompts + parser + metrics + governed benchmark
server -> environment + reward + canonical fixture store
```

Lower-level schema, parsing, and manifest modules must not import the CLI. The
diagnostic server is not part of the GPU trainer process. Local and Kaggle
runners share `ExperimentPlan` and `Pipeline`; runner-specific code changes
paths and lifecycle integration, not scientific behavior.

## Canonical runtime flow

```text
raw data -> governance -> controlled training examples
                                |
                                v
freeze and validate ExperimentPlan
                                |
                                v
BEFORE evaluation -> QLoRA/GRPO -> LoRA adapter -> AFTER evaluation
        |                                               |
        `------------ paired statistical comparison ----'
                                |
                                v
                         multi-seed reporting
```

The training loader accepts only governed training/validation roles. Anthropic
model-written sycophancy rows are benchmark-only. Before and after artifacts
must match model revision, benchmark identity, prompts, and generation settings.

## Runtime boundaries

- `docker-compose.trainer.yml` runs trusted training code as non-root with
  read-only data and root filesystems plus explicit writable cache/output mounts.
- `docker-compose.yml` runs only the localhost diagnostic API.
- `deploy/kaggle/runner.py` loads a frozen staged plan and writes only beneath
  `/kaggle/working`.
- No runner is a hostile-code sandbox. Untrusted code requires a disposable VM
  or microVM.

## Change rules

- New source belongs in the package matching its responsibility; do not create
  new top-level Python packages.
- Cross-stage behavior is changed in shared modules, not duplicated per runner.
- New models require exact revisions and truthful validation status.
- New datasets require role/schema validation, immutable split identity, and
  source provenance.
- Prompt, parser, reward, model, or decoding changes require a new experiment
  identity and focused tests.
