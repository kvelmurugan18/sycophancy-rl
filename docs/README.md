# Documentation index

Start with the root `README.md` for installation and the first complete run.
These documents separate design contracts from operator instructions.

## Design and safety

- `architecture.md` - package ownership, dependency direction, and runtime flow
- `experiment_protocol.md` - frozen before/train/after comparison contract
- `data_governance.md` - dataset roles, imports, leakage rejection, provenance
- `reward_design.md` - reward arithmetic, ablations, and limitations
- `limitations.md` - research scope, known failure modes, and external gates
- `supported_models.md` - registered revisions and custom-model requirements
- `threat_model.md` - isolation guarantees and explicit non-goals

## Operation

- `local_runner.md` - isolated Docker GPU workflow
- `kaggle_runner.md` - governed data and self-contained kernel staging
- `troubleshooting.md` - common failures and safe recovery

## Release and reporting

- `industrial_readiness.md` - verified gates versus external validation
- `licensing.md` - repository, data, and model licensing boundaries
- `model_card.md` - adapter/result publication template

Documentation must describe current behavior. Temporary planning artifacts and
unverifiable audit counts do not belong in this directory.
