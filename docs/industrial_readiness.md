# Industrial readiness

The codebase is an offline-verified release candidate for a controlled research
training environment. “Industrial grade” requires both engineering gates and
recorded external operation. This checklist separates them so static checks are
never presented as completed GPU experiments.

## Verified in this repository

| Gate | State | Evidence |
|---|---|---|
| Schema and role validation | PASS | offline tests cover canonical rows and boundary rejection |
| Fixture/benchmark leakage guards | PASS | trainer, importer, splitter, and Kaggle staging tests |
| Governed user-data import | PASS | CSV/JSON/JSONL normalization, immutable output, provenance tests |
| Pinned model contract | PASS | exact registered revisions and strict custom-model opt-out |
| Reproducible experiment contract | PASS | frozen plans, hashes, status/resume, paired equality checks |
| Local isolated runner configuration | PASS | Compose renders; non-root/read-only/capability controls present |
| Kaggle staging | PASS | offline staging and governance tests; no implicit upload |
| Training failure handling | PASS | preflight, OOM guidance, status capture, atomic adapter promotion |
| API boundary | PASS | session limits, CORS validation, health separation tests |
| Offline quality gates | PASS | Ruff, unit tests, package check, CLI plans, Compose validation |

## External gates still requiring the operator

| Gate | Why it cannot be completed offline | Required evidence |
|---|---|---|
| Qwen2.5 7B local GPU run | needs compatible NVIDIA hardware and model weights | full output/checkpoint tree and completed comparison |
| Mistral 7B compatibility run | needs compatible NVIDIA hardware and model weights | completed smoke/full run or status downgrade |
| Kaggle GPU run | needs account, quota, upload, and network | kernel URL/version plus downloaded output checksums |
| Result validity | depends on real data/model behavior | before/after reports, raw responses, capability checks, multiple seeds |
| Third-party license review | depends on each selected dataset and model | recorded source/model license evidence for the actual run |
| Container security review | native ML stack and host runtime are outside unit tests | image scan, dependency review, least-privilege host validation |

## Release decision

The repository can be handed to an operator for a controlled 7B validation run
under Apache-2.0. Do not describe it as production-proven or claim model
improvement until the external gates have artifacts. The code license does not
override the licenses or terms of selected models and datasets; see
`licensing.md`.
