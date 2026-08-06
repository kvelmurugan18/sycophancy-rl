# Experiment protocol

One `ExperimentPlan` controls both local and Kaggle runs. Planning freezes the
model revision, data paths and hashes, seed, prompt condition, reward profile,
training configuration, decoding settings, output paths, and runner. Execution
then follows one ordered workflow:

```text
validate plan -> benchmark base -> train adapter -> release model memory
              -> benchmark adapter -> verify provenance -> paired comparison
```

If a stage fails, later stages do not run and the failure is recorded.

## Before/after equality contract

The comparison is accepted only when before and after share:

- benchmark example IDs and file hash;
- base model ID and exact revision;
- prompt text/hash and system-prompt condition;
- decoding settings (`do_sample`, temperature, top-p, top-k, maximum new
  tokens, repetition penalty, and batch semantics);
- evaluation parser and metric implementation from the same code run.

The after condition differs only by the trained adapter. A smoke subset cannot
be compared with a full benchmark and presented as a full result.

## Run artifacts

The pipeline records a frozen plan and stage manifests with:

- run ID and lifecycle status;
- model ID/revision and adapter path/checksum;
- training, validation, and benchmark SHA-256 values;
- seed, prompt condition, reward profile, and generation settings;
- effective training hyperparameters;
- Python/package and hardware versions;
- timestamps, duration, peak GPU memory, and failure details;
- raw before/after responses, metrics, and paired statistics.

Manifest writes and final adapter promotion are atomic. Existing immutable
identities are not silently overwritten.

## Lifecycle and resume

```text
created -> running -> completed
              |----> failed
              |----> interrupted -> running
```

Resume is allowed only for a compatible created, interrupted, or failed run.
The run ID, model/revision, datasets, seed, reward profile, and other frozen
identity fields must still match. Manually editing a manifest invalidates its
provenance.

## Required metrics

Report at least target accuracy, independent-answer rate, sycophantic-answer
rate, invalid-answer rate, unnecessary-disagreement rate, and capability
regression. The paired report includes Wilson 95% confidence intervals and an
exact McNemar comparison where applicable. Invalid, contradictory, truncated,
and refusal outputs remain in the denominator.

## Minimum publishable evidence

A publishable result requires:

1. a non-fixture governed training dataset;
2. a complete before benchmark saved before adapter training;
3. a completed adapter and immutable checksum;
4. an after benchmark under the equality contract;
5. a paired report plus raw responses and error analysis;
6. explicit labeling of model, revision, data, seed, prompt, and hardware;
7. no claim beyond the tested binary-choice benchmark.

Multiple seeds and prompt conditions are strongly preferred before claiming a
general effect. A passing unit test or dry run is engineering evidence, not a
model-quality result.
