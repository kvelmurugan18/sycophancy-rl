# LinkedIn evidence plan

Use three separate posts so goals, execution, and results are not mixed. Replace
all placeholders from saved artifacts; never invent metrics or quote a terminal
screenshot as the source of truth.

## Post 1: before training

Publish only after the license decision and after the baseline artifact exists.

> I am testing whether a 7B instruction model changes correct answers to match
> invalid user pressure. The experiment uses a governed training split and the
> Anthropic sycophancy dataset for evaluation only. Before training, the pinned
> `<model>@<revision>` baseline scored `<metrics>` on `<N>` examples. Next I will
> run 4-bit QLoRA + GRPO with the same prompt and decoding contract frozen for
> the after test. Code and artifacts: `<links>`.

Attach the baseline summary/manifest, dataset provenance, run ID, model
revision, seed, prompt condition, and an explicit `pre-training` label.

## Post 2: training completion

> The adapter training stage for `<run-id>` completed on `<hardware>` in
> `<duration>`. The run used `<training rows>`, seed `<seed>`, profile
> `<profile>`, and preserved the Anthropic benchmark as evaluation-only. Peak
> GPU memory was `<value>` and the adapter checksum is `<hash>`. This post is an
> engineering update; the after benchmark determines whether behavior improved.

Attach the completed training manifest and learning curves. Do not treat reward
increase as the final result.

## Post 3: after benchmark and comparison

> Under the identical paired benchmark condition, `<model + adapter>` changed
> sycophantic-answer rate from `<before>` to `<after>` (`<interval/test>`),
> target accuracy from `<before>` to `<after>`, invalid rate from `<before>` to
> `<after>`, and unnecessary disagreement from `<before>` to `<after>`. Results
> cover `<N>` paired examples and `<number of seeds>` seeds. Limitations:
> binary-choice evaluation, `<other limitations>`. Reproduction artifacts:
> `<links>`.

Attach the paired comparison, before/after manifests, raw-response location,
capability-regression report, model card, and error analysis. If the result is
negative or mixed, publish it honestly.

## Claim rules

- Say `smoke` or `subset` whenever `--max-examples` limits the benchmark.
- Do not say “industrial grade” until external readiness gates are recorded.
- Do not say “open source” until a license is committed.
- Report absolute counts with rates and uncertainty.
- Keep invalid/refusal outputs in the denominator.
- State that Anthropic evaluation rows were not used for training.
