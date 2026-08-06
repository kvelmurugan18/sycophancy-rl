# User dataset format

Use `user_choice_dataset.csv` as the smallest supported import template. Required
columns are:

- `question` or `prompt`
- `option_a`
- `option_b`
- `target_option` (`A` or `B`)

Optional columns include `id`, `group_id`, `independent_option`,
`user_preferred_option`, `sycophantic_option`, `user_claim_valid`, `pressure`,
`question_type`, and `topic`.

Import a copy with:

```text
syco import-data --input my-data.csv --dataset-name my-experiment-data
```

The importer writes immutable train/validation/test splits under
`data/generated/user/<dataset-name>/`. The Anthropic benchmark remains separate
and can never be imported as training data.
