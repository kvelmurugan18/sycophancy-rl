# Supported models

The project uses an explicit compatibility registry instead of claiming that
every causal language model will work. Registered models have a fixed Hub ID,
exact commit SHA, architecture contract, context limit, LoRA targets, 4-bit
capability, minimum free VRAM, and validation status.

## Registry

| Model | Exact revision | 4-bit | Min free VRAM | Status |
|---|---|---:|---:|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | yes | 3 GiB | `configured` |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | yes | 3.5 GiB | `configured` |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | yes | 14 GiB | `configured` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | yes | 14 GiB | `configured` |

`validated` is reserved for a retained, reproducible end-to-end GPU artifact.
`configured` means code, pinning, preflight, and offline tests exist, but a real
GPU training/benchmark run has not been claimed from this repository. All
current profiles are therefore `configured`. No status promises that a model
will fit every GPU or driver stack.

All profiles require a tokenizer chat template, use a causal-LM loader,
keep `trust_remote_code=False`, prefer safetensors, and use the PEFT
registered targeting strategy. The 0.5B Qwen profile explicitly targets
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

The first governed Anthropic experiment target is
`Qwen/Qwen2.5-0.5B-Instruct` at the revision shown above. A model name or size absent from the registry is not
an alias for that target and must use the explicit custom-model contract.

## First experiment invocation

Use `kaggle_online_smoke` for the real optimizer smoke and
`qwen25_05b_online` for the full development run. Both require online rollouts,
4-bit loading, and the exact registered revision. `configured` is not GPU
validation; the Kaggle model-load and optimizer gates remain external.

## Recommended 7B invocation

Use `qlora_7b_16gb`, 4-bit loading, benchmark batch size `1`, and the registered
revision:

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl run `
  --run-id qwen25-7b-seed42 `
  --profile qlora_7b_16gb `
  --model-id Qwen/Qwen2.5-7B-Instruct `
  --batch-size 1
```

The registry fills the revision when it is omitted. Supplying a different
revision for a registered model is rejected so before/after provenance cannot
silently drift.

## Custom Hugging Face models

An unregistered model is an advanced, unvalidated path. The caller must supply:

- a plain Hugging Face repository ID;
- an exact 40-64 character hexadecimal commit SHA;
- `--allow-unpinned-model` to acknowledge the registry opt-out;
- `--custom-supports-4bit` when NF4 is known to work;
- explicit LoRA targets, context length, and minimum free VRAM when the defaults
  are inappropriate.

Example planning command:

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl run `
  --run-id custom-plan `
  --model-id owner/model `
  --model-revision 0123456789abcdef0123456789abcdef01234567 `
  --allow-unpinned-model `
  --custom-supports-4bit `
  --custom-lora-targets all-linear `
  --custom-context-length 8192 `
  --custom-min-vram-gib 14 `
  --profile qlora_7b_16gb `
  --batch-size 1
```

URLs, filesystem paths, branch names, tags, empty LoRA targets, unsupported
characters, and custom contexts below 512 are rejected. Passing the flags does
not certify compatibility; the operator owns the first GPU validation.
