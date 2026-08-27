# Local container runner

`Dockerfile.trainer` and `docker-compose.trainer.yml` run training separately
from the API/demo image. The complete GPU services execute the same canonical
pipeline: baseline benchmark, GRPO training, adapter benchmark, and paired
comparison.

## Services

| Compose profile | Service | Purpose |
|---|---|---|
| `preflight` | `trainer-preflight` | CPU-safe dependency/data checks; no weight loading |
| `smollm` | `trainer-smollm` | Complete SmolLM2 1.7B reference workflow |
| `7b` | `trainer-7b` | Complete Qwen2.5/Mistral 7B 4-bit QLoRA workflow |

The 7B profile requires an NVIDIA GPU and at least 14 GiB of available VRAM.
Actual requirements vary with context length, driver, and fragmentation.

## First-run preparation

1. Prepare the governed Anthropic experiment or import another authorized
   source; keep its held-out benchmark protected.
2. Copy `.env.example` to `.env` and adjust model, revision, and container data
   paths. Do not put credentials in tracked files.
3. Ensure Docker Desktop, its NVIDIA runtime, and the host NVIDIA driver work.
4. Keep `HF_HUB_OFFLINE=0` for the first authorized model download. After the
   named volume contains all required model files, set the three offline flags
   to `1` for a network-independent repeat.

## Complete 7B run

Windows PowerShell:

```powershell
.\scripts\run_local_container.ps1 `
  -ModelSize 7b `
  -RunId qwen25-7b-seed42 `
  -TrainPath /data/generated/user/my-dataset/splits/train.jsonl `
  -ValidationPath /data/generated/user/my-dataset/splits/validation.jsonl `
  -BenchmarkPath /data/benchmarks/anthropic_sycophancy.jsonl `
  -MaxExamples 200 `
  -BatchSize 1 `
  -Build
```

Linux:

```bash
MODEL_SIZE=7b \
SYCO_RUN_ID=qwen25-7b-seed42 \
SYCO_TRAIN_PATH=/data/generated/user/my-dataset/splits/train.jsonl \
SYCO_VALIDATION_PATH=/data/generated/user/my-dataset/splits/validation.jsonl \
SYCO_BENCHMARK_PATH=/data/benchmarks/anthropic_sycophancy.jsonl \
SYCO_MAX_EXAMPLES=200 \
SYCO_BATCH_SIZE=1 \
bash scripts/run_local_container.sh --build
```

The script exits with the trainer service's exit code. It does not run in the
background.

## Persistent mounts

| Container path | Host/named volume | Access |
|---|---|---|
| `/data` | `./data` | read-only |
| `/cache/huggingface` | `hf-cache` | writable model cache |
| `/outputs` | `./outputs` | writable reports/manifests |
| `/checkpoints` | `./outputs/checkpoints` | writable adapters/checkpoints |

The Docker socket is never mounted. The process runs as non-root with all Linux
capabilities dropped, `no-new-privileges`, a read-only root filesystem, and a
temporary `/tmp`. SIGTERM is forwarded so interrupted runs can persist state.

## Isolation boundary

This design isolates a trusted training program from the host filesystem and
keeps datasets read-only. It is not a security sandbox for hostile Python,
model repositories that require arbitrary remote code, or malicious native
libraries. Run untrusted workloads inside a disposable VM or microVM. See
`threat_model.md`.

## Static validation without training

```powershell
docker compose -f docker-compose.trainer.yml config --quiet
docker compose -f docker-compose.trainer.yml --profile 7b config --quiet
```

These commands validate rendered Compose configuration; they do not build the
image, download weights, or start a run.
