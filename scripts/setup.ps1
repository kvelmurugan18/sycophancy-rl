param(
    [ValidateSet("cpu", "cu130")]
    [string]$TorchChannel = "cpu",
    [string]$PythonPath = "python",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & $PythonPath -c "import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))"
    if ($LASTEXITCODE -ne 0) { throw "Pass a Python 3.11 or 3.12 executable with -PythonPath." }
    & $PythonPath -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}
$venvPython = (Resolve-Path -LiteralPath ".venv\Scripts\python.exe").Path
& $venvPython -c "import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))"
if ($LASTEXITCODE -ne 0) { throw "Use a virtual environment created with Python 3.11 or 3.12." }
$torchIndex = "https://download.pytorch.org/whl/$TorchChannel"

& $venvPython -m pip install "torch==2.14.1" --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $venvPython -m pip install -e ".[dev,kaggle]"
if ($LASTEXITCODE -ne 0) { throw "Editable package installation failed." }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Installed dependencies are inconsistent." }

# Prepare data/processed/training_pool.jsonl and data/splits/ without
# overwriting existing artifacts. A fresh clone generates the smoke fixture
# pool; an existing repository with real data keeps the user's training pool
# and split files untouched.
& $venvPython -m sycophancy_rl.data_prep.setup_data ensure-fixtures
if ($LASTEXITCODE -ne 0) { throw "Preparing the training pool failed." }
& $venvPython -m sycophancy_rl.data_prep.setup_data ensure-splits
if ($LASTEXITCODE -ne 0) { throw "Preparing the train/validation/test splits failed." }

if (-not $SkipTests) {
    & $venvPython -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
}
& $venvPython -m sycophancy_rl.training.train_grpo --profile smoke --allow-cpu --no-4bit --preflight-only
