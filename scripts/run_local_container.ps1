param(
    [ValidateSet("smollm", "7b")]
    [string]$ModelSize = "7b",
    [string]$RunId = "",
    [Parameter(Mandatory = $true)]
    [string]$TrainPath,
    [Parameter(Mandatory = $true)]
    [string]$ValidationPath,
    [Parameter(Mandatory = $true)]
    [string]$BenchmarkPath,
    [int]$MaxExamples = 200,
    [int]$BatchSize = 1,
    [switch]$Build
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

$service = if ($ModelSize -eq "7b") { "trainer-7b" } else { "trainer-smollm" }
$profile = $ModelSize
if (-not $RunId) {
    $RunId = "$ModelSize-local-seed42"
}

$env:SYCO_RUN_ID = $RunId
$env:SYCO_TRAIN_PATH = $TrainPath
$env:SYCO_VALIDATION_PATH = $ValidationPath
$env:SYCO_BENCHMARK_PATH = $BenchmarkPath
$env:SYCO_MAX_EXAMPLES = [string]$MaxExamples
$env:SYCO_BATCH_SIZE = [string]$BatchSize

$arguments = @("compose", "-f", "docker-compose.trainer.yml", "--profile", $profile, "up")
if ($Build) { $arguments += "--build" }
$arguments += @("--abort-on-container-exit", "--exit-code-from", $service, $service)
& docker @arguments
exit $LASTEXITCODE
