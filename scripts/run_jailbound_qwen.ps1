$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $RepoRoot "src"

python -m jailbound run --config (Join-Path $RepoRoot "configs/qwen25vl_local.json") @args

