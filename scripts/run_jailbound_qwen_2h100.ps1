$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $RepoRoot "src"

accelerate launch `
  --num_processes 2 `
  --mixed_precision bf16 `
  -m jailbound run `
  --config (Join-Path $RepoRoot "configs/qwen25vl_local.json") `
  @args

