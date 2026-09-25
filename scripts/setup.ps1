param([switch]$SkipChecks)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    py -3.12 -m venv (Join-Path $projectRoot ".venv")
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r requirements-lock.txt
& $venvPython -m pip install -e ".[dev]"
& $venvPython -m pre_commit install

if (-not $SkipChecks) {
    & $venvPython -m pytest
    & $venvPython -m ruff check .
    & $venvPython -m ruff format --check .
}

Write-Host "Setup complete. Activate with .\.venv\Scripts\Activate.ps1"
