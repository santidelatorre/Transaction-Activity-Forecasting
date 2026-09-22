param(
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "Preparing Transaction Activity Forecasting..."

function Invoke-Checked {
    param(
        [string]$Description,
        [scriptblock]$Command
    )
    Write-Host $Description
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

$pythonSpec = "3.11"
$hasPython311 = $true
try {
    & py -3.11 -c "import sys; print(sys.version)" 2>$null
} catch {
    $hasPython311 = $false
}
if (-not $hasPython311) {
    Write-Warning "Python 3.11 is not installed. Falling back to Python 3.12, which satisfies this project's compatibility range."
    $pythonSpec = "3.12"
    $hasPython312 = $true
    try {
        & py -3.12 -c "import sys; print(sys.version)" 2>$null
    } catch {
        $hasPython312 = $false
    }
    if (-not $hasPython312) {
        throw "Python 3.11 or 3.12 is required. Install Python 3.11 from python.org and run this script again."
    }
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating .venv with Python $pythonSpec..."
    & py -$pythonSpec -m venv (Join-Path $projectRoot ".venv")
}

Write-Host "Installing project and development dependencies..."
Invoke-Checked "Upgrade packaging tools" { & $venvPython -m pip install --upgrade pip setuptools wheel }
Invoke-Checked "Install project and development dependencies" { & $venvPython -m pip install -e ".[dev]" }
Invoke-Checked "Install pre-commit hooks" { & $venvPython -m pre_commit install }
Invoke-Checked "Register Jupyter kernel" { & $venvPython -m ipykernel install --user --name transaction-forecasting --display-name "Python (transaction-forecasting)" }

if (-not $SkipChecks) {
    Write-Host "Running basic checks..."
    Invoke-Checked "Python version" { & $venvPython --version }
    Invoke-Checked "pip version" { & $venvPython -m pip --version }
    Invoke-Checked "pytest" { & $venvPython -m pytest }
    Invoke-Checked "Ruff lint" { & $venvPython -m ruff check . }
    Invoke-Checked "Ruff format check" { & $venvPython -m ruff format --check . }
    Invoke-Checked "pre-commit" { & $venvPython -m pre_commit run --all-files }
}

Write-Host "Setup complete. Activate with .\.venv\Scripts\Activate.ps1"
