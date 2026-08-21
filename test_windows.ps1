[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "venv_docnorm_web\Scripts\python.exe"
$webSmoke = Join-Path $PSScriptRoot "DocNorm_MD_Web_v1_4\smoke_test_windows.ps1"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "venv_docnorm_web is missing. Run install_windows.ps1 first."
}

$testId = [Guid]::NewGuid().ToString("N").Substring(0, 10)
$baseTemp = Join-Path $env:LOCALAPPDATA "DocNormMD\github_tests\$testId"
$env:PYTHONPATH = Join-Path $PSScriptRoot "pipeline_v2\src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTEST_ADDOPTS = "-p no:cacheprovider"
$pipelineTests = Join-Path $PSScriptRoot "pipeline_v2\tests"
$pipelineTemp = Join-Path $baseTemp "pipeline"

Write-Host "Running pipeline_v2 tests..." -ForegroundColor Cyan
& $python -m pytest `
    $pipelineTests `
    --basetemp $pipelineTemp `
    -p no:cacheprovider
if ($LASTEXITCODE -ne 0) {
    throw "pipeline_v2 tests failed."
}

Write-Host "Running DocNorm web tests..." -ForegroundColor Cyan
& powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File $webSmoke `
    -ProjectRoot $PSScriptRoot
if ($LASTEXITCODE -ne 0) {
    throw "DocNorm web tests failed."
}

Write-Host "DOCNORM_GITHUB_TESTS=PASS" -ForegroundColor Green
